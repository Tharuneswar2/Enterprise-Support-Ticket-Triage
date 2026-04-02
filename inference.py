from __future__ import annotations

import argparse
import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime fallback
    OpenAI = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional runtime fallback
    load_dotenv = None

from env.models import Action, ActionType, Observation, PriorityLevel
from env.support_env import EnterpriseSupportTicketTriageEnv
from env.tasks import TASK_ORDER
from env.utils import extract_json_object


SYSTEM_PROMPT = """
You are an enterprise support triage policy.
Return exactly one JSON object with fields:
- action_type: one of classify|reply|escalate|close|request_info|update_priority|assign_queue
- message: string
- queue: string or null
- priority: low|normal|high|urgent or null
- tags: array of short strings
- resolution_code: string or null
Do not include markdown. Do not include explanations.
""".strip()


def build_user_prompt(observation: Observation) -> str:
    return (
        "Ticket observation JSON:\n"
        f"{observation.model_dump_json(indent=2)}\n\n"
        "Policy goals:\n"
        "1) Correctly classify and route the ticket.\n"
        "2) Ask for missing diagnostics when needed.\n"
        "3) Escalate only when appropriate.\n"
        "4) Avoid premature closure.\n"
        "Return only JSON."
    )


def call_model_for_action(client: Any, model_name: str, observation: Observation) -> Action:
    completion = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        max_tokens=220,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(observation)},
        ],
    )
    content = completion.choices[0].message.content or ""
    payload = extract_json_object(content)
    return Action.model_validate(payload)


def heuristic_action(observation: Observation) -> Action:
    task_id = observation.task_id
    step = observation.step_count

    if task_id == "easy_billing_duplicate_charge":
        if step == 0:
            return Action(
                action_type=ActionType.CLASSIFY,
                message="This is a billing issue for a duplicate charge on invoice INV-88421.",
                tags=["billing", "duplicate_charge", "invoice"],
            )
        if step == 1:
            return Action(action_type=ActionType.ASSIGN_QUEUE, queue="billing-ops", tags=["billing"])
        if step == 2:
            return Action(
                action_type=ActionType.REPLY,
                message=(
                    "Thanks for reporting this duplicate charge. We are initiating a refund review now and will "
                    "update you with next steps once billing confirms the duplicate invoice charge."
                ),
                tags=["refund", "duplicate_charge", "next_steps"],
            )
        return Action(
            action_type=ActionType.CLOSE,
            message="Billing workflow has been initiated and customer informed.",
            resolution_code="duplicate_charge_refund_initiated",
            tags=["billing_closed"],
        )

    if task_id == "medium_security_suspicious_login":
        if step == 0:
            return Action(
                action_type=ActionType.CLASSIFY,
                message="Security incident: suspicious login and possible account compromise.",
                tags=["security", "suspicious_login", "account_compromise"],
            )
        if step == 1:
            return Action(action_type=ActionType.ASSIGN_QUEUE, queue="security-response", tags=["security"])
        if step == 2:
            return Action(action_type=ActionType.UPDATE_PRIORITY, priority=PriorityLevel.HIGH, tags=["high_risk"])
        if step == 3:
            return Action(
                action_type=ActionType.REQUEST_INFO,
                message=(
                    "Please verify identity and share the last login time, login location/IP, and MFA status so we can "
                    "confirm unauthorized access quickly."
                ),
                tags=["last_known_login_time", "login_location", "mfa_status", "verify_identity"],
            )
        if step == 4:
            return Action(
                action_type=ActionType.REPLY,
                message=(
                    "Please reset your password immediately and rotate API credentials. Keep MFA enabled while we "
                    "verify account activity details."
                ),
                tags=["password reset", "credential reset", "mfa"],
            )
        if step == 5:
            return Action(
                action_type=ActionType.REPLY,
                message="Ticket remains open in security response while verification is in progress.",
                tags=["security", "open_ticket"],
            )
        if step == 6:
            return Action(
                action_type=ActionType.REQUEST_INFO,
                message="Please confirm if any admin device was recently replaced and verify identity ownership.",
                tags=["verify", "identity"],
            )
        return Action(
            action_type=ActionType.REPLY,
            message="We are monitoring this case and will continue once verification evidence is received.",
            tags=["monitoring", "security"],
        )

    if task_id == "hard_production_outage":
        if step == 0:
            return Action(
                action_type=ActionType.CLASSIFY,
                message="Critical production incident with checkout outage and deployment impact.",
                tags=["incident", "production", "outage"],
            )
        if step == 1:
            return Action(action_type=ActionType.ASSIGN_QUEUE, queue="platform-incident", tags=["incident"])
        if step == 2:
            return Action(action_type=ActionType.UPDATE_PRIORITY, priority=PriorityLevel.URGENT, tags=["sev1"])
        if step == 3:
            return Action(
                action_type=ActionType.REQUEST_INFO,
                message=(
                    "Please share incident start time, affected service name, request ID/trace ID, and recent error logs "
                    "or stack traces so we can isolate the 503 failure."
                ),
                tags=["incident_start_time", "affected_service", "request_id", "error_logs"],
            )
        return Action(
            action_type=ActionType.ESCALATE,
            message="Escalating to the platform on-call incident commander for urgent mitigation.",
            tags=["escalate", "on-call", "incident_bridge"],
        )

    return Action(
        action_type=ActionType.REPLY,
        message="Acknowledged. Requesting additional details to proceed.",
        tags=["triage"],
    )


def choose_action(client: Any | None, model_name: str, observation: Observation) -> tuple[Action, bool]:
    if client is None:
        return heuristic_action(observation), False
    try:
        return call_model_for_action(client, model_name, observation), True
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] model action parsing failed, fallback to heuristic: {exc}")
        return heuristic_action(observation), False


def run_episode(
    env: EnterpriseSupportTicketTriageEnv,
    task_id: str,
    client: Any | None,
    model_name: str,
) -> float:
    observation = env.reset(task_id=task_id)
    done = False
    model_enabled = client is not None
    consecutive_model_failures = 0

    print(f"\n=== Task: {task_id} ===")
    while not done:
        if model_enabled:
            action, ok = choose_action(client, model_name, observation)
            if not ok:
                consecutive_model_failures += 1
                if consecutive_model_failures >= 3:
                    model_enabled = False
                    print("[warn] disabling model calls after 3 consecutive failures; using heuristic policy.")
            else:
                consecutive_model_failures = 0
        else:
            action = heuristic_action(observation)
        observation, reward, done, info = env.step(action)
        print(
            f"step={observation.step_count} action={action.action_type.value} "
            f"status={observation.current_status.value} reward={reward.value:.3f}"
        )

    score = float(info.get("task_score") or 0.0)
    print(f"task_score={score:.4f}")
    return score


def _normalize_base_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized


def build_client_from_env(disable_api: bool) -> tuple[Any | None, str]:
    if load_dotenv is not None:
        load_dotenv()

    api_key = (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
        or os.getenv("API_KEY", "").strip()
        or os.getenv("HF_API_TOKEN", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    api_base_url = _normalize_base_url(os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"))
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    if disable_api:
        print("[info] API disabled by --disable-api, using deterministic heuristic baseline.")
        return None, model_name
    if not api_key:
        print(
            "[info] No API token found in HF_TOKEN/HUGGING_FACE_HUB_TOKEN/API_KEY/HF_API_TOKEN; "
            "using deterministic heuristic baseline."
        )
        return None, model_name
    if OpenAI is None:
        print("[info] openai package unavailable, using deterministic heuristic baseline.")
        return None, model_name

    client = OpenAI(base_url=api_base_url, api_key=api_key)
    return client, model_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline inference for Enterprise Support Ticket Triage")
    parser.add_argument("--disable-api", action="store_true", help="Run fully deterministic heuristic baseline")
    parser.add_argument("--max-steps", type=int, default=8, help="Global episode step cap")
    args = parser.parse_args()

    env = EnterpriseSupportTicketTriageEnv(max_steps=args.max_steps)
    client, model_name = build_client_from_env(disable_api=args.disable_api)

    scores: list[float] = []
    for task_id in TASK_ORDER:
        score = run_episode(
            env,
            task_id=task_id,
            client=client,
            model_name=model_name,
        )
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    print("\n=== Baseline Summary ===")
    for task_id, score in zip(TASK_ORDER, scores):
        print(f"{task_id}: {score:.4f}")
    print(f"average_score: {avg_score:.4f}")


if __name__ == "__main__":
    main()
