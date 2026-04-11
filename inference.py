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


def emit(line: str) -> None:
    """Emit validator-friendly stdout lines with immediate flush."""
    print(line, flush=True)


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


# ---------------------------------------------------------------------------
# Deterministic heuristic baseline
#
# Intentionally imperfect — demonstrates that grading has discriminating power.
# A real agent trained with RL should outperform this baseline.
# ---------------------------------------------------------------------------


def heuristic_action(observation: Observation) -> Action:
    """
    Deterministic baseline that follows reasonable but imperfect workflows.
    Deliberately misses some milestones to produce realistic (~0.65-0.80) scores.
    """
    task_id = observation.task_id
    step = observation.step_count

    # ── Easy: Billing duplicate charge ──────────────────────────────────
    if task_id == "easy_billing_duplicate_charge":
        if step == 0:
            return Action(
                action_type=ActionType.CLASSIFY,
                message="This is a billing issue for a duplicate charge on the invoice.",
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

    # ── Easy: Password reset ───────────────────────────────────────────
    if task_id == "easy_password_reset_request":
        if step == 0:
            return Action(
                action_type=ActionType.CLASSIFY,
                message="Account access issue — user locked out and needs password reset.",
                tags=["account", "access", "password_reset"],
            )
        if step == 1:
            return Action(action_type=ActionType.ASSIGN_QUEUE, queue="identity-ops", tags=["account"])
        if step == 2:
            # Deliberately skip request_info for verification — imperfect baseline
            return Action(
                action_type=ActionType.REPLY,
                message=(
                    "I understand you're locked out. We can initiate a password reset for you. "
                    "Please check your registered email for the reset link. If you still don't "
                    "receive it, we may need to verify your identity through an alternative method."
                ),
                tags=["reset", "password"],
            )
        return Action(
            action_type=ActionType.CLOSE,
            message="Password reset guidance provided.",
            resolution_code="password_reset_guided",
            tags=["resolved"],
        )

    # ── Easy: Feature request ──────────────────────────────────────────
    if task_id == "easy_feature_request":
        if step == 0:
            return Action(
                action_type=ActionType.CLASSIFY,
                message="This is a feature request for bulk CSV user import.",
                tags=["feature_request", "product_feedback"],
            )
        if step == 1:
            return Action(action_type=ActionType.ASSIGN_QUEUE, queue="product-feedback", tags=["feature"])
        if step == 2:
            return Action(
                action_type=ActionType.REPLY,
                message=(
                    "Thank you for this feature request. We've acknowledged your need for bulk CSV import "
                    "in user management. Our product team will review this and consider it for the roadmap. "
                    "We'll follow up when there's an update on the timeline."
                ),
                tags=["feature_request", "roadmap", "acknowledged"],
            )
        # Deliberately skip setting priority to LOW — imperfect baseline
        return Action(
            action_type=ActionType.CLOSE,
            message="Feature request logged and acknowledged.",
            resolution_code="feature_request_logged",
            tags=["closed"],
        )

    # ── Medium: Security suspicious login ──────────────────────────────
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
            # Deliberately only request 2 of 3 required fields — imperfect baseline
            return Action(
                action_type=ActionType.REQUEST_INFO,
                message=(
                    "Please share the last login time and login location/IP so we can "
                    "investigate the unauthorized access."
                ),
                tags=["last_known_login_time", "login_location"],
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
        # Deliberately send generic reply instead of targeted follow-up — imperfect
        return Action(
            action_type=ActionType.REPLY,
            message="We are monitoring this case and will continue once verification evidence is received.",
            tags=["monitoring", "security"],
        )

    # ── Medium: Data export compliance ─────────────────────────────────
    if task_id == "medium_data_export_compliance":
        if step == 0:
            return Action(
                action_type=ActionType.CLASSIFY,
                message="Compliance request — GDPR Article 15 data subject access request.",
                tags=["compliance", "gdpr", "data_export"],
            )
        if step == 1:
            return Action(action_type=ActionType.ASSIGN_QUEUE, queue="legal-compliance", tags=["compliance"])
        if step == 2:
            # Deliberately skip priority update — goes straight to request_info
            # Only ask for 1 of 3 required fields — imperfect baseline
            return Action(
                action_type=ActionType.REQUEST_INFO,
                message=(
                    "To process your data export request, we need to verify your role and authority "
                    "to make this request on behalf of your organization."
                ),
                tags=["requester_role"],
            )
        if step == 3:
            return Action(
                action_type=ActionType.REPLY,
                message=(
                    "We've received your GDPR Article 15 data export request. Our compliance team will process "
                    "this within the 30-day regulatory timeline. We'll provide updates as the export progresses."
                ),
                tags=["gdpr", "timeline"],
            )
        return Action(
            action_type=ActionType.REPLY,
            message="Your data export request is being processed by our legal compliance team.",
            tags=["compliance", "processing"],
        )

    # ── Medium: API rate limiting ──────────────────────────────────────
    if task_id == "medium_api_rate_limiting":
        if step == 0:
            return Action(
                action_type=ActionType.CLASSIFY,
                message="Technical issue — API rate limiting causing 429 errors on production integration.",
                tags=["incident", "api", "rate_limiting", "429"],
            )
        if step == 1:
            return Action(action_type=ActionType.ASSIGN_QUEUE, queue="platform-support", tags=["platform"])
        if step == 2:
            # Deliberately set NORMAL instead of HIGH — imperfect baseline
            return Action(action_type=ActionType.UPDATE_PRIORITY, priority=PriorityLevel.NORMAL, tags=["technical"])
        if step == 3:
            return Action(
                action_type=ActionType.REQUEST_INFO,
                message=(
                    "Could you share which API endpoint is returning 429 errors and your current "
                    "request volume? Also, is there a pattern to when the errors occur?"
                ),
                tags=["api_endpoint", "request_volume", "error_pattern"],
            )
        return Action(
            action_type=ActionType.REPLY,
            message=(
                "We're investigating the 429 rate limit errors on your API integration. "
                "Our platform team is reviewing your account's rate limit configuration."
            ),
            tags=["rate_limit", "investigating"],
        )

    # ── Hard: Production outage ────────────────────────────────────────
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

    # ── Hard: Data breach investigation ────────────────────────────────
    if task_id == "hard_data_breach_investigation":
        if step == 0:
            return Action(
                action_type=ActionType.CLASSIFY,
                message="Critical security incident — potential data breach with unauthorized PII access.",
                tags=["security", "breach", "unauthorized", "incident"],
            )
        if step == 1:
            return Action(action_type=ActionType.ASSIGN_QUEUE, queue="security-incident", tags=["security"])
        if step == 2:
            return Action(action_type=ActionType.UPDATE_PRIORITY, priority=PriorityLevel.URGENT, tags=["critical"])
        if step == 3:
            # Only request 3 of 4 fields — imperfect
            return Action(
                action_type=ActionType.REQUEST_INFO,
                message=(
                    "We need the compromised token ID, the list of affected user accounts, "
                    "and the timestamps of the unauthorized data exports immediately."
                ),
                tags=["compromised_token_id", "affected_user_list", "export_timestamps"],
            )
        if step == 4:
            return Action(
                action_type=ActionType.REPLY,
                message=(
                    "IMMEDIATE ACTION REQUIRED: Please revoke the compromised API token immediately. "
                    "Block all API access from the suspicious source. We are treating this as a "
                    "confirmed data breach incident requiring containment."
                ),
                tags=["contain", "revoke", "breach"],
            )
        return Action(
            action_type=ActionType.ESCALATE,
            message="Escalating to CISO and incident response team for breach containment and notification.",
            tags=["escalate", "ciso", "breach_response"],
        )

    # ── Hard: Multi-service cascade failure ────────────────────────────
    if task_id == "hard_multi_service_cascade_failure":
        if step == 0:
            return Action(
                action_type=ActionType.CLASSIFY,
                message="Critical production incident — cascading failure across multiple services.",
                tags=["incident", "cascade", "production", "outage"],
            )
        if step == 1:
            return Action(action_type=ActionType.ASSIGN_QUEUE, queue="platform-incident", tags=["incident"])
        if step == 2:
            return Action(action_type=ActionType.UPDATE_PRIORITY, priority=PriorityLevel.URGENT, tags=["sev1"])
        if step == 3:
            # Only request 3 of 5 fields — imperfect
            return Action(
                action_type=ActionType.REQUEST_INFO,
                message=(
                    "We need the incident start time, the list of affected services, "
                    "and any error logs from the failing services."
                ),
                tags=["incident_start_time", "affected_service", "error_logs"],
            )
        return Action(
            action_type=ActionType.ESCALATE,
            message=(
                "Escalating cascade failure to infrastructure on-call team. "
                "Multiple services are down — requesting incident bridge activation."
            ),
            tags=["escalate", "infrastructure", "incident_bridge"],
        )

    # ── Fallback ───────────────────────────────────────────────────────
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
        emit(f"[warn] model action parsing failed, fallback to heuristic: {exc}")
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

    emit(f"[START] task={task_id}")
    while not done:
        if model_enabled:
            action, ok = choose_action(client, model_name, observation)
            if not ok:
                consecutive_model_failures += 1
                if consecutive_model_failures >= 3:
                    model_enabled = False
                    emit("[warn] disabling model calls after 3 consecutive failures; using heuristic policy.")
            else:
                consecutive_model_failures = 0
        else:
            action = heuristic_action(observation)
        observation, reward, done, info = env.step(action)
        emit(
            "[STEP] "
            f"task={task_id} "
            f"step={observation.step_count} "
            f"action={action.action_type.value} "
            f"status={observation.current_status.value} "
            f"reward={reward.value:.3f}"
        )

    score = float(info.get("task_score") or 0.0)
    emit(f"[END] task={task_id} score={score:.4f} steps={observation.step_count}")
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
        emit("[info] API disabled by --disable-api, using deterministic heuristic baseline.")
        return None, model_name
    if not api_key:
        emit(
            "[info] No API token found in HF_TOKEN/HUGGING_FACE_HUB_TOKEN/API_KEY/HF_API_TOKEN; "
            "using deterministic heuristic baseline."
        )
        return None, model_name
    if OpenAI is None:
        emit("[info] openai package unavailable, using deterministic heuristic baseline.")
        return None, model_name

    client = OpenAI(base_url=api_base_url, api_key=api_key)
    return client, model_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline inference for Enterprise Support Ticket Triage")
    parser.add_argument("--disable-api", action="store_true", help="Run fully deterministic heuristic baseline")
    parser.add_argument("--max-steps", type=int, default=10, help="Global episode step cap")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific task IDs to run (default: all)")
    args = parser.parse_args()

    env = EnterpriseSupportTicketTriageEnv(max_steps=args.max_steps)
    client, model_name = build_client_from_env(disable_api=args.disable_api)

    task_ids = args.tasks if args.tasks else TASK_ORDER

    scores: list[float] = []
    for task_id in task_ids:
        score = run_episode(
            env,
            task_id=task_id,
            client=client,
            model_name=model_name,
        )
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    emit("[SUMMARY] per_task_scores")
    for task_id, score in zip(task_ids, scores):
        emit(f"[SUMMARY] task={task_id} score={score:.4f}")
    emit(f"[SUMMARY] average_score={avg_score:.4f}")
    emit(f"[SUMMARY] tasks_run={len(scores)}")


if __name__ == "__main__":
    main()
