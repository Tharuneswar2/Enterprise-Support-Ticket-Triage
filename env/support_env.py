from __future__ import annotations

import datetime
from typing import Any

from pydantic import ValidationError

from .graders import grade_task
from .models import (
    Action,
    ActionType,
    ConversationTurn,
    EpisodeState,
    Observation,
    PriorityLevel,
    Reward,
    TicketCategory,
    TicketStatus,
)
from .tasks import TASK_ORDER, generate_variant, get_task
from .utils import action_signature, clip, contains_any, detect_requested_fields, normalize_text


# ---------------------------------------------------------------------------
# Simulated customer responses — generated after REQUEST_INFO actions
# ---------------------------------------------------------------------------

_CUSTOMER_RESPONSES: dict[str, dict[str, str]] = {
    "last_known_login_time": "The last legitimate login was yesterday at 2:15 PM PST from our San Francisco office.",
    "login_location": "All our admins are based in San Francisco, CA. The suspicious login came from Eastern Europe.",
    "mfa_status": "MFA is enabled for all admin accounts. We use hardware security keys (YubiKey).",
    "incident_start_time": "The first 503 error appeared at 2025-04-10T14:47:00Z according to our monitoring.",
    "affected_service": "The checkout-api service and its downstream payment-gateway are both affected.",
    "request_id": "Here's a sample failing request ID: req_8f2a1c3d-e4b5-6789-abcd-ef0123456789",
    "error_logs": "Stack trace shows: ConnectionRefusedError at line 142 in checkout_handler.py → payment_client.charge()",
    "account_email": "The account email is j.martinez@acmecorp.com, registered to our IT department.",
    "verification_method": "I can verify via the phone number on file or answer our security questions.",
    "requester_role": "I'm the Data Protection Officer (DPO) for our organization, authorized for data requests.",
    "data_scope": "We need all user profiles, activity logs, billing records, and third-party sharing records.",
    "legal_basis": "This is a GDPR Article 15 Subject Access Request. We're an EU-based organization.",
    "api_endpoint": "The errors are on POST /v2/orders and GET /v2/inventory endpoints.",
    "request_volume": "We're making approximately 200 requests per minute, which is well within our plan limits.",
    "error_pattern": "The 429 errors appear in bursts — ~30% fail for 2-3 minutes, then it works for 5 minutes.",
    "compromised_token_id": "The compromised token is tok_api_prod_7x9m2_2025. It was created 6 months ago.",
    "affected_user_list": "So far we've identified users #4821, #4822, and #4830 with unauthorized data exports.",
    "export_timestamps": "The unauthorized exports happened at 02:15, 02:22, and 02:31 UTC today.",
    "audit_log_range": "Attaching audit logs from 2025-04-09T00:00:00Z to 2025-04-10T12:00:00Z.",
    "dependency_health": "Redis cluster shows 2 of 3 nodes as unreachable. Primary node has 99% memory usage.",
}


def _generate_customer_reply(requested_fields: list[str], already_provided: set[str]) -> str | None:
    """Simulate a customer providing requested diagnostic information."""
    new_info = [f for f in requested_fields if f not in already_provided]
    if not new_info:
        return None

    parts = []
    for field in new_info[:2]:  # Customer provides at most 2 fields per response
        response = _CUSTOMER_RESPONSES.get(field)
        if response:
            parts.append(response)

    if not parts:
        return None

    return " ".join(parts)


class EnterpriseSupportTicketTriageEnv:
    """
    OpenEnv-compatible RL environment for enterprise support ticket triage.

    Supports 9 tasks across easy/medium/hard difficulty with stochastic variants,
    multi-turn customer simulation, shaped rewards, and anti-gaming grading.
    """

    def __init__(self, max_steps: int = 8):
        self.default_max_steps = max_steps
        self._state: EpisodeState | None = None
        self._task_cursor = 0
        self._episode_counter = 0
        self._customer_provided_fields: set[str] = set()

    def reset(self, task_id: str | None = None, seed: int | None = None, **kwargs) -> Observation:
        if task_id is None:
            task_id = TASK_ORDER[self._task_cursor % len(TASK_ORDER)]
            self._task_cursor += 1

        task = get_task(task_id)

        # Apply stochastic variant if seed is provided
        if seed is not None:
            task = generate_variant(task, seed)

        self._episode_counter += 1
        self._customer_provided_fields = set()

        effective_max_steps = min(self.default_max_steps, task.max_steps)
        ticket_id = f"TKT-{self._episode_counter:04d}-{task.difficulty.upper()}"

        now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        progress_flags = {
            "classified_correct": False,
            "queue_correct": False,
            "priority_correct": False,
            "acknowledged_billing_issue": False,
            "advised_password_reset": False,
            "escalation_triggered": False,
            "closed_correctly": False,
            "premature_close": False,
            "unsafe_behavior": False,
        }

        self._state = EpisodeState(
            task_id=task.task_id,
            difficulty=task.difficulty,
            ticket_id=ticket_id,
            subject=task.subject,
            customer_message=task.customer_message,
            extracted_entities=task.extracted_entities,
            customer_metadata=task.customer_metadata,
            attachment_refs=task.attachment_refs,
            sla_deadline_minutes=task.sla_deadline_minutes,
            ground_truth_category=task.ground_truth_category,
            gold_queue=task.gold_queue,
            gold_priority=task.gold_priority,
            gold_resolution_path=task.gold_resolution_path,
            required_diagnostic_fields=list(task.required_diagnostic_fields),
            requested_fields=[],
            current_status=TicketStatus.OPEN,
            current_queue="triage-intake",
            current_priority=PriorityLevel.NORMAL,
            progress_flags=progress_flags,
            episode_metadata={
                "task_title": task.title,
                "task_description": task.description,
                "classification_attempts": 0,
                "last_signature": "",
                "repeat_count": 0,
                "milestones_awarded": {},
                "created_at": now_iso,
            },
            conversation_history=[
                ConversationTurn(speaker="customer", message=task.customer_message, timestamp=now_iso),
            ],
            step_count=0,
            max_steps=effective_max_steps,
            done=False,
        )
        return self._observation()

    def step(self, action: Action | dict[str, Any]) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")

        if self._state.done:
            noop_reward = Reward(value=0.0, components={"episode_complete": 0.0}, reason="episode already finished")
            return self._observation(), noop_reward, True, self._info(noop_reward)

        task = get_task(self._state.task_id)
        components: dict[str, float] = {}

        parsed_action, parse_error = self._coerce_action(action)
        self._state.step_count += 1

        if parse_error:
            self._state.last_action_error = parse_error
            components["invalid_action"] = -0.25
        else:
            assert parsed_action is not None
            self._state.last_action_error = None
            self._apply_action(parsed_action, task, components)
            self._apply_repeat_penalty(parsed_action, components)
            self._state.action_history.append(parsed_action)

        if not self._state.done and self._state.step_count >= self._state.max_steps:
            self._state.done = True
            components["max_steps_timeout"] = components.get("max_steps_timeout", 0.0) - 0.10

        reward_value = clip(sum(components.values()), -1.0, 1.0)
        reward_reason = "; ".join(f"{name}:{value:+.2f}" for name, value in sorted(components.items()))
        reward = Reward(value=reward_value, components=components, reason=reward_reason)
        self._state.reward_history.append(reward)

        if self._state.done and self._state.final_score is None:
            self._state.final_score = grade_task(self._state, task)

        return self._observation(), reward, self._state.done, self._info(reward)

    def state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("reset() must be called before state()")
        return self._state.model_copy(deep=True)

    def close(self) -> None:
        """Clean up environment resources. Required by OpenEnv server interface."""
        self._state = None
        self._customer_provided_fields.clear()

    def get_task_ids(self) -> list[str]:
        """Return all available task IDs."""
        return list(TASK_ORDER)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _coerce_action(self, action: Action | dict[str, Any]) -> tuple[Action | None, str | None]:
        try:
            if isinstance(action, Action):
                return action, None
            return Action.model_validate(action), None
        except ValidationError as exc:
            return None, f"invalid action schema: {exc.errors()}"

    def _observation(self) -> Observation:
        assert self._state is not None
        missing_fields = [
            field
            for field in self._state.required_diagnostic_fields
            if field not in set(self._state.requested_fields)
        ]
        return Observation(
            task_id=self._state.task_id,
            difficulty=self._state.difficulty,
            ticket_id=self._state.ticket_id,
            subject=self._state.subject,
            customer_message=self._state.customer_message,
            extracted_entities=self._state.extracted_entities,
            customer_metadata=self._state.customer_metadata,
            attachment_refs=self._state.attachment_refs,
            sla_deadline_minutes=self._state.sla_deadline_minutes,
            current_status=self._state.current_status,
            current_queue=self._state.current_queue,
            current_priority=self._state.current_priority,
            conversation_history=self._state.conversation_history,
            required_missing_fields=missing_fields,
            last_action_error=self._state.last_action_error,
            step_count=self._state.step_count,
            max_steps_remaining=max(0, self._state.max_steps - self._state.step_count),
        )

    def _info(self, reward: Reward) -> dict[str, Any]:
        assert self._state is not None
        return {
            "task_id": self._state.task_id,
            "difficulty": self._state.difficulty,
            "task_score": self._state.final_score if self._state.done else None,
            "current_status": self._state.current_status.value,
            "reward_components": reward.components,
            "requested_fields": list(self._state.requested_fields),
        }

    def _apply_action(self, action: Action, task, components: dict[str, float]) -> None:
        handlers = {
            ActionType.CLASSIFY: self._handle_classify,
            ActionType.ASSIGN_QUEUE: self._handle_assign_queue,
            ActionType.UPDATE_PRIORITY: self._handle_update_priority,
            ActionType.REQUEST_INFO: self._handle_request_info,
            ActionType.REPLY: self._handle_reply,
            ActionType.ESCALATE: self._handle_escalate,
            ActionType.CLOSE: self._handle_close,
        }
        handlers[action.action_type](action, task, components)

    def _award_milestone(
        self,
        milestone: str,
        components: dict[str, float],
        key: str,
        first_value: float,
        repeat_value: float = 0.0,
    ) -> bool:
        """
        Award trajectory-shaping reward only once for a semantic milestone.
        This prevents reward farming by repeating already-completed actions.
        """
        assert self._state is not None
        milestones = self._state.episode_metadata.setdefault("milestones_awarded", {})
        if milestones.get(milestone, False):
            if repeat_value != 0.0:
                components[f"{key}_repeat"] = repeat_value
            return False

        milestones[milestone] = True
        components[key] = first_value
        return True

    def _maybe_simulate_customer(self, newly_requested: list[str]) -> None:
        """After a request_info, simulate the customer providing some diagnostic info."""
        assert self._state is not None
        reply = _generate_customer_reply(newly_requested, self._customer_provided_fields)
        if reply:
            self._customer_provided_fields.update(newly_requested[:2])
            now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            self._state.conversation_history.append(
                ConversationTurn(speaker="customer", message=reply, timestamp=now_iso)
            )

    # ── Action handlers ───────────────────────────────────────────────────

    def _handle_classify(self, action: Action, task, components: dict[str, float]) -> None:
        assert self._state is not None
        guessed_category = self._infer_category(action)
        self._state.episode_metadata["classification_attempts"] += 1

        if guessed_category is None:
            components["classify_missing_intent"] = -0.08
            return

        if guessed_category == task.ground_truth_category:
            self._state.progress_flags["classified_correct"] = True
            self._award_milestone(
                "classify_correct",
                components,
                key="classify_correct",
                first_value=0.22,
                repeat_value=-0.03,
            )
        else:
            components["classify_incorrect"] = -0.10

    def _handle_assign_queue(self, action: Action, task, components: dict[str, float]) -> None:
        assert self._state is not None
        if not action.queue:
            components["queue_missing"] = -0.08
            self._state.last_action_error = "assign_queue requires `queue`"
            return

        self._state.current_queue = action.queue
        is_correct = normalize_text(action.queue) == normalize_text(task.gold_queue)
        self._state.progress_flags["queue_correct"] = is_correct

        if is_correct:
            self._award_milestone(
                "queue_correct",
                components,
                key="queue_correct",
                first_value=0.18,
                repeat_value=-0.03,
            )
        else:
            components["queue_incorrect"] = -0.14

    def _handle_update_priority(self, action: Action, task, components: dict[str, float]) -> None:
        assert self._state is not None
        if action.priority is None:
            components["priority_missing"] = -0.06
            self._state.last_action_error = "update_priority requires `priority`"
            return

        self._state.current_priority = action.priority
        is_correct = action.priority == task.gold_priority
        self._state.progress_flags["priority_correct"] = is_correct

        if is_correct:
            first_reward = 0.16 if task.difficulty == "hard" else 0.12
            self._award_milestone(
                "priority_correct",
                components,
                key="priority_correct",
                first_value=first_reward,
                repeat_value=-0.03,
            )
        else:
            components["priority_incorrect"] = -0.12

    def _handle_request_info(self, action: Action, task, components: dict[str, float]) -> None:
        assert self._state is not None

        if not action.message and not action.tags:
            components["request_info_empty"] = -0.05
            return

        now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Append agent message FIRST (before customer simulation)
        if action.message:
            self._state.conversation_history.append(
                ConversationTurn(speaker="agent", message=action.message, timestamp=now_iso)
            )

        detected_fields = detect_requested_fields(
            action.message,
            action.tags,
            self._state.required_diagnostic_fields,
        )
        new_fields = [field for field in detected_fields if field not in self._state.requested_fields]
        if new_fields:
            self._state.requested_fields.extend(new_fields)
            components["request_required_info"] = min(0.30, 0.10 * len(new_fields))
            # Simulate customer responding with the requested information
            self._maybe_simulate_customer(new_fields)
        elif self._state.required_diagnostic_fields:
            components["request_irrelevant_info"] = -0.04

        self._state.current_status = TicketStatus.PENDING

        if task.task_id == "medium_security_suspicious_login":
            if contains_any(action.message, ["verify", "confirm", "identity", "mfa"]):
                components["request_verification_context"] = 0.05


    def _handle_reply(self, action: Action, task, components: dict[str, float]) -> None:
        assert self._state is not None

        message = action.message.strip()
        if not message:
            components["reply_empty"] = -0.05
            return

        now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._state.conversation_history.append(
            ConversationTurn(speaker="agent", message=message, timestamp=now_iso)
        )

        # ── Easy billing ──
        if task.task_id == "easy_billing_duplicate_charge":
            has_billing_ack = contains_any(message, ["duplicate charge", "duplicate", "charged twice"]) and contains_any(
                message, ["refund", "credit"]
            )
            if has_billing_ack:
                self._state.progress_flags["acknowledged_billing_issue"] = True
                self._award_milestone(
                    "billing_acknowledged",
                    components,
                    key="billing_acknowledged",
                    first_value=0.16,
                    repeat_value=-0.02,
                )
            if contains_any(message, ["next steps", "investigate", "we will follow up", "timeline"]):
                self._award_milestone(
                    "clear_next_steps",
                    components,
                    key="clear_next_steps",
                    first_value=0.08,
                    repeat_value=-0.01,
                )

        # ── Easy password reset ──
        if task.task_id == "easy_password_reset_request":
            if contains_any(message, ["reset", "password reset", "reset link", "forgot password"]):
                self._state.progress_flags["advised_password_reset"] = True
                self._award_milestone(
                    "password_reset_guidance",
                    components,
                    key="password_reset_guidance",
                    first_value=0.14,
                    repeat_value=-0.02,
                )
            if contains_any(message, ["verify", "identity", "confirm your"]):
                self._award_milestone(
                    "identity_verification",
                    components,
                    key="identity_verification",
                    first_value=0.06,
                    repeat_value=-0.01,
                )

        # ── Easy feature request ──
        if task.task_id == "easy_feature_request":
            if contains_any(message, ["feature request", "product feedback", "roadmap", "acknowledged", "noted"]):
                self._award_milestone(
                    "feature_acknowledged",
                    components,
                    key="feature_acknowledged",
                    first_value=0.14,
                    repeat_value=-0.02,
                )
            if contains_any(message, ["timeline", "review", "consider", "prioritize", "team will"]):
                self._award_milestone(
                    "feature_expectation_set",
                    components,
                    key="feature_expectation_set",
                    first_value=0.06,
                    repeat_value=-0.01,
                )

        # ── Medium security ──
        if task.task_id == "medium_security_suspicious_login":
            if contains_any(message, ["password reset", "reset your password", "credential reset"]):
                self._state.progress_flags["advised_password_reset"] = True
                self._award_milestone(
                    "security_reset_guidance",
                    components,
                    key="security_reset_guidance",
                    first_value=0.14,
                    repeat_value=-0.02,
                )
            if contains_any(message, ["verify", "confirm", "identity", "mfa"]):
                self._award_milestone(
                    "security_verification_guidance",
                    components,
                    key="security_verification_guidance",
                    first_value=0.06,
                    repeat_value=-0.01,
                )

        # ── Medium data export ──
        if task.task_id == "medium_data_export_compliance":
            if contains_any(message, ["gdpr", "data export", "article 15", "data subject"]):
                self._award_milestone(
                    "compliance_acknowledged",
                    components,
                    key="compliance_acknowledged",
                    first_value=0.12,
                    repeat_value=-0.02,
                )
            if contains_any(message, ["timeline", "30 day", "processing time", "within"]):
                self._award_milestone(
                    "compliance_timeline",
                    components,
                    key="compliance_timeline",
                    first_value=0.06,
                    repeat_value=-0.01,
                )

        # ── Medium API rate limiting ──
        if task.task_id == "medium_api_rate_limiting":
            if contains_any(message, ["rate limit", "429", "throttl", "quota"]):
                self._award_milestone(
                    "rate_limit_acknowledged",
                    components,
                    key="rate_limit_acknowledged",
                    first_value=0.12,
                    repeat_value=-0.02,
                )
            if contains_any(message, ["investigate", "check", "monitor", "reviewing"]):
                self._award_milestone(
                    "rate_limit_investigation",
                    components,
                    key="rate_limit_investigation",
                    first_value=0.06,
                    repeat_value=-0.01,
                )

        # ── Hard production outage ──
        if task.task_id == "hard_production_outage":
            if contains_any(message, ["incident", "outage", "sev", "critical", "impact"]):
                self._award_milestone(
                    "incident_severity_handling",
                    components,
                    key="incident_severity_handling",
                    first_value=0.08,
                    repeat_value=-0.01,
                )
            if contains_any(message, ["escalate", "on-call", "incident commander", "bridge"]):
                self._award_milestone(
                    "incident_escalation_language",
                    components,
                    key="incident_escalation_language",
                    first_value=0.06,
                    repeat_value=-0.01,
                )

            detected_fields = detect_requested_fields(
                message,
                action.tags,
                self._state.required_diagnostic_fields,
            )
            new_fields = [field for field in detected_fields if field not in self._state.requested_fields]
            if new_fields:
                self._state.requested_fields.extend(new_fields)
                components["diagnostic_questions_in_reply"] = min(0.20, 0.06 * len(new_fields))

        # ── Hard data breach ──
        if task.task_id == "hard_data_breach_investigation":
            if contains_any(message, ["breach", "unauthorized", "compromised", "incident"]):
                self._award_milestone(
                    "breach_acknowledged",
                    components,
                    key="breach_acknowledged",
                    first_value=0.08,
                    repeat_value=-0.01,
                )
            if contains_any(message, ["contain", "revoke", "disable", "block", "isolate"]):
                self._award_milestone(
                    "containment_guidance",
                    components,
                    key="containment_guidance",
                    first_value=0.10,
                    repeat_value=-0.01,
                )

        # ── Hard cascade failure ──
        if task.task_id == "hard_multi_service_cascade_failure":
            if contains_any(message, ["cascade", "multiple services", "dependency", "infrastructure"]):
                self._award_milestone(
                    "cascade_acknowledged",
                    components,
                    key="cascade_acknowledged",
                    first_value=0.08,
                    repeat_value=-0.01,
                )
            if contains_any(message, ["incident", "bridge", "commander", "on-call"]):
                self._award_milestone(
                    "cascade_escalation_language",
                    components,
                    key="cascade_escalation_language",
                    first_value=0.06,
                    repeat_value=-0.01,
                )

        # ── Global checks ──
        if contains_any(message, task.discouraged_keywords):
            components["discouraged_content"] = -0.10

        if contains_any(message, ["share your password", "disable mfa", "ignore this alert",
                                   "send credentials", "turn off security"]):
            self._state.progress_flags["unsafe_behavior"] = True
            components["unsafe_behavior"] = -0.35

    def _handle_escalate(self, action: Action, task, components: dict[str, float]) -> None:
        assert self._state is not None
        self._state.progress_flags["escalation_triggered"] = True
        self._state.escalation_triggered = True

        if task.must_escalate:
            required = set(self._state.required_diagnostic_fields)
            requested = set(self._state.requested_fields)
            coverage_ratio = 1.0 if not required else len(required.intersection(requested)) / len(required)

            base = 0.20
            if self._state.current_priority == task.gold_priority:
                base += 0.08
            if normalize_text(self._state.current_queue) == normalize_text(task.gold_queue):
                base += 0.05

            components["correct_escalation"] = base + (0.07 * coverage_ratio)
            self._state.current_status = TicketStatus.ESCALATED
            self._state.done = True
        else:
            components["unnecessary_escalation"] = -0.18

    def _handle_close(self, action: Action, task, components: dict[str, float]) -> None:
        assert self._state is not None

        if task.allow_close:
            ready_to_close = self._state.progress_flags.get("acknowledged_billing_issue", False) and self._state.progress_flags.get(
                "queue_correct", False
            )
            # For non-billing easy tasks, check different readiness criteria
            if task.task_id == "easy_password_reset_request":
                ready_to_close = (
                    self._state.progress_flags.get("advised_password_reset", False)
                    and self._state.progress_flags.get("queue_correct", False)
                )
            elif task.task_id == "easy_feature_request":
                ready_to_close = self._state.progress_flags.get("queue_correct", False)

            if ready_to_close:
                self._state.current_status = TicketStatus.CLOSED
                self._state.progress_flags["closed_correctly"] = True
                components["correct_closure"] = 0.40
                if action.resolution_code and "duplicate" in normalize_text(action.resolution_code):
                    components["resolution_code_bonus"] = 0.05
                elif action.resolution_code:
                    components["resolution_code_bonus"] = 0.03
            else:
                self._state.current_status = TicketStatus.CLOSED
                self._state.closed_prematurely = True
                self._state.progress_flags["premature_close"] = True
                components["premature_closure"] = -0.35
            self._state.done = True
            return

        self._state.current_status = TicketStatus.CLOSED
        self._state.closed_prematurely = True
        self._state.progress_flags["premature_close"] = True
        self._state.done = True
        components["unsafe_close"] = -0.40

    def _apply_repeat_penalty(self, action: Action, components: dict[str, float]) -> None:
        assert self._state is not None

        signature = action_signature(
            action.action_type.value,
            action.message,
            action.queue,
            action.priority.value if action.priority else None,
        )
        last_signature = self._state.episode_metadata.get("last_signature", "")

        if signature == last_signature:
            repeat_count = int(self._state.episode_metadata.get("repeat_count", 0)) + 1
            self._state.episode_metadata["repeat_count"] = repeat_count
            # Escalating quadratic penalty for repeated actions
            components["repeat_action_penalty"] = max(-0.30, -0.05 * repeat_count * repeat_count)
        else:
            self._state.episode_metadata["repeat_count"] = 0

        self._state.episode_metadata["last_signature"] = signature

    def _infer_category(self, action: Action) -> TicketCategory | None:
        blob = normalize_text(" ".join(action.tags) + " " + action.message)
        if not blob.strip():
            return None

        if any(keyword in blob for keyword in ["billing", "invoice", "refund", "duplicate charge", "charge"]):
            return TicketCategory.BILLING
        if any(keyword in blob for keyword in ["security", "suspicious login", "credential", "compromise",
                                                 "breach", "unauthorized"]):
            return TicketCategory.SECURITY
        if any(keyword in blob for keyword in ["incident", "outage", "503", "502", "500", "production", "sev",
                                                 "deployment", "cascade", "rate limit", "429"]):
            return TicketCategory.INCIDENT
        if any(keyword in blob for keyword in ["feature", "feedback", "password reset", "locked out",
                                                 "gdpr", "compliance", "data export", "access issue"]):
            return TicketCategory.GENERAL
        return TicketCategory.GENERAL


if __name__ == "__main__":
    env = EnterpriseSupportTicketTriageEnv()
    obs = env.reset("easy_billing_duplicate_charge")
    print("Environment initialized")
    print(obs.model_dump_json(indent=2))
