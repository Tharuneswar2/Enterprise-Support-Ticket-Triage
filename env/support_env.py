from __future__ import annotations

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
from .tasks import TASK_ORDER, get_task
from .utils import action_signature, clip, contains_any, detect_requested_fields, normalize_text


class EnterpriseSupportTicketTriageEnv:
    """
    OpenEnv-compatible RL environment for enterprise support ticket triage.
    """

    def __init__(self, max_steps: int = 8):
        self.default_max_steps = max_steps
        self._state: EpisodeState | None = None
        self._task_cursor = 0
        self._episode_counter = 0

    def reset(self, task_id: str | None = None) -> Observation:
        if task_id is None:
            task_id = TASK_ORDER[self._task_cursor % len(TASK_ORDER)]
            self._task_cursor += 1

        task = get_task(task_id)
        self._episode_counter += 1

        effective_max_steps = min(self.default_max_steps, task.max_steps)
        ticket_id = f"TKT-{self._episode_counter:04d}-{task.difficulty.upper()}"

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
            },
            conversation_history=[ConversationTurn(speaker="customer", message=task.customer_message)],
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

        detected_fields = detect_requested_fields(
            action.message,
            action.tags,
            self._state.required_diagnostic_fields,
        )
        new_fields = [field for field in detected_fields if field not in self._state.requested_fields]
        if new_fields:
            self._state.requested_fields.extend(new_fields)
            components["request_required_info"] = min(0.30, 0.10 * len(new_fields))
        elif self._state.required_diagnostic_fields:
            components["request_irrelevant_info"] = -0.04

        self._state.current_status = TicketStatus.PENDING

        if task.task_id == "medium_security_suspicious_login":
            if contains_any(action.message, ["verify", "confirm", "identity", "mfa"]):
                components["request_verification_context"] = 0.05

        if action.message:
            self._state.conversation_history.append(ConversationTurn(speaker="agent", message=action.message))

    def _handle_reply(self, action: Action, task, components: dict[str, float]) -> None:
        assert self._state is not None

        message = action.message.strip()
        if not message:
            components["reply_empty"] = -0.05
            return

        self._state.conversation_history.append(ConversationTurn(speaker="agent", message=message))

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

        if contains_any(message, task.discouraged_keywords):
            components["discouraged_content"] = -0.10

        if contains_any(message, ["share your password", "disable mfa", "ignore this alert"]):
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
            if ready_to_close:
                self._state.current_status = TicketStatus.CLOSED
                self._state.progress_flags["closed_correctly"] = True
                components["correct_closure"] = 0.40
                if action.resolution_code and "duplicate" in normalize_text(action.resolution_code):
                    components["resolution_code_bonus"] = 0.05
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
            components["repeat_action_penalty"] = max(-0.15, -0.03 * repeat_count)
        else:
            self._state.episode_metadata["repeat_count"] = 0

        self._state.episode_metadata["last_signature"] = signature

    def _infer_category(self, action: Action) -> TicketCategory | None:
        blob = normalize_text(" ".join(action.tags) + " " + action.message)
        if not blob.strip():
            return None

        if any(keyword in blob for keyword in ["billing", "invoice", "refund", "duplicate charge", "charge"]):
            return TicketCategory.BILLING
        if any(keyword in blob for keyword in ["security", "suspicious login", "credential", "password", "compromise"]):
            return TicketCategory.SECURITY
        if any(keyword in blob for keyword in ["incident", "outage", "503", "production", "sev", "deployment"]):
            return TicketCategory.INCIDENT
        return TicketCategory.GENERAL


if __name__ == "__main__":
    env = EnterpriseSupportTicketTriageEnv()
    obs = env.reset("easy_billing_duplicate_charge")
    print("Environment initialized")
    print(obs.model_dump_json(indent=2))
