from __future__ import annotations

from .models import ActionType, EpisodeState, TicketStatus
from .tasks import TaskSpec
from .utils import clip, contains_any, normalize_text


def _all_agent_text(state: EpisodeState) -> str:
    messages = [action.message for action in state.action_history if action.message]
    return normalize_text(" ".join(messages))


def _has_action(state: EpisodeState, action_type: ActionType) -> bool:
    return any(action.action_type == action_type for action in state.action_history)


def _required_field_ratio(state: EpisodeState) -> float:
    required = set(state.required_diagnostic_fields)
    if not required:
        return 1.0
    found = set(state.requested_fields)
    return len(required.intersection(found)) / len(required)


def grade_easy_billing(state: EpisodeState, task: TaskSpec) -> float:
    text = _all_agent_text(state)
    flags = state.progress_flags

    score = 0.0
    score += 0.20 if flags.get("classified_correct", False) else 0.0
    score += 0.25 if flags.get("queue_correct", False) else 0.0

    intent_keywords_met = all(
        contains_any(text, [keyword]) for keyword in ["duplicate charge", "refund"]
    )
    score += 0.20 if intent_keywords_met else 0.0

    mentions_next_step = contains_any(text, ["next steps", "we will", "investigate", "follow up"])
    score += 0.10 if mentions_next_step else 0.0

    no_unnecessary_escalation = not flags.get("escalation_triggered", False)
    score += 0.10 if no_unnecessary_escalation else 0.0

    if flags.get("closed_correctly", False):
        score += 0.15
    elif not flags.get("premature_close", False):
        score += 0.05

    return round(clip(score, 0.0, 1.0), 4)


def grade_medium_security(state: EpisodeState, task: TaskSpec) -> float:
    text = _all_agent_text(state)
    flags = state.progress_flags

    score = 0.0
    score += 0.20 if flags.get("classified_correct", False) else 0.0
    score += 0.20 if flags.get("queue_correct", False) else 0.0
    score += 0.15 if flags.get("priority_correct", False) else 0.0

    field_ratio = _required_field_ratio(state)
    score += 0.20 * field_ratio

    advises_reset = contains_any(text, ["password reset", "reset your password", "credential reset"])
    score += 0.15 if advises_reset else 0.0

    asked_verification = contains_any(text, ["verify", "confirm", "identity", "mfa"])
    score += 0.05 if asked_verification else 0.0

    no_premature_close = not flags.get("premature_close", False)
    score += 0.05 if no_premature_close else 0.0

    correct_status_open = state.current_status in {TicketStatus.OPEN, TicketStatus.PENDING}
    score += 0.10 if correct_status_open else 0.0

    if _has_action(state, ActionType.CLOSE):
        score -= 0.10

    return round(clip(score, 0.0, 1.0), 4)


def grade_hard_incident(state: EpisodeState, task: TaskSpec) -> float:
    text = _all_agent_text(state)
    flags = state.progress_flags

    score = 0.0
    score += 0.15 if flags.get("classified_correct", False) else 0.0
    score += 0.20 if flags.get("queue_correct", False) else 0.0
    score += 0.15 if flags.get("priority_correct", False) else 0.0

    field_ratio = _required_field_ratio(state)
    score += 0.25 * field_ratio

    escalated = flags.get("escalation_triggered", False) and _has_action(state, ActionType.ESCALATE)
    score += 0.20 if escalated else 0.0

    incident_language = contains_any(text, ["incident", "outage", "request id", "logs", "impact"])
    score += 0.05 if incident_language else 0.0

    no_premature_close = not flags.get("premature_close", False)
    score += 0.05 if no_premature_close else 0.0

    return round(clip(score, 0.0, 1.0), 4)


def grade_task(state: EpisodeState, task: TaskSpec) -> float:
    if task.task_id == "easy_billing_duplicate_charge":
        return grade_easy_billing(state, task)
    if task.task_id == "medium_security_suspicious_login":
        return grade_medium_security(state, task)
    if task.task_id == "hard_production_outage":
        return grade_hard_incident(state, task)

    return 0.0
