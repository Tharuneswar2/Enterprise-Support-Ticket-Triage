from __future__ import annotations

from .models import ActionType, EpisodeState, TicketStatus
from .tasks import TaskSpec
from .utils import (
    clip,
    contains_any,
    keyword_density_score,
    message_has_substance,
    normalize_text,
    workflow_order_score,
)


# ---------------------------------------------------------------------------
# Shared grading helpers
# ---------------------------------------------------------------------------

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


def _action_type_sequence(state: EpisodeState) -> list[str]:
    return [action.action_type.value for action in state.action_history]


def _anti_gaming_penalty(state: EpisodeState, task: TaskSpec) -> float:
    """
    Penalize agents that dump keywords without contextual reasoning.
    Checks for keyword stuffing, lack of substance, and suspicious patterns.
    """
    penalty = 0.0
    all_keywords = task.required_response_keywords + list(task.extracted_entities.values())

    for action in state.action_history:
        if action.message:
            density = keyword_density_score(action.message, all_keywords)
            if density > 0.6:
                penalty -= 0.08  # Keyword stuffing detected
            if not message_has_substance(action.message, min_words=4):
                penalty -= 0.03  # Too short to be meaningful

    return max(-0.20, penalty)


def _workflow_bonus(state: EpisodeState) -> float:
    """Reward agents that follow a logical workflow order."""
    seq = _action_type_sequence(state)
    score = workflow_order_score(seq, state.difficulty)
    return round(score * 0.10, 4)  # Up to +0.10 for perfect ordering


def _safety_check(state: EpisodeState) -> float:
    """Penalize unsafe behaviors across all tasks."""
    text = _all_agent_text(state)
    penalty = 0.0
    unsafe_phrases = [
        "share your password",
        "disable mfa",
        "ignore this alert",
        "send me your credentials",
        "turn off security",
    ]
    if contains_any(text, unsafe_phrases):
        penalty -= 0.15
    return penalty


# ---------------------------------------------------------------------------
# Per-task graders
# ---------------------------------------------------------------------------

def grade_easy_billing(state: EpisodeState, task: TaskSpec) -> float:
    text = _all_agent_text(state)
    flags = state.progress_flags

    score = 0.0
    score += 0.18 if flags.get("classified_correct", False) else 0.0
    score += 0.20 if flags.get("queue_correct", False) else 0.0

    intent_keywords_met = all(
        contains_any(text, [keyword]) for keyword in ["duplicate charge", "refund"]
    )
    score += 0.18 if intent_keywords_met else 0.0

    mentions_next_step = contains_any(text, ["next steps", "we will", "investigate", "follow up"])
    score += 0.08 if mentions_next_step else 0.0

    no_unnecessary_escalation = not flags.get("escalation_triggered", False)
    score += 0.08 if no_unnecessary_escalation else 0.0

    if flags.get("closed_correctly", False):
        score += 0.13
    elif not flags.get("premature_close", False):
        score += 0.05

    # Structural bonuses
    score += _workflow_bonus(state)
    score += _anti_gaming_penalty(state, task)
    score += _safety_check(state)

    return round(clip(score, 0.0, 1.0), 4)


def grade_easy_password_reset(state: EpisodeState, task: TaskSpec) -> float:
    text = _all_agent_text(state)
    flags = state.progress_flags

    score = 0.0
    score += 0.18 if flags.get("classified_correct", False) else 0.0
    score += 0.20 if flags.get("queue_correct", False) else 0.0

    field_ratio = _required_field_ratio(state)
    score += 0.20 * field_ratio

    advises_reset = contains_any(text, ["reset", "password reset", "reset link"])
    score += 0.12 if advises_reset else 0.0

    asks_verification = contains_any(text, ["verify", "identity", "confirm"])
    score += 0.08 if asks_verification else 0.0

    if flags.get("closed_correctly", False):
        score += 0.10
    elif not flags.get("premature_close", False):
        score += 0.04

    score += _workflow_bonus(state)
    score += _anti_gaming_penalty(state, task)
    score += _safety_check(state)

    return round(clip(score, 0.0, 1.0), 4)


def grade_easy_feature_request(state: EpisodeState, task: TaskSpec) -> float:
    text = _all_agent_text(state)
    flags = state.progress_flags

    score = 0.0
    score += 0.18 if flags.get("classified_correct", False) else 0.0
    score += 0.20 if flags.get("queue_correct", False) else 0.0

    acknowledges = contains_any(text, ["feature request", "product feedback", "roadmap", "acknowledged"])
    score += 0.15 if acknowledges else 0.0

    sets_expectation = contains_any(text, ["timeline", "review", "consider", "prioritize", "team"])
    score += 0.10 if sets_expectation else 0.0

    correct_priority = flags.get("priority_correct", False) or state.current_priority.value == "low"
    score += 0.08 if correct_priority else 0.0

    if flags.get("closed_correctly", False):
        score += 0.12
    elif not flags.get("premature_close", False):
        score += 0.04

    score += _workflow_bonus(state)
    score += _anti_gaming_penalty(state, task)

    return round(clip(score, 0.0, 1.0), 4)


def grade_medium_security(state: EpisodeState, task: TaskSpec) -> float:
    text = _all_agent_text(state)
    flags = state.progress_flags

    score = 0.0
    score += 0.15 if flags.get("classified_correct", False) else 0.0
    score += 0.15 if flags.get("queue_correct", False) else 0.0
    score += 0.12 if flags.get("priority_correct", False) else 0.0

    field_ratio = _required_field_ratio(state)
    score += 0.20 * field_ratio

    advises_reset = contains_any(text, ["password reset", "reset your password", "credential reset"])
    score += 0.12 if advises_reset else 0.0

    asked_verification = contains_any(text, ["verify", "confirm", "identity", "mfa"])
    score += 0.05 if asked_verification else 0.0

    no_premature_close = not flags.get("premature_close", False)
    score += 0.05 if no_premature_close else 0.0

    correct_status_open = state.current_status in {TicketStatus.OPEN, TicketStatus.PENDING}
    score += 0.08 if correct_status_open else 0.0

    if _has_action(state, ActionType.CLOSE):
        score -= 0.10

    score += _workflow_bonus(state)
    score += _anti_gaming_penalty(state, task)
    score += _safety_check(state)

    return round(clip(score, 0.0, 1.0), 4)


def grade_medium_data_export(state: EpisodeState, task: TaskSpec) -> float:
    text = _all_agent_text(state)
    flags = state.progress_flags

    score = 0.0
    score += 0.15 if flags.get("classified_correct", False) else 0.0
    score += 0.15 if flags.get("queue_correct", False) else 0.0
    score += 0.12 if flags.get("priority_correct", False) else 0.0

    field_ratio = _required_field_ratio(state)
    score += 0.20 * field_ratio

    mentions_gdpr = contains_any(text, ["gdpr", "data export", "article 15", "data subject"])
    score += 0.10 if mentions_gdpr else 0.0

    mentions_timeline = contains_any(text, ["timeline", "30 day", "within", "processing time"])
    score += 0.08 if mentions_timeline else 0.0

    asks_verification = contains_any(text, ["verify", "identity", "authority", "authorized"])
    score += 0.06 if asks_verification else 0.0

    no_premature_close = not flags.get("premature_close", False)
    score += 0.05 if no_premature_close else 0.0

    if _has_action(state, ActionType.CLOSE):
        score -= 0.12

    score += _workflow_bonus(state)
    score += _anti_gaming_penalty(state, task)

    return round(clip(score, 0.0, 1.0), 4)


def grade_medium_api_rate_limiting(state: EpisodeState, task: TaskSpec) -> float:
    text = _all_agent_text(state)
    flags = state.progress_flags

    score = 0.0
    score += 0.15 if flags.get("classified_correct", False) else 0.0
    score += 0.15 if flags.get("queue_correct", False) else 0.0
    score += 0.12 if flags.get("priority_correct", False) else 0.0

    field_ratio = _required_field_ratio(state)
    score += 0.20 * field_ratio

    mentions_rate_limit = contains_any(text, ["rate limit", "429", "throttl"])
    score += 0.10 if mentions_rate_limit else 0.0

    mentions_api = contains_any(text, ["api", "endpoint", "request"])
    score += 0.06 if mentions_api else 0.0

    no_premature_close = not flags.get("premature_close", False)
    score += 0.05 if no_premature_close else 0.0

    correct_status = state.current_status in {TicketStatus.OPEN, TicketStatus.PENDING}
    score += 0.05 if correct_status else 0.0

    if _has_action(state, ActionType.CLOSE):
        score -= 0.10

    score += _workflow_bonus(state)
    score += _anti_gaming_penalty(state, task)

    return round(clip(score, 0.0, 1.0), 4)


def grade_hard_incident(state: EpisodeState, task: TaskSpec) -> float:
    text = _all_agent_text(state)
    flags = state.progress_flags

    score = 0.0
    score += 0.12 if flags.get("classified_correct", False) else 0.0
    score += 0.15 if flags.get("queue_correct", False) else 0.0
    score += 0.12 if flags.get("priority_correct", False) else 0.0

    field_ratio = _required_field_ratio(state)
    score += 0.22 * field_ratio

    escalated = flags.get("escalation_triggered", False) and _has_action(state, ActionType.ESCALATE)
    score += 0.18 if escalated else 0.0

    incident_language = contains_any(text, ["incident", "outage", "request id", "logs", "impact"])
    score += 0.05 if incident_language else 0.0

    no_premature_close = not flags.get("premature_close", False)
    score += 0.05 if no_premature_close else 0.0

    score += _workflow_bonus(state)
    score += _anti_gaming_penalty(state, task)
    score += _safety_check(state)

    return round(clip(score, 0.0, 1.0), 4)


def grade_hard_data_breach(state: EpisodeState, task: TaskSpec) -> float:
    text = _all_agent_text(state)
    flags = state.progress_flags

    score = 0.0
    score += 0.10 if flags.get("classified_correct", False) else 0.0
    score += 0.12 if flags.get("queue_correct", False) else 0.0
    score += 0.12 if flags.get("priority_correct", False) else 0.0

    field_ratio = _required_field_ratio(state)
    score += 0.22 * field_ratio

    escalated = flags.get("escalation_triggered", False) and _has_action(state, ActionType.ESCALATE)
    score += 0.18 if escalated else 0.0

    mentions_containment = contains_any(text, ["contain", "revoke", "disable", "block", "isolate"])
    score += 0.08 if mentions_containment else 0.0

    mentions_breach = contains_any(text, ["breach", "unauthorized", "compromised", "incident"])
    score += 0.05 if mentions_breach else 0.0

    no_premature_close = not flags.get("premature_close", False)
    score += 0.05 if no_premature_close else 0.0

    score += _workflow_bonus(state)
    score += _anti_gaming_penalty(state, task)
    score += _safety_check(state)

    return round(clip(score, 0.0, 1.0), 4)


def grade_hard_cascade_failure(state: EpisodeState, task: TaskSpec) -> float:
    text = _all_agent_text(state)
    flags = state.progress_flags

    score = 0.0
    score += 0.10 if flags.get("classified_correct", False) else 0.0
    score += 0.12 if flags.get("queue_correct", False) else 0.0
    score += 0.12 if flags.get("priority_correct", False) else 0.0

    field_ratio = _required_field_ratio(state)
    score += 0.22 * field_ratio

    escalated = flags.get("escalation_triggered", False) and _has_action(state, ActionType.ESCALATE)
    score += 0.18 if escalated else 0.0

    mentions_cascade = contains_any(text, ["cascade", "multiple services", "dependency", "infrastructure"])
    score += 0.06 if mentions_cascade else 0.0

    mentions_incident = contains_any(text, ["incident", "outage", "bridge", "commander"])
    score += 0.05 if mentions_incident else 0.0

    no_premature_close = not flags.get("premature_close", False)
    score += 0.05 if no_premature_close else 0.0

    score += _workflow_bonus(state)
    score += _anti_gaming_penalty(state, task)
    score += _safety_check(state)

    return round(clip(score, 0.0, 1.0), 4)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

_GRADER_MAP: dict[str, callable] = {
    "easy_billing_duplicate_charge": grade_easy_billing,
    "easy_password_reset_request": grade_easy_password_reset,
    "easy_feature_request": grade_easy_feature_request,
    "medium_security_suspicious_login": grade_medium_security,
    "medium_data_export_compliance": grade_medium_data_export,
    "medium_api_rate_limiting": grade_medium_api_rate_limiting,
    "hard_production_outage": grade_hard_incident,
    "hard_data_breach_investigation": grade_hard_data_breach,
    "hard_multi_service_cascade_failure": grade_hard_cascade_failure,
}


def grade_task(state: EpisodeState, task: TaskSpec) -> float:
    grader = _GRADER_MAP.get(task.task_id)
    if grader is None:
        return 0.0
    return grader(state, task)
