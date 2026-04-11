from __future__ import annotations

import json
import re
from typing import Iterable


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(keyword) in normalized for keyword in keywords)


def count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    normalized = normalize_text(text)
    return sum(1 for keyword in keywords if normalize_text(keyword) in normalized)


def action_signature(action_type: str, message: str, queue: str | None, priority: str | None) -> str:
    return "|".join(
        [
            action_type,
            normalize_text(message)[:120],
            (queue or "").strip().lower(),
            (priority or "").strip().lower(),
        ]
    )


# ---------------------------------------------------------------------------
# Field synonym lookup for diagnostic field detection
# ---------------------------------------------------------------------------

FIELD_SYNONYMS: dict[str, list[str]] = {
    "last_known_login_time": [
        "last login time",
        "time of suspicious login",
        "suspicious login time",
        "when did this login happen",
        "timestamp of login",
        "last login",
    ],
    "login_location": [
        "login location",
        "ip address",
        "geo location",
        "where was the login from",
        "location of login",
        "source ip",
    ],
    "mfa_status": [
        "mfa",
        "2fa",
        "two-factor",
        "multi-factor",
        "authentication method",
    ],
    "incident_start_time": [
        "incident start time",
        "when did this start",
        "start timestamp",
        "outage start",
        "first failure time",
        "when did the incident begin",
    ],
    "affected_service": [
        "affected service",
        "service name",
        "which service",
        "impacted component",
        "system affected",
        "what service",
    ],
    "request_id": [
        "request id",
        "trace id",
        "correlation id",
        "transaction id",
    ],
    "error_logs": [
        "error logs",
        "stack trace",
        "log snippet",
        "exception logs",
        "error trace",
        "logs",
    ],
    # New fields for expanded tasks
    "account_email": [
        "email address",
        "account email",
        "registered email",
        "email on file",
    ],
    "verification_method": [
        "verification",
        "verify identity",
        "how to verify",
        "proof of identity",
        "identity verification",
    ],
    "requester_role": [
        "role",
        "your role",
        "position",
        "job title",
        "authority",
        "who is requesting",
    ],
    "data_scope": [
        "data scope",
        "what data",
        "which records",
        "scope of export",
        "data types",
    ],
    "legal_basis": [
        "legal basis",
        "regulation",
        "gdpr",
        "article 15",
        "compliance requirement",
        "regulatory basis",
    ],
    "api_endpoint": [
        "api endpoint",
        "which endpoint",
        "url",
        "route",
        "api path",
    ],
    "request_volume": [
        "request volume",
        "requests per second",
        "rps",
        "traffic volume",
        "call frequency",
        "how many requests",
    ],
    "error_pattern": [
        "error pattern",
        "when do errors occur",
        "intermittent",
        "pattern",
        "error frequency",
    ],
    "compromised_token_id": [
        "token id",
        "compromised token",
        "api token",
        "which token",
        "token identifier",
    ],
    "affected_user_list": [
        "affected users",
        "user list",
        "which accounts",
        "impacted users",
        "compromised accounts",
    ],
    "export_timestamps": [
        "export time",
        "when was data exported",
        "export timestamp",
        "data export time",
    ],
    "audit_log_range": [
        "audit log",
        "log range",
        "time range",
        "audit trail",
        "log period",
    ],
    "dependency_health": [
        "dependency health",
        "redis status",
        "dependency status",
        "shared dependency",
        "upstream health",
        "infrastructure health",
    ],
}


def detect_requested_fields(message: str, tags: list[str], required_fields: list[str]) -> list[str]:
    normalized_message = normalize_text(message)
    normalized_tags = {normalize_text(tag) for tag in tags}

    detected: list[str] = []
    for field_name in required_fields:
        field_as_text = normalize_text(field_name.replace("_", " "))
        aliases = FIELD_SYNONYMS.get(field_name, []) + [field_as_text]

        matched_in_text = any(normalize_text(alias) in normalized_message for alias in aliases)
        matched_in_tags = field_name in tags or field_as_text in normalized_tags
        if matched_in_text or matched_in_tags:
            detected.append(field_name)

    return detected


def extract_json_object(raw_text: str) -> dict:
    """
    Best-effort JSON parser for LLM outputs.
    - Accepts plain JSON objects.
    - If extra text exists, extracts the first JSON object.
    """
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("empty model output")

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model output")

    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Anti-gaming utilities
# ---------------------------------------------------------------------------

def keyword_density_score(text: str, keywords: list[str]) -> float:
    """
    Compute how many of the keywords appear relative to total word count.
    A very high density (keywords but almost no other words) suggests gaming.
    Returns a value in [0, 1] where 1 = extremely suspicious.
    """
    words = normalize_text(text).split()
    if len(words) < 3:
        return 0.0
    hits = count_keyword_hits(text, keywords)
    return min(1.0, hits / max(1, len(words) / 3))


def message_has_substance(message: str, min_words: int = 5) -> bool:
    """Check if a message has meaningful content beyond just keywords."""
    words = normalize_text(message).split()
    return len(words) >= min_words


# ---------------------------------------------------------------------------
# Action sequence analysis
# ---------------------------------------------------------------------------

_EXPECTED_SEQUENCES: dict[str, list[str]] = {
    "easy": ["classify", "assign_queue", "reply", "close"],
    "medium": ["classify", "assign_queue", "update_priority", "request_info", "reply"],
    "hard": ["classify", "assign_queue", "update_priority", "request_info", "escalate"],
}


def workflow_order_score(action_types: list[str], difficulty: str) -> float:
    """
    Score how well the agent's action sequence follows the expected workflow.
    Returns 0.0 (completely disordered) to 1.0 (perfect order).
    """
    expected = _EXPECTED_SEQUENCES.get(difficulty, _EXPECTED_SEQUENCES["easy"])
    if not action_types:
        return 0.0

    # Compute longest common subsequence ratio
    matched = 0
    exp_idx = 0
    for act in action_types:
        if exp_idx < len(expected) and act == expected[exp_idx]:
            matched += 1
            exp_idx += 1
    return matched / len(expected) if expected else 0.0
