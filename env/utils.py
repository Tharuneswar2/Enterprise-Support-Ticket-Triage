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


FIELD_SYNONYMS: dict[str, list[str]] = {
    "last_known_login_time": [
        "last login time",
        "time of suspicious login",
        "suspicious login time",
        "when did this login happen",
        "timestamp of login",
    ],
    "login_location": [
        "login location",
        "ip address",
        "geo location",
        "where was the login from",
        "location of login",
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
    ],
    "affected_service": [
        "affected service",
        "service name",
        "which service",
        "impacted component",
        "system affected",
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
