from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SupportTriageAction(Action):
    action_type: Literal[
        "classify",
        "reply",
        "escalate",
        "close",
        "request_info",
        "update_priority",
        "assign_queue",
    ]
    message: str = ""
    queue: str | None = None
    priority: Literal["low", "normal", "high", "urgent"] | None = None
    tags: list[str] = Field(default_factory=list)
    resolution_code: str | None = None


class SupportTriageObservation(Observation):
    task_id: str
    difficulty: str
    ticket_id: str
    subject: str
    customer_message: str
    extracted_entities: dict = Field(default_factory=dict)
    current_status: str
    current_queue: str
    current_priority: str
    conversation_history: list[dict] = Field(default_factory=list)
    required_missing_fields: list[str] = Field(default_factory=list)
    last_action_error: str | None = None
    step_count: int
    max_steps_remaining: int
