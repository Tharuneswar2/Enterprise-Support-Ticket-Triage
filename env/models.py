from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    CLASSIFY = "classify"
    REPLY = "reply"
    ESCALATE = "escalate"
    CLOSE = "close"
    REQUEST_INFO = "request_info"
    UPDATE_PRIORITY = "update_priority"
    ASSIGN_QUEUE = "assign_queue"


class TicketStatus(str, Enum):
    NEW = "new"
    OPEN = "open"
    PENDING = "pending"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class PriorityLevel(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TicketCategory(str, Enum):
    BILLING = "billing"
    SECURITY = "security"
    INCIDENT = "incident"
    GENERAL = "general"


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker: str
    message: str


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    message: str = ""
    queue: str | None = None
    priority: PriorityLevel | None = None
    tags: list[str] = Field(default_factory=list)
    resolution_code: str | None = None


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    components: dict[str, float] = Field(default_factory=dict)
    reason: str = ""


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: str
    ticket_id: str
    subject: str
    customer_message: str
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    current_status: TicketStatus
    current_queue: str
    current_priority: PriorityLevel
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    required_missing_fields: list[str] = Field(default_factory=list)
    last_action_error: str | None = None
    step_count: int
    max_steps_remaining: int


class EpisodeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: str

    ticket_id: str
    subject: str
    customer_message: str
    extracted_entities: dict[str, Any] = Field(default_factory=dict)

    ground_truth_category: TicketCategory
    gold_queue: str
    gold_priority: PriorityLevel
    gold_resolution_path: list[str] = Field(default_factory=list)

    required_diagnostic_fields: list[str] = Field(default_factory=list)
    requested_fields: list[str] = Field(default_factory=list)

    current_status: TicketStatus
    current_queue: str
    current_priority: PriorityLevel

    progress_flags: dict[str, bool] = Field(default_factory=dict)
    episode_metadata: dict[str, Any] = Field(default_factory=dict)

    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    action_history: list[Action] = Field(default_factory=list)
    reward_history: list[Reward] = Field(default_factory=list)

    step_count: int
    max_steps: int
    done: bool = False
    last_action_error: str | None = None

    closed_prematurely: bool = False
    escalation_triggered: bool = False
    final_score: float | None = None
