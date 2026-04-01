from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .models import PriorityLevel, TicketCategory


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: str
    title: str
    description: str

    subject: str
    customer_message: str
    extracted_entities: dict[str, str] = Field(default_factory=dict)

    ground_truth_category: TicketCategory
    gold_queue: str
    gold_priority: PriorityLevel
    gold_resolution_path: list[str] = Field(default_factory=list)

    required_diagnostic_fields: list[str] = Field(default_factory=list)
    required_response_keywords: list[str] = Field(default_factory=list)
    discouraged_keywords: list[str] = Field(default_factory=list)

    must_escalate: bool = False
    allow_close: bool = False
    max_steps: int = 8


TASKS: dict[str, TaskSpec] = {
    "easy_billing_duplicate_charge": TaskSpec(
        task_id="easy_billing_duplicate_charge",
        difficulty="easy",
        title="Billing Inquiry - Duplicate Charge",
        description=(
            "Customer was charged twice for one subscription cycle and requests correction. "
            "Agent should classify as billing, route to billing operations, acknowledge duplicate charge/refund path, "
            "and close only after a proper acknowledgement."
        ),
        subject="Charged twice for same invoice INV-88421",
        customer_message=(
            "Hi support, I was charged twice today for invoice INV-88421 for our Team plan. "
            "Can you fix this and refund one of the charges?"
        ),
        extracted_entities={
            "invoice_id": "INV-88421",
            "subscription_plan": "Team",
            "reported_issue": "duplicate_charge",
        },
        ground_truth_category=TicketCategory.BILLING,
        gold_queue="billing-ops",
        gold_priority=PriorityLevel.NORMAL,
        gold_resolution_path=["classify", "assign_queue", "reply", "close"],
        required_diagnostic_fields=[],
        required_response_keywords=["duplicate charge", "refund", "next steps"],
        discouraged_keywords=["security breach", "ignore"],
        must_escalate=False,
        allow_close=True,
        max_steps=6,
    ),
    "medium_security_suspicious_login": TaskSpec(
        task_id="medium_security_suspicious_login",
        difficulty="medium",
        title="Account Security - Suspicious Login",
        description=(
            "User reports suspicious login activity. Agent should classify as security, assign to security queue, "
            "set high priority, request verification details, and advise credential reset without closing prematurely."
        ),
        subject="Unrecognized login from another country",
        customer_message=(
            "I just got a login alert from a country where none of our admins are located. "
            "I think someone accessed our account. Please help immediately."
        ),
        extracted_entities={
            "alert_type": "suspicious_login",
            "account_tier": "enterprise",
            "reported_location": "unknown",
        },
        ground_truth_category=TicketCategory.SECURITY,
        gold_queue="security-response",
        gold_priority=PriorityLevel.HIGH,
        gold_resolution_path=["classify", "assign_queue", "update_priority", "request_info", "reply"],
        required_diagnostic_fields=["last_known_login_time", "login_location", "mfa_status"],
        required_response_keywords=["password reset", "verify", "security"],
        discouraged_keywords=["close", "share password"],
        must_escalate=False,
        allow_close=False,
        max_steps=8,
    ),
    "hard_production_outage": TaskSpec(
        task_id="hard_production_outage",
        difficulty="hard",
        title="Production Incident - Service Outage",
        description=(
            "A production outage blocks deployment and service availability. Agent must treat as a critical incident, "
            "set urgent priority, request diagnostics, route to platform incident queue, and escalate properly."
        ),
        subject="Production API returning 503 after deployment, checkout unavailable",
        customer_message=(
            "Our production checkout API started returning 503 right after deployment. "
            "Deployments are blocked and customers cannot complete purchases. "
            "This is impacting revenue right now."
        ),
        extracted_entities={
            "impact": "checkout_down",
            "environment": "production",
            "error_code": "503",
            "business_impact": "revenue_blocked",
        },
        ground_truth_category=TicketCategory.INCIDENT,
        gold_queue="platform-incident",
        gold_priority=PriorityLevel.URGENT,
        gold_resolution_path=[
            "classify",
            "assign_queue",
            "update_priority",
            "request_info",
            "escalate",
        ],
        required_diagnostic_fields=[
            "incident_start_time",
            "affected_service",
            "request_id",
            "error_logs",
        ],
        required_response_keywords=["incident", "request id", "logs", "escalate"],
        discouraged_keywords=["close", "wait a few days"],
        must_escalate=True,
        allow_close=False,
        max_steps=8,
    ),
}


TASK_ORDER: list[str] = [
    "easy_billing_duplicate_charge",
    "medium_security_suspicious_login",
    "hard_production_outage",
]


def get_task(task_id: str) -> TaskSpec:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"unknown task_id: {task_id}") from exc


def list_tasks() -> list[TaskSpec]:
    return [TASKS[task_id] for task_id in TASK_ORDER]
