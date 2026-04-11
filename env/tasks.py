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
    customer_metadata: dict[str, str] = Field(default_factory=dict)
    attachment_refs: list[str] = Field(default_factory=list)
    sla_deadline_minutes: int | None = None

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


# ---------------------------------------------------------------------------
# Task variant templates – used by the stochastic variant system to randomize
# surface-level details while preserving the underlying task structure.
# ---------------------------------------------------------------------------

_CUSTOMER_TONES: dict[str, list[str]] = {
    "frustrated": [
        "I've been waiting for days and nobody has responded.",
        "This is unacceptable — we're paying enterprise rates for this.",
        "I need this resolved NOW.",
    ],
    "calm": [
        "I wanted to report an issue we noticed.",
        "Hi team, just flagging something for your attention.",
        "Could you please look into the following when you get a chance?",
    ],
    "panicked": [
        "URGENT — this is a critical issue impacting production!",
        "We are losing revenue every minute this is down!",
        "Please help, this is an emergency for our entire organization.",
    ],
}

_INVOICE_IDS = ["INV-88421", "INV-91034", "INV-77209", "INV-63582", "INV-50117"]
_PLAN_NAMES = ["Team", "Business", "Enterprise", "Starter", "Scale"]
_SERVICE_NAMES = ["checkout-api", "payment-gateway", "inventory-svc", "auth-service", "order-processor"]
_IP_ADDRESSES = ["185.42.103.7", "91.215.44.22", "203.0.113.55", "198.51.100.12", "45.33.17.89"]
_COUNTRIES = ["Russia", "China", "Brazil", "Nigeria", "Romania"]
_ERROR_CODES = ["503", "502", "500", "504", "520"]


def _vary_string(template: str, seed: int, replacements: dict[str, list[str]]) -> str:
    """Apply deterministic randomized replacements to a template string."""
    result = template
    for placeholder, options in replacements.items():
        if placeholder in result:
            idx = seed % len(options)
            result = result.replace(placeholder, options[idx])
            seed = (seed * 31 + 7) & 0xFFFFFFFF
    return result


def generate_variant(task: TaskSpec, seed: int) -> TaskSpec:
    """
    Create a surface-level variant of a task with randomized details.
    The underlying task structure (gold answers, required fields) stays the same,
    but customer messages, entity values, and tone vary per seed.
    """
    import copy
    variant = task.model_copy(deep=True)

    # Vary tone prefix
    tone_keys = list(_CUSTOMER_TONES.keys())
    tone = tone_keys[seed % len(tone_keys)]
    prefix = _CUSTOMER_TONES[tone][seed % len(_CUSTOMER_TONES[tone])]

    # Apply entity-level variations based on category
    s = seed
    if task.ground_truth_category == TicketCategory.BILLING:
        inv_id = _INVOICE_IDS[s % len(_INVOICE_IDS)]
        plan = _PLAN_NAMES[(s >> 3) % len(_PLAN_NAMES)]
        variant.extracted_entities = {
            **variant.extracted_entities,
            "invoice_id": inv_id,
            "subscription_plan": plan,
        }
        variant.customer_message = (
            f"{prefix} I was charged twice today for invoice {inv_id} on our {plan} plan. "
            f"Can you fix this and refund one of the charges?"
        )
        variant.subject = f"Charged twice for same invoice {inv_id}"

    elif task.ground_truth_category == TicketCategory.SECURITY:
        country = _COUNTRIES[s % len(_COUNTRIES)]
        ip = _IP_ADDRESSES[(s >> 2) % len(_IP_ADDRESSES)]
        variant.extracted_entities = {
            **variant.extracted_entities,
            "reported_location": country,
            "source_ip": ip,
        }
        variant.customer_message = (
            f"{prefix} I just got a login alert from {country} (IP: {ip}) where none of our "
            f"admins are located. I think someone accessed our account. Please help immediately."
        )
        variant.subject = f"Unrecognized login from {country}"

    elif task.ground_truth_category == TicketCategory.INCIDENT:
        svc = _SERVICE_NAMES[s % len(_SERVICE_NAMES)]
        err = _ERROR_CODES[(s >> 2) % len(_ERROR_CODES)]
        variant.extracted_entities = {
            **variant.extracted_entities,
            "affected_service": svc,
            "error_code": err,
        }
        variant.customer_message = (
            f"{prefix} Our production {svc} started returning {err} right after deployment. "
            f"Deployments are blocked and customers cannot complete purchases. "
            f"This is impacting revenue right now."
        )
        variant.subject = f"Production {svc} returning {err} after deployment"

    else:
        variant.customer_message = f"{prefix} {variant.customer_message}"

    return variant


# ---------------------------------------------------------------------------
# Static task definitions
# ---------------------------------------------------------------------------

TASKS: dict[str, TaskSpec] = {
    # ── EASY ──────────────────────────────────────────────────────────────
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
        customer_metadata={
            "account_tier": "business",
            "org_size": "50-200",
            "timezone": "America/New_York",
            "account_age_months": "18",
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

    "easy_password_reset_request": TaskSpec(
        task_id="easy_password_reset_request",
        difficulty="easy",
        title="Account Access - Password Reset",
        description=(
            "User requests a password reset for their enterprise account. Agent should classify as account issue, "
            "route to identity-ops queue, confirm identity verification steps, and guide through self-service reset."
        ),
        subject="Cannot log in - need password reset for admin account",
        customer_message=(
            "Hi, I'm locked out of my admin account (email: j.martinez@acmecorp.com). "
            "I've tried the 'forgot password' link but never received the reset email. "
            "Can you help me regain access? This is blocking my team's work."
        ),
        extracted_entities={
            "user_email": "j.martinez@acmecorp.com",
            "account_role": "admin",
            "reported_issue": "password_reset",
        },
        customer_metadata={
            "account_tier": "enterprise",
            "org_size": "200-1000",
            "timezone": "America/Chicago",
            "account_age_months": "36",
        },
        ground_truth_category=TicketCategory.GENERAL,
        gold_queue="identity-ops",
        gold_priority=PriorityLevel.NORMAL,
        gold_resolution_path=["classify", "assign_queue", "request_info", "reply", "close"],
        required_diagnostic_fields=["account_email", "verification_method"],
        required_response_keywords=["reset", "verify", "identity"],
        discouraged_keywords=["share password", "ignore"],
        must_escalate=False,
        allow_close=True,
        max_steps=6,
    ),

    "easy_feature_request": TaskSpec(
        task_id="easy_feature_request",
        difficulty="easy",
        title="Product Feedback - Feature Request",
        description=(
            "Customer submits a feature request for bulk user import. Agent should classify as feature request, "
            "route to product-feedback queue, acknowledge the request with a timeline expectation, and close."
        ),
        subject="Feature request: Bulk user import via CSV",
        customer_message=(
            "We have 500+ users to onboard and adding them one-by-one is extremely tedious. "
            "Could you add a CSV bulk import feature for user management? "
            "Other platforms we've evaluated have this. It's becoming a dealbreaker for us."
        ),
        extracted_entities={
            "requested_feature": "bulk_csv_import",
            "module": "user_management",
            "reported_issue": "feature_request",
        },
        customer_metadata={
            "account_tier": "enterprise",
            "org_size": "500+",
            "timezone": "Europe/London",
            "account_age_months": "6",
        },
        ground_truth_category=TicketCategory.GENERAL,
        gold_queue="product-feedback",
        gold_priority=PriorityLevel.LOW,
        gold_resolution_path=["classify", "assign_queue", "reply", "close"],
        required_diagnostic_fields=[],
        required_response_keywords=["feature request", "roadmap", "acknowledged"],
        discouraged_keywords=["bug", "outage", "security"],
        must_escalate=False,
        allow_close=True,
        max_steps=5,
    ),

    # ── MEDIUM ────────────────────────────────────────────────────────────
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
            "source_ip": "185.42.103.7",
            "login_timestamp": "2025-04-10T03:22:41Z",
        },
        customer_metadata={
            "account_tier": "enterprise",
            "org_size": "200-1000",
            "timezone": "America/Los_Angeles",
            "account_age_months": "24",
            "sla_tier": "premium",
        },
        attachment_refs=["auth_audit_log_2025-04-10.json"],
        sla_deadline_minutes=120,
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

    "medium_data_export_compliance": TaskSpec(
        task_id="medium_data_export_compliance",
        difficulty="medium",
        title="Compliance - GDPR Data Export Request",
        description=(
            "Customer requests a full data export under GDPR Article 15. Agent must classify as compliance, "
            "route to legal-compliance queue, set high priority, verify the requester's identity and authority, "
            "and provide a timeline. Must NOT close — compliance team must fulfill."
        ),
        subject="GDPR data export request - all user data for our organization",
        customer_message=(
            "Under GDPR Article 15, I'm requesting a complete export of all personal data your platform "
            "holds for our organization (Org ID: ORG-4821). This includes user profiles, activity logs, "
            "billing records, and any third-party data sharing records. "
            "We need this within the 30-day regulatory window. Please confirm receipt and provide a timeline."
        ),
        extracted_entities={
            "request_type": "gdpr_data_export",
            "regulation": "GDPR Article 15",
            "org_id": "ORG-4821",
            "reported_issue": "data_export_compliance",
        },
        customer_metadata={
            "account_tier": "enterprise",
            "org_size": "1000+",
            "timezone": "Europe/Berlin",
            "account_age_months": "42",
            "jurisdiction": "EU",
        },
        sla_deadline_minutes=4320,  # 3 days internal SLA
        ground_truth_category=TicketCategory.GENERAL,
        gold_queue="legal-compliance",
        gold_priority=PriorityLevel.HIGH,
        gold_resolution_path=["classify", "assign_queue", "update_priority", "request_info", "reply"],
        required_diagnostic_fields=["requester_role", "data_scope", "legal_basis"],
        required_response_keywords=["data export", "GDPR", "timeline", "verify"],
        discouraged_keywords=["ignore", "deny", "not possible"],
        must_escalate=False,
        allow_close=False,
        max_steps=8,
    ),

    "medium_api_rate_limiting": TaskSpec(
        task_id="medium_api_rate_limiting",
        difficulty="medium",
        title="Technical Support - API Rate Limiting",
        description=(
            "Customer reports unexpected API rate limiting causing intermittent 429 errors. "
            "Agent should classify as technical/incident, route to platform-support, gather API usage "
            "details and error patterns, and advise on rate limit configuration. Should NOT close."
        ),
        subject="API returning 429 Too Many Requests intermittently",
        customer_message=(
            "Our integration started getting 429 errors from your API about 2 hours ago. "
            "We haven't changed our request volume. Our API key is ak_prod_7x9m2. "
            "The errors are intermittent — roughly 30% of requests fail. "
            "This is breaking our automated order processing pipeline."
        ),
        extracted_entities={
            "error_code": "429",
            "api_key_prefix": "ak_prod_7x9m2",
            "error_rate": "30%",
            "duration": "2 hours",
            "reported_issue": "rate_limiting",
        },
        customer_metadata={
            "account_tier": "business",
            "org_size": "50-200",
            "timezone": "Asia/Tokyo",
            "account_age_months": "12",
        },
        attachment_refs=["api_error_log_sample.json"],
        sla_deadline_minutes=240,
        ground_truth_category=TicketCategory.INCIDENT,
        gold_queue="platform-support",
        gold_priority=PriorityLevel.HIGH,
        gold_resolution_path=["classify", "assign_queue", "update_priority", "request_info", "reply"],
        required_diagnostic_fields=["api_endpoint", "request_volume", "error_pattern"],
        required_response_keywords=["rate limit", "429", "api"],
        discouraged_keywords=["ignore", "wait a few days"],
        must_escalate=False,
        allow_close=False,
        max_steps=8,
    ),

    # ── HARD ──────────────────────────────────────────────────────────────
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
            "deployment_id": "deploy-2025-04-10-1847",
            "affected_region": "us-east-1",
        },
        customer_metadata={
            "account_tier": "enterprise",
            "org_size": "1000+",
            "timezone": "America/New_York",
            "account_age_months": "48",
            "sla_tier": "premium",
        },
        attachment_refs=["deploy_log_2025-04-10.txt", "error_trace_503.json"],
        sla_deadline_minutes=30,
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

    "hard_data_breach_investigation": TaskSpec(
        task_id="hard_data_breach_investigation",
        difficulty="hard",
        title="Security Incident - Potential Data Breach",
        description=(
            "Customer discovers unauthorized data access across multiple user accounts. "
            "Agent must classify as critical security incident, set urgent priority, route to security-incident queue, "
            "request forensic details, advise immediate containment, and escalate to CISO/incident response."
        ),
        subject="Possible data breach - unauthorized access to customer PII",
        customer_message=(
            "We discovered that an unauthorized party accessed our customer database through what appears to be "
            "a compromised API token. At least 3 user accounts show data exports we didn't authorize. "
            "The exports include PII (names, emails, phone numbers). "
            "We noticed this 45 minutes ago in our audit logs. We need immediate help containing this."
        ),
        extracted_entities={
            "incident_type": "data_breach",
            "compromised_vector": "api_token",
            "affected_accounts": "3+",
            "data_types_exposed": "PII",
            "time_since_discovery": "45 minutes",
            "reported_issue": "data_breach",
        },
        customer_metadata={
            "account_tier": "enterprise",
            "org_size": "1000+",
            "timezone": "America/New_York",
            "account_age_months": "60",
            "sla_tier": "premium",
            "compliance_frameworks": "SOC2,HIPAA",
        },
        attachment_refs=["audit_log_export_suspicious.json", "compromised_token_details.txt"],
        sla_deadline_minutes=15,
        ground_truth_category=TicketCategory.SECURITY,
        gold_queue="security-incident",
        gold_priority=PriorityLevel.URGENT,
        gold_resolution_path=[
            "classify",
            "assign_queue",
            "update_priority",
            "request_info",
            "reply",
            "escalate",
        ],
        required_diagnostic_fields=[
            "compromised_token_id",
            "affected_user_list",
            "export_timestamps",
            "audit_log_range",
        ],
        required_response_keywords=["breach", "contain", "revoke", "escalate"],
        discouraged_keywords=["close", "ignore", "wait", "share password"],
        must_escalate=True,
        allow_close=False,
        max_steps=10,
    ),

    "hard_multi_service_cascade_failure": TaskSpec(
        task_id="hard_multi_service_cascade_failure",
        difficulty="hard",
        title="Production Incident - Multi-Service Cascade Failure",
        description=(
            "A cascading failure across multiple microservices causes widespread outage. "
            "Agent must classify as critical incident, set urgent priority, request detailed diagnostics "
            "across multiple services, route to platform-incident, and escalate for incident bridge."
        ),
        subject="Multiple services down - auth, payments, and notifications all failing",
        customer_message=(
            "Starting about 20 minutes ago, we're seeing failures across multiple services simultaneously. "
            "Auth service returns 500, payment processing times out after 30s, and notification delivery "
            "has completely stopped. We suspect a shared dependency (Redis cluster) may be the root cause. "
            "Our entire platform is effectively down. Over 10,000 active users are affected. "
            "We need your infrastructure team on this immediately."
        ),
        extracted_entities={
            "affected_services": "auth,payments,notifications",
            "suspected_root_cause": "redis_cluster",
            "error_types": "500,timeout,delivery_failure",
            "users_affected": "10000+",
            "duration": "20 minutes",
            "environment": "production",
            "reported_issue": "cascade_failure",
        },
        customer_metadata={
            "account_tier": "enterprise",
            "org_size": "1000+",
            "timezone": "Europe/London",
            "account_age_months": "36",
            "sla_tier": "premium",
        },
        attachment_refs=[
            "redis_cluster_health.json",
            "service_dependency_graph.png",
            "error_rate_dashboard.json",
        ],
        sla_deadline_minutes=15,
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
            "dependency_health",
        ],
        required_response_keywords=["incident", "cascade", "escalate", "infrastructure"],
        discouraged_keywords=["close", "wait a few days", "not urgent"],
        must_escalate=True,
        allow_close=False,
        max_steps=10,
    ),
}


TASK_ORDER: list[str] = [
    "easy_billing_duplicate_charge",
    "easy_password_reset_request",
    "easy_feature_request",
    "medium_security_suspicious_login",
    "medium_data_export_compliance",
    "medium_api_rate_limiting",
    "hard_production_outage",
    "hard_data_breach_investigation",
    "hard_multi_service_cascade_failure",
]


def get_task(task_id: str) -> TaskSpec:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"unknown task_id: {task_id}") from exc


def list_tasks() -> list[TaskSpec]:
    return [TASKS[task_id] for task_id in TASK_ORDER]
