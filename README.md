---
title: Enterprise Support Ticket Triage
emoji: 🎫
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: OpenEnv RL environment for enterprise helpdesk ticket triage
tags:
  - openenv
  - reinforcement-learning
  - enterprise-support
  - ticket-triage
---

# Enterprise Support Ticket Triage — OpenEnv RL Environment

A production-grade reinforcement learning environment for training and evaluating AI agents on enterprise helpdesk workflows. Each episode presents one inbound support ticket that the agent must triage through a multi-step decision process.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OpenEnv HTTP Server                         │
│                     (FastAPI on port 7860)                          │
│                                                                     │
│  POST /reset ──► EnterpriseSupportTicketTriageEnv.reset()           │
│  POST /step  ──► EnterpriseSupportTicketTriageEnv.step()            │
│  GET  /state ──► EnterpriseSupportTicketTriageEnv.state()           │
│  GET  /schema ─► Action/Observation Pydantic schemas                │
│  WS   /ws    ──► Persistent WebSocket session                       │
└──────────────┬──────────────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │  Environment Core   │
    │  ┌────────────────┐ │
    │  │   9 Tasks      │ │  easy_billing_duplicate_charge
    │  │   (3 easy,     │ │  easy_password_reset_request
    │  │    3 medium,   │ │  easy_feature_request
    │  │    3 hard)     │ │  medium_security_suspicious_login
    │  └────────────────┘ │  medium_data_export_compliance
    │  ┌────────────────┐ │  medium_api_rate_limiting
    │  │ Shaped Rewards │ │  hard_production_outage
    │  │ + Anti-Gaming  │ │  hard_data_breach_investigation
    │  └────────────────┘ │  hard_multi_service_cascade_failure
    │  ┌────────────────┐ │
    │  │  Deterministic │ │
    │  │  Graders (per  │ │
    │  │  task)         │ │
    │  └────────────────┘ │
    │  ┌────────────────┐ │
    │  │  Customer      │ │
    │  │  Simulation    │ │
    │  └────────────────┘ │
    └─────────────────────┘
```

## What Makes This Environment Interesting

| Feature | Description |
|---------|-------------|
| **Multi-step triage workflow** | Agents must classify → route → investigate → resolve in order |
| **9 diverse tasks** | Billing, security, compliance (GDPR), API issues, data breaches, cascading failures |
| **Customer simulation** | After `request_info`, the customer "responds" with diagnostic details |
| **Anti-gaming grading** | Keyword stuffing is penalized; substance and workflow order are rewarded |
| **Stochastic variants** | Each task can be randomized with different customer tones, entity values, and IP addresses |
| **SLA pressure** | Hard tasks include SLA deadlines that create urgency |
| **Safety checks** | Agents are penalized for unsafe behavior (e.g., suggesting users share passwords) |

## Action Space

```json
{
  "action_type": "classify | reply | escalate | close | request_info | update_priority | assign_queue",
  "message": "Free text response or request",
  "queue": "Target queue name (for assign_queue)",
  "priority": "low | normal | high | urgent (for update_priority)",
  "tags": ["structured", "hints"],
  "resolution_code": "Optional close reason"
}
```

## Observation Space

```json
{
  "task_id": "hard_production_outage",
  "difficulty": "hard",
  "ticket_id": "TKT-0007-HARD",
  "subject": "Production API returning 503 after deployment, checkout unavailable",
  "customer_message": "Our production checkout API started returning 503...",
  "extracted_entities": {
    "impact": "checkout_down",
    "environment": "production",
    "error_code": "503",
    "deployment_id": "deploy-2025-04-10-1847"
  },
  "customer_metadata": {
    "account_tier": "enterprise",
    "org_size": "1000+",
    "sla_tier": "premium"
  },
  "attachment_refs": ["deploy_log_2025-04-10.txt", "error_trace_503.json"],
  "sla_deadline_minutes": 30,
  "current_status": "open",
  "current_queue": "triage-intake",
  "current_priority": "normal",
  "conversation_history": [
    {"speaker": "customer", "message": "Our production checkout API...", "timestamp": "2025-04-10T14:47:00Z"}
  ],
  "required_missing_fields": ["incident_start_time", "affected_service", "request_id", "error_logs"],
  "step_count": 0,
  "max_steps_remaining": 8
}
```

## Tasks (9 total, 3 difficulty levels)

| ID | Difficulty | Category | Key Challenge |
|----|-----------|----------|---------------|
| `easy_billing_duplicate_charge` | Easy | Billing | Classify, acknowledge, close properly |
| `easy_password_reset_request` | Easy | Account | Verify identity before resetting |
| `easy_feature_request` | Easy | General | Acknowledge without over-promising |
| `medium_security_suspicious_login` | Medium | Security | Request verification, advise reset, don't close |
| `medium_data_export_compliance` | Medium | Compliance | GDPR handling, verify authority, set timeline |
| `medium_api_rate_limiting` | Medium | Technical | Diagnose 429 errors, gather API usage data |
| `hard_production_outage` | Hard | Incident | Urgent priority, diagnostics, must escalate |
| `hard_data_breach_investigation` | Hard | Security | Contain breach, revoke tokens, must escalate |
| `hard_multi_service_cascade_failure` | Hard | Incident | Multi-service triage, dependency analysis, escalate |

## Reward Design

Shaped rewards are awarded per-step for meaningful progress:

- **+0.22** Correct classification
- **+0.18** Correct queue assignment
- **+0.16** Correct priority (harder tasks get more)
- **+0.10** Per required diagnostic field requested
- **+0.40** Correct closure (easy tasks only)
- **+0.20–0.40** Correct escalation (hard tasks)
- **+0.10** Workflow order bonus (following expected sequence)

Penalties:

- **-0.25** Invalid action schema
- **-0.35** Premature/unsafe closure
- **-0.18** Unnecessary escalation
- **-0.05n²** Repeated identical actions (quadratic)
- **-0.10** Discouraged content
- **-0.35** Unsafe behavior (suggesting password sharing, etc.)
- **-0.08** Keyword stuffing (anti-gaming)

## Setup

```bash
git clone <repo-url>
cd Enterprise-Support-Ticket-Triage
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Start the API Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### API Examples

**Health check:**
```bash
curl http://localhost:7860/
```

**Reset environment (start new episode):**
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_billing_duplicate_charge"}'
```

**Take a step:**
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "classify",
      "message": "This is a billing issue for a duplicate charge.",
      "tags": ["billing", "duplicate_charge"]
    }
  }'
```

**Get schemas:**
```bash
curl http://localhost:7860/schema
```

### Run Baseline Inference (Heuristic)

```bash
python inference.py --disable-api
```

### Run with LLM (Optional)

```bash
export HF_TOKEN="<your_hf_token>"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
python inference.py
```

### Run Specific Tasks

```bash
python inference.py --disable-api --tasks hard_production_outage hard_data_breach_investigation
```

## Docker

```bash
# Build
docker build -t enterprise-support-ticket-triage .

# Run server
docker run --rm -p 7860:7860 enterprise-support-ticket-triage

# Health check
curl http://localhost:7860/

# Run inference inside container
docker run --rm enterprise-support-ticket-triage python inference.py --disable-api
```

## Baseline Scores

The deterministic heuristic baseline is intentionally imperfect to demonstrate grader discriminating power:

| Task | Baseline Score | Why Not Perfect |
|------|---------------|-----------------|
| Easy tasks | ~0.75-0.85 | Skips some verification steps |
| Medium tasks | ~0.65-0.80 | Misses diagnostic fields, wrong priority |
| Hard tasks | ~0.60-0.75 | Incomplete diagnostics coverage |
| **Average** | **~0.70** | **Room for RL improvement** |

A trained RL agent should significantly outperform this baseline.

## Resources / Runtime

- 2 vCPU, 8 GB RAM
- No GPU required
- Inference completes in under 2 minutes
- Docker-compatible with HF Spaces
