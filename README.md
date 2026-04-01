# Enterprise Support Ticket Triage (OpenEnv Round 1)

## What This Environment Is

`Enterprise Support Ticket Triage` is a realistic reinforcement learning environment for enterprise helpdesk workflows. Each episode is one inbound support ticket. The agent must triage the ticket through a multi-step process:

1. classify issue type
2. assign support queue
3. set priority
4. request missing diagnostics
5. reply safely and clearly
6. decide whether to escalate or close

The environment is deterministic and shaped for trajectory learning, not just terminal success.

## Motivation

Support operations are high-impact and process-sensitive. Bad triage causes delays, unresolved incidents, and security risk. This environment is useful for training and evaluating agents that must make structured, auditable decisions in enterprise support operations.

## Product Decisions

Where the spec left room for design choices, the following practical defaults were selected:

1. Episodes are deterministic per task (no stochastic ticket variants) to maximize grading reproducibility.
2. `step()` returns a typed `Reward` object (not raw float) to keep reward components transparent.
3. Medium and hard tasks intentionally discourage closure; hard task requires escalation.
4. Invalid actions are non-fatal and penalized with `last_action_error` surfaced in observations.
5. Repeated identical actions receive loop penalties to discourage exploitative behavior.

## OpenEnv Interface Coverage

Implemented in [`env/support_env.py`](/home/tharuneswar/Coding/Hackthon/Enterprise-Support-Ticket-Triage/env/support_env.py):

- Typed Pydantic models:
  - [`Action`](/home/tharuneswar/Coding/Hackthon/Enterprise-Support-Ticket-Triage/env/models.py)
  - [`Observation`](/home/tharuneswar/Coding/Hackthon/Enterprise-Support-Ticket-Triage/env/models.py)
  - [`Reward`](/home/tharuneswar/Coding/Hackthon/Enterprise-Support-Ticket-Triage/env/models.py)
  - [`EpisodeState`](/home/tharuneswar/Coding/Hackthon/Enterprise-Support-Ticket-Triage/env/models.py)
- `reset(task_id: Optional[str]) -> Observation`
- `step(action: Action | dict) -> (Observation, Reward, done, info)`
- `state() -> EpisodeState`
- Environment metadata in [`openenv.yaml`](/home/tharuneswar/Coding/Hackthon/Enterprise-Support-Ticket-Triage/openenv.yaml)

## Action Space

Pydantic schema: [`Action`](/home/tharuneswar/Coding/Hackthon/Enterprise-Support-Ticket-Triage/env/models.py)

Fields:

- `action_type`: `classify | reply | escalate | close | request_info | update_priority | assign_queue`
- `message`: free text response/request
- `queue`: queue target string (for routing)
- `priority`: `low | normal | high | urgent`
- `tags`: structured hints/features for policy behavior
- `resolution_code`: optional structured close reason

## Observation Space

Pydantic schema: [`Observation`](/home/tharuneswar/Coding/Hackthon/Enterprise-Support-Ticket-Triage/env/models.py)

Visible fields include:

- `ticket_id`
- `subject`
- `customer_message`
- `extracted_entities`
- `current_status`
- `current_queue`
- `current_priority`
- `conversation_history`
- `required_missing_fields`
- `last_action_error`
- `step_count`
- `max_steps_remaining`

## Hidden/Internal State

Pydantic schema: [`EpisodeState`](/home/tharuneswar/Coding/Hackthon/Enterprise-Support-Ticket-Triage/env/models.py)

Includes deterministic grading signals and full trajectory context:

- ground-truth ticket category
- gold queue and gold priority
- gold resolution path
- required diagnostic fields
- requested-field progress
- action/reward history
- flags (premature close, escalation, unsafe behavior)
- episode metadata (repeat-loop tracking)

## Tasks (Increasing Difficulty)

Task definitions: [`env/tasks.py`](/home/tharuneswar/Coding/Hackthon/Enterprise-Support-Ticket-Triage/env/tasks.py)

1. `easy_billing_duplicate_charge`
2. `medium_security_suspicious_login`
3. `hard_production_outage`

### Easy: Billing inquiry / duplicate charge

Expected policy behavior:

- classify as billing
- assign billing queue
- acknowledge duplicate charge + refund path
- provide next steps
- close only after acknowledgment

### Medium: Account security / suspicious login

Expected policy behavior:

- classify as security
- assign security queue
- set high priority
- request verification details (`last_known_login_time`, `login_location`, `mfa_status`)
- advise password reset / credential reset
- keep ticket open (avoid premature close)

### Hard: Production incident / outage

Expected policy behavior:

- classify as production incident
- assign platform incident queue
- set urgent priority
- request diagnostics (`incident_start_time`, `affected_service`, `request_id`, `error_logs`)
- escalate to incident response
- do not resolve/close prematurely

## Reward Design (Shaped)

Implemented in [`env/support_env.py`](/home/tharuneswar/Coding/Hackthon/Enterprise-Support-Ticket-Triage/env/support_env.py)

Examples of positive shaping:

- correct classification
- correct queue assignment
- correct priority update
- requesting required diagnostics
- safe response content (e.g., password reset guidance)
- correct escalation or correct closure

Penalties include:

- invalid actions / schema mismatch
- wrong queue or priority
- irrelevant repeated loops
- premature or unsafe closure
- unsafe security behavior
- max-step timeout

Per-step reward is clipped to `[-1.0, 1.0]`.

## Deterministic Graders

Implemented in [`env/graders.py`](/home/tharuneswar/Coding/Hackthon/Enterprise-Support-Ticket-Triage/env/graders.py)

Each task has deterministic score output in `[0.0, 1.0]` based on:

- classification correctness
- queue correctness
- priority correctness
- required diagnostics coverage
- required response intent content
- escalation correctness (hard task)
- closure safety / premature closure checks

## Setup

```bash
cd Enterprise-Support-Ticket-Triage
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run baseline inference (heuristic fallback)

```bash
python inference.py --disable-api
```

### Run baseline inference with Hugging Face Router API

Environment variables used by `inference.py`:

- `HF_TOKEN` (preferred; `API_KEY` and `HF_API_TOKEN` are also accepted)
- `API_BASE_URL`
- `MODEL_NAME`

```bash
export HF_TOKEN="<your_hf_token>"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
python inference.py
```

## Validation

If OpenEnv CLI is installed:

```bash
openenv validate
```

## Docker

Build:

```bash
docker build -t enterprise-support-ticket-triage .
```

Run:

```bash
docker run --rm enterprise-support-ticket-triage
```

## Baseline Score Reporting

`inference.py` prints:

- per-task step trace
- per-task deterministic task score
- final average across all 3 tasks

Score interpretation:

- `>= 0.85`: strong triage policy
- `0.60 - 0.84`: partially correct, misses key workflow details
- `< 0.60`: unsafe or low-fidelity triage behavior

## Resource / Runtime Targets

Designed to run within:

- 2 vCPU
- 8 GB RAM
- under 20 minutes inference runtime (typical run is far below this)

## Hugging Face Spaces Deployment Notes

- Docker-compatible (`Dockerfile` included)
- metadata tagged with `openenv` in [`openenv.yaml`](/home/tharuneswar/Coding/Hackthon/Enterprise-Support-Ticket-Triage/openenv.yaml)
- low dependency footprint, no GPU requirement
- suitable for Docker Spaces batch evaluation workflows
