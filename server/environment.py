from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from env.models import Action as NativeAction
from env.support_env import EnterpriseSupportTicketTriageEnv

from .models import SupportTriageAction, SupportTriageObservation


class SupportTriageServerEnvironment(Environment):
    """OpenEnv HTTP adapter around the local Python RL environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        super().__init__()
        self._env = EnterpriseSupportTicketTriageEnv()
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> SupportTriageObservation:
        task_id = kwargs.get("task_id")
        observation = self._env.reset(task_id=task_id)
        episode = episode_id or observation.ticket_id or str(uuid4())
        self._state = State(episode_id=episode, step_count=observation.step_count)
        return self._map_observation(observation, reward=0.0, done=False, metadata={"task_id": observation.task_id})

    def step(self, action: SupportTriageAction, timeout_s: float | None = None, **kwargs) -> SupportTriageObservation:
        native_action = NativeAction.model_validate(
            {
                "action_type": action.action_type,
                "message": action.message,
                "queue": action.queue,
                "priority": action.priority,
                "tags": action.tags,
                "resolution_code": action.resolution_code,
            }
        )
        observation, reward, done, info = self._env.step(native_action)
        self._state.step_count = observation.step_count
        return self._map_observation(
            observation,
            reward=reward.value,
            done=done,
            metadata={
                "reward_components": reward.components,
                "reward_reason": reward.reason,
                "info": info,
            },
        )

    @property
    def state(self) -> State:
        return self._state

    def _map_observation(
        self,
        observation,
        reward: float,
        done: bool,
        metadata: dict,
    ) -> SupportTriageObservation:
        return SupportTriageObservation(
            task_id=observation.task_id,
            difficulty=observation.difficulty,
            ticket_id=observation.ticket_id,
            subject=observation.subject,
            customer_message=observation.customer_message,
            extracted_entities=observation.extracted_entities,
            current_status=observation.current_status.value,
            current_queue=observation.current_queue,
            current_priority=observation.current_priority.value,
            conversation_history=[turn.model_dump() for turn in observation.conversation_history],
            required_missing_fields=observation.required_missing_fields,
            last_action_error=observation.last_action_error,
            step_count=observation.step_count,
            max_steps_remaining=observation.max_steps_remaining,
            reward=reward,
            done=done,
            metadata=metadata,
        )
