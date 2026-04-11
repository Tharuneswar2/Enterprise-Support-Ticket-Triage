"""Smoke tests for Enterprise Support Ticket Triage environment."""

from env.support_env import EnterpriseSupportTicketTriageEnv
from env.models import Action, ActionType, PriorityLevel
from env.tasks import TASK_ORDER


def test_all_tasks_reset():
    """Verify all 9 tasks can be reset and produce valid observations."""
    env = EnterpriseSupportTicketTriageEnv()
    for task_id in TASK_ORDER:
        obs = env.reset(task_id=task_id)
        assert obs.task_id == task_id, f"Task ID mismatch: {obs.task_id} != {task_id}"
        assert obs.step_count == 0
        assert obs.max_steps_remaining > 0
        assert obs.current_status.value == "open"
        assert obs.subject, f"Empty subject for {task_id}"
        assert obs.customer_message, f"Empty customer message for {task_id}"
        print(f"✓ {task_id}: ticket={obs.ticket_id} status={obs.current_status.value}")
    env.close()


def test_step_produces_reward():
    """Verify step() returns valid reward structure."""
    env = EnterpriseSupportTicketTriageEnv()
    env.reset(task_id="easy_billing_duplicate_charge")

    action = Action(
        action_type=ActionType.CLASSIFY,
        message="Billing issue - duplicate charge",
        tags=["billing"],
    )
    obs, reward, done, info = env.step(action)
    assert obs.step_count == 1
    assert isinstance(reward.value, float)
    assert -1.0 <= reward.value <= 1.0
    assert not done
    print(f"✓ Step reward: {reward.value:.3f} ({reward.reason})")
    env.close()


def test_variant_system():
    """Verify stochastic variants produce different observations."""
    env = EnterpriseSupportTicketTriageEnv()
    obs1 = env.reset(task_id="easy_billing_duplicate_charge", seed=42)
    obs2 = env.reset(task_id="easy_billing_duplicate_charge", seed=43)
    assert obs1.customer_message != obs2.customer_message, "Variants should differ"
    print(f"✓ Variant 1: {obs1.customer_message[:60]}...")
    print(f"✓ Variant 2: {obs2.customer_message[:60]}...")
    env.close()


def test_customer_simulation():
    """Verify customer responds after request_info with diagnostic fields."""
    env = EnterpriseSupportTicketTriageEnv()
    env.reset(task_id="medium_security_suspicious_login")

    action = Action(
        action_type=ActionType.REQUEST_INFO,
        message="Please share the last login time and MFA status.",
        tags=["last_known_login_time", "mfa_status"],
    )
    obs, _, _, _ = env.step(action)

    # Should have: initial customer message + agent request + customer reply = 3 turns
    assert len(obs.conversation_history) >= 3, (
        f"Expected customer simulation reply, got {len(obs.conversation_history)} turns"
    )
    last_turn = obs.conversation_history[-1]
    assert last_turn.speaker == "customer", f"Last speaker should be customer, got {last_turn.speaker}"
    print(f"✓ Customer replied: {last_turn.message[:80]}...")
    env.close()


def test_close_method():
    """Verify close() doesn't error."""
    env = EnterpriseSupportTicketTriageEnv()
    env.reset(task_id="easy_billing_duplicate_charge")
    env.close()
    print("✓ close() succeeded")


def test_all_task_ids():
    """Verify get_task_ids() returns all 9 tasks."""
    env = EnterpriseSupportTicketTriageEnv()
    task_ids = env.get_task_ids()
    assert len(task_ids) == 9, f"Expected 9 tasks, got {len(task_ids)}"
    print(f"✓ {len(task_ids)} tasks available: {task_ids}")
    env.close()


if __name__ == "__main__":
    test_all_tasks_reset()
    test_step_produces_reward()
    test_variant_system()
    test_customer_simulation()
    test_close_method()
    test_all_task_ids()
    print("\n✅ All tests passed!")
