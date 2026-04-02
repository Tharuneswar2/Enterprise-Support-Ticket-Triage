from env.support_env import EnterpriseSupportTicketTriageEnv
from env.tasks import TASK_ORDER


def main() -> None:
    env = EnterpriseSupportTicketTriageEnv()
    for task_id in TASK_ORDER:
        obs = env.reset(task_id=task_id)
        print(f"{task_id}: ticket={obs.ticket_id} status={obs.current_status.value}")


if __name__ == "__main__":
    main()
