from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app

from .environment import SupportTriageServerEnvironment
from .models import SupportTriageAction, SupportTriageObservation


app = create_app(
    SupportTriageServerEnvironment,
    SupportTriageAction,
    SupportTriageObservation,
    env_name="enterprise-support-ticket-triage",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    import uvicorn

    if port is None:
        port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
