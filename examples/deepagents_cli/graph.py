"""Re-export the deepagents CLI LangGraph graph for Aegra or ``langgraph`` deployments.

Requires ``deepagents-cli`` (or an editable install of ``libs/cli``) in the
same environment as the API server.

The graph reads runtime configuration from ``ServerConfig`` environment
variables (``DEEPAGENTS_CLI_SERVER_*``) written by the CLI when it starts a
local server; for remote-only clients, set those variables (or rely on server
defaults) in the deployment environment.
"""

from __future__ import annotations

try:
    from deepagents_cli.server_graph import graph
except ModuleNotFoundError as exc:
    name = getattr(exc, "name", "") or ""
    if name == "deepagents_cli" or name.startswith("deepagents_cli."):
        msg = (
            "deepagents-cli with deepagents_cli.server_graph is not installed in this "
            "environment. From the Aegra repository root run:\n"
            "  ./scripts/dev-install-deepagents-cli.sh\n"
            "or install editable:\n"
            "  uv pip install -e /path/to/deepagents/libs/cli[deepseek]"
        )
        raise ImportError(msg) from exc
    raise

__all__ = ["graph"]
