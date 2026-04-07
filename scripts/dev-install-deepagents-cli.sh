#!/usr/bin/env bash
# Install deepagents-cli into the Aegra workspace venv (editable) for local dev.
# Does not modify pyproject.toml or uv.lock — safe for CI clones without deepagents.
#
# Layout: sibling checkouts under a common parent, e.g.
#   agents/aegra/   (this repo)
#   agents/deepagents/
#
# Usage:
#   ./scripts/dev-install-deepagents-cli.sh
#   DEEPAGENTS_CLI_ROOT=/path/to/deepagents/libs/cli ./scripts/dev-install-deepagents-cli.sh
#
# Optional extras (comma-separated, same names as deepagents-cli pyproject optional-dependencies):
#   DEEPAGENTS_CLI_EXTRAS=deepseek          # default; pulls langchain-deepseek for provider deepseek
#   DEEPAGENTS_CLI_EXTRAS=deepseek,openrouter
#   DEEPAGENTS_CLI_EXTRAS=                  # core CLI only, no optional providers
#
# Production: if aegra loads a graph that imports deepagents_cli.server_graph and your model
# uses DeepSeek (or another optional provider), install the matching extra in the deployment
# image/venv, e.g. pip install 'deepagents-cli[deepseek]' (or add langchain-deepseek to your
# requirements). The stock Aegra Docker image does not include deepagents-cli; add it only
# when you actually serve that graph.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${DEEPAGENTS_CLI_ROOT:-}" ]]; then
  CLI_PATH="$(cd "$DEEPAGENTS_CLI_ROOT" && pwd)"
else
  SIBLING="$ROOT/../deepagents/libs/cli"
  if [[ -d "$SIBLING" ]]; then
    CLI_PATH="$(cd "$SIBLING" && pwd)"
  else
    echo "Could not find deepagents-cli package." >&2
    echo "  Place deepagents next to aegra (../deepagents/libs/cli), or set DEEPAGENTS_CLI_ROOT." >&2
    exit 1
  fi
fi

if [[ ! -f "$CLI_PATH/pyproject.toml" ]]; then
  echo "Not a Python project: $CLI_PATH" >&2
  exit 1
fi

echo "Syncing Aegra workspace..."
uv sync --all-packages

PY="$(uv run python -c "import sys; print(sys.executable)")"
EXTRAS="${DEEPAGENTS_CLI_EXTRAS-deepseek}"
if [[ -n "${EXTRAS// /}" ]]; then
  INSTALL_TARGET="${CLI_PATH}[${EXTRAS}]"
  echo "Installing editable deepagents-cli with extras [${EXTRAS}] from: $CLI_PATH"
else
  INSTALL_TARGET="${CLI_PATH}"
  echo "Installing editable deepagents-cli (no optional extras) from: $CLI_PATH"
fi
uv pip install --python "$PY" -e "$INSTALL_TARGET"

echo "Done. Run: uv run aegra dev"
echo "Then point deepagents CLI at this server, e.g.:"
echo "  DEEPAGENTS_CLI_REMOTE_GRAPH_NAME=deepagents deepagents --remote-url http://127.0.0.1:2026"
