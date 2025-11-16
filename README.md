# local-ai-agents

This repository contains an opinionated starting point for running a LangGraph agent
entirely on your Mac. The provided agent scans a repository, infers which package
managers are in use, and executes the appropriate commands (npm, pnpm, Poetry,
pip/uv, Bundler, Cargo, etc.) to bring dependencies up to date. Planning and
summaries are delegated to a local Ollama model so no data ever leaves your
machine.

## Prerequisites

1. **Python** 3.11+ – manage it with [`uv`](https://docs.astral.sh/uv/) or `pyenv`.
   See [Installing `uv`](#installing-uv) for quick setup steps.
2. **Ollama** 0.12+ – already installed per your environment. Pull the models you
   want to use (e.g., `ollama pull llama3`).
3. **Package managers** – the agent shells out to the tool detected in the repo, so
   make sure `npm`, `pnpm`, `uv`, `poetry`, `bundle`, `cargo`, etc. are available in
   your `$PATH`.

## Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

You can also use `pip install -e .` if you prefer stock `pip`.

### Installing `uv`

`uv` is the fastest way to manage Python versions and dependencies for this
project. Pick whichever installation method fits your setup:

| Method | Command |
| --- | --- |
| Homebrew | `brew install uv` |
| Official installer | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

After installation, verify it works and grab the latest Python toolchain with:

```bash
uv --version
uv python install 3.11
```

Then create the virtual environment and install dependencies as shown in the
previous section.

## Usage

```
python agents/update_dependencies_agent.py /path/to/repo \
  --model llama3 --ollama-url http://127.0.0.1:11434
```

- `repo` should point to the local Git repository on your Mac whose dependencies
  you want to update.
- `--model` corresponds to any model name served by Ollama.
- `--ollama-url` defaults to the local Ollama endpoint.

### What happens when you run it

1. **Repository inspection** – `agents/update_dependencies_agent.py` walks the repo
   and identifies package manager files (e.g., `package.json`, `poetry.lock`).
2. **Plan generation** – LangGraph invokes `ChatOllama` to outline a bespoke plan
   for the detected managers.
3. **Execution** – the agent executes deterministic commands (e.g., `npm update`,
   `poetry update`, `uv pip install --upgrade -r requirements.txt`). Command output
   is captured for the final report.
4. **Summary** – the final LangGraph node summarizes the run, highlighting
   successes, failures, and follow-up items.

### Extending the workflow

- Add new detectors/commands by editing `PACKAGE_PATTERNS` and `build_commands`
  inside `agents/update_dependencies_agent.py`.
- If you want additional guardrails (approvals, retries, Git status checks), add
  more nodes to the LangGraph state machine.
- Integrate shell tools (e.g., `just`, `npx npm-check-updates`) by exposing them
  as commands in the execution step.

## Troubleshooting

- If a command fails, the summary includes STDOUT/STDERR; run it manually inside
  the repo to investigate.
- Ensure the virtual environment that runs the agent has access to the same CLI
  tools as your login shell (for example, add Homebrew paths to `$PATH`).
- When updating large repos, prefer quantized Ollama models to keep response
  latency low on the M2 MacBook Air.
