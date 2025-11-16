"""LangGraph agent that updates dependencies inside a local repository.

The module exposes a CLI entry-point so it can be executed directly:

```
python agent/update_dependencies_agent.py /path/to/repo \
    --model llama3 --ollama-url http://127.0.0.1:11434
```

It assumes you already have Ollama running locally with the requested model.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph


PACKAGE_PATTERNS: tuple[tuple[str, Sequence[str]], ...] = (
    ("npm", ("package.json",)),
    ("pnpm", ("pnpm-lock.yaml", "pnpm-workspace.yaml")),
    ("yarn", ("yarn.lock",)),
    ("poetry", ("poetry.lock", "pyproject.toml")),
    ("pip", ("requirements.txt", "requirements.in")),
    ("uv", ("uv.lock",)),
    ("bundler", ("Gemfile", "Gemfile.lock")),
    ("cargo", ("Cargo.toml", "Cargo.lock")),
)


@dataclass
class DependencyUpdateCommand:
    """Represents a concrete command that should be executed."""

    description: str
    command: Sequence[str]


class AgentState(TypedDict):
    repo_path: str
    detected_managers: List[str]
    commands: List[DependencyUpdateCommand]
    plan: str
    execution_log: List[str]
    summary: str


def iter_files(repo_path: Path) -> Iterable[Path]:
    for root, _, files in os.walk(repo_path):
        for file_name in files:
            yield Path(root) / file_name


def detect_package_managers(repo_path: Path) -> list[str]:
    matches: set[str] = set()
    for file_path in iter_files(repo_path):
        for manager, patterns in PACKAGE_PATTERNS:
            if file_path.name in patterns:
                matches.add(manager)
    return sorted(matches)


def build_commands(managers: Sequence[str]) -> list[DependencyUpdateCommand]:
    commands: list[DependencyUpdateCommand] = []
    for manager in managers:
        if manager == "npm":
            commands.append(
                DependencyUpdateCommand(
                    "Update npm packages", ["npm", "update"]
                )
            )
        elif manager == "pnpm":
            commands.append(
                DependencyUpdateCommand(
                    "Update pnpm packages", ["pnpm", "update", "--latest"]
                )
            )
        elif manager == "yarn":
            commands.append(
                DependencyUpdateCommand(
                    "Update yarn packages", ["yarn", "upgrade", "--latest"]
                )
            )
        elif manager == "poetry":
            commands.append(
                DependencyUpdateCommand(
                    "Update Poetry dependencies", ["poetry", "update"]
                )
            )
        elif manager == "pip":
            commands.append(
                DependencyUpdateCommand(
                    "Upgrade requirements via pip", [
                        "uv",
                        "pip",
                        "install",
                        "--upgrade",
                        "-r",
                        "requirements.txt",
                    ]
                )
            )
        elif manager == "uv":
            commands.append(
                DependencyUpdateCommand(
                    "Update uv.lock dependencies", ["uv", "lock", "--upgrade"]
                )
            )
        elif manager == "bundler":
            commands.append(
                DependencyUpdateCommand(
                    "Update Ruby gems", ["bundle", "update"]
                )
            )
        elif manager == "cargo":
            commands.append(
                DependencyUpdateCommand(
                    "Update Cargo crates", ["cargo", "update"]
                )
            )
    return commands


def gather_context(state: AgentState) -> AgentState:
    repo_path = Path(state["repo_path"])
    managers = detect_package_managers(repo_path)
    commands = build_commands(managers)
    return {
        **state,
        "detected_managers": managers,
        "commands": commands,
        "execution_log": [],
    }


def plan_updates(state: AgentState, llm: ChatOllama) -> AgentState:
    messages = [
        SystemMessage(
            content=(
                "You are a release engineer specializing in dependency hygiene. "
                "You will receive detected package managers and you must craft a"
                " concise, bullet-point plan describing how to update them."
            )
        ),
        HumanMessage(
            content=json.dumps(
                {
                    "repo_path": state["repo_path"],
                    "detected_managers": state["detected_managers"],
                    "commands": [cmd.description for cmd in state["commands"]],
                },
                indent=2,
            )
        ),
    ]
    plan = llm.invoke(messages).content
    return {**state, "plan": plan}


def execute_commands(state: AgentState) -> AgentState:
    repo_path = Path(state["repo_path"])
    logs = list(state.get("execution_log", []))
    for command in state["commands"]:
        try:
            completed = subprocess.run(
                list(command.command),
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            logs.append(
                f"✅ {command.description}\n$ {' '.join(command.command)}\n{completed.stdout.strip()}"
            )
        except subprocess.CalledProcessError as error:
            logs.append(
                f"❌ {command.description}\n$ {' '.join(command.command)}\n"
                f"STDOUT: {error.stdout}\nSTDERR: {error.stderr}"
            )
    return {**state, "execution_log": logs}


def summarize(state: AgentState, llm: ChatOllama) -> AgentState:
    transcript = "\n\n".join(state["execution_log"])
    messages = [
        SystemMessage(
            content=(
                "Summarize the dependency update session. Highlight successes, "
                "failures, and recommended follow-up actions."
            )
        ),
        HumanMessage(
            content=f"Plan:\n{state['plan']}\n\nExecution log:\n{transcript}",
        ),
    ]
    summary = llm.invoke(messages).content
    return {**state, "summary": summary}


def build_graph(llm: ChatOllama) -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("gather", gather_context)
    graph.add_node("plan", lambda state: plan_updates(state, llm))
    graph.add_node("execute", execute_commands)
    graph.add_node("summarize", lambda state: summarize(state, llm))

    graph.set_entry_point("gather")
    graph.add_edge("gather", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "summarize")
    graph.add_edge("summarize", END)
    return graph


def run_agent(repo_path: Path, model: str, base_url: str) -> str:
    llm = ChatOllama(model=model, base_url=base_url)
    graph = build_graph(llm)
    final_state = graph.compile().invoke({"repo_path": str(repo_path)})
    return final_state["summary"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo",
        type=Path,
        help="Path to the repository that should be updated",
    )
    parser.add_argument(
        "--model",
        default="llama3",
        help="Name of the Ollama model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Base URL of the Ollama server",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_agent(args.repo, args.model, args.ollama_url)
    print(summary)


if __name__ == "__main__":
    main()
