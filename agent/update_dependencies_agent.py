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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph


SKIP_DIRECTORIES = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "target",
    ".gradle",
}


PACKAGE_PATTERNS: tuple[tuple[str, Sequence[str]], ...] = (
    ("npm", ("package.json", "package-lock.json", "npm-shrinkwrap.json")),
    ("pnpm", ("pnpm-lock.yaml", "pnpm-workspace.yaml")),
    ("yarn", ("yarn.lock",)),
    ("poetry", ("poetry.lock", "pyproject.toml")),
    ("pip", ("requirements.txt", "requirements.in", "setup.py", "setup.cfg")),
    ("pipenv", ("Pipfile", "Pipfile.lock")),
    ("uv", ("uv.lock",)),
    ("bundler", ("Gemfile", "Gemfile.lock")),
    ("cargo", ("Cargo.toml", "Cargo.lock")),
    ("gradle", ("build.gradle", "build.gradle.kts", "settings.gradle", "gradlew", "gradlew.bat")),
    ("maven", ("pom.xml",)),
    ("go", ("go.mod", "go.sum")),
    ("dotnet", ("global.json",)),
)


SUFFIX_MATCHERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dotnet", (".csproj", ".fsproj", ".vbproj", ".sln")),
)


MANAGER_METADATA: dict[str, dict[str, str]] = {
    "npm": {"label": "npm", "ecosystem": "node"},
    "pnpm": {"label": "pnpm", "ecosystem": "node"},
    "yarn": {"label": "yarn", "ecosystem": "node"},
    "poetry": {"label": "Poetry", "ecosystem": "python"},
    "pip": {"label": "pip / requirements", "ecosystem": "python"},
    "pipenv": {"label": "Pipenv", "ecosystem": "python"},
    "uv": {"label": "uv", "ecosystem": "python"},
    "bundler": {"label": "Bundler", "ecosystem": "ruby"},
    "cargo": {"label": "Cargo", "ecosystem": "rust"},
    "gradle": {"label": "Gradle", "ecosystem": "java/kotlin"},
    "maven": {"label": "Maven", "ecosystem": "java"},
    "go": {"label": "Go modules", "ecosystem": "go"},
    "dotnet": {"label": ".NET", "ecosystem": "dotnet"},
}


def _pip_command(repo_path: Path, _: DetectedManager) -> DependencyUpdateCommand:
    requirements = repo_path / "requirements.txt"
    if requirements.exists():
        args = ["uv", "pip", "install", "--upgrade", "-r", "requirements.txt"]
        description = "Upgrade pip requirements"
    else:
        args = ["uv", "pip", "install", "--upgrade", "."]
        description = "Upgrade pip-installed project dependencies"
    return DependencyUpdateCommand(description, args)


def _uv_command(_: Path, __: DetectedManager) -> DependencyUpdateCommand:
    return DependencyUpdateCommand("Update uv lockfile", ["uv", "lock", "--upgrade"])


def _poetry_command(_: Path, __: DetectedManager) -> DependencyUpdateCommand:
    return DependencyUpdateCommand("Update Poetry dependencies", ["poetry", "update"])


def _pipenv_command(_: Path, __: DetectedManager) -> DependencyUpdateCommand:
    return DependencyUpdateCommand("Update Pipenv dependencies", ["pipenv", "update"])


def _npm_command(_: Path, __: DetectedManager) -> DependencyUpdateCommand:
    return DependencyUpdateCommand("Update npm packages", ["npm", "update"])


def _pnpm_command(_: Path, __: DetectedManager) -> DependencyUpdateCommand:
    return DependencyUpdateCommand("Update pnpm packages", ["pnpm", "update", "--latest"])


def _yarn_command(_: Path, __: DetectedManager) -> DependencyUpdateCommand:
    return DependencyUpdateCommand("Update yarn packages", ["yarn", "upgrade", "--latest"])


def _bundler_command(_: Path, __: DetectedManager) -> DependencyUpdateCommand:
    return DependencyUpdateCommand("Update Ruby gems", ["bundle", "update"])


def _cargo_command(_: Path, __: DetectedManager) -> DependencyUpdateCommand:
    return DependencyUpdateCommand("Update Cargo crates", ["cargo", "update"])


def _gradle_command(repo_path: Path, _: DetectedManager) -> DependencyUpdateCommand:
    wrapper = repo_path / "gradlew"
    runner = "./gradlew" if wrapper.exists() else "gradle"
    return DependencyUpdateCommand(
        "Refresh Gradle dependencies",
        [runner, "--refresh-dependencies"],
    )


def _maven_command(_: Path, __: DetectedManager) -> DependencyUpdateCommand:
    return DependencyUpdateCommand(
        "Use latest Maven dependency releases",
        ["mvn", "versions:use-latest-releases"],
    )


def _go_command(_: Path, __: DetectedManager) -> DependencyUpdateCommand:
    return DependencyUpdateCommand("Update Go modules", ["go", "get", "-u", "./..."])


def _dotnet_command(_: Path, __: DetectedManager) -> DependencyUpdateCommand:
    return DependencyUpdateCommand(
        "List and upgrade outdated .NET packages",
        ["dotnet", "list", "package", "--outdated", "--include-transitive"],
    )


COMMAND_BUILDERS: dict[str, CommandBuilder] = {
    "pip": _pip_command,
    "uv": _uv_command,
    "poetry": _poetry_command,
    "pipenv": _pipenv_command,
    "npm": _npm_command,
    "pnpm": _pnpm_command,
    "yarn": _yarn_command,
    "bundler": _bundler_command,
    "cargo": _cargo_command,
    "gradle": _gradle_command,
    "maven": _maven_command,
    "go": _go_command,
    "dotnet": _dotnet_command,
}


@dataclass
class DependencyUpdateCommand:
    """Represents a concrete command that should be executed."""

    description: str
    command: Sequence[str]


@dataclass
class DetectedManager:
    slug: str
    label: str
    ecosystem: str
    evidence: list[str]


CommandBuilder = Callable[[Path, DetectedManager], Optional[DependencyUpdateCommand]]


class AgentState(TypedDict, total=False):
    repo_path: str
    repo_overview: str
    language_summary: str
    repo_samples: list[dict[str, str]]
    detected_managers: List[DetectedManager]
    commands: List[DependencyUpdateCommand]
    plan: str
    execution_log: List[str]
    summary: str


def iter_files(repo_path: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRECTORIES]
        for file_name in files:
            yield Path(root) / file_name


def describe_repo(repo_path: Path, max_entries: int = 25) -> str:
    entries: list[str] = []
    for entry in sorted(repo_path.iterdir(), key=lambda p: p.name.lower()):
        if entry.name in SKIP_DIRECTORIES or entry.name.startswith("."):
            continue
        entries.append(f"{entry.name}{'/' if entry.is_dir() else ''}")
        if len(entries) >= max_entries:
            break
    if not entries:
        return "Repository appears to be empty"
    return "\n".join(entries)


def summarize_languages(repo_path: Path, max_files: int = 3000) -> str:
    """Produce a quick language distribution summary based on file extensions."""

    extension_map = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".jsx": "JavaScript",
        ".java": "Java",
        ".kt": "Kotlin",
        ".kts": "Kotlin",
        ".go": "Go",
        ".rs": "Rust",
        ".rb": "Ruby",
        ".cs": "C#",
        ".cpp": "C++",
        ".cc": "C++",
        ".c": "C",
        ".m": "Objective-C",
        ".swift": "Swift",
        ".php": "PHP",
        ".gradle": "Gradle scripts",
    }

    counts: dict[str, int] = defaultdict(int)
    inspected = 0
    for file_path in iter_files(repo_path):
        inspected += 1
        if inspected > max_files:
            break
        counts[extension_map.get(file_path.suffix, "Other")] += 1

    if not counts:
        return "No source files detected"

    top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
    return ", ".join(f"{language}: {count}" for language, count in top)


INSIGHT_FILES = (
    "README.md",
    "README.MD",
    "package.json",
    "pnpm-workspace.yaml",
    "pyproject.toml",
    "requirements.txt",
    "poetry.lock",
    "build.gradle",
    "build.gradle.kts",
    "pom.xml",
    "go.mod",
    "Cargo.toml",
    "Gemfile",
)


def read_snippet(path: Path, max_chars: int = 1200) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return "<unable to read file>"
    if len(text) > max_chars:
        return text[:max_chars] + "\n…"
    return text


def collect_repo_samples(repo_path: Path, max_files: int = 5) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for candidate in INSIGHT_FILES:
        path = repo_path / candidate
        if not path.exists():
            continue
        samples.append(
            {
                "path": candidate,
                "content": read_snippet(path),
            }
        )
        if len(samples) >= max_files:
            break
    return samples


def detect_package_managers(repo_path: Path) -> list[DetectedManager]:
    evidence: dict[str, set[str]] = defaultdict(set)
    for file_path in iter_files(repo_path):
        relative = str(file_path.relative_to(repo_path))
        file_name = file_path.name
        for manager, patterns in PACKAGE_PATTERNS:
            if file_name in patterns:
                evidence[manager].add(relative)
        for manager, suffixes in SUFFIX_MATCHERS:
            if file_name.endswith(suffixes):
                evidence[manager].add(relative)

        if file_name == "pyproject.toml":
            try:
                content = file_path.read_text(encoding="utf-8")
            except OSError:
                content = ""
            if "[tool.poetry]" in content:
                evidence["poetry"].add(relative)
            elif "[project]" in content:
                evidence["pip"].add(relative)

    detected: list[DetectedManager] = []
    for slug, paths in sorted(evidence.items()):
        if slug not in MANAGER_METADATA:
            continue
        metadata = MANAGER_METADATA[slug]
        detected.append(
            DetectedManager(
                slug=slug,
                label=metadata["label"],
                ecosystem=metadata["ecosystem"],
                evidence=sorted(paths),
            )
        )
    return detected


def build_commands(
    managers: Sequence[DetectedManager], repo_path: Path
) -> list[DependencyUpdateCommand]:
    commands: list[DependencyUpdateCommand] = []
    for manager in managers:
        builder = COMMAND_BUILDERS.get(manager.slug)
        if not builder:
            continue
        command = builder(repo_path, manager)
        if command:
            commands.append(command)
    return commands


def gather_context(state: AgentState) -> AgentState:
    repo_path = Path(state["repo_path"])
    overview = describe_repo(repo_path)
    language_summary = summarize_languages(repo_path)
    repo_samples = collect_repo_samples(repo_path)
    managers = detect_package_managers(repo_path)
    commands = build_commands(managers, repo_path)
    return {
        **state,
        "repo_overview": overview,
        "language_summary": language_summary,
        "repo_samples": repo_samples,
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
                    "repo_overview": state.get("repo_overview", ""),
                    "language_summary": state.get("language_summary", ""),
                    "detected_managers": [
                        {
                            "name": manager.label,
                            "ecosystem": manager.ecosystem,
                            "evidence": manager.evidence,
                        }
                        for manager in state["detected_managers"]
                    ],
                    "repo_samples": state.get("repo_samples", []),
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
