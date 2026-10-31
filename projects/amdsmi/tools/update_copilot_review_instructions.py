#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copy the amd-smi review agents into Copilot code review instructions.

Copilot code review reads instruction files only from the repository root and
cannot run custom agents. This script copies the body of every
projects/amdsmi/.github/agents/amdsmi-review*.agent.md, unchanged, into
.github/instructions/<agent>.instructions.md, scoped to projects/amdsmi/**.

Run it after changing a review agent. --check exits 1 when a copy is stale.
"""

import argparse
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[3]
AGENTS_DIR = REPO_ROOT / "projects" / "amdsmi" / ".github" / "agents"
OUTPUT_DIR = REPO_ROOT / ".github" / "instructions"
AGENT_GLOB = "amdsmi-review*.agent.md"
OUTPUT_GLOB = "amdsmi-review*.instructions.md"

HEADER = """---
applyTo: "projects/amdsmi/**"
excludeAgent: "cloud-agent"
---

<!-- Generated from {source} by {script}. Edit the agent and rerun the script. -->

> Copilot code review: this is the amd-smi `{name}` review agent. Apply its
> review checks to changes under `projects/amdsmi/`. Its instructions about
> tools, subagents, builds and output format apply only when it runs as an
> agent.

"""


def rel(path):
    return path.relative_to(REPO_ROOT).as_posix()


def agent_body(path):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines or lines[0].rstrip("\n") != "---":
        sys.exit(f"{rel(path)}: expected YAML frontmatter")
    for index in range(1, len(lines)):
        if lines[index].rstrip("\n") == "---":
            return "".join(lines[index + 1 :]).lstrip("\n")
    sys.exit(f"{rel(path)}: unterminated YAML frontmatter")


def expected_files():
    files = {}
    for agent in sorted(AGENTS_DIR.glob(AGENT_GLOB)):
        name = agent.name[: -len(".agent.md")]
        header = HEADER.format(source=rel(agent), script=rel(SCRIPT), name=name)
        files[OUTPUT_DIR / f"{name}.instructions.md"] = header + agent_body(agent)
    if not files:
        sys.exit(f"no {AGENT_GLOB} files in {rel(AGENTS_DIR)}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a generated file is stale")
    args = parser.parse_args()

    expected = expected_files()
    stale = [
        path
        for path, text in expected.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != text
    ]
    orphans = sorted(set(OUTPUT_DIR.glob(OUTPUT_GLOB)) - set(expected))

    if args.check:
        for path in stale + orphans:
            print(f"stale: {rel(path)}")
        if stale or orphans:
            print(f"Run: python3 {rel(SCRIPT)}")
            return 1
        print(f"{len(expected)} Copilot review instruction files are up to date")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in stale:
        with open(path, "w", encoding="utf-8", newline="\n") as out:
            out.write(expected[path])
        print(f"wrote {rel(path)}")
    for path in orphans:
        path.unlink()
        print(f"removed {rel(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
