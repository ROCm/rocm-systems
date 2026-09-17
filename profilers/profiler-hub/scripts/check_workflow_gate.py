#!/usr/bin/env python3
"""Fail if a profiler-hub workflow configures profiler-hub without stating the gate.

check_dependency_gate.py guards the callee side: no CMake file can reach the
network outside PROFILER_HUB_FETCH_DEPENDENCIES. This guards the caller side:
every CI configure of profiler-hub passes an explicit value for that option
instead of inheriting whatever it happens to default to. Either value is
acceptable; silence is not.

Scope, so a green run is read for what it is: .github/workflows/profiler-hub-*.yml
and nothing else. TheRock configures profiler-hub from its own profiler/CMakeLists.txt
in another repository, which no check here can see. Green means profiler-hub's own
workflows state the flag, not that every caller does.

Usage: check_workflow_gate.py [REPO_DIR]
"""

import os
import re
import shlex
import sys
from pathlib import Path

GATE = "PROFILER_HUB_FETCH_DEPENDENCIES"
OPTION_PREFIX = "PROFILER_HUB_"

# Repo-relative source directory of the profiler-hub project. tests/find_package
# sits underneath it and is a separate project consuming an installed
# profiler-hub, which is why classify() compares by identity and not by prefix.
PROJECT_DIR = "profilers/profiler-hub"

WORKFLOW_DIR = ".github/workflows"
WORKFLOW_GLOB = "profiler-hub-*.yml"

# cmake modes that are not a configure. Without this, `cmake --build build`
# would offer `build` as a positional source directory.
NON_CONFIGURE = {"--build", "--install", "--version", "-E", "-P", "--open"}

# Options whose value is a separate token, so a value is never mistaken for the
# positional source directory of the `cmake <srcdir>` form.
OPTS_WITH_VALUE = {
    "-S",
    "--source-dir",
    "-B",
    "--build-dir",
    "-G",
    "-T",
    "-A",
    "-D",
    "-U",
    "-C",
    "--preset",
    "--toolchain",
    "--log-level",
    "--graphviz",
}

# A command word can sit behind assignments and wrappers: `CC=gcc cmake …`.
WRAPPERS = {"sudo", "env", "time", "nohup", "exec"}
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

STEP_ITEM = re.compile(r"^(\s*)-\s+(\S.*)$")
KEY = re.compile(r"^(\s*)([A-Za-z0-9_.-]+):\s*(.*)$")
SEPARATORS = re.compile(r"&&|\|\||[;|()]")


def _indent(line):
    return len(line) - len(line.lstrip(" "))


def _block(lines, start, key_indent):
    """Lines of a block scalar: everything indented past the key introducing it."""
    out = []
    i = start
    while i < len(lines):
        if lines[i].strip() and _indent(lines[i]) <= key_indent:
            break
        out.append((i + 1, lines[i]))
        i += 1
    return out, i


def steps(text):
    """Yield one dict per workflow step: job, name, line, workdir, run.

    An indentation scan over the subset of YAML that workflow files use, not a
    parser. `run:` bodies are taken verbatim, which is what makes the cmake
    command inside profiler-hub-ci.yml's heredoc visible.

    Blind spot: flow-style steps (`steps: [{...}]`) and a folded `run: >` body
    are not understood, and their configures would go unseen.
    """
    lines = text.splitlines()
    job = None
    job_workdir = None
    in_jobs = False
    defaults_indent = None
    steps_indent = None
    step = None
    i = 0
    while i < len(lines):
        raw = lines[i]
        if not raw.strip() or raw.lstrip().startswith("#"):
            i += 1
            continue
        ind = _indent(raw)

        if step is not None and ind < step["_indent"]:
            yield step
            step = None
        if steps_indent is not None and ind <= steps_indent:
            steps_indent = None
        if defaults_indent is not None and ind <= defaults_indent:
            defaults_indent = None

        if ind == 0:
            in_jobs = raw.startswith("jobs:")
            job = None
            job_workdir = None
            i += 1
            continue

        item = STEP_ITEM.match(raw)
        if in_jobs and steps_indent is not None and item and ind > steps_indent:
            if step is not None:
                yield step
            step = {
                "job": job,
                "name": None,
                "line": i + 1,
                "workdir": job_workdir,
                "run": [],
                "_indent": ind + 2,
            }
            raw = " " * (ind + 2) + item.group(2)
            ind = ind + 2

        key = KEY.match(raw)
        if not key:
            i += 1
            continue
        name, value = key.group(2), key.group(3).strip()

        if step is None:
            if in_jobs and ind == 2 and not value:
                job = name
                job_workdir = None
            elif name == "defaults":
                defaults_indent = ind
            elif name == "working-directory" and defaults_indent is not None:
                job_workdir = value
            elif name == "steps":
                steps_indent = ind
            i += 1
            continue

        if name == "name" and step["name"] is None:
            step["name"] = value
        elif name == "working-directory":
            step["workdir"] = value
        elif name == "run":
            if value and value not in ("|", "|-", "|+", ">", ">-", ">+"):
                step["run"] = [(i + 1, value)]
                i += 1
                continue
            step["run"], i = _block(lines, i + 1, ind)
            continue
        i += 1

    if step is not None:
        yield step


def logical_lines(run):
    """Join shell line continuations, dropping comments. Yields (line, text)."""
    buf = ""
    start = None
    for lineno, text in run:
        body = text.strip()
        if not body or body.startswith("#"):
            continue
        if start is None:
            start = lineno
        if body.endswith("\\"):
            buf += body[:-1] + " "
            continue
        yield start, buf + body
        buf = ""
        start = None
    if buf:
        yield start, buf


def cmake_commands(run):
    """Yield (line, argv) for every cmake invocation in a run block.

    cmake has to be the command word, not merely a token: `apt-get install -y
    cmake gcc` otherwise reads as a configure of a source directory named gcc.

    Blind spot: this reads text, it does not run a shell. A command assembled at
    runtime, or a configure inside a script file the workflow merely invokes, is
    not seen.
    """
    for lineno, text in logical_lines(run):
        for segment in SEPARATORS.split(text):
            try:
                tokens = shlex.split(segment, posix=True)
            except ValueError:
                tokens = segment.split()
            wrapped = False
            for pos, token in enumerate(tokens):
                if token == "cmake" or token.endswith("/cmake"):
                    yield lineno, tokens[pos + 1 :]
                    break
                if ASSIGNMENT.match(token) or token in WRAPPERS:
                    wrapped = wrapped or token in WRAPPERS
                    continue
                if wrapped and token.startswith("-"):
                    continue
                break


def defines(argv):
    """The -D arguments of a cmake command, as NAME or NAME:TYPE=VALUE strings."""
    out = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "-D" and i + 1 < len(argv):
            out.append(argv[i + 1])
            i += 2
            continue
        if token.startswith("-D") and len(token) > 2:
            out.append(token[2:])
        i += 1
    return out


def source_dir(argv):
    """The configure source directory, or None if this is not a configure."""
    if any(token in NON_CONFIGURE for token in argv):
        return None
    src = None
    positional = None
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ("-S", "--source-dir") and i + 1 < len(argv):
            src = argv[i + 1]
            i += 2
            continue
        if token.startswith("--source-dir="):
            src = token.split("=", 1)[1]
        elif token.startswith("-S") and len(token) > 2:
            src = token[2:]
        elif token in OPTS_WITH_VALUE:
            i += 2
            continue
        elif not token.startswith("-") and positional is None:
            positional = token
        i += 1
    return src if src is not None else positional


def classify(src, workdir, names):
    """Is this a profiler-hub configure? Returns a reason, or None if it is not.

    Identity of the source directory is the rule. The -DPROFILER_HUB_* fallback
    applies only where a shell expansion makes the path unreadable, because on
    its own it would miss a configure that passes no options at all - which is
    the very thing this check exists to forbid.

    Blind spot: a configure whose source path is a shell expansion and which
    passes no PROFILER_HUB_ option either is invisible to both handles.
    """
    unresolvable = "$" in src or "$" in (workdir or "")
    if not unresolvable:
        resolved = os.path.normpath(os.path.join(workdir or ".", src))
        if resolved == os.path.normpath(PROJECT_DIR):
            return f"configures {PROJECT_DIR} (-S {src})"
        return None
    if any(name.startswith(OPTION_PREFIX) for name in names):
        return f"sets {OPTION_PREFIX}* options with an unreadable source path"
    return None


def gate_value(names_values):
    """The value the gate was given on this command line, or None if unstated.

    Two blind spots, in opposite directions. A flag reaching cmake through a
    shell variable (`cmake ${EXTRA_ARGS}`) reads as unstated - a false alarm,
    the safe direction. A matrix-expanded value counts as stated, and whether
    every matrix cell yields a legal one is not checked.
    """
    for define in names_values:
        name, _, value = define.partition("=")
        if name.split(":")[0] == GATE:
            return value
    return None


def scan(text):
    """Yield (line, step, reason, gate) for every profiler-hub configure found."""
    for step in steps(text):
        for line, argv in cmake_commands(step["run"]):
            src = source_dir(argv)
            if src is None:
                continue
            args = defines(argv)
            reason = classify(src, step["workdir"], [d.partition("=")[0] for d in args])
            if reason is None:
                continue
            yield line, step, reason, gate_value(args)


def main(argv):
    repo = (
        Path(argv[1]).resolve()
        if len(argv) > 1
        else Path(__file__).resolve().parents[3]
    )
    failures = 0

    # Two-directional, like the sibling script's allowance list: if the project
    # moves, this check must fail rather than quietly match nothing.
    if not (repo / PROJECT_DIR / "CMakeLists.txt").is_file():
        print(f"::error::PROJECT_DIR {PROJECT_DIR} is not a CMake project under {repo}")
        return 1

    workflows = sorted((repo / WORKFLOW_DIR).glob(WORKFLOW_GLOB))
    if not workflows:
        print(f"::error::no {WORKFLOW_GLOB} files under {repo / WORKFLOW_DIR}")
        return 1

    total = 0
    for path in workflows:
        rel = path.relative_to(repo).as_posix()
        found = list(scan(path.read_text(encoding="utf-8", errors="replace")))
        total += len(found)
        for line, step, reason, gate in found:
            where = f"{step['job']} / {step['name']}"
            if gate is None:
                print(
                    f"::error file={rel},line={line}::{where}: {reason} without "
                    f"passing -D{GATE}=<ON|OFF>; the option default must not be relied on"
                )
                failures += 1
            elif not gate:
                print(
                    f"::error file={rel},line={line}::{where}: -D{GATE}= has an empty value"
                )
                failures += 1
        print(f"checked: {rel} - {len(found)} profiler-hub configure(s)")

    # A scanner that stops matching reports zero problems and reads exactly like
    # a clean branch. The branch has profiler-hub configures; finding none means
    # the scan is broken, not that the workflows are.
    if not total:
        print(
            f"::error::no profiler-hub configure found in any {WORKFLOW_GLOB}; "
            "the workflow scan is not seeing what it should"
        )
        return 1

    if failures:
        print(f"\n{failures} profiler-hub configure(s) do not state {GATE}.")
        print(f"Pass -D{GATE}=ON or =OFF explicitly on the cmake command line.")
        return 1
    print(
        f"OK: {total} profiler-hub configure(s) across {len(workflows)} "
        f"workflow(s) state {GATE} explicitly."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
