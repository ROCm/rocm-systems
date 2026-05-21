#!/usr/bin/env python
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook to reject hardcoded secrets in environment assignments."""

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_PREFIX = "projects/rocprofiler-compute/"
ALLOWLIST_PRAGMA = "# pragma: allowlist secret"
BINARY_SCAN_BYTES = 8192
ENTROPY_MIN_LENGTH = 20
ENTROPY_THRESHOLD = 4.0

EXCLUDED_PATH_PREFIXES = (
    f"{PROJECT_PREFIX}src/vendored/",
)
EXCLUDED_PATH_NAMES = set[str]()

PYTHON_ENV_PATTERNS = (
    re.compile(
        r"os\.environ\[(?P<quote>['\"])(?P<name>\w+)(?P=quote)\]"
        r"\s*=\s*(?P<value>['\"][^'\"]*['\"])",
    ),
    re.compile(
        r"os\.environ\.setdefault\(\s*(?P<quote>['\"])(?P<name>\w+)"
        r"(?P=quote)\s*,\s*(?P<value>['\"][^'\"]*['\"])",
    ),
    re.compile(
        r"os\.putenv\(\s*(?P<quote>['\"])(?P<name>\w+)(?P=quote)"
        r"\s*,\s*(?P<value>['\"][^'\"]*['\"])",
    ),
)
SHELL_ENV_PATTERN = re.compile(
    r"^\s*(?:export\s+)?(?P<name>[A-Z_][A-Z0-9_]+)=(?P<value>[^\s#]+)",
)
DOCKER_ENV_PATTERN = re.compile(
    r"^\s*ENV\s+(?P<name>[A-Z_][A-Z0-9_]+)[=\s]+(?P<value>\S+)",
)
YAML_ENV_BLOCK_PATTERN = re.compile(r"^\s*(?:env|environment):\s*(?:#.*)?$")
YAML_ENV_PATTERN = re.compile(
    r"^\s*(?P<name>[A-Z][A-Z0-9_]{2,}):\s*"
    r"(?P<value>['\"]?[^#\n]+['\"]?)\s*$",
)
CPP_ENV_PATTERNS = (
    re.compile(
        r"setenv\(\s*(?P<quote>['\"])(?P<name>\w+)(?P=quote)"
        r"\s*,\s*(?P<value>['\"][^'\"]*['\"])",
    ),
    re.compile(r"putenv\(\s*(?P<value>['\"]\w+=[^'\"]+['\"])"),
)
SECRET_NAME_PATTERN = re.compile(
    r"(SECRET|TOKEN|PASSWORD|PASSWD|API[_-]?KEY|PRIVATE[_-]?KEY|ACCESS[_-]?KEY)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SecretPattern:
    """A named pattern that identifies a known secret value shape."""

    name: str
    expression: re.Pattern[str]


@dataclass(frozen=True)
class Assignment:
    """A parsed environment-variable assignment from one source line."""

    name: str
    value: str


@dataclass(frozen=True)
class Finding:
    """A secret-like environment-variable assignment found in a file."""

    filepath: str
    line_number: int
    name: str
    reason: str


SECRET_VALUE_PATTERNS = (
    SecretPattern(
        "pem-private-key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    ),
    SecretPattern("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
)


def main() -> int:
    """Run the environment secret check."""
    args = _parse_args()
    repo_paths = _files_to_check(args.all, args.files)
    findings = _scan_repo_paths(repo_paths)

    if not findings:
        return 0

    print("Environment variable secret check failed:\n")
    for finding in findings:
        print(
            f"{finding.filepath}:{finding.line_number}  "
            f"{finding.name} = <redacted>  reason: {finding.reason}",
        )
    print(f"\nAdd `{ALLOWLIST_PRAGMA}` to a line only after verifying it is safe.")
    return 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect secret-like values in environment-variable assignments.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan all tracked project files instead of staged files.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional explicit files to scan. Defaults to staged files.",
    )
    return parser.parse_args()


def _files_to_check(scan_all: bool, explicit_files: list[str]) -> list[str]:
    if explicit_files:
        return [_normalize_repo_path(pathname) for pathname in explicit_files]

    if scan_all:
        return _get_tracked_files()

    return _get_staged_files()


def _scan_repo_paths(repo_paths: list[str]) -> list[Finding]:
    findings = []
    for repo_path in repo_paths:
        if _should_skip_repo_path(repo_path):
            continue

        absolute_path = REPO_ROOT / repo_path
        if _is_binary_file(absolute_path):
            continue

        findings.extend(_scan_text_file(repo_path, absolute_path))

    return findings


def _get_staged_files() -> list[str]:
    result = _run_git(["diff", "--cached", "--name-only", "--diff-filter=ACM"])
    return result.stdout.splitlines()


def _get_tracked_files() -> list[str]:
    result = _run_git(["ls-files", PROJECT_PREFIX])
    return result.stdout.splitlines()


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def _normalize_repo_path(pathname: str) -> str:
    path = Path(pathname)
    if path.is_absolute():
        try:
            return path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return pathname

    return pathname.lstrip("./")


def _should_skip_repo_path(repo_path: str) -> bool:
    if not repo_path.startswith(PROJECT_PREFIX):
        return True

    if repo_path in EXCLUDED_PATH_NAMES:
        return True

    if any(repo_path.startswith(prefix) for prefix in EXCLUDED_PATH_PREFIXES):
        return True

    path = Path(repo_path)
    return path.match("*.lock") or path.match("*.lock.json")


def _is_binary_file(filepath: Path) -> bool:
    try:
        return b"\0" in filepath.read_bytes()[:BINARY_SCAN_BYTES]
    except OSError:
        return True


def _scan_text_file(repo_path: str, absolute_path: Path) -> list[Finding]:
    try:
        lines = absolute_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    return _find_secret_assignments(repo_path, lines)


def _find_secret_assignments(repo_path: str, lines: list[str]) -> list[Finding]:
    findings = []
    yaml_env_block_indent: Optional[int] = None

    for line_number, line in enumerate(lines, start=1):
        yaml_env_block_indent = _updated_yaml_env_block_indent(
            line,
            yaml_env_block_indent,
        )
        if ALLOWLIST_PRAGMA in line:
            continue

        assignments = _line_assignments(line, yaml_env_block_indent is not None)
        for assignment in assignments:
            reason = _secret_reason(assignment.name, assignment.value)
            if reason is None:
                continue

            findings.append(
                Finding(
                    filepath=repo_path,
                    line_number=line_number,
                    name=assignment.name,
                    reason=reason,
                ),
            )

    return findings


def _updated_yaml_env_block_indent(
    line: str,
    current_indent: Optional[int],
) -> Optional[int]:
    stripped_line = line.strip()
    if not stripped_line or stripped_line.startswith("#"):
        return current_indent

    line_indent = len(line) - len(line.lstrip(" "))
    if YAML_ENV_BLOCK_PATTERN.match(line):
        return line_indent

    if current_indent is not None and line_indent <= current_indent:
        return None

    return current_indent


def _line_assignments(line: str, inside_yaml_env_block: bool) -> list[Assignment]:
    assignments = []

    for expression in PYTHON_ENV_PATTERNS:
        assignments.extend(_assignments_from_expression(line, expression))

    assignments.extend(_assignments_from_expression(line, SHELL_ENV_PATTERN))
    assignments.extend(_assignments_from_expression(line, DOCKER_ENV_PATTERN))

    for expression in CPP_ENV_PATTERNS:
        assignments.extend(_assignments_from_expression(line, expression))

    if inside_yaml_env_block:
        assignments.extend(_assignments_from_expression(line, YAML_ENV_PATTERN))

    return assignments


def _assignments_from_expression(
    line: str,
    expression: re.Pattern[str],
) -> list[Assignment]:
    assignments = []
    for match in expression.finditer(line):
        assignment = _assignment_from_match(match)
        if assignment is not None:
            assignments.append(assignment)

    return assignments


def _assignment_from_match(match: re.Match[str]) -> Optional[Assignment]:
    match_groups = match.groupdict()
    if "name" in match_groups and match_groups["name"] is not None:
        return Assignment(
            name=match_groups["name"],
            value=match_groups["value"],
        )

    return _assignment_from_putenv_value(match_groups["value"])


def _assignment_from_putenv_value(value: str) -> Optional[Assignment]:
    cleaned_value = _clean_raw_value(value)
    if "=" not in cleaned_value:
        return None

    name, secret_value = cleaned_value.split("=", 1)
    return Assignment(name=name, value=secret_value)


def _secret_reason(name: str, raw_value: str) -> Optional[str]:
    value = _clean_raw_value(raw_value)
    for secret_pattern in SECRET_VALUE_PATTERNS:
        if secret_pattern.expression.search(value):
            return secret_pattern.name

    if _has_high_entropy(value):
        return "high-entropy"

    if SECRET_NAME_PATTERN.search(name) and _is_literal_secret_value(value):
        return "secret-name"

    return None


def _clean_raw_value(raw_value: str) -> str:
    value = raw_value.strip().rstrip(",;)")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]

    return value


def _has_high_entropy(value: str) -> bool:
    if len(value) < ENTROPY_MIN_LENGTH:
        return False

    if re.fullmatch(r"[A-Za-z0-9_+=-]+", value) is None:
        return False

    if not re.search(r"[A-Z]", value) or not re.search(r"\d", value):
        return False

    return _shannon_entropy(value) >= ENTROPY_THRESHOLD


def _shannon_entropy(value: str) -> float:
    entropy = 0.0
    for character in set(value):
        probability = value.count(character) / len(value)
        entropy -= probability * math.log2(probability)

    return entropy


def _is_literal_secret_value(value: str) -> bool:
    if not value:
        return False

    if value.startswith("$"):
        return False

    return True


if __name__ == "__main__":
    sys.exit(main())
