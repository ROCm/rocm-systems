#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Require changelog and documentation updates for RCCL env-var changes.

Tracks declarations, exact ``NCCL_`` / ``RCCL_`` string reads, and new
``ncclParam`` / ``rcclParam`` uses under ``projects/rccl/src``. Documentation
must change the entry's prose, not only name the variable. NCCL-only findings
may use either env-var page; findings containing an RCCL name require the RCCL
page.

``[no-behavior-change]`` overrides the check. ``[internal-env]`` waives only
the documentation update.
"""

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Literal

SRC_TREE = "projects/rccl/src/"
CHANGELOG = "projects/rccl/CHANGELOG.md"
NCCL_ENV_DOC = "projects/rccl/docs/userguide/source/env.rst"
RCCL_ENV_DOC = "projects/rccl/docs/api-reference/env-variables.rst"
ENV_DOCS = (NCCL_ENV_DOC, RCCL_ENV_DOC)
COMMENT_MARKER = "<!-- rccl-default-behavior -->"
COMMENT_AUTHOR = "github-actions[bot]"
PASS_MESSAGE = "No undocumented RCCL or NCCL environment-variable changes."
SKIP_TOKEN = "[no-behavior-change]"
INTERNAL_TOKEN = "[internal-env]"
SKIPPED_MESSAGE = f"{SKIP_TOKEN} recorded; the RCCL env-var check was skipped."
UNREADABLE_MESSAGE = (
    "The check cannot parse this declaration, so it cannot tell whether an env var changed. "
    f"Use the same argument layout as other declarations of this macro, or put {SKIP_TOKEN} "
    "in the PR title or body."
)
_MAX_CONTINUATION = 40
_PAGE_SIZE = 100

_MACROS = ("DEFINE_NCCL_PARAM", "RCCL_PARAM_NCCL_ALIAS", "RCCL_PARAM", "NCCL_PARAM")
_OPENER_RE = re.compile(rf"^(?:{'|'.join(_MACROS)})\s*\(")
_ENV_EDGE = r"[A-Za-z0-9_]"
_FULL_ENV_RE = re.compile(rf"(?:NCCL|RCCL)_[A-Za-z0-9_]+")
_ENV_NAME_RE = re.compile(rf"(?<!{_ENV_EDGE})(?:NCCL|RCCL)_[A-Za-z0-9_]+(?!{_ENV_EDGE})")
_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_ACCESSOR_RE = re.compile(r"\b(?:ncclParam|rcclParam)[A-Za-z0-9_]+\b")
_DECL_GREP = r"NCCL_PARAM\s*\(|RCCL_PARAM(_NCCL_ALIAS)?\s*\(|DEFINE_NCCL_PARAM\s*\("
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PREFIXES = {
    "NCCL_PARAM": ("NCCL_",),
    "RCCL_PARAM": ("RCCL_",),
    "RCCL_PARAM_NCCL_ALIAS": ("RCCL_", "NCCL_"),
}
_ACCESSOR_PREFIX = {
    "NCCL_PARAM": "ncclParam",
    "RCCL_PARAM": "rcclParam",
    "RCCL_PARAM_NCCL_ALIAS": "rcclParam",
}

Kind = Literal["added", "removed", "default", "referenced"]
_KIND_RANK: dict[Kind, int] = {"default": 0, "removed": 1, "added": 2, "referenced": 3}


@dataclass(frozen=True)
class ParamCall:
    """A declaration, read, or mention of env vars on one source line.

    A declaration carries its default. A string literal that is exactly the
    name is a read, and a name inside longer text, such as a log message, is a
    ``mention``. Neither has a default.
    """

    env_names: tuple[str, ...]
    default: str | None
    path: str
    line: int
    mention: bool = False
    accessor: str | None = None


@dataclass(frozen=True, order=True)
class Place:
    path: str
    line: int


@dataclass(frozen=True)
class Finding:
    """An env-var change and what the PR still has to add for it.

    ``places`` is every line that made the change. A removal's lines are in
    the base revision. ``evaluate`` sets ``needs_changelog`` and ``needs_docs``.
    """

    kind: Kind
    env_names: tuple[str, ...]
    old_default: str | None
    new_default: str | None
    places: tuple[Place, ...]
    needs_changelog: bool = True
    needs_docs: bool = False

    @property
    def primary_env(self) -> str:
        return self.env_names[0]


def calls_in(source: str, path: str) -> tuple[ParamCall, ...]:
    """Env declarations and quoted ``NCCL_`` / ``RCCL_`` names in ``source``.

    A line that continues a ``/* */`` block with ``* `` is a comment and is skipped.
    """
    lines = source.splitlines()
    found: list[ParamCall] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if _OPENER_RE.match(stripped):
            calls, index = _call_at(lines, index, path)
            found.extend(calls)
            continue
        if not stripped.startswith("* "):
            found.extend(_quoted_envs(lines[index], path, index + 1))
        index += 1
    return tuple(found)


def _call_at(lines: list[str], index: int, path: str) -> tuple[list[ParamCall], int]:
    """Parse the declaration opening at ``lines[index]``. Return its calls and the next line."""
    start = index
    parts = [_strip_comments(lines[index]).strip()]
    while _still_open(parts) and index + 1 < len(lines) and len(parts) <= _MAX_CONTINUATION:
        index += 1
        parts.append(_strip_comments(lines[index]).strip())
    joined = " ".join(parts)
    call = parse_declaration(joined, path, start + 1)
    found = [call] if call is not None else []
    found.extend(_quoted_envs(joined, path, start + 1))
    return found, index + 1


def _still_open(parts: list[str]) -> bool:
    code = _STRING_RE.sub("", " ".join(parts))
    return code.count("(") > code.count(")")


def parse_declaration(code: str, path: str, line_no: int) -> ParamCall | None:
    """The declaration in comment-free ``code``, or None when it does not parse.

    ``DEFINE_NCCL_PARAM(name, type, KEY, default, ...)`` names the variable
    ``KEY``. The other macros take ``(name, "ENV", default)`` and prefix ``ENV``.
    """
    head, _, rest = code.removesuffix(";").rstrip().partition("(")
    macro = head.strip()
    if not rest.endswith(")"):
        return None
    args = _split_args(rest[:-1])
    if macro == "DEFINE_NCCL_PARAM":
        if len(args) < 4 or _FULL_ENV_RE.fullmatch(args[2]) is None:
            return None
        names, default = (args[2],), args[3]
    elif len(args) == 3 and _STRING_RE.fullmatch(args[1]):
        names, default = tuple(prefix + args[1][1:-1] for prefix in _PREFIXES[macro]), args[2]
    else:
        return None
    return ParamCall(
        env_names=names,
        default=default,
        path=path,
        line=line_no,
        accessor=_accessor_symbol(macro, args[0]),
    )


def _accessor_symbol(macro: str, c_name: str) -> str | None:
    """Derive the accessor symbol, preserving lowercase C names."""
    if _IDENT_RE.fullmatch(c_name) is None:
        return None
    if macro == "DEFINE_NCCL_PARAM":
        return c_name if _ACCESSOR_RE.fullmatch(c_name) else None
    return _ACCESSOR_PREFIX[macro] + c_name


def _split_args(text: str) -> list[str]:
    """Split macro arguments as the preprocessor does: commas inside strings or parentheses stay put."""
    args: list[str] = []
    buf: list[str] = []
    parens = 0
    quoted = False
    for index, char in enumerate(text):
        if char == '"' and not _escaped(text, index):
            quoted = not quoted
        elif not quoted and char == "(":
            parens += 1
        elif not quoted and char == ")":
            parens -= 1
        elif not quoted and char == "," and parens == 0:
            args.append("".join(buf).strip())
            buf = []
            continue
        buf.append(char)
    if buf or args:
        args.append("".join(buf).strip())
    return args


def _escaped(text: str, index: int) -> bool:
    slashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        slashes += 1
        index -= 1
    return slashes % 2 == 1


def _quoted_envs(line: str, path: str, line_no: int) -> list[ParamCall]:
    """``NCCL_`` / ``RCCL_`` names inside string literals, which carry no default."""
    if '"' not in line or ("NCCL_" not in line and "RCCL_" not in line):
        return []
    found: list[ParamCall] = []
    for literal in _STRING_RE.findall(_strip_comments(line)):
        text = literal[1:-1]
        mention = _FULL_ENV_RE.fullmatch(text) is None
        found.extend(
            ParamCall(env_names=(name,), default=None, path=path, line=line_no, mention=mention)
            for name in _FULL_ENV_RE.findall(text)
        )
    return found


def _strip_comments(line: str) -> str:
    """Drop ``//`` and ``/* */`` comments, leaving string literals intact."""
    out: list[str] = []
    quoted = False
    index = 0
    while index < len(line):
        char = line[index]
        if char == '"' and not _escaped(line, index):
            quoted = not quoted
            out.append(char)
        elif not quoted and line[index : index + 2] == "//":
            break
        elif not quoted and line[index : index + 2] == "/*":
            end = line.find("*/", index + 2)
            if end == -1:
                break
            index = end + 1
        else:
            out.append(char)
        index += 1
    return "".join(out)


def findings_between(before: dict[str, str], after: dict[str, str]) -> list[Finding]:
    """Env names added, removed, or given a new default between two sets of files."""
    old = _by_name(before)
    new = _by_name(after)
    names = sorted(old.keys() | new.keys())
    found = (_pair_name(name, old.get(name, []), new.get(name, [])) for name in names)
    return [item for item in found if item is not None]


def _by_name(sources: dict[str, str]) -> dict[str, list[ParamCall]]:
    """Calls grouped under every name they carry, so adding or dropping an alias pairs on its own."""
    grouped: dict[str, list[ParamCall]] = {}
    for path in sorted(sources):
        for call in calls_in(sources[path], path):
            for name in call.env_names:
                grouped.setdefault(name, []).append(call)
    return grouped


def _pair_name(name: str, old: list[ParamCall], new: list[ParamCall]) -> Finding | None:
    read_before = any(not call.mention for call in old)
    read_after = any(not call.mention for call in new)
    old_values = _values(old)
    new_values = _values(new)
    kind: Kind
    if not read_before and not read_after:
        return None
    if not read_before:
        kind, changed = "added", new
    elif not read_after:
        kind, changed = "removed", old
    elif old_values != new_values:
        kind, changed = "default", [call for call in new if call.default is not None] or new
    else:
        return None
    aliases = (alias for call in (*old, *new) for alias in call.env_names)
    return Finding(
        kind=kind,
        env_names=tuple(dict.fromkeys([name, *aliases])),
        old_default=old_values,
        new_default=new_values,
        places=tuple(sorted({Place(call.path, call.line) for call in changed})),
    )


def reference_findings(
    before: dict[str, str],
    after: dict[str, str],
    accessors: dict[str, tuple[str, ...]],
) -> list[Finding]:
    """Find declared accessors gained by more changed files than lose them."""
    gained: dict[str, list[Place]] = {}
    lost: dict[str, int] = {}
    for path in sorted(set(before) | set(after)):
        old = {symbol for symbol, _line in _accessors_in(before.get(path, ""))}
        by_symbol: dict[str, list[Place]] = {}
        for symbol, line in _accessors_in(after.get(path, "")):
            by_symbol.setdefault(symbol, []).append(Place(path, line))
        for symbol, places in by_symbol.items():
            if symbol not in old:
                gained.setdefault(symbol, []).extend(places)
        for symbol in old - by_symbol.keys():
            lost[symbol] = lost.get(symbol, 0) + 1
    found: list[Finding] = []
    for symbol, places in sorted(gained.items()):
        env_names = accessors.get(symbol)
        if not env_names:
            continue
        if len({place.path for place in places}) <= lost.get(symbol, 0):
            continue
        found.append(
            Finding(
                kind="referenced",
                env_names=env_names,
                old_default=None,
                new_default=None,
                places=tuple(sorted(set(places))),
            )
        )
    return found


def _accessors_in(source: str) -> list[tuple[str, int]]:
    """``ncclParamFoo`` / ``rcclParamFoo`` identifiers outside comments and strings."""
    found: list[tuple[str, int]] = []
    for number, line in enumerate(source.splitlines(), start=1):
        if line.strip().startswith("* "):
            continue
        code = _STRING_RE.sub('""', _strip_comments(line))
        found.extend((match.group(0), number) for match in _ACCESSOR_RE.finditer(code))
    return found


def accessors_defined(sources: dict[str, str]) -> dict[str, tuple[str, ...]]:
    index: dict[str, list[str]] = {}
    for path in sorted(sources):
        for call in calls_in(sources[path], path):
            if call.accessor is None:
                continue
            bucket = index.setdefault(call.accessor, [])
            for name in call.env_names:
                if name not in bucket:
                    bucket.append(name)
    return {symbol: tuple(names) for symbol, names in index.items()}


def _values(calls: list[ParamCall]) -> str | None:
    values = sorted({call.default for call in calls if call.default is not None})
    if not values:
        return None
    return ", ".join(values)


def unreadable_declarations(sources: dict[str, str]) -> list[Place]:
    """Declarations in ``sources`` that do not parse, so their variables cannot be checked."""
    places: list[Place] = []
    for path in sorted(sources):
        parsed = {call.line for call in calls_in(sources[path], path) if call.default is not None}
        places.extend(
            Place(path, number)
            for number, line in enumerate(sources[path].splitlines(), start=1)
            if _OPENER_RE.match(line.strip()) and number not in parsed
        )
    return places


def mentions(text: str, finding: Finding) -> bool:
    """Whether ``text`` names one of the finding's variables as a whole token."""
    return any(
        re.search(rf"(?<!{_ENV_EDGE}){re.escape(name)}(?!{_ENV_EDGE})", text)
        for name in finding.env_names
    )


def suggestion(finding: Finding) -> str:
    """A CHANGELOG bullet with the variable and default filled in."""
    name = f"`{finding.primary_env}`"
    if finding.kind == "referenced":
        return f"* {name}: <why>"
    if finding.kind == "default":
        old = _tick(finding.old_default) or "none"
        new = _tick(finding.new_default) or "none"
        return f"* {name} default changed {old} -> {new}: <why>"
    if finding.kind == "added":
        if finding.new_default is None:
            return f"* {name}: <why>"
        return f"* {name} (default `{finding.new_default}`): <why>"
    if finding.old_default is None:
        return f"* {name} removed: <why>"
    return f"* {name} removed (was default `{finding.old_default}`): <why>"


def _annotations(finding: Finding) -> list[str]:
    """One annotation per place, so each changed file shows it inline.

    A missing CHANGELOG line and a missing doc entry are both ``::error``.
    A removal has no line, so it gets one per file.
    """
    if finding.kind == "referenced":
        detail = "newly referenced"
    elif finding.kind == "added" and finding.new_default is None:
        detail = "added"
    elif finding.kind == "added":
        detail = f"added with default {finding.new_default}"
    elif finding.kind == "default":
        detail = f"default changed {finding.old_default or 'none'} -> {finding.new_default or 'none'}"
    elif finding.old_default is None:
        detail = "removed"
    else:
        detail = f"removed (was default {finding.old_default})"
    asks: list[tuple[str, str]] = []
    if finding.needs_changelog:
        asks.append(("error", f"Name it in {CHANGELOG}, or put {SKIP_TOKEN} in the PR title or body."))
    if finding.needs_docs:
        pages = " or ".join(doc_pages_for(finding))
        asks.append((
            "error",
            f"Describe this change in {pages}, or put {INTERNAL_TOKEN} in the PR title or body "
            "if it is not meant for users.",
        ))
    return list(dict.fromkeys(
        f"::{level} {_where(finding, place)}::{finding.primary_env} {detail}. {ask}"
        for level, ask in asks
        for place in finding.places
    ))


def _where(finding: Finding, place: Place) -> str:
    """Annotation location. A removal's line is in the base revision, so it is left off."""
    if finding.kind == "removed":
        return f"file={place.path}"
    return f"file={place.path},line={place.line}"


def _tick(value: str | None) -> str:
    if not value:
        return ""
    return f"`{value}`"


def _summary(undocumented: list[Finding], unreadable: list[Place]) -> str:
    if _blocks(undocumented, unreadable):
        sections = ["### Undocumented RCCL default-behavior changes\n"]
    else:
        sections = ["### New RCCL env vars without docs\n"]
    if undocumented:
        rows: list[str] = []
        for item in undocumented:
            places = "<br>".join(f"`{place.path}:{place.line}`" for place in item.places)
            rows.append(
                f"| `{item.primary_env}` | {item.kind} | {_tick(item.old_default)} | "
                f"{_tick(item.new_default)} | {_missing(item)} | {places} |"
            )
        table = "\n".join(rows)
        sections.append(
            "| Variable | Trigger | Old default | New default | Missing | Locations |\n"
            f"|---|---|---|---|---|---|\n{table}\n"
        )
        bullets = [suggestion(item) for item in undocumented if item.needs_changelog]
        if bullets:
            sections.append(f"Paste into `{CHANGELOG}`:\n\n" + "\n".join(bullets) + "\n")
        grouped: dict[tuple[str, ...], list[str]] = {}
        for item in undocumented:
            if item.needs_docs:
                grouped.setdefault(doc_pages_for(item), []).append(suggestion(item).removesuffix(": <why>"))
        for pages, entries in grouped.items():
            listed = " or ".join(f"`{path}`" for path in pages)
            sections.append(
                f"Describe these changes in {listed}, or put `{INTERNAL_TOKEN}` in the PR title "
                "or body if the variable is not meant for users:\n\n" + "\n".join(entries) + "\n"
            )
    if unreadable:
        lines = "\n".join(f"* `{place.path}:{place.line}`" for place in unreadable)
        sections.append(f"{UNREADABLE_MESSAGE}\n\n{lines}\n")
    return "\n".join(sections)


def doc_pages_for(finding: Finding) -> tuple[str, ...]:
    """NCCL-only findings may use either page; RCCL findings require its page."""
    nccl = any(name.startswith("NCCL_") for name in finding.env_names)
    rccl = any(name.startswith("RCCL_") for name in finding.env_names)
    if nccl and not rccl:
        return ENV_DOCS
    return (RCCL_ENV_DOC,)


def _doc_regions(text: str) -> dict[str, str]:
    """Each env var's entry, from its name to the next, with whitespace collapsed."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    names = list(_ENV_NAME_RE.finditer(collapsed))
    regions: dict[str, str] = {}
    for index, match in enumerate(names):
        end = names[index + 1].start() if index + 1 < len(names) else len(collapsed)
        name = match.group(0)
        regions[name] = regions.get(name, "") + collapsed[match.start() : end].strip()
    return regions


def _entry_prose(region: str | None, name: str) -> str:
    """Words in an entry besides the variable name. Markup and spacing drop out."""
    if not region:
        return ""
    body = re.sub(rf"(?<!{_ENV_EDGE}){re.escape(name)}(?!{_ENV_EDGE})", " ", region)
    words = re.findall(r"-?\d+|[A-Za-z0-9]+", body)
    return " ".join(word.casefold() for word in words)


def _region_maps(pages: dict[str, str]) -> dict[str, dict[str, str]]:
    return {path: _doc_regions(pages.get(path, "")) for path in ENV_DOCS}


def _docs_cover(
    finding: Finding,
    before: dict[str, dict[str, str]],
    after: dict[str, dict[str, str]],
) -> bool:
    """Accept changed entry prose; it need not repeat the exact code change."""
    for page in doc_pages_for(finding):
        old = before.get(page, {})
        new = after.get(page, {})
        for name in finding.env_names:
            if _entry_prose(old.get(name), name) != _entry_prose(new.get(name), name):
                return True
    return False


def _missing(finding: Finding) -> str:
    needed = [("CHANGELOG", finding.needs_changelog), ("docs", finding.needs_docs)]
    return ", ".join(label for label, missing in needed if missing)


def _blocks(undocumented: list[Finding], unreadable: list[Place]) -> bool:
    return bool(unreadable) or any(item.needs_changelog or item.needs_docs for item in undocumented)


def report(undocumented: list[Finding], unreadable: list[Place], skipped: bool) -> int:
    """Print annotations and a summary, sync the PR comment, and return the exit code."""
    if not undocumented and not unreadable:
        print(PASS_MESSAGE)
        sync_pull_request_comment(PASS_MESSAGE, create=False)
        return 0
    if skipped:
        print(f"::notice::{SKIPPED_MESSAGE}")
        for finding in undocumented:
            for place in finding.places:
                print(f"::notice {_where(finding, place)}::{finding.primary_env} {finding.kind} (override)")
        for place in unreadable:
            print(f"::notice file={place.path},line={place.line}::unreadable declaration (override)")
        sync_pull_request_comment(SKIPPED_MESSAGE, create=False)
        return 0
    for finding in undocumented:
        print(*_annotations(finding), sep="\n")
    for place in unreadable:
        print(f"::error file={place.path},line={place.line}::{UNREADABLE_MESSAGE}")
    summary = _summary(undocumented, unreadable)
    print(summary)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with Path(step_summary).open("a", encoding="utf-8") as handle:
            handle.write(summary)
    sync_pull_request_comment(summary, create=True)
    return 1 if _blocks(undocumented, unreadable) else 0


@dataclass(frozen=True)
class PullRequest:
    """The pull request this run checks and the token that may comment on it."""

    token: str
    comments_url: str


def sync_pull_request_comment(body: str, *, create: bool) -> None:
    """Create or edit the one marker-tagged comment on this pull request.

    A passing check only edits a comment that already exists, so a clean PR
    does not grow a new comment. The comment is edited in place and never
    deleted. Without a token and pull request ref this does nothing, and
    inside Actions it says so with a warning.

    The comment is advisory. A fork's read-only token, rate limiting, or a
    network error becomes a warning and never changes the check result.
    """
    pull = _pull_request()
    if pull is None:
        return
    try:
        _upsert_comment(pull, f"{COMMENT_MARKER}\n{body}", create=create)
    except (OSError, http.client.HTTPException, ValueError) as exc:
        print(f"::warning::Could not update the pull request comment: {exc}")


def _pull_request() -> PullRequest | None:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    match = re.fullmatch(r"refs/pull/(\d+)/merge", os.environ.get("GITHUB_REF", ""))
    if token and "/" in repo and match is not None:
        api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        return PullRequest(token=token, comments_url=f"{api}/repos/{repo}/issues/{match.group(1)}/comments")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print("::warning::No GITHUB_TOKEN or pull request ref; the PR comment was not updated.")
    return None


def _upsert_comment(pull: PullRequest, body: str, *, create: bool) -> None:
    existing = _marked_comment_url(pull)
    if existing is not None:
        _github_json("PATCH", existing, pull.token, {"body": body})
    elif create:
        _github_json("POST", pull.comments_url, pull.token, {"body": body})


def _marked_comment_url(pull: PullRequest) -> str | None:
    """URL of this check's own comment, searching every page of the thread."""
    page = 1
    while True:
        url = f"{pull.comments_url}?per_page={_PAGE_SIZE}&page={page}"
        listed = _github_json("GET", url, pull.token, None)
        if not isinstance(listed, list):
            raise ValueError(f"GitHub returned {type(listed).__name__} for {url}")
        for item in listed:
            if _is_marked(item):
                return str(item["url"])
        if len(listed) < _PAGE_SIZE:
            return None
        page += 1


def _is_marked(item: object) -> bool:
    """A comment by the workflow bot that carries the marker. A person quoting it is skipped."""
    if not isinstance(item, dict) or "url" not in item:
        return False
    user = item.get("user")
    author = user.get("login") if isinstance(user, dict) else None
    return author == COMMENT_AUTHOR and COMMENT_MARKER in str(item.get("body", ""))


def _github_json(method: str, url: str, token: str, payload: dict[str, str] | None) -> object:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode()
    return json.loads(raw) if raw else None


def evaluate(
    changed: list[Finding],
    *,
    changelog_added: str,
    env_docs: dict[str, str],
    internal: bool,
    known_before: set[str],
    known_after: set[str],
    env_docs_before: dict[str, str] | None = None,
) -> list[Finding]:
    """Mark findings lacking changelog or documentation coverage."""
    before_docs = {path: "" for path in ENV_DOCS} if env_docs_before is None else env_docs_before
    old_regions = _region_maps(before_docs)
    new_regions = _region_maps(env_docs)
    pending: list[Finding] = []
    for item in _one_per_variable(_without_existing(changed, known_before, known_after)):
        needs_changelog = not mentions(changelog_added, item)
        needs_docs = not internal and not _docs_cover(item, old_regions, new_regions)
        if needs_changelog or needs_docs:
            pending.append(replace(item, needs_changelog=needs_changelog, needs_docs=needs_docs))
    return pending


def _without_existing(found: list[Finding], known_before: set[str], known_after: set[str]) -> list[Finding]:
    return [
        item
        for item in found
        if not (item.kind == "added" and item.primary_env in known_before)
        and not (item.kind == "removed" and item.primary_env in known_after)
    ]


def _env_names(found: list[Finding], kind: Kind) -> set[str]:
    return {item.primary_env for item in found if item.kind == kind}


def _one_per_variable(found: list[Finding]) -> list[Finding]:
    """Merge findings whose names overlap, such as the two an alias declaration gives.

    The highest-ranked finding supplies the kind and defaults: a default edit,
    then a removal, then a new name, preferring one that carries a default.
    """
    merged: list[Finding] = []
    for item in sorted(found, key=_finding_rank):
        names = set(item.env_names)
        index = next((i for i, kept in enumerate(merged) if names & set(kept.env_names)), None)
        if index is None:
            merged.append(item)
            continue
        kept = merged[index]
        merged[index] = replace(
            kept,
            env_names=tuple(dict.fromkeys((*kept.env_names, *item.env_names))),
            places=tuple(sorted({*kept.places, *item.places})),
        )
    return sorted(merged, key=lambda item: (item.places[0], item.primary_env))


def _finding_rank(item: Finding) -> tuple[int, int, Place, bool]:
    """Merge order. Of the two findings an alias declaration gives, the ``RCCL_`` one leads."""
    has_default = item.old_default is not None or item.new_default is not None
    return (
        _KIND_RANK[item.kind],
        0 if has_default else 1,
        item.places[0],
        not item.primary_env.startswith("RCCL_"),
    )


def _git(repo: Path, args: list[str], *, ok: tuple[int, ...] = (0,)) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, encoding="utf-8", errors="replace"
    )
    if result.returncode not in ok:
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
    return result.stdout


def changed_sources(repo: Path, base_ref: str) -> tuple[dict[str, str], dict[str, str]]:
    """Base and HEAD text of the files under ``SRC_TREE`` that differ between them."""
    fields = _git(
        repo, ["diff", "--no-renames", "--name-status", "-z", base_ref, "HEAD", "--", SRC_TREE]
    ).split("\0")
    statuses = list(zip(fields[0::2], fields[1::2]))
    if statuses:
        # Prefetch base blobs once; repeated `git show` calls fetch them individually.
        _git(repo, ["diff", "--shortstat", base_ref, "HEAD", "--", SRC_TREE, CHANGELOG])
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    for status, path in statuses:
        if status != "A":
            before[path] = _git(repo, ["show", f"{base_ref}:{path}"])
        if status != "D":
            after[path] = _git(repo, ["show", f"HEAD:{path}"])
    return before, after


def known_names(repo: Path, rev: str, names: set[str]) -> set[str]:
    """Which of ``names`` product source declares or reads at ``rev``.

    ``NCCL_PARAM`` and ``RCCL_PARAM`` spell the name without its prefix, so a
    file holding either spelling is parsed the same way as a changed file.
    """
    if not names:
        return set()
    needles = sorted({*names, *(f'"{name.partition("_")[2]}"' for name in names)})
    patterns = [part for needle in needles for part in ("-e", needle)]
    listed = _git(repo, ["grep", "-I", "-l", "-z", "-F", *patterns, rev, "--", SRC_TREE], ok=(0, 1))
    found = {
        name
        for entry in listed.split("\0")
        if entry
        for call in calls_in(_git(repo, ["show", entry]), entry)
        if not call.mention
        for name in call.env_names
    }
    return names & found


def changelog_additions(repo: Path, base_ref: str) -> str:
    """Lines the pull request adds to the changelog."""
    diff = _git(repo, ["diff", "-U0", "--no-color", base_ref, "HEAD", "--", CHANGELOG])
    return "\n".join(
        line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")
    )


def accessor_map(repo: Path) -> dict[str, tuple[str, ...]]:
    """Map ``ncclParamFoo`` / ``rcclParamFoo`` to the env names declared at HEAD."""
    listed = _git(repo, ["grep", "-I", "-l", "-z", "-E", _DECL_GREP, "HEAD", "--", SRC_TREE], ok=(0, 1))
    sources: dict[str, str] = {}
    for entry in listed.split("\0"):
        if not entry:
            continue
        _rev, separator, path = entry.partition(":")
        if not separator:
            continue
        sources[path] = _git(repo, ["show", entry])
    return accessors_defined(sources)


def env_doc_pair(repo: Path, base_ref: str) -> tuple[dict[str, str], dict[str, str]]:
    """Treat pages missing at either revision as empty."""
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    for path in ENV_DOCS:
        before[path] = _git(repo, ["show", f"{base_ref}:{path}"], ok=(0, 128))
        after[path] = _git(repo, ["show", f"HEAD:{path}"], ok=(0, 128))
    return before, after


def git_root() -> Path:
    """Repository root containing the working directory."""
    return Path(_git(Path("."), ["rev-parse", "--show-toplevel"]).strip())


def _new_accessor(before: dict[str, str], after: dict[str, str]) -> bool:
    for path in set(before) | set(after):
        old = {symbol for symbol, _line in _accessors_in(before.get(path, ""))}
        if any(symbol not in old for symbol, _line in _accessors_in(after.get(path, ""))):
            return True
    return False


def check(repo: Path, base_ref: str, override_texts: list[str]) -> int:
    """Run the check against ``base_ref``. Return 0 when the PR may merge."""
    before, after = changed_sources(repo, base_ref)
    changed = findings_between(before, after)
    if _new_accessor(before, after):
        changed.extend(reference_findings(before, after, accessor_map(repo)))
    overrides = "\n".join([*override_texts, _git(repo, ["log", "--pretty=%s", f"{base_ref}..HEAD"])])
    added = _env_names(changed, "added")
    docs_before, docs_after = env_doc_pair(repo, base_ref) if changed else ({}, {})
    pending = evaluate(
        changed,
        changelog_added=changelog_additions(repo, base_ref) if changed else "",
        env_docs=docs_after,
        env_docs_before=docs_before,
        internal=INTERNAL_TOKEN in overrides,
        known_before=known_names(repo, base_ref, added),
        known_after=known_names(repo, "HEAD", _env_names(changed, "removed")),
    )
    return report(pending, unreadable_declarations(after), skipped=SKIP_TOKEN in overrides)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        default="HEAD^",
        help="Git ref the pull request is diffed against (default: HEAD^).",
    )
    args = parser.parse_args(argv)
    texts = [os.environ.get("PR_TITLE", ""), os.environ.get("PR_BODY", "")]
    try:
        return check(git_root(), args.base_ref, texts)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr or str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
