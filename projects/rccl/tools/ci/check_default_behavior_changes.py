#!/usr/bin/env python3
"""Gate RCCL PRs that change user-visible default behavior on an entry in
rcclDefaultBehaviorChanges.md.  JIRA ID: AICOMRCCL-1904

Reads a unified diff and reports every environment variable the PR touches that
has no matching entry in the markdown file.  Three trigger conditions:

  new PARAM       a PARAM macro declares an env var the diff does not also remove
  default change  a PARAM macro's default value changed; old and new are reported
  reference       an added line reads an env var (param accessor, getenv/ncclGetEnv,
                  or an "NCCL_*"/"RCCL_*" string literal)

Stdlib only, and the same command runs in CI and locally:

  # locally, against your branch point
  tools/ci/check_default_behavior_changes.py --base develop

  # in CI, diff piped in from `gh pr diff`
  gh pr diff "$PR" | tools/ci/check_default_behavior_changes.py \\
      --diff - --strip-prefix projects/rccl/ --format github

Exit status is 0 only when every detected variable is covered by an entry or
carries a BEHAVIOR-CHANGE-EXEMPT override in the PR description.

Detection maps a param accessor (ncclParamFoo) back to its env var name using a
registry built by parsing every PARAM declaration in the tree.  The runtime
registry in src/param/param_registry.cc cannot serve this purpose: only the
handful of DEFINE_NCCL_PARAM params register there, not the ~350 legacy
NCCL_PARAM/RCCL_PARAM ones.  If the tree ever migrates fully to
DEFINE_NCCL_PARAM, replace the static parse with a registry dump.
"""

import argparse
import os
import re
import subprocess
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

# Directories that ship as part of the library or its plugins.  A tunable read
# under test/ or tools/ is not a user-visible default, so those stay out.
SCAN_DIRS = ("src", "plugins", "tuner")
SCAN_SUFFIXES = (".cc", ".cpp", ".cu", ".c", ".h", ".hpp", ".cuh")

DEFAULT_MD_PATH = "rcclDefaultBehaviorChanges.md"

# Only tables under these headings count as entries, so the usage examples in the
# markdown file's own header are not mistaken for coverage.
MD_SECTION_RE = re.compile(r"^##\s+(Unreleased|RCCL\b)", re.IGNORECASE)

# PARAM macro forms taking a quoted env suffix.  The value is the prefix
# prepended to that suffix, plus the second prefix RCCL_PARAM_NCCL_ALIAS also
# accepts (it checks RCCL_<env> first and falls back to NCCL_<env>).
QUOTED_MACROS = {
    "NCCL_PARAM": ("NCCL_", None),
    "RCCL_PARAM": ("RCCL_", None),
    "RCCL_PARAM_NCCL_ALIAS": ("RCCL_", "NCCL_"),
}
# DEFINE_NCCL_PARAM(accessor, type, KEY, default, flags, parser, desc): KEY is a
# bare token that is already the full env var name.
DEFINE_MACRO = "DEFINE_NCCL_PARAM"

# RCCL_PARAM_DECLARE(Name) is a forward declaration in a header, not a
# definition -- it carries no default and must never read as a new param, so it
# is deliberately absent from the pattern below.
MACRO_CALL_RE = re.compile(
    r"\b(NCCL_PARAM|RCCL_PARAM|RCCL_PARAM_NCCL_ALIAS|DEFINE_NCCL_PARAM)\s*\("
)

ACCESSOR_RE = re.compile(r"\b((?:nccl|rccl)Param[A-Z][A-Za-z0-9_]*)")
GETENV_RE = re.compile(r"\b(getenv|ncclGetEnv)\s*\(\s*(?:\"([^\"]*)\")?")
ENV_LITERAL_RE = re.compile(r"\"((?:NCCL|RCCL)_[A-Z0-9_]+)\"")
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# BEHAVIOR-CHANGE-EXEMPT: NCCL_FOO, NCCL_BAR - reason
# The separator may be an em dash, an en dash, a colon or a spaced hyphen.
EXEMPT_LINE_RE = re.compile(
    r"^[>\s]*BEHAVIOR-CHANGE-EXEMPT\s*:\s*(?P<rest>.*)$", re.IGNORECASE | re.MULTILINE
)
EXEMPT_SPLIT_RE = re.compile(r"\s*(?:[—–]|(?<=\s)-(?=\s)|:)\s*")

TRIGGER_NEW = "new PARAM"
TRIGGER_DEFAULT = "default change"
TRIGGER_REFERENCE = "reference"
TRIGGER_RANK = {TRIGGER_NEW: 0, TRIGGER_DEFAULT: 1, TRIGGER_REFERENCE: 2}

INTEGER_DEFAULT_RE = re.compile(r"^[+-]?\d+$")

DYNAMIC_ENV = "<dynamic>"

# A newly added variable is documented in its own table, so that a reader can
# tell whether the knob is supported, what it does and what it accepts -- not
# just that it appeared.  Columns are matched by header text, lowercased.
NEW_VAR_KEY_COLUMN = "supported"
NEW_VAR_REQUIRED_COLUMNS = ("supported", "description", "accepted values")
NEW_VAR_COLUMN_LABELS = {
    "supported": "Supported",
    "description": "Description",
    "accepted values": "Accepted values",
}
NEW_VAR_HEADER = (
    "| Variable | Supported | Description | Accepted values | Default | "
    "Reason for Change |"
)
# Emitted with the header so the suggested block is a valid table on its own,
# whether the author pastes the whole thing or just the data rows.
NEW_VAR_SEPARATOR = "|---|---|---|---|---|---|"
# Yes/No are accepted as synonyms so a reviewer writing either reads the same.
SUPPORTED_VALUES = {"supported", "not supported", "yes", "no"}
SUPPORTED_DISPLAY = ("Supported", "Not supported")

# Cells the author pasted but never filled in.
PLACEHOLDER_RE = re.compile(
    r"(?:why|tbd|todo|t\.b\.d\.?|n/?a|none|xxx|\?+|\.{2,}|"
    r"describe[\w \-]*|accepted[\w \-]*|supported\?|fill[\w \-]*)",
    re.IGNORECASE,
)

# Returned by coverage_gap when nothing in the file names the variable at all.
NO_ENTRY = "no entry names it"


class Declaration(NamedTuple):
    """A single PARAM macro invocation, with the line span it occupies."""

    env: str
    accessor: str
    default: str
    path: str
    line: int
    end_line: int


class Detection(NamedTuple):
    """One reason a variable needs an entry in the markdown file."""

    env: str
    trigger: str
    path: str
    line: int
    detail: str
    old: Optional[str] = None
    new: Optional[str] = None


class MdEntry(NamedTuple):
    """One row of an entry table in rcclDefaultBehaviorChanges.md."""

    names: Set[str]  # variable names found in the row's first cell
    text: str  # the whole row, for matching a changed default's new value
    fields: Dict[str, str]  # lowercased column header -> cell


# ---------------------------------------------------------------------------
# Parsing PARAM declarations
# ---------------------------------------------------------------------------


def split_macro_args(text: str, open_paren: int) -> Optional[Tuple[List[str], int]]:
    """Split a macro invocation's arguments, respecting nesting and quotes.

    `text` is whole-file (or whole-hunk) text and `open_paren` indexes the
    macro's `(`.  Returns the argument list and the offset of the closing paren,
    or None when the parentheses never balance -- which happens when a hunk cuts
    a declaration in half.

    Arguments come back as written, because defaults are expressions rather than
    plain literals: `1LL << 30`, `(size_t)1<<20`, `1u << NCCL_LOG_WARN`.
    """
    args: List[str] = []
    depth = 0
    start = open_paren + 1
    i = open_paren
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            quote = ch
            i += 1
            while i < n and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                args.append(text[start:i])
                return [a.strip() for a in args], i
        elif ch == "," and depth == 1:
            args.append(text[start:i])
            start = i + 1
        i += 1
    return None


def normalize_default(value: str) -> str:
    """Collapse whitespace so reformatting alone never reads as a default change."""
    return re.sub(r"\s+", " ", value.strip())


def parse_declarations(text: str, path: str) -> List[Declaration]:
    """Extract every PARAM declaration from a block of text.

    Line numbers are 1-based within `text`.  Declarations split across lines do
    occur in the tree (src/misc/signals.cc, src/misc/rocm_smi_wrap.cc,
    src/transport/net_ib/p2p_resiliency_recovery.cc), hence the span.
    """
    decls: List[Declaration] = []
    for match in MACRO_CALL_RE.finditer(text):
        macro = match.group(1)
        parsed = split_macro_args(text, match.end() - 1)
        if parsed is None:
            continue
        args, close = parsed
        line = text.count("\n", 0, match.start()) + 1
        end_line = text.count("\n", 0, close) + 1

        if macro == DEFINE_MACRO:
            if len(args) < 4:
                continue
            accessor, key, default = args[0], args[2], args[3]
            if not IDENTIFIER_RE.fullmatch(key) or not IDENTIFIER_RE.fullmatch(
                accessor
            ):
                continue
            decls.append(
                Declaration(
                    key, accessor, normalize_default(default), path, line, end_line
                )
            )
            continue

        if len(args) < 3:
            continue
        name, env_arg, default = args[0], args[1], args[2]
        env_match = re.fullmatch(r"\"([^\"]*)\"", env_arg)
        if not env_match or not IDENTIFIER_RE.fullmatch(name):
            continue
        suffix = env_match.group(1)
        prefix, alias_prefix = QUOTED_MACROS[macro]
        accessor = ("rccl" if prefix == "RCCL_" else "nccl") + "Param" + name
        default = normalize_default(default)
        decls.append(
            Declaration(prefix + suffix, accessor, default, path, line, end_line)
        )
        if alias_prefix:
            decls.append(
                Declaration(
                    alias_prefix + suffix, accessor, default, path, line, end_line
                )
            )
    return decls


def in_scan_scope(path: str) -> bool:
    """True for paths inside the shipped source tree, relative to projects/rccl."""
    if not path.endswith(SCAN_SUFFIXES):
        return False
    return path.split("/", 1)[0] in SCAN_DIRS


def build_registry(root: str) -> Dict[str, Declaration]:
    """Map every param accessor in the tree to its declaration.

    An accessor declared with RCCL_PARAM_NCCL_ALIAS resolves to the RCCL_ name,
    which is the one the runtime consults first.
    """
    registry: Dict[str, Declaration] = {}
    for scan_dir in SCAN_DIRS:
        for dirpath, _dirnames, filenames in os.walk(os.path.join(root, scan_dir)):
            for filename in filenames:
                if not filename.endswith(SCAN_SUFFIXES):
                    continue
                full = os.path.join(dirpath, filename)
                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as handle:
                        text = handle.read()
                except OSError:
                    continue
                for decl in parse_declarations(text, os.path.relpath(full, root)):
                    registry.setdefault(decl.accessor, decl)
    return registry


# ---------------------------------------------------------------------------
# Parsing the diff
# ---------------------------------------------------------------------------


class Hunk(NamedTuple):
    # New-side lines (context and additions) as (line number in the head file,
    # text, was_added).  Context is kept so a declaration whose macro name sits
    # in context and whose default sits in an added line still parses.
    new_lines: List[Tuple[int, str, bool]]
    # Old-side lines (context and removals), for parsing the pre-PR declaration.
    old_lines: List[str]


class FileDiff(NamedTuple):
    path: str
    hunks: List[Hunk]


HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def parse_diff(diff_text: str, strip_prefix: str = "") -> List[FileDiff]:
    """Parse a unified diff into per-file hunks.

    `strip_prefix` (e.g. "projects/rccl/") is removed from each path so paths
    line up with the registry; files outside the prefix are dropped.
    """
    files: List[FileDiff] = []
    path: Optional[str] = None
    hunks: List[Hunk] = []
    hunk: Optional[Hunk] = None
    lineno = 0

    def flush_file() -> None:
        if path is not None and hunks:
            files.append(FileDiff(path, list(hunks)))

    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            flush_file()
            hunks, hunk = [], None
            target = raw[4:].strip().split("\t", 1)[0]
            if target == "/dev/null":
                path = None
                continue
            if target.startswith("b/"):
                target = target[2:]
            if strip_prefix:
                if not target.startswith(strip_prefix):
                    path = None
                    continue
                target = target[len(strip_prefix) :]
            path = target
        elif raw.startswith("@@"):
            header = HUNK_HEADER_RE.match(raw)
            if header is None or path is None:
                hunk = None
                continue
            lineno = int(header.group(1))
            hunk = Hunk([], [])
            hunks.append(hunk)
        elif hunk is None:
            continue
        elif raw.startswith("---") or raw.startswith("+++"):
            continue
        elif raw.startswith("+"):
            hunk.new_lines.append((lineno, raw[1:], True))
            lineno += 1
        elif raw.startswith("-"):
            hunk.old_lines.append(raw[1:])
        elif raw.startswith(" ") or raw == "":
            hunk.new_lines.append((lineno, raw[1:], False))
            hunk.old_lines.append(raw[1:])
            lineno += 1
    flush_file()
    return files


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect(
    file_diffs: Sequence[FileDiff], registry: Dict[str, Declaration]
) -> List[Detection]:
    detections: List[Detection] = []
    for fd in file_diffs:
        if not in_scan_scope(fd.path):
            continue
        detections.extend(_detect_declarations(fd))
        detections.extend(_detect_references(fd, registry))
    return _dedupe(detections)


def _detect_declarations(fd: FileDiff) -> List[Detection]:
    """New params and changed defaults, comparing each hunk's two sides."""
    detections: List[Detection] = []
    for hunk in fd.hunks:
        new_text = "\n".join(text for _lineno, text, _added in hunk.new_lines)
        old_by_env = {
            decl.env: decl
            for decl in parse_declarations("\n".join(hunk.old_lines), fd.path)
        }
        for decl in parse_declarations(new_text, fd.path):
            span = hunk.new_lines[decl.line - 1 : decl.end_line]
            touched = [entry for entry in span if entry[2]]
            if not touched:
                # The declaration is entirely context: this hunk changed
                # something else nearby.
                continue
            line = touched[0][0]
            old = old_by_env.get(decl.env)
            if old is None:
                detections.append(
                    Detection(
                        decl.env,
                        TRIGGER_NEW,
                        fd.path,
                        line,
                        "default `%s`" % decl.default,
                        new=decl.default,
                    )
                )
            elif old.default != decl.default:
                detections.append(
                    Detection(
                        decl.env,
                        TRIGGER_DEFAULT,
                        fd.path,
                        line,
                        "`%s` -> `%s`" % (old.default, decl.default),
                        old=old.default,
                        new=decl.default,
                    )
                )
    return detections


def _detect_references(
    fd: FileDiff, registry: Dict[str, Declaration]
) -> List[Detection]:
    detections: List[Detection] = []
    for hunk in fd.hunks:
        for lineno, text, added in hunk.new_lines:
            if not added:
                continue
            for env, detail in _references(text, registry):
                detections.append(
                    Detection(env, TRIGGER_REFERENCE, fd.path, lineno, detail)
                )
    return detections


def _references(text: str, registry: Dict[str, Declaration]) -> List[Tuple[str, str]]:
    """Env vars referenced by one added line, each with a readable detail."""
    found: List[Tuple[str, str]] = []
    seen: Set[str] = set()

    def add(env: str, detail: str) -> None:
        if env not in seen:
            seen.add(env)
            found.append((env, detail))

    # A param named in a comment reads nothing.
    stripped = text.lstrip()
    if stripped.startswith("//") or stripped.startswith("*"):
        return found
    # A declaration line is reported by _detect_declarations instead, with the
    # default value attached.
    if MACRO_CALL_RE.search(text):
        return found

    for accessor in ACCESSOR_RE.findall(text):
        decl = registry.get(accessor)
        # An accessor with no declaration in the tree is still reported, under
        # its own name, rather than silently dropped.
        add(decl.env if decl else accessor, "`%s`" % accessor)

    for func, literal in GETENV_RE.findall(text):
        if literal:
            add(literal, '`%s("%s")`' % (func, literal))
        else:
            add(DYNAMIC_ENV, "`%s()` with a non-literal name" % func)

    for literal in ENV_LITERAL_RE.findall(text):
        add(literal, '`"%s"`' % literal)

    return found


def _dedupe(detections: Sequence[Detection]) -> List[Detection]:
    """One detection per variable: the strongest trigger, at its first site.

    A variable that is both newly declared and referenced keeps the declaration,
    which is what the markdown entry needs to describe.
    """
    best: Dict[str, Detection] = {}
    for det in detections:
        current = best.get(det.env)
        if current is None or TRIGGER_RANK[det.trigger] < TRIGGER_RANK[current.trigger]:
            best[det.env] = det
    return sorted(best.values(), key=lambda d: (TRIGGER_RANK[d.trigger], d.env))


# ---------------------------------------------------------------------------
# Markdown coverage and exemptions
# ---------------------------------------------------------------------------


def _is_separator_row(cells: Sequence[str]) -> bool:
    return bool(cells) and all(cell and set(cell) <= set("-: ") for cell in cells)


def _is_unfilled(cell: str) -> bool:
    """True for a cell the author left as a placeholder rather than filling in.

    The report hands out rows pre-filled with `_describe…_` markers, so an
    untouched paste must not read as documentation.
    """
    text = re.sub(r"<!--.*?-->", "", cell, flags=re.DOTALL).strip()
    if not text:
        return True
    # The report hands out italicised placeholders (`_what it controls_`) and
    # real documentation is not written in italics, so emphasis wrapping the
    # whole cell means the author pasted the row and moved on.
    if re.fullmatch(r"[_*].*[_*]", text, flags=re.DOTALL):
        return True
    return PLACEHOLDER_RE.fullmatch(text.strip("`* ").strip()) is not None


def parse_md_entries(md_text: str) -> List[MdEntry]:
    """Parse the entry tables under each release heading.

    Only tables under an `## Unreleased` or `## RCCL <version>` heading count, so
    the usage examples in the file's header are not read as entries.  Columns are
    keyed by their header text, so the two tables (new variables, changed
    defaults) are told apart by the columns they carry rather than by position.
    """
    entries: List[MdEntry] = []
    in_section = False
    header: Optional[List[str]] = None
    previous: Optional[List[str]] = None

    for line in md_text.splitlines():
        if line.startswith("#"):
            # `###` subsections (the two tables) stay inside the release section;
            # only a `##` heading opens or closes one.
            if len(line) - len(line.lstrip("#")) <= 2:
                in_section = bool(MD_SECTION_RE.match(line))
            header = previous = None
            continue
        stripped = line.strip()
        if not in_section or not stripped.startswith("|"):
            header = previous = None
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if _is_separator_row(cells):
            # The row before a separator is the header row.
            header = [cell.lower() for cell in previous] if previous else None
            previous = None
            continue
        previous = cells
        if header is None or not cells or not cells[0]:
            continue

        fields = {name: value for name, value in zip(header, cells)}
        names = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", cells[0]))
        if names:
            entries.append(MdEntry(names, " | ".join(cells), fields))
    return entries


def coverage_gap(detection: Detection, entries: Sequence[MdEntry]) -> Optional[str]:
    """None when the variable is documented; otherwise why it is not.

    A newly added variable needs a row in the new-variables table with its
    support status, description and accepted values filled in -- a bare mention
    is not documentation.  A changed default needs the new value present in the
    row, so that a row written for an earlier release does not silently satisfy
    a fresh change to the same variable; defaults that are expressions
    (`1LL << 30`) fall back to a name-only match.
    """
    named = [entry for entry in entries if detection.env in entry.names]
    if not named:
        return NO_ENTRY

    if detection.trigger == TRIGGER_NEW:
        documented = [e for e in named if NEW_VAR_KEY_COLUMN in e.fields]
        if not documented:
            return (
                "listed only as a change; a new variable also needs a row in the "
                "New environment variables table"
            )
        best: Optional[str] = None
        for entry in documented:
            gap = _new_variable_gap(entry)
            if gap is None:
                return None
            best = gap if best is None else best
        return best

    if (
        detection.trigger == TRIGGER_DEFAULT
        and detection.new is not None
        and INTEGER_DEFAULT_RE.match(detection.new)
    ):
        pattern = r"(?<![\w-])%s\b" % re.escape(detection.new)
        if not any(re.search(pattern, entry.text) for entry in named):
            return "the entry does not mention the new default `%s`" % detection.new
    return None


def _new_variable_gap(entry: MdEntry) -> Optional[str]:
    """Check one new-variables row for unfilled or invalid cells."""
    unfilled = [
        NEW_VAR_COLUMN_LABELS[column]
        for column in NEW_VAR_REQUIRED_COLUMNS
        if _is_unfilled(entry.fields.get(column, ""))
    ]
    if unfilled:
        return "the entry leaves %s empty" % ", ".join(unfilled)
    supported = entry.fields.get(NEW_VAR_KEY_COLUMN, "").strip().strip("`*_ ")
    if supported.lower() not in SUPPORTED_VALUES:
        return "Supported is `%s`; use one of %s" % (
            supported,
            ", ".join("`%s`" % value for value in SUPPORTED_DISPLAY),
        )
    return None


def is_covered(detection: Detection, entries: Sequence[MdEntry]) -> bool:
    """True when the variable is fully documented."""
    return coverage_gap(detection, entries) is None


def parse_exemptions(pr_body: str) -> Tuple[Dict[str, str], List[str]]:
    """Parse BEHAVIOR-CHANGE-EXEMPT lines into {variable: reason}, plus errors.

    A token with no reason is an error rather than a silent pass: the point of
    the override is that it is recorded.
    """
    exemptions: Dict[str, str] = {}
    errors: List[str] = []
    if not pr_body:
        return exemptions, errors

    for match in EXEMPT_LINE_RE.finditer(pr_body):
        rest = match.group("rest").strip()
        if not rest:
            errors.append("BEHAVIOR-CHANGE-EXEMPT line names no variable.")
            continue
        parts = EXEMPT_SPLIT_RE.split(rest, maxsplit=1)
        names = [name for name in re.split(r"[,\s]+", parts[0]) if name]
        reason = parts[1].strip() if len(parts) > 1 else ""
        if not names:
            errors.append("BEHAVIOR-CHANGE-EXEMPT line names no variable.")
            continue
        if not reason:
            errors.append(
                "BEHAVIOR-CHANGE-EXEMPT for %s has no reason. Use "
                "`BEHAVIOR-CHANGE-EXEMPT: %s — why this is not a behavior change`."
                % (", ".join(names), names[0])
            )
            continue
        for name in names:
            exemptions[name] = reason
    return exemptions, errors


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def suggested_row(detection: Detection) -> str:
    """A pre-filled row for the table this detection belongs in."""
    if detection.trigger == TRIGGER_NEW:
        # Variable | Supported | Description | Accepted values | Default | Reason
        return (
            "| `%s` | Supported | _what it controls_ | _e.g. `0` (off), `1` (on)_ | `%s` | _why_ |"
            % (
                detection.env,
                detection.new,
            )
        )
    if detection.trigger == TRIGGER_DEFAULT:
        change = "`%s` default changed `%s` -> `%s`" % (
            detection.env,
            detection.old,
            detection.new,
        )
    else:
        change = "`%s` newly referenced in `%s`" % (detection.env, detection.path)
    return "| %s | _why_ |" % change


def build_report(
    missing: Sequence[Detection],
    applied: Dict[str, Tuple[str, str]],
    errors: Sequence[str],
    md_path: str,
    gaps: Optional[Dict[str, str]] = None,
) -> str:
    gaps = gaps or {}
    out: List[str] = []
    if missing:
        out.append(
            "### RCCL default-behavior gate — %d variable(s) need an entry\n"
            % len(missing)
        )
        out.append("| Variable | Trigger | Where | Detail | What is missing |")
        out.append("|---|---|---|---|---|")
        for det in missing:
            out.append(
                "| `%s` | %s | `%s:%d` | %s | %s |"
                % (
                    det.env,
                    det.trigger,
                    det.path,
                    det.line,
                    det.detail,
                    gaps.get(det.env, NO_ENTRY),
                )
            )
        out.append("")

        new_params = [d for d in missing if d.trigger == TRIGGER_NEW]
        others = [d for d in missing if d.trigger != TRIGGER_NEW]

        if new_params:
            out.append(
                "A new variable needs a documented row. Paste into `%s` under "
                "`## Unreleased` -> `### New environment variables`, then fill in "
                "the italicised cells:\n" % md_path
            )
            out.append("```markdown")
            out.append(NEW_VAR_HEADER)
            out.append(NEW_VAR_SEPARATOR)
            out.extend(suggested_row(det) for det in new_params)
            out.append("```")
            out.append(
                "\n`Supported` must be %s — say plainly whether users may rely on "
                "this knob.\n"
                % " or ".join("`%s`" % value for value in SUPPORTED_DISPLAY)
            )

        if others:
            out.append(
                "Paste into `%s` under `## Unreleased` -> "
                "`### Changed defaults and other behavior changes`:\n" % md_path
            )
            out.append("```markdown")
            out.extend(suggested_row(det) for det in others)
            out.append("```")

        reference_only = [d.env for d in missing if d.trigger == TRIGGER_REFERENCE]
        if reference_only:
            out.append(
                "\nIf a detection is only a moved or refactored call site and nothing "
                "actually changed, put this in the PR description instead:\n"
            )
            out.append("```")
            out.append(
                "BEHAVIOR-CHANGE-EXEMPT: %s — refactor only, no default or code-path change"
                % ", ".join(reference_only)
            )
            out.append("```")
    else:
        out.append("### RCCL default-behavior gate — pass\n")

    if errors:
        out.append("\n**Override errors**\n")
        out.extend("* %s" % err for err in errors)

    if applied:
        out.append("\n**Overrides applied** — recorded here as part of the run\n")
        out.append("| Variable | Trigger | Reason |")
        out.append("|---|---|---|")
        for env, (trigger, reason) in sorted(applied.items()):
            flag = (
                ""
                if trigger == TRIGGER_REFERENCE
                else " **(this is a behavior change)**"
            )
            out.append("| `%s` | %s | %s%s |" % (env, trigger, reason, flag))

    return "\n".join(out) + "\n"


def emit_annotations(
    missing: Sequence[Detection],
    path_prefix: str,
    gaps: Optional[Dict[str, str]] = None,
) -> None:
    """GitHub workflow commands, so each finding lands inline on the diff."""
    gaps = gaps or {}
    for det in missing:
        if det.trigger == TRIGGER_DEFAULT:
            what = "%s default changed %s -> %s" % (det.env, det.old, det.new)
        elif det.trigger == TRIGGER_NEW:
            what = "%s is new (default %s)" % (det.env, det.new)
        else:
            what = "%s (%s)" % (det.env, det.trigger)
        message = (
            "%s: %s. Document it in %s, or add a BEHAVIOR-CHANGE-EXEMPT line to "
            "the PR description." % (what, gaps.get(det.env, NO_ENTRY), DEFAULT_MD_PATH)
        )
        print(
            "::error file=%s%s,line=%d::%s"
            % (path_prefix, det.path, det.line, message.replace("\n", " "))
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def read_diff(args: argparse.Namespace) -> str:
    if args.diff == "-":
        return sys.stdin.read()
    if args.diff:
        with open(args.diff, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    merge_base = subprocess.run(
        ["git", "merge-base", args.base, "HEAD"],
        cwd=args.root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # --relative emits paths relative to args.root, matching the registry.
    return subprocess.run(
        ["git", "diff", "--relative", merge_base, "--", "."],
        cwd=args.root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def main(argv: Optional[Sequence[str]] = None) -> int:
    rccl_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", default=rccl_root, help="RCCL tree root (default: %(default)s)"
    )
    parser.add_argument(
        "--diff", help="unified diff file, or - for stdin. Omit to use --base."
    )
    parser.add_argument(
        "--base",
        default="develop",
        help="branch to diff against when --diff is omitted (default: %(default)s)",
    )
    parser.add_argument("--pr-body", help="file holding the PR description")
    parser.add_argument(
        "--md",
        help="behavior-changes markdown file (default: <root>/%s)" % DEFAULT_MD_PATH,
    )
    parser.add_argument(
        "--strip-prefix",
        default="",
        help="path prefix to strip from diff paths, e.g. projects/rccl/ for a monorepo diff",
    )
    parser.add_argument(
        "--format",
        choices=("text", "github"),
        default="text",
        help="github adds ::error annotations and appends to $GITHUB_STEP_SUMMARY",
    )
    args = parser.parse_args(argv)

    md_path = args.md or os.path.join(args.root, DEFAULT_MD_PATH)
    md_text = _read_optional(md_path)

    if args.pr_body:
        pr_body = _read_optional(args.pr_body)
    else:
        pr_body = os.environ.get("PR_BODY", "")

    registry = build_registry(args.root)
    detections = detect(parse_diff(read_diff(args), args.strip_prefix), registry)
    entries = parse_md_entries(md_text)
    exemptions, errors = parse_exemptions(pr_body)

    missing: List[Detection] = []
    applied: Dict[str, Tuple[str, str]] = {}
    gaps: Dict[str, str] = {}
    for det in detections:
        gap = coverage_gap(det, entries)
        if gap is None:
            continue
        if det.env in exemptions:
            applied[det.env] = (det.trigger, exemptions[det.env])
            continue
        gaps[det.env] = gap
        missing.append(det)

    display_path = os.path.relpath(md_path, args.root)
    if display_path.startswith(".."):  # --md pointed outside the tree
        display_path = md_path
    report = build_report(missing, applied, errors, display_path, gaps)
    if args.format == "github":
        emit_annotations(missing, args.strip_prefix, gaps)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as handle:
                handle.write(report)
    print(report)
    return 1 if (missing or errors) else 0


def _read_optional(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


if __name__ == "__main__":
    sys.exit(main())
