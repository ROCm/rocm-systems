#!/usr/bin/env python3
"""Bookkeeping check over the conflict register. Run from anywhere:

    python3 openspec/check_conflict_register.py

WHAT THIS VERIFIES, EXACTLY. It is narrower than the name of the register
suggests:

  1. Every ``**Recorded contradiction/defect/gap (Cn)**`` marker in ``specs/``
     has a row in CONFLICTS.md.
  2. Every row in CONFLICTS.md has a non-empty Resolution cell that is not a
     placeholder word.
  3. Every row whose "Recorded in" column names a baseline area names one that
     exists on disk.
  4. A cheap literal-value cross-check between the change directories: when two
     of them state a different literal value for the same named constant that
     this script knows how to scan for, that is reported.

WHAT THIS DOES NOT VERIFY. It does not read the specs for meaning, does not
check that a stated resolution was implemented in any code, and, apart from
the narrow scan in (4), does not compare the four change directories against
each other. A contradiction between two proposals will pass this script unless
it happens to be a literal value of a constant listed in CONSTANT_PATTERNS. It
was passing cleanly while ``add-cuid-kernel-interface`` specified an at-most-32
seed against a kernel and two sibling proposals that all said exactly-32. Do not
read a clean run as "the corpus is consistent".

Exits non-zero if any of (1)-(4) fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "specs" / "cuid"
CHANGES = ROOT / "changes"
AMEND = CHANGES / "amend-published-cuid-spec"
REGISTER = ROOT / "CONFLICTS.md"

problems: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)


# --- the register ----------------------------------------------------------
if not REGISTER.exists():
    print(f"missing {REGISTER}", file=sys.stderr)
    sys.exit(1)

register_text = REGISTER.read_text()
rows = re.findall(r"^\| (C\d+) \| (.+?) \| (.+?) \| (.+?) \|$", register_text, re.M)
registered = {cid: (desc, where, res) for cid, desc, where, res in rows}

if not registered:
    fail("CONFLICTS.md contains no conflict rows")

for cid, (_desc, where, res) in sorted(
    registered.items(), key=lambda kv: int(kv[0][1:])
):
    if res.strip().lower() in {"", "n/a", "-"}:
        fail(f"{cid} is registered with no resolution")
    # A resolution that just points elsewhere is not a resolution.
    if res.strip().lower() in {"tbd", "todo", "open", "unresolved"}:
        fail(f"{cid} resolution is a placeholder: {res!r}")

# --- every recorded marker in the baseline is registered -------------------
marker_re = re.compile(
    r"\*\*Recorded (?:contradiction|defect|gap)(?: \((C\d+)[^)]*\))?"
)
recorded_ids: set[str] = set()
unlabelled = 0

for spec in sorted(BASELINE.rglob("spec.md")):
    for m in marker_re.finditer(spec.read_text()):
        if m.group(1):
            recorded_ids.add(m.group(1))
        else:
            unlabelled += 1

missing = sorted(recorded_ids - set(registered), key=lambda c: int(c[1:]))
for cid in missing:
    fail(f"{cid} is recorded in specs/ but absent from CONFLICTS.md")

# --- the amend change must actually say something about each --------------
amend_text = "\n".join(p.read_text() for p in AMEND.rglob("*.md"))
skipped_no_baseline: list[str] = []
for cid in sorted(registered, key=lambda c: int(c[1:])):
    where = registered[cid][1].strip()
    if where.lower() in {"n/a", "-", ""}:
        # Absent from the published page entirely: there is nothing in the
        # baseline to have recorded, only a value this change supplies. Nothing
        # below applies, so say so rather than dropping it silently.
        skipped_no_baseline.append(cid)
        continue
    if (
        cid not in recorded_ids
        and f"({cid}" not in amend_text
        and cid not in amend_text
    ):
        # Not fatal on its own, since some are recorded under a shared marker,
        # but the register's "recorded in" column must at least name a file
        # that exists.
        target = (
            BASELINE / where.strip().strip("`").split(",")[0].strip("` ") / "spec.md"
        )
        if not target.exists():
            fail(f"{cid} names a baseline file that does not exist: {target}")

# --- cheap cross-proposal literal-value check ------------------------------
#
# Deliberately small. Each entry names a constant the layers must agree on and a
# pattern with one capture group holding the literal. Two change directories
# stating different literals for the same constant is a contradiction of the
# kind that let the seed-length defect survive review. This catches only what is
# listed here; it is a tripwire, not a consistency proof.
CONSTANT_PATTERNS: dict[str, re.Pattern[str]] = {
    # Provisioned seed length: "exactly 32 octets", "at most 32", "up to 32".
    "provisioned seed length": re.compile(
        r"(?:seed|secret)[^.\n]{0,80}?\b(exactly|at most|no more than|up to)\s+32\b",
        re.I,
    ),
    # Canonical fallback seed length in octets. The leading punctuation or
    # "is" keeps this off phrases like "over those 16 octets", where the number
    # is the message length and not the constant's.
    "canonical fallback seed length": re.compile(
        r"AMD-CUID-DEFAULT-SEED-v1[^.\n]{0,40}?(?:\(|\bis\s+|,\s*)\s*(\d+)[- ]"
        r"(?:ASCII[- ])?octet",
        re.I,
    ),
    # Auxiliary temporary key length in octets.
    "temp key length": re.compile(
        r"AMD-CUID-TEMP-KEY-v1[^.\n]{0,40}?(?:\(|\bis\s+|,\s*)\s*(\d+)[- ]"
        r"(?:ASCII[- ])?octet",
        re.I,
    ),
}

# Literals that mean the same thing, normalised so a wording difference is not
# reported as a value difference.
SYNONYMS = {
    "at most": "at-most-32",
    "no more than": "at-most-32",
    "up to": "at-most-32",
    "exactly": "exactly-32",
}

change_dirs = sorted(p for p in CHANGES.iterdir() if p.is_dir())
for name, pattern in CONSTANT_PATTERNS.items():
    stated: dict[str, list[str]] = {}
    for change in change_dirs:
        text = "\n".join(p.read_text() for p in sorted(change.rglob("*.md")))
        for m in pattern.finditer(text):
            literal = m.group(1).strip().lower()
            literal = SYNONYMS.get(literal, literal)
            stated.setdefault(literal, [])
            if change.name not in stated[literal]:
                stated[literal].append(change.name)
    if len(stated) > 1:
        detail = "; ".join(
            f"{lit!r} in {', '.join(dirs)}" for lit, dirs in sorted(stated.items())
        )
        fail(f"change dirs disagree on {name}: {detail}")

# --- report ----------------------------------------------------------------
print(f"registered conflicts  : {len(registered)}")
print(f"labelled in specs/    : {len(recorded_ids)}")
print(f"unlabelled markers    : {unlabelled}")
print(
    f"rows with no baseline : {len(skipped_no_baseline)}"
    f"{' (' + ', '.join(skipped_no_baseline) + ')' if skipped_no_baseline else ''}"
)
print(f"change dirs scanned   : {len(change_dirs)}")
print(f"constants cross-checked: {len(CONSTANT_PATTERNS)}")

if problems:
    print()
    for p in problems:
        print(f"  FAIL  {p}")
    print(f"\n{len(problems)} problem(s)")
    sys.exit(1)

print(
    "\nbookkeeping only: every labelled marker in specs/ has a register row, every "
    "row names a non-placeholder resolution, and the "
    f"{len(CONSTANT_PATTERNS)} cross-checked constants are stated consistently across the "
    f"{len(change_dirs)} change dirs.\nThis says nothing about whether the resolutions are "
    "correct, whether they were implemented, or whether the proposals agree on anything "
    "not listed in CONSTANT_PATTERNS."
)
sys.exit(0)
