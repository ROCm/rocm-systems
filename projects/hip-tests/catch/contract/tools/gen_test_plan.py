#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Generate the contract-tier test plan (TEST_PLAN.md) from the test sources.

The test plan is an inventory of *what each test case asserts*, grouped by tier
(the top-level directory under ``catch/``) and domain (its subdirectory). It is
generated from the sources so it cannot drift: every ``HIP_TEST_CASE`` is listed
with the API it pins and a one-line invariant, read from a structured
``// @asserts:`` doc-comment placed directly above the case:

    // @asserts: hipDeviceReset - reset discards device state, device stays usable
    HIP_TEST_CASE(Contract_DeviceReset_DiscardsStateAndLeavesDeviceUsable) {

The marker is ``@asserts:``; the text before the first `` - `` is the API (or
family) the case exercises, and the remainder is the human-readable invariant.
Cases without the tag are still listed (intent derived from the case name) and
counted in a "missing @asserts tag" summary so the gap is visible.

This is a name/intent-level inventory to support cross-tier de-duplication as the
wider ``catch/*`` suite is organized into tiers (contract, unit, integration,
system, performance, stress). It is pure static analysis - no ROCm, GPU, or build
required - so it doubles as a fast CI staleness gate (``--check``).

Only the ``contract`` tier is wired today; pass ``--tiers`` to include more as they
are organized.
"""

import argparse
import json
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONTRACT_DIR = os.path.dirname(_THIS_DIR)                       # catch/contract
CATCH_DIR = os.path.dirname(CONTRACT_DIR)                       # catch
TEST_PLAN_PATH = os.path.join(CONTRACT_DIR, "TEST_PLAN.md")

# HIP_TEST_CASE(Name) - tolerate whitespace/newlines between '(' and the name so
# a case whose macro is split across lines is still matched.
_CASE_RE = re.compile(r"HIP_TEST_CASE\s*\(\s*([A-Za-z0-9_]+)\s*\)")
_ASSERTS_RE = re.compile(r"//\s*@asserts:\s*(.*\S)\s*$")

# Separator between the API and the invariant in an @asserts tag: a spaced hyphen
# (keyboard-friendly) or an em dash (tolerated for older tags).
_SEP_RE = re.compile(r"\s+[-—]\s+")


def _iter_test_files(tier, catch_dir=CATCH_DIR):
    """Yield (path) of test_hip_*.cc sources under catch/<tier>/."""
    root = os.path.join(catch_dir, tier)
    if not os.path.isdir(root):
        return
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if name.startswith("test_hip_") and name.endswith(".cc"):
                yield os.path.join(dirpath, name)


# Matches the start of a HIP_TEST_CASE macro at the beginning of a line. The name
# may be on the same line or (when the macro is split) empty here and recovered
# from the following lines.
_CASE_START_RE = re.compile(r"\s*HIP_TEST_CASE\s*\(")


def _line_starts_case(line):
    """Return '' if the line opens a HIP_TEST_CASE macro, else None."""
    return "" if _CASE_START_RE.match(line) else None


def _decamel(name):
    """Turn a Contract_Domain_BehaviorInCamel case name into readable text."""
    # Drop a leading Contract_ / tier prefix, split remaining _ groups, and space
    # out CamelCase within each group.
    parts = name.split("_")
    if parts and parts[0].lower() in ("contract", "unit", "integration",
                                      "system", "performance", "stress"):
        parts = parts[1:]
    spaced = []
    for p in parts:
        spaced.append(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", p))
    return " ".join(spaced).strip()


def _extract_asserts_block(lines, case_line_idx):
    """Return the @asserts text from the contiguous // block above a case, or None."""
    j = case_line_idx - 1
    # Skip blank lines directly above the macro.
    while j >= 0 and not lines[j].strip():
        j -= 1
    # Collect the contiguous comment block.
    block = []
    while j >= 0 and lines[j].strip().startswith("//"):
        block.append(lines[j])
        j -= 1
    block.reverse()
    for line in block:
        m = _ASSERTS_RE.search(line)
        if m:
            return m.group(1).strip()
    return None


def _split_api_invariant(asserts_text):
    """Split an @asserts value into (api, invariant)."""
    parts = _SEP_RE.split(asserts_text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return asserts_text.strip(), ""


def collect_cases(tiers, catch_dir=CATCH_DIR):
    """Return a list of case dicts for the given tiers."""
    cases = []
    for tier in tiers:
        for path in _iter_test_files(tier, catch_dir):
            rel = os.path.relpath(path, catch_dir)
            segs = rel.split(os.sep)
            domain = segs[1] if len(segs) > 2 else "(root)"
            text = open(path, encoding="utf-8", errors="replace").read()
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                if _line_starts_case(line) is None:
                    continue
                # Recover the case name from this line plus the next few, so a macro
                # split across lines (HIP_TEST_CASE(\n    Name) {) still resolves.
                joined = " ".join(lines[idx:idx + 4])
                m = _CASE_RE.search(joined)
                name = m.group(1) if m else None
                if not name:
                    continue
                asserts = _extract_asserts_block(lines, idx)
                if asserts:
                    api, invariant = _split_api_invariant(asserts)
                    tagged = True
                else:
                    api, invariant = "", _decamel(name)
                    tagged = False
                cases.append({
                    "tier": tier,
                    "domain": domain,
                    "case": name,
                    "api": api,
                    "invariant": invariant,
                    "tagged": tagged,
                    "file": rel,
                })
    cases.sort(key=lambda c: (c["tier"], c["domain"], c["case"]))
    return cases


def _md_escape(text):
    return text.replace("|", "\\|")


def render_markdown(cases, tiers):
    out = []
    out.append("<!--")
    out.append("Copyright (c) Advanced Micro Devices, Inc., or its affiliates.")
    out.append("SPDX-License-Identifier: MIT")
    out.append("-->")
    out.append("")
    out.append("# HIP test plan")
    out.append("")
    out.append("Generated by `catch/contract/tools/gen_test_plan.py` - **do not edit by "
               "hand**. Regenerate after adding, renaming, or removing a test case; the "
               "`hip-contract-coverage` CI check fails if this file is out of date.")
    out.append("")
    out.append("Each row is one `HIP_TEST_CASE`. The API and invariant come from the "
               "`// @asserts: <API> - <invariant>` tag above the case; rows without a tag "
               "show intent derived from the case name and are flagged below. Tier is the "
               "top-level directory under `catch/`.")
    out.append("")

    total = len(cases)
    tagged = sum(1 for c in cases if c["tagged"])
    out.append("## Summary")
    out.append("")
    out.append("| Tier | Cases | Tagged | Missing `@asserts` |")
    out.append("|---|---:|---:|---:|")
    for tier in tiers:
        tcases = [c for c in cases if c["tier"] == tier]
        ttag = sum(1 for c in tcases if c["tagged"])
        out.append("| `{}` | {} | {} | {} |".format(
            tier, len(tcases), ttag, len(tcases) - ttag))
    out.append("| **total** | **{}** | **{}** | **{}** |".format(
        total, tagged, total - tagged))
    out.append("")

    # Per tier -> domain tables.
    for tier in tiers:
        tcases = [c for c in cases if c["tier"] == tier]
        if not tcases:
            continue
        out.append("## Tier: `{}`".format(tier))
        out.append("")
        domains = []
        for c in tcases:
            if c["domain"] not in domains:
                domains.append(c["domain"])
        for domain in domains:
            dcases = [c for c in tcases if c["domain"] == domain]
            out.append("### `{}` ({} case{})".format(
                domain, len(dcases), "" if len(dcases) == 1 else "s"))
            out.append("")
            out.append("| Case | API | Asserts |")
            out.append("|---|---|---|")
            for c in dcases:
                api = _md_escape(c["api"]) if c["api"] else "_(from name)_"
                inv = _md_escape(c["invariant"]) or "_(none)_"
                out.append("| `{}` | {} | {} |".format(
                    _md_escape(c["case"]), api, inv))
            out.append("")

    missing = [c for c in cases if not c["tagged"]]
    if missing:
        out.append("## Cases missing an `@asserts` tag")
        out.append("")
        out.append("These cases have no `// @asserts:` line; their intent above is derived "
                   "from the case name. Add a tag (see `AUTHORING.md`) to make the plan "
                   "authoritative.")
        out.append("")
        for c in missing:
            out.append("- `{}` ({}/{})".format(c["case"], c["tier"], c["domain"]))
        out.append("")

    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--tiers", default="contract",
                        help="comma-separated tier directories under catch/ to include "
                             "(default: contract)")
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory and fail if TEST_PLAN.md is stale (CI)")
    parser.add_argument("--json", action="store_true",
                        help="emit the raw case inventory as JSON")
    parser.add_argument("--out", default=TEST_PLAN_PATH,
                        help="output path (default: catch/contract/TEST_PLAN.md)")
    parser.add_argument("--catch-dir", default=CATCH_DIR,
                        help="path to the catch/ directory (default: repo-relative)")
    args = parser.parse_args(argv)

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    cases = collect_cases(tiers, args.catch_dir)

    if args.json:
        print(json.dumps(cases, indent=2, sort_keys=True))
        return 0

    rendered = render_markdown(cases, tiers)

    if args.check:
        if not os.path.exists(args.out):
            sys.stderr.write("error: {} does not exist; run gen_test_plan.py to "
                             "create it.\n".format(args.out))
            return 1
        current = open(args.out, encoding="utf-8").read()
        if current != rendered:
            sys.stderr.write(
                "error: {} is out of date. Regenerate it with:\n"
                "  python3 projects/hip-tests/catch/contract/tools/gen_test_plan.py\n"
                .format(os.path.relpath(args.out)))
            return 1
        print("TEST_PLAN.md is up to date ({} cases).".format(len(cases)))
        return 0

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(rendered)
    tagged = sum(1 for c in cases if c["tagged"])
    print("Wrote {} ({} cases, {} tagged, {} missing @asserts).".format(
        os.path.relpath(args.out), len(cases), tagged, len(cases) - tagged))
    return 0


if __name__ == "__main__":
    sys.exit(main())
