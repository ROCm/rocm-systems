#!/usr/bin/env python3
"""Build a platform/family table from Catch2 JUnit files.

CI names the files unit-<os>.xml and integration-<family>.xml. Catch2 also
emits one <testcase> per SECTION, so counts here use only top-level cases
(names without '/'), matching run_catch2.py.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Display order matches .github/workflows/hrr-ci.yml.
COLUMNS = (
    ("unit-ubuntu-24.04.xml", "Unit", "Linux"),
    ("unit-windows-2022.xml", "Unit", "Windows"),
    ("integration-gfx90a.xml", "Integration", "gfx90a"),
    ("integration-gfx94x.xml", "Integration", "gfx94x"),
    ("integration-gfx950.xml", "Integration", "gfx950"),
    ("integration-gfx103x.xml", "Integration", "gfx103x"),
    ("integration-gfx110x.xml", "Integration", "gfx110x"),
    ("integration-gfx120x.xml", "Integration", "gfx120x"),
    ("integration-gfx1151.xml", "Integration", "gfx1151"),
    ("integration-gfx1150.xml", "Integration", "gfx1150"),
    ("integration-gfx1153.xml", "Integration", "gfx1153"),
)

MAX_FAILED_NAMES = 8


def case_result(case: ET.Element) -> str:
    if case.find("failure") is not None or case.find("error") is not None:
        return "FAIL"
    if case.find("skipped") is not None:
        return "SKIP"
    return "PASS"


def summarize_xml(path: Path) -> dict[str, object]:
    cases = [
        case
        for case in ET.parse(path).iterfind(".//testcase")
        if "/" not in case.attrib.get("name", "/")
    ]
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    failed: list[str] = []
    for case in cases:
        result = case_result(case)
        counts[result] += 1
        if result == "FAIL":
            failed.append(case.attrib["name"])
    return {"counts": counts, "failed": failed, "total": len(cases)}


def cell(value: object) -> str:
    return "—" if value is None else str(value)


def failed_cell(names: list[str]) -> str:
    if not names:
        return ""
    shown = names[:MAX_FAILED_NAMES]
    text = ", ".join(f"`{name}`" for name in shown)
    extra = len(names) - len(shown)
    if extra:
        text += f", … +{extra}"
    return text


def render(directory: Path) -> str:
    rows: list[tuple[str, str, dict[str, object] | None]] = []
    for filename, suite, platform in COLUMNS:
        path = directory / filename
        if path.is_file():
            rows.append((suite, platform, summarize_xml(path)))
        else:
            rows.append((suite, platform, None))

    extra = sorted(
        path.name
        for path in directory.glob("*.xml")
        if path.name not in {filename for filename, _, _ in COLUMNS}
    )
    for filename in extra:
        rows.append(("Other", filename, summarize_xml(directory / filename)))

    lines = [
        "## HRR results by platform",
        "",
        "| Suite | Platform / family | Pass | Fail | Skip | Failed tests |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for suite, platform, data in rows:
        if data is None:
            lines.append(f"| {suite} | {platform} | — | — | — | _no results_ |")
            continue
        counts = data["counts"]
        lines.append(
            "| {suite} | {platform} | {passed} | {failed} | {skipped} | {names} |".format(
                suite=suite,
                platform=platform,
                passed=counts["PASS"],
                failed=counts["FAIL"],
                skipped=counts["SKIP"],
                names=failed_cell(data["failed"]),
            )
        )

    lines.extend(
        [
            "",
            "Counts are top-level Catch2 cases. Missing XML means that job did not upload results (cancelled or not started).",
            "",
            "### Counts grid",
            "",
        ]
    )

    headers = ["", *(platform for _, platform, _ in rows)]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| --- |" + " ---: |" * (len(headers) - 1))
    for label, key in (("Pass", "PASS"), ("Fail", "FAIL"), ("Skip", "SKIP")):
        values = []
        for _, _, data in rows:
            values.append(
                cell(None if data is None else data["counts"][key])
            )
        lines.append("| " + " | ".join([label, *values]) + " |")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("."))
    parser.add_argument(
        "--summary",
        type=Path,
        help="Append markdown to this file (GITHUB_STEP_SUMMARY).",
    )
    args = parser.parse_args()
    if not args.dir.is_dir():
        print(f"not a directory: {args.dir}", file=sys.stderr)
        return 1
    markdown = render(args.dir)
    sys.stdout.write(markdown)
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as out:
            out.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
