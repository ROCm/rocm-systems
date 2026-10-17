#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compare CTest and GoogleTest JUnit XML from hardware and rocJITsu runs."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def load_cases(directory: Path, prefix: str) -> dict[str, dict]:
    cases: dict[str, dict] = {}
    for path in sorted(directory.glob(f"{prefix}*.xml")):
        root = ET.parse(path).getroot()
        for case in root.iter("testcase"):
            name = case.attrib.get("name", "")
            classname = case.attrib.get("classname", "")
            suite = path.stem[len(prefix) :]
            key = f"{suite}:{classname}.{name}"
            children = list(case)
            status = "failed" if any(node.tag in {"failure", "error"} for node in children) else (
                "skipped" if any(node.tag == "skipped" for node in children) else "passed"
            )
            cases[key] = {"status": status, "seconds": float(case.attrib.get("time", 0.0))}
    return cases


def main() -> int:
    emu_dir, hw_dir = map(Path, sys.argv[1:3])
    emu = load_cases(emu_dir, "emu-")
    hardware = load_cases(hw_dir, "hw-")
    keys = sorted(set(emu) | set(hardware))
    missing_emu = sorted(set(hardware) - set(emu))
    missing_hardware = sorted(set(emu) - set(hardware))
    status_mismatch = [key for key in keys if key in emu and key in hardware and emu[key]["status"] != hardware[key]["status"]]
    Path("comparison").mkdir(parents=True, exist_ok=True)
    summary = {
        "emulator_test_cases": len(emu),
        "hardware_test_cases": len(hardware),
        "matched_test_cases": len(set(emu) & set(hardware)),
        "missing_from_emulator": missing_emu,
        "missing_from_hardware": missing_hardware,
        "status_mismatches": status_mismatch,
        "emulator_failures": [key for key, case in emu.items() if case["status"] == "failed"],
        "hardware_failures": [key for key, case in hardware.items() if case["status"] == "failed"],
    }
    Path("comparison/summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# gfx942 hardware and rocJITsu quick comparison",
        "",
        f"- Hardware cases: {len(hardware)}",
        f"- rocJITsu cases: {len(emu)}",
        f"- Matched cases: {summary['matched_test_cases']}",
        f"- Missing from hardware: {len(missing_hardware)}",
        f"- Missing from rocJITsu: {len(missing_emu)}",
        f"- Status mismatches: {len(status_mismatch)}",
        "",
    ]
    if missing_hardware:
        lines += ["## Missing from hardware", ""] + [f"- `{key}`" for key in missing_hardware] + [""]
    if missing_emu:
        lines += ["## Missing from rocJITsu", ""] + [f"- `{key}`" for key in missing_emu] + [""]
    if status_mismatch:
        lines += ["## Status mismatches", ""] + [f"- `{key}`: hardware={hardware[key]['status']}, rocJITsu={emu[key]['status']}" for key in status_mismatch] + [""]
    Path("comparison/summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return int(not emu or not hardware or bool(missing_emu or missing_hardware or status_mismatch))


if __name__ == "__main__":
    raise SystemExit(main())
