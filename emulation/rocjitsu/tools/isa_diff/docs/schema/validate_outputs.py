#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Validate rj_isa_diff JSON outputs against rj_isa_diff.v1.schema.json.

Two modes:
  validate_outputs.py FILE_OR_DIR...            validate existing outputs
  validate_outputs.py --tool PATH [--keep]      drive the CLI over built-in fixtures
                                                (analyze, test, cache, manual region,
                                                register map, diagnostics, inject,
                                                replay) and validate everything it writes

Document kinds are detected from content: case.json, results.json, counterexample
bundles and per-case JSONL logs (one schema per line type). Exit 0 when everything
validates, 1 on a validation or CLI failure, 77 when the jsonschema module is missing
(ctest SKIP_RETURN_CODE).
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

try:
    import jsonschema
except ImportError:
    print("jsonschema is not installed; skipping", file=sys.stderr)
    sys.exit(77)

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(HERE, "rj_isa_diff.v1.schema.json")

# gfx950 fixtures (words assembled by llvm-mc; see tests/isa_diff/fixtures/fixtures.asm).
FIXTURES = {
    "shift_old": [
        0x80040302,
        0x82050100,
        0x85060504,
        0x8E888204,
        0xBF078006,
        0x85070908,
        0xBF810000,
    ],
    "shift_new": [
        0x80040302,
        0x82050100,
        0x85060504,
        0x8E888104,
        0x8E888108,
        0xBF078006,
        0x85070908,
        0xBF810000,
    ],
    "constraint_old": [
        0x8604FF04,
        0x000000FF,
        0x8F058404,
        0x80060405,
        0xBF068006,
        0xBF810000,
    ],
    "constraint_new": [
        0x8604FF04,
        0x000000FF,
        0x9285FF04,
        0x00040004,
        0x80060405,
        0xBF068006,
        0xBF810000,
    ],
    "twohunk_old": [
        0x80048102,
        0xBE890087,
        0x80048104,
        0xBF078004,
        0x85058009,
        0xBF068005,
        0xBF810000,
    ],
    "twohunk_new": [
        0x80048202,
        0xBE890087,
        0x80048004,
        0xBF078004,
        0x85058009,
        0xBF068005,
        0xBF810000,
    ],
    "rename_old": [
        0xBE840002,
        0xBF068003,
        0xBF850001,
        0x80048104,
        0xBF800000,
        0x80060504,
        0xBF078006,
        0xBF810000,
    ],
    "rename_new": [
        0xBE880002,
        0xBF068003,
        0xBF850001,
        0x80088108,
        0xBF800000,
        0x80060508,
        0xBF078006,
        0xBF810000,
    ],
    "valu_old": [1745093376, 604374146, 3698360320, 772, 3212902400],
    "valu_new": [1745092865, 604374145, 604374657, 3698360320, 772, 3212902400],
    "valuwrong_new": [1745092865, 604374147, 3698360320, 772, 3212902400],
    "valuclobber_old": [2114323202, 3698360320, 772, 3212902400],
    "valuclobber_new": [
        3196715390,
        3204317633,
        2114323202,
        3204317450,
        3698360320,
        772,
        3212902400,
    ],
}


class Validator:
    def __init__(self):
        with open(SCHEMA_PATH) as f:
            self.root = json.load(f)
        jsonschema.Draft202012Validator.check_schema(self.root)
        self.validators = {}
        self.checked = {}
        self.failures = []

    def _validator(self, kind):
        if kind not in self.validators:
            schema = {
                "$schema": self.root["$schema"],
                "$defs": self.root["$defs"],
                "$ref": "#/$defs/" + kind,
            }
            self.validators[kind] = jsonschema.Draft202012Validator(schema)
        return self.validators[kind]

    def check(self, doc, kind, where):
        errors = sorted(
            self._validator(kind).iter_errors(doc), key=lambda e: list(e.path)
        )
        self.checked[kind] = self.checked.get(kind, 0) + 1
        for e in errors[:5]:
            path = "/".join(str(p) for p in e.path)
            self.failures.append(f"{where}: [{kind}] /{path}: {e.message[:300]}")

    def check_file(self, path):
        if path.endswith(".jsonl"):
            with open(path) as f:
                for n, line in enumerate(f, 1):
                    doc = json.loads(line)
                    self.check(doc, "case_log_" + doc.get("type", "?"), f"{path}:{n}")
            return
        if not path.endswith(".json") or path.endswith(".provenance.json"):
            return
        with open(path) as f:
            doc = json.load(f)
        schema = doc.get("schema_version") if isinstance(doc, dict) else None
        if schema == "rj_isa_diff.counterexample.v1":
            self.check(doc, "counterexample", path)
        elif schema == "rj_isa_diff.v1" and "overall" in doc:
            self.check(doc, "results", path)
        elif schema == "rj_isa_diff.v1":
            self.check(doc, "case", path)

    def check_tree(self, top):
        if os.path.isfile(top):
            self.check_file(top)
            return
        for dirpath, _, files in os.walk(top):
            for name in sorted(files):
                self.check_file(os.path.join(dirpath, name))


def run(tool, args, expect, failures):
    proc = subprocess.run([tool] + args, capture_output=True, text=True)
    if proc.returncode not in expect:
        failures.append(
            f"`{' '.join(args)}` exited {proc.returncode}, expected {sorted(expect)}:\n"
            f"{proc.stdout[-1500:]}{proc.stderr[-1500:]}"
        )
    return proc


def drive_cli(tool, work, v):
    f = v.failures
    for name, words in FIXTURES.items():
        with open(os.path.join(work, name + ".bin"), "wb") as out:
            out.write(struct.pack("<%dI" % len(words), *words))
    path = lambda *p: os.path.join(work, *p)
    cache = path("cache")

    def analyze(case, old, new, *extra):
        run(
            tool,
            [
                "analyze",
                "--old",
                path(old + ".bin"),
                "--new",
                path(new + ".bin"),
                "--out",
                path(case),
            ]
            + list(extra),
            {0},
            f,
        )

    def test(case, expect, *extra):
        return run(
            tool,
            ["test", "--case", path(case), "--cases", "200"] + list(extra),
            expect,
            f,
        )

    # Equivalent rewrite: pass, then the same run again through the result cache.
    analyze(
        "shift",
        "shift_old",
        "shift_new",
        "--diagnostic-entry-prefix",
        "--cache-dir",
        cache,
    )
    analyze(
        "shift",
        "shift_old",
        "shift_new",
        "--diagnostic-entry-prefix",
        "--cache-dir",
        cache,
    )
    test("shift", {0}, "--cache-dir", cache)
    second = test("shift", {0}, "--cache-dir", cache)
    if "1 hit(s)" not in second.stdout:
        f.append(
            "second cached `test` did not hit the result cache:\n"
            + second.stdout[-800:]
        )
    # Derived prefix constraints (also reports the unrestricted run).
    analyze("constraint", "constraint_old", "constraint_new")
    test("constraint", {0, 1, 2})
    # Two hunks in one block; expansion variants.
    analyze("twohunk", "twohunk_old", "twohunk_new")
    test("twohunk", {0, 1, 2})
    # Renaming: automatic plan is inconclusive; a manual region shows the identity
    # mismatch, and a register map reconciles it (still ASSUMED, scope partial).
    analyze("rename", "rename_old", "rename_new")
    test("rename", {2})
    analyze(
        "rename_manual",
        "rename_old",
        "rename_new",
        "--manual-region",
        "old:5-6,new:5-6",
    )
    test("rename_manual", {1})
    analyze(
        "rename_map",
        "rename_old",
        "rename_new",
        "--manual-region",
        "old:5-6,new:5-6",
        "--map",
        "s4=s8",
    )
    test("rename_map", {2})
    analyze(
        "rename_entry_map",
        "rename_old",
        "rename_new",
        "--manual-region",
        "old:5-6,new:5-6",
        "--entry-map",
        "s4=s8",
    )
    test("rename_entry_map", {2})
    # VALU regions: an equivalent rewrite, a wrong shift (lane counterexamples) and an
    # inactive-lane clobber; per-lane VGPR state appears in every document.
    analyze("valu", "valu_old", "valu_new")
    test("valu", {0})
    analyze("valu_wrong", "valu_old", "valuwrong_new")
    test("valu_wrong", {1})
    analyze("valu_clobber", "valuclobber_old", "valuclobber_new")
    test("valu_clobber", {1})
    vb = sorted(os.listdir(path("valu_wrong", "counterexamples")))
    if vb:
        run(
            tool,
            [
                "replay",
                "--counterexample",
                path("valu_wrong", "counterexamples", vb[0]),
            ],
            {0},
            f,
        )
    else:
        f.append("VALU wrong-shift case produced no counterexample bundle")
    # Synthetic error injection, its counterexamples, and replay of one bundle.
    run(
        tool,
        [
            "inject",
            "--case",
            path("shift"),
            "--region",
            "r0",
            "--xor",
            "0x1",
            "--side",
            "new",
            "--inst",
            "0",
            "--out",
            path("shift_injected"),
        ],
        {0},
        f,
    )
    test("shift_injected", {1})
    bundles = sorted(os.listdir(path("shift_injected", "counterexamples")))
    if not bundles:
        f.append("injected case produced no counterexample bundle")
    else:
        run(
            tool,
            [
                "replay",
                "--counterexample",
                path("shift_injected", "counterexamples", bundles[0]),
            ],
            {0},
            f,
        )
    # Malformed input is an error, not a result.
    run(
        tool,
        [
            "analyze",
            "--old",
            path("shift_old.bin"),
            "--new",
            path("shift_new.bin"),
            "--out",
            path("bad"),
            "--manual-region",
            "old:9-3,new:0-1",
        ],
        {3},
        f,
    )
    for case in [
        "shift",
        "constraint",
        "twohunk",
        "rename",
        "rename_manual",
        "rename_map",
        "rename_entry_map",
        "valu",
        "valu_wrong",
        "valu_clobber",
        "shift_injected",
    ]:
        v.check_tree(path(case))
    v.check_tree(path("cache"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="*", help="files or directories to validate")
    ap.add_argument(
        "--tool", help="rj_isa_diff binary: drive the CLI over built-in fixtures"
    )
    ap.add_argument("--keep", action="store_true", help="keep the CLI work directory")
    args = ap.parse_args()
    v = Validator()
    work = None
    if args.tool:
        work = tempfile.mkdtemp(prefix="rj_isa_diff_schema_")
        drive_cli(args.tool, work, v)
    for p in args.paths:
        v.check_tree(p)
    for kind in sorted(v.checked):
        print(f"validated {v.checked[kind]:4d} {kind}")
    for msg in v.failures:
        print("FAIL " + msg)
    if work and not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    elif work:
        print("kept " + work)
    required = (
        {
            "case",
            "results",
            "counterexample",
            "case_log_header",
            "case_log_case",
            "case_log_summary",
        }
        if args.tool
        else set()
    )
    missing = required - set(v.checked)
    if missing:
        print("FAIL no document of kind: " + ", ".join(sorted(missing)))
    if not v.checked:
        print("FAIL nothing validated")
    return 1 if v.failures or missing or not v.checked else 0


if __name__ == "__main__":
    sys.exit(main())
