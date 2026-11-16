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
    "memst_old": [
        0x68040300,
        0xDC708000,
        0x00000204,
        0xBF810000,
    ],
    "memst_new": [
        0x68060101,
        0xDC708000,
        0x00000304,
        0xBF810000,
    ],
    "memstoff_new": [
        0x68040300,
        0xDC708004,
        0x00000204,
        0xBF810000,
    ],
    "memld_old": [
        0xDC508000,
        0x01020000,
        0xBF8C0F70,
        0x68040701,
        0xDC708000,
        0x00000200,
        0xBF810000,
    ],
    "memld_new": [
        0xDC508000,
        0x04020000,
        0xBF8C0F70,
        0x680E0903,
        0xDC708000,
        0x00000700,
        0xBF810000,
    ],
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
    "valuclobberlater_old": [
        0x7E060302,
        0xDC708000,
        0x00000304,
        0xBEFE01C1,
        0xDC708000,
        0x00020304,
        0xBF810000,
    ],
    "valuclobberlater_new": [
        0xBE8A017E,
        0xBEFE01C1,
        0x7E060302,
        0xBEFE010A,
        0xDC708000,
        0x00000304,
        0xBEFE01C1,
        0xDC708000,
        0x00020304,
        0xBF810000,
    ],
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


# A callable function as llc prints it, with inline-asm placeholders (LLVM plugin: the .s
# path needs llvm-mc and ld.lld). TMP is the temporary the def produces, USE what the use
# block reads.
FUNCTION_LISTING = """\t.text
\t.globl\tg
\t.p2align\t8
\t.type\tg,@function
g:
\ts_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
\t;;#ASMSTART
\t; def TMP
\t;;#ASMEND
\ts_nop 0
\tv_add_u32_e32 TMP, v0, TMP
\tv_mul_lo_u32 v0, TMP, v1
\t;;#ASMSTART
\t; use USE
\t;;#ASMEND
\ts_setpc_b64 s[30:31]
.Lfunc_end0:
\t.size\tg, .Lfunc_end0-g
"""
FUNCTIONS = {
    "fn_old": {"TMP": "v2", "USE": "v2"},
    "fn_new": {"TMP": "v3", "USE": "v3"},  # renamed everywhere: equivalent
    "fn_wrong": {"TMP": "v2", "USE": "v0"},  # the use reads another register
}


# Functions for other targets: gfx1100 wave32 (a select on vcc_lo after an f32 add) and
# gfx1200 (an SALU f32 multiply moved to v0), each storing v0 through s[0:1].
def target_listing(body, store, wait):
    return (
        "\t.text\n\t.globl\tf\n\t.p2align\t8\n\t.type\tf,@function\nf:\n"
        + "".join(f"\t{l}\n" for l in body)
        + f"\tv_mov_b32_e32 v4, 0\n\t{store} v4, v0, s[0:1]\n\t{wait}\n"
        + "\ts_setpc_b64 s[30:31]\n.Lfunc_end0:\n\t.size\tf, .Lfunc_end0-f\n"
    )


def w32_listing(add):
    return target_listing(
        [
            "v_cmp_lt_u32_e32 vcc_lo, v2, v3",
            add,
            "v_cndmask_b32_e32 v0, v2, v0, vcc_lo",
        ],
        "global_store_b32",
        "s_waitcnt_vscnt null, 0x0",
    )


def salu_listing(op):
    return target_listing(
        [op, "v_mov_b32_e32 v0, s6"], "global_store_b32", "s_wait_storecnt 0x0"
    )


TARGET_FUNCTIONS = {
    "w32_ab": w32_listing("v_add_f32_e32 v0, v0, v1"),
    "w32_ba": w32_listing("v_add_f32_e32 v0, v1, v0"),
    "salu_ab": salu_listing("s_mul_f32 s6, s2, s3"),
    "salu_ba": salu_listing("s_mul_f32 s6, s3, s2"),
}


# Void functions storing v0 through the inreg pointer s[0:1] after BODY (one op per line).
STORE_LISTING = """\t.text
\t.globl\tf
\t.p2align\t8
\t.type\tf,@function
f:
BODY
\tv_mov_b32_e32 v4, 0
\tglobal_store_dword v4, v0, s[0:1]
\ts_waitcnt vmcnt(0)
\ts_setpc_b64 s[30:31]
.Lfunc_end0:
\t.size\tf, .Lfunc_end0-f
"""
STORE_FUNCTIONS = {
    "nan_ab": "\tv_add_f32_e32 v0, v0, v1",
    "nan_ba": "\tv_add_f32_e32 v0, v1, v0",  # commuted: NaN payloads may differ
    "undef_old": "\tv_mov_b32_e32 v0, v5",  # v5, v7: never written, not arguments
    "undef_new": "\tv_mov_b32_e32 v0, v7",
    "nsw_wrap": "\tv_add_u32_e32 v0, v0, v1",
    "nsw_sat": "\tv_add_i32 v0, v0, v1 clamp",  # differs only where add nsw overflows
}


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
    # Clobbered inactive lanes that a later full-EXEC store observes: a local mismatch.
    analyze(
        "valu_clobber",
        "valuclobberlater_old",
        "valuclobberlater_new",
        "--max-expansions",
        "0",
    )
    test("valu_clobber", {1})
    # At a kernel end nothing reads them again (EXEC never changes after the region).
    analyze("valu_clobber_ctx", "valuclobber_old", "valuclobber_new")
    test("valu_clobber_ctx", {0})
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
    # Memory regions: stores are observable bytes, loads read one shared initial image;
    # a store to a different offset is a memory counterexample that replays.
    analyze("memst", "memst_old", "memst_new")
    test("memst", {0})
    analyze("memld", "memld_old", "memld_new")
    test("memld", {0})
    analyze("memst_off", "memst_old", "memstoff_new")
    test("memst_off", {1})
    mb = sorted(os.listdir(path("memst_off", "counterexamples")))
    if mb:
        run(
            tool,
            [
                "replay",
                "--counterexample",
                path("memst_off", "counterexamples", mb[0]),
            ],
            {0},
            f,
        )
    else:
        f.append("memory store-offset case produced no counterexample bundle")
    # Callable function with comment-only inline asm (.s through the LLVM plugin).
    for name, subst in FUNCTIONS.items():
        text = FUNCTION_LISTING
        for k, val in subst.items():
            text = text.replace(k, val)
        with open(path(name + ".s"), "w") as out:
            out.write(text)
    for case, new_name, code in (("fn", "fn_new", 0), ("fn_wrong", "fn_wrong", 1)):
        run(
            tool,
            [
                "analyze",
                "--old",
                path("fn_old.s"),
                "--new",
                path(new_name + ".s"),
                "--kernel",
                "g",
                "--return-vgprs",
                "1",
                "--out",
                path(case),
            ],
            {0},
            f,
        )
        test(case, {code})
    # NaN policy, undefined values, poison filter (callable functions, LLVM plugin).
    for name, body in STORE_FUNCTIONS.items():
        with open(path(name + ".s"), "w") as out:
            out.write(STORE_LISTING.replace("BODY", body))

    def analyze_fn(case, old, new, *extra):
        run(
            tool,
            [
                "analyze",
                "--old",
                path(old + ".s"),
                "--new",
                path(new + ".s"),
                "--kernel",
                "f",
                "--return-vgprs",
                "0",
                "--out",
                path(case),
            ]
            + list(extra),
            {0},
            f,
        )

    analyze_fn("nan", "nan_ab", "nan_ba", "--entry-vgprs", "2")
    test("nan", {0})
    analyze_fn("nan_strict", "nan_ab", "nan_ba", "--entry-vgprs", "2")
    test("nan_strict", {1}, "--strict-nan")
    nb = sorted(os.listdir(path("nan_strict", "counterexamples")))
    if nb:
        run(
            tool,
            [
                "replay",
                "--counterexample",
                path("nan_strict", "counterexamples", nb[0]),
            ],
            {0},
            f,
        )
    else:
        f.append("strict-NaN case produced no counterexample bundle")
    # Wave32 (gfx1100) and GFX12 SALU float: NaN provenance through a wave32 select and an
    # SGPR, both equal by default and different with --strict-nan.
    for name, text in TARGET_FUNCTIONS.items():
        with open(path(name + ".s"), "w") as out:
            out.write(text)
    analyze_fn("w32", "w32_ab", "w32_ba", "--entry-vgprs", "4", "--target", "gfx1100")
    test("w32", {0})
    analyze_fn(
        "w32_strict",
        "w32_ab",
        "w32_ba",
        "--entry-vgprs",
        "4",
        "--target",
        "gfx1100:w32",
    )
    test("w32_strict", {1}, "--strict-nan")
    analyze_fn(
        "salu", "salu_ab", "salu_ba", "--entry-sgprs", "8", "--target", "gfx1200"
    )
    test("salu", {0})
    analyze_fn(
        "salu_strict", "salu_ab", "salu_ba", "--entry-sgprs", "8", "--target", "gfx1200"
    )
    test("salu_strict", {1}, "--strict-nan")
    analyze_fn("undef", "undef_old", "undef_new", "--entry-vgprs", "0")
    test("undef", {2})  # equal only through the don't-care mask: not a plain pass
    analyze_fn("poison", "nsw_wrap", "nsw_sat", "--entry-vgprs", "2")
    test("poison", {1})
    analyze_fn("poison_filtered", "nsw_wrap", "nsw_sat", "--entry-vgprs", "2")
    test("poison_filtered", {0}, "--poison-filter", "nsw")
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
        "valu_clobber_ctx",
        "memst",
        "memld",
        "memst_off",
        "fn",
        "fn_wrong",
        "nan",
        "nan_strict",
        "w32",
        "w32_strict",
        "salu",
        "salu_strict",
        "undef",
        "poison",
        "poison_filtered",
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
