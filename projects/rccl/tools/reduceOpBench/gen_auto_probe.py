#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Generate fp8_auto_probe.h from the installed ROCm/LLVM BuiltinsAMDGPU database.

Discovers FP8/BF8 AMDGCN builtins automatically (no hand-maintained checklist)
and emits a device kernel that reports __builtin_amdgcn_is_invocable() for each.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


# Match TARGET_BUILTIN(__builtin_amdgcn_..., "type", "attrs", "features")
RE_TARGET_DEF = re.compile(
    r'TARGET_BUILTIN\((__builtin_amdgcn_[A-Za-z0-9_]+),\s*"([^"]*)",\s*"([^"]*)",\s*"([^"]*)"\)'
)

# Newer Clang BuiltinsAMDGPU.inc string-table form (name appears as a C string)
RE_INC_NAME = re.compile(r'"(__builtin_amdgcn_[A-Za-z0-9_]+)\\0"')

# Feature annotation lines near Info entries in .inc (best-effort; optional)
RE_INC_FEATURE_HINT = re.compile(
    r'Builtin::Info\{[^}]*?/\*\s*(__builtin_amdgcn_[A-Za-z0-9_]+)\s*\*/[^}]*?/\*\s*([a-z0-9_,+-]+)\s*\*/'
)

# Builtin names as they appear in a clang executable's embedded string table
RE_BIN_NAME = re.compile(rb"__builtin_amdgcn_[A-Za-z0-9_]+")

FP8_NAME = re.compile(r"(fp8|bf8)", re.IGNORECASE)


def find_builtins_file(rocm_path: Path) -> Path | None:
    candidates = [
        rocm_path / "lib/llvm/include/clang/Basic/BuiltinsAMDGPU.def",
        rocm_path / "lib/llvm/include/clang/Basic/BuiltinsAMDGPU.inc",
        rocm_path / "llvm/include/clang/Basic/BuiltinsAMDGPU.def",
        rocm_path / "llvm/include/clang/Basic/BuiltinsAMDGPU.inc",
    ]
    # Also try hipcc's llvm relative to ROCM_PATH symlink
    for c in candidates:
        if c.is_file():
            return c
    return None


def find_clang_binary(rocm_path: Path) -> Path | None:
    for sub in ("llvm/bin", "lib/llvm/bin"):
        matches = sorted((rocm_path / sub).glob("clang-[0-9]*"))
        for c in matches:
            if c.is_file() and c.suffix != ".cfg":
                return c
    return None


def group_label(short: str) -> str:
    """Display grouping derived from the builtin name.

    Used when the target-feature string is unrecoverable (binary extraction),
    so output stays organised instead of collapsing into one block.
    """
    cat = categorize(short)
    if cat != "cvt":
        return cat
    if "fp8" in short and "bf8" in short:
        tag = "fp8/bf8"
    elif "bf8" in short:
        tag = "bf8"
    else:
        tag = "fp8"
    return f"{tag} {'scale convert' if 'scale' in short else 'convert'}"


def categorize(short: str) -> str:
    if short.startswith("cvt_"):
        return "cvt"
    if short.startswith(("mfma_", "smfmac_")):
        return "mfma"
    if short.startswith(("wmma_", "swmmac_")):
        return "wmma"
    if short.startswith("dot"):
        return "dot"
    return "other"


def parse_def(text: str) -> list[dict]:
    rows = []
    for m in RE_TARGET_DEF.finditer(text):
        full, ty, _attrs, feat = m.groups()
        short = full.replace("__builtin_amdgcn_", "")
        if not FP8_NAME.search(short):
            continue
        rows.append(
            {
                "full": full,
                "short": short,
                "type": ty,
                "feature": feat,
                "category": categorize(short),
            }
        )
    return rows


def parse_inc(text: str) -> list[dict]:
    # Prefer feature-annotated Info entries when present; fall back to name scan.
    by_name: dict[str, dict] = {}
    for m in RE_INC_FEATURE_HINT.finditer(text):
        full, feat = m.groups()
        short = full.replace("__builtin_amdgcn_", "")
        if not FP8_NAME.search(short):
            continue
        by_name[short] = {
            "full": full,
            "short": short,
            "type": "",
            "feature": feat,
            "category": categorize(short),
        }
    if by_name:
        return sorted(by_name.values(), key=lambda r: r["short"])

    for m in RE_INC_NAME.finditer(text):
        full = m.group(1)
        short = full.replace("__builtin_amdgcn_", "")
        if not FP8_NAME.search(short):
            continue
        by_name[short] = {
            "full": full,
            "short": short,
            "type": "",
            "feature": "unknown",
            "category": categorize(short),
        }
    return sorted(by_name.values(), key=lambda r: r["short"])


def parse_clang_binary(path: Path) -> list[dict]:
    """Recover builtin names from a clang executable's string table.

    Runtime-only ROCm installs ship no clang/Basic headers, but the driver
    binary still carries every builtin name. Target-feature strings are not
    recoverable this way, so they are reported as unknown.
    """
    chunk = 1 << 22
    overlap = 128  # exceeds the longest builtin name, so matches never split
    names: set[str] = set()
    tail = b""
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            final = len(buf) < chunk
            window = tail + buf
            for m in RE_BIN_NAME.finditer(window):
                # A match touching the window edge may be truncated; the next
                # window re-scans it via the overlap.
                if not final and m.end() == len(window):
                    continue
                names.add(m.group(0).decode("ascii"))
            if final:
                break
            tail = window[-overlap:]

    rows = []
    for full in sorted(names):
        short = full.replace("__builtin_amdgcn_", "")
        if not FP8_NAME.search(short):
            continue
        rows.append(
            {
                "full": full,
                "short": short,
                "type": "",
                "feature": "unknown",
                "category": categorize(short),
            }
        )
    return rows


def parse_builtins(path: Path) -> list[dict]:
    if path.suffix in (".def", ".inc"):
        text = path.read_text(errors="replace")
        rows = parse_def(text) if path.suffix == ".def" else parse_inc(text)
    else:
        rows = parse_clang_binary(path)
    for r in rows:
        r["group"] = group_label(r["short"])
    # Stable order: category, then group so sections stay contiguous, then name
    rows.sort(key=lambda r: (r["category"], r["group"], r["short"]))
    return rows


def c_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def generate_header(rows: list[dict], source: Path, rocm_path: Path) -> str:
    n = len(rows)
    lines: list[str] = []
    lines.append("/* Auto-generated by gen_auto_probe.py — do not edit by hand. */")
    lines.append(f"/* Source: {source} */")
    lines.append(f"/* ROCm:    {rocm_path} */")
    lines.append("#ifndef FP8_AUTO_PROBE_H_")
    lines.append("#define FP8_AUTO_PROBE_H_")
    lines.append("")
    lines.append("#include <hip/hip_runtime.h>")
    lines.append("")
    lines.append(f"#define FP8_AUTO_NUM_INTRINSICS {n}")
    lines.append("")
    lines.append("struct Fp8AutoBuiltin {")
    lines.append("  const char* name;")
    lines.append("  const char* feature;")
    lines.append("  const char* category;")
    lines.append("  const char* group;")
    lines.append("};")
    lines.append("")
    lines.append("struct Fp8AutoProbeResult {")
    lines.append("  int invocable[FP8_AUTO_NUM_INTRINSICS];")
    lines.append("};")
    lines.append("")
    lines.append("static const Fp8AutoBuiltin kFp8AutoBuiltins[FP8_AUTO_NUM_INTRINSICS] = {")
    for r in rows:
        lines.append(
            f'  {{"{c_escape(r["short"])}", "{c_escape(r["feature"])}", '
            f'"{c_escape(r["category"])}", "{c_escape(r["group"])}"}},'
        )
    lines.append("};")
    lines.append("")
    lines.append("__global__ void fp8AutoProbeKernel(Fp8AutoProbeResult* out) {")
    lines.append("  if (threadIdx.x != 0 || blockIdx.x != 0) return;")
    lines.append("  Fp8AutoProbeResult r{};")
    for i, r in enumerate(rows):
        lines.append(
            f"  r.invocable[{i}] = "
            f"__builtin_amdgcn_is_invocable({r['full']}) ? 1 : 0;"
        )
    lines.append("  *out = r;")
    lines.append("}")
    lines.append("")
    lines.append("#endif  // FP8_AUTO_PROBE_H_")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rocm-path",
        default=os.environ.get("ROCM_PATH", "/opt/rocm"),
        help="ROCm install root (default: $ROCM_PATH or /opt/rocm)",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="fp8_auto_probe.h",
        help="Output header path",
    )
    ap.add_argument(
        "--builtins",
        default=None,
        help="Explicit path to BuiltinsAMDGPU.def/.inc (overrides --rocm-path search)",
    )
    ap.add_argument(
        "--categories",
        default="cvt",
        help="Comma-separated categories to include: cvt,mfma,wmma,dot,other,all "
        "(default: cvt)",
    )
    args = ap.parse_args()

    rocm_path = Path(args.rocm_path).resolve()
    if args.builtins:
        src = Path(args.builtins).resolve()
    else:
        # Runtime-only installs ship no clang/Basic headers; fall back to the
        # builtin names embedded in the clang executable itself.
        src = find_builtins_file(rocm_path) or find_clang_binary(rocm_path)
    if src is None:
        print(
            f"error: found neither BuiltinsAMDGPU.def/.inc nor a clang binary under "
            f"{rocm_path}. Set --rocm-path, or pass --builtins explicitly.",
            file=sys.stderr,
        )
        return 1

    rows = parse_builtins(src)
    if not rows:
        print(f"error: no FP8/BF8 builtins found in {src}", file=sys.stderr)
        return 1

    cats = {c.strip() for c in args.categories.split(",") if c.strip()}
    if "all" not in cats:
        rows = [r for r in rows if r["category"] in cats]
        if not rows:
            print(
                f"error: no builtins left after --categories={args.categories}",
                file=sys.stderr,
            )
            return 1

    out = Path(args.output)
    out.write_text(generate_header(rows, src, rocm_path))
    print(f"Wrote {out} ({len(rows)} builtins from {src})")
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    for k, v in sorted(by_cat.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
