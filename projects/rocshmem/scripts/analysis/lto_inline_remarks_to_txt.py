#!/usr/bin/env python3
###############################################################################
# Flatten an LLVM optimization-record YAML file (as produced by
# `-Xoffload-linker --plugin-opt=opt-remarks-filename=...`) into a sorted,
# demangled, greppable/diffable text file -- one line per remark.
#
# This is a first-cut, exploratory dump: it does NOT try to key/diff remarks
# the structured way resource_usage_diff.py does for kernel-resource-usage
# CSVs. The goal right now is just to see what the raw opt-remarks output for
# a whole-program -fgpu-rdc LTO link actually looks like, before deciding
# whether it's worth a real parser/CSV/chart pipeline.
#
# Usage:
#   lto_inline_remarks_to_txt.py --yaml <remarks.yaml> [--out <file>]
###############################################################################
import argparse
import shutil
import subprocess
import sys

import yaml

# LLVM's opt-record YAML tags each document with the remark kind
# (!Passed / !Missed / !Analysis / ...). PyYAML's SafeLoader has no
# constructor for unknown tags and raises; register one for every kind seen
# in practice and stash the kind itself (tag name minus "!") under a
# synthetic "_Kind" key so it survives into the flattened output.
class RemarkLoader(yaml.SafeLoader):
    pass


def _construct_remark(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mapping["_Kind"] = node.tag.lstrip("!")
    return mapping


for _tag in (
    "!Passed",
    "!Missed",
    "!Analysis",
    "!AnalysisFPCommute",
    "!AnalysisAliasing",
    "!Failure",
):
    RemarkLoader.add_constructor(_tag, _construct_remark)


def _cxxfilt_path():
    for name in ("llvm-cxxfilt", "c++filt"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _demangle_batch(names, cxxfilt):
    if not cxxfilt or not names:
        return {n: n for n in names}
    proc = subprocess.run(
        [cxxfilt], input="\n".join(names), capture_output=True, text=True, check=True
    )
    demangled = proc.stdout.splitlines()
    return dict(zip(names, demangled))


def _flatten_args(args, demangle):
    parts = []
    for arg in args or []:
        for key, value in arg.items():
            if key in ("Callee", "Caller", "Function"):
                value = demangle.get(value, value)
            parts.append(str(value))
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="opt-remarks-filename output to flatten")
    ap.add_argument("--out", help="output text file (default: stdout)")
    args = ap.parse_args()

    with open(args.yaml) as f:
        docs = [d for d in yaml.load_all(f, Loader=RemarkLoader) if d]

    mangled_names = set()
    for d in docs:
        for field in ("Function",):
            if field in d:
                mangled_names.add(d[field])
        for arg in d.get("Args", []) or []:
            for key, value in arg.items():
                if key in ("Callee", "Caller"):
                    mangled_names.add(value)

    demangle = _demangle_batch(sorted(mangled_names), _cxxfilt_path())

    lines = []
    for d in docs:
        kind = d.get("_Kind", "?")
        pass_name = d.get("Pass", "?")
        remark_name = d.get("Name", "?")
        func = demangle.get(d.get("Function", "?"), d.get("Function", "?"))
        loc = d.get("DebugLoc") or {}
        loc_str = f"{loc.get('File', '?')}:{loc.get('Line', '?')}:{loc.get('Column', '?')}"
        msg = _flatten_args(d.get("Args"), demangle)
        lines.append(f"[{kind}] {pass_name}/{remark_name} in {func} @ {loc_str} :: {msg}")

    # Sort before writing: a `--lto-partitions>1` link processes functions
    # across parallel backend workers, so raw remark order is not stable
    # build-to-build even for identical source -- a line-order diff would be
    # dominated by that noise rather than real content changes.
    lines.sort()

    out = "\n".join(lines) + ("\n" if lines else "")
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
