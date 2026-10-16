#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE.txt for license information
"""Guard: a symbol with an owning fakes file must not be re-stubbed in a test TU.

MICROTEST_README.md's ownership table says each production TU's symbols belong to one
fakes/ file. Nothing enforced it, and the failure mode is quiet: a test TU that defines
its own copy of an already-owned symbol links fine as long as its binary does not also
link the owner. The two definitions then drift, and the day someone adds the owner to
that binary -- or merges a branch that moved the symbol into fakes/ -- the build breaks
with a duplicate-symbol error that names neither the rule nor the owner.

That is not hypothetical: moving ncclSetThreadName into fakes/nccl_fakes.cc collided
with a fresh per-TU stub of it added in parallel on develop, and git merged both
cleanly because neither side touched the other's lines.

How it works: with -g, `nm -l` reports the source location of each defined symbol, which
separates the two cases that matter. A symbol a test TU defines because it #includes the
production .cc under test resolves to build/hipify/src/..., and is ignored. A symbol the
TU hand-writes resolves to the test .cc itself, and is a stub. If any fakes/ file also
defines that symbol, the stub has an owner and this guard fails.

Needs a completed build (objects under <build>/CMakeFiles/*/). Run from the `guards`
phase of run_host_tests.sh, after `build`.
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUILD = os.path.join(HERE, "build")

# Symbols a test TU is allowed to define even though a fakes/ file owns them.
# Each entry needs a reason; "it was already like that" is not one. Prefer deleting the
# stub and linking the owner -- that is what this guard exists to push you toward.
ALLOWED = {
    # rccl-UnitTestsMicroConnect compiles the real src/graph/connect.cc, so it cannot link
    # topo_stubs.cc (duplicate Preset/Postset) and by extension does not get the shared
    # logging/param/bootstrap floor either. MICROTEST_README.md documents this binary as
    # the deliberate exception; these four are the "few remaining dependencies" it names.
    ("connect-test.cc", "ncclDebugLog"),
    ("connect-test.cc", "ncclLoadParam"),
    ("connect-test.cc", "bootstrapAllGather"),
    ("connect-test.cc", "ncclParamP2pDisable"),
    # scheduler-test.cc compiles the real scheduler, whose symbol table overlaps
    # enqueue_fakes.cc; the fakes file is not linked into that binary.
    ("scheduler-test.cc", "ncclAlgNameForSymk"),
}

# Not test TUs: these wrap production sources into librccl-source-wrappers and are
# expected to carry their own stubs.
SKIP_SOURCES = re.compile(r"_wrapper\.cc$")

SYM_LINE = re.compile(r"^\s*[0-9a-fA-F]*\s+([A-Za-z])\s+(\S+)(?:\t(.*))?$")


def defined_symbols(obj):
    """[(symbol, source_path, line)] for strong text symbols with a known location."""
    try:
        out = subprocess.run(["nm", "-l", "--defined-only", "--extern-only", obj],
                             capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        sys.exit("error: `nm` not found; this guard needs binutils on PATH")
    found = []
    for line in out.splitlines():
        m = SYM_LINE.match(line)
        if not m or m.group(1) != "T" or not m.group(3):
            continue
        loc = m.group(3).rsplit(":", 1)
        if len(loc) != 2:
            continue
        found.append((m.group(2), loc[0], loc[1]))
    return found


def demangle(names):
    if not names:
        return {}
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True,
                         text=True, check=False).stdout.splitlines()
    pretty = {}
    for mangled, plain in zip(names, out):
        pretty[mangled] = plain.split("(")[0].strip() or mangled
    return pretty


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=DEFAULT_BUILD)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    objroot = os.path.join(args.build_dir, "CMakeFiles")
    if not os.path.isdir(objroot):
        sys.exit(f"error: {objroot} not found -- run the `build` phase first")

    objects = []
    for dirpath, _, files in os.walk(objroot):
        objects.extend(os.path.join(dirpath, f) for f in files if f.endswith(".cc.o"))
    if not objects:
        sys.exit(f"error: no .cc.o objects under {objroot} -- run the `build` phase first")

    fakes_dir = os.path.join(HERE, "fakes") + os.sep
    owners = {}                      # symbol -> owning fakes file
    stubs = defaultdict(set)         # (test source, symbol) -> {"file:line"}

    for obj in objects:
        for sym, src, line in defined_symbols(obj):
            # Objects left behind by a renamed or deleted source are not evidence.
            if not os.path.exists(src):
                continue
            base = os.path.basename(src)
            if src.startswith(fakes_dir):
                owners.setdefault(sym, base)
            elif os.path.dirname(src) + os.sep == HERE + os.sep:
                if SKIP_SOURCES.search(base):
                    continue
                stubs[(base, sym)].add(f"{base}:{line}")
            # Anything else (build/hipify/src/...) is the unit under test. Ignored.

    violations = []
    for (src, sym), sites in sorted(stubs.items()):
        if sym not in owners:
            continue
        pretty = demangle([sym])[sym]
        if (src, pretty) in ALLOWED or (src, sym) in ALLOWED:
            continue
        violations.append((src, sym, pretty, owners[sym], sorted(sites)))

    if args.verbose:
        print(f"scanned {len(objects)} objects: {len(owners)} symbols owned by fakes/, "
              f"{len(stubs)} hand-written stubs in test TUs, {len(ALLOWED)} allowed")

    if not violations:
        print("OK: no test TU re-stubs a symbol that a fakes/ file owns")
        return 0

    print("FAIL: these symbols already have an owning fakes file:\n")
    for src, _sym, pretty, owner, sites in violations:
        print(f"  {pretty}")
        print(f"      stubbed at : {', '.join(sites)}")
        print(f"      owned by   : fakes/{owner}")
        print(f"      fix        : delete the stub in {src} and add fakes/{owner} to that")
        print(f"                   binary's source list in test/host/CMakeLists.txt.")
        print(f"                   If it genuinely cannot link, add it to ALLOWED in")
        print(f"                   {os.path.basename(__file__)} with the reason.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
