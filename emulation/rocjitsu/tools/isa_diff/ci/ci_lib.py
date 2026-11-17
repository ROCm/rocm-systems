# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared pieces of isadiff-ci: configuration, result layout, gh and git, the driver."""

import copy
import datetime
import importlib.machinery
import importlib.util
import json
import os
import subprocess
from pathlib import Path

WORK = Path(os.environ.get("ISADIFF_WORK", "/work"))
TOOL_BIN = Path(os.environ.get("ISADIFF_TOOL_BIN", "/opt/isadiff/bin"))
HOST_LLVM = Path(os.environ.get("ISADIFF_HOST_LLVM", "/host/llvm-project"))
UPSTREAM = "https://github.com/llvm/llvm-project.git"
MARKER = "<!-- rj-isa-diff -->"

DEFAULTS = {
    "repo": "llvm/llvm-project",
    "authors": ["lialan"],
    # Scanned when a change touches a path under one of these prefixes.
    "paths": [
        "llvm/lib/Target/AMDGPU/",
        "llvm/lib/CodeGen/",
        "llvm/include/llvm/CodeGen/",
        "llvm/include/llvm/Target/",
        "llvm/include/llvm/IR/IntrinsicsAMDGPU.td",
        "llvm/lib/Support/KnownBits.cpp",
        "llvm/include/llvm/Support/KnownBits.h",
        "llvm/lib/Analysis/ValueTracking.cpp",
        "llvm/test/CodeGen/AMDGPU/",
    ],
    "local_branches": ["users/lialan/selectiondag_nightly_*"],
    "jobs": 16,
    "budget_hours": {"quick": 3, "deep": 6},
    "run_timeout_hours": {"quick": 2, "deep": 5},
    "job_timeout": {"quick": 300, "deep": 1800},
    "corpus_mcpu": "gfx950",
    "corpus_args": "-O3 -relocation-model=pic",
    "post": {"mode": "manual", "auto_prs": []},
}


def load_config(path=None):
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path) if path else WORK / "config.json"
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    return cfg


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def slug(ident):
    return str(ident).replace("/", "_")


def subject_dir(kind, ident):
    return WORK / "results" / f"{kind}-{slug(ident)}"


def run_dir(kind, ident, head, tier):
    return subject_dir(kind, ident) / f"{head[:12]}-{tier}"


def write_run(d, rec):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    rec = {k: v for k, v in rec.items() if k != "dir"}
    (d / "run.json").write_text(json.dumps(rec, indent=1) + "\n")


def read_run(d):
    p = Path(d) / "run.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    rec["dir"] = str(d)
    return rec


def finished(d):
    rec = read_run(d)
    return bool(rec and rec.get("finished"))


def latest_run(kind, ident, tier=None):
    """The most recently finished run of a subject (of @p tier, if given)."""
    runs = [read_run(p.parent) for p in subject_dir(kind, ident).glob("*/run.json")]
    runs = [r for r in runs if r.get("finished") and tier in (None, r.get("tier"))]
    return max(runs, key=lambda r: r["finished"], default=None)


def gh(args, input=None):
    res = subprocess.run(["gh", *args], input=input, capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f"gh {' '.join(args[:2])}: {res.stderr.strip()[-500:]}")
    return res.stdout


def git(repo, *args):
    res = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    if res.returncode:
        raise RuntimeError(f"git {' '.join(args[:2])}: {res.stderr.strip()[-500:]}")
    return res.stdout


_driver = None


def driver():
    """The installed rj_isa_diff_pr as a module, for verdict() and poison_sensitive()."""
    global _driver
    if _driver is None:
        loader = importlib.machinery.SourceFileLoader(
            "rj_isa_diff_pr", str(TOOL_BIN / "rj_isa_diff_pr")
        )
        spec = importlib.util.spec_from_loader(loader.name, loader)
        _driver = importlib.util.module_from_spec(spec)
        loader.exec_module(_driver)
    return _driver
