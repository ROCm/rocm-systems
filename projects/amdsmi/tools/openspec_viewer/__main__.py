# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Command line entry point: ``python3 -m openspec_viewer [roots...]``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .model import Project, check, load_site, project_slug
from .render import render

# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def default_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "openspec"


def default_out(roots: Sequence[Path]) -> Path:
    tmp = Path(os.environ.get("TMPDIR") or "/tmp")
    name = project_slug(roots[0]) if len(roots) == 1 else "view"
    return tmp / f"openspec-{name or 'view'}.html"


def resolve_root(arg: Optional[str]) -> Path:
    """Accept either an openspec/ directory or the project directory above it."""
    root = Path(arg).expanduser().resolve() if arg else default_root()
    if not _is_root(root) and _is_root(root / "openspec"):
        root = root / "openspec"
    return root


def _is_root(root: Path) -> bool:
    """A root needs specs/ or changes/: a changes-only project is legal."""
    return (root / "specs").is_dir() or (root / "changes").is_dir()


def summary(project: Project) -> str:
    caps = f"{len(project.capabilities)} capabilities"
    reqs = f"{project.requirement_count} requirements, {project.scenario_count} scenarios"
    changes = f", {len(project.changes)} changes" if project.changes else ""
    return f"{project.slug}: {caps}, {reqs}{changes}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Render one or more OpenSpec directories as a single self-contained HTML page.",
        epilog="With no arguments, reads the openspec/ directory beside this "
        "script's project and writes the page under $TMPDIR.",
    )
    ap.add_argument(
        "roots", nargs="*", help="paths to openspec/ directories (default: alongside this script)"
    )
    ap.add_argument("--out", help="output HTML file (default: $TMPDIR/openspec-<project>.html)")
    ap.add_argument(
        "--check",
        action="store_true",
        help="validate spec structure and exit non-zero on problems; writes nothing",
    )
    args = ap.parse_args(argv)

    roots: List[Path] = [resolve_root(a) for a in args.roots] or [resolve_root(None)]
    for root in roots:
        if not _is_root(root):
            print(f"error: no specs/ or changes/ directory under {root}", file=sys.stderr)
            return 2

    site = load_site(roots)
    for project in site.projects:
        if not project.capabilities and not project.changes:
            print(f"error: no spec.md or changes/ content under {project.root}", file=sys.stderr)
            return 2

    if args.check:
        return 1 if sum(check(p) for p in site.projects) else 0

    out = Path(args.out).expanduser() if args.out else default_out(roots)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(site), encoding="utf-8")
    print(out)
    for project in site.projects:
        print(f"  {summary(project)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
