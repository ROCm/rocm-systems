#!/usr/bin/env python3
"""
check_loader_drift.py
=====================

Verify that the dynamic-library loader region in
``py-interface/amdsmi_wrapper.py`` and the f-string template inside
``tools/generator.py`` stay in sync.

The wrapper-side region is delimited by::

    # AMDSMI_LOADER_BEGIN
    ...
    # AMDSMI_LOADER_END

The generator-side region is the f-string assigned to ``new_line`` in
the Linux branch. We extract that f-string via ``ast``, evaluate it with
``library_name="libamd_smi.so"`` (matching what generator.py emits for
Linux), then extract the same marker-bounded slice from the rendered
template and diff against the wrapper region.

Exits non-zero with a unified diff if the regions diverge.
"""

from __future__ import annotations

import ast
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WRAPPER = ROOT / "py-interface" / "amdsmi_wrapper.py"
GENERATOR = ROOT / "tools" / "generator.py"

_MARKER_RE = re.compile(r"# AMDSMI_LOADER_BEGIN.*?\n(?P<body>.*?)# AMDSMI_LOADER_END", re.DOTALL)


def _extract_marker_body(text: str, source_label: str) -> str:
    m = _MARKER_RE.search(text)
    if not m:
        sys.exit(f"[check_loader_drift] missing AMDSMI_LOADER_BEGIN/END markers in {source_label}")
    return m.group("body")


def _render_generator_template() -> str:
    """Find the f-string assigned to ``new_line`` in generator.py and eval it."""
    tree = ast.parse(GENERATOR.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "new_line"
            and isinstance(node.value, ast.JoinedStr)
        ):
            # Found an f-string assigned to new_line. Evaluate it with the
            # only free variable used by the template.
            code = compile(ast.Expression(node.value), filename=str(GENERATOR), mode="eval")
            return eval(code, {"library_name": "libamd_smi.so"})  # noqa: S307
    sys.exit(
        '[check_loader_drift] could not locate `new_line = f"""..."""` '
        "assignment in tools/generator.py"
    )


def main() -> int:
    wrapper_body = _extract_marker_body(WRAPPER.read_text(), str(WRAPPER))
    rendered = _render_generator_template()
    generator_body = _extract_marker_body(rendered, "tools/generator.py (rendered)")
    if wrapper_body == generator_body:
        print("[check_loader_drift] loader regions are in sync")
        return 0
    diff = difflib.unified_diff(
        generator_body.splitlines(),
        wrapper_body.splitlines(),
        fromfile="tools/generator.py (rendered)",
        tofile="py-interface/amdsmi_wrapper.py",
        lineterm="",
    )
    sys.stderr.write("[check_loader_drift] LOADER REGION DRIFT:\n")
    sys.stderr.write("\n".join(diff))
    sys.stderr.write(
        "\n\nFix: update tools/generator.py to match the wrapper, then run "
        "tools/update_wrapper.sh to regenerate.\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
