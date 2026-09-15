#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
run_amdsmi_install_selector_test.py
===================================

The install page shows one installation method at a time. A section is tagged
with ``:method: <value>`` and the browser script reveals it only when the
selected value matches. A value the extension does not define matches nothing,
so the section becomes permanently invisible. The page still builds and Sphinx
reports no warning, so nothing else catches this.

This guard asserts the page and the extension agree:

- every selector value used in install.md is defined in the extension
- the class names the extension emits are parseable by the browser script
- the assets the extension registers actually exist

It parses the extension with ``ast`` instead of importing it, so it runs
without Sphinx or docutils installed.
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTENSION = REPO_ROOT / "docs" / "extension" / "amdsmi_docs" / "install_selector.py"
INSTALL_PAGE = REPO_ROOT / "docs" / "install" / "install.md"
STATIC_DIR = REPO_ROOT / "docs" / "static"

# Must stay in sync with selectionFromClasses() in install-selector.js.
CLASS_PATTERN = re.compile(r"^amdsmi-when--([a-z]+)-(.+)$")

DIRECTIVE_PATTERN = re.compile(r"^\s*:{3,}\{install-(section|when)\}\s*$")
OPTION_PATTERN = re.compile(r"^\s*:([a-z]+):\s*(.+?)\s*$")


def _literal(tree: ast.Module, name: str) -> object:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {EXTENSION}")


def load_extension_config() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return ``(axis -> defined values, axis -> methods it applies to)``."""
    tree = ast.parse(EXTENSION.read_text())
    axes = _literal(tree, "AXES")
    applies_to = _literal(tree, "AXIS_APPLIES_TO")
    return (
        {axis: [value for value, _ in options] for axis, (_, options) in axes.items()},
        dict(applies_to),
    )


def used_values(page: Path) -> Set[Tuple[str, str]]:
    """Collect ``(axis, value)`` pairs used by selector directives in a page."""
    used: Set[Tuple[str, str]] = set()
    in_directive = False
    for line in page.read_text().splitlines():
        if DIRECTIVE_PATTERN.match(line):
            in_directive = True
            continue
        if not in_directive:
            continue
        option = OPTION_PATTERN.match(line)
        if option:
            axis, values = option.group(1), option.group(2)
            used.update((axis, value) for value in values.split())
        else:
            in_directive = False
    return used


def check() -> List[str]:
    errors: List[str] = []
    defined, applies_to = load_extension_config()

    for axis, value in sorted(used_values(INSTALL_PAGE)):
        if axis not in defined:
            errors.append(f"install.md uses undefined axis ':{axis}:'")
        elif value not in defined[axis]:
            errors.append(
                f"install.md uses ':{axis}: {value}' but the extension defines "
                f"{defined[axis]} -- that content would never be shown"
            )

    for axis, values in defined.items():
        for value in values:
            name = f"amdsmi-when--{axis}-{value}"
            match = CLASS_PATTERN.match(name)
            if not match:
                errors.append(f"class '{name}' is not parseable by the browser script")
            elif (match.group(1), match.group(2)) != (axis, value):
                errors.append(
                    f"class '{name}' parses as {match.group(1, 2)}, expected {(axis, value)}"
                )

    methods = defined.get("method", [])
    for axis, limited_to in applies_to.items():
        if axis not in defined:
            errors.append(f"AXIS_APPLIES_TO references undefined axis '{axis}'")
        for method in limited_to:
            if method not in methods:
                errors.append(f"AXIS_APPLIES_TO['{axis}'] references unknown method '{method}'")

    for asset in ("install-selector.css", "install-selector.js"):
        if not (STATIC_DIR / asset).is_file():
            errors.append(f"missing static asset {STATIC_DIR / asset}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="only print output on failure")
    args = parser.parse_args()

    for path in (EXTENSION, INSTALL_PAGE):
        if not path.is_file():
            print(f"FAIL: missing {path}", file=sys.stderr)
            return 1

    errors = check()
    if errors:
        print("FAIL: install selector and docs disagree", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("PASS: install selector values, class names, and assets are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
