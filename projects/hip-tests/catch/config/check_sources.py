# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import argparse
import os
import re
import sys

from common import NON_UNIT_GROUPS, iter_group_configs

CPP_EXTENSIONS = (".cc", ".cpp", ".cxx", ".c++", ".C", ".cp", ".CPP")


def _split_top_level(s):
    """Split a string on commas, ignoring commas inside nested parentheses."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
            continue
        current.append(ch)
    parts.append(''.join(current))
    return parts


def _read_source_files(source_root, group, is_unit):
    """Read all source and header files for a group, return dict of path->content."""
    if is_unit:
        source_dir = os.path.join(source_root, "unit", group)
    else:
        source_dir = os.path.join(source_root, group)

    if not os.path.isdir(source_dir):
        return source_dir, {}

    all_content = {}
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith(CPP_EXTENSIONS) or filename.endswith(
                (".hh", ".h", ".hpp")
            ):
                filepath = os.path.join(root, filename)
                with open(filepath, errors="replace") as f:
                    all_content[filepath] = f.read()
    return source_dir, all_content


def find_source_test_cases(source_root, group, is_unit):
    """Find test names using only HIP_TEST_CASE / HIP_TEMPLATE_TEST_CASE.

    This is the original forward-check logic.
    """
    source_dir, all_content = _read_source_files(source_root, group, is_unit)
    if not all_content:
        return set()

    test_names = set()
    pattern = re.compile(
        r'(?:HIP_TEST_CASE|HIP_TEMPLATE_TEST_CASE)\(\s*(\w+)\s*[,)]'
    )
    for filepath, content in all_content.items():
        if not filepath.endswith(CPP_EXTENSIONS):
            continue
        for match in pattern.finditer(content):
            test_names.add(match.group(1))

    return test_names


def find_all_test_cases(source_root, group, is_unit):
    """Find all test names including TEST_CASE, macro-generated, and chained macros.

    Used for the reverse (stale entry) check to avoid false positives.
    """
    source_dir, all_content = _read_source_files(source_root, group, is_unit)
    if not all_content:
        return set()

    test_names = set()
    hip_test_pattern = re.compile(
        r'(?:HIP_TEST_CASE|HIP_TEMPLATE_TEST_CASE)\(\s*(\w+)\s*[,)]'
    )
    plain_test_pattern = re.compile(
        r'(?:TEST_CASE|TEMPLATE_TEST_CASE)\(\s*"?(\w+)"?\s*[,)]'
    )

    # --- Macro-generated test name detection ---
    macro_def_re = re.compile(r'#define\s+(\w+)\(([^)]*)\)')
    test_in_macro_re = re.compile(
        r'(?:HIP_TEST_CASE|HIP_TEMPLATE_TEST_CASE)\(([^,)]*##[^,)]*)'
    )

    # First pass: collect all macro bodies
    all_macro_defs = {}
    for filepath, content in all_content.items():
        joined = re.sub(r"\\\s*\n", " ", content)
        for line in joined.split("\n"):
            m = macro_def_re.match(line.strip())
            if not m:
                continue
            macro_name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",")]
            body = line.strip()[m.end():]
            all_macro_defs.setdefault(macro_name, []).append((params, body))

    # Second pass: find macros that directly contain HIP_TEST_CASE with ##
    direct_macros = []
    for macro_name, defs in all_macro_defs.items():
        for params, body in defs:
            for tm in test_in_macro_re.finditer(body):
                test_name_expr = tm.group(1).strip()
                direct_macros.append((macro_name, params, test_name_expr))

    # Third pass: find wrapper macros (one level of indirection)
    macros = list(direct_macros)
    direct_names = {m[0] for m in direct_macros}
    for macro_name, defs in all_macro_defs.items():
        if macro_name in direct_names:
            continue
        for params, body in defs:
            for dm_name, dm_params, dm_expr in direct_macros:
                call_re = re.compile(re.escape(dm_name) + r'\(([^)]*)\)')
                cm = call_re.search(body)
                if not cm:
                    continue
                inner_args = [a.strip() for a in cm.group(1).split(",")]
                resolved_expr = dm_expr
                for i, dp in enumerate(dm_params):
                    if i < len(inner_args):
                        resolved_expr = resolved_expr.replace(dp, inner_args[i])
                if "##" in resolved_expr:
                    macros.append((macro_name, params, resolved_expr))

    # Find literal test names (skip macro definition lines)
    for filepath, content in all_content.items():
        if not filepath.endswith(CPP_EXTENSIONS):
            continue
        joined = re.sub(r"\\\s*\n", " ", content)
        for line in joined.split("\n"):
            if line.lstrip().startswith("#define"):
                continue
            for match in hip_test_pattern.finditer(line):
                test_names.add(match.group(1))
            for match in plain_test_pattern.finditer(line):
                test_names.add(match.group(1))

    # Resolve macro-generated test names
    for macro_name, params, test_name_expr in macros:
        call_start_re = re.compile(
            r'(?<!\w)' + re.escape(macro_name) + r'\('
        )
        for filepath, content in all_content.items():
            if not filepath.endswith(CPP_EXTENSIONS):
                continue
            joined = re.sub(r"\\\s*\n", " ", content)
            for match in call_start_re.finditer(joined):
                line_start = joined.rfind("\n", 0, match.start())
                line = joined[line_start:match.start()]
                if "#define" in line:
                    continue
                # Extract balanced parentheses
                start = match.end()
                depth = 1
                i = start
                while i < len(joined) and depth > 0:
                    if joined[i] == '(':
                        depth += 1
                    elif joined[i] == ')':
                        depth -= 1
                    i += 1
                if depth != 0:
                    continue
                args_str = joined[start:i - 1]
                args = _split_top_level(args_str)
                if len(args) < len(params):
                    continue
                name = test_name_expr
                for j, param in enumerate(params):
                    if j < len(args):
                        name = name.replace(param, args[j].strip())
                name = name.replace("##", "")
                name = name.strip()
                if name.isidentifier():
                    test_names.add(name)

    return test_names


def is_unit_group(group):
    return group not in NON_UNIT_GROUPS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check that every Catch2 test case found in the source "
        "tree has a corresponding entry in the YAML configs.",
    )
    parser.add_argument(
        "configs_path",
        help="Path to the directory containing YAML config files.",
    )
    parser.add_argument(
        "source_root",
        help="Root directory of the Catch2 test source files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    configs_path = args.configs_path
    source_root = args.source_root

    missing = []
    stale = []

    for group, cases in iter_group_configs(configs_path):
        yaml_names = set(cases.keys())
        is_unit = is_unit_group(group)
        # Forward check: source tests missing from YAML (original HIP_TEST_CASE only)
        source_names = find_source_test_cases(
            source_root, group, is_unit=is_unit
        )
        for name in sorted(source_names - yaml_names):
            missing.append(f"  {group}/{name}")
        # Reverse check: YAML entries with no matching test in source
        # Uses broader detection (TEST_CASE, macros, chained macros)
        all_source_names = find_all_test_cases(
            source_root, group, is_unit=is_unit
        )
        for name in sorted(yaml_names - all_source_names):
            stale.append(f"  {group}/{name}")

    errors = False

    if missing:
        print(
            "ERROR: The following Catch2 test cases have no entry in their YAML config:",
            file=sys.stderr,
        )
        for entry in missing:
            print(entry, file=sys.stderr)
        errors = True

    if stale:
        print(
            "ERROR: The following YAML config entries have no matching Catch2 "
            "test case in source:",
            file=sys.stderr,
        )
        for entry in stale:
            print(entry, file=sys.stderr)
        errors = True

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
