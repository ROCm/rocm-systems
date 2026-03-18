# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import argparse
import os
import re
import sys

from common import NON_UNIT_GROUPS, iter_group_configs

CPP_EXTENSIONS = (".cc", ".cpp", ".cxx", ".c++", ".C", ".cp", ".CPP")


def find_source_test_cases(source_root, group, is_unit):
    if is_unit:
        source_dir = os.path.join(source_root, "unit", group)
    else:
        source_dir = os.path.join(source_root, group)

    if not os.path.isdir(source_dir):
        return set()

    test_names = set()
    pattern = re.compile(
        r'(?:TEST_CASE|TEMPLATE_TEST_CASE)\(\s*(?:"([^"]+)"|([A-Za-z_]\w*))\s*[,)]'
    )
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if not filename.endswith(CPP_EXTENSIONS):
                continue
            filepath = os.path.join(root, filename)
            with open(filepath, errors="replace") as f:
                content = f.read()
            for match in pattern.finditer(content):
                test_names.add(match.group(1) or match.group(2))

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

    for group, cases in iter_group_configs(configs_path):
        yaml_names = set(cases.keys())
        source_names = find_source_test_cases(
            source_root, group, is_unit=is_unit_group(group)
        )
        for name in sorted(source_names - yaml_names):
            missing.append(f"  {group}/{name}")

    if missing:
        print(
            "ERROR: The following Catch2 test cases have no entry in their YAML config:",
            file=sys.stderr,
        )
        for entry in missing:
            print(entry, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
