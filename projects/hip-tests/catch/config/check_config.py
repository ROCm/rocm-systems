# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import argparse
import sys

from common import iter_group_configs, ERROR, RESET


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check that every test case in the YAML configs has a "
        "'level' field defined.",
    )
    parser.add_argument(
        "configs_path",
        help="Path to the directory containing YAML config files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    configs_path = args.configs_path

    missing = []
    invalid_skip_fields = []

    for group, cases in iter_group_configs(configs_path):
        for case_name, case_config in cases.items():
            if "level" not in case_config:
                missing.append(f"  {group}/{case_name}")
            for field in ("disabled", "unsupported"):
                if field not in case_config:
                    continue
                # Skip fields must be flat YAML lists of tokens. A scalar or
                # mapping would break tag generation (a string is iterated
                # char-by-char; a mapping fails the list concatenation in
                # parse_config). The optional sibling 'reason' scalar is metadata
                # and is intentionally NOT enforced yet — reason enforcement is
                # deferred to a later phase (AIRUNTIME-2744):
                # https://amd-hub.atlassian.net/browse/AIRUNTIME-2744
                if not isinstance(case_config[field], list):
                    invalid_skip_fields.append(f"  {group}/{case_name}: '{field}'")

    if missing:
        print(
            f"[check_config] {ERROR}ERROR: The following test cases are missing a 'level' in their YAML config:{RESET}",
            file=sys.stderr,
        )
        for entry in missing:
            print(f"[check_config] {ERROR}{entry}{RESET}", file=sys.stderr)
        sys.exit(1)

    if invalid_skip_fields:
        print(
            f"[check_config] {ERROR}ERROR: The following test cases have a 'disabled'/'unsupported' field that is not a YAML list:{RESET}",
            file=sys.stderr,
        )
        for entry in invalid_skip_fields:
            print(f"[check_config] {ERROR}{entry}{RESET}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
