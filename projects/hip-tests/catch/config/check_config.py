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
                value = case_config[field]
                if isinstance(value, list):
                    # Legacy flat list of tokens: accept. Reasons are deferred to
                    # the broader migration; do not force list entries to carry one.
                    continue
                if isinstance(value, dict):
                    # Structured form: each token maps to {Reason: [...]}. Accept it
                    # as valid, but do NOT enforce that tokens carry a non-empty
                    # Reason yet. The ~1,100 existing configs have not been migrated
                    # to the reason-bearing form, so Reason enforcement is
                    # intentionally deferred to a later phase (AIRUNTIME-2744):
                    # https://amd-hub.atlassian.net/browse/AIRUNTIME-2744
                    continue
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
            f"[check_config] {ERROR}ERROR: The following test cases have a 'disabled'/'unsupported' field that is neither a YAML list nor a mapping:{RESET}",
            file=sys.stderr,
        )
        for entry in invalid_skip_fields:
            print(f"[check_config] {ERROR}{entry}{RESET}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
