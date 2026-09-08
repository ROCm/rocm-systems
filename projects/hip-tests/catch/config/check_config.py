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
    bad_type = []
    bad_reason = []

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
                    # Structured form: each token maps to {Reason: [...]} where the
                    # Reason must be a non-empty YAML flow list (mirrors the
                    # compute-utils blacklist idiom). Bare scalars are rejected.
                    for token, meta in value.items():
                        reason = meta.get("Reason") if isinstance(meta, dict) else None
                        if not isinstance(reason, list) or not reason:
                            bad_reason.append(
                                f"  {group}/{case_name}: '{field}' token '{token}' "
                                "missing non-empty bracketed Reason"
                            )
                    continue
                bad_type.append(f"  {group}/{case_name}: '{field}'")

    if missing:
        print(
            f"[check_config] {ERROR}ERROR: The following test cases are missing a 'level' in their YAML config:{RESET}",
            file=sys.stderr,
        )
        for entry in missing:
            print(f"[check_config] {ERROR}{entry}{RESET}", file=sys.stderr)
        sys.exit(1)

    if bad_type:
        print(
            f"[check_config] {ERROR}ERROR: The following test cases have a 'disabled'/'unsupported' field that is neither a YAML list nor a mapping:{RESET}",
            file=sys.stderr,
        )
        for entry in bad_type:
            print(f"[check_config] {ERROR}{entry}{RESET}", file=sys.stderr)
        sys.exit(1)

    if bad_reason:
        print(
            f"[check_config] {ERROR}ERROR: The following structured 'disabled'/'unsupported' tokens are missing a non-empty bracketed Reason:{RESET}",
            file=sys.stderr,
        )
        for entry in bad_reason:
            print(f"[check_config] {ERROR}{entry}{RESET}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
