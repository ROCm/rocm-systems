# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import argparse
import re
import sys

from common import iter_group_configs, ERROR, RESET


JIRA_RE = re.compile(r"AIRUNTIME-\d+")


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
    missing_reasons = []
    missing_jira = []

    for group, cases in iter_group_configs(configs_path):
        for case_name, case_config in cases.items():
            if "level" not in case_config:
                missing.append(f"  {group}/{case_name}")
            for field in ("disabled", "unsupported"):
                if field not in case_config:
                    continue
                value = case_config[field]
                # A skip field is either a flat list of targets, or a mapping
                # with a 'targets' list plus a 'reason' scalar. We guard that the
                # targets are a list, since a scalar breaks tag generation (a
                # string is iterated char-by-char) and a malformed mapping fails
                # the parser's list concatenation.
                if isinstance(value, list):
                    targets = value
                    reason = ""
                elif isinstance(value, dict) and isinstance(value.get("targets"), list):
                    targets = value["targets"]
                    reason = value.get("reason", "")
                else:
                    invalid_skip_fields.append(f"  {group}/{case_name}: '{field}'")
                    continue
                # A populated skip section (non-empty targets) must record a
                # non-empty string 'reason'; a non-scalar (list/mapping) or
                # non-string reason is treated as missing. An empty section
                # (targets: []) is a no-op and needs no reason.
                if targets and not (isinstance(reason, str) and reason.strip()):
                    missing_reasons.append(f"  {group}/{case_name}: '{field}'")
                # A populated skip section with a real reason must cite a Jira
                # reference (AIRUNTIME-<number>); the placeholder AIRUNTIME-XXXX
                # has no digits and therefore does not count as a real ref.
                elif targets and not JIRA_RE.search(reason):
                    missing_jira.append(f"  {group}/{case_name}: '{field}'")

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
            f"[check_config] {ERROR}ERROR: The following test cases have a 'disabled'/'unsupported' field that is neither a YAML list nor a mapping with a 'targets' list:{RESET}",
            file=sys.stderr,
        )
        for entry in invalid_skip_fields:
            print(f"[check_config] {ERROR}{entry}{RESET}", file=sys.stderr)
        sys.exit(1)

    if missing_reasons:
        print(
            f"[check_config] {ERROR}ERROR: The following test cases have a 'disabled'/'unsupported' field with targets but no 'reason'. Use the mapping form ({{targets: [...], reason: \"...\"}}) and record why the case is skipped:{RESET}",
            file=sys.stderr,
        )
        for entry in missing_reasons:
            print(f"[check_config] {ERROR}{entry}{RESET}", file=sys.stderr)
        sys.exit(1)

    if missing_jira:
        print(
            f"[check_config] {ERROR}ERROR: The following test cases have a 'disabled'/'unsupported' 'reason' with no real Jira reference. The reason must contain a real AIRUNTIME-<number> (the placeholder AIRUNTIME-XXXX does not count):{RESET}",
            file=sys.stderr,
        )
        for entry in missing_jira:
            print(f"[check_config] {ERROR}{entry}{RESET}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
