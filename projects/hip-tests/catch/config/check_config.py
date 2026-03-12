import argparse
import sys

from common import iter_group_configs


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

    for group, cases in iter_group_configs(configs_path):
        for case_name, case_config in cases.items():
            if "level" not in case_config:
                missing.append(f"  {group}/{case_name}")

    if missing:
        print(
            "ERROR: The following test cases are missing a 'level' in their YAML config:",
            file=sys.stderr,
        )
        for entry in missing:
            print(entry, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
