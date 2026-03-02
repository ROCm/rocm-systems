import sys

from common import iter_group_configs


def main():
    if not len(sys.argv) == 2:
        raise ValueError("1 argument expected")

    config_path = sys.argv[1]

    missing = []

    for group, cases in iter_group_configs(config_path):
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
