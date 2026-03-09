import sys

from common import iter_group_configs


def print_tests_list(tests):
        for entry in tests:
            print(entry, file=sys.stderr)


def main():
    if not len(sys.argv) == 2:
        raise ValueError("1 argument expected")

    config_path = sys.argv[1]

    missing_level = []
    missing_tracker = []

    for group, cases in iter_group_configs(config_path):
        for case_name, case_config in cases.items():
            if "level" not in case_config:
                missing_level.append(f"  {group}/{case_name}")
            if "disabled" in case_config and "tracker" not in case_config:
                missing_tracker.append(f"  {group}/{case_name}")

    if missing_level:
        print(
            "ERROR: The following test cases are missing a 'level' in their YAML config:",
            file=sys.stderr,
        )
        print_tests_list(missing_level)

    if missing_tracker:
        print(
            "ERROR: The following test cases are disabled, but lack a 'tracker' in their YAML config:",
            file=sys.stderr,
        )
        print_tests_list(missing_tracker)

    if missing_level or missing_tracker:
        sys.exit(1)


if __name__ == "__main__":
    main()
