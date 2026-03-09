import sys
import re

from common import iter_group_configs


def report(message, bad_tests):
    if len(bad_tests) == 0:
        return 0
    print(f"ERROR: The following test cases {message}:", file=sys.stderr)
    for entry in bad_tests:
        print(entry, file=sys.stderr)
    return 1


def check_tracker_format(tracker_str):
    jira_format = r'[A-Z]+\d+'  # PROJECT-123
    gh_format = r'[\w\d-]+/[\w\d-]+#\d+'  # org/repo#123
    if re.match(jira_format, tracker_str) is None and re.match(gh_format, tracker_str) is None:
        return False
    return True


def main():
    if not len(sys.argv) == 2:
        raise ValueError("1 argument expected")

    config_path = sys.argv[1]

    missing_level = []
    missing_tracker = []
    invalid_tracker_format = []


    for group, cases in iter_group_configs(config_path):
        for case_name, case_config in cases.items():
            test = f"  {group}/{case_name}"

            if "level" not in case_config:
                missing_level.append(test)
            if "disabled" in case_config:
                if "tracker" not in case_config:
                    missing_tracker.append(test)
                elif not check_tracker_format(case_config["tracker"]):
                    invalid_tracker_format.append(test)

    errors = report("are missing a 'level' in their YAML config", missing_level)
    errors += report("are disabled, but lack a 'tracker' in their YAML config", missing_tracker)
    errors += report("have 'tracker' in incorrect format, it should be ([A-Z]+\d+)|([\w\d-]+/[\w\d-]+#\d+)", invalid_tracker_format)

    if errors != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
