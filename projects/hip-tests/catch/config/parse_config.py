import sys

from common import iter_group_configs


def create_test_definition(group, case_name, case_config, platform, os_name, arch):
    level = case_config.get("level", 2)
    tags = case_config.get("tags", [])
    disabled = case_config.get("disabled", [])

    tags_str = ""

    for tag in tags:
        tags_str += f"[{tag}]"
    tags_str += f"[level_{level}]"
    tags_str += f"[{group}]"

    if f"{platform}_{os_name}" in disabled or arch in disabled:
        # skip case
        tags_str = "[.]"

    return f'#define {case_name} "{case_name}", "{tags_str}"'


def main():
    if not len(sys.argv) == 6:
        raise ValueError("5 arguments expected")

    config_path = sys.argv[1]
    platform = sys.argv[2]
    os_name = sys.argv[3]
    arch = sys.argv[4]
    header_path = sys.argv[5]

    test_macros = []

    for group, cases in iter_group_configs(config_path):
        for case_name, case_config in cases.items():
            test_macros.append(
                create_test_definition(
                    group, case_name, case_config, platform, os_name, arch
                )
            )

    with open(header_path, "w") as file:
        for test_macro in test_macros:
            file.write(test_macro)
            file.write("\n")


if __name__ == "__main__":
    main()
