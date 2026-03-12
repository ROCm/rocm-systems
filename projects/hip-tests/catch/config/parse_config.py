import argparse

from common import iter_group_configs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse YAML test configs and generate a C/C++ header with "
        "Catch2 TEST_CASE macro definitions.",
    )
    parser.add_argument(
        "configs_path",
        help="Path to the directory containing YAML config files.",
    )
    parser.add_argument(
        "platform",
        help="Target platform (e.g. amd, nvidia).",
    )
    parser.add_argument(
        "os_name",
        help="Target operating system (e.g. linux, windows).",
    )
    parser.add_argument(
        "arch",
        help="Target architecture (e.g. gfx90a, gfx942).",
    )
    parser.add_argument(
        "header_path",
        help="Output path for the generated header file.",
    )
    return parser.parse_args()


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
    args = parse_args()

    configs_path = args.configs_path
    platform = args.platform
    os_name = args.os_name
    arch = args.arch
    header_path = args.header_path

    test_macros = []

    for group, cases in iter_group_configs(configs_path):
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
