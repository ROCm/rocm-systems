import sys

import yaml


def create_test_definition(group, case_name, case_config, platform, os, arch):
    level = case_config.get("level", 2)
    tags = case_config.get("tags", [])
    disabled = case_config.get("disabled", [])

    tags_str = ""

    for tag in tags:
        tags_str += f"[{tag}]"
    tags_str += f"[level_{level}]"
    tags_str += f"[{group}]"

    if f"{platform}_{os}" in disabled or arch in disabled:
        # skip case
        tags_str = "[.]"

    return f'#define {case_name} "{case_name}", "{tags_str}"'


if not len(sys.argv) == 6:
    raise ValueError("only 5 arguments expected")

config_path = sys.argv[1]
platform = sys.argv[2]
os = sys.argv[3]
arch = sys.argv[4]
header_path = sys.argv[5]

with open(config_path) as file:
    config = yaml.safe_load(file)

test_definitions = []

for group, cases in config["tests"].items():
    for case_name, case_config in cases.items():
        test_definitions.append(
            create_test_definition(group, case_name, case_config, platform, os, arch)
        )

with open(header_path, "w") as file:
    for test_definition in test_definitions:
        file.write(test_definition)
        file.write("\n")
