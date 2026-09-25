# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import argparse
import re

import yaml

from common import iter_group_configs, load_definitions, parse_size_string


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
    parser.add_argument(
        "param_header_path",
        nargs="?",
        default=None,
        help="Optional output path for the generated parameter header file.",
    )
    parser.add_argument(
        "--asan",
        action="store_true",
        help="Target is an Address Sanitizer (ASAN) build. Test cases that "
        "list 'asan' in their 'disabled' field are skipped.",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Path to test_categories.yaml. When given, each test case gets a "
        "tag for every category whose levels it belongs to, and the category "
        "definitions are written into the parameter header.",
    )
    return parser.parse_args()


def load_categories(categories_path):
    """Load test_categories.yaml as an ordered {category: set(levels)} map."""
    if not categories_path:
        return {}

    with open(categories_path) as file:
        config = yaml.safe_load(file) or {}

    categories = {}
    for name, entry in (config.get("test_categories") or {}).items():
        levels = (entry or {}).get("test_levels")
        if levels:
            categories[name] = set(levels)
    return categories


def create_test_definition(
    group, case_name, case_config, platform, os_name, arch, categories, asan=False
):
    levels = case_config.get("level", [0, 1, 2])
    tags = case_config.get("tags", [])
    disabled = case_config.get("disabled", [])

    # Entries in 'disabled' that apply to this build:
    #   {platform}_{os_name}    amd_linux, nvidia_windows, ...
    #   {arch}                  any platform and OS
    #   {os_name}_{arch}        linux_<arch>, windows_<arch>
    #   asan                    ASAN builds only
    disable_keys = {f"{platform}_{os_name}", arch, f"{os_name}_{arch}"}
    if asan:
        disable_keys.add("asan")

    tags_str = ""

    for tag in tags:
        tags_str += f"[{tag}]"
    for level in levels:
        tags_str += f"[level_{level}]"

    # A category covers a test case when it covers any of its levels, so the
    # categories nest the way TheRock expects.
    level_set = set(levels)
    for category, category_levels in categories.items():
        if category_levels & level_set:
            tags_str += f"[{category}]"

    tags_str += f"[{group}]"

    if any(entry in disable_keys for entry in disabled):
        # Disabled for this build - see disable_keys above for the spellings
        # a 'disabled' entry may use.
        # Use the [disabled] tag (no leading dot) so it is visible in --list-tests
        # and included in a bare ./exe run.
        # CTest registers these as DISABLED TRUE (skipped by default).
        tags_str += f"[disabled][{arch}]"
    else:
        tags_str += f"[{arch}]"

    # Promote every disable reason to a distinct tag so a runner can query which
    # specific labels a test is disabled for (incl. OS labels dropped above).
    # Prefix is "exclude_" (not "disabled_") so it does not substring-match a
    # ctest -LE disabled filter; consumers should match anchored ^exclude_<entry>$.
    for entry in disabled:
        tags_str += f"[exclude_{entry}]"

    return f'#define {case_name} "{case_name}", "{tags_str}"'


def generate_parameter_header(cmd_options, categories, output_path):
    """Generate C++ header with compile-time parameter constants and test categories.

    Args:
        cmd_options: Dict of level_name -> parameters from definitions.yaml
        categories: Dict of category name -> set of levels from test_categories.yaml
        output_path: Path to write hip_test_parameters.hh
    """
    with open(output_path, 'w') as f:
        f.write("// Auto-generated from definitions.yaml\n")
        f.write("// DO NOT EDIT - This file is generated at build time\n")
        f.write("// Contains compile-time test parameters for each level\n\n")
        f.write("#pragma once\n\n")
        f.write("#include <array>\n")
        f.write("#include <vector>\n")
        f.write("#include <cstddef>\n")
        f.write("#include <cstdint>\n")
        f.write("#include <string>\n")
        f.write("#include <map>\n\n")
        
        f.write("namespace TestParameters {\n\n")
        
        levels_found = []
        
        for level_name, options in cmd_options.items():
            levels_found.append(level_name)
            f.write(f"// {'=' * 76}\n")
            f.write(f"// {level_name.upper()} PARAMETERS\n")
            f.write(f"// {'=' * 76}\n\n")
            
            # Memory sizes
            if "memory_sizes" in options:
                sizes = [parse_size_string(s) for s in options["memory_sizes"]]
                f.write(f"inline constexpr std::array<size_t, {len(sizes)}> {level_name}_memory_sizes = {{\n")
                f.write("    " + ",\n    ".join(str(s) for s in sizes) + "\n")
                f.write("};\n\n")
            
            # Block sizes
            if "block_sizes" in options:
                sizes = options["block_sizes"]
                f.write(f"inline constexpr std::array<int, {len(sizes)}> {level_name}_block_sizes = {{\n")
                f.write("    " + ", ".join(str(s) for s in sizes) + "\n")
                f.write("};\n\n")
            
            # Iterations
            if "iterations" in options:
                f.write(f"inline constexpr int {level_name}_iterations = {options['iterations']};\n\n")
            
            # Warmups
            if "warmups" in options:
                f.write(f"inline constexpr int {level_name}_warmups = {options['warmups']};\n\n")
            
            # CG iterations
            if "cg_iterations" in options:
                f.write(f"inline constexpr int {level_name}_cg_iterations = {options['cg_iterations']};\n\n")

            # Math accuracy iterations
            if "math_accuracy_iterations" in options:
                f.write(f"inline constexpr uint64_t {level_name}_math_accuracy_iterations = {options['math_accuracy_iterations']}ULL;\n\n")

            # Math accuracy max memory percentage
            if "math_accuracy_max_memory_percentage" in options:
                f.write(f"inline constexpr int {level_name}_math_accuracy_max_memory_percentage = {options['math_accuracy_max_memory_percentage']};\n\n")

            # Math max memory
            if "math_max_memory" in options:
                math_max_mem = parse_size_string(str(options["math_max_memory"]))
                f.write(f"inline constexpr size_t {level_name}_math_max_memory = {math_max_mem};\n\n")

            # Math reduction factor
            if "math_reduction_factor" in options:
                f.write(f"inline constexpr double {level_name}_math_reduction_factor = {options['math_reduction_factor']};\n\n")
        
        # Generate LevelParameters struct and initialization function
        f.write(f"// {'=' * 76}\n")
        f.write("// LEVEL REGISTRY - Maps level names to their parameters\n")
        f.write(f"// {'=' * 76}\n\n")
        
        f.write("struct LevelParameters {\n")
        f.write("    std::vector<size_t> memory_sizes;\n")
        f.write("    std::vector<int> block_sizes;\n")
        f.write("    int iterations = 0;\n")
        f.write("    int warmups = 0;\n")
        f.write("    int cg_iterations = 0;\n")
        f.write("    uint64_t math_accuracy_iterations = 0;\n")
        f.write("    int math_accuracy_max_memory_percentage = 0;\n")
        f.write("    size_t math_max_memory = 0;\n")
        f.write("    double math_reduction_factor = 0.0;\n")
        f.write("};\n\n")
        
        f.write("inline std::map<std::string, LevelParameters> initializeLevelParameters() {\n")
        f.write("    std::map<std::string, LevelParameters> params;\n\n")
        
        for level_name in levels_found:
            f.write(f"    // {level_name}\n")
            f.write(f"    params[\"{level_name}\"] = {{\n")
            f.write(f"        std::vector<size_t>({level_name}_memory_sizes.begin(), "
                    f"{level_name}_memory_sizes.end()),\n")
            f.write(f"        std::vector<int>({level_name}_block_sizes.begin(), "
                    f"{level_name}_block_sizes.end()),\n")
            f.write(f"        {level_name}_iterations,\n")
            f.write(f"        {level_name}_warmups,\n")
            f.write(f"        {level_name}_cg_iterations,\n")
            f.write(f"        {level_name}_math_accuracy_iterations,\n")
            f.write(f"        {level_name}_math_accuracy_max_memory_percentage,\n")
            f.write(f"        {level_name}_math_max_memory,\n")
            f.write(f"        {level_name}_math_reduction_factor\n")
            f.write(f"    }};\n\n")
        
        f.write("    return params;\n")
        f.write("}\n\n")

        f.write("} // namespace TestParameters\n\n")

        # Test categories
        def category_identifier(name):
            return re.sub(r"[^0-9a-zA-Z_]", "_", name) + "_levels"

        f.write(f"// {'=' * 76}\n")
        f.write("// TEST CATEGORIES - from test_categories.yaml\n")
        f.write(f"// {'=' * 76}\n\n")
        f.write("namespace TestCategories {\n\n")

        for name, levels in categories.items():
            values = ", ".join(str(level) for level in sorted(levels))
            f.write(f"inline constexpr std::array<int, {len(levels)}> "
                    f"{category_identifier(name)} = {{{values}}};\n")
        f.write("\n")

        f.write("inline std::map<std::string, std::vector<int>> initializeTestCategories() {\n")
        f.write("    std::map<std::string, std::vector<int>> categories;\n\n")
        for name in categories:
            f.write(f'    categories["{name}"] = std::vector<int>('
                    f"{category_identifier(name)}.begin(), "
                    f"{category_identifier(name)}.end());\n")
        f.write("\n    return categories;\n")
        f.write("}\n\n")
        f.write("} // namespace TestCategories\n")

    print(f"[parse_config] Generated parameter header: {output_path}")
    print(f"[parse_config]   Levels defined: {', '.join(levels_found)}")
    print(f"[parse_config]   Categories defined: {', '.join(categories) or 'none'}")


def main():
    args = parse_args()

    configs_path = args.configs_path
    platform = args.platform
    os_name = args.os_name
    arch = args.arch
    header_path = args.header_path
    param_header_path = args.param_header_path
    asan = args.asan

    categories = load_categories(args.categories)

    test_macros = []

    for group, cases in iter_group_configs(configs_path):
        for case_name, case_config in cases.items():
            test_macros.append(
                create_test_definition(
                    group, case_name, case_config, platform, os_name, arch,
                    categories, asan
                )
            )

    with open(header_path, "w") as file:
        file.write("// Auto-generated from YAML config files\n")
        file.write("// DO NOT EDIT - This file is generated at build time\n\n")
        for test_macro in test_macros:
            file.write(test_macro)
            file.write("\n")
    
    print(f"[parse_config] Generated test definitions: {header_path}")
    print(f"[parse_config]   Test cases: {len(test_macros)}")

    # Generate parameter header if path provided
    if param_header_path:
        definitions = load_definitions(configs_path)
        cmd_options = definitions.get("cmd_options", {})
        if cmd_options:
            generate_parameter_header(cmd_options, categories, param_header_path)
        else:
            print("[parse_config] Warning: No cmd_options found in definitions.yaml")


if __name__ == "__main__":
    main()
