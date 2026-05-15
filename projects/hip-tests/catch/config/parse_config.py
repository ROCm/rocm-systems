# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import argparse

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


def generate_parameter_header(cmd_options, output_path):
    """Generate C++ header with compile-time parameter constants.
    
    Args:
        cmd_options: Dict of level_name -> parameters from definitions.yaml
        output_path: Path to write hip_test_parameters.hh
    """
    with open(output_path, 'w') as f:
        f.write("// Auto-generated from definitions.yaml\n")
        f.write("// DO NOT EDIT - This file is generated at build time\n")
        f.write("// Contains compile-time test parameters for each level\n\n")
        f.write("#pragma once\n\n")
        f.write("#include <vector>\n")
        f.write("#include <cstddef>\n")
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
                f.write(f"inline const std::vector<size_t> {level_name}_memory_sizes = {{\n")
                f.write("    " + ",\n    ".join(str(s) for s in sizes) + "\n")
                f.write("};\n\n")
            
            # Block sizes
            if "block_sizes" in options:
                sizes = options["block_sizes"]
                f.write(f"inline const std::vector<int> {level_name}_block_sizes = {{\n")
                f.write("    " + ", ".join(str(s) for s in sizes) + "\n")
                f.write("};\n\n")
            
            # Iterations
            if "iterations" in options:
                f.write(f"inline const int {level_name}_iterations = {options['iterations']};\n\n")
            
            # Warmups
            if "warmups" in options:
                f.write(f"inline const int {level_name}_warmups = {options['warmups']};\n\n")
            
            # Max memory
            if "max_memory" in options:
                max_mem = parse_size_string(str(options["max_memory"]))
                f.write(f"inline const size_t {level_name}_max_memory = {max_mem};\n\n")
            
            # Reduction factor
            if "reduction_factor" in options:
                f.write(f"inline const double {level_name}_reduction_factor = {options['reduction_factor']};\n\n")
        
        # Generate LevelParameters struct and initialization function
        f.write(f"// {'=' * 76}\n")
        f.write("// LEVEL REGISTRY - Maps level names to their parameters\n")
        f.write(f"// {'=' * 76}\n\n")
        
        f.write("struct LevelParameters {\n")
        f.write("    std::vector<size_t> memory_sizes;\n")
        f.write("    std::vector<int> block_sizes;\n")
        f.write("    int iterations = 0;\n")
        f.write("    int warmups = 0;\n")
        f.write("    size_t max_memory = 0;\n")
        f.write("    double reduction_factor = 0.0;\n")
        f.write("};\n\n")
        
        f.write("inline std::map<std::string, LevelParameters> initializeLevelParameters() {\n")
        f.write("    std::map<std::string, LevelParameters> params;\n\n")
        
        for level_name in levels_found:
            f.write(f"    // {level_name}\n")
            f.write(f"    params[\"{level_name}\"] = {{\n")
            f.write(f"        {level_name}_memory_sizes,\n")
            f.write(f"        {level_name}_block_sizes,\n")
            f.write(f"        {level_name}_iterations,\n")
            f.write(f"        {level_name}_warmups,\n")
            f.write(f"        {level_name}_max_memory,\n")
            f.write(f"        {level_name}_reduction_factor\n")
            f.write(f"    }};\n\n")
        
        f.write("    return params;\n")
        f.write("}\n\n")
        
        f.write("} // namespace TestParameters\n")
    
    print(f"[parse_config] Generated parameter header: {output_path}")
    print(f"[parse_config]   Levels defined: {', '.join(levels_found)}")


def main():
    args = parse_args()

    configs_path = args.configs_path
    platform = args.platform
    os_name = args.os_name
    arch = args.arch
    header_path = args.header_path
    param_header_path = args.param_header_path

    test_macros = []

    for group, cases in iter_group_configs(configs_path):
        for case_name, case_config in cases.items():
            test_macros.append(
                create_test_definition(
                    group, case_name, case_config, platform, os_name, arch
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
            generate_parameter_header(cmd_options, param_header_path)
        else:
            print("[parse_config] Warning: No cmd_options found in definitions.yaml")


if __name__ == "__main__":
    main()
