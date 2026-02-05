
import sys
import yaml


def parse_size_string(size_str):
    """Convert size string (1K, 1M, 1G) to bytes"""
    # Handle integers directly (YAML may parse numbers as int)
    if isinstance(size_str, int):
        return size_str
    
    size_str = str(size_str).strip().upper()
    multipliers = {'K': 1024, 'M': 1024*1024, 'G': 1024*1024*1024}
    
    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            return int(float(size_str[:-1]) * multiplier)
    return int(size_str)


def create_test_definition(group, case_name, case_config, platform, os_name, arch):
    """Generate test macro with tags"""
    level = case_config.get("level", 2)
    tags = case_config.get("tags", [])
    disabled = case_config.get("disabled", [])

    tags_str = ""
    for tag in tags:
        tags_str += f"[{tag}]"
    tags_str += f"[level_{level}]"
    tags_str += f"[{group}]"

    if f"{platform}_{os_name}" in disabled or arch in disabled:
        return f'#define {case_name} "{case_name}", "[.]"'
    
    return f'#define {case_name} "{case_name}", "{tags_str}"'


def generate_test_definitions_header(config, platform, os_name, arch, output_path):
    """Generate header with test case macros"""
    test_definitions = []
    
    # Process unit tests
    for group, cases in config.get("unit", {}).items():
        for case_name, case_config in cases.items():
            test_definitions.append(
                create_test_definition(group, case_name, case_config, platform, os_name, arch)
            )
    
    # Process non-unit test groups
    non_unit_groups = ["ABM", "multiproc", "performance", "perftests", "stress", "TypeQualifiers"]
    for group in non_unit_groups:
        if group in config:
            for case_name, case_config in config[group].items():
                test_definitions.append(
                    create_test_definition(group, case_name, case_config, platform, os_name, arch)
                )
    
    # Write header file
    with open(output_path, "w") as f:
        f.write("// Auto-generated from hip_tests_config.yaml\n")
        f.write("// DO NOT EDIT - This file is generated at build time\n\n")
        for test_definition in test_definitions:
            f.write(test_definition)
            f.write("\n")
    
    print(f"[parse_config] Generated test definitions: {output_path}")
    print(f"[parse_config]   Test cases: {len(test_definitions)}")


def generate_parameter_header(config, output_path):
    """Generate C++ header with compile-time parameter constants"""
    
    with open(output_path, 'w') as f:
        f.write("// Auto-generated from hip_tests_config.yaml\n")
        f.write("// DO NOT EDIT - This file is generated at build time\n")
        f.write("// Contains compile-time test parameters for each level\n\n")
        f.write("#pragma once\n\n")
        f.write("#include <vector>\n")
        f.write("#include <cstddef>\n")
        f.write("#include <string>\n")
        f.write("#include <map>\n\n")
        
        f.write("namespace TestParameters {\n\n")
        
        cmd_options = config.get("cmd_options", {})
        levels_found = []
        
        for level_name, options in cmd_options.items():
            levels_found.append(level_name)
            f.write(f"// ============================================================================\n")
            f.write(f"// {level_name.upper()} PARAMETERS\n")
            f.write(f"// ============================================================================\n\n")
            
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
        
        # Generate initialization map
        f.write("// ============================================================================\n")
        f.write("// LEVEL REGISTRY - Maps level names to their parameters\n")
        f.write("// ============================================================================\n\n")
        
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


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: parse_config.py <yaml_path> <platform> <os> <arch> <test_defs_header> <param_header>")
        sys.exit(1)

    config_path = sys.argv[1]
    platform = sys.argv[2]
    os_name = sys.argv[3]
    arch = sys.argv[4]
    test_defs_header_path = sys.argv[5]
    param_header_path = sys.argv[6]

    print(f"[parse_config] Loading YAML config: {config_path}")
    print(f"[parse_config] Platform: {platform}, OS: {os_name}, Arch: {arch}")
    
    # Load YAML config
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Generate test definitions header
    generate_test_definitions_header(config, platform, os_name, arch, test_defs_header_path)
    
    # Generate parameter header
    generate_parameter_header(config, param_header_path)
    
    print("[parse_config] Code generation complete!")

