import yaml
import sys
import re
import platform
import argparse
import contextlib


def gpu_arch_matches(specific_arch, pattern_arch):
    """
    Check if a specific GPU architecture matches a pattern with X wildcards.
    E.g., gfx1150 matches gfx1150 (exact), gfx115X, gfx11X, etc.
    """
    if specific_arch == pattern_arch:
        return True

    # Check if pattern_arch has X wildcards
    if "X" not in pattern_arch:
        return False

    # Split at the first X and check if specific_arch starts with the prefix
    prefix = pattern_arch.split("X")[0]
    return specific_arch.startswith(prefix)


def load_yaml(yaml_file):
    """Load and parse a YAML file, exiting with a descriptive error on failure."""
    try:
        with open(yaml_file, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: YAML file not found: {yaml_file}", file=sys.stderr)
    except PermissionError:
        print(
            f"Error: Permission denied reading YAML file: {yaml_file}", file=sys.stderr
        )
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML syntax in {yaml_file}: {e}", file=sys.stderr)
    except Exception as e:
        print(
            f"Error: Unexpected failure loading {yaml_file}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Parse test_categories.yaml and generate CMake test definitions"
    )
    parser.add_argument("yaml_file", help="Path to the test_categories.yaml file")
    parser.add_argument(
        "target_name", help="Name of the test target (e.g., miopen_gtest)"
    )
    parser.add_argument("working_dir", help="Working directory for running tests")
    parser.add_argument(
        "install_test_file",
        nargs="?",
        default=None,
        help="Optional: Path to write install-time test definitions with relative paths",
    )

    args = parser.parse_args()

    yaml_file = args.yaml_file
    target_name = args.target_name
    working_dir = args.working_dir
    install_test_file = args.install_test_file

    config = load_yaml(yaml_file)

    # Open install test file if provided, using context manager for automatic cleanup
    try:
        install_cm = (
            open(install_test_file, "a", buffering=1)
            if install_test_file
            else contextlib.nullcontext()
        )
    except OSError as e:
        print(
            f"Warning: I/O error opening install test file {install_test_file}: {e}",
            file=sys.stderr,
        )
        install_cm = contextlib.nullcontext()
    except Exception as e:
        print(
            f"Warning: Unexpected error opening install test file {install_test_file}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        install_cm = contextlib.nullcontext()

    with install_cm as install_file_handle:
        # ===============================================================================================================
        # Parse the YAML file, add excludes (including OS-specific), and write tests to CMake and install file.
        # ===============================================================================================================
        categories = config.get("test_categories", {})
        execution_settings = config.get("execution_settings", {})
        timeouts = execution_settings.get("category_timeouts", {})
        timeout_multiplier = execution_settings.get("timeout_multiplier", 1)
        env_dict = execution_settings.get("environment", {}) or {}
        env_string = (
            ";".join(f"{k}={v}" for k, v in env_dict.items()) if env_dict else None
        )
        exclude_gpu_config = config.get("exclude_gpu", {})

        # Detect OS
        is_windows = platform.system() == "Windows"
        is_linux = platform.system() == "Linux"

        print("# Generated CMake code for test categories")
        print(f"# Detected OS: {platform.system()}")
        print(f"# Timeout multiplier: {timeout_multiplier}")

        # Store category information for later use with GPU exclusions
        category_data = {}

        for category_name, category_info in categories.items():
            patterns = category_info.get("test_patterns", [])
            if not patterns:
                print(
                    f"Warning: Category '{category_name}' has no test_patterns defined, skipping.",
                    file=sys.stderr,
                )
                continue
            labels = category_info.get("labels", [])
            exclude = category_info.get("exclude", [])
            if exclude is None:
                exclude = []

            # Add OS-specific exclusions
            if is_windows:
                exclude_windows = category_info.get("exclude_windows", [])
                if exclude_windows:
                    exclude.extend(exclude_windows)

            if is_linux:
                exclude_linux = category_info.get("exclude_linux", [])
                if exclude_linux:
                    exclude.extend(exclude_linux)

            base_timeout = timeouts.get(category_name, 300)
            timeout = int(base_timeout * timeout_multiplier)
            print(f"# Category: {category_name}")
            print(f'# Description: {category_info.get("description", "")}')

            # Build positive pattern string and exclude string
            positive_string = ":".join(patterns)
            exclude_string = ":".join(exclude) if exclude else ""

            # Store positive and exclude strings separately for GPU exclusion processing
            category_data[category_name] = {
                "positive_string": positive_string,
                "exclude_string": exclude_string,
                "labels": labels[:],  # Make a copy
                "timeout": timeout,
            }

            # Build complete pattern string for this category test
            if exclude_string:
                pattern_string = positive_string + "-" + exclude_string
            else:
                pattern_string = positive_string

            label_string = '"' + ";".join(labels) + '"'

            # =======================================================================
            # Write category test to CMake file and install file.
            # =======================================================================
            print("add_test(")
            print(f"  NAME {target_name}_{category_name}_suite")
            print(f"  COMMAND {target_name} --gtest_filter={pattern_string}")
            print(f"  WORKING_DIRECTORY {working_dir}")
            print(")")

            print(
                f"set_tests_properties({target_name}_{category_name}_suite PROPERTIES"
            )
            print(f"  LABELS {label_string}")
            print(f"  TIMEOUT {timeout}")
            if env_string:
                print(f'  ENVIRONMENT "{env_string}"')
            print(")")
            print()

            # Write install-time test with relative path if install file is provided
            if install_file_handle:
                try:
                    install_file_handle.write(
                        f'add_test({target_name}_{category_name}_suite "../{target_name}" --gtest_filter={pattern_string})\n'
                    )
                    env_prop = f' ENVIRONMENT "{env_string}"' if env_string else ""
                    install_file_handle.write(
                        f"set_tests_properties({target_name}_{category_name}_suite PROPERTIES LABELS {label_string} TIMEOUT {timeout}{env_prop})\n\n"
                    )
                    install_file_handle.flush()
                except OSError as e:
                    print(
                        f"Warning: I/O error writing category {category_name} to install test file: {e}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"Warning: Unexpected error writing category {category_name} to install test file: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )

        # ========================================================================
        # GPU Exclusion Tests with Hierarchical Pattern Matching
        # ========================================================================
        #
        # This section generates GPU-specific exclusion tests

        # - Uses wildcard 'X' for pattern matching (e.g., gfx11X matches gfx1100, gfx1150, etc.)

        # - Generates one test per category (quick, standard, etc.) per unique ex_gpu_* label
        # - Test name format: {target_name}_{category}_{gpu_arch}_suite
        # - Uses gtest filter: "{category_patterns}:-{gpu_exclusion_patterns}"
        # ========================================================================

        # Collect all ex_gpu labels and their corresponding GPU architectures

        ex_gpu_labels_to_process = set()
        for gpu_key, gpu_config in exclude_gpu_config.items():
            match = re.match(r"exclude_gpu_(gfx\w+)", gpu_key)
            if match:
                gpu_labels = gpu_config.get("labels", [])
                for label in gpu_labels:
                    if label.startswith("ex_gpu_"):
                        ex_gpu_labels_to_process.add(label)

        # For each unique ex_gpu label, create tests with hierarchical pattern matching
        # Sort to ensure consistent test order
        for ex_gpu_label in sorted(ex_gpu_labels_to_process):
            # Extract the GPU architecture from the label (e.g., ex_gpu_gfx1150 -> gfx1150)
            gpu_arch = ex_gpu_label.replace("ex_gpu_", "")

            # Collect all patterns that apply to this GPU architecture
            # This includes exact matches and hierarchical matches (e.g., gfx1150 matches gfx115X, gfx11X)
            all_applicable_patterns = []
            all_applicable_categories = set()

            for gpu_key, gpu_config in exclude_gpu_config.items():
                match = re.match(r"exclude_gpu_(gfx\w+)", gpu_key)
                if not match:
                    continue

                config_arch = match.group(1)

                # Check if this config applies to our target GPU architecture
                if gpu_arch_matches(gpu_arch, config_arch):
                    patterns = gpu_config.get("test_patterns", [])
                    if patterns:
                        for p in patterns:
                            if isinstance(p, list):
                                all_applicable_patterns.extend(p)
                            else:
                                all_applicable_patterns.append(p)

                    # Collect applicable categories from this config
                    gpu_labels = gpu_config.get("labels", [])
                    for label in gpu_labels:
                        if label in category_data:
                            all_applicable_categories.add(label)

            if not all_applicable_patterns:
                continue

            # Remove duplicates from all_applicable_patterns while preserving order
            seen = set()
            unique_patterns = []
            for pattern in all_applicable_patterns:
                if pattern not in seen:
                    seen.add(pattern)
                    unique_patterns.append(pattern)

            # Build GPU exclusion pattern string - format: pattern1:pattern2
            gpu_exclude_string = ":".join(unique_patterns)

            # Create one test for each applicable category
            for category_name in all_applicable_categories:
                cat_data = category_data[category_name]
                positive_string = cat_data["positive_string"]
                cat_exclude_string = cat_data["exclude_string"]
                cat_labels = cat_data["labels"]
                timeout = cat_data["timeout"]

                # Build combined pattern string: positive - category_excludes:gpu_excludes
                combined_exclude_string = ""
                if cat_exclude_string:
                    combined_exclude_string = (
                        cat_exclude_string + ":" + gpu_exclude_string
                    )
                else:
                    combined_exclude_string = gpu_exclude_string

                pattern_string = positive_string + "-" + combined_exclude_string

                # Build label string: category_labels + ex_gpu_<arch> label
                combined_labels = cat_labels + [ex_gpu_label]
                label_string = '"' + ";".join(combined_labels) + '"'

                # =======================================================================
                # Write GPU exclusion tests to CMake file and install file.
                # =======================================================================
                print(f"# GPU exclusion for {gpu_arch} - {category_name} category")
                print("add_test(")
                print(f"  NAME {target_name}_{category_name}_{gpu_arch}_suite")
                print(f"  COMMAND {target_name} --gtest_filter={pattern_string}")
                print(f"  WORKING_DIRECTORY {working_dir}")
                print(")")

                print(
                    f"set_tests_properties({target_name}_{category_name}_{gpu_arch}_suite PROPERTIES"
                )
                print(f"  LABELS {label_string}")
                print(f"  TIMEOUT {timeout}")
                if env_string:
                    print(f'  ENVIRONMENT "{env_string}"')
                print(")")
                print()

                # Write install-time test with relative path if install file is provided
                if install_file_handle:
                    try:
                        install_file_handle.write(
                            f'add_test({target_name}_{category_name}_{gpu_arch}_suite "../{target_name}" --gtest_filter={pattern_string})\n'
                        )
                        env_prop = f' ENVIRONMENT "{env_string}"' if env_string else ""
                        install_file_handle.write(
                            f"set_tests_properties({target_name}_{category_name}_{gpu_arch}_suite PROPERTIES LABELS {label_string} TIMEOUT {timeout}{env_prop})\n\n"
                        )
                        install_file_handle.flush()
                    except OSError as e:
                        print(
                            f"Warning: I/O error writing GPU exclude {category_name}_{gpu_arch} to install test file: {e}",
                            file=sys.stderr,
                        )
                    except Exception as e:
                        print(
                            f"Warning: Unexpected error writing GPU exclude {category_name}_{gpu_arch} to install test file: {type(e).__name__}: {e}",
                            file=sys.stderr,
                        )


if __name__ == "__main__":
    main()
