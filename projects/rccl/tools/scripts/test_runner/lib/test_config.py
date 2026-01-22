#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Test Configuration Processor Module
Handles hierarchical test configuration with inheritance and merging
"""

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType

# Set default WORKDIR to rccl root directory if not already defined
# This file is at: rccl/tools/scripts/test_runner/lib/test_config.py
# rccl root is 5 directories up
if "WORKDIR" not in os.environ:
    _rccl_root = Path(__file__).resolve().parents[4]
    os.environ["WORKDIR"] = str(_rccl_root)


class TestConfigProcessor:
    """
    Processes hierarchical test configurations with support for:
    - Configuration inheritance ('using' directive)
    - Environment variable merging
    - Test parameter inheritance
    - Environment variable expansion in paths
    """

    def __init__(self, config_file):
        """
        Initialize the TestConfigProcessor with the configuration file.

        Args:
            config_file: Path to JSON configuration file
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        # Load the JSON configuration file
        with open(config_file, 'r') as file:
            config_data = json.load(file)

        # Expand environment variables in paths section
        if "paths" in config_data:
            config_data["paths"] = self._expand_env_vars_in_dict(config_data["paths"])

        # Make the configuration immutable (frozen)
        self.config = MappingProxyType(config_data)
        self.config_file = config_file

    def _expand_env_var(self, value):
        """
        Expand environment variables in a string.

        Supports both ${VAR} and $VAR syntax.
        If an environment variable is not set, it will be left unexpanded
        or replaced with an empty string based on the pattern.

        Args:
            value: String that may contain environment variables

        Returns:
            str: String with environment variables expanded

        Examples:
            "${HOME}/code" -> "/home/user/code"
            "$ROCM_PATH/bin" -> "/opt/rocm/bin"
            "${UNDEFINED:-/default}" -> "/default" (bash-style default)
            "${WORKDIR:-$HOME/code}" -> expands $HOME in default if WORKDIR not set
        """
        if not isinstance(value, str):
            return value

        # Pattern to match ${VAR}, ${VAR:-default}, or $VAR
        # First, handle ${VAR:-default} pattern
        def replace_with_default(match):
            var_name = match.group(1)
            default_value = match.group(2)
            # Get the env var, or use default
            result = os.environ.get(var_name)
            if result is None:
                # Recursively expand env vars in the default value
                result = self._expand_env_var(default_value)
            return result

        # Replace ${VAR:-default} patterns
        value = re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}', replace_with_default, value)

        # Replace ${VAR} patterns
        value = re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}',
                      lambda m: os.environ.get(m.group(1), m.group(0)), value)

        # Replace $VAR patterns (but not ${ to avoid double replacement)
        value = re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)',
                      lambda m: os.environ.get(m.group(1), m.group(0)), value)

        return value

    def _expand_env_vars_in_dict(self, data):
        """
        Recursively expand environment variables in all string values in a dictionary.

        Args:
            data: Dictionary that may contain environment variables in string values

        Returns:
            dict: Dictionary with all environment variables expanded
        """
        if isinstance(data, dict):
            return {key: self._expand_env_vars_in_dict(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars_in_dict(item) for item in data]
        elif isinstance(data, str):
            return self._expand_env_var(data)
        else:
            return data

    def combine_configs(self, config_name):
        """
        Combines configurations generically using the 'extends' directive.

        Merging rules:
        - env_variables: Overwrite duplicate keys (child overwrites parent)
        - mpi_args: Append and remove duplicates
        - args: Append and remove duplicates
        - tests: Merge by test name
        - Other fields: Child overwrites parent

        Args:
            config_name: Name of configuration to combine

        Returns:
            dict: Combined configuration
        """
        test_configs = self.config.get("test_configurations", {})
        if config_name not in test_configs:
            raise ValueError(
                f"Configuration '{config_name}' not found in test_configurations. "
                f"Available: {', '.join(test_configs.keys())}"
            )

        # Start with a deep copy of the target configuration
        combined_config = deepcopy(test_configs[config_name])

        # Process the 'extends' directive if it exists
        while "extends" in combined_config:
            parent_configs = combined_config.pop("extends")
            if not isinstance(parent_configs, list):
                parent_configs = [parent_configs]

            for parent_config_name in parent_configs:
                if parent_config_name not in test_configs:
                    raise ValueError(
                        f"Parent configuration '{parent_config_name}' not found."
                    )

                parent_config = deepcopy(test_configs[parent_config_name])

                # Recursively process parent's 'extends' directive
                if "extends" in parent_config:
                    parent_config = self.combine_configs(parent_config_name)

                # Merge all keys from parent into combined configuration
                for key, value in parent_config.items():
                    if key == "env_variables":
                        # Merge env_variables (child overwrites parent)
                        current_env = combined_config.get("env_variables", {})
                        combined_env = {**value, **current_env}
                        combined_config["env_variables"] = combined_env
                    elif key in ["args", "mpi_args"]:
                        # Append lists and remove duplicates (preserve order)
                        current_items = combined_config.get(key, [])
                        if isinstance(current_items, list) and isinstance(value, list):
                            combined_config[key] = list(dict.fromkeys(value + current_items))
                        elif isinstance(value, list):
                            combined_config[key] = value
                    elif key == "tests":
                        # Merge tests by name
                        current_tests = combined_config.get("tests", [])
                        combined_tests = self._merge_tests(value, current_tests)
                        combined_config["tests"] = combined_tests
                    else:
                        # Child overwrites parent for other keys
                        if key not in combined_config:
                            combined_config[key] = value

        return combined_config

    def _merge_tests(self, parent_tests, child_tests):
        """
        Merges two lists of tests by name.

        Args:
            parent_tests: List of parent tests
            child_tests: List of child tests

        Returns:
            list: Merged list of tests
        """
        merged_tests = []
        test_map = {}

        # Process parent tests
        for test in parent_tests:
            if isinstance(test, str):
                test_map[test] = {"name": test}
            elif isinstance(test, dict):
                name = test.get("name")
                if name:
                    test_map[name] = test

        # Process child tests (child overwrites parent)
        for test in child_tests:
            if isinstance(test, str):
                test_map[test] = {"name": test}
            elif isinstance(test, dict):
                name = test.get("name")
                if name:
                    # Merge with parent test if exists
                    if name in test_map:
                        parent_test = test_map[name]
                        merged_test = {**parent_test, **test}
                        test_map[name] = merged_test
                    else:
                        test_map[name] = test

        # Convert map back to list
        merged_tests = list(test_map.values())
        return merged_tests

    def _apply_test_defaults(self, tests, config_defaults):
        """
        Apply configuration-level defaults to individual tests.

        Test-specific values override configuration defaults.

        Args:
            tests: List of test dictionaries
            config_defaults: Dictionary with default values from configuration

        Returns:
            list: Tests with defaults applied
        """
        # Fields that can have defaults at config level
        default_fields = ["is_gtest", "binary", "num_ranks", "num_nodes", "num_gpus", "timeout"]

        processed_tests = []
        for test in tests:
            # Start with config defaults
            merged_test = {}

            # Apply defaults for each field if not already in test
            for field in default_fields:
                if field in config_defaults:
                    merged_test[field] = config_defaults[field]

            # Override with test-specific values
            merged_test.update(test)

            processed_tests.append(merged_test)

        return processed_tests

    def parse_test_suites(self):
        """
        Parses the test_suites section and processes each test suite.

        Applies hierarchical defaults in order (test-specific overrides suite, suite overrides config):
        1. Configuration-level defaults
        2. Test suite-level defaults (override config)
        3. Individual test values (override both)

        Returns:
            list: List of combined configurations for each test suite
        """
        test_suites = self.config.get("test_suites", [])
        combined_suites = []

        for suite in test_suites:
            config_name = suite.get("config")
            if not config_name:
                raise ValueError(
                    f"Test suite '{suite.get('name')}' does not specify a configuration."
                )

            # Combine the configuration for the test suite
            combined_config = self.combine_configs(config_name)

            # Extract configuration-level defaults
            config_defaults = {
                "is_gtest": combined_config.get("is_gtest"),
                "binary": combined_config.get("binary"),
                "num_ranks": combined_config.get("num_ranks"),
                "num_nodes": combined_config.get("num_nodes"),
                "num_gpus": combined_config.get("num_gpus", 8),
                "timeout": combined_config.get("timeout")
            }
            # Remove None values
            config_defaults = {k: v for k, v in config_defaults.items() if v is not None}

            # Extract suite-level defaults (override config-level)
            suite_defaults = {
                "is_gtest": suite.get("is_gtest"),
                "binary": suite.get("binary"),
                "num_ranks": suite.get("num_ranks"),
                "num_nodes": suite.get("num_nodes"),
                "num_gpus": suite.get("num_gpus"),
                "timeout": suite.get("timeout")
            }
            # Remove None values
            suite_defaults = {k: v for k, v in suite_defaults.items() if v is not None}

            # Merge defaults: suite-level overrides config-level
            merged_defaults = {**config_defaults, **suite_defaults}

            # Apply merged defaults to tests
            tests = combined_config.get("tests", [])
            if tests and merged_defaults:
                combined_config["tests"] = self._apply_test_defaults(tests, merged_defaults)

            # Add suite-specific details
            combined_config["suite_details"] = {
                "name": suite.get("name"),
                "description": suite.get("description", ""),
                "num_nodes": suite.get("num_nodes", 1),
                "num_ranks": suite.get("num_ranks", 1),
                "num_gpus": suite.get("num_gpus", 8),
                "enabled": suite.get("enabled", True)
            }

            combined_suites.append(combined_config)

        return combined_suites

    def get_system_config(self):
        """
        Get system-wide configuration settings.

        Returns:
            dict: System configuration
        """
        return self.config.get("system_configurations", {})

    def get_env_variables(self):
        """
        Get global environment variables.

        Returns:
            dict: Global environment variables
        """
        return self.config.get("env_variables", {})

    def get_paths(self):
        """
        Get system paths (ROCM, MPI, etc.).

        Returns:
            dict: System paths
        """
        return self.config.get("paths", {})

    def get_build_config(self):
        """
        Get build configuration settings.

        Returns:
            dict: Build configuration with CMake options, environment variables, etc.
        """
        return self.config.get("build_configuration", {})

    def validate_config(self):
        """
        Validate the configuration for required fields.

        Raises:
            ValueError: If configuration is invalid
        """
        # Check for required top-level keys
        required_keys = ["test_configurations", "test_suites"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Validate test suites
        test_suites = self.config.get("test_suites", [])
        if not test_suites:
            raise ValueError("No test suites defined in configuration")

        for suite in test_suites:
            if "name" not in suite:
                raise ValueError("Test suite missing 'name' field")
            if "config" not in suite:
                raise ValueError(f"Test suite '{suite['name']}' missing 'config' field")

        return True

