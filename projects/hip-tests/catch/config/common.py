# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import glob
import os

import yaml


NON_UNIT_GROUPS = [
    "ABM",
    "multiproc",
    "performance",
    "perftests",
    "stress",
    "TypeQualifiers",
]


def parse_size_string(size_str):
    """Convert size string (1K, 1M, 1G) to bytes.
    
    Args:
        size_str: Size as string ("1K", "1M", "1G") or int
        
    Returns:
        Size in bytes as integer
    """
    # Handle integers directly (YAML may parse numbers as int)
    if isinstance(size_str, int):
        return size_str
    
    size_str = str(size_str).strip().upper()
    multipliers = {'K': 1024, 'M': 1024*1024, 'G': 1024*1024*1024}
    
    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            return int(float(size_str[:-1]) * multiplier)
    return int(size_str)


def load_definitions(config_path):
    """Load definitions.yaml and return the full config including cmd_options.
    
    Args:
        config_path: Path to the config directory (containing configs/)
        
    Returns:
        Dict with 'definitions' and 'cmd_options' keys
    """
    definitions_path = os.path.join(config_path, "definitions.yaml")
    with open(definitions_path) as file:
        return yaml.safe_load(file)


def load_config(file_path, definitions_text):
    with open(file_path) as file:
        content = file.read()
    # Strip leading document separator so we can prepend definitions
    content = content.lstrip("-").lstrip("\n").lstrip()
    combined = definitions_text + "\n" + content
    return yaml.safe_load(combined)


def iter_group_configs(configs_path):
    """Yield (group, config) pairs for all unit and non-unit config files.

    Iterates unit configs first (sorted alphabetically), then non-unit groups
    in the order defined by NON_UNIT_GROUPS.
    """
    definitions_path = os.path.join(configs_path, "definitions.yaml")
    with open(definitions_path) as file:
        definitions_text = file.read()

    unit_config_dir = os.path.join(configs_path, "unit")
    for config_file in sorted(glob.glob(os.path.join(unit_config_dir, "*.yaml"))):
        config = load_config(config_file, definitions_text)
        group = os.path.splitext(os.path.basename(config_file))[0]
        if group not in config:
            continue
        yield group, config[group]

    for group in NON_UNIT_GROUPS:
        config_file = os.path.join(configs_path, f"{group}.yaml")
        config = load_config(config_file, definitions_text)
        yield group, config[group]
