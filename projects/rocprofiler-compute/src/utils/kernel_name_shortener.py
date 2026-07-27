# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import re
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.logger import console_debug, console_error, console_log

# Module-level cache for demangled kernel names
_NAME_CACHE: dict[str, str] = {}

# Constants

# NOTE: c++filt is a Linux-only solution for demangling C++ symbols.
# TODO: We need to think about Windows support in the future.
# Windows equivalent might be undname.exe or using llvm-cxxfilt.
# CONCERN: Using absolute path here is brittle - c++filt location may vary
# across distributions. TODO: Consider using shutil.which() or PATH lookup instead.
CPP_FILT_PATH = "/usr/bin/c++filt"
MAX_SHORTENING_LEVEL = 5
KERNEL_NAME_COLUMNS = ["Kernel_Name", "Name"]


def validate_cpp_filt(cpp_filt_path: str = CPP_FILT_PATH) -> bool:
    """Validate that c++filt binary exists and is executable."""
    if not Path(cpp_filt_path).is_file():
        console_error(
            f"Could not resolve c++filt in expected directory: {cpp_filt_path}"
        )
        return False
    return True


def demangle_kernel_name(original_name: str, cpp_filt_path: str = CPP_FILT_PATH) -> str:
    cmd = [cpp_filt_path, original_name]

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        demangled_name, error = proc.communicate()

        if proc.returncode != 0:
            console_error(f"c++filt failed for {original_name}: {error}", exit=False)
            return original_name

        return demangled_name.strip()

    except (subprocess.SubprocessError, OSError) as e:
        console_error(f"Error running c++filt: {e}", exit=False)
        return original_name


def parse_template_depth(text: str, level: int, current_level: int) -> tuple[str, int]:
    result = ""
    curr_index = 0

    while curr_index < len(text) and ">" in text:
        if current_level < level:
            result += text[curr_index:]
            current_level -= text[curr_index:].count(">")
            break
        elif text[curr_index] == ">":
            current_level -= 1
        curr_index += 1

    return result, current_level


def shorten_demangled_name(demangled_name: str, level: int) -> str:
    names_and_args_pattern = re.compile(r"(?P<name>[( )A-Za-z0-9_]+)([ ,*<>()]+)(::)?")

    matches = names_and_args_pattern.findall(demangled_name)

    if not matches:
        # Handle cases like '__amd_rocclr_fillBuffer.kd'
        return demangled_name

    shortened_name = ""
    current_level = 0

    for name_part, args_part, scope_op in matches:
        # Skip 'clone' parts as they can cause errors
        if name_part == "clone":
            continue

        # Skip scope operators
        if scope_op == "::":
            continue

        # Add name part if within level limit
        if current_level < level:
            shortened_name += name_part

        # Handle template arguments
        if ">" not in args_part:
            if current_level < level:
                # Don't add opening brackets at the deepest level
                if not (current_level == level - 1 and "<" in args_part):
                    shortened_name += args_part
            current_level += args_part.count("<")
        else:
            # Handle closing template brackets
            if current_level < level:
                shortened_name += args_part
                current_level -= args_part.count(">")
            else:
                _, current_level = parse_template_depth(args_part, level, current_level)

    return shortened_name if shortened_name else demangled_name


def _argument_list_start(name: str) -> int:
    """Index of the ``(`` that opens the argument list, or -1 if none."""
    first_paren = name.find("(")
    if first_paren == -1:
        return -1
    operator_index = name.find("operator()")
    if operator_index != -1 and first_paren == operator_index + len("operator"):
        return name.find("(", operator_index + len("operator()"))
    return first_paren


def format_kernel_signature(name: str) -> str:
    """Trim a demangled kernel name to function(argument types)."""
    if not name or "(" not in name:
        return name

    opening_parenthesis_index = _argument_list_start(name)
    if opening_parenthesis_index == -1:
        return name
    parenthesis_depth = 0
    closing_parenthesis_index = -1
    for character_index in range(opening_parenthesis_index, len(name)):
        if name[character_index] == "(":
            parenthesis_depth += 1
        elif name[character_index] == ")":
            parenthesis_depth -= 1
            if parenthesis_depth == 0:
                closing_parenthesis_index = character_index
                break
    if closing_parenthesis_index == -1:
        return name

    argument_signature = name[opening_parenthesis_index : closing_parenthesis_index + 1]
    name_without_templates = _strip_template_parameters(
        name[:opening_parenthesis_index]
    ).strip()
    # The function is what follows the return type; a
    # demangled return type, when present, is separated by a top-level space.
    function_name = name_without_templates.rsplit(" ", 1)[-1]
    if not function_name:
        return name
    return f"{function_name}{argument_signature}"


def process_single_kernel_name(
    original_name: str, level: int, cpp_filt_path: str = CPP_FILT_PATH
) -> str:
    if original_name in _NAME_CACHE:
        return _NAME_CACHE[original_name]

    demangled_name = demangle_kernel_name(original_name, cpp_filt_path)
    shortened_name = shorten_demangled_name(demangled_name, level)

    final_name = shortened_name if shortened_name else demangled_name
    _NAME_CACHE[original_name] = final_name

    return final_name


def get_kernel_column_name(df: pd.DataFrame) -> Optional[str]:
    for column_name in KERNEL_NAME_COLUMNS:
        if column_name in df.columns:
            return column_name
    return None


def shorten_file(
    df: pd.DataFrame, level: int, cpp_filt_path: str = CPP_FILT_PATH
) -> pd.DataFrame:
    column_name = get_kernel_column_name(df)
    if not column_name:
        console_debug("No kernel name column found")
        return df

    df_copy = df.copy()

    df_copy[column_name] = df_copy[column_name].apply(
        lambda name: process_single_kernel_name(name, level, cpp_filt_path)
    )

    return df_copy


def kernel_name_shortener(df: pd.DataFrame, level: int) -> Optional[pd.DataFrame]:
    """Shorten kernel names in a DataFrame.

    NOTE: shortener is now dependent on a rocprof install with llvm

    Args:
        df: DataFrame containing kernel names
        level: Shortening level (0-4)

    Returns:
        DataFrame with shortened kernel names, or None if processing fails
    """

    if level >= MAX_SHORTENING_LEVEL:
        console_debug("profiling", "Skipping kernel name shortening: level >= 5")
        return df

    cpp_filt = CPP_FILT_PATH
    if not validate_cpp_filt(cpp_filt):
        return df

    try:
        modified_df = shorten_file(df, level, cpp_filt)
        console_log("profiling", "Kernel_Name shortening complete.")
        return modified_df
    except pd.errors.EmptyDataError:
        console_debug("profiling", "Skipping shortening on empty csv")
        return df
    except Exception as e:
        console_error(f"Error during kernel name shortening: {e}")
        return df


def _strip_template_parameters(text: str) -> str:
    """Remove nested C++ template arguments from a demangled name."""
    characters_outside_templates = []
    template_depth = 0
    for character in text:
        if character == "<":
            template_depth += 1
        elif character == ">":
            template_depth = max(template_depth - 1, 0)
        elif template_depth == 0:
            characters_outside_templates.append(character)
    return "".join(characters_outside_templates)
