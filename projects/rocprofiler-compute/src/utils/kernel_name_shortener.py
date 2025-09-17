##############################################################################
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

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
