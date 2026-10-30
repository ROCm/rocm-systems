# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Names of the kernel folders in a workload folder of an analyze run.

A kernel folder is kernel-<NNN>-<name>. NNN is the kernel's place in the top
stats order, and <name> is its function name without the template arguments,
the parameter list or the namespaces. kernels.csv gives the full kernel name.
"""

import re
import subprocess
from typing import Dict, Iterable

KERNEL_FOLDER_PREFIX = "kernel"
MIN_INDEX_DIGITS = 3
MAX_NAME_LENGTH = 64
UNNAMED_FUNCTION_NAME = "unnamed"
MANGLED_NAME_PREFIX = "_Z"
DEMANGLER_COMMAND = "c++filt"

KERNEL_DESCRIPTOR_SUFFIX = re.compile(r"(?: \(\.kd\)|\.kd)$")
UNSAFE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9_]")
# An operator and its symbol, such as operator<, operator() or operator new[]
OPERATOR_NAME = re.compile(
    r"\boperator(\s*(?:\(\)|\[\]|(?:new|delete)\b(?:\s*\[\])?|[^\sA-Za-z0-9_()\[\]]+))"
)
# The cv-qualifiers and ref-qualifiers that follow a parameter list
TRAILING_QUALIFIERS = re.compile(
    r"(?:\s*(?:\bconst\b|\bvolatile\b|\bnoexcept\b|&&|&))+\s*$"
)


def kernel_index_width(kernel_count: int) -> int:
    """Return the digits of a kernel index: at least 3, and enough for the last.

    The last index is kernel_count - 1, so the folders of a run list in
    top-stats order when sorted by name.
    """
    last_index = max(kernel_count - 1, 0)
    return max(MIN_INDEX_DIGITS, len(str(last_index)))


def kernel_folder_name(index: int, index_width: int, function_name: str) -> str:
    """Return the folder name of the kernel at index, such as kernel-000-vecCopy.

    Args:
        index: Place of the kernel in the top-stats order, counting from 0.
        index_width: Digits of the index, from kernel_index_width.
        function_name: The name from kernel_function_name.
    """
    return f"{KERNEL_FOLDER_PREFIX}-{index:0{index_width}d}-{function_name}"


def kernel_function_name(kernel_name: str) -> str:
    """Return the folder-safe function name of a kernel.

    The name is cut to its bare function name, then every character outside
    [A-Za-z0-9_] becomes an underscore, and the result is cut to 64 characters.
    A name with no function name left is written as "unnamed".
    """
    identifier = _bare_function_name(kernel_name) or UNNAMED_FUNCTION_NAME
    return UNSAFE_NAME_CHARACTERS.sub("_", identifier)[:MAX_NAME_LENGTH]


def demangle_kernel_names(kernel_names: Iterable[str]) -> Dict[str, str]:
    """Return the demangled form of each mangled kernel name, using c++filt.

    Only names that start with _Z are sent to c++filt, all in one call. A name
    c++filt returns unchanged maps to itself. Every name is left out when
    c++filt is missing, fails, or prints a different number of lines, so the
    caller falls back to the raw name.
    """
    mangled_names = sorted({
        name for name in kernel_names if name.startswith(MANGLED_NAME_PREFIX)
    })
    if not mangled_names:
        return {}
    try:
        completed = subprocess.run(
            [DEMANGLER_COMMAND],
            input="\n".join(mangled_names) + "\n",
            capture_output=True,
            encoding="utf-8",
            check=False,
        )
    except OSError:
        return {}
    demangled_lines = completed.stdout.splitlines()
    if completed.returncode != 0 or len(demangled_lines) != len(mangled_names):
        return {}
    return dict(zip(mangled_names, demangled_lines))


def _bare_function_name(kernel_name: str) -> str:
    """Return the function name left once the signature is removed.

    The ' (.kd)' suffix goes first. An operator keeps its symbol, which the
    folder name then sanitizes. Then the template arguments, the parameter
    lists and the GCC clone suffix are removed, nested ones included, so a
    parameter such as std::vector<int> cannot end the name early. The trailing
    cv- and ref-qualifiers go, and the last namespace component is the function
    name, with a return type before it dropped.
    """
    name = KERNEL_DESCRIPTOR_SUFFIX.sub("", kernel_name.strip()).strip()
    name = OPERATOR_NAME.sub(_operator_spelling, name)
    unbracketed = TRAILING_QUALIFIERS.sub("", _without_brackets(name))
    words = unbracketed.split("::")[-1].split()
    return words[-1] if words else ""


def _operator_spelling(match: re.Match[str]) -> str:
    """Return an operator's name with each character outside [A-Za-z0-9_] as _.

    The symbol is masked here, so the brackets it holds are not taken for
    template arguments or parameter lists.
    """
    return "operator" + UNSAFE_NAME_CHARACTERS.sub("_", match.group(1))


def _without_brackets(text: str) -> str:
    """Return text without the characters inside <>, () or [], nesting included."""
    kept_characters = []
    angle_depth = 0
    paren_depth = 0
    square_depth = 0
    for character in text:
        if character == "<":
            angle_depth += 1
        elif character == ">":
            angle_depth = max(angle_depth - 1, 0)
        elif character == "(":
            paren_depth += 1
        elif character == ")":
            paren_depth = max(paren_depth - 1, 0)
        elif character == "[":
            square_depth += 1
        elif character == "]":
            square_depth = max(square_depth - 1, 0)
        elif angle_depth == 0 and paren_depth == 0 and square_depth == 0:
            kept_characters.append(character)
    return "".join(kept_characters)
