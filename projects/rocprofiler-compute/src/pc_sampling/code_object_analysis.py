# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Code-object disassembly analysis utilities.

Parse the per-process code-object info artifact emitted by the native PC
sampling collector into a per-code-object instruction tree. The artifact holds
the full disassembly of every loaded code object, including un-sampled ones.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple, Optional

from utils.logger import console_warning

CODE_OBJ_INFO_GLOB = "**/*_code_obj_info.json"


class CodeObjectInstruction(NamedTuple):
    """One disassembled instruction within a code object."""

    virtual_address: int
    instruction: str
    comment: str
    kernel_name: str


class CodeObjectDisassembly(NamedTuple):
    """A code object and every instruction disassembled from it."""

    code_object_id: int
    instructions: list[CodeObjectInstruction]


def parse_code_object_info(data: dict[str, Any]) -> list[CodeObjectDisassembly]:
    """Flatten a parsed code-object info dict into per-object instructions.

    Each code object owns several symbols; every symbol carries its own
    instruction list, flattened here into one list per code object.
    """
    return [
        CodeObjectDisassembly(
            code_object_id=code_object["id"],
            instructions=[
                _to_instruction(instruction, symbol.get("name"))
                for symbol in code_object.get("symbols", [])
                for instruction in symbol.get("instructions", [])
            ],
        )
        for code_object in data.get("code_objects", [])
    ]


def load_code_object_disassemblies(
    workload_path: str,
) -> dict[int, list[CodeObjectDisassembly]]:
    """Discover and parse every code-object info file under a workload.

    Returns a pid to disassemblies map. A file whose name does not start with
    an integer pid, or that fails to parse, is skipped with a warning.
    """
    disassemblies_per_pid: dict[int, list[CodeObjectDisassembly]] = {}
    for json_path in sorted(Path(workload_path).glob(CODE_OBJ_INFO_GLOB)):
        pid = _parse_pid(json_path)
        if pid is None:
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            console_warning(f"Code object info: failed to parse {json_path}: {error}")
            continue
        disassemblies_per_pid[pid] = parse_code_object_info(data)
    return disassemblies_per_pid


def _parse_pid(json_path: Path) -> Optional[int]:
    """Extract the leading integer pid from a code-object info filename."""
    pid_text = json_path.name.split("_", 1)[0]
    if not pid_text.isdigit():
        console_warning(f"Code object info: no pid prefix in {json_path.name}")
        return None
    return int(pid_text)


def _to_instruction(
    instruction: dict[str, Any], symbol_name: str
) -> CodeObjectInstruction:
    """Convert one raw instruction dict into a CodeObjectInstruction."""
    return CodeObjectInstruction(
        virtual_address=instruction["virtual_address"],
        instruction=instruction.get("name"),
        comment=instruction.get("comment"),
        kernel_name=symbol_name,
    )
