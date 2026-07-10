# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Code-object disassembly analysis utilities.

Helpers for parsing the ``<pid>_code_obj_info.json`` artifact emitted by the
native PC sampling collector into a normalized per-code-object instruction tree.
The artifact holds the full disassembly of every loaded code object, including
instructions that were never sampled.
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


class CodeObjectDisassembly(NamedTuple):
    """A code object and every instruction disassembled from it."""

    code_object_id: int
    instructions: list[CodeObjectInstruction]


def parse_code_object_info(data: dict[str, Any]) -> list[CodeObjectDisassembly]:
    """Flatten a parsed ``code_obj_info.json`` dict into per-object instructions.

    Each code object owns several symbols; every symbol carries its own
    instruction list. The instructions are flattened per code object and keyed
    by ``virtual_address`` (the axis the analysis DB joins on).
    """
    return [
        CodeObjectDisassembly(
            code_object_id=code_object["id"],
            instructions=[
                _to_instruction(instruction)
                for symbol in code_object.get("symbols", [])
                for instruction in symbol.get("instructions", [])
            ],
        )
        for code_object in data.get("code_objects", [])
    ]


def load_code_object_disassemblies(
    workload_path: str,
) -> dict[int, list[CodeObjectDisassembly]]:
    """Discover and parse every ``<pid>_code_obj_info.json`` under a workload.

    Returns a ``{pid: disassemblies}`` map. A file whose name does not start with
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
    """Extract the leading integer pid from a ``<pid>_code_obj_info.json`` name."""
    pid_text = json_path.name.split("_", 1)[0]
    if not pid_text.isdigit():
        console_warning(f"Code object info: no pid prefix in {json_path.name}")
        return None
    return int(pid_text)


def _to_instruction(instruction: dict[str, Any]) -> CodeObjectInstruction:
    """Convert one raw instruction dict into a CodeObjectInstruction."""
    return CodeObjectInstruction(
        virtual_address=instruction["virtual_address"],
        instruction=instruction.get("name"),
        comment=instruction.get("comment"),
    )
