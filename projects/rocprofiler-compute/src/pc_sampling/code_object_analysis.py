# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Code-object disassembly analysis utilities.

Parse the per-process code-object info artifact emitted by the native PC
sampling collector into a per-code-object instruction tree. The artifact holds
the full disassembly of every loaded code object, including un-sampled ones.

Also maps a disassembled instruction to the execution pipeline that runs it.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple, Optional

import yaml

import config
from utils.logger import console_warning

CODE_OBJ_INFO_GLOB = "**/*_code_obj_info.json"


class CodeObjectInstruction(NamedTuple):
    """One disassembled instruction within a code object."""

    virtual_address: int
    instruction: Optional[str]
    source: Optional[str]


class CodeObjectSymbol(NamedTuple):
    """One kernel symbol within a code object, and the instructions it spans."""

    name: Optional[str]
    virtual_address: int
    instructions: list[CodeObjectInstruction]


class CodeObjectDisassembly(NamedTuple):
    """A code object and every symbol disassembled from it."""

    code_object_id: int
    symbols: list[CodeObjectSymbol]


def parse_code_object_info(data: dict[str, Any]) -> list[CodeObjectDisassembly]:
    """Parse a code-object info dict into a per-object symbol tree.

    Keeps only virtual addresses: the artifact's own ``code_object_offset`` is
    a file offset, which callers cannot rebase onto the runtime load_base.
    """
    return [
        CodeObjectDisassembly(
            code_object_id=code_object["id"],
            symbols=[_to_symbol(symbol) for symbol in code_object.get("symbols", [])],
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


def _to_symbol(symbol: dict[str, Any]) -> CodeObjectSymbol:
    """Convert one raw symbol dict into a CodeObjectSymbol."""
    return CodeObjectSymbol(
        name=symbol.get("name"),
        virtual_address=symbol["virtual_address"],
        instructions=[
            _to_instruction(instruction)
            for instruction in symbol.get("instructions", [])
        ],
    )


def _to_instruction(instruction: dict[str, Any]) -> CodeObjectInstruction:
    """Convert one raw instruction dict into a CodeObjectInstruction."""
    return CodeObjectInstruction(
        virtual_address=instruction["virtual_address"],
        instruction=instruction.get("name"),
        source=instruction.get("comment"),
    )


class InstructionPipelines:
    """The generated mnemonic-to-pipeline table, read once and kept."""

    table: Optional[dict[str, str]] = None

    @classmethod
    def lookup(cls, instruction: Optional[str]) -> Optional[str]:
        """Return the execution pipeline that runs a disassembled instruction.

        The pipeline (VALU, MATRIX, SCALAR, and so on) is a property of the
        instruction itself, so it applies to every disassembled line, not only
        the offsets PC sampling landed on. Returns None for an instruction the
        table does not hold, leaving the type unset rather than guessing.
        """
        if not instruction:
            return None
        if cls.table is None:
            cls.table = cls.load()
        return cls.table.get(instruction.split(maxsplit=1)[0])

    @classmethod
    def load(cls) -> dict[str, str]:
        """Read the table, inverting it into the mnemonic lookup."""
        path = (
            config.rocprof_compute_home
            / "rocprof_compute_soc"
            / "analysis_configs"
            / "instruction_pipelines.yaml"
        )
        if not path.is_file():
            return {}
        # The C loader is far faster on a file this size, and ships with PyYAML
        # wherever libyaml is available.
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=loader)
        return {
            mnemonic: pipeline
            for pipeline, mnemonics in ((document or {}).get("pipelines") or {}).items()
            for mnemonic in mnemonics
        }
