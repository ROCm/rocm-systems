# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared dataclasses for hazard mutation testing."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


@dataclass
class WaitInstruction:
    """A single s_wait_* instruction found in assembly."""

    line_number: int  # 0-based line index
    instruction: str  # e.g. "s_wait_loadcnt"
    operand: str  # e.g. "0x0"
    full_line: str  # original line text


@dataclass
class MemoryMutation:
    """A sub-dword load that can be rewritten in place as a store.

    Turning a byte/short *read* into a *write* to the same address is the
    single-edit that makes an otherwise race-free all-reads kernel racy: the
    new writer must race the foreign readers of that byte. It exposes the
    global shadow's sub-dword eviction gap the way removing an ``s_wait``
    exposes a missing dependency.
    """

    line_number: int  # 0-based line index
    load_mnemonic: str  # e.g. "flat_load_ubyte"
    store_mnemonic: str  # e.g. "flat_store_byte"
    full_line: str  # original line text
    rewritten_line: str  # the store that replaces it (no trailing newline)


@dataclass
class MutantResult:
    """Result of building and running one mutant."""

    shader: str
    wait_index: int  # index into the shader's wait list
    wait_instruction: str  # e.g. "s_wait_loadcnt 0x0"
    wait_line: int  # line number in assembly
    kind: str = "wait"  # "wait" (s_wait removal) or "memory" (load->store)
    compiled: bool = False
    linked: bool = False
    ran: bool = False
    correct: bool = False
    exit_code: Optional[int] = None
    stdout_match: bool = False
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    hazard_count: int = 0
    hazard_report_path: str = ""
    hazard_report_raw: str = ""


@dataclass
class ShaderReport:
    """Aggregate report for one shader."""

    shader: str
    asm_file: str
    asm_content: str = ""
    waits_found: List[WaitInstruction] = field(default_factory=list)
    compile_ok: bool = False
    build_ok: bool = False
    error: str = ""
    baseline_ok: bool = False
    baseline_stdout: str = ""
    baseline_exit_code: int = -1
    baseline_hazard_count: int = 0
    baseline_hazard_report_path: str = ""
    baseline_hazard_report_raw: str = ""
    mutants: List[MutantResult] = field(default_factory=list)


class DetectionLabel(str, Enum):
    """Detection-quality verdicts for mutant hazard reporting."""

    FN = "FN"  # False negative: killed but plugin missed it
    DETECTED = "Detected"  # True positive: killed and plugin caught it
    FP = "FP"  # False positive: survived but plugin flagged it
    OK = "OK"  # True negative: survived and no hazards
    NA = "-"  # Not applicable (build fail, timeout, etc.)
