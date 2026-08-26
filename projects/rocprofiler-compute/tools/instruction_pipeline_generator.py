#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""
Static instruction execution pipeline table generator for rocprofiler-compute.

Builds ``src/rocprof_compute_soc/analysis_configs/instruction_pipelines.json``,
the mnemonic-to-execution-pipeline map analyze uses to fill
``InstructionLine.instruction_type_uuid``. The mapping comes from the AMDGPU
TableGen files of the LLVM commit TheRock pins, so the names in the table are
the names the installed disassembler prints.

Everything happens in a temporary directory that is removed on exit. Nothing
from LLVM is needed at analyze time.

Setup and workflow are documented in CONTRIBUTING.md.

Usage (from the ``rocprofiler-compute`` project root):
    ./tools/instruction_pipeline_generator.py [--output PATH]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple, Optional

THEROCK_REPO = "https://github.com/ROCm/TheRock.git"
THEROCK_LLVM_SUBMODULE = "compiler/amd-llvm"
LLVM_REPO = "https://github.com/ROCm/llvm-project"
LLVM_SPARSE_PATHS = ("llvm/lib/Target/AMDGPU", "llvm/include/llvm")

DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "rocprof_compute_soc"
    / "analysis_configs"
    / "instruction_pipelines.json"
)

# Order matters: a record can set several flags at once, so the first pipeline
# whose rule matches wins. VALU and SALU are catch-alls that everything else is
# carved out of, so they come last.
PIPELINE_PRECEDENCE = (
    "MATRIX",
    "BRANCH",
    "BARRIER",
    "EXP",
    "INTERNAL",
    "LDS",
    "VMEM",
    "FLAT",
    "SCALAR",
    "VALU",
)

# TableGen encoding-class flag -> execution pipeline. SCALAR covers scalar ALU
# and scalar memory because the hardware serves both from one pipeline.
FLAG_TO_PIPELINE = {
    "IsMAI": "MATRIX",
    "IsWMMA": "MATRIX",
    "IsSWMMAC": "MATRIX",
    "isBranch": "BRANCH",
    "EXP": "EXP",
    "DS": "LDS",
    "LDSDIR": "LDS",
    "MUBUF": "VMEM",
    "MTBUF": "VMEM",
    "MIMG": "VMEM",
    "FLAT": "FLAT",
    "SMRD": "SCALAR",
    "SALU": "SCALAR",
    "VALU": "VALU",
}

# Nothing in TableGen names these pipelines, so they are matched by name. The
# barrier, message and wait instructions share the scalar encoding with plain
# scalar ALU work, and global, scratch and flat share the FLAT encoding while
# running on different pipelines. Tested in order, first match wins, and the
# order is folded into PIPELINE_PRECEDENCE.
NAME_RULES = (
    (
        "BARRIER",
        ("s_barrier",),
        ("s_wakeup_barrier", "s_get_barrier_state"),
    ),
    ("EXP", ("s_sendmsg",), ()),
    (
        "INTERNAL",
        (
            "s_wait",
            "s_soft_wait",
            "s_delay",
            "s_set",
            "s_sleep",
            "s_monitor_sleep",
            "s_ttrace",
            "s_clause",
            "s_inst_prefetch",
            "s_incperflevel",
            "s_decperflevel",
            "s_icache_inv",
            "s_denorm_mode",
            "s_round_mode",
            "s_singleuse_vdst",
            "s_endpgm",
            "s_code_end",
            "s_nop",
            "s_trap",
            "s_wakeup",
        ),
        (),
    ),
    ("VMEM", ("global_", "scratch_"), ()),
    ("FLAT", ("flat_",), ()),
)

# The encoding classes each name rule is allowed to win over. Everything the
# name rules match is either a scalar instruction or a FLAT-encoded one.
NAME_RULE_ENCODINGS = {
    "BARRIER": frozenset({"SCALAR"}),
    "EXP": frozenset({"SCALAR"}),
    "INTERNAL": frozenset({"SCALAR"}),
    "VMEM": frozenset({"VMEM", "FLAT"}),
    "FLAT": frozenset({"VMEM", "FLAT"}),
}

# Printed encoding suffixes, appended to a record's base name so analyze can
# look a line up without stripping anything.
ENCODING_SUFFIX_TOKENS = ("e32", "e64", "sdwa", "dpp", "dpp8")

MNEMONIC_PATTERN = re.compile(r"^[a-z][a-z0-9_]*")
# A disassembly line is "<mnemonic> <operands>" possibly followed by an
# encoding comment; the mnemonic is always the first token.
DISASSEMBLY_LINE_PATTERN = re.compile(r"^([a-z][a-z0-9_]*)\s")
CODE_OBJECT_SUFFIXES = ("*.hsaco", "*.co")


class InstructionRecord(NamedTuple):
    """One TableGen instruction record, reduced to what classification needs."""

    record_name: str
    spellings: frozenset[str]
    flags: frozenset[str]


def instruction_records(dump: dict) -> list[InstructionRecord]:
    """Reduce a tblgen JSON dump to one InstructionRecord per instruction."""
    records: list[InstructionRecord] = []
    for record_name in dump.get("!instanceof", {}).get("Instruction", []):
        record = dump[record_name]
        spellings = spellings_for_record(record_name, record)
        if not spellings:
            continue
        records.append(
            InstructionRecord(
                record_name=record_name,
                spellings=frozenset(spellings),
                flags=frozenset(_set_flags(record)),
            )
        )
    return records


def spellings_for_record(record_name: str, record: dict) -> set[str]:
    """Return every name the disassembler may print for one record.

    A name is spread over three fields: ``Mnemonic`` holds the short form,
    ``PseudoInstr`` the encoding-suffixed form, and ``AsmString`` the form after
    any per-family rename. Each base is also combined with the encoding suffix
    in the record name, which is how ``v_add_f32_e32`` gets a key of its own.
    """
    bases: set[str] = set()
    for field in ("Mnemonic", "PseudoInstr", "AsmString"):
        base = _leading_mnemonic(record.get(field))
        if base is not None:
            bases.add(base)

    suffix = _encoding_suffix(record_name)
    if suffix is None:
        return bases
    return bases | {
        f"{base}_{suffix}" for base in bases if not base.endswith(f"_{suffix}")
    }


def classify(spelling: str, flags: Iterable[str]) -> Optional[str]:
    """Return the execution pipeline for a name and its merged TableGen flags.

    Returns None when no rule matches, which leaves the column NULL rather than
    guessing a pipeline.
    """
    flag_set = set(flags)
    for pipeline in PIPELINE_PRECEDENCE:
        if _matches_pipeline(pipeline, spelling, flag_set):
            return pipeline
    return None


def merge_flags_by_spelling(
    records: Iterable[InstructionRecord],
) -> dict[str, set[str]]:
    """Union the flags of every record that prints the same name.

    One name has many records, one per encoding variant per GPU family, and a
    variant that carries no flags of its own would otherwise classify as
    something weaker than the instruction really is.
    """
    merged_flags: dict[str, set[str]] = {}
    for record in records:
        for spelling in record.spellings:
            merged_flags.setdefault(spelling, set()).update(record.flags)
    return merged_flags


def build_pipeline_table(merged_flags: dict[str, set[str]]) -> dict[str, str]:
    """Classify every merged name, dropping the ones no rule claims."""
    table: dict[str, str] = {}
    for spelling, flags in merged_flags.items():
        pipeline = classify(spelling, flags)
        if pipeline is not None:
            table[spelling] = pipeline
    return table


def find_rule_conflicts(merged_flags: dict[str, set[str]]) -> dict[str, str]:
    """Return names a hand-written name rule claims against their encoding.

    The name rules are the only hand-maintained part of the mapping, and the
    way they go wrong is by being too greedy: a prefix that was meant for the
    wait instructions swallowing every scalar one, say. A name rule that wins
    over an encoding class it was never meant to cover is that mistake, so the
    generator refuses to write the table.
    """
    conflicts: dict[str, str] = {}
    for spelling, flags in merged_flags.items():
        claimed = _name_rule_pipeline(spelling)
        if claimed is None or classify(spelling, flags) != claimed:
            continue
        encoding_pipeline = _flag_pipeline(flags)
        expected = NAME_RULE_ENCODINGS[claimed]
        if encoding_pipeline is not None and encoding_pipeline not in expected:
            conflicts[spelling] = f"{claimed} claims a {encoding_pipeline} encoding"
    return conflicts


def find_uncovered_mnemonics(
    table: dict[str, str], corpus_mnemonics: Iterable[str]
) -> list[str]:
    """Return corpus mnemonics the table has no entry for."""
    return sorted(set(corpus_mnemonics) - set(table))


def build_document(commit: str, table: dict[str, str]) -> dict:
    """Wrap the table in the provenance and the full rule set behind it.

    Every rule the classifier applies is recorded, the flag ones and the
    hand-written name ones alike, so a reader can tell where any entry came
    from and a regenerated file shows in its diff what moved. Analyze reads
    only ``pipelines``.
    """
    return {
        "repo": LLVM_REPO,
        "commit": commit,
        "precedence": list(PIPELINE_PRECEDENCE),
        "flag_to_pipeline": FLAG_TO_PIPELINE,
        "name_rules": {
            pipeline: {"prefixes": list(prefixes), "names": list(names)}
            for pipeline, prefixes, names in NAME_RULES
        },
        "pipelines": dict(sorted(table.items())),
    }


def parse_corpus_mnemonics(disassembly: str) -> set[str]:
    """Collect the mnemonics of an llvm-objdump disassembly listing."""
    mnemonics: set[str] = set()
    for line in disassembly.splitlines():
        match = DISASSEMBLY_LINE_PATTERN.match(line.strip())
        if match:
            mnemonics.add(match.group(1))
    return mnemonics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the generated table.",
    )
    args = parser.parse_args()

    rocm_path = Path(os.getenv("ROCM_PATH", "/opt/rocm"))
    tblgen = rocm_path / "lib" / "llvm" / "bin" / "llvm-tblgen"
    objdump = rocm_path / "lib" / "llvm" / "bin" / "llvm-objdump"
    for binary in (tblgen, objdump):
        if not binary.is_file():
            print(f"error: {binary} not found; set ROCM_PATH", file=sys.stderr)
            return 1

    corpus_mnemonics = _collect_corpus_mnemonics(rocm_path, objdump)
    if not corpus_mnemonics:
        print(
            f"error: no code objects found under {rocm_path}; "
            "coverage cannot be checked",
            file=sys.stderr,
        )
        return 1

    with tempfile.TemporaryDirectory(prefix="instruction_pipelines_") as work_dir:
        work_path = Path(work_dir)
        commit = _read_llvm_pin(work_path / "therock")
        llvm_path = _fetch_llvm_sources(work_path / "llvm", commit)
        print(f"Running llvm-tblgen on the AMDGPU target of {commit[:12]}")
        dump = _run_tblgen(tblgen, llvm_path, work_path / "amdgpu.json")

    records = instruction_records(dump)
    merged_flags = merge_flags_by_spelling(records)
    table = build_pipeline_table(merged_flags)
    print(f"Classified {len(table)} names from {len(records)} records")

    conflicts = find_rule_conflicts(merged_flags)
    if conflicts:
        print(
            f"error: {len(conflicts)} names a name rule claims wrongly: "
            f"{sorted(conflicts.items())[:5]}",
            file=sys.stderr,
        )
        return 1

    uncovered = find_uncovered_mnemonics(table, corpus_mnemonics)
    if uncovered:
        print(
            f"error: {len(uncovered)} corpus mnemonics missing from the table: "
            f"{uncovered[:10]}",
            file=sys.stderr,
        )
        return 1

    _write_table(args.output, commit, table)
    print(f"Wrote {args.output}")
    return 0


def _matches_pipeline(pipeline: str, spelling: str, flags: set[str]) -> bool:
    """Test one pipeline's rule, which is a name rule, a flag rule, or both."""
    return _name_rule_pipeline(spelling) == pipeline or _has_flag_for(pipeline, flags)


def _name_rule_pipeline(spelling: str) -> Optional[str]:
    """Return the pipeline a hand-written name rule gives this name, if any."""
    for pipeline, prefixes, names in NAME_RULES:
        if spelling.startswith(prefixes) or spelling in names:
            return pipeline
    return None


def _flag_pipeline(flags: set[str]) -> Optional[str]:
    """Return the pipeline the encoding-class flags alone give."""
    for pipeline in PIPELINE_PRECEDENCE:
        if _has_flag_for(pipeline, flags):
            return pipeline
    return None


def _has_flag_for(pipeline: str, flags: set[str]) -> bool:
    return any(FLAG_TO_PIPELINE.get(flag) == pipeline for flag in flags)


def _set_flags(record: dict) -> Iterator[str]:
    """Yield the flag names the record sets, ignoring everything else."""
    for flag in FLAG_TO_PIPELINE:
        if record.get(flag):
            yield flag


def _leading_mnemonic(value: object) -> Optional[str]:
    """Return the name a field starts with, dropping whatever follows it.

    Operands follow the name in several fields, sometimes with no separating
    space (``v_add_co_ci_u32$vdst``), and pseudo records carry placeholder text
    that is never printed.
    """
    if not isinstance(value, str):
        return None
    match = MNEMONIC_PATTERN.match(value.strip().lower())
    return match.group(0) if match else None


def _encoding_suffix(record_name: str) -> Optional[str]:
    """Return the printed encoding suffix of a record name, if it has one.

    The suffix is not at the end: a per-family suffix follows it, as in
    ``V_ADD_CO_CI_U32_e64_dpp_gfx12``.
    """
    tokens = [
        token
        for token in record_name.lower().split("_")
        if token in ENCODING_SUFFIX_TOKENS
    ]
    return "_".join(tokens) if tokens else None


def _read_llvm_pin(therock_path: Path) -> str:
    """Return the llvm-project commit TheRock pins as its compiler submodule."""
    therock_path.mkdir(parents=True)
    _git(therock_path, "init", "--quiet")
    _git(therock_path, "remote", "add", "origin", THEROCK_REPO)
    print(f"Reading the compiler pin from {THEROCK_REPO}")
    _git(therock_path, "fetch", "--quiet", "--depth", "1", "origin", "main")
    # ls-tree reads the gitlink from the object store, so no files are written.
    listing = _git(therock_path, "ls-tree", "FETCH_HEAD", THEROCK_LLVM_SUBMODULE)
    return listing.split()[2]


def _fetch_llvm_sources(llvm_path: Path, commit: str) -> Path:
    """Sparsely check out the AMDGPU TableGen files at one llvm-project commit.

    The whole ``llvm/include/llvm`` tree is taken because the TableGen includes
    chain, and listing the exact subdirectories breaks whenever LLVM adds one.
    """
    llvm_path.mkdir(parents=True)
    _git(llvm_path, "init", "--quiet")
    _git(llvm_path, "remote", "add", "origin", f"{LLVM_REPO}.git")
    _git(llvm_path, "sparse-checkout", "init", "--cone")
    _git(llvm_path, "sparse-checkout", "set", *LLVM_SPARSE_PATHS)
    print(f"Fetching {commit[:12]} of {LLVM_REPO}")
    _git(
        llvm_path,
        "fetch",
        "--quiet",
        "--depth",
        "1",
        "--filter=blob:none",
        "origin",
        commit,
    )
    _git(llvm_path, "checkout", "--quiet", "FETCH_HEAD")
    return llvm_path / "llvm"


def _run_tblgen(tblgen: Path, llvm_path: Path, dump_path: Path) -> dict:
    """Dump every AMDGPU TableGen record as JSON and load it."""
    target_path = llvm_path / "lib" / "Target" / "AMDGPU"
    subprocess.run(
        [
            str(tblgen),
            "--dump-json",
            "-I",
            str(target_path),
            "-I",
            str(llvm_path / "include"),
            str(target_path / "AMDGPU.td"),
            "-o",
            str(dump_path),
        ],
        check=True,
    )
    with open(dump_path, encoding="utf-8") as dump_file:
        return json.load(dump_file)


def _collect_corpus_mnemonics(rocm_path: Path, objdump: Path) -> set[str]:
    """Disassemble the code objects of a ROCm install to see what real code emits."""
    mnemonics: set[str] = set()
    for suffix in CODE_OBJECT_SUFFIXES:
        for code_object in sorted(rocm_path.rglob(suffix)):
            disassembly = subprocess.run(
                [str(objdump), "-d", str(code_object)],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            mnemonics |= parse_corpus_mnemonics(disassembly)
    print(f"Corpus holds {len(mnemonics)} distinct mnemonics")
    return mnemonics


def _write_table(output_path: Path, commit: str, table: dict[str, str]) -> None:
    """Write the generated document."""
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(build_document(commit, table), output_file, indent=2)
        output_file.write("\n")


def _git(repository_path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository_path), *arguments],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


if __name__ == "__main__":
    sys.exit(main())
