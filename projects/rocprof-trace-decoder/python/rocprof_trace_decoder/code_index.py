from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .records import Instruction, Pc, Wave

CODE_HEADER = "ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle"


@dataclass
class CodeEntry:
    pc: Pc
    inst: str
    line_number: int
    source: str = ""
    hitcount: int = 0
    latency: int = 0
    stall: int = 0
    idle: int = 0
    expected_hitcount: int = 0
    expected_latency: int = 0
    expected_stall: int = 0
    expected_idle: int = 0
    memory_size: int = 4


class CodeIndex:
    def __init__(self, entries: Iterable[CodeEntry], document: dict | None = None):
        self.document = document or {"code": [], "header": CODE_HEADER}
        self.entries: dict[Pc, CodeEntry] = {}
        self.line_numbers: dict[Pc, int] = {}

        for entry in entries:
            self.entries[entry.pc] = entry
            self.line_numbers[entry.pc] = entry.line_number

        self._set_memory_sizes()

    @classmethod
    def from_code_json(cls, path: str | Path) -> "CodeIndex":
        code_path = Path(path)
        doc = json.loads(code_path.read_text())
        entries: list[CodeEntry] = []

        for row in doc.get("code", []):
            if len(row) < 10:
                continue
            inst = str(row[0])
            if inst.startswith(";"):
                continue
            pc = Pc(address=int(row[5]), code_object_id=int(row[4]))
            entries.append(
                CodeEntry(
                    pc=pc,
                    inst=inst,
                    line_number=int(row[2]),
                    source=str(row[3]),
                    hitcount=int(row[6] or 0),
                    latency=int(row[7] or 0),
                    stall=int(row[8] or 0),
                    idle=int(row[9] or 0),
                )
            )

        index = cls(entries, doc)
        index.path = code_path
        return index

    @classmethod
    def from_stats_csv(cls, paths: Iterable[str | Path]) -> "CodeIndex":
        entries: dict[Pc, CodeEntry] = {}
        for path in paths:
            with Path(path).open("r", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if not row:
                        continue
                    pc = Pc(address=int(row["Vaddr"].strip(), 0), code_object_id=int(row["CodeObj"].strip(), 0))
                    inst = row["Instruction"].strip()
                    if inst.startswith(";"):
                        continue

                    entry = entries.setdefault(
                        pc,
                        CodeEntry(
                            pc=pc,
                            inst=inst,
                            line_number=len(entries) + 1,
                            source=row.get("Source", ""),
                        ),
                    )
                    entry.expected_hitcount += _int_cell(row.get("Hitcount"))
                    entry.expected_latency += _int_cell(row.get("Latency"))
                    entry.expected_stall += _int_cell(row.get("Stall"))
                    entry.expected_idle += _int_cell(row.get("Idle"))
        return cls(entries.values())

    def isa_for_pc(self, pc: Pc) -> tuple[str, int] | None:
        entry = self.entries.get(pc)
        if entry is None:
            return None
        return entry.inst, entry.memory_size

    def get_or_create(self, pc: Pc) -> CodeEntry:
        entry = self.entries.get(pc)
        if entry is not None:
            return entry
        entry = CodeEntry(pc=pc, inst="", line_number=len(self.entries) + 1)
        self.entries[pc] = entry
        self.line_numbers[pc] = entry.line_number
        self._set_memory_sizes()
        return entry

    def line_number(self, pc: Pc) -> int:
        return self.entries[pc].line_number

    def accumulate_wave(self, wave: Wave) -> None:
        prev_inst_time = wave.begin_time
        for inst in wave.instructions:
            if inst.pc.code_object_id == 0 and inst.pc.address == 0:
                continue
            entry = self.get_or_create(inst.pc)
            entry.hitcount += 1
            entry.latency += inst.duration
            entry.stall += inst.stall
            entry.idle += max(inst.time - prev_inst_time, 0)
            prev_inst_time = max(prev_inst_time, inst.time + inst.duration)

    def write_code_json(self, path: str | Path) -> None:
        out_doc = dict(self.document)
        rows = []
        for row in self.document.get("code", []):
            if len(row) < 10 or str(row[0]).startswith(";"):
                rows.append(row)
                continue
            pc = Pc(address=int(row[5]), code_object_id=int(row[4]))
            entry = self.entries.get(pc)
            if entry is None:
                rows.append(row)
                continue
            updated = list(row)
            updated[6] = entry.hitcount
            updated[7] = entry.latency
            updated[8] = entry.stall
            updated[9] = entry.idle
            rows.append(updated)
        out_doc["code"] = rows
        out_doc.setdefault("header", CODE_HEADER)
        Path(path).write_text(json.dumps(out_doc, indent=2))

    def write_stats_csv(self, path: str | Path) -> None:
        with Path(path).open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["CodeObj", "Vaddr", "Instruction", "Hitcount", "Latency", "Stall", "Idle", "Source"])
            for entry in sorted(self.entries.values(), key=lambda e: (e.pc.code_object_id, e.pc.address)):
                writer.writerow(
                    [
                        entry.pc.code_object_id,
                        entry.pc.address,
                        entry.inst,
                        entry.hitcount,
                        entry.latency,
                        entry.stall,
                        entry.idle,
                        entry.source,
                    ]
                )

    def validate_expected(self) -> list[str]:
        errors = []
        for entry in sorted(self.entries.values(), key=lambda e: (e.pc.code_object_id, e.pc.address)):
            if (
                entry.hitcount == entry.expected_hitcount
                and entry.latency == entry.expected_latency
                and entry.stall == entry.expected_stall
                and entry.idle == entry.expected_idle
            ):
                continue
            errors.append(
                f"{entry.inst} - PC: {entry.pc.code_object_id},{entry.pc.address}\n"
                f"Hitcount: {entry.hitcount}/{entry.expected_hitcount}, "
                f"Stall: {entry.stall}/{entry.expected_stall}, "
                f"Latency: {entry.latency}/{entry.expected_latency}, "
                f"Idle: {entry.idle}/{entry.expected_idle}"
            )
        return errors

    def _set_memory_sizes(self) -> None:
        by_codeobj: dict[int, list[CodeEntry]] = {}
        for entry in self.entries.values():
            by_codeobj.setdefault(entry.pc.code_object_id, []).append(entry)
        for entries in by_codeobj.values():
            entries.sort(key=lambda e: e.pc.address)
            for idx, entry in enumerate(entries):
                if idx + 1 < len(entries):
                    delta = entries[idx + 1].pc.address - entry.pc.address
                    entry.memory_size = max(delta, 4)
                else:
                    entry.memory_size = 4


def _int_cell(value: str | None) -> int:
    value = (value or "").strip()
    return int(value, 0) if value else 0


def copy_snapshots(src_dir: str | Path, dst_dir: str | Path) -> None:
    src = Path(src_dir)
    dst = Path(dst_dir)
    snap = src / "snapshots.json"
    if snap.exists() and snap.resolve() != (dst / "snapshots.json").resolve():
        shutil.copy2(snap, dst / "snapshots.json")
    for path in src.glob("source_*"):
        if path.is_file() and path.resolve() != (dst / path.name).resolve():
            shutil.copy2(path, dst / path.name)


def touched_pcs(waves: Iterable[Wave]) -> set[Pc]:
    pcs: set[Pc] = set()
    for wave in waves:
        for inst in wave.instructions:
            if inst.pc.code_object_id == 0 and inst.pc.address == 0:
                continue
            pcs.add(inst.pc)
    return pcs
