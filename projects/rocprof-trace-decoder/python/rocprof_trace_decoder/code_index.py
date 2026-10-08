from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .records import Pc, Wave

CODE_HEADER = "ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle"
CODE_ROW_INST = 0
CODE_ROW_LINE = 2
CODE_ROW_SOURCE = 3
CODE_ROW_CODE_OBJECT = 4
CODE_ROW_VADDR = 5
CODE_ROW_HIT = 6
CODE_ROW_LATENCY = 7
CODE_ROW_STALL = 8
CODE_ROW_IDLE = 9
CODE_ROW_COUNTERS = (CODE_ROW_HIT, CODE_ROW_LATENCY, CODE_ROW_STALL, CODE_ROW_IDLE)
CODE_ROW_WIDTH = CODE_ROW_IDLE + 1

STATS_COL_CODE_OBJECT = "CodeObj"
STATS_COL_VADDR = "Vaddr"
STATS_COL_INSTRUCTION = "Instruction"
STATS_COL_HITCOUNT = "Hitcount"
STATS_COL_LATENCY = "Latency"
STATS_COL_STALL = "Stall"
STATS_COL_IDLE = "Idle"
STATS_COL_SOURCE = "Source"
STATS_CSV_HEADER = [
    STATS_COL_CODE_OBJECT,
    STATS_COL_VADDR,
    STATS_COL_INSTRUCTION,
    STATS_COL_HITCOUNT,
    STATS_COL_LATENCY,
    STATS_COL_STALL,
    STATS_COL_IDLE,
    STATS_COL_SOURCE,
]


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
        self._memory_sizes_dirty = False

        for entry in entries:
            self.entries[entry.pc] = entry
            self.line_numbers[entry.pc] = entry.line_number

        self._next_line_number = max(self.line_numbers.values(), default=0) + 1
        self._mark_memory_sizes_dirty()

    @classmethod
    def from_code_json(cls, path: str | Path, *, load_counts: bool = True) -> "CodeIndex":
        return cls.from_document(json.loads(Path(path).read_text()), load_counts=load_counts)

    @classmethod
    def from_document(cls, doc: dict, *, load_counts: bool = True) -> "CodeIndex":
        document = json.loads(json.dumps(doc))
        if not load_counts:
            _clear_document_counts(document)

        entries: list[CodeEntry] = []

        for row in document.get("code", []):
            if len(row) < CODE_ROW_WIDTH:
                continue
            inst = str(row[CODE_ROW_INST])
            if _is_code_label(inst):
                continue
            pc = Pc(address=int(row[CODE_ROW_VADDR]), code_object_id=int(row[CODE_ROW_CODE_OBJECT]))
            entries.append(
                CodeEntry(
                    pc=pc,
                    inst=inst,
                    line_number=int(row[CODE_ROW_LINE]),
                    source=str(row[CODE_ROW_SOURCE]),
                    hitcount=int(row[CODE_ROW_HIT] or 0) if load_counts else 0,
                    latency=int(row[CODE_ROW_LATENCY] or 0) if load_counts else 0,
                    stall=int(row[CODE_ROW_STALL] or 0) if load_counts else 0,
                    idle=int(row[CODE_ROW_IDLE] or 0) if load_counts else 0,
                )
            )

        return cls(entries, document)

    @classmethod
    def from_stats_csv(cls, paths: Iterable[str | Path]) -> "CodeIndex":
        entries: dict[Pc, CodeEntry] = {}
        for path in paths:
            with Path(path).open("r", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if not row:
                        continue
                    pc = Pc(
                        address=int(row[STATS_COL_VADDR].strip(), 0),
                        code_object_id=int(row[STATS_COL_CODE_OBJECT].strip(), 0),
                    )
                    inst = row[STATS_COL_INSTRUCTION].strip()
                    if _is_code_label(inst):
                        continue

                    entry = entries.setdefault(
                        pc,
                        CodeEntry(
                            pc=pc,
                            inst=inst,
                            line_number=len(entries) + 1,
                            source=row.get(STATS_COL_SOURCE, ""),
                        ),
                    )
                    entry.expected_hitcount += _int_cell(row.get(STATS_COL_HITCOUNT))
                    entry.expected_latency += _int_cell(row.get(STATS_COL_LATENCY))
                    entry.expected_stall += _int_cell(row.get(STATS_COL_STALL))
                    entry.expected_idle += _int_cell(row.get(STATS_COL_IDLE))
        return cls(entries.values())

    def isa_for_pc(self, pc: Pc) -> tuple[str, int] | None:
        entry = self.entries.get(pc)
        if entry is None:
            return None
        self._ensure_memory_sizes()
        return entry.inst, entry.memory_size

    def get_or_create(self, pc: Pc) -> CodeEntry:
        entry = self.entries.get(pc)
        if entry is not None:
            return entry
        entry = CodeEntry(
            pc=pc,
            inst="",
            line_number=self._next_line_number,
        )
        self._next_line_number += 1
        self.entries[pc] = entry
        self.line_numbers[pc] = entry.line_number
        self._mark_memory_sizes_dirty()
        return entry

    def line_number(self, pc: Pc) -> int:
        """Return the source line number for *pc*.

        Raises KeyError if *pc* is not present in the index.
        """
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
        seen: set[Pc] = set()
        for row in self.document.get("code", []):
            if len(row) < CODE_ROW_WIDTH or _is_code_label(str(row[CODE_ROW_INST])):
                rows.append(row)
                continue
            pc = _pc_from_code_row(row)
            seen.add(pc)
            entry = self.entries.get(pc)
            if entry is None:
                rows.append(row)
                continue
            updated = list(row)
            _set_code_row_counts(updated, entry)
            rows.append(updated)
        for entry in sorted(
            (entry for pc, entry in self.entries.items() if pc not in seen),
            key=lambda e: e.line_number,
        ):
            rows.append(_code_row(entry))
        out_doc["code"] = rows
        out_doc.setdefault("header", CODE_HEADER)
        Path(path).write_text(json.dumps(out_doc, indent=2))

    def write_stats_csv(self, path: str | Path) -> None:
        with Path(path).open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(STATS_CSV_HEADER)
            seen: set[Pc] = set()
            for row in self.document.get("code", []):
                if len(row) < CODE_ROW_WIDTH:
                    continue
                pc = _pc_from_code_row(row)
                if _is_code_label(str(row[CODE_ROW_INST])):
                    writer.writerow(
                        [
                            pc.code_object_id,
                            pc.address,
                            row[CODE_ROW_INST],
                            0,
                            0,
                            0,
                            0,
                            row[CODE_ROW_SOURCE],
                        ]
                    )
                    continue
                entry = self.entries.get(pc)
                if entry is not None:
                    _write_stats_row(writer, entry)
                    seen.add(pc)
            for entry in sorted(
                (entry for pc, entry in self.entries.items() if pc not in seen),
                key=lambda e: (e.pc.code_object_id, e.pc.address),
            ):
                _write_stats_row(writer, entry)

    def validate_expected(self) -> list[str]:
        errors = []
        for entry in sorted(
            self.entries.values(),
            key=lambda e: (e.pc.code_object_id, e.pc.address),
        ):
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

    def _mark_memory_sizes_dirty(self) -> None:
        self._memory_sizes_dirty = True

    def _ensure_memory_sizes(self) -> None:
        if self._memory_sizes_dirty:
            self._set_memory_sizes()
            self._memory_sizes_dirty = False

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


def _is_code_label(inst: str) -> bool:
    return inst.startswith(";") or inst.startswith("label")


def _pc_from_code_row(row: list) -> Pc:
    return Pc(
        address=int(row[CODE_ROW_VADDR]),
        code_object_id=int(row[CODE_ROW_CODE_OBJECT]),
    )


def _set_code_row_counts(row: list, entry: CodeEntry) -> None:
    row[CODE_ROW_HIT] = entry.hitcount
    row[CODE_ROW_LATENCY] = entry.latency
    row[CODE_ROW_STALL] = entry.stall
    row[CODE_ROW_IDLE] = entry.idle


def _code_row(entry: CodeEntry) -> list[object]:
    row: list[object] = [0] * CODE_ROW_WIDTH
    row[CODE_ROW_INST] = entry.inst
    row[CODE_ROW_LINE] = entry.line_number
    row[CODE_ROW_SOURCE] = entry.source
    row[CODE_ROW_CODE_OBJECT] = entry.pc.code_object_id
    row[CODE_ROW_VADDR] = entry.pc.address
    _set_code_row_counts(row, entry)
    return row


def _clear_document_counts(document: dict) -> None:
    for row in document.get("code", []):
        if len(row) >= CODE_ROW_WIDTH:
            for column in CODE_ROW_COUNTERS:
                row[column] = 0


def _write_stats_row(writer: object, entry: CodeEntry) -> None:
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
