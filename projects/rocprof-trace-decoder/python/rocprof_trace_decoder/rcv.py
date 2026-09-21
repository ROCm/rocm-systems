from __future__ import annotations

import json
import shutil
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from pathlib import Path

from .code_index import CodeIndex
from .records import (
    Dispatch,
    Event,
    EventType,
    Occupancy,
    OtherSimdInstruction,
    Pc,
    PerfEvent,
    Realtime,
    ShaderData,
    TraceRecords,
    Wave,
)

TOOL_VERSION = "3.1.0"
DEFAULT_RUN_BASE_NAME = "att"
VALID_OUTPUT_FORMATS = frozenset({"csv", "json"})


def normalize_output_formats(formats: str | Iterable[str]) -> frozenset[str]:
    if isinstance(formats, str):
        raw_items = [formats]
    else:
        raw_items = formats

    normalized = frozenset(
        part.strip().lower() for item in raw_items for part in str(item).split(",") if part.strip()
    )
    if not normalized:
        raise ValueError("No output formats requested.")

    unknown = normalized - VALID_OUTPUT_FORMATS
    if unknown:
        valid = ", ".join(sorted(VALID_OUTPUT_FORMATS))
        invalid = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unsupported output format(s): {invalid}. Expected one or more of: {valid}"
        )

    return normalized


class RcvOutputWriter:
    """Writes ROCprof Compute Viewer sidecar files."""

    def __init__(
        self,
        output_dir: Path,
        code_index: CodeIndex,
        *,
        formats: str | Iterable[str],
    ):
        self.output_dir = output_dir
        self.code_index = code_index
        self.formats = normalize_output_formats(formats)
        self.gfxip = 9
        self.wave_names: dict[int, dict[int, dict[int, dict[int, list[object]]]]] = {}
        self.wave_counts: dict[tuple[int, int, int], int] = defaultdict(int)
        self.occupancy: dict[int, list[Occupancy]] = defaultdict(list)
        self.perf: dict[int, list[PerfEvent]] = defaultdict(list)
        self.events: dict[int, list[Event]] = defaultdict(list)
        self.dispatches: dict[int, list[Dispatch]] = defaultdict(list)
        self.realtime: dict[int, list[Realtime]] = defaultdict(list)
        self.realtime_frequency = 0
        self.other_simd_files: dict[int, list[list[object]]] = defaultdict(list)
        self.shaderdata_files: dict[int, list[list[object]]] = defaultdict(list)
        self.kernel_ids: dict[Pc, int] = {Pc(0, 0): 0}
        self.kernel_names = _kernel_names(code_index.document)

    def add_shader_records(self, se: int, records: TraceRecords) -> None:
        if records.gfxip is not None:
            self.gfxip = records.gfxip
        if records.realtime_frequency is not None:
            self.realtime_frequency = records.realtime_frequency

        self.occupancy[se].extend(records.occupancy)
        self.perf[se].extend(records.perf_events)
        self.events[se].extend(records.events)
        self.dispatches[se].extend(records.dispatches)
        self.realtime[se].extend(records.realtime)

        if records.perf_events and self._want_json:
            self._write_perf(se, records.perf_events)
        if records.other_simd and self._want_json:
            self._write_other_simd(se, records.other_simd)
        if records.shaderdata and self._want_json:
            self._write_shaderdata(se, records.shaderdata)

        for wave in records.waves:
            self.code_index.accumulate_wave(wave)
            self._write_wave(se, wave)

    def finish(self) -> None:
        if not self._want_json:
            return
        self._write_filenames()
        self._write_occupancy()
        self._write_realtime()

    @property
    def _want_json(self) -> bool:
        return "json" in self.formats

    def _write_wave(self, se: int, wave: Wave) -> None:
        if not self._want_json:
            return
        if not wave.instructions and len(wave.timeline) < 3:
            return

        key = (se, wave.simd, wave.wave_id)
        wave_id = self.wave_counts[key]
        self.wave_counts[key] += 1
        filename = f"se{se}_sm{wave.simd}_sl{wave.wave_id}_wv{wave_id}.json"

        self.wave_names.setdefault(se, {}).setdefault(wave.simd, {}).setdefault(wave.wave_id, {})[
            wave_id
        ] = [filename, wave.begin_time, wave.end_time]

        instructions = []
        for inst in wave.instructions:
            line_number = self.code_index.get_or_create(inst.pc).line_number
            instructions.append([inst.time, inst.category, inst.stall, inst.duration, line_number])

        timeline = []
        for state in wave.timeline:
            timeline.append([state.type, state.duration])

        data = {
            "name": f"SE{se}",
            "duration": wave.end_time - wave.begin_time,
            "wave": {
                "cu": wave.cu,
                "id": wave_id,
                "simd": wave.simd,
                "slot": wave.wave_id,
                "begin": wave.begin_time,
                "end": wave.end_time,
                "instructions": instructions,
                "timeline": timeline,
                "waitcnt": [],
            },
            "num_stitched": len(wave.instructions),
            "num_insts": len(wave.instructions),
        }
        _write_json(self.output_dir / filename, data)

    def _write_perf(self, se: int, events: list[PerfEvent]) -> None:
        data = {
            "data": [
                [e.time, e.events0, e.events1, e.events2, e.events3, e.cu, e.bank] for e in events
            ]
        }
        _write_json(self.output_dir / f"se{se}_perfcounter.json", data)

    def _write_other_simd(self, se: int, records: list[OtherSimdInstruction]) -> None:
        begin = records[0].time
        end = records[-1].time + records[-1].cycles
        idx = len(self.other_simd_files[se])
        filename = f"other_simd_se{se}_{idx}.json"
        self.other_simd_files[se].append([filename, begin, end])
        data = {
            "type": "OTHER_SIMD_INSTRUCTIONS",
            "begin_time": begin,
            "end_time": end,
            "wgp": records[0].wgp,
            "instructions_schema": ["time", "duration", "category"],
            "instructions_count": len(records),
            "instructions": [[r.time, r.cycles, r.category] for r in records],
        }
        _write_json(self.output_dir / filename, data)

    def _write_shaderdata(self, se: int, records: list[ShaderData]) -> None:
        begin = records[0].time
        end = records[-1].time
        idx = len(self.shaderdata_files[se])
        filename = f"shaderdata_{se}_{idx}.json"
        self.shaderdata_files[se].append([filename, begin, end])
        data = {
            "type": "shaderdata",
            "begin_time": begin,
            "end_time": end,
            "records_schema": ["time", "value", "cu", "simd", "wave_id", "flags"],
            "records_count": len(records),
            "records": [[r.time, r.value, r.cu, r.simd, r.wave_id, r.flags] for r in records],
        }
        _write_json(self.output_dir / filename, data)

    def _write_filenames(self) -> None:
        data = {
            "global_begin_time": 0,
            "gfxv": "navi" if self.gfxip > 9 else "vega",
            "gfxip": self.gfxip,
            "version": TOOL_VERSION,
            "counter_names": [],
            "wave_filenames": _stringify_nested(self.wave_names),
            "other_simd_filenames": {str(k): v for k, v in self.other_simd_files.items()},
            "shaderdata_filenames": {str(k): v for k, v in self.shaderdata_files.items()},
        }
        _write_json(self.output_dir / "filenames.json", data)

    def _write_occupancy(self) -> None:
        data: dict[str, object] = {
            "occupancy_fields": [
                "time",
                "cu",
                "simd",
                "wave_id",
                "start",
                "kernel_id",
                "me_id",
                "pipe_id",
                "is_ext",
                "workgroup_id",
                "cluster_id",
            ],
            "events": {},
            "dispatches": {},
            "version": TOOL_VERSION,
        }

        for se, events in self.occupancy.items():
            rows = []
            for event in events:
                rows.append(
                    [
                        event.time,
                        event.cu,
                        event.simd,
                        event.wave_id,
                        event.start,
                        self._kernel_id(event.pc),
                        event.me_id,
                        event.pipe_id,
                        event.is_ext,
                        event.workgroup_id,
                        event.cluster_id,
                    ]
                )
            data[str(se)] = rows

        timelines = {}
        for se, dispatches in self.dispatches.items():
            timelines.setdefault(se, [])
            for dispatch in dispatches:
                timelines[se].append((dispatch.time, self._dispatch_json(dispatch)))
        for se, events in self.events.items():
            timelines.setdefault(se, [])
            for event in events:
                timelines[se].append((event.time, self._event_json(event)))

        data["events"] = {
            str(se): [entry for _time, entry in sorted(items, key=lambda x: x[0])]
            for se, items in timelines.items()
        }
        data["dispatches"] = {
            str(kernel_id): self._kernel_name(pc) for pc, kernel_id in self.kernel_ids.items()
        }
        _write_json(self.output_dir / "occupancy.json", data)

    def _write_realtime(self) -> None:
        if not any(self.realtime.values()):
            return
        data: dict[str, object] = {
            "metadata": {
                "descriptor": "[gfx_clock, realtime_clock]",
                "frequency": self.realtime_frequency,
            }
        }
        for se, events in self.realtime.items():
            data[f"SE{se}"] = [[e.shader_clock, e.realtime_clock] for e in events]
        _write_json(self.output_dir / "realtime.json", data)

    def _kernel_id(self, pc: Pc) -> int:
        if pc in self.kernel_ids:
            return self.kernel_ids[pc]
        self.kernel_ids[pc] = len(self.kernel_ids)
        return self.kernel_ids[pc]

    def _kernel_name(self, pc: Pc) -> str:
        return self.kernel_names.get(pc, _default_kernel_name(pc))

    def _dispatch_json(self, dispatch: Dispatch) -> dict[str, object]:
        kernel_id = self._kernel_id(dispatch.entry_point)
        return {
            "kind": "dispatch",
            "time": dispatch.time,
            "me_id": dispatch.me_id,
            "pipe_id": dispatch.pipe_id,
            "kernel_id": kernel_id,
            "kernel_name": self._kernel_name(dispatch.entry_point),
            "entry_point": {
                "address": dispatch.entry_point.address,
                "code_object_id": dispatch.entry_point.code_object_id,
            },
            "user_sgprs": dispatch.user_sgprs,
            "vgprs": dispatch.vgprs,
            "sgprs": dispatch.sgprs,
            "lds_size": dispatch.lds_size,
            "thread_dim_x": dispatch.thread_dim_x,
            "thread_dim_y": dispatch.thread_dim_y,
            "thread_dim_z": dispatch.thread_dim_z,
            "dispatch_pkt_addr": dispatch.dispatch_pkt_addr,
            "byte_offset": dispatch.byte_offset,
            "flags": dispatch.flags,
        }

    def _event_json(self, event: Event) -> dict[str, object]:
        out = {
            "kind": "event",
            "time": event.time,
            "type": event.type,
            "me_id": event.me_id,
            "pipe_id": event.pipe_id,
            "flags": event.flags,
            "payload": event.payload.raw,
            "byte_offset": event.byte_offset,
        }
        if event.type in (int(EventType.CODE_OBJECT_LOAD), int(EventType.CODE_OBJECT_UNLOAD)):
            out["code_object_id"] = event.payload.code_object_id
        if event.type == int(EventType.CLUSTER_BARRIER):
            out["cluster_id"] = event.payload.cluster_id
            out["barrier_id"] = event.payload.barrier_id
        return out


def _stringify_nested(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _stringify_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_nested(v) for v in value]
    return value


def _kernel_names(document: dict) -> dict[Pc, str]:
    out: dict[Pc, str] = {}
    for kernel in document.get("kernels", []):
        pc = Pc(
            address=int(kernel.get("address", -1)),
            code_object_id=int(kernel.get("codeobj", -1)),
        )
        name = kernel.get("name") or kernel.get("demangled")
        out[pc] = str(name or _default_kernel_name(pc))
    return out


def _default_kernel_name(pc: Pc) -> str:
    return f"{pc.code_object_id} / 0x{pc.address:x}"


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2))


def run_output_dir(
    output_root: str | Path,
    base_name: str,
    run: int,
    *,
    use_root: bool = False,
) -> Path:
    root = Path(output_root)
    if use_root:
        return root
    return root / f"ui_output_{base_name}{run}"


def write_code_json(output_dir: str | Path, code_index: CodeIndex) -> None:
    code_index.write_code_json(Path(output_dir) / "code.json")


def write_stats_csv(output_dir: str | Path, code_index: CodeIndex, run_dir_name: str) -> None:
    code_index.write_stats_csv(Path(output_dir) / f"stats_{run_dir_name}.csv")


def write_source_snapshots(source_paths: Iterable[str | Path], output_dir: str | Path) -> int:
    """Write RCV source snapshots and return the number of copied files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tree: dict = {}
    seen: OrderedDict[Path, None] = OrderedDict()
    for raw in source_paths:
        path = Path(raw)
        if path and path.is_file():
            seen.setdefault(path, None)

    for idx, source in enumerate(seen):
        filename = f"source_{idx}_{source.name}"
        shutil.copyfile(source, out_dir / filename)
        _insert_snapshot_path(tree, source, filename)

    if seen:
        _write_json(out_dir / "snapshots.json", tree)
    return len(seen)


def copy_source_snapshots(src_dir: str | Path, dst_dir: str | Path) -> None:
    src = Path(src_dir)
    dst = Path(dst_dir)
    snap = src / "snapshots.json"
    if snap.exists() and snap.resolve() != (dst / "snapshots.json").resolve():
        shutil.copy2(snap, dst / "snapshots.json")
    for path in src.glob("source_*"):
        if path.is_file() and path.resolve() != (dst / path.name).resolve():
            shutil.copy2(path, dst / path.name)


def _insert_snapshot_path(tree: dict, source: Path, filename: str) -> None:
    node = tree
    for idx, part in enumerate(source.parts):
        if idx == len(source.parts) - 1:
            node[part] = filename
        else:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                return
            node = child
