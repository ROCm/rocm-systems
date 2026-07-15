from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .bindings import Decoder
from .code_index import CodeIndex
from .codegen import CodeObject, generate_code_artifacts
from .records import ShaderDataFlags
from .rcv import (
    DEFAULT_RUN_BASE_NAME,
    RcvOutputWriter,
    normalize_output_formats,
    run_output_dir,
    write_code_json,
    write_source_snapshots,
    write_stats_csv,
)


@dataclass(frozen=True)
class AttTrace:
    """A trace file plus metadata needed for viewer output.

    The API intentionally does not derive this metadata from file names. Scripts
    that know a producer's naming convention should parse names before calling
    into this module.
    """

    path: str | Path
    shader_engine: int
    run: int = 1


def generate_att_outputs(
    traces: Iterable[AttTrace],
    *,
    code_index: CodeIndex | None = None,
    code_objects: Iterable[CodeObject] | None = None,
    source_paths: Iterable[str | Path] | None = None,
    output_dir: str | Path | None = None,
    lib_path: str | Path | None = None,
    formats: str | Iterable[str] = "json,csv",
    base_name: str = DEFAULT_RUN_BASE_NAME,
    on_warning: Callable[[Path, str], None] | None = None,
    decode_markers: bool = True,
) -> list[Path]:
    output_formats = normalize_output_formats(formats)
    trace_list = [
        AttTrace(
            path=Path(trace.path).expanduser().resolve(),
            shader_engine=int(trace.shader_engine),
            run=int(trace.run),
        )
        for trace in traces
    ]
    if not trace_list:
        raise ValueError("No ATT traces were provided.")

    if code_index is not None and code_objects is not None:
        raise ValueError("Provide only one of code_index or code_objects.")

    base_dir = Path(output_dir).expanduser().resolve() if output_dir else trace_list[0].path.parent
    snapshot_sources = (
        None
        if source_paths is None
        else tuple(Path(path).expanduser().resolve() for path in source_paths)
    )
    if code_objects is not None:
        artifacts = generate_code_artifacts(code_objects)
        code_index = artifacts.code_index
        if snapshot_sources is None:
            snapshot_sources = artifacts.source_paths

    runs = _group_traces(trace_list)

    output_dirs: list[Path] = []
    for run, run_traces in runs.items():
        run_dir = run_output_dir(
            base_dir,
            base_name,
            run,
            use_root=bool(output_dir and len(runs) == 1),
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        run_code_index = (
            CodeIndex.from_document(code_index.document, load_counts=False)
            if code_index is not None
            else CodeIndex([])
        )
        writer = RcvOutputWriter(run_dir, run_code_index, formats=output_formats)

        decode_json = (
            decode_markers
            and "json" in output_formats
            and bool(run_code_index.document.get("sqtt_funcmap_layout"))
        )
        decoded = [] if decode_json else None
        with Decoder(lib_path) as decoder:
            for trace in run_traces:
                records = decoder.parse_file(
                    trace.path,
                    isa=run_code_index if code_index else None,
                )
                if on_warning is not None:
                    for info in records.info:
                        on_warning(trace.path, decoder.info_string(info))
                if decoded is None:
                    writer.add_shader_records(trace.shader_engine, records)
                else:
                    decoded.append((trace.shader_engine, records))

        if decoded is not None:
            _correct_marker_timestamps(decoded, run_code_index.document)
            for shader_engine, records in decoded:
                writer.add_shader_records(shader_engine, records)

        writer.finish()
        write_code_json(run_dir, run_code_index)
        if "csv" in output_formats:
            write_stats_csv(run_dir.parent, run_code_index, run_dir.name)
        if snapshot_sources:
            write_source_snapshots(snapshot_sources, run_dir)
        output_dirs.append(run_dir)

    return output_dirs


def _correct_marker_timestamps(decoded: list[tuple[int, object]], document: dict) -> None:
    """Correct packed marker headers in-place, leaving legacy streams alone."""
    layouts: dict[int, tuple[int, int]] = {}
    for row in document.get("sqtt_funcmap_layout", []):
        try:
            codeobj, bits, shift = map(int, row[:3])
        except (TypeError, ValueError):
            continue
        if codeobj >= 0 and 0 < bits <= 29 and 0 <= shift and bits + shift <= 32:
            layouts[codeobj] = (bits, shift)
    if not layouts:
        return

    marker_ids: dict[int, set[int]] = defaultdict(set)
    for row in document.get("sqtt_funcmap", []):
        try:
            codeobj, marker_id, kind = int(row[0]), int(row[1]), str(row[2])
        except (IndexError, TypeError, ValueError):
            continue
        if codeobj in layouts and kind.lower() not in {"k", "kernel"} and marker_id >= 0:
            marker_ids[codeobj].add(marker_id)

    payloads: dict[tuple[int, int], int] = {}
    for row in document.get("sqtt_funcmap_payloads", []):
        try:
            codeobj, marker_id, count = map(int, row[:3])
        except (TypeError, ValueError):
            continue
        if marker_id in marker_ids.get(codeobj, ()) and count > 0:
            payloads[codeobj, marker_id] = count

    shaderdata: dict[tuple[int, int, int, int], list[object]] = defaultdict(list)
    occupancy: dict[tuple[int, int, int, int], list[object]] = defaultdict(list)
    for se, records in decoded:
        for record in records.shaderdata:
            shaderdata[se, record.cu, record.simd, record.wave_id].append(record)
        for record in records.occupancy:
            occupancy[se, record.cu, record.simd, record.wave_id].append(record)

    headers: list[tuple[object, int, int, int, list[object]]] = []
    minimum_delay: dict[tuple[int, int], int] = {}
    for location, stream in shaderdata.items():
        se = location[0]
        stream.sort(key=lambda record: record.time)
        intervals = _active_codeobj_intervals(occupancy.get(location, ()))
        interval = 0
        payload_remaining = 0
        header_payloads: list[object] | None = None
        for record in stream:
            if payload_remaining:
                if header_payloads is not None:
                    header_payloads.append(record)
                payload_remaining -= 1
                continue
            if record.flags & ((1 << ShaderDataFlags.IMM) | (1 << ShaderDataFlags.PRIV)):
                continue

            while interval < len(intervals) and intervals[interval][1] < record.time:
                interval += 1
            while interval + 1 < len(intervals) and intervals[interval + 1][0] <= record.time:
                interval += 1
            codeobj = (
                intervals[interval][2]
                if interval < len(intervals) and intervals[interval][0] <= record.time
                else None
            )
            layout = layouts.get(codeobj)
            known_ids = marker_ids.get(codeobj, ())
            if layout is None:
                continue

            bits, shift = layout
            raw = int(record.value) & 0xFFFFFFFF
            marker_id = (raw >> 2) & ((1 << (30 - bits)) - 1)
            if marker_id not in known_ids and not (marker_id == 0 and (raw & 3) == 1):
                continue

            record.value = (marker_id << 2) | (raw & 3)
            header_payloads = []
            if marker_id in known_ids:
                payload_remaining = payloads.get((codeobj, marker_id), 0)
            mask = ((1 << bits) - 1) << shift
            sampled = ((raw >> (32 - bits)) & ((1 << bits) - 1)) << shift
            delay = ((int(record.time) & mask) - sampled) & mask
            headers.append((record, se, mask, delay, header_payloads))
            key = se, mask
            minimum_delay[key] = min(minimum_delay.get(key, delay), delay)

    changed = False
    for record, se, mask, delay, header_payloads in headers:
        correction = delay - minimum_delay[se, mask]
        if correction:
            record.time -= correction
            # The output writer sorts shaderdata after correction. Move raw
            # payloads with their header so declared header/payload blocks
            # remain ordered for readers that consume R: metadata.
            for payload in header_payloads:
                payload.time -= correction
            changed = True
    if changed:
        for _se, records in decoded:
            records.shaderdata.sort(key=lambda record: record.time)


def _active_codeobj_intervals(occupancy: Iterable[object]) -> list[tuple[int, int, int]]:
    events = sorted(occupancy, key=lambda record: (record.time, record.start))
    intervals: list[tuple[int, int, int]] = []
    begin = 0
    codeobj: int | None = None
    for record in events:
        if record.start:
            if codeobj is not None:
                intervals.append((begin, record.time, codeobj))
            begin, codeobj = record.time, record.pc.code_object_id
        elif codeobj is not None:
            intervals.append((begin, record.time, codeobj))
            codeobj = None
    if codeobj is not None:
        intervals.append((begin, (1 << 63) - 1, codeobj))
    return intervals


def _group_traces(traces: Iterable[AttTrace]) -> dict[int, list[AttTrace]]:
    grouped: dict[int, list[AttTrace]] = defaultdict(list)
    for trace in traces:
        grouped[trace.run].append(trace)
    return dict(sorted(grouped.items()))
