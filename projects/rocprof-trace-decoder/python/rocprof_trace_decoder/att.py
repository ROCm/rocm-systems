from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .bindings import Decoder
from .code_index import CodeIndex
from .codegen import CodeObject, generate_code_artifacts
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

        with Decoder(lib_path) as decoder:
            for trace in run_traces:
                records = decoder.parse_file(
                    trace.path,
                    isa=run_code_index if code_index else None,
                )
                if on_warning is not None:
                    for info in records.info:
                        on_warning(trace.path, decoder.info_string(info))
                writer.add_shader_records(trace.shader_engine, records)

        writer.finish()
        write_code_json(run_dir, run_code_index)
        if "csv" in output_formats:
            write_stats_csv(run_dir.parent, run_code_index, run_dir.name)
        if snapshot_sources:
            write_source_snapshots(snapshot_sources, run_dir)
        output_dirs.append(run_dir)

    return output_dirs


def _group_traces(traces: Iterable[AttTrace]) -> dict[int, list[AttTrace]]:
    grouped: dict[int, list[AttTrace]] = defaultdict(list)
    for trace in traces:
        grouped[trace.run].append(trace)
    return dict(sorted(grouped.items()))
