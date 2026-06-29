from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .bindings import Decoder
from .code_index import CodeIndex, copy_snapshots
from .rcv import RcvOutputWriter


@dataclass(frozen=True)
class AttTrace:
    """A trace file plus metadata needed for viewer output.

    The API intentionally does not derive this metadata from file names. Scripts
    that know a producer's naming convention should parse names before calling
    into this module.
    """

    path: Path
    shader_engine: int
    run: int = 1


def generate_att_outputs(
    traces: Iterable[AttTrace],
    *,
    code_json: str | Path,
    output_dir: str | Path | None = None,
    lib_path: str | Path | None = None,
    formats: str = "json,csv",
    base_name: str = "att",
) -> list[Path]:
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

    code_json_path = Path(code_json).expanduser().resolve()
    if not code_json_path.exists():
        raise FileNotFoundError(code_json_path)

    base_dir = Path(output_dir).expanduser().resolve() if output_dir else trace_list[0].path.parent
    runs = _group_traces(trace_list)

    output_dirs: list[Path] = []
    for run, run_traces in runs.items():
        run_dir = base_dir if output_dir and len(runs) == 1 else base_dir / f"ui_output_{base_name}{run}"
        run_dir.mkdir(parents=True, exist_ok=True)

        code_index = CodeIndex.from_code_json(code_json_path)
        writer = RcvOutputWriter(run_dir, code_index, formats=formats)

        with Decoder(lib_path) as decoder:
            for trace in run_traces:
                records = decoder.parse_file(trace.path, isa=code_index)
                writer.add_shader_records(trace.shader_engine, records)

        writer.finish()
        code_index.write_code_json(run_dir / "code.json")
        if "csv" in formats.lower():
            code_index.write_stats_csv(run_dir.parent / f"stats_{run_dir.name}.csv")
        copy_snapshots(code_json_path.parent, run_dir)
        output_dirs.append(run_dir)

    return output_dirs


def _group_traces(traces: Iterable[AttTrace]) -> dict[int, list[AttTrace]]:
    grouped: dict[int, list[AttTrace]] = defaultdict(list)
    for trace in sorted(traces, key=lambda item: (item.run, item.shader_engine, item.path.name)):
        grouped[trace.run].append(trace)
    return dict(sorted(grouped.items()))
