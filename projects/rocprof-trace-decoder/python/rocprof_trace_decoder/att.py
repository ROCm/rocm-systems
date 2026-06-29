from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .bindings import Decoder
from .code_index import CodeIndex, copy_snapshots
from .rcv import RcvOutputWriter, normalize_output_formats


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


class _UnknownIsa:
    def isa_for_pc(self, _pc: object) -> tuple[str, int]:
        return "", 4


def generate_att_outputs(
    traces: Iterable[AttTrace],
    *,
    code_json: str | Path | None = None,
    output_dir: str | Path | None = None,
    lib_path: str | Path | None = None,
    formats: str | Iterable[str] = "json,csv",
    base_name: str = "att",
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

    code_json_path = Path(code_json).expanduser().resolve() if code_json else None
    if code_json_path is not None and not code_json_path.exists():
        raise FileNotFoundError(code_json_path)

    base_dir = Path(output_dir).expanduser().resolve() if output_dir else trace_list[0].path.parent
    runs = _group_traces(trace_list)

    output_dirs: list[Path] = []
    for run, run_traces in runs.items():
        if output_dir and len(runs) == 1:
            run_dir = base_dir
        else:
            run_dir = base_dir / f"ui_output_{base_name}{run}"
        run_dir.mkdir(parents=True, exist_ok=True)

        code_index = CodeIndex.from_code_json(code_json_path) if code_json_path else CodeIndex([])
        writer = RcvOutputWriter(run_dir, code_index, formats=output_formats)

        with Decoder(lib_path) as decoder:
            for trace in run_traces:
                isa = code_index if code_json_path else _UnknownIsa()
                records = decoder.parse_file(trace.path, isa=isa)
                if on_warning is not None:
                    for info in records.info:
                        on_warning(trace.path, decoder.info_string(info))
                writer.add_shader_records(trace.shader_engine, records)

        writer.finish()
        code_index.write_code_json(run_dir / "code.json")
        if "csv" in output_formats:
            code_index.write_stats_csv(run_dir.parent / f"stats_{run_dir.name}.csv")
        if code_json_path:
            copy_snapshots(code_json_path.parent, run_dir)
        output_dirs.append(run_dir)

    return output_dirs


def _group_traces(traces: Iterable[AttTrace]) -> dict[int, list[AttTrace]]:
    grouped: dict[int, list[AttTrace]] = defaultdict(list)
    for trace in sorted(traces, key=lambda item: (item.run, item.shader_engine, item.path.name)):
        grouped[trace.run].append(trace)
    return dict(sorted(grouped.items()))
