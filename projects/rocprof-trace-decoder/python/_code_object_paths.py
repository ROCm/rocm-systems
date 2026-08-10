from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from rocprof_trace_decoder.codegen import CodeObject

CODE_OBJECT_ID_RE = re.compile(r"(?:^|_)code_object_id_(\d+)(?:_|$)", re.IGNORECASE)


def code_objects_from_paths(paths: Iterable[str | Path]) -> list[CodeObject]:
    parsed: list[tuple[Path, int | None]] = []
    for raw_path in paths:
        path = Path(raw_path)
        parsed.append((path.expanduser().resolve(), _code_object_id_from_path(path)))

    untagged = [str(path) for path, code_object_id in parsed if code_object_id is None]
    parsed_ids = {
        code_object_id for _path, code_object_id in parsed if code_object_id is not None
    }

    if len(untagged) > 1:
        raise ValueError(
            "Cannot infer code object IDs for multiple unnamed inputs: " + ", ".join(untagged)
        )
    if untagged and 0 in parsed_ids:
        raise ValueError(
            f"Cannot assign code object id 0 to {untagged[0]}: another input already uses id 0."
        )

    return [
        CodeObject(
            path=path,
            code_object_id=code_object_id if code_object_id is not None else 0,
        )
        for path, code_object_id in parsed
    ]


def _code_object_id_from_path(path: Path) -> int | None:
    stem = path.stem
    match = CODE_OBJECT_ID_RE.search(stem)
    if match:
        return int(match.group(1))

    pos = stem.rfind("_")
    if pos == -1:
        return None
    try:
        return int(stem[pos + 1 :])
    except ValueError:
        return None
