#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rocprof_trace_decoder import (
    CodeIndex,
    Decoder,
    DecoderStatus,
    HiddenLatency,
    InstCategory,
    RecordType,
    __version__,
)
from rocprof_trace_decoder.bindings import _find_library


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the rocprof_trace_decoder Python API")
    parser.add_argument("--lib", help="Path to librocprof-trace-decoder")
    parser.add_argument(
        "--expect-lib", help="Fail unless library discovery selects this file"
    )
    args = parser.parse_args()

    # Discovery can silently fall back to an installed decoder, which then
    # tests the wrong library.
    if args.expect_lib:
        found = Path(_find_library(args.lib)).resolve()
        expected = Path(args.expect_lib).resolve()
        if found != expected:
            raise RuntimeError(f"Library discovery selected {found}, expected {expected}")

    assert __version__
    assert InstCategory.VALU.name == "VALU"
    assert CodeIndex([]).isa_for_pc
    assert HiddenLatency(idle=1, stall=2, issue=3).total() == 6

    lib_path = Path(args.lib) if args.lib else None
    if lib_path is not None and not lib_path.is_file():
        raise FileNotFoundError(lib_path)

    with Decoder(lib_path) as decoder:
        status = decoder.status_string(DecoderStatus.SUCCESS)
        if not status:
            raise RuntimeError("Decoder returned an empty SUCCESS status string")
        decoder.set_record_filter(
            [RecordType.MARKER, RecordType.SHADERDATA],
            immediate=[RecordType.MARKER],
        )
        try:
            decoder.set_record_filter(immediate=[RecordType.MARKER])
        except ValueError:
            pass
        else:
            raise RuntimeError("record filter accepted immediate types without an explicit record set")
        decoder.set_record_filter()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
