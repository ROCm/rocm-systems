#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rocprof_trace_decoder import (
    AnalysisFlags,
    CodeIndex,
    Decoder,
    DecoderStatus,
    InstCategory,
    HiddenLatency,
    Pc,
    RecordType,
    __version__,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the rocprof_trace_decoder Python API")
    parser.add_argument("--lib", help="Path to librocprof-trace-decoder")
    args = parser.parse_args()

    assert __version__
    assert InstCategory.VALU.name == "VALU"
    assert CodeIndex([]).isa_for_pc
    assert RecordType.HIDDEN_LATENCY < RecordType.LAST
    assert (
        HiddenLatency(
            size=0, pc=Pc(0, 0), idle=1, stall=2, issue=3, simd=0
        ).total()
        == 6
    )

    lib_path = Path(args.lib) if args.lib else None
    if lib_path is not None and not lib_path.is_file():
        raise FileNotFoundError(lib_path)

    with Decoder(lib_path) as decoder:
        status = decoder.status_string(DecoderStatus.SUCCESS)
        if not status:
            raise RuntimeError("Decoder returned an empty SUCCESS status string")
        decoder.set_analysis(AnalysisFlags.HIDDEN_LATENCY)
        decoder.set_analysis(AnalysisFlags.NONE)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
