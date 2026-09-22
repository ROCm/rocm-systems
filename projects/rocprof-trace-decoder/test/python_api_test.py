#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rocprof_trace_decoder import (
    CodeIndex,
    Decoder,
    DecoderStatus,
    InstCategory,
    __version__,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the rocprof_trace_decoder Python API")
    parser.add_argument("--lib", help="Path to librocprof-trace-decoder")
    args = parser.parse_args()

    assert __version__
    assert InstCategory.VALU.name == "VALU"
    assert CodeIndex([]).isa_for_pc

    lib_path = Path(args.lib) if args.lib else None
    if lib_path is not None and not lib_path.is_file():
        raise FileNotFoundError(lib_path)

    with Decoder(lib_path) as decoder:
        status = decoder.status_string(DecoderStatus.SUCCESS)
        if not status:
            raise RuntimeError("Decoder returned an empty SUCCESS status string")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
