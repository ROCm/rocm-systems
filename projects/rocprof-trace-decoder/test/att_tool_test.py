#!/usr/bin/env python3
"""Focused checks for packed-marker JSON normalization and timestamp correction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from att_tool import build_argparser
from rocprof_trace_decoder.att import AttTrace, _correct_marker_timestamps, generate_att_outputs
from rocprof_trace_decoder.code_index import CodeIndex
from rocprof_trace_decoder.records import Occupancy, Pc, ShaderData, ShaderDataFlags, TraceRecords


def _shaderdata(time: int, value: int, flags: int = 0) -> ShaderData:
    return ShaderData(time, value, cu=1, simd=2, wave_id=3, flags=flags, reserved=0)


def _occupancy(start: int, time: int, codeobj: int = 7) -> Occupancy:
    return Occupancy(Pc(0, codeobj), time, 0, 1, 2, 3, start, 0, 0, 0, 0, 0)


class _Decoder:
    def __init__(self, records: TraceRecords):
        self.records = records

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def parse_file(self, *_args, **_kwargs):
        return self.records


class _Writer:
    records: TraceRecords | None = None

    def __init__(self, *_args, **_kwargs):
        pass

    def add_shader_records(self, _se: int, records: TraceRecords):
        type(self).records = records

    def finish(self):
        pass


class MarkerTimestampCorrectionTest(unittest.TestCase):
    def test_corrects_headers_and_moves_declared_payloads_with_their_header(self):
        first_value = (0x100 << 20) | (3 << 2)
        second_value = (0x100 << 20) | (4 << 2)
        first = _shaderdata(0x1010, first_value)
        second = _shaderdata(0x1080, second_value)
        payload = _shaderdata(0x1090, first_value)
        records = TraceRecords(
            occupancy=[_occupancy(1, 0, 0), _occupancy(0, 0x2000, 0)],
            shaderdata=[first, second, payload],
        )
        document = {
            "sqtt_funcmap": [[0, 3, "P", "first", "", 0], [0, 4, "P", "second", "", 0]],
            "sqtt_funcmap_layout": [[0, 12, 4]],
            "sqtt_funcmap_payloads": [[0, 4, 1]],
        }

        _correct_marker_timestamps([(0, records)], document)

        self.assertEqual([record.time for record in records.shaderdata], [0x1010, 0x1010, 0x1020])
        self.assertIs(records.shaderdata[0], first)
        self.assertIs(records.shaderdata[1], second)
        self.assertIs(records.shaderdata[2], payload)
        self.assertEqual([record.value for record in records.shaderdata], [12, 16, first_value])

    def test_imm_record_is_not_a_packed_header(self):
        record = _shaderdata(0x1010, 3 << 2, flags=1 << ShaderDataFlags.IMM)
        records = TraceRecords(occupancy=[_occupancy(1, 0)], shaderdata=[record])

        _correct_marker_timestamps(
            [(0, records)],
            {"sqtt_funcmap": [[7, 3, "P", "marker"]], "sqtt_funcmap_layout": [[7, 12, 4]]},
        )

        self.assertEqual(record.time, 0x1010)
        self.assertEqual(record.value, 3 << 2)

    def test_legacy_metadata_leaves_records_unchanged(self):
        records = TraceRecords(shaderdata=[_shaderdata(20, 12), _shaderdata(10, 16)])
        original = list(records.shaderdata)

        _correct_marker_timestamps([(0, records)], {"sqtt_funcmap": [[7, 3, "P", "marker"]]})

        self.assertEqual(records.shaderdata, original)

    def test_no_active_code_object_is_not_code_object_zero(self):
        value = (0x100 << 20) | (3 << 2)
        record = _shaderdata(0x1080, value)
        records = TraceRecords(shaderdata=[record])

        _correct_marker_timestamps(
            [(0, records)],
            {"sqtt_funcmap": [[0, 3, "P", "marker"]], "sqtt_funcmap_layout": [[0, 12, 4]]},
        )

        self.assertEqual(record.time, 0x1080)
        self.assertEqual(record.value, value)

    def test_exit_without_a_marker_row_is_decoded(self):
        record = _shaderdata(0x1010, (0x100 << 20) | 1)
        records = TraceRecords(occupancy=[_occupancy(1, 0)], shaderdata=[record])

        _correct_marker_timestamps(
            [(0, records)],
            {
                "sqtt_funcmap": [[7, 0, "K", "kernel", "", 0]],
                "sqtt_funcmap_layout": [[7, 12, 4]],
            },
        )

        self.assertEqual(record.value, 1)

    def test_unknown_packed_values_are_unchanged(self):
        records = TraceRecords(
            occupancy=[_occupancy(1, 0)],
            shaderdata=[_shaderdata(20, 4 << 20), _shaderdata(10, 4 << 20)],
        )
        original = list(records.shaderdata)

        _correct_marker_timestamps(
            [(0, records)],
            {"sqtt_funcmap": [[7, 3, "P", "marker"]], "sqtt_funcmap_layout": [[7, 12, 4]]},
        )

        self.assertEqual(records.shaderdata, original)

    def test_correction_is_default(self):
        value = (0x100 << 20) | (3 << 2)
        records = TraceRecords(
            occupancy=[_occupancy(1, 0)],
            shaderdata=[_shaderdata(0x1080, value), _shaderdata(0x1010, value)],
        )
        code_index = CodeIndex.from_document(
            {
                "sqtt_funcmap": [[7, 3, "P", "marker", "", 0]],
                "sqtt_funcmap_layout": [[7, 12, 4]],
            }
        )

        with TemporaryDirectory() as directory:
            with patch("rocprof_trace_decoder.att.Decoder", return_value=_Decoder(records)), \
                patch("rocprof_trace_decoder.att.RcvOutputWriter", _Writer), \
                patch("rocprof_trace_decoder.att.write_code_json"):
                generate_att_outputs(
                    [AttTrace(Path(directory) / "trace.att", shader_engine=0)],
                    code_index=code_index,
                    output_dir=directory,
                    formats="json",
                )

        self.assertEqual([record.time for record in records.shaderdata], [0x1010, 0x1010])
        self.assertEqual([record.value for record in records.shaderdata], [12, 12])

    def test_no_decode_markers_preserves_packed_values(self):
        value = (0x100 << 20) | (3 << 2)
        records = TraceRecords(
            occupancy=[_occupancy(1, 0)],
            shaderdata=[_shaderdata(0x1080, value), _shaderdata(0x1010, value)],
        )
        code_index = CodeIndex.from_document(
            {
                "sqtt_funcmap": [[7, 3, "P", "marker", "", 0]],
                "sqtt_funcmap_layout": [[7, 12, 4]],
            }
        )

        _Writer.records = None
        with TemporaryDirectory() as directory:
            with patch("rocprof_trace_decoder.att.Decoder", return_value=_Decoder(records)), \
                patch("rocprof_trace_decoder.att.RcvOutputWriter", _Writer), \
                patch("rocprof_trace_decoder.att.write_code_json"):
                generate_att_outputs(
                    [AttTrace(Path(directory) / "trace.att", shader_engine=0)],
                    code_index=code_index,
                    output_dir=directory,
                    formats="json",
                    decode_markers=False,
                )

        self.assertIs(_Writer.records, records)
        self.assertEqual([record.time for record in records.shaderdata], [0x1080, 0x1010])
        self.assertEqual([record.value for record in records.shaderdata], [value, value])

    def test_no_decode_markers_cli_flag(self):
        parser = build_argparser()
        self.assertTrue(parser.parse_args(["trace.att"]).decode_markers)
        self.assertFalse(parser.parse_args(["--no-decode-markers", "trace.att"]).decode_markers)


if __name__ == "__main__":
    unittest.main()
