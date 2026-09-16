#!/usr/bin/env python3
"""Focused checks for packed-marker JSON normalization and timestamp correction."""

from __future__ import annotations

import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

from att_tool import build_argparser
from rocprof_trace_decoder.att import AttTrace, _correct_marker_timestamps, generate_att_outputs
from rocprof_trace_decoder.code_index import CodeIndex
from rocprof_trace_decoder.records import Occupancy, Pc, ShaderData, ShaderDataFlags, TraceRecords
from sqtt_data import FuncMap, ShaderRecord, WaveSpan, find_wave_span_at, preprocess_records
from sqtt_flamegraph import build_stacks, render_flamegraph_svg
from sqtt_perfetto import emit_wave_events


def _shaderdata(time: int, value: int, flags: int = 0, simd: int = 2) -> ShaderData:
    return ShaderData(time, value, cu=1, simd=simd, wave_id=3, flags=flags, reserved=0)


def _occupancy(start: int, time: int, codeobj: int = 7, simd: int = 2) -> Occupancy:
    return Occupancy(Pc(0, codeobj), time, 0, 1, simd, 3, start, 0, 0, 0, 0, 0)


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
    def test_payload_headers_are_normalized_without_clock_correction(self):
        first_value = (0x100 << 20) | (3 << 2)
        second_value = (0x101 << 20) | (4 << 2)
        first = _shaderdata(0x1000, first_value)
        payload = _shaderdata(0x1020, 0xDEADBEEF)
        second = _shaderdata(0x1040, second_value)
        records = TraceRecords(
            occupancy=[_occupancy(1, 0)],
            shaderdata=[first, payload, second],
        )
        document = {
            "sqtt_funcmap": [[7, 3, "P", "first", "", 0], [7, 4, "P", "second", "", 0]],
            "sqtt_funcmap_layout": [[7, 12, 4]],
            "sqtt_funcmap_payloads": [[7, 3, 1]],
        }

        _correct_marker_timestamps([(0, records)], document)

        self.assertEqual([record.time for record in records.shaderdata], [0x1000, 0x1020, 0x1040])
        self.assertIs(records.shaderdata[0], first)
        self.assertIs(records.shaderdata[1], payload)
        self.assertIs(records.shaderdata[2], second)
        self.assertEqual([record.value for record in records.shaderdata], [12, 0xDEADBEEF, 16])

    def test_payload_records_stay_adjacent_when_another_code_object_is_corrected(self):
        header = _shaderdata(0x1000, 3 << 2)
        payload = _shaderdata(0x1020, 0xDEADBEEF)
        first = _shaderdata(0x1040, (0x100 << 20) | (4 << 2))
        second = _shaderdata(0x1080, (0x107 << 20) | (4 << 2))
        records = TraceRecords(
            occupancy=[_occupancy(1, 0, codeobj=7), _occupancy(1, 0x1030, codeobj=8)],
            shaderdata=[header, payload, first, second],
        )

        _correct_marker_timestamps(
            [(0, records)],
            {
                "sqtt_funcmap": [[7, 3, "P", "payload", "", 0], [8, 4, "P", "marker", "", 0]],
                "sqtt_funcmap_layout": [[8, 12, 4]],
                "sqtt_funcmap_payloads": [[7, 3, 1]],
            },
        )

        self.assertEqual([record.time for record in records.shaderdata], [0x1000, 0x1020, 0x101F, 0x1080])
        self.assertEqual([record.value for record in records.shaderdata], [12, 0xDEADBEEF, 16, 16])

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
        early_value = (0x100 << 20) | (3 << 2)
        late_value = (0x101 << 20) | (3 << 2)
        records = TraceRecords(
            occupancy=[_occupancy(1, 0)],
            shaderdata=[_shaderdata(0x1080, late_value), _shaderdata(0x1010, early_value)],
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

        self.assertEqual([record.time for record in records.shaderdata], [0x1010, 0x102F])
        self.assertEqual([record.value for record in records.shaderdata], [12, 12])

    def test_constant_clock_samples_are_normalized_without_correction(self):
        late = _shaderdata(0x1080, 3 << 2)
        early = _shaderdata(0x1010, 4 << 2)
        records = TraceRecords(
            occupancy=[_occupancy(1, 0)], shaderdata=[late, early]
        )

        _correct_marker_timestamps(
            [(0, records)],
            {
                "sqtt_funcmap": [[7, 3, "P", "first", "", 0], [7, 4, "P", "second", "", 0]],
                "sqtt_funcmap_layout": [[7, 12, 4]],
            },
        )

        self.assertEqual([late.time, early.time], [0x1080, 0x1010])
        self.assertEqual([late.value, early.value], [12, 16])

    def test_correction_does_not_share_the_reference_between_simds(self):
        slow_first = _shaderdata(0x1000, (0x100 << 20) | (3 << 2), simd=2)
        slow_second = _shaderdata(0x1080, (0x101 << 20) | (3 << 2), simd=2)
        fast_first = _shaderdata(0x3000, (0x100 << 20) | (3 << 2), simd=3)
        fast_second = _shaderdata(0x3080, (0x101 << 20) | (3 << 2), simd=3)
        records = TraceRecords(
            occupancy=[_occupancy(1, 0, simd=2), _occupancy(1, 0, simd=3)],
            shaderdata=[slow_first, slow_second, fast_first, fast_second],
        )

        _correct_marker_timestamps(
            [(0, records)],
            {
                "sqtt_funcmap": [[7, 3, "P", "marker", "", 0]],
                "sqtt_funcmap_layout": [[7, 12, 4]],
            },
        )

        self.assertEqual([record.time for record in records.shaderdata], [0x1000, 0x101F, 0x3000, 0x301F])
        self.assertEqual([record.value for record in records.shaderdata], [12, 12, 12, 12])

    def test_correction_cancels_unknown_clock_phase(self):
        # The fixed inter-clock phase crosses the sampled window. The
        # correction must use clock/time deltas, not absolute values.
        marker_id = 3 << 2
        delayed = _shaderdata(0x101D0, marker_id)
        reference = _shaderdata(0x10DFC, (0x100 << 20) | marker_id)
        records = TraceRecords(
            occupancy=[_occupancy(1, 0)], shaderdata=[delayed, reference])

        _correct_marker_timestamps(
            [(0, records)],
            {
                "sqtt_funcmap": [[7, 3, "P", "marker", "", 0]],
                "sqtt_funcmap_layout": [[7, 12, 4]],
            },
        )

        self.assertEqual([record.time for record in records.shaderdata], [0xFE0B, 0x10DFC])
        self.assertEqual([record.value for record in records.shaderdata], [marker_id, marker_id])

    def test_packed_bare_exit_after_retirement_is_normalized_and_corrected(self):
        before_retirement = _shaderdata(0x1010, (0x100 << 20) | (3 << 2))
        after_retirement = _shaderdata(0x1080, (0x101 << 20) | 1)
        records = TraceRecords(
            occupancy=[_occupancy(1, 0), _occupancy(0, 0x1020)],
            shaderdata=[before_retirement, after_retirement],
        )

        _correct_marker_timestamps(
            [(0, records)],
            {
                "sqtt_funcmap": [[7, 3, "P", "marker", "", 0]],
                "sqtt_funcmap_layout": [[7, 12, 4]],
            },
        )

        self.assertEqual(before_retirement.time, 0x1010)
        self.assertEqual(after_retirement.time, 0x102F)
        self.assertEqual([record.value for record in records.shaderdata], [12, 1])

    def test_later_launch_supersedes_a_retired_wave_slot(self):
        value = (0x80 << 21) | (4 << 2)
        first = _shaderdata(0x1020, value)
        second = _shaderdata(0x1080, value)
        records = TraceRecords(
            occupancy=[
                _occupancy(1, 0, codeobj=7),
                _occupancy(0, 0x20, codeobj=7),
                _occupancy(1, 0x40, codeobj=8),
            ],
            shaderdata=[first, second],
        )

        _correct_marker_timestamps(
            [(0, records)],
            {
                "sqtt_funcmap": [[7, 3, "P", "first", "", 0], [8, 4, "P", "second", "", 0]],
                "sqtt_funcmap_layout": [[7, 12, 4], [8, 11, 5]],
            },
        )

        self.assertEqual([record.time for record in records.shaderdata], [0x1020, 0x1080])
        self.assertEqual([record.value for record in records.shaderdata], [16, 16])

    def test_multi_payload_headers_are_normalized_without_clock_correction(self):
        value = (0x100 << 20) | (3 << 2)
        header = _shaderdata(0x1080, value)
        payload0 = _shaderdata(0x1084, 0xDEADBEEF)
        payload1 = _shaderdata(0x1088, 0xCAFEBABE)
        records = TraceRecords(
            occupancy=[_occupancy(1, 0)],
            shaderdata=[header, payload0, payload1],
        )

        _correct_marker_timestamps(
            [(0, records)],
            {
                "sqtt_funcmap": [[7, 3, "P", "multi", "", 0]],
                "sqtt_funcmap_layout": [[7, 12, 4]],
                "sqtt_funcmap_payloads": [[7, 3, 2]],
            },
        )

        self.assertEqual(header.value, 12)
        self.assertEqual(header.time, 0x1080)
        self.assertEqual([payload0.time, payload1.time], [0x1084, 0x1088])
        self.assertEqual([payload0.value, payload1.value], [0xDEADBEEF, 0xCAFEBABE])

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


class MarkerPipelineTest(unittest.TestCase):
    def test_late_records_recover_after_an_orphan_and_render(self):
        funcmap = FuncMap(markers={1: ("outer", "function"), 2: ("inner", "function")})
        spans = [WaveSpan(100, 130, 1), WaveSpan(300, 360, 2)]
        wave_spans = {(0, 1, 2, 3): spans}
        dispatches = {"1": "kernel"}
        records = [
            ShaderRecord(160, 1, 1, 2, 3, 0),  # orphan exit after retirement
            ShaderRecord(191, 6, 1, 2, 3, 0),  # enter outer
            ShaderRecord(207, 10, 1, 2, 3, 0), # enter inner
            ShaderRecord(223, 1, 1, 2, 3, 0),  # exit inner
            ShaderRecord(239, 1, 1, 2, 3, 0),  # exit outer
        ]

        self.assertIs(find_wave_span_at(spans, 99), spans[0])
        self.assertIs(find_wave_span_at(spans, 299), spans[0])
        self.assertIs(find_wave_span_at(spans, 300), spans[1])

        markers, addresses = preprocess_records(records, funcmap)
        self.assertFalse(addresses)
        folded = build_stacks(markers, {7: funcmap}, dispatches, wave_spans, {"kernel": 7})
        self.assertEqual(
            folded,
            {
                "code_object_7;kernel;outer": (32, 2),
                "code_object_7;kernel;outer;inner": (16, 1),
            },
        )
        ET.fromstring(render_flamegraph_svg(folded))

        events, unmatched = emit_wave_events(
            markers,
            (0, 1, 2, 3, 0),
            dispatches,
            {"kernel": 7},
            {7: funcmap},
            funcmap,
            wave_spans,
            1.0,
        )
        self.assertEqual(unmatched, 1)
        self.assertEqual([event["ph"] for event in events], ["B", "B", "E", "E"])


if __name__ == "__main__":
    unittest.main()
