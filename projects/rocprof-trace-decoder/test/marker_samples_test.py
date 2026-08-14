#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "samples" / "markers")]

from address_trace import decode_block
from flamegraph import folded_stacks
from perfetto import trace_events
from rocprof_trace_decoder import (
    Marker,
    MarkerFlags,
    MarkerKind,
    MarkerRecordKind,
    Pc,
    ShaderData,
)


def marker(time: int, marker_id: int, name: str, flags: int) -> Marker:
    return Marker(
        size=104,
        shaderdata=ShaderData(time, marker_id << 2, 0, 0, 0, 0, 0),
        kernel_entry=Pc(0x100, 7),
        code_object_id=7,
        name=name,
        source_location=None,
        marker_id=marker_id,
        record_kind=MarkerRecordKind.HEADER,
        marker_kind=MarkerKind.USER_SCOPE,
        marker_flags=flags,
        payload_index=0xFFFFFFFF,
        payload_count=0,
        delay=0,
        reserved=0,
    )


def address_block(name: str, values: list[int]) -> tuple[Marker, list[Marker]]:
    header = marker(10, 1, name, 0)
    header.marker_kind = MarkerKind.POINT
    header.payload_count = len(values)
    payloads = []
    for index, value in enumerate(values):
        payload = marker(11 + index, 1, name, 0)
        payload.record_kind = MarkerRecordKind.PAYLOAD
        payload.marker_kind = MarkerKind.POINT
        payload.shaderdata.value = value
        payload.payload_index = index
        payload.payload_count = len(values)
        payloads.append(payload)
    return header, payloads


class MarkerSamplesTest(unittest.TestCase):
    def test_folded_stacks_and_perfetto_events(self):
        records = [
            marker(10, 1, "outer", MarkerFlags.ENTER | MarkerFlags.NEW_WAVE),
            marker(20, 2, "inner", MarkerFlags.ENTER),
            marker(30, 2, "inner", MarkerFlags.EXIT_PREVIOUS),
            marker(40, 1, "outer", MarkerFlags.EXIT_PREVIOUS),
        ]

        self.assertEqual(
            folded_stacks(records),
            {
                "outer": 20,
                "outer;inner": 10,
            },
        )
        self.assertEqual([event["ph"] for event in trace_events(records)], ["B", "B", "E", "E"])

    def test_perfetto_closes_open_scopes_at_new_wave(self):
        records = [
            marker(10, 1, "outer", MarkerFlags.ENTER | MarkerFlags.NEW_WAVE),
            marker(20, 2, "next", MarkerFlags.NEW_WAVE),
        ]
        records[1].marker_kind = MarkerKind.POINT
        self.assertEqual([event["ph"] for event in trace_events(records)], ["B", "E", "i"])

    def test_address_payload_protocols(self):
        global_addresses = [0x100000000 + lane for lane in range(32)]
        global_words = [
            word
            for address in global_addresses
            for word in (address & 0xFFFFFFFF, address >> 32)
        ]
        cases = [
            ("addr_trace_lds_load", [0b101, 0, *range(32)], [0, 2]),
            ("addr_trace_load", [0b11, 0, *global_words], global_addresses[:2]),
            (
                "addr_trace_buffer_load",
                [0b11, 0, 0x1000, 0, 0x20, *range(32)],
                [0x1020, 0x1021],
            ),
        ]
        for name, values, expected in cases:
            with self.subTest(name=name):
                trace = decode_block(*address_block(name, values))
                self.assertIsNotNone(trace)
                self.assertEqual(trace.addresses, expected)


if __name__ == "__main__":
    unittest.main()
