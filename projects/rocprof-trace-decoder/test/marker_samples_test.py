#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "samples" / "markers")]

from address_trace import decode_address_traces, decode_block
from flamegraph import folded_stacks
from marker_common import marker_time
from perfetto import trace_events
from rocprof_trace_decoder import (
    Decoder,
    DecoderStatus,
    Marker,
    MarkerFlags,
    MarkerKind,
    MarkerMetadata,
    Pc,
    RecordType,
    ShaderData,
)
from rocprof_trace_decoder.bindings import _Handle, _Marker, _MarkerMetadata


def marker(time: int, marker_id: int, name: str, flags: int) -> Marker:
    return Marker(
        size=ctypes.sizeof(_Marker),
        shaderdata=ShaderData(time, marker_id << 2, 0, 0, 0, 0, 0),
        kernel_entry=Pc(0x100, 7),
        metadata=MarkerMetadata(
            size=ctypes.sizeof(_MarkerMetadata),
            code_object_id=7,
            name=name,
            source_location=None,
            marker_id=marker_id,
            marker_kind=MarkerKind.USER_SCOPE,
            payload_count=0,
            wave_size=32,
            reserved=b"\0\0\0",
        ),
        payload=[],
        marker_flags=flags,
        payload_count=0,
    )


def address_block(name: str, values: list[int]) -> Marker:
    header = marker(10, 1, name, 0)
    header.metadata = replace(
        header.metadata, marker_kind=MarkerKind.POINT, payload_count=len(values)
    )
    header.payload_count = len(values)
    header.payload = list(values)
    return header


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

    def test_new_wave_does_not_join_scopes_in_a_reused_slot(self):
        records = [
            marker(10, 1, "outer", MarkerFlags.ENTER | MarkerFlags.NEW_WAVE),
            marker(20, 2, "next", MarkerFlags.NEW_WAVE),
            marker(30, 2, "next", MarkerFlags.EXIT_PREVIOUS),
        ]
        records[1].metadata = replace(records[1].metadata, marker_kind=MarkerKind.POINT)
        self.assertEqual(folded_stacks(records), {})
        events = trace_events(records)
        self.assertEqual([event["ph"] for event in events], ["B", "E", "i"])
        self.assertEqual(events[0]["tid"], events[1]["tid"])
        self.assertEqual(events[1]["ts"], 20)

    def test_marker_samples_use_public_timestamp(self):
        record = marker(91, 1, "scope", MarkerFlags.ENTER)
        self.assertEqual(marker_time(record), 91)
        self.assertEqual(trace_events([record])[0]["ts"], 91)

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
                trace = decode_block(address_block(name, values))
                self.assertIsNotNone(trace)
                self.assertEqual(trace.addresses, expected)

    def test_address_payload_rejects_incomplete_blocks(self):
        complete = address_block("addr_trace_lds_load", [1, 0, *range(32)])
        incomplete = replace(
            complete,
            marker_flags=MarkerFlags.INCOMPLETE,
            payload=complete.payload[:-1],
            payload_count=len(complete.payload) - 1,
        )
        self.assertIsNone(decode_block(incomplete))
        self.assertIsNone(
            decode_block(replace(complete, metadata=replace(complete.metadata, wave_size=64)))
        )
        self.assertEqual(len(decode_address_traces([incomplete, complete], None)), 1)


class MarkerBindingsTest(unittest.TestCase):
    def make_native_batch(self):
        class ExtendedMarker(ctypes.Structure):
            _fields_ = [("base", _Marker), ("future_field", ctypes.c_uint64)]

        metadata = _MarkerMetadata()
        metadata.size = ctypes.sizeof(metadata)
        metadata.code_object_id = 7
        metadata.name = b"point"
        metadata.marker_id = 42
        metadata.marker_kind = 123  # A future kind must survive conversion.
        metadata.payload_count = 2
        metadata.wave_size = 32
        payload = (ctypes.c_uint32 * 2)(1234, 5678)
        native = (ExtendedMarker * 2)()
        for index, entry in enumerate(native):
            entry.base.size = ctypes.sizeof(ExtendedMarker)
            entry.base.shaderdata.time = index * 100
            entry.base.metadata = ctypes.pointer(metadata)
            entry.base.payload = payload
            entry.base.payload_count = 2
        return native, metadata, payload

    def test_extended_records_own_payloads_and_share_metadata(self):
        native, metadata, payload = self.make_native_batch()
        decoder = Decoder.__new__(Decoder)
        records = decoder._convert_batch(RecordType.MARKER, ctypes.addressof(native), 2, {})
        self.assertEqual([record.shaderdata.time for record in records], [0, 100])
        self.assertIs(records[0].metadata, records[1].metadata)
        self.assertEqual(records[0].marker_kind, 123)
        self.assertEqual(records[0].payload, [1234, 5678])
        metadata.name = b"changed"
        payload[0] = 0
        self.assertEqual(records[0].name, "point")
        self.assertEqual(records[0].payload[0], 1234)

    def test_batch_callback_can_avoid_collection_and_ignores_new_record_types(self):
        native, metadata, payload = self.make_native_batch()
        decoder = Decoder.__new__(Decoder)
        decoder._closed = False
        decoder._handle = _Handle(1)

        def native_parse(_handle, _data, _size, callback, userdata):
            self.assertEqual(callback(999, None, 1, userdata), DecoderStatus.SUCCESS)
            return callback(RecordType.MARKER, ctypes.addressof(native), 2, userdata)

        decoder._lib = SimpleNamespace(
            rocprof_trace_decoder_set_isa_callback=lambda *_args: DecoderStatus.SUCCESS,
            rocprof_trace_decoder_parse=native_parse,
        )
        batches = []
        records = decoder.parse(
            b"trace", on_batch=lambda kind, batch: batches.append((kind, batch)), collect=False
        )
        self.assertEqual(records.markers, [])
        self.assertEqual(records.batches, [])
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0], RecordType.MARKER)
        self.assertEqual(len(batches[0][1]), 2)
        records = decoder.parse(b"trace")
        self.assertEqual(len(records.markers), 2)


if __name__ == "__main__":
    unittest.main()
