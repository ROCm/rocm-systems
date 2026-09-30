#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``amdsmi_get_cper_json`` with the C library faked at the wrapper boundary."""

import ctypes
import unittest
from unittest import mock

from common.common import amdsmi

_wrapper = amdsmi.amdsmi_wrapper
_REPORT = b'{"schema_version": "3.3.0"}\0'


def _c_signature(cper_buffer, buf_size, json_buffer, json_buffer_size):
    """Signature stub for ``create_autospec``."""


def _fake_report(cper_buffer, buf_size, json_buffer, json_buffer_size):
    """Follow the C contract: NULL or too small reports the size, otherwise writes the text."""
    capacity = json_buffer_size._obj
    if json_buffer is None:
        capacity.value = len(_REPORT)
        return _wrapper.AMDSMI_STATUS_SUCCESS
    if capacity.value < len(_REPORT):
        capacity.value = len(_REPORT)
        return _wrapper.AMDSMI_STATUS_INSUFFICIENT_SIZE
    ctypes.memmove(json_buffer, _REPORT, len(_REPORT))
    capacity.value = len(_REPORT)
    return _wrapper.AMDSMI_STATUS_SUCCESS


def _patched(side_effect):
    fake = mock.create_autospec(_c_signature, side_effect=side_effect)
    return mock.patch.object(_wrapper, "amdsmi_get_cper_json", fake), fake


class TestAmdSmiGetCperJson(unittest.TestCase):
    def test_returns_report_text(self):
        patcher, _ = _patched(_fake_report)
        with patcher:
            report = amdsmi.amdsmi_get_cper_json(b"CPER-record")
        self.assertEqual(report, '{"schema_version": "3.3.0"}')

    def test_sizes_the_buffer_with_a_null_first_call(self):
        patcher, fake = _patched(_fake_report)
        with patcher:
            amdsmi.amdsmi_get_cper_json(b"CPER-record")
        self.assertEqual(fake.call_count, 2)
        self.assertIsNone(fake.call_args_list[0].args[2])
        self.assertIsNotNone(fake.call_args_list[1].args[2])

    def test_passes_the_record_bytes(self):
        patcher, fake = _patched(_fake_report)
        record = b"CPER-record"
        with patcher:
            amdsmi.amdsmi_get_cper_json(record)
        for call in fake.call_args_list:
            self.assertEqual(ctypes.string_at(call.args[0], len(record)), record)

    def test_second_call_failure_raises(self):
        # The sizing call succeeds; the call that fills the buffer fails.
        def fail_fill(cper_buffer, buf_size, json_buffer, json_buffer_size):
            if json_buffer is None:
                json_buffer_size._obj.value = len(_REPORT)
                return _wrapper.AMDSMI_STATUS_SUCCESS
            return _wrapper.AMDSMI_STATUS_INSUFFICIENT_SIZE

        patcher, fake = _patched(fail_fill)
        with patcher:
            with self.assertRaises(amdsmi.AmdSmiLibraryException) as ctx:
                amdsmi.amdsmi_get_cper_json(b"CPER-record")
        self.assertEqual(ctx.exception.get_error_code(), _wrapper.AMDSMI_STATUS_INSUFFICIENT_SIZE)
        self.assertEqual(fake.call_count, 2)

    def test_passes_the_record_length(self):
        patcher, fake = _patched(_fake_report)
        record = b"CPER-record"
        with patcher:
            amdsmi.amdsmi_get_cper_json(record)
        self.assertEqual(fake.call_args_list[0].args[1].value, len(record))

    def test_library_failure_raises_with_its_status(self):
        patcher, _ = _patched(lambda *_: _wrapper.AMDSMI_STATUS_UNEXPECTED_DATA)
        with patcher:
            with self.assertRaises(amdsmi.AmdSmiLibraryException) as ctx:
                amdsmi.amdsmi_get_cper_json(b"CPER-record")
        self.assertEqual(ctx.exception.get_error_code(), _wrapper.AMDSMI_STATUS_UNEXPECTED_DATA)

    def test_sizing_call_failure_raises_before_any_buffer_is_read(self):
        # Only the first (sizing) call fails; the second would succeed on an empty buffer.
        def fail_sizing(cper_buffer, buf_size, json_buffer, json_buffer_size):
            if json_buffer is None:
                return _wrapper.AMDSMI_STATUS_UNEXPECTED_DATA
            return _wrapper.AMDSMI_STATUS_SUCCESS

        patcher, fake = _patched(fail_sizing)
        with patcher:
            with self.assertRaises(amdsmi.AmdSmiLibraryException):
                amdsmi.amdsmi_get_cper_json(b"CPER-record")
        self.assertEqual(fake.call_count, 1)

    def test_rejects_non_bytes_without_calling_the_library(self):
        patcher, fake = _patched(_fake_report)
        with patcher:
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                amdsmi.amdsmi_get_cper_json("not bytes")
        fake.assert_not_called()


def _afid_signature(cper_buffer, buf_size, afid_array, num_afids):
    """Signature stub for ``create_autospec``."""


def _fake_afids(total):
    """Follow the C contract: write up to the capacity, always report the full count."""

    def fake(cper_buffer, buf_size, afid_array, num_afids):
        capacity = num_afids._obj.value
        for i in range(min(capacity, total)):
            afid_array[i] = 100 + i
        num_afids._obj.value = total
        return _wrapper.AMDSMI_STATUS_SUCCESS

    return fake


class TestAmdSmiGetAfidsFromCper(unittest.TestCase):
    def _call(self, total):
        fake = mock.create_autospec(_afid_signature, side_effect=_fake_afids(total))
        with mock.patch.object(_wrapper, "amdsmi_get_afids_from_cper", fake):
            result = amdsmi.amdsmi_get_afids_from_cper(b"CPER-record")
        return result, fake

    def test_returns_afids_within_the_default_capacity(self):
        (afids, count), fake = self._call(3)
        self.assertEqual(afids, [100, 101, 102])
        self.assertEqual(count, 3)
        self.assertEqual(fake.call_count, 1)

    def test_grows_the_array_when_the_library_reports_a_larger_count(self):
        total = amdsmi.amdsmi_interface.AMDSMI_MAX_NUMBER_OF_AFIDS_PER_RECORD + 5
        (afids, count), fake = self._call(total)
        self.assertEqual(afids, [100 + i for i in range(total)])
        self.assertEqual(count, total)
        self.assertEqual(fake.call_count, 2)

    def test_second_overrun_raises_instead_of_looping(self):
        # A library that reports a larger count on every call is misbehaving.
        calls = []

        def always_larger(cper_buffer, buf_size, afid_array, num_afids):
            calls.append(1)
            num_afids._obj.value = num_afids._obj.value + 1
            return _wrapper.AMDSMI_STATUS_SUCCESS

        fake = mock.create_autospec(_afid_signature, side_effect=always_larger)
        with mock.patch.object(_wrapper, "amdsmi_get_afids_from_cper", fake):
            with self.assertRaises(amdsmi.AmdSmiLibraryException) as ctx:
                amdsmi.amdsmi_get_afids_from_cper(b"CPER-record")
        self.assertEqual(ctx.exception.get_error_code(), _wrapper.AMDSMI_STATUS_INTERNAL_EXCEPTION)
        self.assertEqual(len(calls), 2)

    def test_library_failure_raises_with_its_status(self):
        fake = mock.create_autospec(
            _afid_signature, side_effect=lambda *_: _wrapper.AMDSMI_STATUS_UNEXPECTED_DATA
        )
        with mock.patch.object(_wrapper, "amdsmi_get_afids_from_cper", fake):
            with self.assertRaises(amdsmi.AmdSmiLibraryException) as ctx:
                amdsmi.amdsmi_get_afids_from_cper(b"CPER-record")
        self.assertEqual(ctx.exception.get_error_code(), _wrapper.AMDSMI_STATUS_UNEXPECTED_DATA)
