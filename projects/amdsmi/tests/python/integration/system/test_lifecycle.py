#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Library init, discovery and version APIs."""

import unittest

import common.api_test as api
import common.common as common

_INIT_FLAGS = [(flag.name, flag, common.PASS) for flag in common.amdsmi.AmdSmiInitFlags]
_PROCESSOR_TYPES = [
    (member.name, member, common.PASS) for member in common.amdsmi.AmdSmiProcessorType
]


class TestSystemLifecycle(api.ApiTestCase):
    def _sockets(self):
        return api.Handle("socket", common.amdsmi.amdsmi_get_socket_handles())

    def test_init(self):
        # Zero selects no processor family and is rejected by the flag mask; a
        # positive re-init belongs to the fixture, not to a test.
        self.reject_only(
            "amdsmi_init",
            api.Param("flag", _INIT_FLAGS[0][:2], [("zero", 0), ("bad-type", api.BAD_INT)]),
        )

    def test_shut_down(self):
        # No arguments and no payload; re-init afterwards so the fixture teardown
        # still has an initialised library to close.
        self.expect_only("amdsmi_shut_down", validate=False)
        self.common.amdsmi_smart_init()

    def test_get_socket_handles(self):
        self.assertGreaterEqual(len(self.expect_only("amdsmi_get_socket_handles")), 1)

    def test_get_socket_info(self):
        self.both("amdsmi_get_socket_info", self._sockets())

    def test_get_processor_handles(self):
        processors = self.expect_only("amdsmi_get_processor_handles")
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), self.common.max_num_physical_devices)

    def test_get_processor_handles_by_type(self):
        self.both(
            "amdsmi_get_processor_handles_by_type",
            self._sockets(),
            api.enum("processor_type", _PROCESSOR_TYPES),
        )

    def test_get_processor_info(self):
        self.both("amdsmi_get_processor_info", self.handle)

    def test_get_processor_type(self):
        self.both("amdsmi_get_processor_type", self.handle)

    def test_get_processor_count_from_handles(self):
        handles = api.Param(
            "processors", ("[all gpus]", self.common.processors), [("bad-type", api.BAD_SEQUENCE)]
        )
        self.both("amdsmi_get_processor_count_from_handles", handles)

    def test_get_processor_handle_from_bdf(self):
        # The rejection path needs no live BDF, so run it before fetching one.
        bad = [("bad-type", api.BAD_STR), ("malformed", "not-a-bdf")]
        self._announce()
        self.api.reject(
            "amdsmi_get_processor_handle_from_bdf",
            api.Param("bdf", ("0000:00:00.0", "0000:00:00.0"), bad),
        )
        bdf = self.prerequisite("amdsmi_get_gpu_device_bdf", self.common.processors[0])
        self.api.expect("amdsmi_get_processor_handle_from_bdf", api.Param("bdf", (bdf, bdf), bad))

    def test_get_lib_version(self):
        self.assertIn(
            "major",
            self.expect_only(
                "amdsmi_get_lib_version", require_success=True, require_populated=True
            ),
        )

    def test_get_rocm_version(self):
        self.expect_only("amdsmi_get_rocm_version")

    def test_status_code_to_string(self):
        status = common.amdsmi.amdsmi_wrapper.amdsmi_status_t(0)
        self.both(
            "amdsmi_status_code_to_string",
            api.Param("status", ("SUCCESS", status), [("invalid", api.BAD_HANDLE)]),
        )

    def test_get_node_handle(self):
        self.both("amdsmi_get_node_handle", self.handle)

    def test_get_npm_info(self):
        # The rejection path needs no node handle, so run it before fetching one.
        self._announce()
        self.api.reject("amdsmi_get_npm_info", api.opaque("node_handle"))
        node = self.prerequisite("amdsmi_get_node_handle", self.common.processors[0])
        self.api.expect(
            "amdsmi_get_npm_info",
            api.Param("node_handle", ("0", node), [("invalid", api.BAD_HANDLE)]),
        )


if __name__ == "__main__":
    unittest.main()
