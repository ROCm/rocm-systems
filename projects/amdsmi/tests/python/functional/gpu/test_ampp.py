#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""GPU AMPP (amdsmi power profile): get_ampp_profiles, get_ampp_fields,
activate_ampp_profile, and configure_ampp_profile (exercised via
AMDSMI_DRY_RUN)."""

import os
import unittest

import common.common as common
from common.common import amdsmi


class TestGpuAmpp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(common.verbose)

    @classmethod
    def tearDownClass(cls):
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException:
            pass

    def setUp(self):
        self.raise_exception = None
        self.common.amdsmi_smart_init()
        self.common.processors = amdsmi.amdsmi_get_processor_handles()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_get_ampp_profiles(self):
        """Test AMPP profile enumeration and returned dict shape"""
        self.common.print_func_name("")
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)

        for i in range(0, len(processors)):
            self.common.print_device_header(i)

            msg = f"\t### amdsmi_get_ampp_profiles(gpu={i}):"
            try:
                version, profiles = amdsmi.amdsmi_get_ampp_profiles(processors[i])
                self.common.print(msg, (version, profiles))
                self.common.check_ret("", "", self.common.PASS)
            except amdsmi.AmdSmiLibraryException as e:
                self.common.print(msg, e)
                self.assertEqual(
                    e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED, msg
                )
                continue

            # version is a single tree-wide app_modes/profile_abi string
            # (e.g. "1.0"), not a per-profile value.
            self.assertIsInstance(version, str)

            active_count = 0
            for profile in profiles:
                for key in ("name", "index", "is_active", "is_writable", "is_configured"):
                    self.assertIn(key, profile)
                self.assertTrue(profile["name"].startswith("profile_"))
                if profile["is_active"]:
                    active_count += 1

            # Exactly one profile is active whenever any profiles exist.
            if profiles:
                self.assertEqual(active_count, 1)

    def test_get_ampp_fields(self):
        """Test AMPP field enumeration for configured / unconfigured slots
        and confirm control directories (not real profiles) are rejected."""
        self.common.print_func_name("")
        processors = amdsmi.amdsmi_get_processor_handles()

        for i in range(0, len(processors)):
            self.common.print_device_header(i)

            try:
                _, profiles = amdsmi.amdsmi_get_ampp_profiles(processors[i])
            except amdsmi.AmdSmiLibraryException as e:
                self.assertEqual(
                    e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
                )
                continue

            # Names that don't resolve to a published "profile_<N>" slot
            # (sysfs control dirs, or a path-traversal attempt) must raise
            # INVAL, never SUCCESS -- this is the path-traversal /
            # validation-order guard.
            for bad_name in ("config", "limits", "../../etc", "profile_"):
                msg = f"\t### amdsmi_get_ampp_fields(gpu={i}, profile_name={bad_name}):"
                with self.assertRaises(amdsmi.AmdSmiLibraryException, msg=msg) as ctx:
                    amdsmi.amdsmi_get_ampp_fields(processors[i], bad_name)
                self.assertEqual(
                    ctx.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL, msg
                )

            for profile in profiles:
                msg = f"\t### amdsmi_get_ampp_fields(gpu={i}, profile={profile['name']}):"
                try:
                    fields = amdsmi.amdsmi_get_ampp_fields(processors[i], profile["name"])
                    self.common.print(msg, fields)
                except amdsmi.AmdSmiLibraryException as e:
                    self.common.print(msg, e)
                    if profile["is_writable"] and not profile["is_configured"]:
                        self.assertEqual(
                            e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NO_DATA, msg
                        )
                    else:
                        raise
                    continue

                for field in fields:
                    for key in ("name", "unit", "value", "min_value", "max_value", "has_limits"):
                        self.assertIn(key, field)

    def test_set_ampp_profile_dry_run(self):
        """Test ACTIVATE and CONFIGURE write paths in DRY_RUN mode"""
        self.common.print_func_name("")
        processors = amdsmi.amdsmi_get_processor_handles()

        # Enable DRY_RUN mode; ensure cleanup even if test fails
        os.environ["AMDSMI_DRY_RUN"] = "1"
        self.addCleanup(os.environ.pop, "AMDSMI_DRY_RUN", None)

        for i in range(0, len(processors)):
            self.common.print_device_header(i)

            try:
                _, profiles = amdsmi.amdsmi_get_ampp_profiles(processors[i])
            except amdsmi.AmdSmiLibraryException as e:
                self.assertEqual(
                    e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
                )
                continue

            if not profiles:
                continue

            # ACTIVATE the currently-active profile is a no-op but must
            # still succeed in DRY_RUN.
            active_profile = next((p for p in profiles if p["is_active"]), profiles[0])
            msg = (
                f"\t### amdsmi_activate_ampp_profile(gpu={i}, {active_profile['name']}) (DRY_RUN):"
            )
            try:
                amdsmi.amdsmi_activate_ampp_profile(processors[i], active_profile["name"])
                self.common.print(msg, "OK")
            except amdsmi.AmdSmiLibraryException as e:
                self.fail(f"Failed to ACTIVATE {active_profile['name']} in DRY_RUN mode: {e}")

            # ACTIVATE of an unpublished/invalid name must fail with INVAL,
            # not silently succeed.
            msg = f"\t### amdsmi_activate_ampp_profile(gpu={i}, ../../etc) (DRY_RUN):"
            with self.assertRaises(amdsmi.AmdSmiLibraryException, msg=msg) as ctx:
                amdsmi.amdsmi_activate_ampp_profile(processors[i], "../../etc")
            self.assertEqual(
                ctx.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL, msg
            )

            # CONFIGURE a writable slot with an empty field list must fail
            # INVAL: the driver rejects config/commit with -EINVAL if no
            # field was ever staged for this attempt, and AMDSMI defensively
            # rejects this before issuing any sysfs writes at all.
            writable = next((p for p in profiles if p["is_writable"]), None)
            if writable is not None:
                msg = (
                    f"\t### amdsmi_configure_ampp_profile(gpu={i}, {writable['name']}, "
                    "no fields) (DRY_RUN):"
                )
                with self.assertRaises(amdsmi.AmdSmiLibraryException, msg=msg) as ctx:
                    amdsmi.amdsmi_configure_ampp_profile(processors[i], writable["name"], [])
                self.assertEqual(
                    ctx.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL, msg
                )

                # CONFIGURE with an unrecognized field name must fail INVAL.
                msg = (
                    f"\t### amdsmi_configure_ampp_profile(gpu={i}, {writable['name']}, "
                    "bogus field) (DRY_RUN):"
                )
                with self.assertRaises(amdsmi.AmdSmiLibraryException, msg=msg) as ctx:
                    amdsmi.amdsmi_configure_ampp_profile(
                        processors[i],
                        writable["name"],
                        [{"name": "not_a_real_field_xyz", "value": 1}],
                    )
                self.assertEqual(
                    ctx.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL, msg
                )
