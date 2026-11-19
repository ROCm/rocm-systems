#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU device identity: ASIC info, board info, IDs, BDF, UUID, firmware, VBIOS, enumeration."""

import os
import re
import unittest
import warnings

import common.common as common
from common.common import amdsmi


def _dkms_version_of_running_kernel():
    """The amdgpu-dkms version DKMS recorded for the running kernel, or None.

    Read here and not through the library: the test must not share the lookup it checks.
    """
    uname = os.uname()
    root = "/var/lib/dkms/amdgpu"
    try:
        target = os.readlink(os.path.join(root, f"kernel-{uname.release}-{uname.machine}"))
        # DKMS writes <version>/<release>/<machine>, relative to the root.
        version = target.split("/")[0]
        with open(os.path.join(root, version, "source", "dkms.conf"), encoding="utf-8") as conf:
            lines = conf.read().splitlines()
    except OSError:
        return None
    if version and 'PACKAGE_NAME="amdgpu"' in lines and f'PACKAGE_VERSION="{version}"' in lines:
        return version
    return None


class TestGpuIdentity(unittest.TestCase):
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

    def test_get_fw_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_fw_info=amdsmi.amdsmi_get_fw_info)
        return

    def test_get_gpu_asic_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_asic_info=amdsmi.amdsmi_get_gpu_asic_info)
        return

    def test_get_gpu_bdf_id(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_bdf_id=amdsmi.amdsmi_get_gpu_bdf_id)
        return

    def test_get_gpu_board_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_board_info=amdsmi.amdsmi_get_gpu_board_info)
        return

    def test_get_gpu_device_bdf(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_device_bdf=amdsmi.amdsmi_get_gpu_device_bdf)
        return

    def test_get_gpu_device_uuid(self):
        self.common.print_func_name("")
        # Devices with no real ASIC serial (e.g. gfx1151/WSL) now correctly
        # refuse to synthesize a placeholder UUID, so NOT_SUPPORTED is expected
        # alongside SUCCESS here.
        accept = [amdsmi.AmdSmiStatus.SUCCESS, amdsmi.AmdSmiStatus.NOT_SUPPORTED]
        with self.common.status_sweep():
            for i, gpu in enumerate(self.common.processors):
                self.common.print_device_header(i)
                msg = f"\t### amdsmi_get_gpu_device_uuid(gpu={i}):"
                with self.common.expect_status(msg, accept):
                    amdsmi.amdsmi_get_gpu_device_uuid(gpu)
                self.common.print("")

        # Invalid gpu handle: kept on the check_ret path since it raises
        # AmdSmiParameterException, which expect_status cannot classify
        # (no get_error_code()).
        raise_exception = None
        msg = "\t### amdsmi_get_gpu_device_uuid(gpu=invalid):"
        try:
            amdsmi.amdsmi_get_gpu_device_uuid(self.common.bad_gpu)
            self.common.check_ret("", "", self.common.FAIL)
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            if self.common.check_ret(msg, e, self.common.FAIL):
                raise_exception = e
        if raise_exception:
            raise raise_exception
        return

    def test_get_gpu_device_cuid(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_device_cuid=amdsmi.amdsmi_get_gpu_device_cuid)
        return

    def test_get_gpu_driver_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_driver_info=amdsmi.amdsmi_get_gpu_driver_info)
        return

    def test_get_gpu_driver_info_v2(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_driver_info_v2=amdsmi.amdsmi_get_gpu_driver_info_v2
        )

        # A.B.C[.D] for the module, A.B.C[.D]-<build>[.<distro>] for the package.
        dkms = re.fullmatch(
            r"((\d+\.\d+\.\d+)(?:\.\d+)*-(\d+))(?:\..*)?", _dkms_version_of_running_kernel() or ""
        )
        answered = 0
        for gpu in self.common.processors:
            try:
                info = amdsmi.amdsmi_get_gpu_driver_info_v2(gpu)
            except amdsmi.AmdSmiLibraryException:
                continue
            answered += 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                base = amdsmi.amdsmi_get_gpu_driver_info(gpu)
            self.assertEqual({key: info[key] for key in base}, base)
            self.assertEqual(
                set(info) - set(base),
                {
                    "driver_kernel_version",
                    "amdgpu_driver_version",
                    "driver_build_version",
                    "driver_full_version",
                    "driver_package",
                    "driver_package_version",
                },
            )

            version = base["driver_version"]
            module = re.fullmatch(r"(\d+\.\d+\.\d+)(?:\.(\d+))?", version)
            self.assertEqual(info["driver_kernel_version"], module.group(1) if module else "N/A")
            self.assertEqual(
                info["amdgpu_driver_version"], (module.group(2) or "N/A") if module else "N/A"
            )
            if module and dkms and dkms.group(2) == module.group(1):
                # DKMS recorded a package with the kernel version of the loaded module.
                self.assertEqual(info["driver_package"], "amdgpu-dkms")
                self.assertEqual(info["driver_package_version"], dkms.group(1))
                self.assertEqual(info["driver_build_version"], dkms.group(3))
                self.assertEqual(info["driver_full_version"], f"{version}-{dkms.group(3)}")
            else:
                # A package is reported only together with its build.
                self.assertEqual(
                    info["driver_package"] == "N/A", info["driver_build_version"] == "N/A"
                )
        if not answered:
            self.skipTest("no device reported driver info")
        return

    def test_get_gpu_enumeration_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_enumeration_info=amdsmi.amdsmi_get_gpu_enumeration_info
        )
        return

    def test_get_gpu_id(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_id=amdsmi.amdsmi_get_gpu_id)
        return

    def test_get_gpu_kfd_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_kfd_info=amdsmi.amdsmi_get_gpu_kfd_info)
        return

    def test_get_gpu_revision(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_revision=amdsmi.amdsmi_get_gpu_revision)
        return

    def test_get_gpu_subsystem_id(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_subsystem_id=amdsmi.amdsmi_get_gpu_subsystem_id)
        return

    def test_get_gpu_subsystem_name(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_subsystem_name=amdsmi.amdsmi_get_gpu_subsystem_name
        )
        return

    def test_get_gpu_vbios_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_vbios_info=amdsmi.amdsmi_get_gpu_vbios_info)
        return

    def test_get_gpu_vendor_name(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_vendor_name=amdsmi.amdsmi_get_gpu_vendor_name)
        return

    def test_get_gpu_virtualization_mode(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_virtualization_mode=amdsmi.amdsmi_get_gpu_virtualization_mode
        )
        return

    def test_get_gpu_vram_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_vram_info=amdsmi.amdsmi_get_gpu_vram_info)
        return

    def test_get_lib_version(self):
        self.common.print_func_name("")
        self.common.Test_API(amdsmi_get_lib_version=amdsmi.amdsmi_get_lib_version)
        return

    def test_get_processor_count_from_handles(self):
        self.common.print_func_name("")
        self.common.Test_API(
            amdsmi_get_processor_count_from_handles=amdsmi.amdsmi_get_processor_count_from_handles,
            processors=self.common.processors,
        )
        return

    # print data issues

    def test_get_processor_handles(self):
        self.common.print_func_name("")
        msg = "\t### amdsmi_get_processor_handles():"
        try:
            procs = amdsmi.amdsmi_get_processor_handles()
            self.common.print(msg, [id(addr) for addr in procs])
            self.assertGreaterEqual(len(self.common.processors), 1)
            self.assertLessEqual(len(self.common.processors), self.common.max_num_physical_devices)
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            if self.common.check_ret(msg, e, self.common.PASS):
                self.raise_exception = e
        self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_processor_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_processor_info=amdsmi.amdsmi_get_processor_info)
        return

    def test_get_processor_type(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_processor_type=amdsmi.amdsmi_get_processor_type)
        return

    # data print issues
