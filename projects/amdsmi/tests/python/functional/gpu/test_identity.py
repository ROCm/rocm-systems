#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU device identity: ASIC info, board info, IDs, BDF, UUID, firmware, VBIOS, enumeration."""

import os
import re
import subprocess
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


def _kmod_versions_of_running_kernel():
    """The VERSION-RELEASE of each amdgpu-kmod package rpm has for the running kernel.

    Empty when rpm or its database is not there. Like the library, rpm is asked only where its
    database has a packages file: a query creates a database where there is none.
    """

    def rpm(*args):
        try:
            done = subprocess.run(
                ["/usr/bin/rpm", *args],
                env={"LC_ALL": "C", "HOME": "/"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.splitlines() if done.returncode == 0 else None

    dbpath = rpm("--eval", "%{_dbpath}")
    if not dbpath or len(dbpath) != 1 or not dbpath[0].startswith("/"):
        return []
    if not any(
        os.path.isfile(os.path.join(dbpath[0], name))
        for name in ("Packages", "Packages.db", "rpmdb.sqlite")
    ):
        return []
    return rpm("-q", "--qf", "%{VERSION}-%{RELEASE}\n", f"amdgpu-kmod-{os.uname().release}") or []


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
        # Asked before the library's time-limited rpm query, so rpm's own caches are warm and
        # the library's query is not the cold one.
        kmod_versions = _kmod_versions_of_running_kernel()
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_driver_info_v2=amdsmi.amdsmi_get_gpu_driver_info_v2
        )

        # A.B.C[.D] for the module, A.B.C[.D]-<build>[.<distro>] for the package.
        package_format = re.compile(r"(\d+\.\d+\.\d+)(?:\.\d+)*-(\d+)(?:\..*)?")
        dkms = package_format.fullmatch(_dkms_version_of_running_kernel() or "")
        answered = 0
        for gpu in self.common.processors:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                try:
                    base = amdsmi.amdsmi_get_gpu_driver_info(gpu)
                except amdsmi.AmdSmiLibraryException as v1_error:
                    # v2 starts with the v1 call, so it must fail with the same status.
                    with self.assertRaises(amdsmi.AmdSmiLibraryException) as v2_error:
                        amdsmi.amdsmi_get_gpu_driver_info_v2(gpu)
                    self.assertEqual(v2_error.exception.get_error_code(), v1_error.get_error_code())
                    continue
            info = amdsmi.amdsmi_get_gpu_driver_info_v2(gpu)
            answered += 1
            self.assertEqual({key: info[key] for key in base}, base)
            self.assertEqual(
                set(info) - set(base),
                {
                    "driver_kernel_version",
                    "amdgpu_driver_version",
                    "driver_build_version",
                    "driver_full_version",
                    "driver_package",
                },
            )

            version = base["driver_version"]
            module = re.fullmatch(r"(\d+\.\d+\.\d+)(?:\.(\d+))?", version)
            self.assertEqual(info["driver_kernel_version"], module.group(1) if module else "N/A")
            self.assertEqual(
                info["amdgpu_driver_version"], (module.group(2) or "N/A") if module else "N/A"
            )
            # The builds rpm lists for amdgpu-kmod with the kernel version of the loaded module.
            kmod_builds = {
                kmod.group(2)
                for kmod in map(package_format.fullmatch, kmod_versions)
                if module and kmod and kmod.group(1) == module.group(1)
            }
            if module and dkms and dkms.group(1) == module.group(1):
                # DKMS recorded a package with the kernel version of the loaded module.
                self.assertEqual(info["driver_package"], "amdgpu-dkms")
                self.assertEqual(info["driver_build_version"], dkms.group(2))
                self.assertEqual(info["driver_full_version"], f"{version}-{dkms.group(2)}")
            elif len(kmod_builds) == 1:
                # No DKMS package of the module, and rpm has one amdgpu-kmod build of it. Several
                # would give none: which of them is loaded is not known.
                (kmod_build,) = kmod_builds
                self.assertEqual(info["driver_package"], "amdgpu-kmod")
                self.assertEqual(info["driver_build_version"], kmod_build)
                self.assertEqual(info["driver_full_version"], f"{version}-{kmod_build}")
            elif len(kmod_builds) > 1:
                self.assertEqual(info["driver_package"], "N/A")
                self.assertEqual(info["driver_build_version"], "N/A")
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
