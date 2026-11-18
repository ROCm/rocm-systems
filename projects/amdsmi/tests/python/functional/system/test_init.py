#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""System initialization and shutdown functional tests."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import common.common as common
from common.common import amdsmi


class TestAmdSmiInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(common.verbose)
        return

    def test_init_shutdown(self):
        self.common.print("## test_init_shutdown()")

        msg = "\t### amdsmi_init():"
        try:
            ret = self.common.amdsmi_smart_init()[0]
            self.common.print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            self.common.print(msg, e)
            raise e

        msg = "\t### amdsmi_shut_down():"
        try:
            ret = amdsmi.amdsmi_shut_down()
            self.common.print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            self.common.print(msg, e)
            raise e
        return

    def test_init_with_another_library_copy_loaded_global(self):
        # rocm_sdk preloads libamd_smi RTLD_GLOBAL for PyTorch; a second copy
        # must not take over this library's internal calls (#7643).
        self.common.print("## test_init_with_another_library_copy_loaded_global()")
        probe = (
            "import ctypes, os, sys\n"
            "if len(sys.argv) > 1:\n"
            "    ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)\n"
            "import amdsmi\n"
            "amdsmi.amdsmi_init()\n"
            "print(len(amdsmi.amdsmi_get_processor_handles()))\n"
            "libs = [l.split()[-1] for l in open('/proc/self/maps') if '.so' in l]\n"
            "print([p for p in libs if '/libamd_smi' in p][0])\n"
            "print(':'.join(sorted({os.path.dirname(p) for p in libs if p.startswith('/')})))\n"
            "amdsmi.amdsmi_shut_down()\n"
        )
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))

        def run(*args):
            out = subprocess.run(
                [sys.executable, "-c", probe, *args], env=env, capture_output=True, text=True
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            return out.stdout.split()[:3]

        expected, lib, lib_dirs = run()
        if int(expected) == 0:
            self.skipTest("no GPUs enumerated")
        # The copy loses the original's $ORIGIN RPATH, so point the loader at
        # the directories its dependencies were found in.
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            d for d in (lib_dirs, os.environ.get("LD_LIBRARY_PATH")) if d
        )
        with tempfile.TemporaryDirectory() as tmp:
            other = os.path.join(tmp, os.path.basename(lib))
            shutil.copy(lib, other)
            self.assertEqual(run(other)[0], expected)
