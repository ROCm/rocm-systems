#!/usr/bin/env python3
# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Guard that the staged nccl_device.h compiles in a C99 host translation unit.

NCCL 2.31.2 made nccl_device.h C99-compatible (NVIDIA/nccl@6431a138db), but
RCCL's own additions still broke a plain C compile:
<cstddef> in core.h, a C++ functional cast in hip_compat.h's host nccl_umulhi,
the static_assert ABI pins in host.h, and unguarded <atomic> / rccl_ptr.h
includes in lsa_barrier__funcs.h and ll_a2a__funcs.h. Each one alone fails
test_nccl_device_h_compiles_as_c99.

Probes compile against $RCCL_BUILD/include, the hipify-staged tree RCCL
installs, with a plain C compiler at -std=c99. The control compiles nccl.h and
nccl_device/net_device.h with the same flags and HIP headers, so its failure
means a broken toolchain or build tree. Run after building RCCL:
python3 test/test_nccl_device_c99.py. Exits 0 on pass, 1 on a failed test,
2 on a harness error, 77 when every selected test skipped.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

RCCL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.environ.get("RCCL_BUILD") or os.path.join(RCCL_ROOT, "build", "release")
INCLUDE_DIR = os.path.join(BUILD_DIR, "include")
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
HIP_INCLUDE_DIR = os.path.join(ROCM_PATH, "include")


def _find_cc():
    for cc in (os.environ.get("CC"), "cc", "gcc", "clang", os.path.join(ROCM_PATH, "llvm", "bin", "clang")):
        if cc and shutil.which(cc):
            return cc
    return None


CC = _find_cc()

# Every probe starts with this, so a compiler that is not doing plain C99 fails the control too.
PLAIN_C99 = """\
#if defined(__cplusplus) || defined(__HIPCC__) || defined(__CUDACC__) || __STDC_VERSION__ != 199901L
#error "not a plain C99 compile"
#endif
"""


def _compile(src):
    """Return (returncode, command line + compiler output) for a C99 compile of `src`."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "probe.c")
        with open(path, "w") as f:
            f.write(PLAIN_C99 + textwrap.dedent(src))
        cmd = [CC, "-std=c99", "-D__HIP_PLATFORM_AMD__=1", "-I" + INCLUDE_DIR, "-I" + HIP_INCLUDE_DIR,
               "-c", path, "-o", os.path.join(tmp, "probe.o")]
        proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, " ".join(cmd) + "\n" + proc.stdout + proc.stderr


@unittest.skipUnless(CC, "no C compiler found (set CC)")
@unittest.skipUnless(os.path.isfile(os.path.join(HIP_INCLUDE_DIR, "hip", "hip_runtime.h")),
                     "no HIP headers in %s (set ROCM_PATH)" % HIP_INCLUDE_DIR)
@unittest.skipUnless(os.path.isfile(os.path.join(INCLUDE_DIR, "nccl.h")),
                     "no configured RCCL build in %s (set RCCL_BUILD)" % BUILD_DIR)
class NcclDeviceC99Test(unittest.TestCase):
    def test_control_nccl_h_compiles_as_c99(self):
        self.assertTrue(os.path.isfile(os.path.join(INCLUDE_DIR, "nccl_device.h")),
                        "nccl_device.h is not staged in %s: build rccl or copy_nccl_device_headers" % INCLUDE_DIR)
        rc, out = _compile("""
            #include <nccl.h>
            #include <nccl_device/net_device.h>

            ncclResult_t probe(int* version, ncclNetDeviceHandle_v7_t* handle) {
              handle->netDeviceType = NCCL_NET_DEVICE_HOST;
              return ncclGetVersion(version);
            }
            """)
        self.assertEqual(rc, 0, msg=out)

    def test_nccl_device_h_compiles_as_c99(self):
        rc, out = _compile("""
            #include <nccl_device.h>

            ncclResult_t probe(ncclComm_t comm, ncclDevComm_t* devComm) {
              ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
              ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
              ncclResult_t res = ncclCommQueryProperties(comm, &props);
              if (res != ncclSuccess) return res;
              reqs.lsaBarrierCount = props.nRanks;
              return ncclDevCommCreate(comm, &reqs, devComm);
            }
            """)
        self.assertEqual(rc, 0, msg=out)


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False).result
    if result.errors:
        sys.exit(2)
    if result.failures:
        sys.exit(1)
    sys.exit(77 if result.testsRun and len(result.skipped) == result.testsRun else 0)
