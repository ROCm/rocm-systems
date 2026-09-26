/* Copyright (c) 2026 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

/* Portable capability probes for feature tests.

   Each probe is cheap and returns a bool so tests can early-skip via
   GTEST_SKIP() when their prerequisites are absent.  On Windows the
   KFD probe always returns false; on Linux the KMD probe always
   returns false.  enumerate_gfx_targets() returns the gfx_target_version
   ints reported by the kernel (e.g. 90402 for gfx942) filtered to GPU
   nodes (gpu_id != 0).  */

#ifndef DBGAPI_FEATURE_SUPPORT_CAPABILITIES_H
#define DBGAPI_FEATURE_SUPPORT_CAPABILITIES_H

#include <cstdint>
#include <vector>

namespace amd::dbgapi::test
{

/* True if the AMD GPU kernel driver (KFD on Linux) appears usable from
   this process.  Probes /dev/kfd existence and R_OK access.  Always
   false on non-Linux.  */
bool is_kfd_available ();

/* True if the AMD GPU kernel-mode driver (KMD on Windows) appears
   usable.  Always false on non-Windows.  Stubbed pending Windows port.  */
bool is_kmd_available ();

/* Convenience: KFD on Linux, KMD on Windows.  */
bool is_driver_available ();

/* True if at least one GPU node is visible in KFD topology
   (gpu_id != 0).  Implies is_kfd_available() on Linux.  Always false
   on non-Linux until KMD probes land.  */
bool is_gpu_available ();

/* gfx_target_version ints (e.g. 90402 == gfx942) for every GPU node
   the driver reports.  Empty when no driver / no GPUs.  Useful for
   per-arch skip predicates.  */
std::vector<uint32_t> enumerate_gfx_targets ();

} /* namespace amd::dbgapi::test */

#endif /* DBGAPI_FEATURE_SUPPORT_CAPABILITIES_H */
