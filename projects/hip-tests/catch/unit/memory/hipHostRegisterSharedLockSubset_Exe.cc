/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Standalone helper spawned by Unit_hipHostRegister_SharedLockDeviceSubset (see
// hipHostRegister.cc). The parent restricts this process to a subset of the machine's
// GPUs via HIP_VISIBLE_DEVICES before spawning it, then checks this program's exit code.
//
// This exercises the multi-device hipHostRegister "shared lock" path end to end:
//   - every device visible to this process gets a working, identical device pointer
//     from one hipHostRegister call (validates cross-device pointer sharing);
//   - a kernel on every visible device can read/write through that pointer
//     (validates the mapping is actually usable, not just non-null);
//   - hipHostUnregister immediately followed by re-registering the same range works
//     cleanly, with no leaked or double-released lock.
//
// Not a Catch test binary: plain main(), consistent with the other *_Exe helpers in
// this directory. Exit code: 1 on success, 0 on any failure.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void MarkIndex(int* ptr, int idx, int value) { ptr[idx] = value; }

namespace {

bool Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
    return false;
  }
  return true;
}

// Registers buf (numDevices ints), confirms every device in [0, numDevices) gets an
// identical, working device pointer, exercises it with a kernel from every device, then
// unregisters. Returns false on any failure.
bool RegisterAccessUnregisterOnce(int* buf, int numDevices) {
  if (!Check(hipHostRegister(buf, numDevices * sizeof(int), 0), "hipHostRegister")) {
    return false;
  }

  std::vector<void*> devPtrs(numDevices, nullptr);
  for (int dev = 0; dev < numDevices; ++dev) {
    if (!Check(hipSetDevice(dev), "hipSetDevice")) return false;
    if (!Check(hipHostGetDevicePointer(&devPtrs[dev], buf, 0), "hipHostGetDevicePointer")) {
      return false;
    }
  }
  for (int dev = 1; dev < numDevices; ++dev) {
    if (devPtrs[dev] != devPtrs[0]) {
      fprintf(stderr, "device %d pointer %p differs from device 0 pointer %p\n", dev,
              devPtrs[dev], devPtrs[0]);
      return false;
    }
  }

  for (int dev = 0; dev < numDevices; ++dev) {
    buf[dev] = -1;
  }

  // Exercise the mapping from every device, not just device 0, so a mapping that is
  // only valid for the owning device (but reused by siblings) would show up as a
  // kernel fault or a stale/incorrect value instead of silently passing.
  for (int dev = 0; dev < numDevices; ++dev) {
    if (!Check(hipSetDevice(dev), "hipSetDevice")) return false;
    int* devPtr = reinterpret_cast<int*>(devPtrs[dev]);
    hipLaunchKernelGGL(MarkIndex, dim3(1), dim3(1), 0, 0, devPtr, dev, 100 + dev);
    if (!Check(hipGetLastError(), "kernel launch")) return false;
    if (!Check(hipDeviceSynchronize(), "hipDeviceSynchronize")) return false;
  }
  for (int dev = 0; dev < numDevices; ++dev) {
    if (buf[dev] != 100 + dev) {
      fprintf(stderr, "buf[%d] = %d, expected %d\n", dev, buf[dev], 100 + dev);
      return false;
    }
  }

  return Check(hipHostUnregister(buf), "hipHostUnregister");
}

}  // namespace

int main() {
  int numDevices = 0;
  if (!Check(hipGetDeviceCount(&numDevices), "hipGetDeviceCount")) return 0;
  if (numDevices < 2) {
    fprintf(stderr, "need at least 2 visible devices, got %d\n", numDevices);
    return 0;
  }

  std::vector<int> buf(static_cast<size_t>(numDevices), 0);

  // Round 1: exercise the shared-lock path end to end.
  if (!RegisterAccessUnregisterOnce(buf.data(), numDevices)) return 0;

  // Round 2: re-register the same range immediately after unregistering it, to
  // confirm round 1's unlock/ownership bookkeeping left nothing dangling.
  if (!RegisterAccessUnregisterOnce(buf.data(), numDevices)) return 0;

  return 1;
}
