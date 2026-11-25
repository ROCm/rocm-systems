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

// Each device marks/verifies its own 16 KiB region of the shared buffer.
constexpr int kElemsPerDevice = 4096;

__global__ void MarkRange(int* ptr, int count, int value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    ptr[i] = value;
  }
}

namespace {

bool Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
    return false;
  }
  return true;
}

// Registers buf (numDevices * kElemsPerDevice ints), confirms every device in
// [0, numDevices) gets an identical, working device pointer, exercises each device's own
// region of the mapping with a kernel, then unregisters. Returns false on any failure.
bool RegisterAccessUnregisterOnce(int* buf, int numDevices) {
  const size_t totalElems = static_cast<size_t>(numDevices) * kElemsPerDevice;
  if (!Check(hipHostRegister(buf, totalElems * sizeof(int), 0), "hipHostRegister")) {
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

  for (size_t i = 0; i < totalElems; ++i) {
    buf[i] = -1;
  }

  // Each device writes its own region, so a bad/aliased mapping shows up as a kernel
  // fault or wrong/corrupted values instead of silently passing.
  for (int dev = 0; dev < numDevices; ++dev) {
    if (!Check(hipSetDevice(dev), "hipSetDevice")) return false;
    int* devRegion = reinterpret_cast<int*>(devPtrs[dev]) + (dev * kElemsPerDevice);
    constexpr int kThreadsPerBlock = 256;
    int blocks = (kElemsPerDevice + kThreadsPerBlock - 1) / kThreadsPerBlock;
    hipLaunchKernelGGL(MarkRange, dim3(blocks), dim3(kThreadsPerBlock), 0, 0, devRegion,
                        kElemsPerDevice, 100 + dev);
    if (!Check(hipGetLastError(), "kernel launch")) return false;
    if (!Check(hipDeviceSynchronize(), "hipDeviceSynchronize")) return false;
  }
  for (int dev = 0; dev < numDevices; ++dev) {
    for (int i = 0; i < kElemsPerDevice; ++i) {
      int value = buf[dev * kElemsPerDevice + i];
      if (value != 100 + dev) {
        fprintf(stderr, "buf[%d] = %d, expected %d (device %d region)\n",
                dev * kElemsPerDevice + i, value, 100 + dev, dev);
        return false;
      }
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

  // The excluded GPU is outside HIP_VISIBLE_DEVICES and thus invisible here; this only
  // confirms the visible subset shares one correctly working mapping.
  std::vector<int> buf(static_cast<size_t>(numDevices) * kElemsPerDevice, 0);

  // Round 1: exercise the shared-lock path end to end.
  if (!RegisterAccessUnregisterOnce(buf.data(), numDevices)) return 0;

  // Round 2: re-register the same range immediately after unregistering it, to
  // confirm round 1's unlock/ownership bookkeeping left nothing dangling.
  if (!RegisterAccessUnregisterOnce(buf.data(), numDevices)) return 0;

  return 1;
}
