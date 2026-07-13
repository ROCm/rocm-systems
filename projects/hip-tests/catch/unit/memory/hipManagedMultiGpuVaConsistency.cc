/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup managed managed
 * @{
 * Multi-GPU virtual-address consistency for __managed__ variables.
 *
 * A __managed__ variable is shared, coherent, cross-device memory: the single
 * canonical pointer the app holds must resolve to the SAME allocation on every
 * device in the context. On the multi-GPU PAL path a peer device's managed
 * backing could be created at a device-local VA that diverged from the owner's
 * canonical SVM VA; a kernel launched on that peer then faulted or accessed the
 * wrong memory. This test pins that behavior down.
 */

__managed__ int g_val = 0;  // written by device, read by host / other device
__managed__ int g_out = 0;  // device reads g_val into here for host to verify

static __global__ void write_val(int v) { g_val = v; }
static __global__ void read_into_out() { g_out = g_val; }

/**
 * Test Description
 * ------------------------
 *  - Verifies a __managed__ variable is addressable at the same canonical VA on
 *    every device, in both directions and cross-device:
 *      1. device d writes the symbol  -> host reads it back
 *      2. host writes the symbol      -> device d reads it into g_out
 *      3. device 1 writes the symbol  -> device 0 reads it (cross-device coherence)
 *  - Skips on single-GPU systems and on devices without managed-memory support.
 * Test source
 * ------------------------
 *  - unit/memory/hipManagedMultiGpuVaConsistency.cc
 * Test requirements
 * ------------------------
 *  - Multiple devices supporting managed memory
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE("Unit_hipManagedMultiGpuVaConsistency_MultiDevice") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 2) {
    HIP_SKIP_TEST("Multi-device test requires at least 2 GPUs");
    return;
  }
  for (int d = 0; d < numDevices; ++d) {
    if (!isManagedMemorySupportedOnDevice(d)) {
      HIP_SKIP_TEST("Managed memory is not supported on all devices");
      return;
    }
  }

  // Direction 1: each device writes a device-unique sentinel through the
  // canonical symbol; the host must read exactly that value back.
  for (int d = 0; d < numDevices; ++d) {
    const int sentinel = 0xA000 + d;
    HIP_CHECK(hipSetDevice(d));
    g_val = 0;
    HIP_CHECK(hipDeviceSynchronize());
    write_val<<<1, 1>>>(sentinel);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    INFO("device " << d << " wrote 0x" << std::hex << sentinel << ", host read 0x" << g_val);
    REQUIRE(g_val == sentinel);
  }

  // Direction 2: the host writes the sentinel; a kernel on device d copies
  // g_val into g_out. A matching g_out proves device d sees the same VA.
  for (int d = 0; d < numDevices; ++d) {
    const int sentinel = 0xB000 + d;
    HIP_CHECK(hipSetDevice(d));
    g_val = sentinel;
    g_out = 0;
    HIP_CHECK(hipDeviceSynchronize());
    read_into_out<<<1, 1>>>();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    INFO("host wrote 0x" << std::hex << sentinel << ", device " << d << " read 0x" << g_out);
    REQUIRE(g_out == sentinel);
  }

  // Direction 3: device 1 writes the symbol, device 0 reads it -> cross-device
  // coherence through the shared canonical address.
  {
    const int sentinel = 0xC0DE;
    HIP_CHECK(hipSetDevice(1));
    g_val = 0;
    HIP_CHECK(hipDeviceSynchronize());
    write_val<<<1, 1>>>(sentinel);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipSetDevice(0));
    g_out = 0;
    read_into_out<<<1, 1>>>();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    INFO("device 1 wrote 0x" << std::hex << sentinel << ", device 0 read 0x" << g_out);
    REQUIRE(g_out == sentinel);
  }

  HIP_CHECK(hipSetDevice(0));
}
/**
 * End doxygen group managed.
 * @}
 */
