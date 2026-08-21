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
 */
HIP_TEST_CASE(Unit_hipManagedMultiGpuVaConsistency_MultiDevice) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 2) {
    HIP_SKIP_TEST("Multi-device test requires at least 2 GPUs");
    return;
  }
  for (int d = 0; d < numDevices; ++d) {
    if (!HipTest::isManagedMemorySupportedOnDevice(d)) {
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

static __global__ void write_int_ptr(int* p, int v) { *p = v; }

/**
 * Test Description
 * ------------------------
 *  - Verifies that, with a persistent __managed__ variable resident (g_val/g_out
 *    above occupy the shared fine-grain SVM chunk), a multi-GPU fine-grain
 *    hipHostMalloc remains reachable at its one canonical address on every
 *    device. If a peer device's mapping diverged or aliased, a kernel on that
 *    peer writing through the canonical pointer faults or lands elsewhere, so
 *    the host reads back a stale value.
 *  - For every fine-grain flag combo, and driven from every device in turn, a
 *    device-side write through the canonical pointer must be visible at the host
 *    pointer.
 *  - Skips on single-GPU systems.
 * Test source
 * ------------------------
 *  - unit/memory/hipManagedMultiGpuVaConsistency.cc
 * Test requirements
 * ------------------------
 *  - Multiple devices
 */
HIP_TEST_CASE(Unit_hipManagedMultiGpuVaConsistency_HostAllocNoDivergence) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 2) {
    HIP_SKIP_TEST("Multi-device test requires at least 2 GPUs");
    return;
  }

  // The file-scope __managed__ globals above are allocated at module load, so a
  // persistent managed buffer already occupies the shared SVM chunk for the
  // whole process; no explicit managed access is needed here.
  const unsigned int flags[] = {
      hipHostMallocDefault,
      hipHostMallocPortable,
      hipHostMallocMapped,
      hipHostMallocWriteCombined,
      hipHostMallocPortable | hipHostMallocMapped,
      hipHostMallocPortable | hipHostMallocWriteCombined,
      hipHostMallocMapped | hipHostMallocWriteCombined,
      hipHostMallocPortable | hipHostMallocMapped | hipHostMallocWriteCombined,
  };

  for (unsigned int flag : flags) {
    int* hostPtr = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&hostPtr), sizeof(int), flag));
    REQUIRE(hostPtr != nullptr);

    // The invariant: every device (including the peer) reaches this one
    // canonical pointer and its write is host-visible.
    for (int d = 0; d < numDevices; ++d) {
      const int sentinel = 0xD000 + d * 0x100 + static_cast<int>(flag);
      HIP_CHECK(hipSetDevice(d));
      *hostPtr = 0;
      write_int_ptr<<<1, 1>>>(hostPtr, sentinel);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      INFO("flag 0x" << std::hex << flag << " device " << d << " ptr=" << hostPtr
                     << " read 0x" << *hostPtr);
      REQUIRE(*hostPtr == sentinel);
    }

    HIP_CHECK(hipFreeHost(hostPtr));
  }

  HIP_CHECK(hipSetDevice(0));
}
/**
 * End doxygen group managed.
 * @}
 */
