/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipExecutionCtxSmAlignment hipExecutionCtxSmAlignment
 * @{
 * @ingroup ExecutionContextTest
 * Validates that SM co-scheduling alignment returned by device resource APIs
 * is consistent with the device's WGP/CU mode (arch-dependent).
 */

#include <hip_test_common.hh>
#include "hip_executionctx_common.hh"

#include <algorithm>
#include <vector>

/**
 * Test Description
 * ------------------------
 *  - Verifies that hipDeviceGetDevResource returns a valid smCoscheduledAlignment
 *    (1 or 2), that minSmPartitionSize matches it, and that smCount equals
 *    multiProcessorCount.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxSmAlignment_DeviceGetDevResource) {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));

  for (int dev = 0; dev < deviceCount; dev++) {
    HIP_CHECK(hipSetDevice(dev));

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));

    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, dev));

    hipDevResource resource{};
    HIP_CHECK(hipDeviceGetDevResource(device, &resource, hipDevResourceTypeSm));

    unsigned int alignment = resource.sm.smCoscheduledAlignment;

    INFO("Device " << dev << " (" << prop.gcnArchName << "): "
         << "smCoscheduledAlignment=" << alignment
         << " minSmPartitionSize=" << resource.sm.minSmPartitionSize);
#if HT_AMD
    REQUIRE((alignment == 1 || alignment == 2));
#else
    REQUIRE((alignment >= 1));
#endif
    REQUIRE(resource.sm.minSmPartitionSize == alignment);
    REQUIRE(resource.sm.smCount == static_cast<unsigned int>(prop.multiProcessorCount) * alignment);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Verifies that hipStreamGetDevResource returns alignment values matching
 *    hipDeviceGetDevResource for the same device, for both the NULL stream
 *    and a user-created stream.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxSmAlignment_StreamGetDevResource) {
  HIP_CHECK(hipSetDevice(0));

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipDevResource devResource{};
  HIP_CHECK(hipDeviceGetDevResource(device, &devResource, hipDevResourceTypeSm));

  // NULL stream
  hipDevResource nullStreamRes{};
  HIP_CHECK(hipStreamGetDevResource(nullptr, &nullStreamRes, hipDevResourceTypeSm));
  REQUIRE(nullStreamRes.sm.smCoscheduledAlignment == devResource.sm.smCoscheduledAlignment);
  REQUIRE(nullStreamRes.sm.minSmPartitionSize == devResource.sm.minSmPartitionSize);

  // User-created stream
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  hipDevResource streamRes{};
  HIP_CHECK(hipStreamGetDevResource(stream, &streamRes, hipDevResourceTypeSm));
  REQUIRE(streamRes.sm.smCoscheduledAlignment == devResource.sm.smCoscheduledAlignment);
  REQUIRE(streamRes.sm.minSmPartitionSize == devResource.sm.minSmPartitionSize);

  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Verifies that alignment propagates correctly through resource splitting.
 *    After hipDevSmResourceSplitByCount, each result partition must carry the
 *    same alignment as the input resource. Remainder resource, if present,
 *    must have minimum architectural alignment. On AMD this is same as WGP alignment,
 *    but CUDA devices set remainder alignment to minimum architectural alignment which
 *    may be less than the input alignment.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxSmAlignment_PropagatesThroughSplit) {
  HIP_CHECK(hipSetDevice(0));

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipDevResource input{};
  HIP_CHECK(hipDeviceGetDevResource(device, &input, hipDevResourceTypeSm));

  unsigned int alignment = input.sm.smCoscheduledAlignment;
  unsigned int minCount = alignment;

  unsigned int nbGroups = 0;
  HIP_CHECK(hipDevSmResourceSplitByCount(nullptr, &nbGroups, &input, nullptr, 0, minCount));
  REQUIRE(nbGroups > 0);

  unsigned int requestedGroups = std::min(nbGroups, 2u);
  std::vector<hipDevResource> result(requestedGroups);
  hipDevResource remainder{};
  HIP_CHECK(hipDevSmResourceSplitByCount(result.data(), &requestedGroups, &input,
                                         &remainder, 0, minCount));

  for (unsigned int i = 0; i < requestedGroups; i++) {
    REQUIRE(result[i].sm.smCoscheduledAlignment == alignment);
    REQUIRE(result[i].sm.minSmPartitionSize == alignment);
  }

  if (remainder.type == hipDevResourceTypeSm) {
#if HT_AMD
    REQUIRE(remainder.sm.smCoscheduledAlignment == alignment);
    REQUIRE(remainder.sm.minSmPartitionSize == alignment);
#else 
    REQUIRE(remainder.sm.smCoscheduledAlignment >= 1);
    REQUIRE(remainder.sm.minSmPartitionSize >= 1);
#endif
  }
}

/**
 * Test Description
 * ------------------------
 *  - Verifies that hipExecutionCtxGetDevResource on a green context created
 *    from a split resource preserves the alignment from the original resource.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxSmAlignment_GreenCtxPreservesAlignment) {
  HIP_CHECK(hipSetDevice(0));

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipDevResource input{};
  HIP_CHECK(hipDeviceGetDevResource(device, &input, hipDevResourceTypeSm));

  unsigned int alignment = input.sm.smCoscheduledAlignment;
  unsigned int groupSize = (input.sm.smCount / 2 / alignment) * alignment;
  REQUIRE(groupSize >= alignment);

  hipDevSmResourceGroupParams params[1] = {};
  params[0].smCount = groupSize;

  hipDevResource splitResult[1] = {};
  hipDevResource remainder{};
  HIP_CHECK(hipDevSmResourceSplit(splitResult, 1, &input, &remainder, 0, params));

  hipDevResourceDesc_t desc{};
  HIP_CHECK(hipDevResourceGenerateDesc(&desc, splitResult, 1));

  hipExecutionCtx_t ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&ctx, desc, 0, 0));
  REQUIRE(ctx != nullptr);

  hipDevResource ctxResource{};
  HIP_CHECK(hipExecutionCtxGetDevResource(ctx, &ctxResource, hipDevResourceTypeSm));

  REQUIRE(ctxResource.sm.smCoscheduledAlignment == alignment);
  REQUIRE(ctxResource.sm.minSmPartitionSize >= 1);
  REQUIRE(ctxResource.sm.minSmPartitionSize <= alignment);
  REQUIRE(ctxResource.sm.smCount == groupSize);

  HIP_CHECK(hipExecutionCtxDestroy(ctx));
}

/**
 * End doxygen group hipExecutionCtxSmAlignment.
 * @}
 */
