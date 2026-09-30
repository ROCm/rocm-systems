/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipStreamGetDevResource hipStreamGetDevResource
 * @{
 * @ingroup ExecutionContextTest
 * `hipStreamGetDevResource` API - functional and negative tests
 */

#include <hip_test_common.hh>
#include "hip_executionctx_common.hh"

#include <vector>

/**
 * Test Description
 * ------------------------
 *  - Creates a green context from a split SM resource, creates a stream from
 *    it, then calls hipStreamGetDevResource on that stream.  Verifies the
 *    returned SM count matches the partition size used to create the green
 *    context.
 */
HIP_TEST_CASE(Unit_hipStreamGetDevResource_GreenCtxStream_Functional) {
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
  REQUIRE(splitResult[0].sm.smCount == groupSize);

  hipDevResourceDesc_t desc{};
  HIP_CHECK(hipDevResourceGenerateDesc(&desc, splitResult, 1));

  hipExecutionCtx_t ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&ctx, desc, 0, 0));
  REQUIRE(ctx != nullptr);

  hipStream_t stream = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream, ctx, hipStreamNonBlocking, 0));
  REQUIRE(stream != nullptr);

  // Query the stream's SM resource
  hipDevResource streamRes{};
  HIP_CHECK(hipStreamGetDevResource(stream, &streamRes, hipDevResourceTypeSm));

  REQUIRE(streamRes.type == hipDevResourceTypeSm);
  REQUIRE(streamRes.sm.smCount == groupSize);
  REQUIRE(streamRes.sm.smCoscheduledAlignment >= 1);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipExecutionCtxDestroy(ctx));
}

/**
 * Test Description
 * ------------------------
 *  - Creates a regular HIP stream (not from a green context), calls
 *    hipStreamGetDevResource to get its SM resource, and verifies the resource
 *    matches hipDeviceGetDevResource for the same device.  Then partitions the
 *    derived resource via hipDevSmResourceSplit, creates a green context and
 *    stream from the partition, launches a vectorADD kernel, and verifies output.
 */
HIP_TEST_CASE(Unit_hipStreamGetDevResource_RegularStream_Functional) {
  HIP_CHECK(hipSetDevice(0));

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  // Get device resource for comparison
  hipDevResource devResource{};
  HIP_CHECK(hipDeviceGetDevResource(device, &devResource, hipDevResourceTypeSm));

  // Create a regular stream and get its resource
  hipStream_t regularStream = nullptr;
  HIP_CHECK(hipStreamCreate(&regularStream));

  hipDevResource streamRes{};
  HIP_CHECK(hipStreamGetDevResource(regularStream, &streamRes, hipDevResourceTypeSm));

  REQUIRE(streamRes.type == hipDevResourceTypeSm);
  REQUIRE(streamRes.sm.smCount == devResource.sm.smCount);
  REQUIRE(streamRes.sm.smCoscheduledAlignment == devResource.sm.smCoscheduledAlignment);
  REQUIRE(streamRes.sm.minSmPartitionSize == devResource.sm.minSmPartitionSize);

  HIP_CHECK(hipStreamDestroy(regularStream));

  // Partition the stream-derived resource
  unsigned int alignment = streamRes.sm.smCoscheduledAlignment;
  unsigned int groupSize = (streamRes.sm.smCount / 2 / alignment) * alignment;
  REQUIRE(groupSize >= alignment);

  hipDevSmResourceGroupParams params[1] = {};
  params[0].smCount = groupSize;

  hipDevResource splitResult[1] = {};
  hipDevResource remainder{};
  HIP_CHECK(hipDevSmResourceSplit(splitResult, 1, &streamRes, &remainder, 0, params));
  REQUIRE(splitResult[0].sm.smCount == groupSize);

  // Create green context from the partitioned resource and launch kernel
  RunVectorAddOnResource(&splitResult[0], 0);
}

/**
 * Test Description
 * ------------------------
 *  - Validates error codes from hipStreamGetDevResource for invalid parameters:
 *    NULL resource pointer, unsupported resource types (workqueue, workqueue
 *    config), and invalid resource type.
 */
HIP_TEST_CASE(Unit_hipStreamGetDevResource_Negative) {
  HIP_CHECK(hipSetDevice(0));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  // NULL resource pointer
  REQUIRE(hipStreamGetDevResource(stream, nullptr, hipDevResourceTypeSm)
          == hipErrorInvalidValue);

  // Unsupported resource types
  hipDevResource resource{};
  REQUIRE(hipStreamGetDevResource(stream, &resource, hipDevResourceTypeWorkqueue)
          == hipErrorInvalidResourceType);
  REQUIRE(hipStreamGetDevResource(stream, &resource, hipDevResourceTypeWorkqueueConfig)
          == hipErrorInvalidResourceType);

  // Invalid resource type
  REQUIRE(hipStreamGetDevResource(stream, &resource, hipDevResourceTypeInvalid)
          == hipErrorInvalidResourceType);

  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group hipStreamGetDevResource.
 * @}
 */
