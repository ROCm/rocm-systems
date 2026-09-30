/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipExecutionCtxResourceSplit hipExecutionCtxResourceSplit
 * @{
 * @ingroup ExecutionContextTest
 * `hipDevSmResourceSplit` and `hipDevSmResourceSplitByCount` APIs
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include "hip_executionctx_common.hh"

#include <algorithm>
#include <vector>

/**
 * Test Description
 * ------------------------
 *  - Splits device SM resources into a single group containing half the total
 *    SMs (aligned to smCoscheduledAlignment) via hipDevSmResourceSplit.
 *    Verifies the result SM count and that the remainder captures the rest.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxResourceSplit_Sanity) {
  HIP_CHECK(hipSetDevice(0));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipDevResource input{};
  HIP_CHECK(hipDeviceGetDevResource(device, &input, hipDevResourceTypeSm));

  unsigned int totalSMs = input.sm.smCount;
  unsigned int alignment = input.sm.smCoscheduledAlignment;
  unsigned int halfSMs = (totalSMs / 2 / alignment) * alignment;
  REQUIRE(halfSMs >= alignment);

  hipDevSmResourceGroupParams params[1] = {};
  params[0].smCount = halfSMs;

  hipDevResource result[1] = {};
  hipDevResource remainder{};
  HIP_CHECK(hipDevSmResourceSplit(result, 1, &input, &remainder, 0, params));

  REQUIRE(result[0].type == hipDevResourceTypeSm);
  REQUIRE(result[0].sm.smCount == halfSMs);

  REQUIRE(remainder.type == hipDevResourceTypeSm);
  REQUIRE(remainder.sm.smCount == totalSMs - halfSMs);
}

/**
 * Test Description
 * ------------------------
 *  - Splits device SM resources via hipDevSmResourceSplitByCount using discovery
 *    mode (result=NULL) to query possible groups, then performs the actual split
 *    and verifies group counts and SM totals.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxResourceSplit_By_Count_Sanity) {
  HIP_CHECK(hipSetDevice(0));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipDevResource input{};
  HIP_CHECK(hipDeviceGetDevResource(device, &input, hipDevResourceTypeSm));

  unsigned int nbGroups = 0;
  HIP_CHECK(hipDevSmResourceSplitByCount(nullptr, &nbGroups, &input, nullptr, 0, 2));
  REQUIRE(nbGroups > 0);

  std::vector<hipDevResource> result(nbGroups);
  hipDevResource remainder{};
  unsigned int actualGroups = nbGroups;
  HIP_CHECK(hipDevSmResourceSplitByCount(result.data(), &actualGroups, &input, &remainder, 0, 2));
  REQUIRE(actualGroups == nbGroups);

  unsigned int totalAssigned = 0;
  for (unsigned int i = 0; i < actualGroups; i++) {
    REQUIRE(result[i].type == hipDevResourceTypeSm);
    REQUIRE(result[i].sm.smCount >= 2);
    totalAssigned += result[i].sm.smCount;
  }
  unsigned int remainderSMs = 0;
  if (remainder.type == hipDevResourceTypeSm) {
    remainderSMs = remainder.sm.smCount;
  }
  REQUIRE(totalAssigned + remainderSMs == input.sm.smCount);
}

/**
 * Test Description
 * ------------------------
 *  - Validates error codes from hipDevSmResourceSplit for invalid parameters:
 *    NULL input, NULL groupParams, non-zero flags, wrong resource type,
 *    smCount < 2, and smCount exceeding total SMs.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxResourceSplit_Negative) {
  HIP_CHECK(hipSetDevice(0));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipDevResource input{};
  HIP_CHECK(hipDeviceGetDevResource(device, &input, hipDevResourceTypeSm));

  hipDevResource result[2] = {};
  hipDevSmResourceGroupParams params[2] = {};
  params[0].smCount = 2;
  params[1].smCount = 2;

  // NULL input
  REQUIRE(hipDevSmResourceSplit(result, 2, nullptr, nullptr, 0, params)
          == hipErrorInvalidValue);

  // NULL groupParams
  REQUIRE(hipDevSmResourceSplit(result, 2, &input, nullptr, 0, nullptr)
          == hipErrorInvalidValue);

  // Non-zero flags (currently unsupported)
  hipDevSmResourceGroupParams validParams[1] = {};
  validParams[0].smCount = 2;
  REQUIRE(hipDevSmResourceSplit(result, 1, &input, nullptr, 1, validParams)
          == hipErrorInvalidValue);

  // Wrong resource type
  hipDevResource badInput = input;
  badInput.type = hipDevResourceTypeInvalid;
  REQUIRE(hipDevSmResourceSplit(result, 1, &badInput, nullptr, 0, validParams)
          == hipErrorInvalidResourceType);

  // smCount < minSmPartitionSize (must be >= minSmPartitionSize)
  if (input.sm.minSmPartitionSize > 1) {
    hipDevSmResourceGroupParams tooSmall[1] = {};
    tooSmall[0].smCount = input.sm.minSmPartitionSize - 1;
    REQUIRE(hipDevSmResourceSplit(result, 1, &input, nullptr, 0, tooSmall)
            == hipErrorInvalidResourceConfiguration);
  }

  // smCount exceeds total SMs
  hipDevSmResourceGroupParams tooLarge[1] = {};
  tooLarge[0].smCount = input.sm.smCount + 2;
  REQUIRE(hipDevSmResourceSplit(result, 1, &input, nullptr, 0, tooLarge)
          == hipErrorInvalidResourceConfiguration);

  // Total smCount across groups exceeds available SMs
  hipDevSmResourceGroupParams overcommit[2] = {};
  overcommit[0].smCount = input.sm.smCount;
  overcommit[1].smCount = 2;
  REQUIRE(hipDevSmResourceSplit(result, 2, &input, nullptr, 0, overcommit)
          == hipErrorInvalidResourceConfiguration);
}

/**
 * Test Description
 * ------------------------
 *  - Uses hipDevSmResourceSplitByCount discovery to find the supported number
 *    of groups.  If at least 3 groups are possible, performs an asymmetric
 *    hipDevSmResourceSplit (1x and 2x alignment); otherwise falls back to a
 *    symmetric split (alignment + alignment).  Creates a execution context and
 *    stream from each partition, runs a vectorADD kernel, and verifies
 *    correctness.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxResourceSplit_Functional) {
  HIP_CHECK(hipSetDevice(0));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipDevResource input{};
  HIP_CHECK(hipDeviceGetDevResource(device, &input, hipDevResourceTypeSm));

  unsigned int totalSMs = input.sm.smCount;
  unsigned int alignment = input.sm.smCoscheduledAlignment;

  unsigned int nbGroups = 0;
  HIP_CHECK(hipDevSmResourceSplitByCount(nullptr, &nbGroups, &input, nullptr, 0, alignment));
  REQUIRE(nbGroups >= 2);

  unsigned int groupA = alignment;
  unsigned int groupB = (nbGroups >= 3) ? alignment * 2 : alignment;

  hipDevSmResourceGroupParams params[2] = {};
  params[0].smCount = groupA;
  params[1].smCount = groupB;

  hipDevResource result[2] = {};
  hipDevResource remainder{};
  HIP_CHECK(hipDevSmResourceSplit(result, 2, &input, &remainder, 0, params));

  REQUIRE(result[0].type == hipDevResourceTypeSm);
  REQUIRE(result[0].sm.smCount == groupA);
  REQUIRE(result[1].type == hipDevResourceTypeSm);
  REQUIRE(result[1].sm.smCount == groupB);

  unsigned int remainderSMs = 0;
  if (remainder.type == hipDevResourceTypeSm) {
    remainderSMs = remainder.sm.smCount;
  }
  REQUIRE(result[0].sm.smCount + result[1].sm.smCount + remainderSMs == totalSMs);

  RunVectorAddOnResource(&result[0], 0);
  RunVectorAddOnResource(&result[1], 0);
}

/**
 * Test Description
 * ------------------------
 *  - Validates error codes from hipDevSmResourceSplitByCount for invalid
 *    parameters: NULL nbGroups, NULL input, and wrong resource type.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxResourceSplit_By_Count_Negative) {
  HIP_CHECK(hipSetDevice(0));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipDevResource input{};
  HIP_CHECK(hipDeviceGetDevResource(device, &input, hipDevResourceTypeSm));

  unsigned int nbGroups = 2;

  // NULL nbGroups
  REQUIRE(hipDevSmResourceSplitByCount(nullptr, nullptr, &input, nullptr, 0, 2)
          == hipErrorInvalidValue);

  // NULL input
  REQUIRE(hipDevSmResourceSplitByCount(nullptr, &nbGroups, nullptr, nullptr, 0, 2)
          == hipErrorInvalidValue);

  // Wrong resource type
  hipDevResource badInput = input;
  badInput.type = hipDevResourceTypeInvalid;
  REQUIRE(hipDevSmResourceSplitByCount(nullptr, &nbGroups, &badInput, nullptr, 0, 2)
          == hipErrorInvalidResourceType);
}

/**
 * Test Description
 * ------------------------
 *  - Splits SM resources via hipDevSmResourceSplitByCount into 2 partitions
 *    using a minCount of ~40% of total SMs.  Any SMs left over due to
 *    alignment are captured in the remainder.  Creates a execution context and
 *    stream from each partition, runs a vectorADD kernel, and verifies
 *    correctness.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxResourceSplit_By_Count_Functional) {
  HIP_CHECK(hipSetDevice(0));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipDevResource input{};
  HIP_CHECK(hipDeviceGetDevResource(device, &input, hipDevResourceTypeSm));

  unsigned int totalSMs = input.sm.smCount;
  unsigned int alignment = input.sm.smCoscheduledAlignment;
  unsigned int minCount = min(alignment, (static_cast<unsigned int>(totalSMs * 0.4) / alignment) * alignment);

  unsigned int nbGroups = 0;
  HIP_CHECK(hipDevSmResourceSplitByCount(nullptr, &nbGroups, &input, nullptr, 0, minCount));
  REQUIRE(nbGroups >= 2);

  unsigned int requestedGroups = 2;
  hipDevResource result[2] = {};
  hipDevResource remainder{};
  HIP_CHECK(hipDevSmResourceSplitByCount(result, &requestedGroups, &input, &remainder, 0,
                                         minCount));
  REQUIRE(requestedGroups == 2);

  unsigned int assignedSMs = result[0].sm.smCount + result[1].sm.smCount;
  unsigned int remainderSMs = 0;
  if (remainder.type == hipDevResourceTypeSm) {
    remainderSMs = remainder.sm.smCount;
  }

  REQUIRE(assignedSMs + remainderSMs == totalSMs);

  RunVectorAddOnResource(&result[0], 0);
  RunVectorAddOnResource(&result[1], 0);
}

/**
 * Test Description
 * ------------------------
 *  - Uses hipDevSmResourceSplitByCount discovery to find the supported number
 *    of groups, then splits SM resources into 2 groups via hipDevSmResourceSplit
 *    where the first group gets ~65% of SMs (capped to the discoverable range)
 *    and the second group uses the backfill flag to absorb all remaining SMs.
 *    Verifies that the remainder is invalid and runs a vectorADD kernel on each
 *    partition.
 */
HIP_TEST_CASE(Unit_hipExecutionCtxResourceSplit_Backfill_Functional) {
  HIP_CHECK(hipSetDevice(0));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipDevResource input{};
  HIP_CHECK(hipDeviceGetDevResource(device, &input, hipDevResourceTypeSm));

  unsigned int totalSMs = input.sm.smCount;
  unsigned int alignment = input.sm.smCoscheduledAlignment;

  unsigned int nbGroups = 0;
  HIP_CHECK(hipDevSmResourceSplitByCount(nullptr, &nbGroups, &input, nullptr, 0, alignment));
  REQUIRE(nbGroups >= 2);

  unsigned int maxExplicit = (nbGroups - 1) * alignment;
  unsigned int group0 = (static_cast<unsigned int>(totalSMs * 0.65) / alignment) * alignment;
  group0 = std::min(group0, maxExplicit);
  REQUIRE(group0 >= alignment);

  hipDevSmResourceGroupParams params[2] = {};
  params[0].smCount = group0;
  params[1].smCount = 0;
  params[1].flags = hipDevSmResourceGroupBackfill;

  hipDevResource result[2] = {};
  hipDevResource remainder{};
  HIP_CHECK(hipDevSmResourceSplit(result, 2, &input, &remainder, 0, params));

  REQUIRE(result[0].sm.smCount == group0);
  REQUIRE(result[1].sm.smCount == totalSMs - group0);
  REQUIRE(remainder.type == hipDevResourceTypeInvalid);

  RunVectorAddOnResource(&result[0], 0);
  RunVectorAddOnResource(&result[1], 0);
}

#if HT_AMD
/**
 * Test Description
 * ------------------------
 *  - Splits SM resources into 3 equal groups via hipDevSmResourceSplit, creates
 *    a execution context and stream from each partition, retrieves CU masks via
 *    hipExtStreamGetCUMask, and verifies that every pair of masks is disjoint
 *    (bitwise AND is zero).
 */
HIP_TEST_CASE(Unit_hipExecutionCtxResourceSplit_Disjoint_Sets) {
  HIP_CHECK(hipSetDevice(0));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipDevResource input{};
  HIP_CHECK(hipDeviceGetDevResource(device, &input, hipDevResourceTypeSm));

  unsigned int totalSMs = input.sm.smCount;
  unsigned int alignment = input.sm.smCoscheduledAlignment;
  unsigned int groupSize = (totalSMs / 3 / alignment) * alignment;
  if (groupSize < alignment) {
    HIP_SKIP_TEST(HipTest::SkipReason::kSmCountTooSmall);
  }
  hipDevSmResourceGroupParams params[3] = {};
  params[0].smCount = groupSize;
  params[1].smCount = groupSize;
  params[2].smCount = groupSize;

  hipDevResource result[3] = {};
  hipDevResource remainder{};
  HIP_CHECK(hipDevSmResourceSplit(result, 3, &input, &remainder, 0, params));

  for (int i = 0; i < 3; i++) {
    REQUIRE(result[i].type == hipDevResourceTypeSm);
    REQUIRE(result[i].sm.smCount == groupSize);
  }

  unsigned int totalWGPs = (alignment > 1) ? totalSMs / alignment : totalSMs;
  unsigned int maskWords = (totalWGPs + 31) / 32;
  std::vector<std::vector<uint32_t>> masks(3);
  hipExecutionCtx_t ctx[3] = {};
  hipStream_t stream[3] = {};

  for (int i = 0; i < 3; i++) {
    hipDevResourceDesc_t desc{};
    HIP_CHECK(hipDevResourceGenerateDesc(&desc, &result[i], 1));
    HIP_CHECK(hipGreenCtxCreate(&ctx[i], desc, 0, 0));
    REQUIRE(ctx[i] != nullptr);
    HIP_CHECK(hipExecutionCtxStreamCreate(&stream[i], ctx[i], hipStreamNonBlocking, 0));
    REQUIRE(stream[i] != nullptr);

    masks[i].resize(maskWords, 0);
    HIP_CHECK(hipExtStreamGetCUMask(stream[i], maskWords, masks[i].data()));
  }

  for (int i = 0; i < 3; i++) {
    for (int j = i + 1; j < 3; j++) {
      for (unsigned int w = 0; w < maskWords; w++) {
        REQUIRE((masks[i][w] & masks[j][w]) == 0);
      }
    }
  }

  for (int i = 0; i < 3; i++) {
    HIP_CHECK(hipStreamDestroy(stream[i]));
    HIP_CHECK(hipExecutionCtxDestroy(ctx[i]));
  }
}
#endif
/**
 * End doxygen group hipExecutionCtxResourceSplit.
 * @}
 */
