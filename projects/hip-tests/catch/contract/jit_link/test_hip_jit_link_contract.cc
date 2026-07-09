/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

#if HT_AMD
namespace {
constexpr uint32_t kDummyInput = 0x12345678;
constexpr char kMissingFile[] = "hip-contract-test-missing.spv";

hipLinkState_t CreateLinkState() {
  hipLinkState_t state = nullptr;
  HIP_CHECK(hipLinkCreate(0, nullptr, nullptr, &state));
  REQUIRE(state != nullptr);
  return state;
}
}  // namespace

HIP_TEST_CASE(Contract_JitLink_Create_NullState_IsRejected) {
  REQUIRE(hipLinkCreate(0, nullptr, nullptr, nullptr) != hipSuccess);
}

HIP_TEST_CASE(Contract_JitLink_CreateDestroy_RoundTrips) {
  hipLinkState_t state = CreateLinkState();

  HIP_CHECK(hipLinkDestroy(state));
}

HIP_TEST_CASE(Contract_JitLink_Destroy_InvalidHandle_IsRejected) {
  REQUIRE(hipLinkDestroy(nullptr) != hipSuccess);
}

HIP_TEST_CASE(Contract_JitLink_Complete_NullOutputs_AreRejected) {
  hipLinkState_t state = CreateLinkState();

  REQUIRE(hipLinkComplete(state, nullptr, nullptr) != hipSuccess);

  HIP_CHECK(hipLinkDestroy(state));
}

HIP_TEST_CASE(Contract_JitLink_AddData_InvalidImage_IsRejected) {
  hipLinkState_t state = CreateLinkState();

  REQUIRE(hipLinkAddData(state, hipJitInputSpirv, nullptr, 0, "invalid", 0, nullptr, nullptr) !=
          hipSuccess);
  REQUIRE(hipLinkAddData(state, hipJitInputPtx, const_cast<uint32_t*>(&kDummyInput),
                         sizeof(kDummyInput), "ptx", 0, nullptr, nullptr) != hipSuccess);

  HIP_CHECK(hipLinkDestroy(state));
}

HIP_TEST_CASE(Contract_JitLink_AddFile_InvalidInputType_IsRejected) {
  hipLinkState_t state = CreateLinkState();

  REQUIRE(hipLinkAddFile(state, hipJitInputFatBinary, kMissingFile, 0, nullptr, nullptr) !=
          hipSuccess);

  HIP_CHECK(hipLinkDestroy(state));
}
#endif  // HT_AMD
