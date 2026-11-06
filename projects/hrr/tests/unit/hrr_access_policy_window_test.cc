/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Access Policy Window
 * @{
 * @ingroup HRRTest
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstdint>
#include <cstring>

// Defined in hip_playback.cpp; not declared in the public header.
hipError_t playback_hipStreamSetAttribute(PlaybackContext& ctx, const uint8_t* pl);

namespace {

struct HipCall {
  int calls = 0;
  int attr = -1;
  void* base_ptr = nullptr;
};
HipCall g_stream;
HipCall g_node;

constexpr uint64_t kRecordedBase = 0x7f0000005000ull;

}  // namespace

extern "C" hipError_t hipStreamSetAttribute(hipStream_t, hipStreamAttrID attr,
                                            const hipStreamAttrValue* value) {
  ++g_stream.calls;
  g_stream.attr = static_cast<int>(attr);
  g_stream.base_ptr = value ? value->accessPolicyWindow.base_ptr : nullptr;
  return hipSuccess;
}

extern "C" hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t, hipKernelNodeAttrID attr,
                                                     const hipKernelNodeAttrValue* value) {
  ++g_node.calls;
  g_node.attr = static_cast<int>(attr);
  g_node.base_ptr = value ? value->accessPolicyWindow.base_ptr : nullptr;
  return hipSuccess;
}

TEST_CASE("Unit_HRR_AccessPolicyWindow_StreamAttributeTranslatesBasePtr", "[hrr]") {
  char live[256];
  PlaybackContext ctx;
  ctx.record_alloc(kRecordedBase, live, sizeof(live));

  hrr_test::ExactRecord<hrr_args_hipStreamSetAttribute> rec;
  auto* a = rec.get();
  a->attr = hipStreamAttributeAccessPolicyWindow;
  hipStreamAttrValue val{};
  val.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(kRecordedBase);
  std::memcpy(a->stream_attr_bytes, &val, sizeof(val));

  g_stream = {};
  REQUIRE(playback_hipStreamSetAttribute(ctx, rec.bytes()) == hipSuccess);
  CHECK(g_stream.calls == 1);
  CHECK(g_stream.attr == hipStreamAttributeAccessPolicyWindow);
  CHECK(g_stream.base_ptr == static_cast<void*>(live));

  a->attr = hipStreamAttributeSynchronizationPolicy;
  g_stream = {};
  REQUIRE(playback_hipStreamSetAttribute(ctx, rec.bytes()) == hipSuccess);
  CHECK(g_stream.calls == 1);
  CHECK(g_stream.base_ptr == reinterpret_cast<void*>(kRecordedBase));
}

TEST_CASE("Unit_HRR_AccessPolicyWindow_KernelNodeAttributeTranslatesBasePtr", "[hrr]") {
  char live[256];
  PlaybackContext ctx;
  ctx.record_alloc(kRecordedBase, live, sizeof(live));

  hrr_test::ExactRecord<hrr_args_hipGraphKernelNodeSetAttribute> rec;
  auto* a = rec.get();
  a->hNode = 0;  // no node to look up; the handler goes straight to the call
  a->attr = hipKernelNodeAttributeAccessPolicyWindow;
  a->value_present = 1;
  hipKernelNodeAttrValue val{};
  val.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(kRecordedBase);
  std::memcpy(a->value_bytes, &val, sizeof(val));

  REQUIRE(hrr_playback_dispatch[HRR_API_HIPGRAPHKERNELNODESETATTRIBUTE] != nullptr);
  g_node = {};
  REQUIRE(hrr_playback_dispatch[HRR_API_HIPGRAPHKERNELNODESETATTRIBUTE](ctx, rec.bytes()) ==
          hipSuccess);
  CHECK(g_node.calls == 1);
  CHECK(g_node.attr == hipKernelNodeAttributeAccessPolicyWindow);
  CHECK(g_node.base_ptr == static_cast<void*>(live));
}

/** @} */
