/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Refusals
 * @{
 * @ingroup HRRTest
 * Handler-level tests that an archive integer does not reach HIP as a pointer
 * or handle: the generated handlers are called through the dispatch table with
 * crafted records, the hand-written ones directly. Every case here is a
 * refusal or a failed lookup, so it returns before any HIP call and needs no
 * GPU.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstdint>
#include <cstring>
#include <string>

// Defined in hip_playback.cpp; not declared in the public header.
hipError_t playback_hipStreamSetAttribute(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipMemcpy2D(PlaybackContext& ctx, const uint8_t* pl);

namespace {

hipError_t dispatch(PlaybackContext& ctx, hrr_api_id_t id, const uint8_t* record) {
  REQUIRE(hrr_playback_dispatch[id] != nullptr);
  return hrr_playback_dispatch[id](ctx, record);
}

}  // namespace

/**
 * Test Description
 * ----------------
 *   - hipLinkAddFile is refused whatever its record holds: the path would be
 *     one the archive chose. The replay records it as unreplayable by name.
 */
TEST_CASE("Unit_HRR_Refusals_LinkAddFileIsRefused", "[hrr]") {
  PlaybackContext ctx;
  hrr_test::ExactRecord<hrr_args_hipLinkAddFile> rec;
  auto* a = rec.get();
  a->path_present = 1;
  std::strncpy(reinterpret_cast<char*>(a->path_bytes), "/etc/passwd", sizeof(a->path_bytes) - 1);

  hipError_t r = hipSuccess;
  (void)hrr_test::capture_stderr([&] { r = dispatch(ctx, HRR_API_HIPLINKADDFILE, rec.bytes()); });
  CHECK(r == hipErrorNotSupported);
  CHECK(ctx.unreplayable_apis.count("hipLinkAddFile") == 1);
}

/**
 * Test Description
 * ----------------
 *   - hipLinkCreate is refused when any recorded JIT option is one whose value
 *     is a pointer (a log buffer, a symbol table) or a number the table does
 *     not know, and the replay records it as unreplayable by name. A 256-byte
 *     option-value block full of pointer-looking values is never forwarded.
 */
TEST_CASE("Unit_HRR_Refusals_LinkCreateRefusesPointerCarryingOptions", "[hrr]") {
  for (int option : {static_cast<int>(hipJitOptionInfoLogBuffer),
                     static_cast<int>(hipJitOptionErrorLogBuffer),
                     static_cast<int>(hipJitOptionGlobalSymbolNames), 12345}) {
    PlaybackContext ctx;
    hrr_test::ExactRecord<hrr_args_hipLinkCreate> rec;
    auto* a = rec.get();
    a->options_present = 1;
    a->optionValues_present = 1;
    a->options_n = 1;
    a->optionValues_n = 1;
    std::memcpy(a->options_bytes, &option, sizeof(option));
    std::memset(a->optionValues_bytes, 0xAB, sizeof(a->optionValues_bytes));

    hipError_t r = hipSuccess;
    (void)hrr_test::capture_stderr([&] { r = dispatch(ctx, HRR_API_HIPLINKCREATE, rec.bytes()); });
    INFO("option " << option);
    CHECK(r == hipErrorNotSupported);
    CHECK(ctx.unreplayable_apis.count("hipLinkCreate") == 1);
  }
}

/**
 * Test Description
 * ----------------
 *   - A sample of the APIs the generator refuses because an argument would
 *     reach HIP as a raw pointer or handle (a texture reference, external
 *     memory, an execution context, a graphics resource, a library) are
 *     refused by name through the dispatch table.
 */
TEST_CASE("Unit_HRR_Refusals_GeneratedHandlersRefuseUntranslatableSlots", "[hrr]") {
  struct Case { hrr_api_id_t id; const char* api; size_t record_size; };
  const Case cases[] = {
      {HRR_API_HIPBINDTEXTURE, "hipBindTexture", sizeof(hrr_args_hipBindTexture)},
      {HRR_API_HIPIMPORTEXTERNALMEMORY, "hipImportExternalMemory", sizeof(hrr_args_hipImportExternalMemory)},
      {HRR_API_HIPEXECUTIONCTXSYNCHRONIZE, "hipExecutionCtxSynchronize", sizeof(hrr_args_hipExecutionCtxSynchronize)},
  };
  for (const Case& c : cases) {
    PlaybackContext ctx;
    std::vector<uint8_t> rec(c.record_size, 0);  // exact-size, zero-filled record
    hipError_t r = hipSuccess;
    (void)hrr_test::capture_stderr([&] { r = dispatch(ctx, c.id, rec.data()); });
    INFO(c.api);
    CHECK(r == hipErrorNotSupported);
    CHECK(ctx.unreplayable_apis.count(c.api) == 1);
  }
}

/**
 * Test Description
 * ----------------
 *   - hipGraphKernelNodeSetAttribute (generated) and hipStreamSetAttribute
 *     (hand-written) refuse the access policy window attribute, whose union
 *     member holds a host base pointer from the capturing process, and record
 *     the API as unreplayable.
 */
TEST_CASE("Unit_HRR_Refusals_AccessPolicyWindowAttributeIsRefused", "[hrr]") {
  {
    PlaybackContext ctx;
    hrr_test::ExactRecord<hrr_args_hipGraphKernelNodeSetAttribute> rec;
    rec.get()->hNode = 0;
    rec.get()->attr = hipKernelNodeAttributeAccessPolicyWindow;
    rec.get()->value_present = 1;
    std::memset(rec.get()->value_bytes, 0xCD, sizeof(rec.get()->value_bytes));
    hipError_t r = hipSuccess;
    (void)hrr_test::capture_stderr(
        [&] { r = dispatch(ctx, HRR_API_HIPGRAPHKERNELNODESETATTRIBUTE, rec.bytes()); });
    CHECK(r == hipErrorNotSupported);
    CHECK(ctx.unreplayable_apis.count("hipGraphKernelNodeSetAttribute") == 1);
  }
  {
    PlaybackContext ctx;
    hrr_test::ExactRecord<hrr_args_hipStreamSetAttribute> rec;
    rec.get()->stream = 0;
    rec.get()->attr = hipStreamAttributeAccessPolicyWindow;
    std::memset(rec.get()->stream_attr_bytes, 0xCD, sizeof(rec.get()->stream_attr_bytes));
    hipError_t r = hipSuccess;
    (void)hrr_test::capture_stderr([&] { r = playback_hipStreamSetAttribute(ctx, rec.bytes()); });
    CHECK(r == hipErrorNotSupported);
    CHECK(ctx.unreplayable_apis.count("hipStreamSetAttribute") == 1);
  }
}

/**
 * Test Description
 * ----------------
 *   - A 2D device-to-device copy whose destination or source is in no replay
 *     map is refused with hipErrorInvalidValue, naming the end that is
 *     missing, instead of being run against the recorded address.
 */
TEST_CASE("Unit_HRR_Refusals_Memcpy2DWithUnmappedEndIsRefused", "[hrr]") {
  PlaybackContext ctx;
  char device_storage[256];
  ctx.record_alloc(0x5000, device_storage, sizeof(device_storage));

  struct Case { uint64_t dst, src; const char* missing; };
  for (const Case& c : {Case{0x9000, 0x5000, "destination"}, Case{0x5000, 0x9000, "source"},
                        Case{0x9000, 0x9100, "destination"}}) {
    hrr_test::ExactRecord<hrr_args_hipMemcpy2D> rec;
    auto* a = rec.get();
    a->dst = c.dst;
    a->src = c.src;
    a->dpitch = a->spitch = a->width = 16;
    a->height = 2;
    a->kind = hipMemcpyDeviceToDevice;
    hipError_t r = hipSuccess;
    const std::string err = hrr_test::capture_stderr([&] { r = playback_hipMemcpy2D(ctx, rec.bytes()); });
    CHECK(r == hipErrorInvalidValue);
    CHECK(err.find(c.missing) != std::string::npos);
  }
}

/** @} */
