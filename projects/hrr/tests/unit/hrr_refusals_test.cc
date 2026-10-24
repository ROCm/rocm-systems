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
#include "hrr_payload_bounds.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

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

/**
 * Test Description
 * ----------------
 *   - A 2D copy that does nothing is not refused: one with a zero width or
 *     height (ihipMemcpy2D returns success before it looks at either pointer, so
 *     its ends can be null or in no map), and a host-to-host copy (a no-op, as in
 *     the 1D path). Both return hipSuccess without a HIP call and with no
 *     message.
 */
TEST_CASE("Unit_HRR_Refusals_Memcpy2DNoOpCopiesAreNotRefused", "[hrr]") {
  PlaybackContext ctx;
  struct Case { uint64_t width, height; hipMemcpyKind kind; };
  for (const Case& c : {Case{0, 2, hipMemcpyDeviceToDevice}, Case{16, 0, hipMemcpyDeviceToDevice},
                        Case{16, 2, hipMemcpyHostToHost}}) {
    hrr_test::ExactRecord<hrr_args_hipMemcpy2D> rec;
    auto* a = rec.get();
    a->dst = 0x9000;  // in no map
    a->src = 0x9100;
    a->dpitch = a->spitch = 16;
    a->width = c.width;
    a->height = c.height;
    a->kind = c.kind;
    hipError_t r = hipErrorUnknown;
    const std::string err = hrr_test::capture_stderr([&] { r = playback_hipMemcpy2D(ctx, rec.bytes()); });
    INFO(err);
    CHECK(r == hipSuccess);
    CHECK(err.empty());
  }
}

/**
 * Test Description
 * ----------------
 *   - A hipSetupArgument record whose offset and size do not fit the legacy
 *     argument buffer is refused before the HIP call: an offset near the top of
 *     the 64-bit range (where offset + size wraps in size_t, the runtime's own
 *     arithmetic), one just past the buffer, and one that asks for terabytes.
 *     The handler returns hipErrorInvalidValue and its message names the call
 *     and the event's sequence number.
 */
TEST_CASE("Unit_HRR_Refusals_SetupArgumentRefusesAWrappingOrOversizedOffset", "[hrr]") {
  struct Case { uint64_t offset; uint32_t n; };
  for (const Case& c : {Case{UINT64_MAX, 8}, Case{UINT64_MAX - 3, 8}, Case{UINT64_MAX - 255, 256},
                        Case{hrr::kMaxLegacyArgBytes - 7, 8}, Case{1ull << 40, 4}}) {
    PlaybackContext ctx;
    hrr_test::ExactRecord<hrr_args_hipSetupArgument> rec;
    auto* a = rec.get();
    a->arg_present = 1;
    a->arg_n = c.n;
    a->size = c.n;
    a->offset = c.offset;
    std::memset(a->arg_bytes, 0xAB, sizeof(a->arg_bytes));
    hipError_t r = hipSuccess;
    const std::string err =
        hrr_test::capture_stderr([&] { r = dispatch(ctx, HRR_API_HIPSETUPARGUMENT, rec.bytes()); });
    INFO("offset " << c.offset << " size " << c.n << ": " << err);
    CHECK(r == hipErrorInvalidValue);
    CHECK(err.find("hipSetupArgument") != std::string::npos);
    CHECK(err.find("sequence number") != std::string::npos);
    CHECK(err.find("refusing the call") != std::string::npos);
  }
}

/**
 * Test Description
 * ----------------
 *   - A hipSetupArgument record whose range fits the legacy argument buffer is
 *     still refused when no hipConfigureCall has been replayed on its thread:
 *     the runtime would write into the top of an empty execution-frame stack.
 *     The record runs on a fresh thread, so no earlier test's configure counts.
 */
TEST_CASE("Unit_HRR_Refusals_SetupArgumentRefusesWithoutAConfigureCall", "[hrr]") {
  PlaybackContext ctx;
  hrr_test::ExactRecord<hrr_args_hipSetupArgument> rec;
  auto* a = rec.get();
  a->arg_present = 1;
  a->arg_n = 8;
  a->size = 8;
  a->offset = 0;
  std::memset(a->arg_bytes, 0xAB, sizeof(a->arg_bytes));
  hipError_t r = hipSuccess;
  const std::string err = hrr_test::capture_stderr([&] {
    std::thread t([&] { r = hrr_playback_dispatch[HRR_API_HIPSETUPARGUMENT](ctx, rec.bytes()); });
    t.join();
  });
  INFO(err);
  CHECK(r == hipErrorInvalidValue);
  CHECK(err.find("no hipConfigureCall") != std::string::npos);
  CHECK(err.find("refusing the call") != std::string::npos);
}

/** @} */
