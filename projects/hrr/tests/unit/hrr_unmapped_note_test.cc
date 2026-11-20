/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Unmapped Note
 * @{
 * @ingroup HRRTest
 * Handler-level tests of which pointer misses reach the unmapped-pointer
 * policy (playback/hip_playback.h): the handlers are called through the
 * dispatch table with crafted records and the per-thread note is read back.
 * Every case returns before any HIP call, so no GPU is needed.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstdint>

namespace {

hipError_t dispatch(PlaybackContext& ctx, hrr_api_id_t id, const uint8_t* record) {
  REQUIRE(hrr_playback_dispatch[id] != nullptr);
  hrr_unmapped_note = {};
  return hrr_playback_dispatch[id](ctx, record);
}

constexpr uint64_t kReserved = 0x7f0000000000ull;

}  // namespace

/**
 * Test Description
 * ----------------
 *   - hipMemMap at an address in no reservation or allocation is noted for the
 *     policy, so it is counted and stops --strict-unmapped.
 *   - At an address inside a reservation it is skipped as before, unnoted, and
 *     hipMemUnmap, a release, is never noted.
 */
TEST_CASE("Unit_HRR_UnmappedNote_MemMap", "[hrr]") {
  PlaybackContext ctx;
  char live[16];
  ctx.vmm_va_map[kReserved] = {live, 0x200000};

  hrr_test::ExactRecord<hrr_args_hipMemMap> map;
  map.get()->size = 0x1000;
  map.get()->handle = 0x1234;

  map.get()->ptr = 0x5a0000000000ull;
  CHECK(dispatch(ctx, HRR_API_HIPMEMMAP, map.bytes()) == hipSuccess);
  CHECK(hrr_unmapped_note.count == 1);
  CHECK(hrr_unmapped_note.first == 0x5a0000000000ull);

  map.get()->ptr = kReserved + 0x1000;
  CHECK(dispatch(ctx, HRR_API_HIPMEMMAP, map.bytes()) == hipSuccess);
  CHECK(hrr_unmapped_note.count == 0);

  hrr_test::ExactRecord<hrr_args_hipMemUnmap> unmap;
  unmap.get()->ptr = 0x5a0000000000ull;
  CHECK(dispatch(ctx, HRR_API_HIPMEMUNMAP, unmap.bytes()) == hipSuccess);
  CHECK(hrr_unmapped_note.count == 0);
}

/**
 * Test Description
 * ----------------
 *   - hipIpcOpenMemHandle with a handle no replayed export matches is skipped,
 *     so its pointer is never mapped. Closing that pointer is a release of
 *     memory replay never had: it succeeds without a HIP call and is not noted,
 *     so --strict-unmapped does not stop there.
 */
TEST_CASE("Unit_HRR_UnmappedNote_IpcCloseOfSkippedImport", "[hrr]") {
  PlaybackContext ctx;
  const uint64_t imported = 0x7e0000000000ull;

  hrr_test::ExactRecord<hrr_args_hipIpcOpenMemHandle> open;
  open.get()->devPtr = imported;
  open.get()->handle_bytes[0] = 0x42;
  hipError_t r = hipErrorUnknown;
  (void)hrr_test::capture_stderr(
      [&] { r = dispatch(ctx, HRR_API_HIPIPCOPENMEMHANDLE, open.bytes()); });
  REQUIRE(r == hipSuccess);
  REQUIRE(ctx.lookup_ptr(imported) == nullptr);

  hrr_test::ExactRecord<hrr_args_hipIpcCloseMemHandle> close;
  close.get()->devPtr = imported;
  CHECK(dispatch(ctx, HRR_API_HIPIPCCLOSEMEMHANDLE, close.bytes()) == hipSuccess);
  CHECK(hrr_unmapped_note.count == 0);
}

/** @} */
