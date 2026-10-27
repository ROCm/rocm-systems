/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR hipMemcpy3D Array Handles
 * @{
 * @ingroup HRRTest
 * Handler-level tests that the srcArray and dstArray of a replayed
 * hipMemcpy3D, hipMemcpy3DAsync and their _spt forms are array handles replay
 * created, never the address the archive recorded: CLR dereferences those
 * fields, so a recorded address would make the archive choose what the runtime
 * reads. The refusals return before any HIP call, so no GPU is needed. A
 * handler that wrongly reaches HIP with a recorded address crashes in the
 * runtime, which Catch2 reports against the running test.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_reader.h"
#include "hrr_handler_test_util.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct Spelling {
  hrr_api_id_t id;
  const char* api;
  size_t parms_offset;
  size_t hashes_offset;  // blob_hash_lo, blob_hash_hi, d2h_hash_lo, d2h_hash_hi
};

// The four spellings differ only in their record type.
#define HRR_SPELLING(ID, API, T)                                                  \
  { ID, API, offsetof(T, parms_bytes), offsetof(T, blob_hash_lo) }
const Spelling kSpellings[] = {
    HRR_SPELLING(HRR_API_HIPMEMCPY3D, "hipMemcpy3D", hrr_args_hipMemcpy3D),
    HRR_SPELLING(HRR_API_HIPMEMCPY3DASYNC, "hipMemcpy3DAsync", hrr_args_hipMemcpy3DAsync),
    HRR_SPELLING(HRR_API_HIPMEMCPY3D_SPT, "hipMemcpy3D_spt", hrr_args_hipMemcpy3D_spt),
    HRR_SPELLING(HRR_API_HIPMEMCPY3DASYNC_SPT, "hipMemcpy3DAsync_spt",
                 hrr_args_hipMemcpy3DAsync_spt),
};
#undef HRR_SPELLING

constexpr uint64_t kRecordedSrc = 0x7f1100000000ull;
constexpr uint64_t kRecordedDst = 0x7f2200000000ull;

// A copy of 16 x 2 elements with the given kind and array ends (0: none); the
// pointers are recorded addresses in no map, so without the array check the copy
// would reach HIP. `hash` is the blob hash of an H2D source or D2H result.
std::vector<uint8_t> make_record(const Spelling& s, hipMemcpyKind kind, uint64_t src_array,
                                 uint64_t dst_array, uint64_t hash_lo = 0) {
  std::vector<uint8_t> rec(hrr_api_min_payload_size[s.id], 0);
  hipMemcpy3DParms parms{};
  parms.srcArray = reinterpret_cast<hipArray_t>(src_array);
  parms.dstArray = reinterpret_cast<hipArray_t>(dst_array);
  parms.srcPtr.pitch = parms.dstPtr.pitch = 16;
  parms.srcPtr.ysize = parms.dstPtr.ysize = 2;
  parms.extent = make_hipExtent(16, 2, 1);
  parms.kind = kind;
  std::memcpy(rec.data() + s.parms_offset, &parms, sizeof(parms));
  const uint64_t hash[2] = {hash_lo, hash_lo ? 2 : 0};
  const size_t at = s.hashes_offset + (kind == hipMemcpyHostToDevice ? 0 : 2 * sizeof(uint64_t));
  std::memcpy(rec.data() + at, hash, sizeof(hash));
  return rec;
}

// An archive directory holding one 32-byte blob with hash (1, 2).
struct BlobDir {
  fs::path root = fs::temp_directory_path() / "hrr_memcpy3d_array_test";
  BlobDir() {
    fs::remove_all(root);
    const std::string hex = hrr::hash_hex(1, 2);
    fs::create_directories(root / "blobs" / hex.substr(0, 2));
    std::ofstream(root / "blobs" / hex.substr(0, 2) / (hex + ".blob"), std::ios::binary)
        << std::string(32, 'x');
  }
  ~BlobDir() { fs::remove_all(root); }
};

}  // namespace

/**
 * Test Description
 * ----------------
 *   - A copy whose srcArray or dstArray was never created at replay is refused
 *     with hipErrorInvalidValue, naming the end and the recorded address, in
 *     all four spellings and whatever the copy kind. Nothing reaches HIP.
 *   - A handle that was created for the other end does not excuse the missing
 *     one.
 */
TEST_CASE("Unit_HRR_Memcpy3DArray_UnknownHandleIsRefused", "[hrr]") {
  PlaybackContext ctx;
  // Only the destination array exists at replay.
  ctx.record_array(kRecordedDst, reinterpret_cast<hipArray_t>(0x1000));

  for (hipMemcpyKind kind : {hipMemcpyDeviceToDevice, hipMemcpyHostToDevice, hipMemcpyDeviceToHost}) {
    for (const Spelling& s : kSpellings) {
      INFO(s.api << " kind " << kind);
      {  // srcArray never created
        auto rec = make_record(s, kind, kRecordedSrc, kRecordedDst);
        hipError_t r = hipSuccess;
        const std::string err = hrr_test::capture_stderr(
            [&] { r = hrr_test::dispatch(ctx, s.id, rec.data()); });
        CHECK(r == hipErrorInvalidValue);
        CHECK(err.find(s.api) != std::string::npos);
        CHECK(err.find("source array 0x7f1100000000") != std::string::npos);
      }
      {  // dstArray never created
        auto rec = make_record(s, kind, kRecordedDst, kRecordedSrc);
        hipError_t r = hipSuccess;
        const std::string err = hrr_test::capture_stderr(
            [&] { r = hrr_test::dispatch(ctx, s.id, rec.data()); });
        CHECK(r == hipErrorInvalidValue);
        CHECK(err.find("destination array 0x7f1100000000") != std::string::npos);
      }
    }
  }
}

/**
 * Test Description
 * ----------------
 *   - A host-to-array copy with a captured blob, which replay would otherwise
 *     issue, and an array-to-host copy, are not issued: the host rect is sized
 *     in bytes and the copy in array elements, so issuing them would read past
 *     the blob or write past the scratch buffer. The handler returns
 *     hipErrorNotSupported and lists the API as unreplayable, in all four
 *     spellings, and nothing reaches HIP (the live handle here is not a real
 *     array).
 *   - An array-to-host copy with a D2H check counts as a failed check, as it
 *     did before replay translated arrays; one without a check does not.
 */
TEST_CASE("Unit_HRR_Memcpy3DArray_HostCopiesAreNotIssued", "[hrr]") {
  BlobDir dir;
  PlaybackContext ctx;
  ctx.archive_dir = dir.root.string();
  ctx.validate_d2h = true;
  ctx.record_array(kRecordedSrc, reinterpret_cast<hipArray_t>(0x1000));

  for (const Spelling& s : kSpellings) {
    INFO(s.api);
    hipError_t r = hipSuccess;
    {  // host to array, blob present
      auto rec = make_record(s, hipMemcpyHostToDevice, 0, kRecordedSrc, 1);
      (void)hrr_test::capture_stderr([&] { r = hrr_test::dispatch(ctx, s.id, rec.data()); });
      CHECK(r == hipErrorNotSupported);
      CHECK(ctx.unreplayable_apis.count(s.api) == 1);
    }
    const size_t failed = ctx.d2h_fail.load();
    {  // array to host, no D2H check
      auto rec = make_record(s, hipMemcpyDeviceToHost, kRecordedSrc, 0);
      (void)hrr_test::capture_stderr([&] { r = hrr_test::dispatch(ctx, s.id, rec.data()); });
      CHECK(r == hipErrorNotSupported);
      CHECK(ctx.d2h_fail.load() == failed);
    }
    {  // array to host, D2H check
      auto rec = make_record(s, hipMemcpyDeviceToHost, kRecordedSrc, 0, 1);
      (void)hrr_test::capture_stderr([&] { r = hrr_test::dispatch(ctx, s.id, rec.data()); });
      CHECK(r == hipErrorNotSupported);
      CHECK(ctx.d2h_fail.load() == failed + 1);
    }
  }
}

/**
 * Test Description
 * ----------------
 *   - The translation step replaces a recorded srcArray and dstArray with the
 *     live handles, leaves a null field null, and does not touch the pointers.
 *     The copy handlers above stop before HIP for every handle, so this is
 *     where a known handle is seen to become the live one.
 */
TEST_CASE("Unit_HRR_Memcpy3DArray_TranslatesRecordedHandles", "[hrr]") {
  PlaybackContext ctx;
  auto* const live_src = reinterpret_cast<hipArray_t>(0x1000);
  auto* const live_dst = reinterpret_cast<hipArray_t>(0x2000);
  ctx.record_array(kRecordedSrc, live_src);
  ctx.record_array(kRecordedDst, live_dst);

  hipMemcpy3DParms parms{};
  parms.srcArray = reinterpret_cast<hipArray_t>(kRecordedSrc);
  parms.dstArray = reinterpret_cast<hipArray_t>(kRecordedDst);
  parms.srcPtr.ptr = reinterpret_cast<void*>(0xabc0);
  REQUIRE(hrr_translate_memcpy3d_arrays(ctx, "hipMemcpy3D", parms));
  CHECK(parms.srcArray == live_src);
  CHECK(parms.dstArray == live_dst);
  CHECK(parms.srcPtr.ptr == reinterpret_cast<void*>(0xabc0));

  hipMemcpy3DParms none{};
  REQUIRE(hrr_translate_memcpy3d_arrays(ctx, "hipMemcpy3D", none));
  CHECK(none.srcArray == nullptr);
  CHECK(none.dstArray == nullptr);

  hipMemcpy3DParms one{};
  one.dstArray = reinterpret_cast<hipArray_t>(kRecordedDst);
  REQUIRE(hrr_translate_memcpy3d_arrays(ctx, "hipMemcpy3D", one));
  CHECK(one.srcArray == nullptr);
  CHECK(one.dstArray == live_dst);
}

/**
 * Test Description
 * ----------------
 *   - hipMemcpy3DBatchAsync refuses a batch in which an operand, source or
 *     destination, is an array that replay never created, an array with a null
 *     handle, or of a type that is neither pointer nor array: CLR dereferences
 *     the array and leaves the operand of an unknown type uninitialised. The
 *     refusal is hipErrorInvalidValue with a stderr line naming the end.
 */
TEST_CASE("Unit_HRR_Memcpy3DBatchArray_BadOperandIsRefused", "[hrr]") {
  PlaybackContext ctx;
  ctx.record_array(kRecordedDst, reinterpret_cast<hipArray_t>(0x1000));

  struct Bad {
    const char* what;
    hipMemcpy3DOperandType type;
    uint64_t array;
  };
  const Bad bads[] = {
      {"an unknown array", hipMemcpyOperandTypeArray, kRecordedSrc},
      {"a null array", hipMemcpyOperandTypeArray, 0},
      {"a type that is neither pointer nor array", static_cast<hipMemcpy3DOperandType>(7),
       kRecordedDst},
  };
  for (const Bad& bad : bads) {
    for (bool src_side : {true, false}) {
      INFO(bad.what << (src_side ? ", source" : ", destination"));
      hrr_test::ExactRecord<hrr_args_hipMemcpy3DBatchAsync> r;
      hipMemcpy3DBatchOp op{};
      op.src.type = op.dst.type = hipMemcpyOperandTypeArray;
      op.src.op.array.array = op.dst.op.array.array = reinterpret_cast<hipArray_t>(kRecordedDst);
      hipMemcpy3DOperand& bad_end = src_side ? op.src : op.dst;
      bad_end.type = bad.type;
      bad_end.op.array.array = reinterpret_cast<hipArray_t>(bad.array);
      op.extent = make_hipExtent(16, 2, 1);
      r.get()->opList_present = 1;
      r.get()->opList_n = 1;
      r.get()->numOps = 1;
      std::memcpy(r.get()->opList_bytes, &op, sizeof(op));
      hipError_t ret = hipSuccess;
      const std::string err = hrr_test::capture_stderr(
          [&] { ret = hrr_test::dispatch(ctx, HRR_API_HIPMEMCPY3DBATCHASYNC, r.bytes()); });
      CHECK(ret == hipErrorInvalidValue);
      CHECK(err.find("hipMemcpy3DBatchAsync") != std::string::npos);
      CHECK(err.find(src_side ? "source" : "destination") != std::string::npos);
    }
  }
}

/** @} */
