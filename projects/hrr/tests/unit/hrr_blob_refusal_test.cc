/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Blob Refusal
 * @{
 * @ingroup HRRTest
 * Handler-level tests (hrr-handler-tests links hip_playback.cpp in): a blob the
 * file cap refuses ends the replay, a blob that is merely absent does not, and
 * contents a metadata-only capture withheld, a blob or an inline argument value,
 * end it with the mode named.
 * No HIP call is made and no GPU is needed.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr_handler_test_util.h"
#include "hrr_reader.h"
#include "hrr/hrr_api_args.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Defined in hip_playback.cpp; not declared in the public header.
hipError_t playback_hipGraphAddKernelNode(PlaybackContext& ctx, const uint8_t* pl);

namespace {

struct FileCapGuard {
  uint64_t saved = hrr::max_file_bytes();
  ~FileCapGuard() { hrr::set_max_file_bytes(saved); }
};

}  // namespace

/**
 * Test Description
 * ----------------
 *   - A blob over --max-file-bytes is not reported as a missing blob: load_blob
 *     returns null and sets fatal_error at once, so the replay ends even under
 *     --continue-on-error and before a consuming handler reaches its HIP call.
 */
TEST_CASE("Unit_HRR_Blob_CapRefusalEndsTheReplay", "[hrr]") {
  hrr_test::BlobDir dir("0123456789abcdef");
  FileCapGuard guard;
  PlaybackContext ctx;
  ctx.archive_dir = dir.root.string();
  hrr::set_max_file_bytes(8);
  size_t sz = 0;
  const void* p = nullptr;
  const std::string err = hrr_test::capture_stderr([&] { p = ctx.load_blob(1, 2, &sz); });
  CHECK(p == nullptr);
  CHECK(ctx.fatal_error.load());
  CHECK(err.find("file cap") != std::string::npos);
}

/**
 * Test Description
 * ----------------
 *   - A blob under the cap loads, and one that is absent from the archive is
 *     null without ending the replay.
 */
TEST_CASE("Unit_HRR_Blob_AbsentBlobDoesNotEndTheReplay", "[hrr]") {
  hrr_test::BlobDir dir("0123456789abcdef");
  PlaybackContext ctx;
  ctx.archive_dir = dir.root.string();
  size_t sz = 0;
  CHECK(ctx.load_blob(1, 2, &sz) != nullptr);
  CHECK(sz == 16);
  const void* p = nullptr;
  (void)hrr_test::capture_stderr([&] { p = ctx.load_blob(7, 7, &sz); });
  CHECK(p == nullptr);
  CHECK_FALSE(ctx.fatal_error.load());
}

/**
 * Test Description
 * ----------------
 *   - A blob or code-object hash of HRR_HASH_WITHHELD, which a capture without
 *     HIP_HRR_CAPTURE_CONTENT records, is refused: the loader returns null,
 *     ends the replay, and says which capture mode would have recorded it.
 */
TEST_CASE("Unit_HRR_Blob_WithheldContentIsRefused", "[hrr]") {
  hrr_test::BlobDir dir("0123456789abcdef");
  const std::string kMode =
      "captured in metadata-only mode. Content mode (HIP_HRR_CAPTURE_CONTENT=1 at capture) "
      "would have recorded it.";
  {
    PlaybackContext ctx;
    ctx.archive_dir = dir.root.string();
    size_t sz = 0;
    const void* p = nullptr;
    const std::string err = hrr_test::capture_stderr(
        [&] { p = ctx.load_blob(HRR_HASH_WITHHELD, HRR_HASH_WITHHELD, &sz); });
    CHECK(p == nullptr);
    CHECK(ctx.fatal_error.load());
    CHECK(err.find("needs a host buffer") != std::string::npos);
    CHECK(err.find(kMode) != std::string::npos);
  }
  {
    PlaybackContext ctx;
    ctx.archive_dir = dir.root.string();
    size_t sz = 0;
    const void* p = nullptr;
    const std::string err = hrr_test::capture_stderr(
        [&] { p = ctx.load_code_object(HRR_HASH_WITHHELD, HRR_HASH_WITHHELD, &sz); });
    CHECK(p == nullptr);
    CHECK(ctx.fatal_error.load());
    CHECK(err.find("needs a code object") != std::string::npos);
    CHECK(err.find(kMode) != std::string::npos);
  }
}

/**
 * Test Description
 * ----------------
 *   - A kernel node whose code-object hash is HRR_HASH_WITHHELD carries zeros
 *     for its by-value arguments. With --replace-kernel resolved for its
 *     kernel, which skips the code-object load, it is still refused: the
 *     replay ends and the message names the withheld arguments. The node also
 *     has an incomplete dependency list, which would refuse it without ending
 *     the replay had the kernel been resolved.
 */
TEST_CASE("Unit_HRR_Blob_WithheldArgumentsRefuseReplacedKernel", "[hrr]") {
  const std::string kName = "hrr_scalar_kernel";
  PlaybackContext ctx;
  ctx.record_graph(0x1000, reinterpret_cast<hipGraph_t>(0x1));
  ctx.kernel_replacements.emplace_back(kName, "/nonexistent/replacement.hsaco");
  ctx.replacement_funcs[kName] = reinterpret_cast<hipFunction_t>(0x2);

  hrr_args_hipGraphAddKernelNode a{};
  a.graph = 0x1000;
  a.pNodeParams_present = 1;
  a.pDependencies_present = 1;
  a.pDependencies_n = 1;
  a.numDependencies = 2;
  // Tail: name, code-object hash, one four-byte by-value argument of zeros.
  std::vector<uint8_t> tail;
  auto push = [&](const void* v, size_t n) {
    tail.insert(tail.end(), static_cast<const uint8_t*>(v), static_cast<const uint8_t*>(v) + n);
  };
  const uint16_t name_len = static_cast<uint16_t>(kName.size());
  const uint64_t withheld = HRR_HASH_WITHHELD;
  const uint16_t num_args = 1, arg_size = 4;
  const uint8_t kind = 0;
  const uint32_t zero = 0;
  push(&name_len, 2); push(kName.data(), kName.size());
  push(&withheld, 8); push(&withheld, 8);
  push(&num_args, 2); push(&kind, 1); push(&arg_size, 2); push(&zero, 4);
  a.hdr.payload_length = static_cast<uint32_t>(sizeof(a) + tail.size());
  std::vector<uint8_t> rec(sizeof(a));
  std::memcpy(rec.data(), &a, sizeof(a));
  rec.insert(rec.end(), tail.begin(), tail.end());

  hipError_t r = hipErrorUnknown;
  const std::string err =
      hrr_test::capture_stderr([&] { r = playback_hipGraphAddKernelNode(ctx, rec.data()); });
  INFO("stderr: " << err);
  CHECK(ctx.fatal_error.load());
  CHECK(err.find("needs a code object and its by-value kernel arguments") != std::string::npos);
}

/**
 * Test Description
 * ----------------
 *   - A __hipRegisterFatBinary whose blob a metadata-only capture withheld is
 *     refused as a fat binary, in one line: no "blob not found" follows it.
 */
TEST_CASE("Unit_HRR_Blob_WithheldFatBinaryIsNamed", "[hrr]") {
  PlaybackContext ctx;
  hrr_test::ExactRecord<hrr_args___hipRegisterFatBinary> rec;
  rec.get()->blob_hash_lo = HRR_HASH_WITHHELD;
  rec.get()->blob_hash_hi = HRR_HASH_WITHHELD;
  rec.get()->blob_size = 4096;

  REQUIRE(hrr_playback_dispatch[HRR_API_HIPREGISTERFATBINARY] != nullptr);
  const std::string err = hrr_test::capture_stderr(
      [&] { (void)hrr_playback_dispatch[HRR_API_HIPREGISTERFATBINARY](ctx, rec.bytes()); });
  INFO("stderr: " << err);
  CHECK(ctx.fatal_error.load());
  CHECK(err.find("needs a fat binary") != std::string::npos);
  CHECK(std::count(err.begin(), err.end(), '\n') == 1);
}

/**
 * Test Description
 * ----------------
 *   - A legacy hipSetupArgument whose argument value a metadata-only capture
 *     withheld (present flag HRR_PRESENT_WITHHELD, zero bytes) is refused
 *     through the generated handler before any HIP call: the replay ends and
 *     the message names the kernel argument value and the capture mode.
 */
TEST_CASE("Unit_HRR_Blob_WithheldSetupArgumentIsRefused", "[hrr]") {
  PlaybackContext ctx;
  hrr_test::ExactRecord<hrr_args_hipSetupArgument> rec;
  auto* a = rec.get();
  a->size = 8;
  a->arg_n = 8;
  a->arg_present = HRR_PRESENT_WITHHELD;

  REQUIRE(hrr_playback_dispatch[HRR_API_HIPSETUPARGUMENT] != nullptr);
  hipError_t r = hipSuccess;
  const std::string err = hrr_test::capture_stderr(
      [&] { r = hrr_playback_dispatch[HRR_API_HIPSETUPARGUMENT](ctx, rec.bytes()); });
  INFO("stderr: " << err);
  CHECK(r == hipErrorNotSupported);
  CHECK(ctx.fatal_error.load());
  CHECK(err.find("needs a kernel argument value, which this archive does not hold because it "
                 "was captured in metadata-only mode") != std::string::npos);
}

/** @} */
