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
 * contents a metadata-only capture withheld end it with the mode named.
 * No HIP call is made and no GPU is needed.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr_handler_test_util.h"
#include "hrr_reader.h"
#include "hrr/hrr_api_args.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

namespace fs = std::filesystem;

struct FileCapGuard {
  uint64_t saved = hrr::max_file_bytes();
  ~FileCapGuard() { hrr::set_max_file_bytes(saved); }
};

// An archive directory holding one 16-byte blob with hash (1, 2).
struct BlobDir {
  fs::path root;
  BlobDir() {
    root = fs::temp_directory_path() / "hrr_blob_refusal_test";
    fs::remove_all(root);
    const std::string hex = hrr::hash_hex(1, 2);
    fs::create_directories(root / "blobs" / hex.substr(0, 2));
    std::ofstream(root / "blobs" / hex.substr(0, 2) / (hex + ".blob"), std::ios::binary)
        << "0123456789abcdef";
  }
  ~BlobDir() { fs::remove_all(root); }
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
  BlobDir dir;
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
  BlobDir dir;
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
  BlobDir dir;
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
  {
    // Only both halves withheld means withheld; a real hash may have one.
    PlaybackContext ctx;
    ctx.archive_dir = dir.root.string();
    size_t sz = 0;
    (void)hrr_test::capture_stderr([&] { (void)ctx.load_blob(HRR_HASH_WITHHELD, 2, &sz); });
    CHECK_FALSE(ctx.fatal_error.load());
  }
}

/** @} */
