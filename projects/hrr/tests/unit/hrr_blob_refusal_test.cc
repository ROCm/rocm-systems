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
 * with the digests of a checked archive a code object or region stream swapped
 * after the check is refused where it is read. No HIP call is made and no GPU is
 * needed.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr_handler_test_util.h"
#include "hrr/hrr_regions.h"
#include "hrr_reader.h"
#include "hrr_region_map.h"

#include <cstdint>
#include <string>

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

#ifdef HRR_HAVE_ARCHIVE_SIGN
namespace {

std::string sha256_of(const std::string& bytes) {
  hrr::StreamHash h;
  h.update(bytes.data(), bytes.size());
  return h.hex();
}

void write_file(const fs::path& p, const std::string& bytes) {
  fs::create_directories(p.parent_path());
  std::ofstream(p, std::ios::binary | std::ios::trunc) << bytes;
}

}  // namespace

/**
 * Test Description
 * ----------------
 *   A replay opens a code object at each module's first launch, long after the
 *   archive check, and passes its bytes to hipModuleLoadData. With the checked
 *   archive's digests (PlaybackContext::verified):
 *   - negative control: the code object as checked loads;
 *   - the same file replaced afterwards, by a different one of the same size, is
 *     null and sets fatal_error, so the replay ends without loading it;
 *   - a code object the check never listed is refused the same way.
 */
TEST_CASE("Unit_HRR_Sign_ReplaySwappedCodeObjectIsRefused", "[hrr]") {
  const fs::path root = fs::temp_directory_path() / "hrr_verified_co_test";
  fs::remove_all(root);
  const std::string good = "gpu code A", evil = "gpu code B";
  const std::string rel = "code_objects/" + hrr::hash_hex(5, 6) + ".hsaco";
  write_file(root / rel, good);
  hrr::VerifiedFiles verified{{rel, sha256_of(good)}};

  {
    PlaybackContext ctx;
    ctx.archive_dir = root.string();
    ctx.verified = &verified;
    size_t sz = 0;
    const void* p = ctx.load_code_object(5, 6, &sz);
    REQUIRE(p != nullptr);
    CHECK(sz == good.size());
    CHECK_FALSE(ctx.fatal_error.load());
  }
  write_file(root / rel, evil);
  {
    PlaybackContext ctx;
    ctx.archive_dir = root.string();
    ctx.verified = &verified;
    size_t sz = 0;
    const void* p = nullptr;
    const std::string err = hrr_test::capture_stderr([&] { p = ctx.load_code_object(5, 6, &sz); });
    CHECK(p == nullptr);
    CHECK(ctx.fatal_error.load());
    CHECK(err.find("changed after the archive was verified") != std::string::npos);
  }
  {
    write_file(root / ("code_objects/" + hrr::hash_hex(7, 8) + ".hsaco"), good);
    PlaybackContext ctx;
    ctx.archive_dir = root.string();
    ctx.verified = &verified;
    size_t sz = 0;
    const void* p = nullptr;
    const std::string err = hrr_test::capture_stderr([&] { p = ctx.load_code_object(7, 8, &sz); });
    CHECK(p == nullptr);
    CHECK(ctx.fatal_error.load());
    CHECK(err.find("not in the signed manifest") != std::string::npos);
  }
  {
    // No digests, as without --trusted-key: the swapped file loads, as before.
    PlaybackContext ctx;
    ctx.archive_dir = root.string();
    size_t sz = 0;
    CHECK(ctx.load_code_object(5, 6, &sz) != nullptr);
  }
  fs::remove_all(root);
}

/**
 * Test Description
 * ----------------
 *   Region streams (regions/*.hrrr) are read after the archive check too:
 *   - negative control: a stream as checked loads;
 *   - a stream changed afterwards, one added, or one removed, is refused: load
 *     returns 0, refused() is set and no record is kept.
 */
TEST_CASE("Unit_HRR_Sign_RegionStreamsAreHeldToTheVerifiedDigests", "[hrr]") {
  const fs::path root = fs::temp_directory_path() / "hrr_verified_regions_test";
  fs::remove_all(root);
  hrr_file_header fh{HRR_REGION_MAGIC, HRR_REGION_VERSION, 0};
  const std::string stream(reinterpret_cast<const char*>(&fh), sizeof(fh));
  write_file(root / "regions/a.hrrr", stream);
  hrr::VerifiedFiles verified{{"regions/a.hrrr", sha256_of(stream)}};

  {
    hrr::RegionMap map;
    (void)map.load(root.string(), &verified);
    CHECK_FALSE(map.refused());
    CHECK(map.streams() == 1);
  }
  write_file(root / "regions/a.hrrr", stream + "x");  // a torn tail parses; the digest differs
  {
    hrr::RegionMap map;
    const std::string err = hrr_test::capture_stderr([&] { CHECK(map.load(root.string(), &verified) == 0); });
    CHECK(map.refused());
    CHECK(map.streams() == 0);
    CHECK(err.find("changed after the archive was verified") != std::string::npos);
  }
  write_file(root / "regions/a.hrrr", stream);
  write_file(root / "regions/b.hrrr", stream);  // never listed
  {
    hrr::RegionMap map;
    (void)hrr_test::capture_stderr([&] { (void)map.load(root.string(), &verified); });
    CHECK(map.refused());
  }
  fs::remove(root / "regions/b.hrrr");
  fs::remove(root / "regions/a.hrrr");  // listed, now missing
  {
    hrr::RegionMap map;
    (void)hrr_test::capture_stderr([&] { (void)map.load(root.string(), &verified); });
    CHECK(map.refused());
  }
  {
    hrr::RegionMap map;  // no digests: nothing is checked and nothing is there
    CHECK(map.load(root.string()) == 0);
    CHECK_FALSE(map.refused());
  }
  fs::remove_all(root);
}
#endif

/** @} */
