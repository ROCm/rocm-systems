/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Blob Refusal
 * @{
 * @ingroup HRRTest
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr_handler_test_util.h"
#include "hrr_reader.h"

#include <cstdint>
#include <string>

namespace {

struct FileCapGuard {
  uint64_t saved = hrr::max_file_bytes();
  ~FileCapGuard() { hrr::set_max_file_bytes(saved); }
};

}  // namespace

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

/** @} */
