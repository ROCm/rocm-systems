/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Capture Smoke
 * @{
 * @ingroup HRRTest
 * Minimal HIP-side sentinel for the HRR capture path.
 *
 * Full HRR format, playback, and GPU round-trip coverage lives in projects/hrr.
 * This test only protects the HIP runtime integration point: a capture-enabled
 * amdhip64 must honor HIP_HRR_CAPTURE_OUTPUT and write an archive.
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

#include <cstdlib>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

#ifndef HRR_CLR_LIB
#define HRR_CLR_LIB ""
#endif

#ifdef _WIN32
static constexpr char kPathSep = ';';
#else
static constexpr char kPathSep = ':';
#endif

static void set_proc_runtime_env(hip::SpawnProc& proc) {
  const char* cur_path = getenv("PATH");
  proc.setEnv("PATH",
              std::string(ROCM_BIN_PATH) + kPathSep + (cur_path ? cur_path : ""));
#if !defined(_WIN32)
  if (HRR_CLR_LIB[0] != '\0') {
    const char* cur_ld = getenv("LD_LIBRARY_PATH");
    std::string ld = std::string(HRR_CLR_LIB) + kPathSep + (cur_ld ? cur_ld : "");
    proc.setEnv("LD_LIBRARY_PATH", ld);
  }
#endif
}

struct ScopedDir {
  fs::path path;
  explicit ScopedDir(fs::path p) : path(std::move(p)) { fs::remove_all(path); }
  ~ScopedDir() { fs::remove_all(path); }
};

static fs::path hrr_single_process_archive(const fs::path& root) {
  if (fs::exists(root / "events.bin")) return root;

  std::vector<fs::path> archives;
  for (const auto& ent : fs::directory_iterator(root)) {
    if (!ent.is_directory()) continue;
    const std::string name = ent.path().filename().string();
    if (name.rfind("pid-", 0) == 0 && fs::exists(ent.path() / "events.bin")) {
      archives.push_back(ent.path());
    }
  }
  INFO("Process archive count: " << archives.size());
  REQUIRE(archives.size() == 1);
  return archives.front();
}

TEST_CASE("Unit_HRR_CaptureSmoke_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipFree(nullptr));
}

HIP_TEST_CASE(Unit_HRR_CaptureSmoke) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_capture_smoke"};

  hip::SpawnProc proc(HRR_TEST_EXE);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
  set_proc_runtime_env(proc);
  int ret = proc.run("\"Unit_HRR_CaptureSmoke_Direct\"");
  INFO("Capture subprocess exit code: " << ret);
  REQUIRE(ret == 0);

  fs::path archive_path = hrr_single_process_archive(cap.path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
}

/**
 * End doxygen group HRR.
 * @}
 */
