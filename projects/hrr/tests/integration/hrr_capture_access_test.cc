/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR capture access
 * @{
 * @ingroup HRRTest
 * What the capture writer does with the file system, as opposed to what it
 * records:
 *
 *   Unit_HRR_CaptureArchiveIsPrivate:
 *     every directory the capture creates is 0700 and every file 0600 (POSIX).
 *
 *   Unit_HRR_CaptureRefusesPlantedLinks:
 *     a symbolic link planted at pid-<pid>/events.bin, at pid-<pid> itself, or
 *     at pid-<pid>/blobs, or a hard link at pid-<pid>/manifest.json, disables
 *     the write instead of being written through (POSIX).
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include <filesystem>
#include <fstream>
#include <string>

namespace {

#ifndef _WIN32
fs::perms perms_of(const fs::path& p) {
  return fs::symlink_status(p).permissions() & fs::perms::mask;
}

void write_text(const fs::path& p, const std::string& text) {
  std::ofstream out(p, std::ios::binary);
  out << text;
}

// Runs the GPU workload through /bin/sh so the script can plant entries named
// after its own pid; exec keeps that pid for the workload, and therefore for
// the pid-<pid> directory the capture writes to.
int capture_after_planting(const fs::path& base, const fs::path& script, const std::string& body) {
  write_text(script, "#!/bin/sh\nset -e\n" + body + "exec \"$HRR_TEST_WORKLOAD\" " +
                         "Unit_HRR_GpuWorkload_Direct\n");
  hrr::test::SpawnProc proc("/bin/sh");
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", base.string());
  proc.setEnv("HRR_TEST_BASE", base.string());
  proc.setEnv("HRR_TEST_WORKLOAD", HRR_TEST_EXE);
  set_proc_search_path(proc);
  return proc.run(script.string());
}
#endif

}  // namespace

// ---------------------------------------------------------------------------
/**
 * Test Description
 * ----------------
 *   - Captures Unit_HRR_GpuWorkload_Direct into a directory that does not exist
 *     yet, so the capture creates the whole tree.
 *   - Every directory in the archive is 0700 and every regular file 0600: the
 *     archive holds host buffers, kernel arguments and code objects.
 */
HRR_TEST_CASE(Unit_HRR_CaptureArchiveIsPrivate) {
#ifdef _WIN32
  HRR_SKIP("POSIX permission bits");
#else
  ScopedDir cap{fs::temp_directory_path() / "hrr_access_private"};
  hrr_capture_direct("Unit_HRR_GpuWorkload_Direct", cap.path);

  CHECK(perms_of(cap.path) == fs::perms::owner_all);
  size_t files = 0;
  for (const auto& ent : fs::recursive_directory_iterator(cap.path)) {
    INFO("Path: " << ent.path().string());
    const fs::file_status st = fs::symlink_status(ent.path());
    if (fs::is_directory(st)) {
      CHECK(perms_of(ent.path()) == fs::perms::owner_all);
    } else {
      REQUIRE(fs::is_regular_file(st));
      CHECK(perms_of(ent.path()) == (fs::perms::owner_read | fs::perms::owner_write));
      ++files;
    }
  }
  CHECK(files >= 3);  // events.bin, manifest.json, at least one blob
#endif
}

// ---------------------------------------------------------------------------
/**
 * Test Description
 * ----------------
 *   - Plants a symbolic link at pid-<pid>/events.bin pointing at a file with
 *     known contents, at pid-<pid> pointing at an empty directory, or at
 *     pid-<pid>/blobs pointing at an empty directory; and a hard link at
 *     pid-<pid>/manifest.json pointing at a file with known contents.
 *   - The workload still succeeds and each target stays untouched: a resume
 *     would otherwise truncate and append to the file, and a fresh capture
 *     would fill the directory or overwrite the hard-linked manifest.
 */
HRR_TEST_CASE(Unit_HRR_CaptureRefusesPlantedLinks) {
#ifdef _WIN32
  HRR_SKIP("POSIX symbolic links");
#else
  ScopedDir work{fs::temp_directory_path() / "hrr_access_links"};
  const fs::path base = work.path / "capture";
  const fs::path victim_file = work.path / "victim.txt";
  const fs::path victim_dir = work.path / "victim-dir";
  const fs::path script = work.path / "plant.sh";
  const std::string contents = "must survive a capture\n";
  fs::create_directories(base);
  fs::create_directories(victim_dir);
  write_text(victim_file, contents);

  SECTION("link at pid-<pid>/events.bin") {
    const int ret = capture_after_planting(
        base, script,
        "mkdir -p \"$HRR_TEST_BASE/pid-$$\"\n"
        "ln -s '" + victim_file.string() + "' \"$HRR_TEST_BASE/pid-$$/events.bin\"\n");
    INFO("Workload exit code: " << ret);
    REQUIRE(ret == 0);
    CHECK(read_text_file(victim_file) == contents);
    CHECK_FALSE(fs::exists(base / "manifest.json"));
  }

  SECTION("link at pid-<pid>") {
    const int ret = capture_after_planting(
        base, script,
        "ln -s '" + victim_dir.string() + "' \"$HRR_TEST_BASE/pid-$$\"\n");
    INFO("Workload exit code: " << ret);
    REQUIRE(ret == 0);
    CHECK(fs::is_empty(victim_dir));
  }

  SECTION("link at pid-<pid>/blobs") {
    const int ret = capture_after_planting(
        base, script,
        "mkdir -p \"$HRR_TEST_BASE/pid-$$\"\n"
        "ln -s '" + victim_dir.string() + "' \"$HRR_TEST_BASE/pid-$$/blobs\"\n");
    INFO("Workload exit code: " << ret);
    REQUIRE(ret == 0);
    CHECK(fs::is_empty(victim_dir));
    CHECK_FALSE(fs::exists(base / "manifest.json"));
  }

  SECTION("hard link at pid-<pid>/manifest.json") {
    const int ret = capture_after_planting(
        base, script,
        "mkdir -p \"$HRR_TEST_BASE/pid-$$\"\n"
        "ln '" + victim_file.string() + "' \"$HRR_TEST_BASE/pid-$$/manifest.json\"\n");
    INFO("Workload exit code: " << ret);
    REQUIRE(ret == 0);
    CHECK(read_text_file(victim_file) == contents);
  }
#endif
}

/**
 * @}
 */
