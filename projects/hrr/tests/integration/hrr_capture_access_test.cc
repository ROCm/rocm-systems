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
 *     every archive directory the capture creates or reuses is 0700 and every
 *     file 0600, with the umask cleared and, where HIP can run under it, set
 *     to 0277, while a missing parent of the archive gets the default mode
 *     (POSIX).
 *
 *   Unit_HRR_CaptureRefusesPlantedLinks:
 *     a symbolic link planted at pid-<pid>/events.bin, at pid-<pid> itself, or
 *     at pid-<pid>/blobs, or a hard link planted at pid-<pid>/events.bin,
 *     disables the capture; links planted at every pid-<pid>/blobs/<prefix>
 *     are not written through and leave the archive marked incomplete; and a
 *     hard link planted at pid-<pid>/manifest.json
 *     is not written through, neither at exit nor from the crash callback
 *     (POSIX).
 *
 *   Unit_HRR_CaptureDoesNotBlockOnPlantedFifo:
 *     a FIFO planted at pid-<pid>/manifest.json does not hold up the exit
 *     (POSIX).
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace {

#ifndef _WIN32
constexpr const char* kCaptureDisabled = "[HRR capture] Capture disabled: ";
// Far above what one small workload needs; a capture blocked on a planted FIFO
// never finishes.
constexpr int kCaptureTimeoutSeconds = 120;

// True when a line of the output says capture is disabled for this reason: `what`
// right after the prefix and `then` later on the same line, so the path and the
// errno text between them don't matter. Other refusals share the prefix.
bool disabled_because(const std::string& output, const std::string& what, const std::string& then) {
  const std::string head = kCaptureDisabled + what;
  for (size_t at = output.find(head); at != std::string::npos; at = output.find(head, at + 1)) {
    const size_t eol = output.find('\n', at);
    const std::string line = output.substr(at, eol == std::string::npos ? eol : eol - at);
    if (line.find(then, head.size()) != std::string::npos) return true;
  }
  return false;
}

bool refused_archive_dir(const std::string& output) {
  return disabled_because(output, "cannot use ", " as a private archive directory (");
}

bool refused_events_file(const std::string& output) {
  return disabled_because(output, "cannot open ", "/events.bin (");
}

fs::perms perms_of(const fs::path& p) {
  return fs::symlink_status(p).permissions() & fs::perms::all;
}

// Sets the umask for its lifetime; the workloads a test spawns inherit it.
struct ScopedUmask {
  mode_t saved;
  explicit ScopedUmask(mode_t mask) : saved(::umask(mask)) {}
  ~ScopedUmask() { ::umask(saved); }
};

// HIP writes into temporary directories of its own, which a umask that takes
// an owner bit away leaves unusable unless the process has CAP_DAC_OVERRIDE.
bool can_write_in_new_dir(const fs::path& dir) {
  if (::mkdir(dir.c_str(), 0700) != 0) return false;
  return static_cast<bool>(std::ofstream(dir / "probe"));
}

void write_text(const fs::path& p, const std::string& text) {
  std::ofstream out(p, std::ios::binary);
  out << text;
}

struct PlantedRun {
  int ret;
  std::string output;  // stdout and stderr
};

// Runs a workload through /bin/sh so the script can plant entries named after
// its own pid; exec keeps that pid for the workload, and therefore for the
// pid-<pid> directory the capture writes to.
PlantedRun capture_after_planting(const fs::path& base, const fs::path& script,
                                  const std::string& body,
                                  const std::string& workload = "Unit_HRR_GpuWorkload_Direct") {
  write_text(script,
             "#!/bin/sh\nset -e\n" + body + "exec \"$HRR_TEST_WORKLOAD\" " + workload + "\n");
  hrr::test::SpawnProc proc("/bin/sh", /*capture_stdout=*/true, /*capture_stderr=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", base.string());
  proc.setEnv("HRR_TEST_BASE", base.string());
  proc.setEnv("HRR_TEST_WORKLOAD", HRR_TEST_EXE);
  set_proc_search_path(proc);
  const int ret = proc.runWithTimeout(script.string(), kCaptureTimeoutSeconds);
  return {ret, proc.getOutput()};
}

// Size of the one pid-<pid>/events.bin under base, 0 unless there is exactly
// one. More than a file header tells a capture that ran from one that never
// started.
std::uintmax_t events_bytes(const fs::path& base) {
  const std::vector<fs::path> archives = hrr_process_archives(base);
  if (archives.size() != 1) return 0;
  std::error_code ec;
  const std::uintmax_t n = fs::file_size(archives.front() / "events.bin", ec);
  return ec ? 0 : n;
}
#endif

}  // namespace

#ifndef _WIN32
// ---------------------------------------------------------------------------
// Hidden ([.]) workload for the crash-callback section: records a few events,
// then dies on SIGABRT so the CLR crash callback finalizes the archive through
// emergency_finalize.
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_CaptureAbort_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  void* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, 256));
  HRR_HIP_CHECK(hipMemset(d, 0, 256));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  std::raise(SIGABRT);
}
#endif

// ---------------------------------------------------------------------------
/**
 * Test Description
 * ----------------
 *   - With the umask cleared, captures Unit_HRR_GpuWorkload_Direct into a
 *     directory whose parent does not exist yet, then captures it again into
 *     the same directory after creating that run's pid-<pid> with mode 0755.
 *   - Captures it a third time with the umask set to 0277, which takes the
 *     owner's write bit off every file and directory created under it, when
 *     the process can still write in a directory created under that umask.
 *   - Every directory in the archive is 0700 and every regular file 0600: the
 *     archive holds host buffers, kernel arguments and code objects. The parent
 *     the capture had to create is not part of the archive and gets the
 *     default mode.
 */
HRR_TEST_CASE(Unit_HRR_CaptureArchiveIsPrivate) {
#ifdef _WIN32
  HRR_SKIP("POSIX permission bits");
#else
  ScopedDir work{fs::temp_directory_path() / "hrr_access_private"};
  // Owner-only, since the second capture runs a script from it.
  REQUIRE(::mkdir(work.path.c_str(), 0700) == 0);
  const fs::path parent = work.path / "parent";
  const fs::path base = parent / "capture";
  const ScopedUmask no_umask(0);
  hrr_capture_direct("Unit_HRR_GpuWorkload_Direct", base);
  const PlantedRun run = capture_after_planting(base, work.path / "plant.sh",
                                                "mkdir -m 0755 \"$HRR_TEST_BASE/pid-$$\"\n");
  INFO("Workload exit code: " << run.ret << "\n" << run.output);
  REQUIRE(run.ret == 0);
  size_t archives = 2;
  {
    const ScopedUmask strict(0277);
    if (can_write_in_new_dir(work.path / "strict")) {
      const PlantedRun strict_run = capture_after_planting(base, work.path / "strict.sh", "");
      INFO("Exit code under umask 0277: " << strict_run.ret << "\n" << strict_run.output);
      REQUIRE(strict_run.ret == 0);
      archives = 3;
    } else {
      WARN("Not capturing under umask 0277: HIP cannot run under it without CAP_DAC_OVERRIDE");
    }
  }
  REQUIRE(hrr_process_archives(base).size() == archives);

  CHECK(perms_of(parent) == fs::perms::all);
  CHECK(perms_of(base) == fs::perms::owner_all);
  size_t files = 0;
  for (const auto& ent : fs::recursive_directory_iterator(base)) {
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
  CHECK(files >= 3 * archives);  // events.bin, manifest.json and a blob in each archive
#endif
}

// ---------------------------------------------------------------------------
/**
 * Test Description
 * ----------------
 *   - Plants a symbolic link at pid-<pid>/events.bin pointing at a file with
 *     known contents, at pid-<pid> pointing at an empty directory, at
 *     pid-<pid>/blobs pointing at an empty directory, or at every
 *     pid-<pid>/blobs/<prefix> pointing at an empty directory; a hard link at
 *     pid-<pid>/events.bin to a file with known contents; and a hard link at
 *     pid-<pid>/manifest.json pointing at a file with known contents, followed
 *     by a workload that exits cleanly or one that aborts and leaves the
 *     manifest to the crash callback.
 *   - Each target stays untouched: a resume would otherwise truncate and
 *     append to the file, and a fresh capture would fill the directory or
 *     overwrite the hard-linked manifest.
 *   - Each symbolic link and the hard link at events.bin disable the capture,
 *     which says so on stderr, and the workload still succeeds; with the links
 *     at the blob prefixes the capture runs without its blobs and its manifest
 *     says complete: false; with the hard link at manifest.json the capture
 *     runs and writes events.bin, and a hard-linked file shaped like a process
 *     manifest is not read into the root manifest.
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
    const PlantedRun run = capture_after_planting(
        base, script,
        "mkdir -p \"$HRR_TEST_BASE/pid-$$\"\n"
        "ln -s '" + victim_file.string() + "' \"$HRR_TEST_BASE/pid-$$/events.bin\"\n");
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    CHECK(refused_events_file(run.output));
    CHECK(read_text_file(victim_file) == contents);
    CHECK_FALSE(fs::exists(base / "manifest.json"));
  }

  SECTION("link at pid-<pid>") {
    const PlantedRun run = capture_after_planting(
        base, script,
        "ln -s '" + victim_dir.string() + "' \"$HRR_TEST_BASE/pid-$$\"\n");
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    CHECK(refused_archive_dir(run.output));
    CHECK(fs::is_empty(victim_dir));
    CHECK_FALSE(fs::exists(base / "manifest.json"));
  }

  SECTION("link at pid-<pid>/blobs") {
    const PlantedRun run = capture_after_planting(
        base, script,
        "mkdir -p \"$HRR_TEST_BASE/pid-$$\"\n"
        "ln -s '" + victim_dir.string() + "' \"$HRR_TEST_BASE/pid-$$/blobs\"\n");
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    CHECK(refused_archive_dir(run.output));
    CHECK(fs::is_empty(victim_dir));
    CHECK_FALSE(fs::exists(base / "manifest.json"));
  }

  SECTION("link at every pid-<pid>/blobs/<prefix>") {
    const PlantedRun run = capture_after_planting(
        base, script,
        "mkdir -m 0700 \"$HRR_TEST_BASE/pid-$$\" \"$HRR_TEST_BASE/pid-$$/blobs\"\n"
        "for a in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do\n"
        "  for b in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do\n"
        "    ln -s '" + victim_dir.string() + "' \"$HRR_TEST_BASE/pid-$$/blobs/$a$b\"\n"
        "  done\n"
        "done\n");
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    CHECK(fs::is_empty(victim_dir));
    // The manifest, not stderr: the INCOMPLETE breadcrumb may go through
    // AMD_LOG_LEVEL, which the workload leaves at its default.
    const std::vector<fs::path> archives = hrr_process_archives(base);
    REQUIRE(archives.size() == 1);
    CHECK(read_text_file(archives.front() / "manifest.json").find("\"complete\": false") !=
          std::string::npos);
  }

  SECTION("hard link at pid-<pid>/events.bin") {
    const PlantedRun run = capture_after_planting(
        base, script,
        "mkdir -p \"$HRR_TEST_BASE/pid-$$\"\n"
        "ln '" + victim_file.string() + "' \"$HRR_TEST_BASE/pid-$$/events.bin\"\n");
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    CHECK(refused_events_file(run.output));
    CHECK(read_text_file(victim_file) == contents);
    CHECK_FALSE(fs::exists(base / "manifest.json"));
  }

  SECTION("hard link at pid-<pid>/manifest.json") {
    const PlantedRun run = capture_after_planting(
        base, script,
        "mkdir -p \"$HRR_TEST_BASE/pid-$$\"\n"
        "ln '" + victim_file.string() + "' \"$HRR_TEST_BASE/pid-$$/manifest.json\"\n");
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    CHECK(events_bytes(base) > sizeof(hrr_file_header));
    CHECK(read_text_file(victim_file) == contents);
  }

  SECTION("hard-linked manifest is not read into the root manifest") {
    // Shaped like a process manifest, so only the link check keeps it out.
    write_text(victim_file, "{\n  \"pid\": 424242,\n  \"complete\": true\n}\n");
    const PlantedRun run = capture_after_planting(
        base, script,
        "mkdir -p \"$HRR_TEST_BASE/pid-$$\"\n"
        "ln '" + victim_file.string() + "' \"$HRR_TEST_BASE/pid-$$/manifest.json\"\n");
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    REQUIRE(fs::exists(base / "manifest.json"));
    CHECK(read_text_file(base / "manifest.json").find("424242") == std::string::npos);
  }

  SECTION("hard link at pid-<pid>/manifest.json, crash callback") {
    const PlantedRun run = capture_after_planting(
        base, script,
        "ulimit -c 0\n"
        "mkdir -p \"$HRR_TEST_BASE/pid-$$\"\n"
        "ln '" + victim_file.string() + "' \"$HRR_TEST_BASE/pid-$$/manifest.json\"\n",
        "Unit_HRR_CaptureAbort_Direct");
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 128 + SIGABRT);
    CHECK(events_bytes(base) > sizeof(hrr_file_header));
    CHECK(read_text_file(victim_file) == contents);
  }
#endif
}

// ---------------------------------------------------------------------------
/**
 * Test Description
 * ----------------
 *   - Plants a FIFO at pid-<pid>/manifest.json with nothing on the other end.
 *   - The workload exits cleanly before the deadline and events.bin is
 *     written: neither the manifest write at exit nor the root manifest's read
 *     of it waits for the other end of the FIFO.
 */
HRR_TEST_CASE(Unit_HRR_CaptureDoesNotBlockOnPlantedFifo) {
#ifdef _WIN32
  HRR_SKIP("POSIX FIFOs");
#else
  ScopedDir work{fs::temp_directory_path() / "hrr_access_fifo"};
  const fs::path base = work.path / "capture";
  fs::create_directories(base);

  const PlantedRun run = capture_after_planting(base, work.path / "plant.sh",
                                                "mkdir -p \"$HRR_TEST_BASE/pid-$$\"\n"
                                                "mkfifo \"$HRR_TEST_BASE/pid-$$/manifest.json\"\n");
  INFO("Workload exit code: " << run.ret << "\n" << run.output);
  REQUIRE(run.ret == 0);
  CHECK(events_bytes(base) > sizeof(hrr_file_header));
#endif
}

/**
 * @}
 */
