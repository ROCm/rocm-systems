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
 *     is not written through, neither at exit nor from the crash callback; a
 *     link planted at another pid-* in the base directory, or at the root
 *     manifest's old temporary file name, is not followed (POSIX).
 *
 *   Unit_HRR_CaptureDoesNotBlockOnPlantedFifo:
 *     a FIFO planted at pid-<pid>/manifest.json does not hold up the exit
 *     (POSIX).
 *
 *   Unit_HRR_CaptureSurvivesUnusableOutputPath:
 *     an output path that cannot be created disables the capture with a
 *     message instead of failing the application's HIP calls (POSIX).
 *
 *   Unit_HRR_CaptureDisabledInForkedChild:
 *     a forked child whose archive cannot be opened says capture is disabled
 *     and leaves no empty pid-<pid> behind (POSIX).
 *
 *   Unit_HRR_CaptureKeepsRootManifestWhenBaseIsUnreadable:
 *     a base directory that cannot be listed leaves the root manifest as it
 *     was, and the application exits cleanly (POSIX, without
 *     CAP_DAC_OVERRIDE).
 *
 *   Unit_HRR_CaptureMarksUnwrittenFilesIncomplete:
 *     a blob or code object that cannot be written leaves the archive marked
 *     incomplete (POSIX).
 *
 *   Unit_HRR_CaptureResumeTrustsOnlyItsOwnFiles:
 *     resuming an archive whose directories and events.bin are readable by
 *     others makes them private again, and a hard-linked blob found there is
 *     written again rather than trusted (POSIX).
 *
 *   Unit_HRR_CaptureActiveMarker:
 *     pid-<pid>/active, the file producers read as "capture is on", exists
 *     while the capture runs and is gone after a clean exit; a stale one is
 *     removed when the archive is refused; and a resume that cannot create it
 *     disables the capture and leaves the earlier events.bin as it was (POSIX).
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <dirent.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
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

// True if p holds exactly `contents`. A bool rather than a string comparison,
// so that a failed check does not print the file: one written through holds
// binary events, which the CI's log reader cannot decode.
bool file_holds(const fs::path& p, const std::string& contents) {
  return read_text_file(p) == contents;
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

// Runs a workload with capture into base, stdout and stderr captured.
PlantedRun capture_workload(const fs::path& base, const std::string& workload) {
  hrr::test::SpawnProc proc(HRR_TEST_EXE, /*capture_stdout=*/true, /*capture_stderr=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", base.string());
  set_proc_search_path(proc);
  const int ret = proc.runWithTimeout(workload, kCaptureTimeoutSeconds);
  return {ret, proc.getOutput()};
}

// Paths, relative to the archive, of its files under `sub` with extension `ext`.
std::vector<fs::path> archive_files(const fs::path& archive, const char* sub, const char* ext) {
  std::vector<fs::path> out;
  for (const auto& ent : fs::recursive_directory_iterator(archive / sub)) {
    if (ent.is_regular_file() && ent.path().extension() == ext)
      out.push_back(fs::relative(ent.path(), archive));
  }
  std::sort(out.begin(), out.end());
  return out;
}

// The one pid-* entry under base, whatever it holds; empty unless there is
// exactly one.
fs::path only_pid_dir(const fs::path& base) {
  std::vector<fs::path> found;
  for (const auto& ent : fs::directory_iterator(base))
    if (ent.path().filename().string().rfind("pid-", 0) == 0) found.push_back(ent.path());
  return found.size() == 1 ? found.front() : fs::path{};
}

bool refused_active_marker(const std::string& output) {
  return disabled_because(output, "cannot create ", "/active (");
}

bool manifest_says_complete(const fs::path& archive, bool complete) {
  return read_text_file(archive / "manifest.json")
             .find(complete ? "\"complete\": true" : "\"complete\": false") != std::string::npos;
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

// ---------------------------------------------------------------------------
// Hidden ([.]) workload for Unit_HRR_CaptureDisabledInForkedChild: records a
// few events, then forks with RLIMIT_NOFILE at 3. The capture writer's atfork
// handler opens the child's archive before fork() returns in the child, so that
// open cannot get a descriptor and fails. The child restores the limit and
// leaves at once; it touches no HIP state.
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_CaptureForkWithoutFds_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  void* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, 256));
  HRR_HIP_CHECK(hipMemset(d, 0, 256));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // With 0, 1 and 2 taken, the first descriptor the child asks for is over the
  // limit.
  for (int fd = 0; fd < 3; ++fd)
    if (::fcntl(fd, F_GETFD) < 0) REQUIRE(::open("/dev/null", O_RDWR) == fd);
  struct rlimit saved{};
  REQUIRE(::getrlimit(RLIMIT_NOFILE, &saved) == 0);
  struct rlimit tight = saved;
  tight.rlim_cur = 3;
  // Nothing between the two setrlimit calls in the parent may need a descriptor.
  REQUIRE(::setrlimit(RLIMIT_NOFILE, &tight) == 0);
  const pid_t child = ::fork();
  if (child == 0) {
    (void)::setrlimit(RLIMIT_NOFILE, &saved);
    ::_exit(0);
  }
  const int restored = ::setrlimit(RLIMIT_NOFILE, &saved);
  REQUIRE(child > 0);
  REQUIRE(restored == 0);
  int status = 0;
  REQUIRE(::waitpid(child, &status, 0) == child);
  CHECK(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 0);
  HRR_HIP_CHECK(hipFree(d));
}

// ---------------------------------------------------------------------------
// Hidden ([.]) workload for Unit_HRR_CaptureActiveMarker: once HIP is up, and
// with it the capture, its pid-<pid>/active is a private regular file.
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_CaptureActiveMarker_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  void* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, 256));
  HRR_HIP_CHECK(hipMemset(d, 0, 256));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  const char* base = std::getenv("HIP_HRR_CAPTURE_OUTPUT");
  REQUIRE(base != nullptr);
  const std::string marker =
      std::string(base) + "/pid-" + std::to_string(::getpid()) + "/active";
  INFO("Marker: " << marker);
  struct stat st{};
  REQUIRE(::lstat(marker.c_str(), &st) == 0);
  CHECK(S_ISREG(st.st_mode));
  CHECK((st.st_mode & 07777) == 0600);
  CHECK(st.st_nlink == 1);
  HRR_HIP_CHECK(hipFree(d));
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
 *     manifest, planted in another pid-*, is not read into the root manifest.
 *   - A symbolic link at another pid-* in the base directory, pointing at a
 *     directory with a process manifest in it, is not read into the root
 *     manifest; a symbolic link at manifest.json.<pid>.tmp in the base
 *     directory, the root manifest's old predictable temporary name, is not
 *     written through.
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
    CHECK(file_holds(victim_file, contents));
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
    CHECK(file_holds(victim_file, contents));
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
    CHECK(file_holds(victim_file, contents));
  }

  SECTION("hard-linked manifest is not read into the root manifest") {
    // Shaped like a process manifest, so only the link check keeps it out. It
    // sits in another process's archive: the capture would replace one in its
    // own before the root manifest reads it.
    write_text(victim_file, "{\n  \"pid\": 424242,\n  \"complete\": true\n}\n");
    const PlantedRun run = capture_after_planting(
        base, script,
        "mkdir -p \"$HRR_TEST_BASE/pid-424242\"\n"
        "ln '" + victim_file.string() + "' \"$HRR_TEST_BASE/pid-424242/manifest.json\"\n");
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
    CHECK(file_holds(victim_file, contents));
  }

  SECTION("link at another pid-* is not read into the root manifest") {
    // A directory that holds a well-formed process manifest, reachable only
    // through the link.
    const fs::path elsewhere = work.path / "elsewhere";
    fs::create_directories(elsewhere);
    write_text(elsewhere / "manifest.json", "{\n  \"pid\": 424243,\n  \"complete\": true\n}\n");
    const PlantedRun run = capture_after_planting(
        base, script,
        "ln -s '" + elsewhere.string() + "' \"$HRR_TEST_BASE/pid-424243\"\n");
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    REQUIRE(fs::exists(base / "manifest.json"));
    CHECK(read_text_file(base / "manifest.json").find("424243") == std::string::npos);
  }

  SECTION("link at the root manifest's <pid>.tmp name") {
    // The name the root manifest's temporary file used to have, which anyone
    // who can write to the base directory could predict.
    const PlantedRun run = capture_after_planting(
        base, script,
        "ln -s '" + victim_file.string() + "' \"$HRR_TEST_BASE/manifest.json.$$.tmp\"\n");
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    CHECK(file_holds(victim_file, contents));
    CHECK(fs::is_regular_file(fs::symlink_status(base / "manifest.json")));
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

// ---------------------------------------------------------------------------
/**
 * Test Description
 * ----------------
 *   - Points HIP_HRR_CAPTURE_OUTPUT below a regular file, so no directory of
 *     the archive can be created, and runs Unit_HRR_GpuWorkload_Direct.
 *   - The workload succeeds and stderr says capture is disabled: setting up the
 *     archive runs inside hip::init, and a failure there must not reach the
 *     application's HIP calls.
 */
HRR_TEST_CASE(Unit_HRR_CaptureSurvivesUnusableOutputPath) {
#ifdef _WIN32
  HRR_SKIP("POSIX paths");
#else
  ScopedDir work{fs::temp_directory_path() / "hrr_access_unusable"};
  fs::create_directories(work.path);
  const fs::path not_a_dir = work.path / "file";
  write_text(not_a_dir, "a regular file where the capture wants a directory\n");

  const PlantedRun run = capture_workload(not_a_dir / "capture", "Unit_HRR_GpuWorkload_Direct");
  INFO("Workload exit code: " << run.ret << "\n" << run.output);
  REQUIRE(run.ret == 0);
  CHECK(refused_archive_dir(run.output));
  CHECK(fs::is_regular_file(fs::symlink_status(not_a_dir)));
#endif
}

// ---------------------------------------------------------------------------
/**
 * Test Description
 * ----------------
 *   - Runs Unit_HRR_CaptureForkWithoutFds_Direct, whose child is forked with
 *     no descriptor to spare, so its own archive cannot be opened.
 *   - The child says capture is disabled, and the pid-<pid> it created for
 *     itself is removed again, so a refused capture leaves nothing behind.
 *     The parent's archive is the only one left.
 */
HRR_TEST_CASE(Unit_HRR_CaptureDisabledInForkedChild) {
#ifdef _WIN32
  HRR_SKIP("POSIX fork");
#else
  ScopedDir work{fs::temp_directory_path() / "hrr_access_fork"};
  const fs::path base = work.path / "capture";

  const PlantedRun run = capture_workload(base, "Unit_HRR_CaptureForkWithoutFds_Direct");
  INFO("Workload exit code: " << run.ret << "\n" << run.output);
  REQUIRE(run.ret == 0);
  CHECK(refused_archive_dir(run.output));
  size_t pid_dirs = 0;
  for (const auto& ent : fs::directory_iterator(base)) {
    INFO("Path: " << ent.path().string());
    if (ent.path().filename().string().rfind("pid-", 0) == 0) ++pid_dirs;
  }
  CHECK(pid_dirs == 1);
  CHECK(hrr_process_archives(base).size() == 1);
#endif
}

// ---------------------------------------------------------------------------
/**
 * Test Description
 * ----------------
 *   - Captures Unit_HRR_GpuWorkload_Direct once, then sets the base directory
 *     to 0333 (searchable and writable, not readable) and captures it again.
 *   - The second workload exits cleanly and its archive is written, while the
 *     root manifest stays as the first capture left it: one written from a
 *     listing that stopped early would drop the archives it did not reach.
 *   - Skipped where the mode does not stop the listing (CAP_DAC_OVERRIDE).
 */
HRR_TEST_CASE(Unit_HRR_CaptureKeepsRootManifestWhenBaseIsUnreadable) {
#ifdef _WIN32
  HRR_SKIP("POSIX permission bits");
#else
  ScopedDir work{fs::temp_directory_path() / "hrr_access_unreadable"};
  const fs::path base = work.path / "capture";
  hrr_capture_direct("Unit_HRR_GpuWorkload_Direct", base);
  const std::string before = read_text_file(base / "manifest.json");
  REQUIRE(before.find("\"pid\"") != std::string::npos);

  struct RestoreMode {
    const fs::path& dir;
    ~RestoreMode() { ::chmod(dir.c_str(), 0700); }
  } restore{base};
  REQUIRE(::chmod(base.c_str(), 0333) == 0);
  if (DIR* listing = ::opendir(base.c_str())) {
    ::closedir(listing);
    HRR_SKIP("The base directory can still be listed at mode 0333 (CAP_DAC_OVERRIDE)");
  }
  const PlantedRun run = capture_workload(base, "Unit_HRR_GpuWorkload_Direct");
  REQUIRE(::chmod(base.c_str(), 0700) == 0);
  INFO("Workload exit code: " << run.ret << "\n" << run.output);
  REQUIRE(run.ret == 0);
  CHECK(hrr_process_archives(base).size() == 2);
  CHECK(read_text_file(base / "manifest.json") == before);
#endif
}

// ---------------------------------------------------------------------------
/**
 * Test Description
 * ----------------
 *   - Captures Unit_HRR_GpuWorkload_Direct once to learn the names of its
 *     blobs and code objects, then captures it again after creating a
 *     directory at each of those names in the second run's pid-<pid>, so the
 *     final rename of every blob, or of every code object, fails.
 *   - The first archive is complete; the second says complete: false, since
 *     its events refer to files it does not have.
 */
HRR_TEST_CASE(Unit_HRR_CaptureMarksUnwrittenFilesIncomplete) {
#ifdef _WIN32
  HRR_SKIP("POSIX rename semantics");
#else
  ScopedDir work{fs::temp_directory_path() / "hrr_access_unwritten"};
  const fs::path first = work.path / "first";
  const fs::path base = work.path / "capture";
  fs::create_directories(base);
  hrr_capture_direct("Unit_HRR_GpuWorkload_Direct", first);
  const fs::path first_archive = hrr_single_process_archive(first);
  REQUIRE(manifest_says_complete(first_archive, true));

  std::vector<fs::path> blocked;
  SECTION("blobs") { blocked = archive_files(first_archive, "blobs", ".blob"); }
  SECTION("code objects") { blocked = archive_files(first_archive, "code_objects", ".hsaco"); }
  REQUIRE_FALSE(blocked.empty());

  std::string body;
  for (const fs::path& rel : blocked)
    body += "mkdir -p \"$HRR_TEST_BASE/pid-$$/" + rel.string() + "\"\n";
  const PlantedRun run = capture_after_planting(base, work.path / "plant.sh", body);
  INFO("Workload exit code: " << run.ret << "\n" << run.output);
  REQUIRE(run.ret == 0);
  const std::vector<fs::path> archives = hrr_process_archives(base);
  REQUIRE(archives.size() == 1);
  CHECK(manifest_says_complete(archives.front(), false));
  for (const fs::path& rel : blocked) {
    INFO("Path: " << rel.string());
    CHECK(fs::is_directory(fs::symlink_status(archives.front() / rel)));
  }
#endif
}

// ---------------------------------------------------------------------------
/**
 * Test Description
 * ----------------
 *   - Captures Unit_HRR_GpuWorkload_Direct, copies the archive into the next
 *     run's pid-<pid> with its directories made 0755 and events.bin 0644, and
 *     replaces the largest blob with a hard link to a copy of it outside the
 *     archive; the run then resumes that archive.
 *   - The archive is resumed (events.bin grows), its directories end up 0700
 *     and events.bin and manifest.json 0600, and the hard-linked blob is not
 *     trusted: it is written again, so neither name shares its inode any more.
 */
HRR_TEST_CASE(Unit_HRR_CaptureResumeTrustsOnlyItsOwnFiles) {
#ifdef _WIN32
  HRR_SKIP("POSIX permission bits and hard links");
#else
  ScopedDir work{fs::temp_directory_path() / "hrr_access_resume"};
  const fs::path first = work.path / "first";
  const fs::path base = work.path / "capture";
  const fs::path victim = work.path / "victim.blob";
  fs::create_directories(base);
  hrr_capture_direct("Unit_HRR_GpuWorkload_Direct", first);
  const fs::path first_archive = hrr_single_process_archive(first);
  const std::uintmax_t first_bytes = fs::file_size(first_archive / "events.bin");

  // The largest blob is a host buffer the workload copies on every run.
  const std::vector<fs::path> blobs = archive_files(first_archive, "blobs", ".blob");
  REQUIRE_FALSE(blobs.empty());
  const fs::path linked = *std::max_element(
      blobs.begin(), blobs.end(), [&](const fs::path& a, const fs::path& b) {
        return fs::file_size(first_archive / a) < fs::file_size(first_archive / b);
      });
  fs::copy_file(first_archive / linked, victim);
  const std::string victim_contents = read_text_file(victim);

  const PlantedRun run = capture_after_planting(
      base, work.path / "plant.sh",
      "mkdir \"$HRR_TEST_BASE/pid-$$\"\n"
      "cp -R '" + first_archive.string() + "/.' \"$HRR_TEST_BASE/pid-$$/\"\n"
      "find \"$HRR_TEST_BASE/pid-$$\" -type d -exec chmod 0755 {} +\n"
      "chmod 0644 \"$HRR_TEST_BASE/pid-$$/events.bin\"\n"
      "ln -f '" + victim.string() + "' \"$HRR_TEST_BASE/pid-$$/" + linked.string() + "\"\n");
  INFO("Workload exit code: " << run.ret << "\n" << run.output);
  REQUIRE(run.ret == 0);
  const std::vector<fs::path> archives = hrr_process_archives(base);
  REQUIRE(archives.size() == 1);
  const fs::path archive = archives.front();
  CHECK(fs::file_size(archive / "events.bin") > first_bytes);

  CHECK(perms_of(archive) == fs::perms::owner_all);
  for (const auto& ent : fs::recursive_directory_iterator(archive)) {
    INFO("Path: " << ent.path().string());
    if (fs::is_directory(fs::symlink_status(ent.path())))
      CHECK(perms_of(ent.path()) == fs::perms::owner_all);
  }
  for (const char* name : {"events.bin", "manifest.json"}) {
    INFO("File: " << name);
    CHECK(perms_of(archive / name) == (fs::perms::owner_read | fs::perms::owner_write));
  }

  INFO("Hard-linked blob: " << linked.string());
  CHECK(fs::hard_link_count(archive / linked) == 1);
  CHECK(fs::hard_link_count(victim) == 1);
  CHECK(file_holds(victim, victim_contents));
#endif
}

// ---------------------------------------------------------------------------
/**
 * Test Description
 * ----------------
 *   - Runs Unit_HRR_CaptureActiveMarker_Direct, which finds its
 *     pid-<pid>/active while the capture runs; after a clean exit it is gone.
 *   - Plants a stale pid-<pid>/active next to a symbolic link at
 *     pid-<pid>/events.bin: the capture is refused and the stale marker is
 *     removed, so producers do not take the refused archive for a live one.
 *   - Copies a finished archive into the next run's pid-<pid>, with a
 *     non-empty directory at pid-<pid>/active: the writer opens and resumes
 *     events.bin, then cannot create the marker. Capture is disabled with a
 *     message and nothing is appended to events.bin, which stays without a
 *     marker next to it: the state a resume that fails after opening
 *     events.bin leaves behind.
 */
HRR_TEST_CASE(Unit_HRR_CaptureActiveMarker) {
#ifdef _WIN32
  HRR_SKIP("POSIX paths");
#else
  ScopedDir work{fs::temp_directory_path() / "hrr_access_marker"};
  const fs::path base = work.path / "capture";
  const fs::path victim_file = work.path / "victim.txt";
  fs::create_directories(base);
  write_text(victim_file, "must survive a capture\n");

  SECTION("present while capturing, gone after exit") {
    const PlantedRun run = capture_workload(base, "Unit_HRR_CaptureActiveMarker_Direct");
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    const std::vector<fs::path> archives = hrr_process_archives(base);
    REQUIRE(archives.size() == 1);
    CHECK(events_bytes(base) > sizeof(hrr_file_header));
    CHECK_FALSE(fs::exists(fs::symlink_status(archives.front() / "active")));
  }

  SECTION("stale marker removed when the archive is refused") {
    const PlantedRun run = capture_after_planting(
        base, work.path / "plant.sh",
        "mkdir -p \"$HRR_TEST_BASE/pid-$$\"\n"
        "touch \"$HRR_TEST_BASE/pid-$$/active\"\n"
        "ln -s '" + victim_file.string() + "' \"$HRR_TEST_BASE/pid-$$/events.bin\"\n");
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    CHECK(refused_events_file(run.output));
    const fs::path pid_dir = only_pid_dir(base);
    REQUIRE_FALSE(pid_dir.empty());
    CHECK_FALSE(fs::exists(fs::symlink_status(pid_dir / "active")));
  }

  SECTION("resume that cannot create the marker") {
    const fs::path first = work.path / "first";
    hrr_capture_direct("Unit_HRR_GpuWorkload_Direct", first);
    const fs::path first_archive = hrr_single_process_archive(first);
    const std::uintmax_t first_bytes = fs::file_size(first_archive / "events.bin");
    REQUIRE(first_bytes > sizeof(hrr_file_header));

    const PlantedRun run = capture_after_planting(
        base, work.path / "plant.sh",
        "mkdir \"$HRR_TEST_BASE/pid-$$\"\n"
        "cp -R '" + first_archive.string() + "/.' \"$HRR_TEST_BASE/pid-$$/\"\n"
        "mkdir -p \"$HRR_TEST_BASE/pid-$$/active/keep\"\n");
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    CHECK(refused_active_marker(run.output));
    const std::vector<fs::path> archives = hrr_process_archives(base);
    REQUIRE(archives.size() == 1);
    // No larger: the resume may have stripped the trailer, but nothing was
    // appended.
    CHECK(fs::file_size(archives.front() / "events.bin") <= first_bytes);
    CHECK_FALSE(fs::is_regular_file(fs::symlink_status(archives.front() / "active")));
  }
#endif
}

/**
 * @}
 */
