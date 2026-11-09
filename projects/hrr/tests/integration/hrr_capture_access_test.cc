/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR capture access
 * @{
 * @ingroup HRRTest
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"
#include "hrr_clock_hook.hh"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
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
// Far above what a small workload needs; a capture blocked on a planted FIFO never finishes.
constexpr int kCaptureTimeoutSeconds = 120;

// True when an output line says capture is disabled for this reason: `what` right after the prefix
// and `then` later on the same line, so the path and errno text between them don't matter. Other refusals share the prefix.
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

struct ScopedUmask {
  mode_t saved;
  explicit ScopedUmask(mode_t mask) : saved(::umask(mask)) {}
  ~ScopedUmask() { ::umask(saved); }
};

// HIP writes into temporary directories of its own, which a umask that removes an owner bit
// leaves unusable unless the process has CAP_DAC_OVERRIDE.
bool can_write_in_new_dir(const fs::path& dir) {
  if (::mkdir(dir.c_str(), 0700) != 0) return false;
  return static_cast<bool>(std::ofstream(dir / "probe"));
}

void write_text(const fs::path& p, const std::string& text) {
  std::ofstream out(p, std::ios::binary);
  out << text;
}

// A bool rather than a string comparison, so a failed check does not print the file:
// one written through holds binary events, which the CI's log reader cannot decode.
bool file_holds(const fs::path& p, const std::string& contents) {
  return read_text_file(p) == contents;
}

struct PlantedRun {
  int ret;
  std::string output;  // stdout and stderr
};

// Runs a workload through /bin/sh so the script can plant entries named after its own pid;
// exec keeps that pid for the workload, and therefore for the pid-<pid> directory the capture writes to.
PlantedRun capture_after_planting(const fs::path& base, const fs::path& script,
                                  const std::string& body,
                                  const std::string& workload = "Unit_HRR_GpuWorkload_Direct") {
  write_text(script,
             "#!/bin/sh\nset -e\n" + body + "exec \"$HRR_TEST_WORKLOAD\" " + workload + "\n");
  hrr::test::SpawnProc proc("/bin/sh", /*capture_stdout=*/true, /*capture_stderr=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", base.string());
  proc.setEnv("HRR_TEST_BASE", base.string());
  proc.setEnv("HRR_TEST_WORKLOAD", hrr_self_exe());
  set_proc_search_path(proc);
  const int ret = proc.runWithTimeout(script.string(), kCaptureTimeoutSeconds);
  return {ret, proc.getOutput()};
}

// Size of the one pid-<pid>/events.bin under base, 0 unless there is exactly one.
// More than a file header tells a capture that ran from one that never started.
std::uintmax_t events_bytes(const fs::path& base) {
  const std::vector<fs::path> archives = hrr_process_archives(base);
  if (archives.size() != 1) return 0;
  std::error_code ec;
  const std::uintmax_t n = fs::file_size(archives.front() / "events.bin", ec);
  return ec ? 0 : n;
}

PlantedRun capture_workload(const fs::path& base, const std::string& workload) {
  hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true, /*capture_stderr=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", base.string());
  set_proc_search_path(proc);
  const int ret = proc.runWithTimeout(workload, kCaptureTimeoutSeconds);
  return {ret, proc.getOutput()};
}

std::vector<fs::path> archive_files(const fs::path& archive, const char* sub, const char* ext) {
  std::vector<fs::path> out;
  for (const auto& ent : fs::recursive_directory_iterator(archive / sub)) {
    if (ent.is_regular_file() && ent.path().extension() == ext)
      out.push_back(fs::relative(ent.path(), archive));
  }
  std::sort(out.begin(), out.end());
  return out;
}

fs::path only_pid_dir(const fs::path& base) {
  std::vector<fs::path> found;
  for (const auto& ent : fs::directory_iterator(base))
    if (ent.path().filename().string().rfind("pid-", 0) == 0) found.push_back(ent.path());
  return found.size() == 1 ? found.front() : fs::path{};
}

bool refused_active_marker(const std::string& output) {
  return disabled_because(output, "cannot create ", "/active (");
}

bool same_tree(const fs::path& a, const fs::path& b) {
  const auto tree = [](const fs::path& dir) {
    std::map<std::string, std::string> out;
    for (const auto& ent : fs::recursive_directory_iterator(dir))
      out[fs::relative(ent.path(), dir).string()] =
          ent.is_regular_file() ? read_text_file(ent.path()) : "(not a file)";
    return out;
  };
  return tree(a) == tree(b);
}

bool manifest_says_complete(const fs::path& archive, bool complete) {
  return read_text_file(archive / "manifest.json")
             .find(complete ? "\"complete\": true" : "\"complete\": false") != std::string::npos;
}
#endif

}  // namespace

#ifndef _WIN32
TEST_CASE("Unit_HRR_CaptureAbort_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  void* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, 256));
  HRR_HIP_CHECK(hipMemset(d, 0, 256));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  std::raise(SIGABRT);
}

// Forks with RLIMIT_NOFILE at 3 so the child's archive open cannot get a descriptor and fails.
// The child counts clock reads to detect a capture shim left in either dispatch table.
// Exit 1: runtime call read the clock, 2: compiler call did, 3: both.
static int g_child_clock_reads = 0;

TEST_CASE("Unit_HRR_CaptureForkWithoutFds_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  void* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, 256));
  HRR_HIP_CHECK(hipMemset(d, 0, 256));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  for (int fd = 0; fd < 3; ++fd)
    if (::fcntl(fd, F_GETFD) < 0) REQUIRE(::open("/dev/null", O_RDWR) == fd);
  struct rlimit saved{};
  REQUIRE(::getrlimit(RLIMIT_NOFILE, &saved) == 0);
  struct rlimit tight = saved;
  tight.rlim_cur = 3;
  REQUIRE(::setrlimit(RLIMIT_NOFILE, &tight) == 0);
  const pid_t child = ::fork();
  if (child == 0) {
    (void)hipGetLastError();  // opens the child's archive
    (void)::setrlimit(RLIMIT_NOFILE, &saved);
    t_hrr_clock_hook = [] { ++g_child_clock_reads; };
    (void)hipGetLastError();
    const bool runtime_shim = g_child_clock_reads != 0;
    g_child_clock_reads = 0;
    dim3 grid, block;
    size_t shared = 0;
    hipStream_t stream = nullptr;
    (void)__hipPushCallConfiguration(dim3(1), dim3(1), 0, nullptr);
    (void)__hipPopCallConfiguration(&grid, &block, &shared, &stream);
    const bool compiler_shim = g_child_clock_reads != 0;
    t_hrr_clock_hook = nullptr;
    ::_exit((runtime_shim ? 1 : 0) | (compiler_shim ? 2 : 0));
  }
  const int restored = ::setrlimit(RLIMIT_NOFILE, &saved);
  REQUIRE(child > 0);
  REQUIRE(restored == 0);
  int status = 0;
  REQUIRE(::waitpid(child, &status, 0) == child);
  CHECK(WIFEXITED(status));
  INFO("1: a runtime call still runs a capture shim, 2: a compiler call does, 3: both");
  CHECK(WEXITSTATUS(status) == 0);
  HRR_HIP_CHECK(hipFree(d));
}

// Once HIP is up, pid-<pid>/active is a private regular file. On Linux it holds the boot id and start time from /proc/self/stat.
// The command name is set to one with spaces and parentheses, which the writer has to read past.
TEST_CASE("Unit_HRR_CaptureActiveMarker_Direct", "[.][hrr-direct]") {
#ifdef __linux__
  std::ofstream("/proc/self/comm") << "hrr) a (b) 1 2";
#endif
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
#ifdef __linux__
  std::string boot = read_text_file("/proc/sys/kernel/random/boot_id");
  REQUIRE_FALSE(boot.empty());
  boot.pop_back();  // the newline
  const std::string stat_line = read_text_file("/proc/self/stat");
  INFO("/proc/self/stat: " << stat_line);
  std::istringstream fields(stat_line.substr(stat_line.rfind(") ") + 2));
  std::string start;
  for (int field = 3; field <= 22; ++field) fields >> start;
  CHECK(read_text_file(marker) == boot + " " + start + "\n");
#endif
  HRR_HIP_CHECK(hipFree(d));
}

// The writer removes a stale pid-<pid>/active before reading free space and creates its own after.
// The statvfs hook plants a hard link to HRR_TEST_VICTIM in between, so the writer opens and refuses it.
extern std::atomic<void (*)(const char*)> g_hrr_statvfs_hook;

namespace {
void plant_marker_link(const char* path) {
  const std::string dir(path);
  const std::string pid_dir = "/pid-" + std::to_string(::getpid());
  if (dir.size() < pid_dir.size() ||
      dir.compare(dir.size() - pid_dir.size(), pid_dir.size(), pid_dir) != 0)
    return;
  g_hrr_statvfs_hook = nullptr;
  if (const char* victim = std::getenv("HRR_TEST_VICTIM"))
    (void)::link(victim, (dir + "/active").c_str());
}
}  // namespace

TEST_CASE("Unit_HRR_CaptureMarkerRefused_Direct", "[.][hrr-direct]") {
  g_hrr_statvfs_hook = plant_marker_link;
  HRR_HIP_CHECK(hipSetDevice(0));
  g_hrr_statvfs_hook = nullptr;
}

// Renames its own pid-<pid> to "moved" and plants a symlink to HRR_TEST_VICTIM in its place, then copies 32 distinct buffers.
// Without both variables it does nothing.
TEST_CASE("Unit_HRR_CaptureRenamedArchive_Direct", "[.][hrr-direct]") {
  const char* base = std::getenv("HIP_HRR_CAPTURE_OUTPUT");
  const char* victim = std::getenv("HRR_TEST_VICTIM");
  if (base == nullptr || victim == nullptr) return;
  HRR_HIP_CHECK(hipSetDevice(0));
  const std::string pid_dir = std::string(base) + "/pid-" + std::to_string(::getpid());
  REQUIRE(::rename(pid_dir.c_str(), (std::string(base) + "/moved").c_str()) == 0);
  REQUIRE(::symlink(victim, pid_dir.c_str()) == 0);
  void* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, 256));
  for (uint32_t i = 0; i < 32; ++i) {
    const std::vector<uint32_t> host(64, i);
    HRR_HIP_CHECK(hipMemcpy(d, host.data(), 256, hipMemcpyHostToDevice));
  }
  HRR_HIP_CHECK(hipFree(d));
}

#ifdef __linux__
// Copies 512 distinct buffers, then prints how many descriptors are directories in its pid-<pid>, once live and once from an exit handler.
// The handler is registered before the first HIP call, so it runs after the capture's own. Without HIP_HRR_CAPTURE_OUTPUT it does nothing.
namespace {
char g_fd_archive[4096];  // pid-<pid>, resolved

int archive_dir_fds() {
  DIR* fds = ::opendir("/proc/self/fd");
  if (fds == nullptr) return -1;
  const size_t len = strlen(g_fd_archive);
  int n = 0;
  while (const dirent* e = ::readdir(fds)) {
    char target[4096];
    const ssize_t got = ::readlinkat(::dirfd(fds), e->d_name, target, sizeof(target) - 1);
    if (got <= 0) continue;
    target[got] = '\0';
    struct stat st{};
    if (strncmp(target, g_fd_archive, len) == 0 && (target[len] == '\0' || target[len] == '/') &&
        ::fstatat(::dirfd(fds), e->d_name, &st, 0) == 0 && S_ISDIR(st.st_mode))
      ++n;
  }
  ::closedir(fds);
  return n;
}

void report_archive_dir_fds_at_exit() {
  fprintf(stderr, "archive directory descriptors after the capture: %d\n", archive_dir_fds());
}
}  // namespace

TEST_CASE("Unit_HRR_CaptureClosesArchiveDirs_Direct", "[.][hrr-direct]") {
  const char* base = std::getenv("HIP_HRR_CAPTURE_OUTPUT");
  if (base == nullptr) return;
  REQUIRE(std::atexit(report_archive_dir_fds_at_exit) == 0);
  HRR_HIP_CHECK(hipSetDevice(0));
  const std::string pid_dir = std::string(base) + "/pid-" + std::to_string(::getpid());
  REQUIRE(::realpath(pid_dir.c_str(), g_fd_archive) != nullptr);
  void* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, 256));
  for (uint32_t i = 0; i < 512; ++i) {
    const std::vector<uint32_t> host(64, i);
    HRR_HIP_CHECK(hipMemcpy(d, host.data(), 256, hipMemcpyHostToDevice));
  }
  HRR_HIP_CHECK(hipFree(d));
  fprintf(stderr, "archive directory descriptors during the capture: %d\n", archive_dir_fds());
}
#endif

#endif

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
    // Shaped like a process manifest, so only the link check keeps it out.
    // It sits in another process's archive, because the capture would replace one in its own before the root manifest reads it.
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

  SECTION("marker refused, with the reason") {
    const PlantedRun run = capture_after_planting(
        base, work.path / "plant.sh",
        "export HRR_TEST_VICTIM='" + victim_file.string() + "'\n",
        "Unit_HRR_CaptureMarkerRefused_Direct");
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    // The inode is refused after open() succeeded, so errno is whatever an earlier call left unless the writer sets it.
    CHECK(disabled_because(run.output, "cannot create ",
                           "/active (" + std::string(std::strerror(EPERM)) + ")"));
    CHECK(file_holds(victim_file, "must survive a capture\n"));
  }
#endif
}

HRR_TEST_CASE(Unit_HRR_CaptureMovesEarlierArchiveAside) {
#ifdef _WIN32
  HRR_SKIP("planting relies on /bin/sh and exec keeping the pid");
#else
  ScopedDir work{fs::temp_directory_path() / "hrr_access_earlier"};
  const fs::path first = work.path / "first";
  const fs::path base = work.path / "capture";
  fs::create_directories(base);
  hrr_capture_direct("Unit_HRR_GpuWorkload_Direct", first);
  const fs::path earlier = hrr_single_process_archive(first);
  REQUIRE(manifest_says_complete(earlier, true));
  const std::string plant = "mkdir \"$HRR_TEST_BASE/pid-$$\"\n"
                            "cp -R '" + earlier.string() + "/.' \"$HRR_TEST_BASE/pid-$$/\"\n";

  SECTION("moved aside") {
    const PlantedRun run = capture_after_planting(base, work.path / "plant.sh", plant);
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    const std::vector<fs::path> archives = hrr_process_archives(base);
    REQUIRE(archives.size() == 1);
    const fs::path archive = archives.front();
    const fs::path prev = archive.string() + ".prev-1";
    REQUIRE(fs::is_directory(fs::symlink_status(prev)));
    CHECK(same_tree(prev, earlier));
    CHECK(run.output.find("moved it to " + prev.string() + ".") != std::string::npos);

    hrr::Archive fresh;
    REQUIRE(hrr::load_archive(archive.string(), fresh));
    REQUIRE_FALSE(fresh.events.empty());
    CHECK(fresh.events.front().header().sequence_id == 0);
    CHECK(fresh.trailer);
    CHECK(fresh.trailer_events == fresh.events.size());
    CHECK(fresh.complete);

    const std::string root = read_text_file(base / "manifest.json");
    size_t listed = 0;
    for (size_t at = root.find("\"pid\":"); at != std::string::npos; at = root.find("\"pid\":", at + 1))
      ++listed;
    CHECK(listed == 1);
    hrr::Archive resolved;
    REQUIRE(hrr::load_archive(base.string(), resolved));
    CHECK(resolved.path == archive.string());
  }

  SECTION("cannot be moved") {
    struct RestoreMode {
      const fs::path& dir;
      ~RestoreMode() { ::chmod(dir.c_str(), 0700); }
    } restore{base};
    const PlantedRun run = capture_after_planting(base, work.path / "plant.sh",
                                                  plant + "chmod 0500 \"$HRR_TEST_BASE\"\n");
    REQUIRE(::chmod(base.c_str(), 0700) == 0);
    INFO("Workload exit code: " << run.ret << "\n" << run.output);
    REQUIRE(run.ret == 0);
    const fs::path pid_dir = only_pid_dir(base);
    if (pid_dir.empty())
      HRR_SKIP("The earlier archive was moved from a base directory at mode 0500 (CAP_DAC_OVERRIDE)");
    CHECK(disabled_because(run.output, "cannot move the earlier archive at ", " aside ("));
    CHECK(same_tree(pid_dir, earlier));
  }
#endif
}

HRR_TEST_CASE(Unit_HRR_CaptureWritesThroughCheckedDirs) {
#ifdef _WIN32
  HRR_SKIP("POSIX symbolic links");
#else
  ScopedDir work{fs::temp_directory_path() / "hrr_access_renamed"};
  const fs::path base = work.path / "capture";
  const fs::path victim_dir = work.path / "victim-dir";
  fs::create_directories(base);
  fs::create_directories(victim_dir);

  const PlantedRun run = capture_after_planting(
      base, work.path / "plant.sh", "export HRR_TEST_VICTIM='" + victim_dir.string() + "'\n",
      "Unit_HRR_CaptureRenamedArchive_Direct");
  INFO("Workload exit code: " << run.ret << "\n" << run.output);
  REQUIRE(run.ret == 0);
  CHECK(fs::is_empty(victim_dir));
  const fs::path moved = base / "moved";
  REQUIRE(fs::is_directory(fs::symlink_status(moved)));
  CHECK(archive_files(moved, "blobs", ".blob").size() >= 32);
  CHECK(manifest_says_complete(moved, true));
  CHECK_FALSE(fs::exists(fs::symlink_status(moved / "active")));
  hrr::Archive archive;
  REQUIRE(hrr::load_archive(moved.string(), archive));
  CHECK(archive.complete);
#endif
}

HRR_TEST_CASE(Unit_HRR_CaptureClosesArchiveDirs) {
#ifndef __linux__
  HRR_SKIP("/proc/self/fd");
#else
  ScopedDir work{fs::temp_directory_path() / "hrr_access_dir_fds"};
  const fs::path base = work.path / "capture";
  fs::create_directories(base);
  const PlantedRun run = capture_workload(base, "Unit_HRR_CaptureClosesArchiveDirs_Direct");
  INFO("Workload exit code: " << run.ret << "\n" << run.output);
  REQUIRE(run.ret == 0);
  const std::vector<fs::path> archives = hrr_process_archives(base);
  REQUIRE(archives.size() == 1);
  size_t prefixes = 0;
  for (const auto& ent : fs::directory_iterator(archives.front() / "blobs"))
    if (ent.is_directory()) ++prefixes;
  CHECK(prefixes >= 128);
  CHECK(manifest_says_complete(archives.front(), true));
  const auto reported = [&run](const std::string& when) {
    const std::string head = "archive directory descriptors " + when + ": ";
    const size_t at = run.output.find(head);
    return at == std::string::npos ? -1 : std::atoi(run.output.c_str() + at + head.size());
  };
  // Each blob is written inside its hipMemcpy on the calling thread,
  // so the blobs/<xx> it opened is closed again before the workload counts.
  const int during = reported("during the capture");
  CHECK(during >= 0);
  CHECK(during <= 3);
  CHECK(reported("after the capture") == 0);
#endif
}

/**
 * @}
 */
