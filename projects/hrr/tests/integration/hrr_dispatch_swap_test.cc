/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR dispatch-table swap and capture shutdown
 * @{
 * @ingroup HRRTest
 * What the capture leaves behind when it shuts down, and what it refuses to
 * record once it has.
 *
 *   Unit_HRR_DispatchSwap_WrapperOutlivesCapture:
 *     A profiler-style wrapper goes into the hipMemcpy slot after the shims are
 *     installed, and a thread keeps calling hipMemcpy through it across exit.
 *     Uninstall must leave the wrapper in its slot while restoring the others,
 *     and the shim the wrapper still calls must add nothing after the trailer:
 *     no event record, and no blob the manifest does not count.
 *
 *   Unit_HRR_FailedBlobWrite_LeavesCaptureIncomplete:
 *     A blob that cannot be written leaves an event naming a missing file, so
 *     the archive must not claim to be complete.
 *
 *   Unit_HRR_FailedBlobClose_LeavesCaptureIncomplete:
 *     The same, when the blob is small enough to sit in the stdio buffer and
 *     its write fails only when the file is closed.
 *
 *   Unit_HRR_DispatchSwap_NoUnreplayableAfterTrailer:
 *     A shim reached after the trailer, whose event is dropped, must not list
 *     its API under manifest.unreplayable_apis; one reached before must.
 *
 *   Unit_HRR_Fork_ChildArchiveCompleteAfterParentFailure:
 *     The failure above marks the parent's archive only. A child forked
 *     afterwards writes its own archive and must still get its trailer, and
 *     the parent must keep capturing after fork().
 *
 * The live dispatch table is reached through rocprofiler-register, the way a
 * profiler reaches it: hrr_dispatch_tool.cc is loaded as the tool library.
 * Preloading the same library lets a workload act as the capture opens its
 * manifest, after the trailer is written.
 * Linux only, like the rocprofiler-register integration in libamdhip64.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"
#include "hrr_reader.h"
#include "hrr/hrr_api_args.h"

#if !defined(_WIN32) && defined(HRR_TEST_EXE) && defined(HRR_DISPATCH_TOOL)

// hip_api_trace.hpp names deprecated and GL interop types without including them.
#include <hip/hip_deprecated.h>
#include <hip/amd_detail/amd_hip_gl_interop.h>
#include <hip/amd_detail/hip_api_trace.hpp>

#include <dlfcn.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

#define HRR_SWAP_MARKER "HRR_DISPATCH_SWAP"
#define HRR_FORK_MARKER "HRR_FORK_CHILD"
#define HRR_LATE_MARKER "HRR_LATE_UNREPLAYABLE"
#define HRR_CLOSE_MARKER "HRR_BLOB_CLOSE"

// Big enough that the failing write is cut off by the file size limit, which
// is set well below it and well above anything else the capture writes then.
constexpr size_t kBigBlob = 8u << 20;
constexpr rlim_t kFileSizeLimit = 2u << 20;

// Small enough that fwrite() only fills the stdio buffer, which is the file
// system's block size up to BUFSIZ, so the write that fails is fclose()'s.
constexpr size_t kSmallBlob = 2048;
constexpr rlim_t kSmallFileSizeLimit = 1024;

HipDispatchTable* tool_hip_table() {
  using Fn = void* (*)();
  auto fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "hrr_dispatch_tool_hip_table"));
  return fn ? static_cast<HipDispatchTable*>(fn()) : nullptr;
}

bool tool_set_fopen_hook(void (*hook)(const char*)) {
  using Fn = void (*)(void (*)(const char*));
  auto fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "hrr_dispatch_tool_set_fopen_hook"));
  if (!fn) return false;
  fn(hook);
  return true;
}

bool tool_set_late_hook(void (*hook)()) {
  using Fn = void (*)(void (*)());
  auto fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "hrr_dispatch_tool_set_late_hook"));
  if (!fn) return false;
  fn(hook);
  return true;
}

// Makes one H2D hipMemcpy of `bytes` whose blob cannot be written: the file
// size limit cuts the write short. Unlike a permission, the limit also applies
// to root, which is what the container runners use.
hipError_t memcpy_with_failing_blob(void* dev, size_t bytes = kBigBlob,
                                    rlim_t limit = kFileSizeLimit) {
  std::vector<unsigned char> host(bytes, 0xb5);
  struct rlimit old {};
  if (getrlimit(RLIMIT_FSIZE, &old) != 0) return hipErrorUnknown;
  struct rlimit lim = old;
  if (lim.rlim_max != RLIM_INFINITY && lim.rlim_max < limit) return hipErrorUnknown;
  lim.rlim_cur = limit;
  // Past the limit write() fails with EFBIG once SIGXFSZ is ignored.
  void (*old_handler)(int) = signal(SIGXFSZ, SIG_IGN);
  if (setrlimit(RLIMIT_FSIZE, &lim) != 0) return hipErrorUnknown;
  const hipError_t err = hipMemcpy(dev, host.data(), host.size(), hipMemcpyHostToDevice);
  setrlimit(RLIMIT_FSIZE, &old);
  signal(SIGXFSZ, old_handler);
  return err;
}

// ---------------------------------------------------------------------------
// Wrapped slot
// ---------------------------------------------------------------------------

using MemcpyFn = decltype(HipDispatchTable::hipMemcpy_fn);

struct SwapState {
  HipDispatchTable* table = nullptr;
  std::vector<unsigned char> installed;  // the table with the shims in
  std::atomic<MemcpyFn> next{nullptr};
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> calls{0};
  std::thread* caller = nullptr;  // never destroyed: it may outlive exit
};
SwapState g_swap;

hipError_t wrapped_hipMemcpy(void* dst, const void* src, size_t bytes, hipMemcpyKind kind) {
  return g_swap.next.load(std::memory_order_acquire)(dst, src, bytes, kind);
}

void caller_loop(void* dev) {
  unsigned char buf[64] = {0x5e};
  for (uint64_t i = 1; !g_swap.stop.load(std::memory_order_acquire); ++i) {
    // A fresh payload each call, so each one is a new blob rather than a dedup hit.
    std::memcpy(buf + 8, &i, sizeof(i));
    (void)hipMemcpy(dev, buf, sizeof(buf), hipMemcpyHostToDevice);
    g_swap.calls.fetch_add(1, std::memory_order_release);
  }
}

// Runs after hip_capture_shutdown(). Lets the caller go on through the wrapper
// for a while first, so calls after the capture closed are certain to exist.
void swap_late_check() {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  const uint64_t at_hook = g_swap.calls.load(std::memory_order_acquire);
  while (g_swap.calls.load(std::memory_order_acquire) < at_hook + 32 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  g_swap.stop.store(true, std::memory_order_release);
  if (g_swap.calls.load(std::memory_order_acquire) < at_hook + 32) {
    printf(HRR_SWAP_MARKER " caller stalled after shutdown at %llu calls\n",
           static_cast<unsigned long long>(at_hook));
    fflush(stdout);
    _exit(5);
  }
  g_swap.caller->join();

  const HipDispatchTable* t = g_swap.table;
  const bool kept = t->hipMemcpy_fn == &wrapped_hipMemcpy;
  const size_t wrapped = offsetof(HipDispatchTable, hipMemcpy_fn);
  const auto* now = reinterpret_cast<const unsigned char*>(t);
  size_t restored = 0;
  for (size_t off = sizeof(size_t); off + sizeof(void*) <= g_swap.installed.size();
       off += sizeof(void*)) {
    if (off != wrapped && std::memcmp(now + off, g_swap.installed.data() + off, sizeof(void*)) != 0)
      ++restored;
  }
  printf(HRR_SWAP_MARKER " kept=%d restored=%zu calls=%llu\n", kept ? 1 : 0, restored,
         static_cast<unsigned long long>(g_swap.calls.load()));
  fflush(stdout);
}

TEST_CASE("Unit_HRR_DispatchSwap_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));  // hip::init() installs the shims

  HipDispatchTable* t = tool_hip_table();
  INFO("rocprofiler-register did not pass the HIP table to " HRR_DISPATCH_TOOL);
  REQUIRE(t != nullptr);
  REQUIRE(t->size >= offsetof(HipDispatchTable, hipMemcpy_fn) + sizeof(void*));

  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, 64));

  g_swap.table = t;
  const auto* bytes = reinterpret_cast<const unsigned char*>(t);
  g_swap.installed.assign(bytes, bytes + t->size);
  // What a profiler attaching after hip::init() does: chain to whatever the
  // slot holds now, which is the capture shim.
  g_swap.next.store(t->hipMemcpy_fn, std::memory_order_release);
  t->hipMemcpy_fn = &wrapped_hipMemcpy;

  REQUIRE(tool_set_late_hook(swap_late_check));
  g_swap.caller = new std::thread(caller_loop, dev);
  while (g_swap.calls.load(std::memory_order_acquire) < 64)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  // Returning starts exit with the caller still running.
}

// ---------------------------------------------------------------------------
// Failed blob write
// ---------------------------------------------------------------------------

TEST_CASE("Unit_HRR_FailedBlobWrite_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kBigBlob));
  HRR_HIP_CHECK(memcpy_with_failing_blob(dev));
  HRR_HIP_CHECK(hipFree(dev));
}

// Reports the block size and the copy's result for the driver to check: the
// blob is buffered whole only if the stdio buffer is larger than it.
TEST_CASE("Unit_HRR_FailedBlobClose_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  const char* out = std::getenv("HIP_HRR_CAPTURE_OUTPUT");
  struct stat st {};
  const long long blksize =
      out && stat(out, &st) == 0 ? static_cast<long long>(st.st_blksize) : -1;

  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kSmallBlob));
  // Under the limit no other file may grow past it, so the runtime loads
  // whatever a small copy needs first, with a copy of another size.
  std::vector<unsigned char> warm(kSmallBlob - 512, 0x4b);
  HRR_HIP_CHECK(hipMemcpy(dev, warm.data(), warm.size(), hipMemcpyHostToDevice));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  const hipError_t err = memcpy_with_failing_blob(dev, kSmallBlob, kSmallFileSizeLimit);
  printf(HRR_CLOSE_MARKER " blksize=%lld err=%d\n", blksize, static_cast<int>(err));
  fflush(stdout);
  HRR_HIP_CHECK(hipFree(dev));
}

// ---------------------------------------------------------------------------
// Unreplayable call after the trailer
// ---------------------------------------------------------------------------

using UserObjectCreateFn = decltype(HipDispatchTable::hipUserObjectCreate_fn);
std::atomic<UserObjectCreateFn> g_late_shim{nullptr};
std::string g_own_manifest;

void noop_host_fn(void*) {}

// Runs when flush() opens this process's manifest: the trailer is written and
// the manifest is not. Calls the hipUserObjectCreate shim, as a wrapper that
// kept it would, for the first time in this process.
void before_own_manifest(const char* path) {
  if (!path) return;
  const size_t n = std::strlen(path);
  if (n < g_own_manifest.size() ||
      g_own_manifest.compare(0, std::string::npos, path + n - g_own_manifest.size()) != 0)
    return;
  const UserObjectCreateFn shim = g_late_shim.exchange(nullptr);
  if (!shim) return;
  static int payload = 0;
  hipUserObject_t obj = nullptr;
  const hipError_t err = shim(&obj, &payload, noop_host_fn, 1, hipUserObjectNoDestructorSync);
  printf(HRR_LATE_MARKER " err=%d\n", static_cast<int>(err));
  fflush(stdout);
}

TEST_CASE("Unit_HRR_DispatchSwap_Unreplayable_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));  // hip::init() installs the shims
  HipDispatchTable* t = tool_hip_table();
  INFO("rocprofiler-register did not pass the HIP table to " HRR_DISPATCH_TOOL);
  REQUIRE(t != nullptr);
  REQUIRE(t->size >= offsetof(HipDispatchTable, hipUserObjectCreate_fn) + sizeof(void*));

  // Recorded while capture runs, so listed with its event.
  HRR_HIP_CHECK(hipLaunchHostFunc(nullptr, noop_host_fn, nullptr));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  g_own_manifest = "/pid-" + std::to_string(getpid()) + "/manifest.json";
  g_late_shim.store(t->hipUserObjectCreate_fn);
  INFO("the dispatch tool was not preloaded");
  REQUIRE(tool_set_fopen_hook(before_own_manifest));
}

// ---------------------------------------------------------------------------
// fork() after a failure
// ---------------------------------------------------------------------------

pid_t g_fork_parent = 0;

// The child's exit runs hip_capture_shutdown() and then this. Leaving straight
// away skips the HIP runtime's own teardown, which a forked child cannot run.
void fork_late_hook() {
  if (getpid() != g_fork_parent) _exit(0);
}

TEST_CASE("Unit_HRR_Fork_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  INFO("rocprofiler-register did not pass the HIP table to " HRR_DISPATCH_TOOL);
  REQUIRE(tool_hip_table() != nullptr);

  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kBigBlob));
  HRR_HIP_CHECK(memcpy_with_failing_blob(dev));

  g_fork_parent = getpid();
  REQUIRE(tool_set_late_hook(fork_late_hook));
  fflush(stdout);
  fflush(stderr);
  const pid_t child = fork();
  if (child == 0) std::exit(0);
  REQUIRE(child > 0);
  printf(HRR_FORK_MARKER " %d\n", static_cast<int>(child));
  fflush(stdout);

  int status = 0;
  pid_t done = 0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  while ((done = waitpid(child, &status, WNOHANG)) == 0 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  if (done == 0) {
    kill(child, SIGKILL);
    waitpid(child, &status, 0);
    FAIL("the forked child did not exit within 120 s");
  }
  REQUIRE(done == child);
  INFO("child wait status " << status);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);

  // fork() must hand the parent back both writer locks: this call takes them.
  unsigned char buf[64] = {0x3c};
  HRR_HIP_CHECK(hipMemcpy(dev, buf, sizeof(buf), hipMemcpyHostToDevice));
  HRR_HIP_CHECK(hipFree(dev));
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

// The baked-in path, or the same file next to this binary when the build was
// unpacked somewhere else. CMake builds both into the same directory.
std::string dispatch_tool_path() {
  std::error_code ec;
  if (fs::exists(HRR_DISPATCH_TOOL, ec)) return HRR_DISPATCH_TOOL;
  const fs::path self = fs::read_symlink("/proc/self/exe", ec);
  if (ec) return HRR_DISPATCH_TOOL;
  return (self.parent_path() / fs::path(HRR_DISPATCH_TOOL).filename()).string();
}

std::string capture_workload(const fs::path& cap, const char* direct_case, bool with_tool,
                             bool preload_tool = false) {
  hrr::test::SpawnProc proc(hrr_test_exe(), /*capture_stdout=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.string());
  if (with_tool) proc.setEnv("ROCPROFILER_REGISTER_LIBRARY", dispatch_tool_path());
  if (preload_tool) proc.setEnv("LD_PRELOAD", dispatch_tool_path());
  set_proc_search_path(proc);
  const int ret = proc.runWithTimeout(std::string("\"") + direct_case + "\"", 600);
  std::string out = proc.getOutput();
  INFO(direct_case << " exit: " << ret << "\n" << out);
  REQUIRE(ret == 0);
  return out;
}

// The token after "key": in a manifest.json, or "" if the key is absent.
std::string manifest_value(const fs::path& archive, const std::string& key) {
  const std::string text = read_text_file(archive / "manifest.json");
  const std::string needle = "\"" + key + "\":";
  size_t at = text.find(needle);
  if (at == std::string::npos) return "";
  at += needle.size();
  while (at < text.size() && text[at] == ' ') ++at;
  size_t end = at;
  while (end < text.size() && text[end] != ',' && text[end] != '\n' && text[end] != '}') ++end;
  return text.substr(at, end - at);
}

// Files under `dir` with `extension`, and of them those exactly `size` bytes long.
std::pair<size_t, size_t> count_files(const fs::path& dir, const std::string& extension,
                                      uintmax_t size = 0) {
  std::pair<size_t, size_t> n{0, 0};
  std::error_code ec;
  for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
    if (!it->is_regular_file() || it->path().extension() != extension) continue;
    ++n.first;
    if (fs::file_size(it->path()) == size) ++n.second;
  }
  return n;
}

struct TrailerScan {
  bool trailer = false;
  size_t after = 0;  // records past the first trailer
};

// load_archive() keeps reading past the trailer, so look at the raw stream.
TrailerScan scan_events(const fs::path& archive) {
  TrailerScan scan;
  uint16_t version = 0;
  FILE* f = hrr::open_record_stream((archive / "events.bin").string(), HRR_MAGIC, HRR_VERSION,
                                    &version);
  REQUIRE(f != nullptr);
  std::vector<uint8_t> raw;
  while (hrr::read_raw_record(f, raw) == hrr::RecordStatus::Ok) {
    const auto* hdr = reinterpret_cast<const hrr_event_header*>(raw.data());
    if (scan.trailer)
      ++scan.after;
    else if (hdr->event_type == HRR_EOF_MARKER)
      scan.trailer = true;
  }
  fclose(f);
  return scan;
}

}  // namespace

// Before the slot-by-slot swap, uninstall copied the whole table back and the
// wrapper vanished; and a shim reached through it after flush() appended to a
// finalized archive.
HRR_TEST_CASE(Unit_HRR_DispatchSwap_WrapperOutlivesCapture) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_dispatch_swap.hrr");
  const std::string out = capture_workload(cap.path, "Unit_HRR_DispatchSwap_Direct", true);
  INFO("Workload output:\n" << out);

  const size_t at = out.find(HRR_SWAP_MARKER " kept=");
  REQUIRE(at != std::string::npos);
  int kept = -1;
  size_t restored = 0;
  unsigned long long calls = 0;
  REQUIRE(sscanf(out.c_str() + at, HRR_SWAP_MARKER " kept=%d restored=%zu calls=%llu", &kept,
                 &restored, &calls) == 3);
  // Uninstall ran before the check, and left the wrapper where it was.
  CHECK(restored > 0);
  CHECK(kept == 1);

  const fs::path archive = hrr_single_process_archive(cap.path);
  hrr::Archive ar;
  REQUIRE(hrr::load_archive(archive.string(), ar));
  CHECK(ar.complete);
  size_t memcpys = 0;
  for (const auto& ev : ar.events)
    if (ev.header().event_type == HRR_API_HIPMEMCPY) ++memcpys;
  INFO("hipMemcpy calls " << calls << ", recorded " << memcpys);
  // The shim was reached through the wrapper, and was still being called once
  // the archive stopped taking records.
  REQUIRE(memcpys > 0);
  REQUIRE(calls > memcpys);

  const TrailerScan scan = scan_events(archive);
  CHECK(scan.trailer);
  CHECK(scan.after == 0);

  // A blob written after the manifest is a file no count accounts for.
  const size_t files = count_files(archive / "blobs", ".blob").first +
                       count_files(archive / "code_objects", ".hsaco").first;
  CHECK(manifest_value(archive, "blob_count") == std::to_string(files));
}

// Before, the failed write was only a warning and the archive got its trailer.
HRR_TEST_CASE(Unit_HRR_FailedBlobWrite_LeavesCaptureIncomplete) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_failed_blob.hrr");
  capture_workload(cap.path, "Unit_HRR_FailedBlobWrite_Direct", false);

  const fs::path archive = hrr_single_process_archive(cap.path);
  {
    INFO("the blob write was meant to fail, but the blob is on disk");
    REQUIRE(count_files(archive / "blobs", ".blob", kBigBlob).second == 0);
  }

  hrr::Archive ar;
  REQUIRE(hrr::load_archive(archive.string(), ar));
  CHECK_FALSE(ar.complete);
  CHECK(manifest_value(archive, "complete") == "false");
}

// Before, fclose()'s result was ignored and the truncated blob was published.
HRR_TEST_CASE(Unit_HRR_FailedBlobClose_LeavesCaptureIncomplete) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_failed_blob_close.hrr");
  const std::string out = capture_workload(cap.path, "Unit_HRR_FailedBlobClose_Direct", false);
  INFO("Workload output:\n" << out);

  const size_t at = out.find(HRR_CLOSE_MARKER " blksize=");
  REQUIRE(at != std::string::npos);
  long long blksize = 0;
  int err = -1;
  REQUIRE(sscanf(out.c_str() + at, HRR_CLOSE_MARKER " blksize=%lld err=%d", &blksize, &err) == 2);
  REQUIRE(err == hipSuccess);
  {
    INFO("this block size makes fwrite() fail, not fclose()");
    REQUIRE((blksize <= 0 || static_cast<unsigned long long>(blksize) > kSmallBlob));
  }

  const fs::path archive = hrr_single_process_archive(cap.path);
  {
    INFO("a blob cut short at the file size limit was published");
    CHECK(count_files(archive / "blobs", ".blob", kSmallFileSizeLimit).second == 0);
  }
  {
    INFO("the blob write was meant to fail, but the blob is on disk");
    REQUIRE(count_files(archive / "blobs", ".blob", kSmallBlob).second == 0);
  }

  hrr::Archive ar;
  REQUIRE(hrr::load_archive(archive.string(), ar));
  CHECK_FALSE(ar.complete);
  CHECK(manifest_value(archive, "complete") == "false");
}

// Before, the shim listed its API before the cut-off dropped its event.
HRR_TEST_CASE(Unit_HRR_DispatchSwap_NoUnreplayableAfterTrailer) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_late_unreplayable.hrr");
  const std::string out =
      capture_workload(cap.path, "Unit_HRR_DispatchSwap_Unreplayable_Direct", true, true);
  INFO("Workload output:\n" << out);

  // The shim ran between the trailer and the manifest, and the call succeeded.
  const size_t at = out.find(HRR_LATE_MARKER " err=");
  REQUIRE(at != std::string::npos);
  int err = -1;
  REQUIRE(sscanf(out.c_str() + at, HRR_LATE_MARKER " err=%d", &err) == 1);
  REQUIRE(err == hipSuccess);

  const fs::path archive = hrr_single_process_archive(cap.path);
  hrr::Archive ar;
  REQUIRE(hrr::load_archive(archive.string(), ar));
  CHECK(ar.complete);
  CHECK(scan_events(archive).after == 0);
  size_t host_funcs = 0, user_objects = 0;
  for (const auto& ev : ar.events) {
    if (ev.header().event_type == HRR_API_HIPLAUNCHHOSTFUNC) ++host_funcs;
    if (ev.header().event_type == HRR_API_HIPUSEROBJECTCREATE) ++user_objects;
  }
  CHECK(host_funcs == 1);
  CHECK(user_objects == 0);

  const std::string manifest = read_text_file(archive / "manifest.json");
  INFO("manifest.json:\n" << manifest);
  CHECK(manifest.find("\"hipLaunchHostFunc\"") != std::string::npos);
  CHECK(manifest.find("\"hipUserObjectCreate\"") == std::string::npos);
}

// Before, the child inherited the parent's incomplete flag and lost its trailer.
HRR_TEST_CASE(Unit_HRR_Fork_ChildArchiveCompleteAfterParentFailure) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_fork_incomplete.hrr");
  const std::string out = capture_workload(cap.path, "Unit_HRR_Fork_Direct", true);
  INFO("Workload output:\n" << out);

  const size_t at = out.find(HRR_FORK_MARKER " ");
  REQUIRE(at != std::string::npos);
  int child = 0;
  REQUIRE(sscanf(out.c_str() + at, HRR_FORK_MARKER " %d", &child) == 1);

  const std::vector<fs::path> archives = hrr_process_archives(cap.path);
  REQUIRE(archives.size() == 2);
  const fs::path child_dir = cap.path / ("pid-" + std::to_string(child));
  const fs::path parent_dir = archives[0] == child_dir ? archives[1] : archives[0];
  REQUIRE(fs::exists(child_dir / "events.bin"));

  hrr::Archive parent;
  REQUIRE(hrr::load_archive(parent_dir.string(), parent));
  {
    INFO("the parent's blob write was meant to fail and mark it incomplete");
    REQUIRE_FALSE(parent.complete);
  }

  hrr::Archive child_ar;
  REQUIRE(hrr::load_archive(child_dir.string(), child_ar));
  CHECK(child_ar.complete);
  CHECK(manifest_value(child_dir, "complete") == "true");
}

#endif  // !_WIN32 && HRR_TEST_EXE && HRR_DISPATCH_TOOL

/** @} */
