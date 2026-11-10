/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR dispatch-table swap and capture shutdown
 * @{
 * @ingroup HRRTest
 * When the capture runs, what it leaves behind when it shuts down, and what it
 * refuses to record once it has.
 *
 *   Unit_HRR_DispatchSwap_NoCaptureWithProfiler:
 *     Profilers write the live dispatch table too, so with a rocprofiler tool
 *     attached at startup, or the CLR profiler started by GPU_CLR_PROFILE_OUTPUT,
 *     capture must stay off. The same workload without a profiler is recorded.
 *
 *   Unit_HRR_FailedBlobWrite_LeavesCaptureIncomplete:
 *     A blob that cannot be written leaves an event naming a missing file, so
 *     the archive must not claim to be complete.
 *
 *   Unit_HRR_FailedBlobClose_LeavesCaptureIncomplete:
 *     The same, when the blob is small enough to sit in the stdio buffer and
 *     its write fails only when the file is closed.
 *
 *   Unit_HRR_Fork_ChildArchiveCompleteAfterParentFailure:
 *     The failure above marks the parent's archive only. A child forked
 *     afterwards writes its own archive and must still get its trailer, and
 *     the parent must keep capturing after fork().
 *
 * A rocprofiler tool is attached the way a profiler is: rocprofiler-register
 * loads hrr_dispatch_tool.cc as the tool library. Linux only, like the
 * rocprofiler-register integration in libamdhip64. A runtime built without that
 * integration attaches no tool, so that part of the test skips.
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

#define HRR_FORK_MARKER "HRR_FORK_CHILD"
#define HRR_CLOSE_MARKER "HRR_BLOB_CLOSE"
#define HRR_NO_TABLE_MARKER "HRR_NO_HIP_TABLE"

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

// The leading fields of rocprofiler_register_registration_info_t.
struct RegistrationInfo {
  size_t size;
  const char* common_name;
};

// Whether libamdhip64 passed its tables to rocprofiler-register. A static build
// or one with HIP_ENABLE_ROCPROFILER_REGISTER=OFF does not. Read through dlsym
// so the tests do not need the rocprofiler-register headers.
bool hip_registered() {
  if (!dlsym(RTLD_DEFAULT, "rocprofiler_register_library_api_table")) return false;
  using Cb = int (*)(const RegistrationInfo*, void*);
  using IterateFn = int (*)(Cb, void*);
  auto iterate = reinterpret_cast<IterateFn>(
      dlsym(RTLD_DEFAULT, "rocprofiler_register_iterate_registration_info"));
  if (!iterate) return true;  // cannot tell, so expect the table
  bool hip = false;
  iterate(
      [](const RegistrationInfo* info, void* data) {
        if (info->common_name && std::strcmp(info->common_name, "hip") == 0)
          *static_cast<bool*>(data) = true;
        return 0;
      },
      &hip);
  return hip;
}

// True, after telling the driver, when no tool can get the HIP table because
// the runtime never registered it. Call after hip::init().
bool runtime_has_no_table() {
  if (tool_hip_table() || hip_registered()) return false;
  printf(HRR_NO_TABLE_MARKER "\n");
  fflush(stdout);
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
// Profiler attached
// ---------------------------------------------------------------------------

// hip::init() decides whether capture runs, which is all the drivers look at.
TEST_CASE("Unit_HRR_DispatchSwap_Profiled_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  runtime_has_no_table();
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
// fork() after a failure
// ---------------------------------------------------------------------------

pid_t g_fork_parent = 0;

void noop_host_fn(void*) {}

// The child's exit runs hip_capture_shutdown() and then this, which was
// registered before it. Leaving straight away skips the HIP runtime's own
// teardown, which a forked child cannot run.
void fork_late_hook() {
  if (getpid() != g_fork_parent) _exit(0);
}

// One more dependency than an event records, so the call is noted as unreplayable.
constexpr size_t kTooManyDeps = 17;

hipError_t add_node_with_too_many_deps(hipGraph_t graph, const hipGraphNode_t* deps) {
  hipGraphNode_t node = nullptr;
  return hipGraphAddEmptyNode(&node, graph, deps, kTooManyDeps);
}

TEST_CASE("Unit_HRR_Fork_Direct", "[.][hrr-direct]") {
  g_fork_parent = getpid();
  // Before hip::init() registers hip_capture_shutdown(), so it runs after that.
  std::atexit(fork_late_hook);
  HRR_HIP_CHECK(hipSetDevice(0));

  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kBigBlob));
  HRR_HIP_CHECK(memcpy_with_failing_blob(dev));

  // Unreplayable in the parent's archive only.
  HRR_HIP_CHECK(hipLaunchHostFunc(nullptr, noop_host_fn, nullptr));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  // Truncated in both archives: the child makes the same call.
  hipGraph_t graph = nullptr;
  HRR_HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t deps[kTooManyDeps] = {};
  for (auto& dep : deps) HRR_HIP_CHECK(hipGraphAddEmptyNode(&dep, graph, nullptr, 0));
  HRR_HIP_CHECK(add_node_with_too_many_deps(graph, deps));

  fflush(stdout);
  fflush(stderr);
  const pid_t child = fork();
  if (child == 0) {
    if (add_node_with_too_many_deps(graph, deps) != hipSuccess) _exit(3);
    std::exit(0);
  }
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
  HRR_HIP_CHECK(hipGraphDestroy(graph));
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
                             bool clr_profiler = false) {
  hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.string());
  if (with_tool) proc.setEnv("ROCPROFILER_REGISTER_LIBRARY", dispatch_tool_path());
  if (clr_profiler) proc.setEnv("GPU_CLR_PROFILE_OUTPUT", (cap / "clr_profile.json").string());
  set_proc_search_path(proc);
  const int ret = proc.runWithTimeout(std::string("\"") + direct_case + "\"", 600);
  std::string out = proc.getOutput();
  INFO(direct_case << " exit: " << ret << "\n" << out);
  REQUIRE(ret == 0);
  return out;
}

// True when the workload found no HIP table to use; the caller skips.
bool skipped_without_table(const std::string& out) {
  if (out.find(HRR_NO_TABLE_MARKER) == std::string::npos) return false;
  SUCCEED("libamdhip64 does not register with rocprofiler-register - skipping");
  return true;
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

// Files under `dir` with `extension` that are exactly `size` bytes long.
size_t count_files(const fs::path& dir, const std::string& extension, uintmax_t size) {
  size_t n = 0;
  std::error_code ec;
  for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
    if (it->is_regular_file() && it->path().extension() == extension &&
        fs::file_size(it->path()) == size)
      ++n;
  }
  return n;
}

}  // namespace

HRR_TEST_CASE(Unit_HRR_DispatchSwap_NoCaptureWithProfiler) {
  constexpr const char* kCase = "Unit_HRR_DispatchSwap_Profiled_Direct";
  {
    ScopedDir cap(hrr_temp_dir("hrr_no_profiler.hrr"));
    capture_workload(cap.path, kCase, false);
    REQUIRE(hrr_process_archives(cap.path).size() == 1);
  }
  {
    ScopedDir cap(hrr_temp_dir("hrr_clr_profiler.hrr"));
    capture_workload(cap.path, kCase, false, true);
    CHECK(hrr_process_archives(cap.path).empty());
  }
  ScopedDir cap(hrr_temp_dir("hrr_rocprofiler_tool.hrr"));
  const std::string out = capture_workload(cap.path, kCase, true);
  INFO("Workload output:\n" << out);
  if (skipped_without_table(out)) return;
  CHECK(hrr_process_archives(cap.path).empty());
}

// Before, the failed write was only a warning and the archive got its trailer.
HRR_TEST_CASE(Unit_HRR_FailedBlobWrite_LeavesCaptureIncomplete) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_failed_blob.hrr");
  capture_workload(cap.path, "Unit_HRR_FailedBlobWrite_Direct", false);

  const fs::path archive = hrr_single_process_archive(cap.path);
  {
    INFO("the blob write was meant to fail, but the blob is on disk");
    REQUIRE(count_files(archive / "blobs", ".blob", kBigBlob) == 0);
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
    CHECK(count_files(archive / "blobs", ".blob", kSmallFileSizeLimit) == 0);
  }
  {
    INFO("the blob write was meant to fail, but the blob is on disk");
    REQUIRE(count_files(archive / "blobs", ".blob", kSmallBlob) == 0);
  }

  hrr::Archive ar;
  REQUIRE(hrr::load_archive(archive.string(), ar));
  CHECK_FALSE(ar.complete);
  CHECK(manifest_value(archive, "complete") == "false");
}

// Before, the child inherited the parent's incomplete flag and lost its trailer.
HRR_TEST_CASE(Unit_HRR_Fork_ChildArchiveCompleteAfterParentFailure) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_fork_incomplete.hrr");
  const std::string out = capture_workload(cap.path, "Unit_HRR_Fork_Direct", false);
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

// Before, the child listed the parent's unreplayable APIs, and a truncation the
// parent had already warned about was not listed for the child's own event.
HRR_TEST_CASE(Unit_HRR_Fork_ChildListsOnlyItsOwnUnreplayableApis) {
  ScopedDir cap(fs::temp_directory_path() / "hrr_fork_unreplayable.hrr");
  const std::string out = capture_workload(cap.path, "Unit_HRR_Fork_Direct", false);
  INFO("Workload output:\n" << out);

  const size_t at = out.find(HRR_FORK_MARKER " ");
  REQUIRE(at != std::string::npos);
  int child = 0;
  REQUIRE(sscanf(out.c_str() + at, HRR_FORK_MARKER " %d", &child) == 1);

  const std::vector<fs::path> archives = hrr_process_archives(cap.path);
  REQUIRE(archives.size() == 2);
  const fs::path child_dir = cap.path / ("pid-" + std::to_string(child));
  const fs::path parent_dir = archives[0] == child_dir ? archives[1] : archives[0];

  const std::string parent = read_text_file(parent_dir / "manifest.json");
  {
    INFO("parent manifest.json:\n" << parent);
    REQUIRE(parent.find("\"hipLaunchHostFunc\"") != std::string::npos);
    REQUIRE(parent.find("\"hipGraphAddEmptyNode\"") != std::string::npos);
  }

  hrr::Archive child_ar;
  REQUIRE(hrr::load_archive(child_dir.string(), child_ar));
  size_t host_funcs = 0, empty_nodes = 0;
  for (const auto& ev : child_ar.events) {
    if (ev.header().event_type == HRR_API_HIPLAUNCHHOSTFUNC) ++host_funcs;
    if (ev.header().event_type == HRR_API_HIPGRAPHADDEMPTYNODE) ++empty_nodes;
  }
  REQUIRE(host_funcs == 0);
  REQUIRE(empty_nodes == 1);

  const std::string manifest = read_text_file(child_dir / "manifest.json");
  INFO("child manifest.json:\n" << manifest);
  CHECK(manifest.find("\"hipLaunchHostFunc\"") == std::string::npos);
  CHECK(manifest.find("\"hipGraphAddEmptyNode\"") != std::string::npos);
}

#endif  // !_WIN32 && HRR_TEST_EXE && HRR_DISPATCH_TOOL

/** @} */
