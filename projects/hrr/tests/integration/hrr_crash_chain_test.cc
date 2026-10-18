/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR crash handler chaining
 * @{
 * @ingroup HRRTest
 * What CLR's crash handler does with a handler the application installed first:
 *
 *   Unit_HRR_CaptureCrashChainsEarlierHandler:
 *     The workload installs a SIGSEGV handler, records a few copies and raises
 *     SIGSEGV twice. Each time the capture must flush its archive and then the
 *     earlier handler must run, which returns. The handler sees events.bin at
 *     its final size, so the flush came first, and the second signal reaches
 *     both again, so CLR's handler is still installed after the first.
 *
 *   Unit_HRR_CaptureCrashChainsWithAsyncLog:
 *     The same with CLR's asynchronous logging on, which registers its own
 *     crash callback before capture starts. The capture's callback must still
 *     run.
 *
 * Linux only, like the signal path it covers.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"
#include "hrr_reader.h"
#include "hrr/hrr_api_args.h"

#if !defined(_WIN32)

#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kCrashChainCopies = 3;
constexpr const char* kCrashChainMarker = "hrr-crash-chain";

// Set before the signals, read by the handler: no allocation in it.
char g_events_path[4096];
volatile sig_atomic_t g_earlier_calls = 0;
volatile off_t g_events_size_seen[2] = {-1, -1};

void earlier_segv_handler(int) {
  struct stat st;
  const int call = g_earlier_calls;
  if (call < 2) g_events_size_seen[call] = stat(g_events_path, &st) == 0 ? st.st_size : -1;
  g_earlier_calls = call + 1;
}

struct ChainRun {
  int ret;
  std::string out;
};

ChainRun run_crash_chain(const fs::path& cap, bool async_log) {
  hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true, /*capture_stderr=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.string());
  if (async_log) {
    proc.setEnv("AMD_LOG_LEVEL", "1");
    proc.setEnv("AMD_LOG_ASYNC", "1");
  }
  set_proc_search_path(proc);
  const int ret = proc.runWithTimeout("\"Unit_HRR_CaptureCrashChain_Direct\"", 120);
  return {ret, proc.getOutput()};
}

void check_crash_chain(const fs::path& cap, bool async_log) {
  const ChainRun run = run_crash_chain(cap, async_log);
  INFO("Workload exit code: " << run.ret << "\nWorkload output:\n" << run.out);
  // Before, CLR's handler reset itself and killed the process: 128 + SIGSEGV.
  REQUIRE(run.ret == 0);

  const size_t at = run.out.find(kCrashChainMarker);
  REQUIRE(at != std::string::npos);
  int calls = -1;
  long long first = -1, second = -1, final_size = -1;
  REQUIRE(sscanf(run.out.c_str() + at, "hrr-crash-chain calls=%d first=%lld second=%lld size=%lld",
                 &calls, &first, &second, &final_size) == 4);
  CHECK(calls == 2);
  // The capture flushed before the earlier handler looked at the file.
  CHECK(first == final_size);
  CHECK(second == final_size);

  const fs::path archive = hrr_single_process_archive(cap);
  hrr::Archive ar;
  REQUIRE(hrr::load_archive(archive.string(), ar));
  CHECK_FALSE(ar.complete);
  size_t copies = 0;
  for (const auto& ev : ar.events)
    if (ev.header().event_type == HRR_API_HIPMEMCPY) ++copies;
  // The workload ends with _exit, so only the crash flush can have written them.
  CHECK(copies == static_cast<size_t>(kCrashChainCopies));
  const std::string manifest = read_text_file(archive / "manifest.json");
  INFO("Process manifest:\n" << manifest);
  CHECK(manifest.find("\"complete\": false") != std::string::npos);
}

}  // namespace

// Installs its SIGSEGV handler before HIP starts, so the capture's crash
// handler finds it already there. Ends with _exit: no atexit flush.
TEST_CASE("Unit_HRR_CaptureCrashChain_Direct", "[.][hrr-direct]") {
  // Run on its own, outside the driver, the signals would only test the
  // handler, and there would be no archive to look at.
  const char* root = std::getenv("HIP_HRR_CAPTURE_OUTPUT");
  if (root == nullptr) HRR_SKIP("not running under capture");
  snprintf(g_events_path, sizeof(g_events_path), "%s/pid-%d/events.bin", root,
           static_cast<int>(getpid()));

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sigemptyset(&sa.sa_mask);
  sa.sa_handler = earlier_segv_handler;
  REQUIRE(sigaction(SIGSEGV, &sa, nullptr) == 0);

  HRR_HIP_CHECK(hipSetDevice(0));
  std::vector<int> h(256, 7);
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, h.size() * sizeof(int)));
  for (int i = 0; i < kCrashChainCopies; ++i)
    HRR_HIP_CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(int), hipMemcpyHostToDevice));
  struct stat st;
  REQUIRE(stat(g_events_path, &st) == 0);

  raise(SIGSEGV);
  raise(SIGSEGV);

  REQUIRE(stat(g_events_path, &st) == 0);
  printf("%s calls=%d first=%lld second=%lld size=%lld\n", kCrashChainMarker,
         static_cast<int>(g_earlier_calls), static_cast<long long>(g_events_size_seen[0]),
         static_cast<long long>(g_events_size_seen[1]), static_cast<long long>(st.st_size));
  fflush(stdout);
  _exit(0);
}

HRR_TEST_CASE(Unit_HRR_CaptureCrashChainsEarlierHandler) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_crash_chain"};
  check_crash_chain(cap.path, /*async_log=*/false);
}

HRR_TEST_CASE(Unit_HRR_CaptureCrashChainsWithAsyncLog) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_crash_chain_async_log"};
  check_crash_chain(cap.path, /*async_log=*/true);
}

#endif  // !_WIN32

/** @} */
