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
 *     The workload installs a SIGSEGV handler, records a few copies, raises
 *     SIGSEGV, records one more copy and raises it again. Each time the capture
 *     must flush its archive and then the earlier handler must run, which
 *     returns. The second time the handler sees events.bin grown to its final
 *     size, so CLR's handler is still installed and flushes again. The handler
 *     runs with its own sa_mask and SA_NODEFER, and CLR's handler keeps its
 *     SA_RESTART and, as the earlier handler has none, no SA_ONSTACK.
 *
 *   Unit_HRR_CaptureCrashChainsWithAsyncLog:
 *     The same with CLR's asynchronous logging on, which registers its own
 *     crash callback before capture starts. The capture's callback must still
 *     run, and so must the logger's: the workload holds stderr's lock from
 *     before its last HIP call, so the logger thread cannot write that call's
 *     log line, and the line can only reach stderr through the crash flush,
 *     after a marker the workload writes just before the signal.
 *
 *   Unit_HRR_CaptureCrashChainsAsyncLogRecovers:
 *     With asynchronous logging on, the logger thread is blocked writing a log
 *     line when a SIGSEGV arrives that the earlier handler recovers from. The
 *     crash flush must leave that line and the ones after it to the logger, so
 *     that after the recovery the logger goes on and the process exits instead
 *     of hanging.
 *
 *   Unit_HRR_CaptureCrashChainsKeepErrno:
 *     The capture's crash callback cannot write its manifest and changes errno.
 *     The earlier handler, and the code that resumes after it, still see the
 *     errno from before the signal.
 *
 *   Unit_HRR_CaptureCrashChainsResetHandOnce:
 *     The earlier handler is SA_RESETHAND and SA_ONSTACK. The first SIGSEGV
 *     runs it, the second kills the process, as it would without CLR's
 *     handler. CLR's handler keeps the SA_ONSTACK.
 *
 *   Unit_HRR_CaptureCrashChainsFaultKeepsThread: no earlier handler; a second
 *     thread faults under ptrace. The capture flushes, then the fatal SIGSEGV
 *     is the kernel's, on that thread and address, not one sent to the process.
 *   Unit_HRR_CaptureCrashChainsKeepIgnore: an ignored, raised SIGSEGV is ignored.
 *   Unit_HRR_CaptureCrashChainsRecoverWhileRecording: threads record while the
 *     earlier handler recovers 200 SIGSEGVs; the archive ends complete.
 *
 * Linux only, like the signal path it covers.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"
#include "hrr_reader.h"
#include "hrr/hrr_api_args.h"

#if !defined(_WIN32)

#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kCrashChainCopies = 3;
constexpr const char* kCrashChainMarker = "hrr-crash-chain";
constexpr const char* kCrashResetMarker = "hrr-crash-reset";
constexpr const char* kCrashSignalMarker = "hrr-last-signal";  // Not a prefix of the others
// Called once, right before the last signal, so only that call logs it
constexpr const char* kLastHipCall = "hipDeviceSynchronize";

// Set before the signals, read by the handler: no allocation in it.
char g_events_path[4096];
volatile sig_atomic_t g_earlier_calls = 0;
volatile off_t g_events_size_seen[2] = {-1, -1};
volatile sig_atomic_t g_usr1_blocked = -1;
volatile sig_atomic_t g_segv_blocked = -1;

void earlier_segv_handler(int) {
  struct stat st;
  const int call = g_earlier_calls;
  if (call < 2) g_events_size_seen[call] = stat(g_events_path, &st) == 0 ? st.st_size : -1;
  if (call == 0) {
    sigset_t blocked;
    pthread_sigmask(SIG_BLOCK, nullptr, &blocked);
    g_usr1_blocked = sigismember(&blocked, SIGUSR1);
    g_segv_blocked = sigismember(&blocked, SIGSEGV);
  }
  g_earlier_calls = call + 1;
}

volatile sig_atomic_t g_reset_calls = 0;

void reset_segv_handler(int) { g_reset_calls = g_reset_calls + 1; }

constexpr const char* kLogRecoveryMarker = "hrr-crash-log-recovery";
constexpr const char* kErrnoMarker = "hrr-crash-errno";
volatile sig_atomic_t g_errno_seen = -1;

void errno_segv_handler(int) { g_errno_seen = errno; }

constexpr const char* kIgnoreMarker = "hrr-crash-ignore";
constexpr const char* kRecordingMarker = "hrr-crash-recording";
constexpr int kRecoveredSignals = 200;
constexpr int kRecorders = 4;

struct SegvDelivery {
  pid_t tid;
  int code;
  void* addr;
};

// The workload's wait status, or -1 when ptrace is not allowed here, and every
// SIGSEGV delivered to any of its threads.
struct TracedRun {
  int status = -1;
  std::vector<SegvDelivery> segv;
};

TracedRun run_traced(const fs::path& cap, const char* workload) {
  TracedRun run;
  const std::string exe = hrr_self_exe();
  const std::string filter = std::string("\"") + workload + "\"";
  const pid_t pid = fork();
  if (pid == 0) {
    setenv("HIP_HRR_CAPTURE_OUTPUT", cap.c_str(), 1);
    if (ptrace(PTRACE_TRACEME, 0, nullptr, nullptr) != 0) _exit(126);
    raise(SIGSTOP);
    execl(exe.c_str(), exe.c_str(), filter.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }
  int st = 0;
  if (pid < 0 || waitpid(pid, &st, 0) != pid || !WIFSTOPPED(st)) return run;
  const long options = PTRACE_O_TRACECLONE | PTRACE_O_TRACEEXEC | PTRACE_O_EXITKILL;
  ptrace(PTRACE_SETOPTIONS, pid, nullptr, reinterpret_cast<void*>(options));
  ptrace(PTRACE_CONT, pid, nullptr, nullptr);
  for (;;) {
    const pid_t tid = waitpid(-1, &st, __WALL);
    if (tid < 0) break;  // No traced thread left
    if (!WIFSTOPPED(st)) {
      if (tid == pid) run.status = st;
      continue;
    }
    int sig = WSTOPSIG(st);
    // ptrace events, a new thread's first stop and the exec trap are not the
    // workload's signals
    if ((st >> 16) != 0 || sig == SIGSTOP || sig == SIGTRAP) {
      sig = 0;
    } else if (sig == SIGSEGV) {
      siginfo_t si;
      if (ptrace(PTRACE_GETSIGINFO, tid, nullptr, &si) == 0)
        run.segv.push_back({tid, si.si_code, si.si_addr});
    }
    ptrace(PTRACE_CONT, tid, nullptr, reinterpret_cast<void*>(static_cast<long>(sig)));
  }
  return run;
}

struct ChainRun {
  int ret;
  std::string out;
};

ChainRun run_crash_chain(const fs::path& cap, bool async_log, const char* workload) {
  hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true, /*capture_stderr=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.string());
  if (async_log) {
    proc.setEnv("AMD_LOG_LEVEL", "3");  // LOG_INFO: API calls are logged
    proc.setEnv("AMD_LOG_ASYNC", "1");
  }
  set_proc_search_path(proc);
  const int ret = proc.runWithTimeout(std::string("\"") + workload + "\"", 120);
  return {ret, proc.getOutput()};
}

void check_crash_chain(const fs::path& cap, bool async_log) {
  const ChainRun run = run_crash_chain(cap, async_log, "Unit_HRR_CaptureCrashChain_Direct");
  INFO("Workload exit code: " << run.ret << "\nWorkload output:\n" << run.out);
  // Before, CLR's handler reset itself and killed the process: 128 + SIGSEGV.
  REQUIRE(run.ret == 0);

  const size_t at = run.out.find(kCrashChainMarker);
  REQUIRE(at != std::string::npos);
  int calls = -1, chained = -1, restart = -1, onstack = -1, usr1 = -1, segv = -1;
  long long first = -1, second = -1, final_size = -1;
  REQUIRE(sscanf(run.out.c_str() + at,
                 "hrr-crash-chain calls=%d first=%lld second=%lld size=%lld chained=%d "
                 "restart=%d onstack=%d usr1=%d segv=%d",
                 &calls, &first, &second, &final_size, &chained, &restart, &onstack, &usr1,
                 &segv) == 9);
  CHECK(calls == 2);
  // The second signal flushed the copy recorded after the first, before the
  // earlier handler looked at the file.
  CHECK(first < second);
  CHECK(second == final_size);
  // CLR's handler is in front, keeps the earlier handler's SA_RESTART, does not
  // add an SA_ONSTACK it lacks, and runs it with its sa_mask (SIGUSR1) and
  // SA_NODEFER (SIGSEGV not blocked).
  CHECK(chained == 1);
  CHECK(restart == 1);
  CHECK(onstack == 0);
  CHECK(usr1 == 1);
  CHECK(segv == 0);

  if (async_log) {
    // The last HIP call's log line comes after the marker, so the logger's
    // crash callback ran too. Synchronous logging would print it before the
    // marker, and without the callback it is never printed.
    const size_t signal_at = run.out.find(kCrashSignalMarker);
    REQUIRE(signal_at != std::string::npos);
    const size_t logged_at = run.out.rfind(kLastHipCall);
    CHECK(logged_at != std::string::npos);
    CHECK(logged_at > signal_at);
  }

  const fs::path archive = hrr_single_process_archive(cap);
  hrr::Archive ar;
  REQUIRE(hrr::load_archive(archive.string(), ar));
  CHECK_FALSE(ar.complete);
  size_t copies = 0;
  for (const auto& ev : ar.events)
    if (ev.header().event_type == HRR_API_HIPMEMCPY) ++copies;
  // The workload ends with _exit, so only the crash flush can have written them.
  CHECK(copies == static_cast<size_t>(kCrashChainCopies + 1));
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
  sigaddset(&sa.sa_mask, SIGUSR1);
  sa.sa_handler = earlier_segv_handler;
  sa.sa_flags = SA_RESTART | SA_NODEFER;
  REQUIRE(sigaction(SIGSEGV, &sa, nullptr) == 0);

  HRR_HIP_CHECK(hipSetDevice(0));
  std::vector<int> h(256, 7);
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, h.size() * sizeof(int)));
  for (int i = 0; i < kCrashChainCopies; ++i)
    HRR_HIP_CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(int), hipMemcpyHostToDevice));
  struct stat st;
  REQUIRE(stat(g_events_path, &st) == 0);
  struct sigaction cur;
  REQUIRE(sigaction(SIGSEGV, nullptr, &cur) == 0);

  raise(SIGSEGV);
  HRR_HIP_CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(int), hipMemcpyHostToDevice));
  // From here to _exit, a logger thread blocks on stderr; this thread can
  // still write it, and so can the crash callbacks run on it. The logger polls
  // every millisecond: it writes the lines logged so far, then blocks with no
  // line in hand, so the next one is left to the crash flush.
  usleep(200 * 1000);
  flockfile(stderr);
  usleep(200 * 1000);
  HRR_HIP_CHECK(hipDeviceSynchronize());
  fprintf(stderr, "%s\n", kCrashSignalMarker);
  raise(SIGSEGV);

  REQUIRE(stat(g_events_path, &st) == 0);
  printf("%s calls=%d first=%lld second=%lld size=%lld chained=%d restart=%d onstack=%d "
         "usr1=%d segv=%d\n",
         kCrashChainMarker, static_cast<int>(g_earlier_calls),
         static_cast<long long>(g_events_size_seen[0]),
         static_cast<long long>(g_events_size_seen[1]), static_cast<long long>(st.st_size),
         cur.sa_handler != earlier_segv_handler ? 1 : 0, (cur.sa_flags & SA_RESTART) ? 1 : 0,
         (cur.sa_flags & SA_ONSTACK) ? 1 : 0, static_cast<int>(g_usr1_blocked),
         static_cast<int>(g_segv_blocked));
  fflush(stdout);
  _exit(0);
}

// Installs a one-shot SIGSEGV handler before HIP starts. The second signal
// must kill the process; if it does not, the workload says so and exits 0.
TEST_CASE("Unit_HRR_CaptureCrashChainResetHand_Direct", "[.][hrr-direct]") {
  if (std::getenv("HIP_HRR_CAPTURE_OUTPUT") == nullptr) HRR_SKIP("not running under capture");

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sigemptyset(&sa.sa_mask);
  sa.sa_handler = reset_segv_handler;
  sa.sa_flags = SA_RESETHAND | SA_ONSTACK;
  REQUIRE(sigaction(SIGSEGV, &sa, nullptr) == 0);

  HRR_HIP_CHECK(hipSetDevice(0));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, 64));
  struct sigaction cur;
  REQUIRE(sigaction(SIGSEGV, nullptr, &cur) == 0);

  raise(SIGSEGV);
  printf("%s onstack=%d calls=%d\n", kCrashResetMarker, (cur.sa_flags & SA_ONSTACK) ? 1 : 0,
         static_cast<int>(g_reset_calls));
  fflush(stdout);
  raise(SIGSEGV);
  printf("%s survived calls=%d\n", kCrashResetMarker, static_cast<int>(g_reset_calls));
  fflush(stdout);
  _exit(0);
}

// The earlier handler recovers from a SIGSEGV raised while the logger thread is
// blocked on stderr in the middle of a log line.
TEST_CASE("Unit_HRR_CaptureCrashChainLogRecovery_Direct", "[.][hrr-direct]") {
  if (std::getenv("HIP_HRR_CAPTURE_OUTPUT") == nullptr || std::getenv("AMD_LOG_ASYNC") == nullptr)
    HRR_SKIP("not running under capture with asynchronous logging");

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sigemptyset(&sa.sa_mask);
  sa.sa_handler = reset_segv_handler;
  REQUIRE(sigaction(SIGSEGV, &sa, nullptr) == 0);

  HRR_HIP_CHECK(hipSetDevice(0));
  flockfile(stderr);
  HRR_HIP_CHECK(hipDeviceSynchronize());
  usleep(200 * 1000);  // The logger thread is now blocked writing a line
  HRR_HIP_CHECK(hipDeviceSynchronize());
  raise(SIGSEGV);
  funlockfile(stderr);
  int version = 0;
  HRR_HIP_CHECK(hipRuntimeGetVersion(&version));
  printf("%s calls=%d\n", kLogRecoveryMarker, static_cast<int>(g_reset_calls));
  fflush(stdout);
  // A normal exit flushes the log and stops the logger thread
  exit(0);
}

// HRR's crash callback finds a directory where its manifest goes.
TEST_CASE("Unit_HRR_CaptureCrashChainErrno_Direct", "[.][hrr-direct]") {
  const char* root = std::getenv("HIP_HRR_CAPTURE_OUTPUT");
  if (root == nullptr) HRR_SKIP("not running under capture");

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sigemptyset(&sa.sa_mask);
  sa.sa_handler = errno_segv_handler;
  REQUIRE(sigaction(SIGSEGV, &sa, nullptr) == 0);

  HRR_HIP_CHECK(hipSetDevice(0));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, 64));
  char manifest[4096];
  snprintf(manifest, sizeof(manifest), "%s/pid-%d/manifest.json", root,
           static_cast<int>(getpid()));
  unlink(manifest);
  REQUIRE(mkdir(manifest, 0700) == 0);

  errno = EDOM;
  raise(SIGSEGV);
  const int after = errno;
  printf("%s seen=%d after=%d\n", kErrnoMarker, static_cast<int>(g_errno_seen), after);
  fflush(stdout);
  _exit(0);
}

// No SIGSEGV handler before HIP, not even Catch2's. A second thread writes to a
// page it cannot write, after a few recorded copies.
TEST_CASE("Unit_HRR_CaptureCrashChainFault_Direct", "[.][hrr-direct]") {
  if (std::getenv("HIP_HRR_CAPTURE_OUTPUT") == nullptr) HRR_SKIP("not running under capture");

  REQUIRE(signal(SIGSEGV, SIG_DFL) != SIG_ERR);
  HRR_HIP_CHECK(hipSetDevice(0));
  std::vector<int> h(256, 7);
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, h.size() * sizeof(int)));
  for (int i = 0; i < kCrashChainCopies; ++i)
    HRR_HIP_CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(int), hipMemcpyHostToDevice));
  void* page = mmap(nullptr, 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  REQUIRE(page != MAP_FAILED);
  std::thread([page] { *static_cast<volatile int*>(page) = 1; }).join();
  _exit(0);
}

// Ignores SIGSEGV before HIP, then raises one.
TEST_CASE("Unit_HRR_CaptureCrashChainIgnored_Direct", "[.][hrr-direct]") {
  if (std::getenv("HIP_HRR_CAPTURE_OUTPUT") == nullptr) HRR_SKIP("not running under capture");

  REQUIRE(signal(SIGSEGV, SIG_IGN) != SIG_ERR);
  HRR_HIP_CHECK(hipSetDevice(0));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, 64));
  raise(SIGSEGV);
  usleep(200 * 1000);  // Time for a SIGSEGV sent to the process to arrive
  printf("%s survived\n", kIgnoreMarker);
  fflush(stdout);
  _exit(0);
}

// Threads record copies until the main thread has raised its SIGSEGVs, each of
// which the earlier handler recovers from. Ends with a normal exit.
TEST_CASE("Unit_HRR_CaptureCrashChainRecording_Direct", "[.][hrr-direct]") {
  if (std::getenv("HIP_HRR_CAPTURE_OUTPUT") == nullptr) HRR_SKIP("not running under capture");

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sigemptyset(&sa.sa_mask);
  sa.sa_handler = reset_segv_handler;
  REQUIRE(sigaction(SIGSEGV, &sa, nullptr) == 0);

  HRR_HIP_CHECK(hipSetDevice(0));
  std::atomic<bool> done{false};
  std::atomic<int> copies{0}, failed{0}, ready{0};
  std::vector<std::thread> recorders;
  for (int t = 0; t < kRecorders; ++t) {
    recorders.emplace_back([&] {
      std::vector<int> h(256, 7);
      int* d = nullptr;
      if (hipSetDevice(0) != hipSuccess || hipMalloc(&d, h.size() * sizeof(int)) != hipSuccess) {
        ++failed;
        ++ready;
        return;
      }
      bool first = true;
      do {
        if (hipMemcpy(d, h.data(), h.size() * sizeof(int), hipMemcpyHostToDevice) == hipSuccess)
          ++copies;
        else
          ++failed;
        if (first) ++ready;
        first = false;
      } while (!done.load());
    });
  }
  // A thread's first HIP calls can build CLR's blit kernels through comgr, and
  // LLVM's own SIGSEGV handler, installed over CLR's, deletes comgr's temporary
  // files: raise only once every recorder is past them.
  while (ready.load() < kRecorders) usleep(1000);
  for (int i = 0; i < kRecoveredSignals; ++i) {
    raise(SIGSEGV);
    usleep(1000);
  }
  done = true;
  for (auto& t : recorders) t.join();
  printf("%s calls=%d copies=%d failed=%d\n", kRecordingMarker,
         static_cast<int>(g_reset_calls), copies.load(), failed.load());
  fflush(stdout);
  exit(0);  // The atexit flush finalizes the archive
}

HRR_TEST_CASE(Unit_HRR_CaptureCrashChainsEarlierHandler) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_crash_chain"};
  check_crash_chain(cap.path, /*async_log=*/false);
}

HRR_TEST_CASE(Unit_HRR_CaptureCrashChainsWithAsyncLog) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_crash_chain_async_log"};
  check_crash_chain(cap.path, /*async_log=*/true);
}

HRR_TEST_CASE(Unit_HRR_CaptureCrashChainsResetHandOnce) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_crash_chain_reset_hand"};
  const ChainRun run =
      run_crash_chain(cap.path, /*async_log=*/false, "Unit_HRR_CaptureCrashChainResetHand_Direct");
  INFO("Workload exit code: " << run.ret << "\nWorkload output:\n" << run.out);
  // CLR's handler keeps the earlier handler's SA_ONSTACK
  CHECK(run.out.find(std::string(kCrashResetMarker) + " onstack=1 calls=1") != std::string::npos);
  CHECK(run.out.find("survived") == std::string::npos);
  CHECK(run.ret == 128 + SIGSEGV);
}

HRR_TEST_CASE(Unit_HRR_CaptureCrashChainsAsyncLogRecovers) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_crash_chain_log_recovery"};
  const ChainRun run =
      run_crash_chain(cap.path, /*async_log=*/true, "Unit_HRR_CaptureCrashChainLogRecovery_Direct");
  INFO("Workload exit code: " << run.ret << "\nWorkload output:\n" << run.out);
  // Before, the crash flush wrote the line the logger thread held and the ones
  // after it; the logger then waited forever for a line already written, and
  // so did the flush at exit.
  REQUIRE(run.ret == 0);
  CHECK(run.out.find(std::string(kLogRecoveryMarker) + " calls=1") != std::string::npos);
  CHECK(run.out.find("hipRuntimeGetVersion") != std::string::npos);
}

HRR_TEST_CASE(Unit_HRR_CaptureCrashChainsKeepErrno) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_crash_chain_errno"};
  const ChainRun run =
      run_crash_chain(cap.path, /*async_log=*/false, "Unit_HRR_CaptureCrashChainErrno_Direct");
  INFO("Workload exit code: " << run.ret << "\nWorkload output:\n" << run.out);
  REQUIRE(run.ret == 0);
  // Before, both saw the EISDIR from the manifest open
  CHECK(run.out.find(std::string(kErrnoMarker) + " seen=" + std::to_string(EDOM) +
                     " after=" + std::to_string(EDOM)) != std::string::npos);
}

HRR_TEST_CASE(Unit_HRR_CaptureCrashChainsFaultKeepsThread) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_crash_chain_fault"};
  const TracedRun run = run_traced(cap.path, "Unit_HRR_CaptureCrashChainFault_Direct");
  if (run.status == -1) HRR_SKIP("ptrace is not allowed here");
  INFO("Wait status: " << run.status << ", SIGSEGV deliveries: " << run.segv.size());
  REQUIRE(WIFSIGNALED(run.status));
  CHECK(WTERMSIG(run.status) == SIGSEGV);
  // The fault, into CLR's handler, then the fatal one
  REQUIRE(run.segv.size() >= 2);
  for (const auto& d : run.segv) {
    INFO("tid " << d.tid << " si_code " << d.code << " si_addr " << d.addr);
    // Before, CLR's handler sent SIGSEGV to the process with kill(): SI_USER,
    // no address, taken by whichever thread did not block it
    CHECK(d.tid == run.segv[0].tid);
    CHECK(d.code == SEGV_ACCERR);
    CHECK(d.addr == run.segv[0].addr);
  }

  const fs::path archive = hrr_single_process_archive(cap.path);
  hrr::Archive ar;
  REQUIRE(hrr::load_archive(archive.string(), ar));
  CHECK_FALSE(ar.complete);
  size_t copies = 0;
  for (const auto& ev : ar.events)
    if (ev.header().event_type == HRR_API_HIPMEMCPY) ++copies;
  CHECK(copies == static_cast<size_t>(kCrashChainCopies));
}

HRR_TEST_CASE(Unit_HRR_CaptureCrashChainsKeepIgnore) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_crash_chain_ignore"};
  const ChainRun run =
      run_crash_chain(cap.path, /*async_log=*/false, "Unit_HRR_CaptureCrashChainIgnored_Direct");
  INFO("Workload exit code: " << run.ret << "\nWorkload output:\n" << run.out);
  // Before, CLR's handler killed the process with the ignored signal
  CHECK(run.ret == 0);
  CHECK(run.out.find(std::string(kIgnoreMarker) + " survived") != std::string::npos);
}

HRR_TEST_CASE(Unit_HRR_CaptureCrashChainsRecoverWhileRecording) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_crash_chain_recording"};
  const ChainRun run =
      run_crash_chain(cap.path, /*async_log=*/false, "Unit_HRR_CaptureCrashChainRecording_Direct");
  INFO("Workload exit code: " << run.ret << "\nWorkload output:\n" << run.out);
  REQUIRE(run.ret == 0);
  const size_t at = run.out.find(kRecordingMarker);
  REQUIRE(at != std::string::npos);
  int calls = -1, recorded = -1, failed = -1;
  REQUIRE(sscanf(run.out.c_str() + at, "hrr-crash-recording calls=%d copies=%d failed=%d", &calls,
                 &recorded, &failed) == 3);
  CHECK(calls == kRecoveredSignals);
  CHECK(failed == 0);
  CHECK(recorded > 0);

  const fs::path archive = hrr_single_process_archive(cap.path);
  hrr::Archive ar;
  REQUIRE(hrr::load_archive(archive.string(), ar));
  // The crash flushes left no torn or repeated record, and the exit flush
  // marked the archive complete again
  CHECK(ar.complete);
  size_t copies = 0;
  for (const auto& ev : ar.events)
    if (ev.header().event_type == HRR_API_HIPMEMCPY) ++copies;
  CHECK(copies == static_cast<size_t>(recorded));
}

#endif  // !_WIN32

/** @} */
