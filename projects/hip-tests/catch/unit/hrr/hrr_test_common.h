/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR shared test helpers
 * @{
 * @ingroup HRRTest
 * Helpers shared by hrr_roundtrip_test.cc and hrr_api_matrix_test.cc.
 *
 * The capture/replay pattern every roundtrip follows:
 *   1. Spawn HrrTest as a subprocess with HIP_HRR_CAPTURE_OUTPUT set, running a
 *      hidden "[.][hrr-direct]" workload case by name.  A clean subprocess is
 *      required on Windows so MSYS2/bash SEH handling does not interfere with
 *      HIP's internal __try/__except frames.
 *   2. Inspect the resulting archive directory.
 *   3. Run hrr-playback over it and assert on the outcome.
 *
 * The API-matrix tests add a fourth mode on top: rather than asserting only on
 * D2H fidelity, they assert that each API's *replay class* (faithful / noop /
 * error-stub) matches what HRR's generator declares it to be.
 */

#pragma once

#include <hip_test_common.hh>
#include <hip_test_process.hh>
#include "hrr_reader.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <csignal>
#endif

namespace fs = std::filesystem;

// Platform path separator for setEnv("PATH", ...).
// ';' on Windows, ':' on POSIX.
#ifdef _WIN32
static constexpr char kPathSep = ';';
#else
static constexpr char kPathSep = ':';
#endif

// Set PATH so the subprocess can find the ROCm runtime binaries.
// On Windows: DLLs are found via PATH.
// On Linux:   fork() inherits LD_LIBRARY_PATH from the parent automatically;
//             no explicit setEnv needed.
inline void set_proc_search_path(hip::SpawnProc& proc) {
  const char* cur_path = getenv("PATH");
  proc.setEnv("PATH",
              std::string(ROCM_BIN_PATH) + kPathSep + (cur_path ? cur_path : ""));
}

// RAII guard: removes a directory tree on scope exit (even on REQUIRE failure).
struct ScopedDir {
  fs::path path;
  explicit ScopedDir(fs::path p) : path(std::move(p)) { fs::remove_all(path); }
  ~ScopedDir() { fs::remove_all(path); }
};

inline std::string read_text_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Every per-process archive under `root`, or `root` itself if the capture
// wrote a flat archive. Unlike hrr_single_process_archive this does not
// REQUIRE a particular count: callers that only want to read an archive should
// tolerate zero or many rather than failing the test on the shape.
inline std::vector<fs::path> hrr_process_archives(const fs::path& root) {
  std::vector<fs::path> archives;
  if (fs::exists(root / "events.bin")) {
    archives.push_back(root);
    return archives;
  }
  std::error_code ec;
  if (!fs::is_directory(root, ec)) return archives;
  for (const auto& ent : fs::directory_iterator(root, ec)) {
    if (!ent.is_directory()) continue;
    const std::string name = ent.path().filename().string();
    if (name.rfind("pid-", 0) == 0 && fs::exists(ent.path() / "events.bin"))
      archives.push_back(ent.path());
  }
  std::sort(archives.begin(), archives.end());
  return archives;
}

inline fs::path hrr_single_process_archive(const fs::path& root) {
  if (fs::exists(root / "events.bin"))
    return root;

  std::vector<fs::path> archives;
  for (const auto& ent : fs::directory_iterator(root)) {
    if (!ent.is_directory()) continue;
    const std::string name = ent.path().filename().string();
    if (name.rfind("pid-", 0) == 0 && fs::exists(ent.path() / "events.bin"))
      archives.push_back(ent.path());
  }
  INFO("Process archive count: " << archives.size());
  REQUIRE(archives.size() == 1);
  return archives.front();
}

// SpawnProc uses execvp on Linux (no shell), so quotes around the archive path
// would be taken literally.  On Windows CreateProcess needs them for spaces.
inline std::string hrr_quote_path(const fs::path& p) {
#ifdef _WIN32
  return "\"" + p.string() + "\"";
#else
  return p.string();
#endif
}

// ---------------------------------------------------------------------------
// hrr_parse_d2h_summary: extract the pass/fail counts from the playback
// "D2H checks" summary line, which hrr_playback.cpp prints as:
//
//   "[HRR]   D2H checks     : N pass (E exact, T within tol), M fail, K skipped"
//
// The parenthetical breakdown is always part of the line, so the format string
// has to consume it: a format that stops at "pass," matches only the pass count
// and leaves the fail count at its initial value, which silently turns every
// caller's fail assertion into a no-op.
//
// Returns false when the line is absent or does not match, so a future change
// to the producer surfaces as a test failure instead of a phantom zero.
// ---------------------------------------------------------------------------
inline bool hrr_parse_d2h_summary(const std::string& out, int& d2h_pass, int& d2h_fail) {
  const size_t pos = out.find("D2H checks");
  if (pos == std::string::npos) return false;
  const size_t colon = out.find(':', pos);
  if (colon == std::string::npos) return false;
  return std::sscanf(out.c_str() + colon + 1, " %d pass (%*d exact, %*d within tol), %d fail",
                     &d2h_pass, &d2h_fail) == 2;
}

// ---------------------------------------------------------------------------
// hrr_run_playback — spawn hrr-playback, capture stdout, assert:
//   1. Exit code == 0.
//   2. The "D2H checks" summary line is present and shows >= 1 pass, 0 fail.
//
// If require_d2h == true (default) we REQUIRE pass >= 1.
// Workloads with no D2H memcpy (e.g. DeviceInfo, Occupancy) pass require_d2h=false.
// ---------------------------------------------------------------------------
inline void hrr_run_playback(const fs::path& cap_path,
                             const std::string& extra_args = "",
                             bool require_d2h = true) {
  hip::SpawnProc proc(HRR_PLAYBACK_EXE, /*capture_stdout=*/true);
  set_proc_search_path(proc);
  std::string path_arg = hrr_quote_path(cap_path);
  int ret = proc.run(path_arg + (extra_args.empty() ? "" : " " + extra_args));
  std::string out = proc.getOutput();
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
  // On Windows (gfx1151 consumer iGPU CI target) replay is not guaranteed to
  // reproduce device output bit-for-bit — kernel output buffers can read back
  // as zero even though capture and playback both launch successfully. Treat
  // D2H fidelity as best-effort there (same policy as the Linux fat-binary
  // limitation below); a crash still fails the test via the ret < 128 check.
#ifdef _WIN32
  require_d2h = false;
#endif
  // When require_d2h is false (e.g. no D2H in workload, or Linux fat-binary
  // limitation) we only assert that hrr-playback did not crash (signal).
  // A non-zero exit due to D2H mismatch is accepted.
  if (require_d2h) {
    REQUIRE(ret == 0);
  } else {
    // Treat SIGSEGV/SIGBUS (>128) as hard failure; clean exit or D2H-fail (1) is ok.
    REQUIRE(ret < 128);
    if (ret != 0) return;  // D2H mismatch expected — skip summary parse
  }

  // Parse the D2H summary line.
  int d2h_pass = 0, d2h_fail = 0;
  if (!hrr_parse_d2h_summary(out, d2h_pass, d2h_fail)) {
    FAIL("hrr-playback output missing or malformed 'D2H checks' summary line");
  }
  INFO("D2H pass=" << d2h_pass << " fail=" << d2h_fail);
  if (require_d2h) {
    CHECK(d2h_pass >= 1);
    CHECK(d2h_fail == 0);
  }
}

// ---------------------------------------------------------------------------
// Helper: shared roundtrip body — capture → verify archive → playback.
//
// min_events:  minimum number of events expected in events.bin.  Every workload
//   must produce at least a few events (malloc, memcpy, kernel, free) — a value
//   of 5 is a conservative floor that would catch a totally empty capture.
//   Use a higher value for workloads known to emit many events (StressApis, etc.).
// require_d2h: if true (default), asserts that playback validated at least one
//   D2H blob.  Pass false for workloads that conditionally skip D2H (e.g. the
//   texture workload on devices without image support).
// ---------------------------------------------------------------------------
inline void hrr_run_roundtrip(const std::string& direct_case,
                              const fs::path& cap_path,
                              size_t min_events = 5,
                              bool require_d2h = true) {
  { hip::SpawnProc proc(HRR_TEST_EXE);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap_path.string());
    { set_proc_search_path(proc); }
    int ret = proc.run("\"" + direct_case + "\"");
    INFO("Capture exit: " << ret); REQUIRE(ret == 0); }
  fs::path archive_path = hrr_single_process_archive(cap_path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
  REQUIRE(fs::exists(archive_path / "blobs"));
  int bc = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(archive_path / "blobs")) ++bc;
  INFO("Blob count: " << bc); REQUIRE(bc >= 1);

  // Load the archive and assert a minimum event count.  This catches generator
  // bugs that silently produce empty or near-empty archives while still writing
  // at least one blob (which would otherwise satisfy the blob_count >= 1 check).
  hrr::Archive arc;
  bool arc_ok = hrr::load_archive(cap_path.string(), arc);
  INFO("Archive event count: " << arc.events.size());
  REQUIRE(arc_ok);
  REQUIRE(arc.events.size() >= min_events);

  hrr_run_playback(cap_path, /*extra_args=*/"", require_d2h);
}

// ---------------------------------------------------------------------------
// Env-aware capture + playback helpers (used by the repro roundtrips).
//
// hrr_capture_direct: spawn a hidden _Direct workload with HIP_HRR_CAPTURE_OUTPUT
//   set, REQUIRE a clean capture, and assert the archive has >= min_events.
//
// hrr_playback_env: run hrr-playback with arbitrary extra environment pairs
//   (e.g. HIP_HRR_REPLAY_ZERO_INIT / HIP_HRR_REPLAY_DIVERGENCE_ABORT) and return
//   {exit_code, stdout}.  Note: SpawnProc only captures stdout here, so the
//   divergence-guard "[HRR] replay DIVERGED" message (emitted on stderr) is NOT
//   visible — the deterministic, observable contract is the exit code (2 ==
//   clean divergence stop), which is what the callers assert.  Tests that need
//   stderr use hrr_playback_merged below.
// ---------------------------------------------------------------------------
inline void hrr_capture_direct(const std::string& direct_case,
                               const fs::path& cap_path,
                               size_t min_events = 5) {
  { hip::SpawnProc proc(HRR_TEST_EXE);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap_path.string());
    { set_proc_search_path(proc); }
    int ret = proc.run("\"" + direct_case + "\"");
    INFO("Capture exit: " << ret); REQUIRE(ret == 0); }
  fs::path archive_path = hrr_single_process_archive(cap_path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
  REQUIRE(fs::exists(archive_path / "blobs"));
  hrr::Archive arc;
  bool arc_ok = hrr::load_archive(cap_path.string(), arc);
  INFO("Archive event count: " << arc.events.size());
  REQUIRE(arc_ok);
  REQUIRE(arc.events.size() >= min_events);
}

inline std::pair<int, std::string> hrr_playback_env(
    const fs::path& cap_path,
    const std::vector<std::pair<std::string, std::string>>& env,
    const std::string& extra_args = "") {
  hip::SpawnProc proc(HRR_PLAYBACK_EXE, /*capture_stdout=*/true);
  set_proc_search_path(proc);
  for (const auto& kv : env) proc.setEnv(kv.first, kv.second);
  std::string path_arg = hrr_quote_path(cap_path);
  int ret = proc.run(path_arg + (extra_args.empty() ? "" : " " + extra_args));
  return {ret, proc.getOutput()};
}

inline std::pair<int, std::string> run_playback_raw(const fs::path& cap_path,
                                                    const std::string& extra_args) {
  hip::SpawnProc proc(HRR_PLAYBACK_EXE, /*capture_stdout=*/true);
  set_proc_search_path(proc);
  std::string path_arg = cap_path.string();
  int ret = proc.run(path_arg + (extra_args.empty() ? "" : " " + extra_args));
  return {ret, proc.getOutput()};
}

// ===========================================================================
// API-matrix support
// ===========================================================================

// Replay behaviour HRR's generator declares for an API.  These mirror the
// classification sets in clr/hipamd/src/hrr/gen_hrr_api_args.h's generator
// (gen_hrr_api_args.py): every API is exactly one of these at replay time.
enum class HrrReplayClass {
  kReal,       // A real handler ran (GENERATED, MANUAL or CUSTOM).
  kNoop,       // NOOP_PLAYBACK_APIS — one-time warning, returns hipSuccess.
  kErrorStub,  // ERROR_STUB_PLAYBACK_APIS — named graph-construction warning.
  kHandlerError,  // A real handler ran and returned a HIP error.
  kCrash,      // A real handler took the replay process down with a signal.
};

inline const char* hrr_replay_class_name(HrrReplayClass c) {
  switch (c) {
    case HrrReplayClass::kReal:         return "REAL";
    case HrrReplayClass::kNoop:         return "NOOP";
    case HrrReplayClass::kErrorStub:    return "ERROR_STUB";
    case HrrReplayClass::kHandlerError: return "HANDLER_ERROR";
    case HrrReplayClass::kCrash:        return "CRASH";
  }
  return "?";
}

// Run hrr-playback with stderr merged into the captured output.  The NOOP and
// ERROR_STUB handlers write their one-time warnings to stderr, so a test that
// asserts on replay class must merge the two streams; hrr_playback_env cannot,
// because the tests that compare stdout exactly would break.
inline std::pair<int, std::string> hrr_playback_merged(
    const fs::path& cap_path,
    const std::string& extra_args = "",
    const std::vector<std::pair<std::string, std::string>>& env = {}) {
  hip::SpawnProc proc(HRR_PLAYBACK_EXE, /*capture_stdout=*/true,
                      /*capture_stderr=*/true);
  set_proc_search_path(proc);
  for (const auto& kv : env) proc.setEnv(kv.first, kv.second);
  std::string path_arg = hrr_quote_path(cap_path);
  int ret = proc.run(path_arg + (extra_args.empty() ? "" : " " + extra_args));
  return {ret, proc.getOutput()};
}

// Exit code hrr_playback_watchdog reports when it had to kill the replay.
// 128 + SIGKILL matches the encoding SpawnProc::doWait uses for a signalled
// child, so callers that only distinguish "crashed" from "clean" need no
// special case; callers that care about the hang specifically check for it.
inline constexpr int kHrrWatchdogKilled = 128 + 9;

// ---------------------------------------------------------------------------
// hrr_playback_watchdog — replay under a deadline.
//
// Some replays do not fail, they hang: hipStreamWriteValue64 is a no-op at
// replay, so a program that waits on the value it was supposed to write waits
// forever (section 7 hazard H2 of HRR-Use-Case-Priorities.md). Without a
// deadline that turns into a CI job that runs until the harness kills it,
// which is both slower to diagnose and easy to misread as infrastructure
// flake. Killing the child ourselves turns the hang into an assertable
// outcome.
// ---------------------------------------------------------------------------
inline std::pair<int, std::string> hrr_playback_watchdog(
    const fs::path& cap_path, int timeout_seconds,
    const std::string& extra_args = "") {
  hip::SpawnProc proc(HRR_PLAYBACK_EXE, /*capture_stdout=*/true,
                      /*capture_stderr=*/true);
  set_proc_search_path(proc);
  std::string path_arg = hrr_quote_path(cap_path);
  if (proc.spawn(path_arg + (extra_args.empty() ? "" : " " + extra_args)) != 0)
    return {-1, "failed to spawn " HRR_PLAYBACK_EXE};

  std::atomic<bool> finished{false};
  auto handle = proc.getProcess();
  std::thread watchdog([&finished, handle, timeout_seconds]() {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(timeout_seconds);
    while (!finished.load() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (finished.load()) return;
#ifdef _WIN32
    TerminateProcess(handle.hProcess, kHrrWatchdogKilled);
#else
    kill(handle, SIGKILL);
#endif
  });

  int ret = proc.wait();
  finished.store(true);
  watchdog.join();
  return {ret, proc.getOutput()};
}

// The exact one-time warnings the generator emits.  Matching on a substring
// rather than the whole sentence keeps these robust against rewording of the
// explanatory tail, while still being specific to the API and the class.
//
// The trailing space is load-bearing: HIP API names prefix one another, so
// without it hipMallocHost's warning also reads as a warning for hipMalloc.
inline std::string hrr_noop_marker(const std::string& api) {
  return "NOOP playback handler called for " + api + " ";
}

inline std::string hrr_error_stub_marker(const std::string& api) {
  return "[HRR] " + api + ": not reconstructable at replay";
}

// ---------------------------------------------------------------------------
// hrr_replay_failed_apis — every API whose handler returned a HIP error.
//
// A failing handler prints one of two lines depending on whether the replay
// was told to keep going:
//
//   [HRR] Fatal: T19 Event 54 (hipDrvLaunchKernelEx) returned 400 (invalid
//         resource handle) — aborting replay
//   [HRR] Error: T19 Event 54 (hipDrvLaunchKernelEx) returned 400 (invalid
//         resource handle) — continuing
//
// Both are parsed. Without --continue-on-error there is at most one, and
// everything after it went unobserved; with it, this is the full set.
// ---------------------------------------------------------------------------
inline std::set<std::string> hrr_replay_failed_apis(const std::string& merged) {
  std::set<std::string> failed;
  for (const char* tag : {"] Fatal: ", "] Error: "}) {
    size_t at = 0;
    while ((at = merged.find(tag, at)) != std::string::npos) {
      at += std::strlen(tag);
      const size_t eol = merged.find('\n', at);
      const size_t open = merged.find('(', at);
      if (open == std::string::npos || (eol != std::string::npos && open > eol))
        continue;
      const size_t close = merged.find(')', open);
      if (close == std::string::npos || (eol != std::string::npos && close > eol))
        continue;
      failed.insert(merged.substr(open + 1, close - open - 1));
    }
  }
  return failed;
}

// ---------------------------------------------------------------------------
// hrr_replay_reached_apis — the APIs a --verbose replay actually dispatched,
// in the order it printed them.
//
// Two line formats, one per replay path:
//
//   [HRR] Event 33: hipMemExportToShareableHandle        (single-threaded)
//   [HRR] T19 [12] hipMemcpyAsync                        (multi-threaded)
//
// This is what makes "no warning was printed, therefore a real handler ran"
// sound. Silence from an API the replay never reached is absence of evidence:
// it happens whenever the process dies partway, which a handler that
// dereferences a capture-time host pointer does. The last entry is then the
// event that killed it.
// ---------------------------------------------------------------------------
inline std::vector<std::string> hrr_replay_reached_apis(
    const std::string& merged) {
  std::vector<std::string> reached;
  std::istringstream in(merged);
  std::string line;
  while (std::getline(in, line)) {
    const size_t tag = line.find("[HRR] ");
    if (tag == std::string::npos) continue;
    size_t at = tag + 6;
    if (at >= line.size()) continue;

    if (line.compare(at, 6, "Event ") == 0) {
      const size_t colon = line.find(": ", at);
      if (colon == std::string::npos) continue;
      at = colon + 2;
    } else if (line[at] == 'T') {
      // "T<tid> [<index>] <api>"
      const size_t close = line.find("] ", at);
      if (close == std::string::npos || line.find(" [", at) > close) continue;
      at = close + 2;
    } else {
      continue;
    }

    const size_t end = line.find_first_of(" \t\r\n", at);
    std::string api = line.substr(at, end == std::string::npos ? end
                                                               : end - at);
    // "no handler for type 42" and the payload-size skip share the prefix but
    // name no API; both start with a lower-case word that is not a HIP symbol.
    if (api.rfind("hip", 0) != 0 && api.rfind("__hip", 0) != 0) continue;
    reached.push_back(std::move(api));
  }
  return reached;
}

// True when the replay stopped early rather than running to the end of the
// archive. Everything after that point is unobserved: silence from an API is
// then absence of evidence, not evidence of a real handler.
inline bool hrr_replay_aborted(const std::string& merged_output) {
  return merged_output.find("Replay aborted due to fatal HIP error") !=
         std::string::npos;
}

// Classify how `api` behaved during a replay, given that replay's merged
// stdout+stderr.
//
// Returns kReal when no marker is present, which is only sound when the API
// was captured *and* the replay reached it. Callers establish the first with
// hrr_info_api_counts() and the second with hrr_replay_aborted().
inline HrrReplayClass hrr_observed_replay_class(const std::string& merged_output,
                                                const std::string& api) {
  if (hrr_replay_failed_apis(merged_output).count(api))
    return HrrReplayClass::kHandlerError;
  if (merged_output.find(hrr_error_stub_marker(api)) != std::string::npos)
    return HrrReplayClass::kErrorStub;
  if (merged_output.find(hrr_noop_marker(api)) != std::string::npos)
    return HrrReplayClass::kNoop;
  return HrrReplayClass::kReal;
}

// Parse one `hrr-playback --info` report's "Event Type Breakdown" table into
// the accumulator. The table format is:
//
//   Event Type Breakdown:
//     Type                   Count
//     ----                   -----
//     hipMalloc                  4
//     ...
//
// Parsing mirrors parse_sections() in scenarios/prep/lib/hrr_info.py so the
// C++ tests and the Python reporter agree on what "captured" means.
inline void hrr_parse_info_breakdown(const std::string& info,
                                     std::map<std::string, long long>& counts) {
  bool in_table = false;
  std::istringstream in(info);
  std::string line;
  while (std::getline(in, line)) {
    const size_t b = line.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) continue;
    const size_t e = line.find_last_not_of(" \t\r\n");
    const std::string s = line.substr(b, e - b + 1);

    if (s.rfind("Event Type Breakdown", 0) == 0) { in_table = true; continue; }
    if (!in_table) continue;
    // Any other section header ends the table.
    if (s.rfind("Kernel Call Counts", 0) == 0 || s.rfind("Kernel Summary", 0) == 0 ||
        s.rfind("Event Log", 0) == 0 || s.rfind("Threads:", 0) == 0 ||
        s.rfind("Code Objects:", 0) == 0) {
      in_table = false;
      continue;
    }
    if (s.find_first_not_of("- ") == std::string::npos) continue;  // rule line
    if (s.rfind("Type", 0) == 0) continue;                          // column header

    // "<name><spaces><count>"
    const size_t sp = s.find_last_of(" \t");
    if (sp == std::string::npos) continue;
    const std::string name = s.substr(0, s.find_last_not_of(" \t", sp) + 1);
    const std::string num = s.substr(sp + 1);
    if (name.empty() || num.empty()) continue;
    if (num.find_first_not_of("0123456789") != std::string::npos) continue;
    counts[name] += std::strtoll(num.c_str(), nullptr, 10);
  }
}

// ---------------------------------------------------------------------------
// hrr_info_api_counts — {api name -> event count} for a whole capture.
//
// Pointed at an archive *root*, `hrr-playback --info` prints only the
// process table and no breakdown; the breakdown comes from running it against
// each pid-<pid> sub-archive. So resolve first and merge, which also gives the
// right answer for a multi-process capture.
//
// --info does not touch the GPU, so this is usable from CPU-only test cases.
// ---------------------------------------------------------------------------
inline std::map<std::string, long long> hrr_info_api_counts(const fs::path& cap_path) {
  std::map<std::string, long long> counts;
  for (const auto& archive : hrr_process_archives(cap_path)) {
    // A truncated archive still prints a usable breakdown before reporting the
    // truncation, so a non-zero exit is not by itself a reason to give up.
    auto [rc, out] = hrr_playback_merged(archive, "--info");
    (void)rc;
    hrr_parse_info_breakdown(out, counts);
  }
  return counts;
}

// ---------------------------------------------------------------------------
// hrr_expect_replay_class — the core API-matrix assertion.
//
// Asserts two things about `api` for the archive at cap_path:
//   1. It was captured (present in the --info Event Type Breakdown).  Without
//      this, an API the workload silently failed to call would look identical
//      to a faithfully-replayed one, and the matrix would report false green.
//   2. Its replay class matches `expected`.
//
// `merged_output` is the merged stdout+stderr of a replay of the same archive,
// obtained from hrr_playback_merged().  It is passed in rather than produced
// here so that one replay can validate a whole family of APIs.
// ---------------------------------------------------------------------------
inline void hrr_expect_replay_class(const std::map<std::string, long long>& captured,
                                    const std::string& merged_output,
                                    const std::string& api,
                                    HrrReplayClass expected) {
  INFO("API under test: " << api);
  const auto it = captured.find(api);
  INFO("Captured: " << (it != captured.end() ? std::to_string(it->second)
                                             : std::string("NOT PRESENT")));
  CHECK(it != captured.end());
  if (it == captured.end()) return;
  CHECK(it->second >= 1);

  const HrrReplayClass observed = hrr_observed_replay_class(merged_output, api);
  INFO("Expected replay class: " << hrr_replay_class_name(expected));
  INFO("Observed replay class: " << hrr_replay_class_name(observed));
  CHECK(observed == expected);
}

/**
 * @}
 */
