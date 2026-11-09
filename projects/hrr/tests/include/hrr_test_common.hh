/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR shared test helpers
 * @{
 * @ingroup HRRTest
 */

#pragma once

#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime.h>

#include "hrr_test_process.hh"
#include "hrr_reader.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Replaces the few Catch2 pieces used, so HRR tests do not depend on the hip-tests infrastructure.

#define HRR_TEST_CASE(name) TEST_CASE(#name, "[hrr]")

#define HRR_HIP_CHECK(expr)                                                     \
  do {                                                                          \
    hipError_t _hrr_err = (expr);                                              \
    INFO("HIP call failed: " #expr);                                           \
    INFO("hipError_t: " << static_cast<int>(_hrr_err));                        \
    INFO("hipGetErrorString: " << hipGetErrorString(_hrr_err));                \
    REQUIRE(_hrr_err == hipSuccess);                                           \
  } while (0)

// Catch2 records SKIP() as a skipped testcase. WARN plus return does not,
// so the case would be missing from the JUnit report.
#define HRR_SKIP(message)                                                       \
  do {                                                                          \
    std::ostringstream _hrr_skip_msg;                                          \
    _hrr_skip_msg << message;                                                  \
    SKIP(_hrr_skip_msg.str());                                                 \
  } while (0)


namespace fs = std::filesystem;

#ifdef _WIN32
static constexpr char kPathSep = ';';
#else
static constexpr char kPathSep = ':';
#endif

inline void set_proc_search_path(hrr::test::SpawnProc& proc) {
  const char* cur_path = getenv("PATH");
  proc.setEnv("PATH",
              std::string(ROCM_BIN_PATH) + kPathSep + (cur_path ? cur_path : ""));
}

// HRR_TEST_EXE is an absolute path baked in at build time. A container that bind-mounts the checkout elsewhere
// reaches it only through a symlink, and exec of a symlink on a noexec mount fails with EACCES (SpawnProc reports 127).
// /proc/self/exe is the file the parent already executed.
inline std::string hrr_self_exe() {
#if !defined(_WIN32) && defined(HRR_TEST_EXE)
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n > 0) return std::string(buf, static_cast<size_t>(n));
#endif
#if defined(HRR_TEST_EXE)
  return HRR_TEST_EXE;
#else
  return {};
#endif
}

// hrr-playback sits next to the test binary. Resolving it from the running executable
// keeps it on the same mount the parent was started from.
inline std::string hrr_playback_exe() {
#if !defined(_WIN32) && defined(HRR_PLAYBACK_EXE)
  const std::string self = hrr_self_exe();
  const std::size_t slash = self.find_last_of('/');
  if (slash != std::string::npos) {
    const std::string candidate =
        self.substr(0, slash) + "/../../playback/hrr-playback";
    char resolved[4096];
    if (::realpath(candidate.c_str(), resolved) != nullptr) return resolved;
  }
#endif
#if defined(HRR_PLAYBACK_EXE)
  return HRR_PLAYBACK_EXE;
#else
  return {};
#endif
}

struct ScopedDir {
  fs::path path;
  explicit ScopedDir(fs::path p) : path(std::move(p)) { fs::remove_all(path); }
  ~ScopedDir() { fs::remove_all(path); }
};

inline fs::path hrr_temp_dir(const std::string& name) {
#ifdef _WIN32
  const unsigned long pid = GetCurrentProcessId();
#else
  const long pid = static_cast<long>(::getpid());
#endif
  return fs::temp_directory_path() / (name + "_" + std::to_string(pid));
}

inline std::vector<uint8_t> read_file(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE(f.good());
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

inline void write_file(const fs::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  REQUIRE(f.good());
}

inline std::string read_text_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

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
    if (hrr::pid_from_archive_dir_name(name) != 0 && fs::exists(ent.path() / "events.bin"))
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
    if (hrr::pid_from_archive_dir_name(name) != 0 && fs::exists(ent.path() / "events.bin"))
      archives.push_back(ent.path());
  }
  INFO("Process archive count: " << archives.size());
  REQUIRE(archives.size() == 1);
  return archives.front();
}

// SpawnProc uses execvp on Linux (no shell), so quotes would be taken literally.
// On Windows CreateProcess needs them for spaces.
inline std::string hrr_quote_path(const fs::path& p) {
#ifdef _WIN32
  return "\"" + p.string() + "\"";
#else
  return p.string();
#endif
}

// The format must consume the parenthetical, or the fail count keeps its initial value and fail assertions become no-ops.
// Returns false when the line is absent so a producer change fails the test.
inline bool hrr_parse_d2h_summary(const std::string& out, int& d2h_pass, int& d2h_fail) {
  const size_t pos = out.find("D2H checks");
  if (pos == std::string::npos) return false;
  const size_t colon = out.find(':', pos);
  if (colon == std::string::npos) return false;
  return std::sscanf(out.c_str() + colon + 1, " %d pass (%*d exact, %*d within tol), %d fail",
                     &d2h_pass, &d2h_fail) == 2;
}

// Exit 2 is the divergence guard; exit 1 needs a summary counting a failed D2H check. A fatal HIP error never counts.
inline bool hrr_d2h_mismatch_exit(int ret, const std::string& out) {
  if (ret == 2) return true;
  int d2h_pass = 0, d2h_fail = 0;
  return ret == 1 && hrr_parse_d2h_summary(out, d2h_pass, d2h_fail) && d2h_fail > 0;
}

inline void hrr_run_playback(const fs::path& cap_path,
                             const std::string& extra_args = "",
                             bool require_d2h = true) {
  hrr::test::SpawnProc proc(hrr_playback_exe(), /*capture_stdout=*/true);
  set_proc_search_path(proc);
  std::string path_arg = hrr_quote_path(cap_path);
  int ret = proc.run(path_arg + (extra_args.empty() ? "" : " " + extra_args));
  std::string out = proc.getOutput();
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
  // On Windows (gfx1151 iGPU) replay may not reproduce device output bit-for-bit, so D2H fidelity is best-effort there.
  // A non-zero exit passes only when D2H mismatches alone explain it. A crash or fatal HIP error fails as on Linux.
#ifdef _WIN32
  if (require_d2h && ret != 0) {
    REQUIRE(hrr_d2h_mismatch_exit(ret, out));
    WARN("D2H mismatch accepted: D2H fidelity is best-effort on Windows");
    return;
  }
#endif
  if (require_d2h) {
    REQUIRE(ret == 0);
  } else {
    REQUIRE(ret < 128);
    if (ret != 0) return;  // D2H mismatch expected — skip summary parse
  }

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

#if defined(HRR_TEST_EXE)
// min_events is a floor that catches an empty capture.
// require_d2h=false is for workloads that conditionally skip D2H.
inline void hrr_run_roundtrip(const std::string& direct_case,
                              const fs::path& cap_path,
                              size_t min_events = 5,
                              bool require_d2h = true) {
  { hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap_path.string());
    { set_proc_search_path(proc); }
    int ret = proc.run("\"" + direct_case + "\"");
    INFO("Capture exit: " << ret);
    if (ret == 4) HRR_SKIP(direct_case << " skipped");
    REQUIRE(ret == 0); }
  fs::path archive_path = hrr_single_process_archive(cap_path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
  REQUIRE(fs::exists(archive_path / "blobs"));
  int bc = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(archive_path / "blobs")) ++bc;
  INFO("Blob count: " << bc); REQUIRE(bc >= 1);

  hrr::Archive arc;
  bool arc_ok = hrr::load_archive(cap_path.string(), arc);
  INFO("Archive event count: " << arc.events.size());
  REQUIRE(arc_ok);
  REQUIRE(arc.events.size() >= min_events);

  hrr_run_playback(cap_path, /*extra_args=*/"", require_d2h);
}

// hrr_playback_env captures only stdout, so the stderr divergence message is invisible.
// Callers assert the exit code (2 is a clean divergence stop); use hrr_playback_merged for stderr.
inline void hrr_capture_direct(const std::string& direct_case,
                               const fs::path& cap_path,
                               size_t min_events = 5,
                               const std::vector<std::pair<std::string, std::string>>& env = {}) {
  { hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap_path.string());
    for (const auto& kv : env) proc.setEnv(kv.first, kv.second);
    { set_proc_search_path(proc); }
    int ret = proc.run("\"" + direct_case + "\"");
    INFO("Capture exit: " << ret);
    if (ret == 4) HRR_SKIP(direct_case << " skipped");
    REQUIRE(ret == 0); }
  fs::path archive_path = hrr_single_process_archive(cap_path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
  REQUIRE(fs::exists(archive_path / "blobs"));
  hrr::Archive arc;
  bool arc_ok = hrr::load_archive(cap_path.string(), arc);
  INFO("Archive event count: " << arc.events.size());
  REQUIRE(arc_ok);
  REQUIRE(arc.events.size() >= min_events);
}
#endif  // HRR_TEST_EXE

// Every forked child records a call before it exits, starting with the first.
// The roundtrip expects one archive for each of those besides the parent's.
inline constexpr int kHrrForkWhileRecordingForks = 200;
inline constexpr int kHrrForkWhileRecordingArchives = 1 + (kHrrForkWhileRecordingForks + 1) / 2;

inline std::pair<int, std::string> hrr_playback_env(
    const fs::path& cap_path,
    const std::vector<std::pair<std::string, std::string>>& env,
    const std::string& extra_args = "") {
  hrr::test::SpawnProc proc(hrr_playback_exe(), /*capture_stdout=*/true);
  set_proc_search_path(proc);
  for (const auto& kv : env) proc.setEnv(kv.first, kv.second);
  std::string path_arg = hrr_quote_path(cap_path);
  int ret = proc.run(path_arg + (extra_args.empty() ? "" : " " + extra_args));
  return {ret, proc.getOutput()};
}

inline std::pair<int, std::string> run_playback_raw(const fs::path& cap_path,
                                                    const std::string& extra_args) {
  hrr::test::SpawnProc proc(hrr_playback_exe(), /*capture_stdout=*/true);
  set_proc_search_path(proc);
  std::string path_arg = cap_path.string();
  int ret = proc.run(path_arg + (extra_args.empty() ? "" : " " + extra_args));
  return {ret, proc.getOutput()};
}


enum class HrrReplayClass {
  kReal,       // A real handler ran (GENERATED, MANUAL or CUSTOM).
  kNoop,       // NOOP_PLAYBACK_APIS — one-time warning, returns hipSuccess.
  kErrorStub,  // ERROR_STUB_PLAYBACK_APIS — named graph-construction warning.
  kHandlerError,  // A real handler ran and returned a HIP error.
  kCrash,      // A real handler took the replay process down with a signal.
  kUnreplayable,  // UNREPLAYABLE_PLAYBACK_APIS — refused by name, with the
                  // reason, rather than attempted with recorded arguments that
                  // cannot mean anything here.
};

inline const char* hrr_replay_class_name(HrrReplayClass c) {
  switch (c) {
    case HrrReplayClass::kReal:         return "REAL";
    case HrrReplayClass::kNoop:         return "NOOP";
    case HrrReplayClass::kErrorStub:    return "ERROR_STUB";
    case HrrReplayClass::kHandlerError: return "HANDLER_ERROR";
    case HrrReplayClass::kCrash:        return "CRASH";
    case HrrReplayClass::kUnreplayable: return "UNREPLAYABLE";
  }
  return "?";
}

// A v10 kernel launch record as capture writes it, followed by the pinned host snapshot record `rec` if any, cut `cut` bytes short.
// A cut record has no attribute tail, so the event runs out inside the snapshot.
inline std::vector<uint8_t> make_launch_with_snapshot(uint64_t seq,
                                                      const std::vector<uint8_t>& rec,
                                                      size_t cut = 0,
                                                      uint16_t num_args = 0,
                                                      bool pointer_args = false) {
  std::vector<uint8_t> p(sizeof(hrr_event_header), 0);
  auto put = [&](const void* d, size_t n) {
    p.insert(p.end(), static_cast<const uint8_t*>(d), static_cast<const uint8_t*>(d) + n);
  };
  const uint64_t stream = 0, co_hash[2] = {0, 0};
  const uint16_t name_len = 1;
  const uint16_t counts[2] = {num_args, static_cast<uint16_t>(rec.empty() ? 0 : 1)};
  const uint32_t dims[7] = {1, 1, 1, 1, 1, 1, 0}, attrs[2] = {0, 0};
  put(&stream, 8); put(&name_len, 2); put("k", 1); put(co_hash, 16);
  put(dims, sizeof(dims)); put(counts, sizeof(counts));
  for (uint16_t i = 0; i < num_args; ++i) {
    const uint8_t kind = pointer_args ? 3 : 0;
    const uint16_t size = pointer_args ? 1 : 0;
    put(&kind, 1); put(&size, 2);
    if (pointer_args) {
      const uint8_t data = 0;
      const uint16_t n_ptrs = 1, offset = 0;
      put(&data, 1); put(&n_ptrs, 2); put(&offset, 2);
    }
  }
  put(rec.data(), rec.size() - cut);
  if (!cut) put(attrs, sizeof(attrs));
  auto* h = reinterpret_cast<hrr_event_header*>(p.data());
  h->event_type     = static_cast<uint16_t>(HRR_API_HIPMODULELAUNCHKERNEL);
  h->sequence_id    = seq;
  h->payload_length = static_cast<uint32_t>(p.size());
  return p;
}

// The NOOP and ERROR_STUB handlers write their one-time warnings to stderr, so a test asserting on replay class must merge the streams.
// hrr_playback_env cannot merge them, because tests that compare stdout exactly would break.
inline std::pair<int, std::string> hrr_playback_merged(
    const fs::path& cap_path,
    const std::string& extra_args = "",
    const std::vector<std::pair<std::string, std::string>>& env = {}) {
  hrr::test::SpawnProc proc(hrr_playback_exe(), /*capture_stdout=*/true,
                      /*capture_stderr=*/true);
  set_proc_search_path(proc);
  for (const auto& kv : env) proc.setEnv(kv.first, kv.second);
  std::string path_arg = hrr_quote_path(cap_path);
  int ret = proc.run(path_arg + (extra_args.empty() ? "" : " " + extra_args));
  return {ret, proc.getOutput()};
}
// 128 + SIGKILL matches SpawnProc's encoding of a signalled child,
// so callers that only tell "crashed" from "clean" need no special case.
inline constexpr int kHrrWatchdogKilled = hrr::test::SpawnProc::kKilledOnTimeout;

// A no-op hipStreamWriteValue64 would leave a waiting program hanging until the harness kills it.
// Killing the child makes the hang an assertable outcome.
inline std::pair<int, std::string> hrr_playback_watchdog(
    const fs::path& cap_path, int timeout_seconds,
    const std::string& extra_args = "") {
  hrr::test::SpawnProc proc(hrr_playback_exe(), /*capture_stdout=*/true,
                            /*capture_stderr=*/true);
  set_proc_search_path(proc);
  std::string path_arg = hrr_quote_path(cap_path);
  int ret = proc.runWithTimeout(
      path_arg + (extra_args.empty() ? "" : " " + extra_args), timeout_seconds);
  return {ret, proc.getOutput()};
}


// Matching a substring keeps these robust against rewording of the explanatory tail.
// The trailing space is load-bearing: HIP API names prefix one another,
// so without it hipMallocHost's warning also reads as one for hipMalloc.
inline std::string hrr_noop_marker(const std::string& api) {
  return "NOOP playback handler called for " + api + " ";
}

inline std::string hrr_error_stub_marker(const std::string& api) {
  return "[HRR] " + api + ": not reconstructable at replay";
}

// An unreplayable API also returns hipErrorNotSupported, so it appears in the failed-API list too.
// This marker tells a declared refusal with a reason from a handler that tried and failed.
inline std::string hrr_unreplayable_marker(const std::string& api) {
  return "[HRR] " + api + ": NOT REPLAYABLE";
}

// Parses "[HRR] Fatal: ... aborting replay" and "[HRR] Error: ... continuing" lines.
// Without --continue-on-error there is at most one; with it, the full set.
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

// Silence from an API the replay never reached is no evidence, as when a handler kills the process.
// The last entry is then the event that killed it.
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
    // "no handler for type 42" and the payload-size skip name no API;
    // both start with a lower-case word that is not a HIP symbol.
    if (api.rfind("hip", 0) != 0 && api.rfind("__hip", 0) != 0) continue;
    reached.push_back(std::move(api));
  }
  return reached;
}

// True when the replay stopped early. Events after that point are unobserved,
// so silence from an API is absence of evidence, not evidence of a real handler.
inline bool hrr_replay_aborted(const std::string& merged_output) {
  return merged_output.find("Replay aborted due to fatal HIP error") !=
         std::string::npos;
}

// Returns kReal when no marker is present, which is sound only when the API was captured
// (hrr_info_api_counts) and the replay reached it (hrr_replay_aborted).
inline HrrReplayClass hrr_observed_replay_class(const std::string& merged_output,
                                                const std::string& api) {
  // Checked before the failed-API list: an unreplayable handler also reports the event as failed,
  // and the refusal is the more specific fact.
  if (merged_output.find(hrr_unreplayable_marker(api)) != std::string::npos)
    return HrrReplayClass::kUnreplayable;
  if (hrr_replay_failed_apis(merged_output).count(api))
    return HrrReplayClass::kHandlerError;
  if (merged_output.find(hrr_error_stub_marker(api)) != std::string::npos)
    return HrrReplayClass::kErrorStub;
  if (merged_output.find(hrr_noop_marker(api)) != std::string::npos)
    return HrrReplayClass::kNoop;
  return HrrReplayClass::kReal;
}

// check_matrix.py reads the observations this feeds, so "captured" here is what its report counts.
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
    if (s.rfind("Kernel Call Counts", 0) == 0 || s.rfind("Kernel Summary", 0) == 0 ||
        s.rfind("Event Log", 0) == 0 || s.rfind("Threads:", 0) == 0 ||
        s.rfind("Code Objects:", 0) == 0) {
      in_table = false;
      continue;
    }
    if (s.find_first_not_of("- ") == std::string::npos) continue;  // rule line
    if (s.rfind("Type", 0) == 0) continue;                          // column header

    const size_t sp = s.find_last_of(" \t");
    if (sp == std::string::npos) continue;
    const std::string name = s.substr(0, s.find_last_not_of(" \t", sp) + 1);
    const std::string num = s.substr(sp + 1);
    if (name.empty() || num.empty()) continue;
    if (num.find_first_not_of("0123456789") != std::string::npos) continue;
    counts[name] += std::strtoll(num.c_str(), nullptr, 10);
  }
}

// On an archive root --info prints no breakdown, so resolve each pid-<pid> sub-archive and merge. It does not touch the GPU.
inline std::map<std::string, long long> hrr_info_api_counts(const fs::path& cap_path) {
  std::map<std::string, long long> counts;
  for (const auto& archive : hrr_process_archives(cap_path)) {
    // A truncated archive still prints a usable breakdown before reporting the truncation,
    // so a non-zero exit alone is no reason to give up.
    auto [rc, out] = hrr_playback_merged(archive, "--info");
    (void)rc;
    hrr_parse_info_breakdown(out, counts);
  }
  return counts;
}

// `api` must have been captured, else a missed call looks like a faithful replay, and its replay class must match `expected`.
// `merged_output` comes from hrr_playback_merged() so one replay can validate a family of APIs.
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
