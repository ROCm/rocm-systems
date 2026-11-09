/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR per-API playback matrix
 * @{
 * @ingroup HRRTest
 * Set HRR_MATRIX_RESULTS_DIR to have each tier write its observations as JSON
 * for projects/hrr/tools/api-matrix/check_matrix.py --results.
 */

#include "hrr_test_process.hh"

#include "hrr_test_common.hh"
#include "hrr_api_matrix_expectations.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

// A no-op hipStreamWriteValue64 would leave a workload waiting forever, so every replay runs under a deadline.
constexpr int kReplayTimeoutSeconds = 300;

HrrReplayClass expect_code_to_class(int code) {
  switch (code) {
    case kHrrExpectNoop:      return HrrReplayClass::kNoop;
    case kHrrExpectErrorStub: return HrrReplayClass::kErrorStub;
    case kHrrExpectHandlerError: return HrrReplayClass::kHandlerError;
    case kHrrExpectCrash:     return HrrReplayClass::kCrash;
    case kHrrExpectUnreplayable: return HrrReplayClass::kUnreplayable;
    default:                  return HrrReplayClass::kReal;
  }
}

// REAL, HANDLER_ERROR and CRASH all mean a real handler ran. Two workloads landing on two of them
// saw one handler behave differently on different arguments, not a different handler.
bool is_real_handler_outcome(HrrReplayClass c) {
  return c == HrrReplayClass::kReal || c == HrrReplayClass::kHandlerError ||
         c == HrrReplayClass::kCrash;
}

bool is_real_handler_failure(HrrReplayClass a, HrrReplayClass b) {
  return is_real_handler_outcome(a) && is_real_handler_outcome(b);
}

HrrReplayClass worse_of(HrrReplayClass a, HrrReplayClass b) {
  auto rank = [](HrrReplayClass c) {
    return c == HrrReplayClass::kCrash          ? 2
           : c == HrrReplayClass::kHandlerError ? 1
                                                : 0;
  };
  return rank(a) >= rank(b) ? a : b;
}

const HrrTierFloor& tier_floor(const std::string& tier) {
  for (size_t i = 0; i < kHrrTierFloorCount; ++i)
    if (tier == kHrrTierFloors[i].tier) return kHrrTierFloors[i];
  FAIL("no tier floor recorded for " + tier);
  return kHrrTierFloors[0];  // unreachable; FAIL throws
}

int visible_device_count() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) return 0;
  return count;
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c == '\n')        { out += "\\n"; }
    else                       { out += c; }
  }
  return out;
}

struct TierObservation {
  std::string tier;
  std::vector<std::string> workloads;
  std::map<std::string, long long> captured;
  std::map<std::string, HrrReplayClass> observed;
  int worst_replay_exit = 0;
};

void write_observation(const TierObservation& obs) {
  const char* dir = getenv("HRR_MATRIX_RESULTS_DIR");
  if (dir == nullptr || *dir == '\0') return;

  std::error_code ec;
  fs::create_directories(dir, ec);
  const fs::path out = fs::path(dir) / (obs.tier + ".json");
  std::ofstream f(out);
  if (!f) {
    WARN("could not write matrix observations to " << out);
    return;
  }

  f << "{\n  \"tier\": \"" << obs.tier << "\",\n";
  f << "  \"replay_exit\": " << obs.worst_replay_exit << ",\n";
  f << "  \"workloads\": [";
  for (size_t i = 0; i < obs.workloads.size(); ++i)
    f << (i ? ", " : "") << '"' << json_escape(obs.workloads[i]) << '"';
  f << "],\n";

  f << "  \"captured\": {";
  bool first = true;
  for (const auto& kv : obs.captured) {
    f << (first ? "\n    " : ",\n    ");
    f << '"' << json_escape(kv.first) << "\": " << kv.second;
    first = false;
  }
  f << (first ? "" : "\n  ") << "},\n";

  f << "  \"observed\": {";
  first = true;
  for (const auto& kv : obs.observed) {
    f << (first ? "\n    " : ",\n    ");
    f << '"' << json_escape(kv.first) << "\": \""
      << hrr_replay_class_name(kv.second) << '"';
    first = false;
  }
  f << (first ? "" : "\n  ") << "}\n}\n";
}

void observe_workload(const std::string& direct_case, TierObservation& obs) {
  ScopedDir cap{fs::temp_directory_path() /
                ("hrr_matrix_" + direct_case)};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    const int ret = proc.run("\"" + direct_case + "\"");
    // Catch2 exits 4 when the selected case skipped. A workload that gated itself off (no VMM, no cooperative launch)
    // is not a failure and contributes no coverage. Any other non-zero exit is.
    if (ret == 4)
      return;
    if (ret != 0) {
      FAIL_CHECK("workload " << direct_case << " exited " << ret);
      return;
    }
  }

  if (!fs::exists(cap.path)) {
    FAIL_CHECK("workload " << direct_case << " produced no archive");
    return;
  }

  const auto counts = hrr_info_api_counts(cap.path);
  if (counts.empty()) {
    FAIL_CHECK("workload " << direct_case << " captured an archive but --info "
                     << "reported no events for it");
    return;
  }

// --continue-on-error lets the replay survey the whole archive, since several APIs fail by design.
// --verbose names every event as dispatched, which separates "replayed silently" from "never reached".
  auto [replay_rc, merged] = hrr_playback_watchdog(
      cap.path, kReplayTimeoutSeconds, "--continue-on-error --verbose");
  INFO("Replay of " << direct_case << " exited " << replay_rc);
  // A hang is never acceptable: it is how a no-op stream value write looks to a waiter, and it would wedge CI.
  if (replay_rc == kHrrWatchdogKilled) {
    FAIL_CHECK("replay of " << direct_case << " did not finish within "
                      << kReplayTimeoutSeconds << "s (watchdog killed it)");
  }
  obs.worst_replay_exit = std::max(obs.worst_replay_exit, replay_rc);

  // --continue-on-error should make the replay reach the end of the archive. If it stopped anyway,
  // the later events never ran, and their silence is not evidence of a real handler.
  // Record nothing for them rather than a tier-wide sweep of false REALs.
  if (hrr_replay_aborted(merged)) {
    const auto failed = hrr_replay_failed_apis(merged);
    FAIL_CHECK("replay of " << direct_case << " stopped early despite "
         << "--continue-on-error"
         << (failed.empty() ? "" : ", last failing API " + *failed.rbegin())
         << "; the events after it went unobserved");
  }

  // A handler that dereferences a capture-time host pointer ends the replay with a signal and leaves no message.
  // The last event in the verbose log is the one that did not come back.
  // Everything after it stays unobserved and surfaces as not-exercised.
  const std::vector<std::string> reached = hrr_replay_reached_apis(merged);
  const std::set<std::string> reached_set(reached.begin(), reached.end());
  const bool crashed = replay_rc >= 128;
  std::string crash_api;
  if (crashed) {
    if (reached.empty()) {
      FAIL_CHECK("replay of " << direct_case << " died with exit " << replay_rc
           << " before dispatching a single event");
    }
    crash_api = reached.back();
  }

  obs.workloads.push_back(direct_case);
  for (const auto& kv : counts) {
    obs.captured[kv.first] += kv.second;

    if (!reached_set.count(kv.first)) continue;

    const HrrReplayClass klass = kv.first == crash_api
                                     ? HrrReplayClass::kCrash
                                     : hrr_observed_replay_class(merged,
                                                                 kv.first);
    const auto it = obs.observed.find(kv.first);
    if (it == obs.observed.end()) {
      obs.observed.emplace(kv.first, klass);
    } else if (it->second != klass) {
// REAL versus HANDLER_ERROR or CRASH is one handler on different arguments. Keep that failure: it is declared in api_matrix.yaml.
// Any other pairing means the API was reclassified, which this matrix exists to catch.
      if (is_real_handler_failure(it->second, klass)) {
        it->second = worse_of(it->second, klass);
      } else {
        FAIL_CHECK("replay class of " << kv.first
                   << " differs between workloads: "
                   << hrr_replay_class_name(it->second) << " vs "
                   << hrr_replay_class_name(klass));
      }
    }
  }
}

void run_tier(const std::string& tier) {
  const HrrTierFloor& floor = tier_floor(tier);

  if (floor.gpus > 1 && visible_device_count() < floor.gpus) {
    HRR_SKIP("tier requires at least two visible GPUs");
  }

  TierObservation obs;
  obs.tier = tier;
  for (const char* const* w = floor.workloads; *w != nullptr; ++w)
    observe_workload(*w, obs);

  write_observation(obs);

  REQUIRE_FALSE(obs.workloads.empty());

  int covered = 0, not_exercised = 0, skipped = 0;
  int absent_asserted = 0;
  std::vector<std::string> missing;
  for (size_t i = 0; i < kHrrApiMatrixCount; ++i) {
    const HrrApiExpectation& e = kHrrApiMatrix[i];
    if (tier != e.tier) continue;
    if (e.skip) { ++skipped; continue; }

    const auto it = obs.captured.find(e.api);
    const long long count = it == obs.captured.end() ? 0 : it->second;

    // Declared as folded away by HIP before the capture layer runs. Its absence is the assertion:
    // if it starts appearing, HIP's lowering changed.
    if (!e.expect_captured) {
      INFO("API: " << e.api << " (tier " << tier
                   << ") is declared as never captured");
      CHECK(count == 0);
      ++absent_asserted;
      continue;
    }

    if (it == obs.captured.end() || it->second < 1) {
      ++not_exercised;
      missing.emplace_back(e.api);
      continue;
    }

    const auto obs_it = obs.observed.find(e.api);
    if (obs_it == obs.observed.end()) {
      ++not_exercised;
      missing.emplace_back(std::string(e.api) +
                           " (captured; replay never reached it)");
      continue;
    }
    ++covered;

    const HrrReplayClass expected = expect_code_to_class(e.expect);
    const HrrReplayClass observed = obs_it->second;
    INFO("API: " << e.api << " (tier " << tier << ")");
    INFO("Expected replay class: " << hrr_replay_class_name(expected));
    INFO("Observed replay class: " << hrr_replay_class_name(observed));
    // A payload-loss API still runs a real handler, which is what hides the loss. check_matrix.py records the loss as XFAIL.
    INFO("Known payload loss: " << (e.payload_loss ? "yes" : "no"));
    if (e.handler_error_ok && expected == HrrReplayClass::kReal &&
        observed == HrrReplayClass::kHandlerError) {
      INFO("Declared in api_matrix.yaml as failing for this tier's arguments");
      continue;
    }
    CHECK(observed == expected);
  }

  std::sort(missing.begin(), missing.end());
  std::string missing_list;
  for (const auto& m : missing) {
    if (!missing_list.empty()) missing_list += ", ";
    missing_list += m;
  }
// Rows with expect_captured == false assert an absence, so they are subtracted from the floor and reported separately.
// min_covered is generated counting these rows; check_matrix.py must re-emit it over assertable rows before the floor can tighten.
  const int evidence_floor = floor.min_covered - absent_asserted;
  INFO("Tier " << tier << ": " << covered << " covered with evidence, "
               << absent_asserted << " absence-asserted, " << not_exercised
               << " not exercised, " << skipped << " skipped");
  INFO("Evidence floor: " << evidence_floor << " (min_covered "
               << floor.min_covered << " less " << absent_asserted
               << " absence rows)");
  INFO("Not exercised: " << missing_list);
  INFO("Workloads that contributed: " << obs.workloads.size());
  CHECK(covered >= evidence_floor);
}

}  // namespace

TEST_CASE("Unit_HRR_ApiMatrix_T0_Roundtrip", "[.][hrr][api-matrix]") {
  run_tier("T0");
}

TEST_CASE("Unit_HRR_ApiMatrix_T1_Roundtrip", "[.][hrr][api-matrix]") {
  run_tier("T1");
}

TEST_CASE("Unit_HRR_ApiMatrix_T2_Roundtrip", "[.][hrr][api-matrix]") {
  run_tier("T2");
}

TEST_CASE("Unit_HRR_ApiMatrix_T3_Roundtrip", "[.][hrr][api-matrix]") {
  run_tier("T3");
}

TEST_CASE("Unit_HRR_ApiMatrix_T4_Roundtrip", "[.][hrr][api-matrix]") {
  run_tier("T4");
}

// T5 is tagged [T5], not [hrr], so CI's "[hrr]~[direct]" skips it; run it by name or tag.
TEST_CASE("Unit_HRR_ApiMatrix_T5_Roundtrip", "[.][api-matrix][T5]") {
  run_tier("T5");
}

TEST_CASE("Unit_HRR_ApiMatrix_ManifestWellFormed", "[.][hrr][api-matrix][cpu]") {
  REQUIRE(kHrrApiMatrixCount == static_cast<size_t>(HRR_API_COUNT));
  REQUIRE(kHrrTierFloorCount >= 1);

  std::set<std::string> tiers;
  for (size_t i = 0; i < kHrrTierFloorCount; ++i) {
    REQUIRE(kHrrTierFloors[i].tier != nullptr);
    REQUIRE(kHrrTierFloors[i].workloads != nullptr);
    REQUIRE(kHrrTierFloors[i].gpus >= 1);
    REQUIRE(kHrrTierFloors[i].min_covered >= 0);
    tiers.insert(kHrrTierFloors[i].tier);
    CHECK(*kHrrTierFloors[i].workloads != nullptr);
  }

  std::set<std::string> seen;
  for (size_t i = 0; i < kHrrApiMatrixCount; ++i) {
    const HrrApiExpectation& e = kHrrApiMatrix[i];
    INFO("row " << i << ": " << (e.api ? e.api : "(null)"));
    REQUIRE(e.api != nullptr);
    REQUIRE(e.tier != nullptr);
    CHECK(e.expect >= kHrrExpectReal);
    CHECK(e.expect <= kHrrExpectUnreplayable);
    CHECK(tiers.count(e.tier) == 1);
    CHECK(seen.insert(e.api).second);
  }
  CHECK(seen.size() == kHrrApiMatrixCount);

  // The matrix must name exactly the APIs HRR can record, or a stale header would pass every check above.
  std::set<std::string> known;
  for (int i = 0; i < HRR_API_COUNT; ++i) known.insert(hrr_api_names[i]);
  CHECK(known.size() == static_cast<size_t>(HRR_API_COUNT));
  std::string absent, unknown;
  for (const auto& name : known)
    if (!seen.count(name)) absent += (absent.empty() ? "" : ", ") + name;
  for (const auto& name : seen)
    if (!known.count(name)) unknown += (unknown.empty() ? "" : ", ") + name;
  INFO("APIs missing from the matrix: " << absent);
  INFO("matrix APIs unknown to hrr_api_names: " << unknown);
  CHECK(absent.empty());
  CHECK(unknown.empty());
}

TEST_CASE("Unit_HRR_ApiMatrix_ReplayClassMarkers", "[.][hrr][api-matrix][cpu]") {
  const std::string noop_line =
      "[HRR] NOOP playback handler called for hipHostAlloc \xE2\x80\x94 this "
      "API is not replayed; results may differ from capture.\n";
  const std::string stub_line =
      "[HRR] hipGraphAddNode: not reconstructable at replay, so the call is "
      "skipped and the graph it belongs to is marked incomplete; "
      "instantiating that graph fails loudly rather than running a graph "
      "that is missing work.\n";
  const std::string clean = "[HRR]   D2H checks     : 4 pass, 0 fail\n";

  CHECK(hrr_observed_replay_class(noop_line, "hipHostAlloc") ==
        HrrReplayClass::kNoop);
  CHECK(hrr_observed_replay_class(stub_line, "hipGraphAddNode") ==
        HrrReplayClass::kErrorStub);
  CHECK(hrr_observed_replay_class(clean, "hipMalloc") == HrrReplayClass::kReal);

  CHECK(hrr_observed_replay_class(noop_line, "hipHostMalloc") ==
        HrrReplayClass::kReal);
  CHECK(hrr_observed_replay_class(stub_line, "hipGraphAddMemsetNode") ==
        HrrReplayClass::kReal);

// HIP API names prefix one another: an unbounded marker would read hipMallocHost's warning as one for hipMalloc.
  const std::string prefix_line =
      "[HRR] NOOP playback handler called for hipMallocHost \xE2\x80\x94 this "
      "API is not replayed; results may differ from capture.\n";
  CHECK(hrr_observed_replay_class(prefix_line, "hipMallocHost") ==
        HrrReplayClass::kNoop);
  CHECK(hrr_observed_replay_class(prefix_line, "hipMalloc") ==
        HrrReplayClass::kReal);

  const std::string both = noop_line + stub_line;
  CHECK(hrr_observed_replay_class(both, "hipHostAlloc") ==
        HrrReplayClass::kNoop);
  CHECK(hrr_observed_replay_class(both, "hipGraphAddNode") ==
        HrrReplayClass::kErrorStub);

  const std::string fatal_line =
      "[HRR] Fatal: T19 Event 54 (hipDrvLaunchKernelEx) returned 400 "
      "(invalid resource handle) \xE2\x80\x94 aborting replay\n";
  const std::string continued_line =
      "[HRR] Error: T19 Event 54 (hipDrvLaunchKernelEx) returned 400 "
      "(invalid resource handle) \xE2\x80\x94 continuing\n";
  CHECK(hrr_observed_replay_class(fatal_line, "hipDrvLaunchKernelEx") ==
        HrrReplayClass::kHandlerError);
  CHECK(hrr_observed_replay_class(continued_line, "hipDrvLaunchKernelEx") ==
        HrrReplayClass::kHandlerError);
  CHECK(hrr_observed_replay_class(continued_line, "hipMalloc") ==
        HrrReplayClass::kReal);

  const std::string many =
      continued_line +
      "[HRR] Error: T19 Event 71 (hipStreamAddCallback) returned 1 "
      "(invalid argument) \xE2\x80\x94 continuing\n" + noop_line;
  const auto failed = hrr_replay_failed_apis(many);
  CHECK(failed.size() == 2);
  CHECK(failed.count("hipDrvLaunchKernelEx") == 1);
  CHECK(failed.count("hipStreamAddCallback") == 1);
  CHECK(hrr_observed_replay_class(many, "hipHostAlloc") ==
        HrrReplayClass::kNoop);

  // An unreplayable API also appears in the failed-API list. The refusal must win,
  // or a declared scope exclusion would read as a broken handler.
  const std::string unreplayable_line =
      "[HRR] hipStreamAddCallback: NOT REPLAYABLE \xE2\x80\x94 the callback is "
      "a host function pointer belonging to the capturing process. The call is "
      "in the archive but its effect cannot be reproduced here.\n";
  const std::string unreplayable_pair =
      unreplayable_line +
      "[HRR] Error: T19 Event 71 (hipStreamAddCallback) returned 801 "
      "(operation not supported) \xE2\x80\x94 continuing\n";
  CHECK(hrr_observed_replay_class(unreplayable_pair, "hipStreamAddCallback") ==
        HrrReplayClass::kUnreplayable);
  CHECK(hrr_observed_replay_class(unreplayable_pair, "hipMalloc") ==
        HrrReplayClass::kReal);
  CHECK(hrr_observed_replay_class(unreplayable_line + continued_line,
                                  "hipDrvLaunchKernelEx") ==
        HrrReplayClass::kHandlerError);

  // Only the fatal form means the replay stopped; the continuing form must not read as a truncated run.
  CHECK(hrr_replay_aborted(fatal_line +
                           "[HRR] Replay aborted due to fatal HIP error\n"));
  CHECK_FALSE(hrr_replay_aborted(many));

// A handler that segfaults prints nothing, so without the verbose event log every API after it would read as a clean real handler.
  const std::string verbose_single =
      "[HRR] Event 0: __hipRegisterFatBinary\n"
      "[HRR] Loaded fat binary blob (47144 bytes) -> hipModule_t\n"
      "[HRR] Event 1: hipMemGetHandleForAddressRange\n"
      "[HRR] NOOP playback handler called for hipMemGetHandleForAddressRange "
      "\xE2\x80\x94 this API is not replayed.\n"
      "[HRR] Event 2: hipMemExportToShareableHandle\n";
  const auto reached = hrr_replay_reached_apis(verbose_single);
  REQUIRE(reached.size() == 3);
  CHECK(reached.front() == "__hipRegisterFatBinary");
  CHECK(reached.back() == "hipMemExportToShareableHandle");
  CHECK(std::find(reached.begin(), reached.end(), "Loaded") == reached.end());

  const std::string verbose_mt =
      "[HRR] T140234 [0] hipMalloc\n"
      "[HRR] T140234 [1] hipMemcpyAsync\n"
      "[HRR] T140235 Event 7: no handler for type 512\n";
  const auto reached_mt = hrr_replay_reached_apis(verbose_mt);
  REQUIRE(reached_mt.size() == 2);
  CHECK(reached_mt[0] == "hipMalloc");
  CHECK(reached_mt[1] == "hipMemcpyAsync");
}

TEST_CASE("Unit_HRR_ApiMatrix_InfoBreakdownParse", "[.][hrr][api-matrix][cpu]") {
  const std::string sample =
      "HRR Archive: /tmp/cap/pid-7\n"
      "========================================\n"
      "Complete:     YES\n"
      "Events:       1326\n"
      "Kernels:      3\n"
      "Blobs:        1277\n"
      "Code Objects: 2\n"
      "Threads:      1\n"
      "\n"
      "Event Type Breakdown:\n"
      "  Type                   Count\n"
      "  ----                   -----\n"
      "  __hipPushCallConfiguration 3\n"
      "  __hipRegisterFatBinary 1277\n"
      "  hipGetDevice           17\n"
      "  hipMalloc              1\n"
      "  hipModuleLaunchKernel  3\n"
      "\n"
      "Kernel Summary (first 20):\n"
      "  ID   Kernel                     Grid            Block\n"
      "  --   ------                     ----            -----\n"
      "  0    hrr_fault_kernel           [16,1,1]        [256,1,1]\n";

  std::map<std::string, long long> counts;
  hrr_parse_info_breakdown(sample, counts);

  CHECK(counts.size() == 5);
  CHECK(counts["hipMalloc"] == 1);
  CHECK(counts["hipGetDevice"] == 17);
  CHECK(counts["__hipRegisterFatBinary"] == 1277);
  CHECK(counts["__hipPushCallConfiguration"] == 3);
  CHECK(counts["hipModuleLaunchKernel"] == 3);

  // The Kernel Summary rows sit past the terminator and are not APIs: "hrr_fault_kernel" is a kernel name.
  CHECK(counts.count("hrr_fault_kernel") == 0);
  CHECK(counts.count("ID") == 0);
  CHECK(counts.count("Complete:") == 0);

  hrr_parse_info_breakdown(sample, counts);
  CHECK(counts["hipMalloc"] == 2);

  // A report with no breakdown, as `--info` prints for an archive root, must yield nothing rather than misparse the process table.
  const std::string root_report =
      "HRR Archive Root: /tmp/cap\n"
      "========================================\n"
      "Capture Mode: in-tree\n"
      "Owner PID:    7\n"
      "Processes:    1\n"
      "\n"
      "  PID          Parent PID   Complete   Events       Blobs      Path\n"
      "  ---          ----------   --------   ------       -----      ----\n"
      "  7            1            NO         1326         1279       /tmp/cap/pid-7\n";
  std::map<std::string, long long> root_counts;
  hrr_parse_info_breakdown(root_report, root_counts);
  CHECK(root_counts.empty());
}

/**
 * @}
 */
