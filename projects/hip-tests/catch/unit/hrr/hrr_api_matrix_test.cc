/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR per-API playback matrix
 * @{
 * @ingroup HRRTest
 * One test per priority tier. Each captures that tier's workloads, replays the
 * archives, and asserts that every HIP API the workloads reached behaved at
 * replay the way hrr_api_matrix_expectations.h says it should.
 *
 * What is being asserted
 * ----------------------
 * Not "does replay produce the right numbers" — the roundtrip tests in
 * hrr_roundtrip_test.cc already do that. These assert the weaker but broader
 * property that an API's *replay class* has not changed: an API HRR declares a
 * no-op is still a no-op, and one it declares faithfully replayed still runs a
 * real handler. That distinction is what decides whether a recording of a real
 * workload means anything, and it is exactly the thing that can change
 * silently when the generator's classification sets are edited upstream.
 *
 * How the class is observed
 * -------------------------
 * Two independent readings of the same archive:
 *
 *   captured  the API appears in the Event Type Breakdown of
 *             `hrr-playback --info`. Without this an API the workload never
 *             managed to call is indistinguishable from one that replayed
 *             perfectly, and the matrix would report false green.
 *
 *   class     the merged stdout+stderr of a replay. NOOP handlers emit a
 *             one-time "NOOP playback handler called for <api>" warning;
 *             ERROR_STUB handlers emit a named graph-construction warning.
 *             Neither warning, plus a clean run, means a real handler ran.
 *
 * Coverage, not exhaustiveness
 * ----------------------------
 * Reaching all 551 APIs from a test binary is not achievable — some need a
 * second process, some need hardware this host does not have, some would tear
 * down the context the rest of the capture depends on. So an API a tier did
 * not reach is reported as not-exercised rather than failed, and each tier
 * instead has to clear a coverage floor. A workload that dies on its second
 * line then fails loudly, instead of reporting a clean sweep of nothing.
 *
 * Set HRR_MATRIX_RESULTS_DIR to have each tier write its observations as JSON
 * for aim-labs/scenarios/prep/hrr-api-matrix/check_matrix.py to turn into a
 * coverage report.
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

#include "hrr_test_common.h"
#include "hrr_api_matrix_expectations.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

// A replay of a NOOP-heavy tier can take a while on a busy machine, but it
// should never take minutes. hipStreamWriteValue64's no-op means a workload
// that waits on the value it should have written waits forever (section 7,
// hazard H2 of HRR-Use-Case-Priorities.md), so every replay here runs under a
// deadline and a hang becomes an assertable outcome rather than a stuck job.
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

// A real handler that ran is REAL when it returned cleanly, HANDLER_ERROR
// when it did not, and CRASH when it did not come back at all. Two workloads
// landing on two of those three saw the same handler behave differently on
// different arguments, not a different handler.
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

// Minimal JSON string escaping. The only values written are HIP API names and
// Catch2 test-case names, so this covers the cases that can actually occur
// rather than the whole grammar.
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

// What one tier's run saw, aggregated over all its workloads.
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

// ---------------------------------------------------------------------------
// Capture one workload and fold what it recorded into `obs`.
//
// Everything that can go wrong here is reported with FAIL_CHECK rather than
// FAIL: the tier fails either way, but the remaining workloads still run and
// the observation file is still written. One workload crashing should cost
// its own APIs, not the report for the fifteen next to it.
// ---------------------------------------------------------------------------
void observe_workload(const std::string& direct_case, TierObservation& obs) {
  ScopedDir cap{fs::temp_directory_path() /
                ("hrr_matrix_" + direct_case)};

  {
    hip::SpawnProc proc(HRR_TEST_EXE);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    const int ret = proc.run("\"" + direct_case + "\"");
    // The workloads gate unsupported hardware features internally and still
    // exit clean, so a non-zero exit is a genuine failure. Reporting it as a
    // warning would let the tier quietly degrade to zero coverage.
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

  // --continue-on-error is what makes a whole-archive survey possible. Without
  // it the replay stops at the first failing handler, and since several APIs
  // in every tier fail by design, the events after that point would go
  // unobserved and the tier would report a handful of APIs instead of all of
  // them. The failures are still counted; they are just not fatal here.
  //
  // --verbose names every event as it is dispatched, which is the only way to
  // tell "this API replayed silently" from "the replay never got to it".
  auto [replay_rc, merged] = hrr_playback_watchdog(
      cap.path, kReplayTimeoutSeconds, "--continue-on-error --verbose");
  INFO("Replay of " << direct_case << " exited " << replay_rc);
  // A hang is the one replay outcome that is never acceptable: it is the H2
  // symptom, and it is also what would wedge CI.
  if (replay_rc == kHrrWatchdogKilled) {
    FAIL_CHECK("replay of " << direct_case << " did not finish within "
                      << kReplayTimeoutSeconds << "s (watchdog killed it)");
  }
  obs.worst_replay_exit = std::max(obs.worst_replay_exit, replay_rc);

  // --continue-on-error should mean the replay reaches the end of the archive.
  // If it stopped anyway — a special event failed, or the GPU went away — the
  // events after that point never ran, and silence from them is absence of
  // evidence rather than evidence of a real handler. Record nothing for those
  // rather than a tier-wide sweep of false REALs.
  if (hrr_replay_aborted(merged)) {
    const auto failed = hrr_replay_failed_apis(merged);
    FAIL_CHECK("replay of " << direct_case << " stopped early despite "
         << "--continue-on-error"
         << (failed.empty() ? "" : ", last failing API " + *failed.rbegin())
         << "; the events after it went unobserved");
  }

  // A handler can also end the replay by dereferencing a capture-time host
  // pointer, which is a signal rather than an error return and leaves no
  // message behind. The verbose event log still names it: the last event
  // printed is the one that did not come back. Everything the archive holds
  // after it is unobservable until that handler is fixed, so it is left
  // unobserved and surfaces as not-exercised.
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

    // Not reached: no evidence either way. The API stays unobserved, and the
    // tier's coverage floor is what notices if too much of it went that way.
    if (!reached_set.count(kv.first)) continue;

    const HrrReplayClass klass = kv.first == crash_api
                                     ? HrrReplayClass::kCrash
                                     : hrr_observed_replay_class(merged,
                                                                 kv.first);
    const auto it = obs.observed.find(kv.first);
    if (it == obs.observed.end()) {
      obs.observed.emplace(kv.first, klass);
    } else if (it->second != klass) {
      // Two workloads saw the same API do different things. Which of the two
      // shapes that is matters:
      //
      // REAL versus HANDLER_ERROR or CRASH is the same real handler
      // succeeding on one archive's arguments and failing on another's —
      // hipGraphInstantiate replays fine for a stream-captured graph and
      // fails for one built out of ERROR_STUB node-adds. Keep the failure:
      // it is the finding, and api_matrix.yaml is where it gets declared.
      //
      // Anything else — NOOP against REAL, ERROR_STUB against NOOP — is the
      // API having been reclassified, which is what this matrix exists to
      // catch.
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

// ---------------------------------------------------------------------------
// Run a whole tier: capture every workload, then assert every expectation.
// ---------------------------------------------------------------------------
void run_tier(const std::string& tier) {
  const HrrTierFloor& floor = tier_floor(tier);

  if (floor.gpus > 1 && visible_device_count() < floor.gpus) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
    return;
  }

  TierObservation obs;
  obs.tier = tier;
  for (const char* const* w = floor.workloads; *w != nullptr; ++w)
    observe_workload(*w, obs);

  write_observation(obs);

  REQUIRE_FALSE(obs.workloads.empty());

  int covered = 0, not_exercised = 0, skipped = 0;
  std::vector<std::string> missing;
  for (size_t i = 0; i < kHrrApiMatrixCount; ++i) {
    const HrrApiExpectation& e = kHrrApiMatrix[i];
    if (tier != e.tier) continue;
    if (e.skip) { ++skipped; continue; }

    const auto it = obs.captured.find(e.api);
    const long long count = it == obs.captured.end() ? 0 : it->second;

    // Declared as folded away by HIP before the capture layer runs. The
    // absence is the assertion: a workload does call it, and it still must
    // not appear. If it starts appearing, HIP's lowering changed and the
    // matrix is describing the past.
    if (!e.expect_captured) {
      INFO("API: " << e.api << " (tier " << tier
                   << ") is declared as never captured");
      CHECK(count == 0);
      ++covered;
      continue;
    }

    if (it == obs.captured.end() || it->second < 1) {
      ++not_exercised;
      missing.emplace_back(e.api);
      continue;
    }

    // Captured, but the replay died before reaching it. Counting this as
    // covered would mean asserting on a class nobody measured.
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
    // A payload-loss API is still expected to run a real handler — that is
    // precisely what hides the loss. The class is what this test asserts; the
    // loss itself is what check_matrix.py records as XFAIL.
    INFO("Known payload loss (section 8.3): " << (e.payload_loss ? "yes" : "no"));
    // A handler declared as failing on this tier's arguments may come back
    // either way: hipGraphInstantiate replays fine for the stream-captured
    // graphs and fails for the one built out of ERROR_STUB node-adds, and
    // which of those a run lands on depends on which workloads ran.
    if (e.handler_error_ok && expected == HrrReplayClass::kReal &&
        observed == HrrReplayClass::kHandlerError) {
      INFO("Declared in api_matrix.yaml as failing for this tier's arguments");
      continue;
    }
    CHECK(observed == expected);
  }

  // Report before asserting, so a floor failure comes with the list of what
  // went missing rather than just a number.
  std::sort(missing.begin(), missing.end());
  std::string missing_list;
  for (const auto& m : missing) {
    if (!missing_list.empty()) missing_list += ", ";
    missing_list += m;
  }
  INFO("Tier " << tier << ": " << covered << " covered, " << not_exercised
               << " not exercised, " << skipped << " skipped");
  INFO("Not exercised: " << missing_list);
  INFO("Workloads that contributed: " << obs.workloads.size());
  CHECK(covered >= floor.min_covered);
}

}  // namespace

/**
 * Test Description
 * ----------------
 *   - T0, the UC1 dense LLM inference and serving substrate: the rank-1
 *     use-case from section 2 of HRR-Use-Case-Priorities.md.
 *   - Captures Unit_HRR_ApiMatrix_UC1Dense_Direct, which exercises the
 *     PyTorch + hipBLASLt surface and, in particular, the launch and module
 *     holes no other workload reaches: hipDrvLaunchKernelEx (Triton's default
 *     modern launch path), hipLaunchKernelExC (CK cluster launches),
 *     hipLaunchHostFunc recorded into a stream-captured graph (reached through
 *     aiter's mha_bwd) and hipModuleUnload.
 *   - Asserts each API's replay class still matches the matrix, so a change to
 *     the substrate every Instinct serving deployment depends on cannot land
 *     silently.
 */
TEST_CASE("Unit_HRR_ApiMatrix_T0_Roundtrip", "[.][hrr][api-matrix]") {
  run_tier("T0");
}

/**
 * Test Description
 * ----------------
 *   - T1, the payload-loss class from section 8.3 (priorities P1 and P2).
 *   - Every API here has a real playback handler and is counted in the "273
 *     faithfully replayed" figure, yet cannot be replayed: the generator
 *     lowers a const-struct pointer to a bare capture-time address, or a
 *     >8-byte by-value struct to a single 8-byte field. hipIpcOpenMemHandle is
 *     the sharpest case — 56 of the handle's 64 bytes are discarded.
 *   - The assertion is deliberately that the handler still runs. That is what
 *     makes the loss invisible today, and it is why these are XFAIL in the
 *     report rather than PASS: they turn into XPASS the day P2 lands, which is
 *     the signal to update api_matrix.yaml.
 */
TEST_CASE("Unit_HRR_ApiMatrix_T1_Roundtrip", "[.][hrr][api-matrix]") {
  run_tier("T1");
}

/**
 * Test Description
 * ----------------
 *   - T2, the silent failures. Section 9 ranks these above breadth: a replay
 *     that reports success while producing wrong numbers is worse than one
 *     that refuses to run.
 *   - Covers the graph-mutation family that makes ggml/llama.cpp replay every
 *     token with the first token's parameters (section 8.4a), the stream value
 *     operations whose no-op makes XLA's VMM allocator hang rather than fail
 *     (hazard H2), hipHostAlloc allocating nothing (section 8.6), the
 *     __device__ symbol path MoRI's globalGpuStates arrives through, and
 *     hipModuleLaunchCooperativeKernel, live on Instinct via MIOpen's Winograd
 *     Fury solver.
 *   - Runs under the replay watchdog, so the H2 hang fails the test in bounded
 *     time instead of wedging the job.
 */
TEST_CASE("Unit_HRR_ApiMatrix_T2_Roundtrip", "[.][hrr][api-matrix]") {
  run_tier("T2");
}

/**
 * Test Description
 * ----------------
 *   - T3, multi-GPU: device identity, peer access, IPC and VMM (P3, P4, P7).
 *   - Section 5 puts the multi-GPU delta at 42 APIs and roughly 31% faithful,
 *     and the problem is structural rather than per-API: events carry no
 *     device ID and alloc_map has no device field, so replay cannot know which
 *     GPU an allocation belonged to.
 *   - Requires two visible devices; run-api-matrix.sh supplies them with
 *     HIP_VISIBLE_DEVICES=6,7. Skips cleanly on a single-GPU host rather than
 *     failing, so the rest of the matrix stays runnable anywhere.
 */
TEST_CASE("Unit_HRR_ApiMatrix_T3_Roundtrip", "[.][hrr][api-matrix]") {
  run_tier("T3");
}

/**
 * Test Description
 * ----------------
 *   - T4, the breadth sweep: every API not called out by a higher tier.
 *   - Captures the new Unit_HRR_ApiMatrix_Breadth_Direct workload together
 *     with the suite's existing device, stream, memcpy, memset, mempool,
 *     module, occupancy and graph workloads, rather than writing a second copy
 *     of coverage that already exists.
 *   - The 254 NOOP APIs mostly land here. Asserting they are still NOOP is the
 *     point: a NOOP that quietly became a real handler, or the reverse,
 *     changes what every existing recording means.
 */
TEST_CASE("Unit_HRR_ApiMatrix_T4_Roundtrip", "[.][hrr][api-matrix]") {
  run_tier("T4");
}

/**
 * Test Description
 * ----------------
 *   - T5, the deprioritised families: textures, hipArray, surfaces, mipmaps,
 *     managed memory and the cooperative/extended launch spellings with no
 *     library caller.
 *   - Section 10 records verified-negative evidence for all of it: MIOpen has
 *     zero texture occurrences across its whole repository, shipped
 *     rocFFT/rocSPARSE/rocRAND import zero texture symbols, and the
 *     managed-memory caller that does exist belongs to a recommender workload
 *     absent from Instinct MLPerf submissions.
 *   - Hidden ([.]) so it does not run by default. run-api-matrix.sh
 *     --include-deprioritised runs it by name.
 */
TEST_CASE("Unit_HRR_ApiMatrix_T5_Roundtrip", "[.][hrr][api-matrix]") {
  run_tier("T5");
}

/**
 * Test Description
 * ----------------
 *   - CPU-only structural check on the generated expectations header: the
 *     matrix must name every API exactly once, every entry must belong to a
 *     declared tier, and the expectation codes must be in range.
 *   - This is what catches a bad regeneration before a GPU run wastes time on
 *     it, and it is why the header is generated rather than hand-written.
 */
TEST_CASE("Unit_HRR_ApiMatrix_ManifestWellFormed", "[.][hrr][api-matrix][cpu]") {
  REQUIRE(kHrrApiMatrixCount > 500);
  REQUIRE(kHrrTierFloorCount >= 1);

  std::set<std::string> tiers;
  for (size_t i = 0; i < kHrrTierFloorCount; ++i) {
    REQUIRE(kHrrTierFloors[i].tier != nullptr);
    REQUIRE(kHrrTierFloors[i].workloads != nullptr);
    REQUIRE(kHrrTierFloors[i].gpus >= 1);
    REQUIRE(kHrrTierFloors[i].min_covered >= 0);
    tiers.insert(kHrrTierFloors[i].tier);
    // A tier with no workload can never be exercised, so its results would be
    // vacuously clean.
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
}

/**
 * Test Description
 * ----------------
 *   - CPU-only check that the two ends of the observation pipeline agree.
 *   - hrr_observed_replay_class() classifies a replay by matching the one-time
 *     warnings the generator emits. Those strings are the only contract
 *     between hrr-playback and this matrix, and nothing else in the build
 *     would notice if the wording changed, so pin the exact shapes here.
 */
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

  // A warning about one API must not classify another. The NOOP marker embeds
  // the API name for exactly this reason.
  CHECK(hrr_observed_replay_class(noop_line, "hipHostMalloc") ==
        HrrReplayClass::kReal);
  CHECK(hrr_observed_replay_class(stub_line, "hipGraphAddMemsetNode") ==
        HrrReplayClass::kReal);

  // HIP API names prefix one another, so a marker that is not bounded at the
  // end classifies the wrong API: hipMallocHost's warning would otherwise read
  // as a warning for hipMalloc, and hipMalloc is REAL in every tier that has
  // it. That mismatch is reported as a cross-workload conflict, which sends
  // the reader looking for a behaviour change that never happened.
  const std::string prefix_line =
      "[HRR] NOOP playback handler called for hipMallocHost \xE2\x80\x94 this "
      "API is not replayed; results may differ from capture.\n";
  CHECK(hrr_observed_replay_class(prefix_line, "hipMallocHost") ==
        HrrReplayClass::kNoop);
  CHECK(hrr_observed_replay_class(prefix_line, "hipMalloc") ==
        HrrReplayClass::kReal);

  // An ERROR_STUB API also matches nothing in the NOOP marker and vice versa,
  // so the two classes cannot be confused when both appear in one replay.
  const std::string both = noop_line + stub_line;
  CHECK(hrr_observed_replay_class(both, "hipHostAlloc") ==
        HrrReplayClass::kNoop);
  CHECK(hrr_observed_replay_class(both, "hipGraphAddNode") ==
        HrrReplayClass::kErrorStub);

  // A handler that returns a HIP error reports itself in one of two forms
  // depending on whether the replay was told to keep going. Both name the API,
  // and the matrix relies on that to tell "this API failed" apart from "this
  // API ran fine".
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

  // --continue-on-error means several can fail in one replay, and each must be
  // attributed to its own API rather than to whichever failed first.
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

  // An unreplayable API returns hipErrorNotSupported, so it appears in the
  // failed-API list as well. The refusal is the more specific fact and must
  // win, or a declared scope exclusion would read as a handler that broke.
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
  // A handler that genuinely failed is still HANDLER_ERROR when some other API
  // was the unreplayable one.
  CHECK(hrr_observed_replay_class(unreplayable_line + continued_line,
                                  "hipDrvLaunchKernelEx") ==
        HrrReplayClass::kHandlerError);

  // Only the fatal form means the replay stopped; the continuing form must not
  // be read as a truncated run, or every tier would fail on its own design.
  CHECK(hrr_replay_aborted(fatal_line +
                           "[HRR] Replay aborted due to fatal HIP error\n"));
  CHECK_FALSE(hrr_replay_aborted(many));

  // The verbose event log is what separates "replayed silently" from "never
  // reached". A handler that segfaults says nothing at all, so without this
  // every API recorded after it would be read as a clean real handler — which
  // is how a crashing hipMemExportToShareableHandle turned three unrelated
  // NOOP APIs into reported reclassifications.
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
  // The last event of a replay that died is the one that killed it.
  CHECK(reached.back() == "hipMemExportToShareableHandle");
  // The explanatory lines between events name no API and must not be read as
  // one, or the crash would be attributed to whatever was logged last.
  CHECK(std::find(reached.begin(), reached.end(), "Loaded") == reached.end());

  // The multi-threaded replay path prints a different shape for the same
  // thing, and the tiers do produce multi-threaded archives.
  const std::string verbose_mt =
      "[HRR] T140234 [0] hipMalloc\n"
      "[HRR] T140234 [1] hipMemcpyAsync\n"
      "[HRR] T140235 Event 7: no handler for type 512\n";
  const auto reached_mt = hrr_replay_reached_apis(verbose_mt);
  REQUIRE(reached_mt.size() == 2);
  CHECK(reached_mt[0] == "hipMalloc");
  CHECK(reached_mt[1] == "hipMemcpyAsync");
}

/**
 * Test Description
 * ----------------
 *   - CPU-only check on the other half of the observation pipeline: the
 *     "Event Type Breakdown" parser that decides whether an API was captured.
 *   - Pinned here because a parser that silently returns nothing is
 *     indistinguishable from a workload that captured nothing, and the matrix
 *     would then report a confident sweep of not-exercised across all 551
 *     APIs. The sample below is a verbatim `hrr-playback --info` report,
 *     including the surrounding sections the parser has to stop at.
 */
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

  // The Kernel Summary rows sit past the terminator and must not be read as
  // APIs — "hrr_fault_kernel" is a kernel name, not a HIP entry point.
  CHECK(counts.count("hrr_fault_kernel") == 0);
  CHECK(counts.count("ID") == 0);
  CHECK(counts.count("Complete:") == 0);

  // Two process archives of the same capture accumulate rather than overwrite.
  hrr_parse_info_breakdown(sample, counts);
  CHECK(counts["hipMalloc"] == 2);

  // A report with no breakdown at all — which is what `--info` prints when it
  // is pointed at an archive root instead of a pid-<pid> sub-archive — must
  // yield nothing rather than misparse the process table.
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
