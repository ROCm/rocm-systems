// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "differential.h"

#include "boundary.h"

#include <chrono>
#include <filesystem>

#ifndef RJ_ISA_DIFF_RJ_REVISION
#define RJ_ISA_DIFF_RJ_REVISION "unknown"
#endif
#ifndef RJ_ISA_DIFF_RJ_VERSION
#define RJ_ISA_DIFF_RJ_VERSION "unknown"
#endif

namespace rocjitsu::isa_diff {
namespace {

using Clock = std::chrono::steady_clock;

uint64_t ns_since(Clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

bool is_unsupported_status(RunStatus s) {
  return s == RunStatus::DecodeFailure || s == RunStatus::MissingExecutionBackend ||
         s == RunStatus::Unimplemented || s == RunStatus::UnsupportedOperand ||
         s == RunStatus::IllegalInstruction;
}

bool is_harness_status(RunStatus s) {
  return s == RunStatus::HostException || s == RunStatus::ExecutionError ||
         s == RunStatus::InvalidRegisterAccess || s == RunStatus::BudgetExceeded ||
         s == RunStatus::PcWritten || s == RunStatus::WaveStateChanged;
}

Json trace_json(const std::vector<TraceStep> &trace) {
  Json arr = Json::array();
  for (const auto &step : trace) {
    Json s = Json::object();
    s.set("index", static_cast<uint64_t>(step.index));
    s.set("pc", hex64(step.pc));
    s.set("asm", step.text);
    Json changed = Json::object();
    for (unsigned l = 0; l < kNumLocs; ++l)
      if (step.changed.test(l))
        changed.set(loc_name(l), hex32(step.after.get(l)));
    s.set("changed", std::move(changed));
    if (!step.frame_changed.empty()) {
      Json f = Json::array();
      for (const auto &name : step.frame_changed)
        f.push(name);
      s.set("frame_changed", std::move(f));
    }
    arr.push(std::move(s));
  }
  return arr;
}

Json outcome_json(const RunOutcome &o) {
  Json j = Json::object();
  j.set("status", run_status_name(o.status));
  j.set("executed", static_cast<uint64_t>(o.executed));
  if (o.status != RunStatus::Ok) {
    j.set("failed_index", static_cast<uint64_t>(o.failed_index));
    j.set("detail", o.detail);
  }
  return j;
}

/// Compare one executed pair. Returns the mismatch kind, or empty when they agree.
std::string compare_pair(const RegionPair &region, const MachineState &init_old,
                         const MachineState &init_new, const MachineState &final_old,
                         const MachineState &final_new, Mismatch &m) {
  const RegionContract &k = region.contract;
  // Effect model: execution must not write anything the model did not declare.
  const LocSet extra_old = differing_locs(init_old, final_old) & ~k.declared_writes_old;
  const LocSet extra_new = differing_locs(init_new, final_new) & ~k.declared_writes_new;
  if (extra_old.any() || extra_new.any()) {
    m.locs = extra_old | extra_new;
    m.detail = "execution wrote undeclared locations (old: " + format_locs(extra_old) +
               "; new: " + format_locs(extra_new) + ")";
    return "effect_model";
  }
  // Frame preservation on each side.
  auto fo = differing_frame_fields(init_old.frame, final_old.frame);
  auto fn = differing_frame_fields(init_new.frame, final_new.frame);
  if (!fo.empty() || !fn.empty()) {
    m.frame_fields = fo;
    m.frame_fields.insert(m.frame_fields.end(), fn.begin(), fn.end());
    m.detail = "frame state changed by the region";
    return "frame_preservation";
  }
  // R_out: equality on the observable set.
  const LocSet diff = differing_locs(final_old, final_new) & k.observable;
  if (diff.any()) {
    m.locs = diff;
    for (unsigned l = 0; l < kNumLocs; ++l)
      if (diff.test(l)) {
        m.detail = "first observable difference " + loc_name(l) + ": old " +
                   hex32(final_old.get(l)) + " new " + hex32(final_new.get(l));
        break;
      }
    return "observable";
  }
  return {};
}

} // namespace

const char *test_status_name(TestStatus s) {
  switch (s) {
  case TestStatus::NoMismatchFound:
    return "NO_MISMATCH_FOUND";
  case TestStatus::LocalMismatch:
    return "LOCAL_MISMATCH";
  case TestStatus::Inconclusive:
    return "INCONCLUSIVE";
  case TestStatus::Unsupported:
    return "UNSUPPORTED";
  case TestStatus::HarnessError:
    return "HARNESS_ERROR";
  case TestStatus::NoSemanticDiff:
    return "NO_SEMANTIC_DIFF";
  }
  return "?";
}

const char *witness_scope_name(WitnessScope s) {
  switch (s) {
  case WitnessScope::None:
    return "NONE";
  case WitnessScope::LocalOnly:
    return "LOCAL_ONLY";
  case WitnessScope::ReachableAtCut:
    return "REACHABLE_AT_CUT";
  case WitnessScope::ObservableKernelDivergence:
    return "OBSERVABLE_KERNEL_DIVERGENCE";
  }
  return "?";
}

const char *coverage_scope_name(CoverageScope s) {
  switch (s) {
  case CoverageScope::CompleteForDeclaredScope:
    return "COMPLETE_FOR_DECLARED_SCOPE";
  case CoverageScope::Partial:
    return "PARTIAL";
  case CoverageScope::NotEstablished:
    return "NOT_ESTABLISHED";
  }
  return "?";
}

Json build_identity() {
  Json j = Json::object();
  j.set("tool", kToolName);
  j.set("tool_version", kToolVersion);
  j.set("schema_version", kSchemaVersion);
  j.set("rocjitsu_revision", RJ_ISA_DIFF_RJ_REVISION);
  j.set("rocjitsu_version", RJ_ISA_DIFF_RJ_VERSION);
  j.set("semantics", "RocJITsu ComputeUnitCore::execute_instruction, gfx950 (CDNA4) wave64, "
                     "default arithmetic semantics");
#if defined(__VERSION__)
  j.set("compiler", __VERSION__);
#endif
  return j;
}

void build_initial_states(const RegionContract &contract, const Sample &sample, const Frame &frame,
                          uint64_t seed, uint64_t case_index, MachineState &old_state,
                          MachineState &new_state) {
  old_state = {};
  new_state = {};
  old_state.frame = frame;
  new_state.frame = frame;
  for (unsigned l = 0; l < kNumLocs; ++l) {
    if (contract.entry_equal.test(l)) {
      old_state.set(l, sample.value[l]);
      new_state.set(l, sample.value[l]);
    } else {
      old_state.set(l, canary_value(seed, case_index, l, /*new_side=*/false));
      new_state.set(l, canary_value(seed, case_index, l, /*new_side=*/true));
    }
  }
}

Json RegionResult::to_json() const {
  Json j = Json::object();
  j.set("region", region_id);
  j.set("synthetic", synthetic);
  j.set("test_status", test_status_name(test_status));
  j.set("contract_status", contract_status_name(contract_status));
  j.set("witness_scope", witness_scope_name(witness_scope));
  j.set("coverage_scope", coverage_scope_name(coverage_scope));
  j.set("proof_status", "NOT_RUN");
  if (!reason.empty())
    j.set("reason", reason);
  j.set("prepared_key", prepared_key);
  j.set("policy", policy);
  j.set("seed", seed);
  Json counts = Json::object();
  counts.set("attempted", attempted);
  counts.set("accepted", accepted);
  counts.set("executed_pairs", executed_pairs);
  counts.set("valid_pairs", valid_pairs);
  counts.set("mismatches", mismatches);
  counts.set("harness_errors", harness_errors);
  Json rej = Json::object();
  for (const auto &[k, v] : rejections)
    rej.set(k, v);
  counts.set("rejections", std::move(rej));
  Json mk = Json::object();
  for (const auto &[k, v] : mismatch_kinds)
    mk.set(k, v);
  counts.set("mismatch_kinds", std::move(mk));
  j.set("counts", std::move(counts));
  auto phase_json = [](const PhaseCounts &p) {
    Json o = Json::object();
    o.set("cases", p.cases);
    o.set("accepted", p.accepted);
    o.set("valid", p.valid);
    o.set("mismatches", p.mismatches);
    return o;
  };
  Json bj = phase_json(boundary);
  bj.set("mandatory", true);
  bj.set("generated", boundary_generated);
  bj.set("duplicates_dropped", boundary_duplicates);
  bj.set("truncated", boundary_truncated);
  Json fam = Json::object();
  for (const auto &[k, v] : boundary_families)
    fam.set(k, v);
  bj.set("valid_by_family", std::move(fam));
  j.set("boundary_sweep", std::move(bj));
  j.set("random_phase", phase_json(random));
  Json exec = Json::object();
  exec.set("old_region_insts", static_cast<uint64_t>(old_len));
  exec.set("new_region_insts", static_cast<uint64_t>(new_len));
  exec.set("old_insts_executed", old_insts_executed);
  exec.set("new_insts_executed", new_insts_executed);
  j.set("execution", std::move(exec));
  Json tj = Json::object();
  tj.set("setup_ns", t.setup_ns);
  tj.set("propose_ns", t.propose_ns);
  tj.set("state_load_ns", t.load_ns);
  tj.set("old_execute_ns", t.old_exec_ns);
  tj.set("new_execute_ns", t.new_exec_ns);
  tj.set("compare_ns", t.compare_ns);
  tj.set("triage_ns", t.triage_ns);
  j.set("timing", std::move(tj));
  if (!examples.empty()) {
    const Mismatch &m = examples.front();
    Json first = Json::object();
    first.set("case_index", m.case_index);
    first.set("phase", m.phase);
    first.set("kind", m.kind);
    first.set("locations", format_locs(m.locs));
    first.set("detail", m.detail);
    j.set("first_mismatch", std::move(first));
  }
  Json files = Json::array();
  for (const auto &f : counterexample_files)
    files.push(f);
  j.set("counterexamples", std::move(files));
  return j;
}

Json counterexample_json(const RegionPair &region, const Mismatch &m, uint64_t seed,
                         const std::string &policy, const std::vector<TraceStep> &trace_old,
                         const std::vector<TraceStep> &trace_new) {
  Json j = Json::object();
  j.set("schema_version", "rj_isa_diff.counterexample.v1");
  j.set("synthetic", region.synthetic);
  j.set("evidence", region.synthetic
                        ? "synthetic mismatch from injected mutation; not an LLVM finding"
                        : "region-level reproducer under the recorded contract; kernel/context "
                          "reproducer PENDING (reachability not established in P0)");
  j.set("witness_scope", "LOCAL_ONLY");
  j.set("region_id", region.id);
  j.set("prepared_key", region.prepared_key);
  j.set("seed", seed);
  j.set("policy", policy);
  j.set("case_index", m.case_index);
  j.set("phase", m.phase);
  j.set("kind", m.kind);
  j.set("locations", format_locs(m.locs));
  j.set("detail", m.detail);
  Json firsts = Json::array();
  for (unsigned l = 0; l < kNumLocs; ++l) {
    if (!m.locs.test(l))
      continue;
    Json d = Json::object();
    d.set("loc", loc_name(l));
    d.set("old", hex32(m.final_old.get(l)));
    d.set("new", hex32(m.final_new.get(l)));
    firsts.push(std::move(d));
  }
  j.set("differences", std::move(firsts));
  Json events = Json::array();
  for (const auto &e : m.events)
    events.push(e);
  j.set("input_events", std::move(events));
  j.set("region", region.to_json());
  Json init = Json::object();
  init.set("old", m.init_old.to_json());
  init.set("new", m.init_new.to_json());
  j.set("initial_state", std::move(init));
  Json fin = Json::object();
  fin.set("old", m.final_old.to_json());
  fin.set("new", m.final_new.to_json());
  j.set("final_state", std::move(fin));
  Json exit = Json::object();
  exit.set("old", outcome_json(m.out_old));
  exit.set("new", outcome_json(m.out_new));
  j.set("exit", std::move(exit));
  Json tr = Json::object();
  tr.set("old", trace_json(trace_old));
  tr.set("new", trace_json(trace_new));
  j.set("trace", std::move(tr));
  j.set("identity", build_identity());
  return j;
}

RegionResult test_region(const RegionPair &region, const TestOptions &options,
                         InputPolicy &policy) {
  RegionResult r;
  r.region_id = region.id;
  r.synthetic = region.synthetic;
  r.contract_status = region.contract.status;
  r.prepared_key = region.prepared_key;
  r.policy = policy.name();
  r.seed = options.seed;
  r.old_len = region.old_side.insts.size();
  r.new_len = region.new_side.insts.size();
  if (region.status != RegionStatus::Ready) {
    r.test_status = region.status == RegionStatus::Unsupported ? TestStatus::Unsupported
                                                               : TestStatus::Inconclusive;
    r.reason = region.reason;
    return r;
  }

  auto setup_start = Clock::now();
  std::string error;
  auto old_runner = RegionRunner::create(region.old_side.insts, region.id + "_old", error);
  auto new_runner =
      old_runner ? RegionRunner::create(region.new_side.insts, region.id + "_new", error) : nullptr;
  r.t.setup_ns = ns_since(setup_start);
  if (!old_runner || !new_runner) {
    r.test_status = TestStatus::HarnessError;
    r.reason = error;
    return r;
  }
  if (!(old_runner->baseline_frame() == new_runner->baseline_frame())) {
    r.test_status = TestStatus::HarnessError;
    r.reason = "old and new runners start from different frames";
    return r;
  }
  const Frame frame = old_runner->baseline_frame();

  const RegionFeatures features =
      make_region_features(region.old_side.insts, region.new_side.insts);
  auto sweep_start = Clock::now();
  const BoundarySweep sweep = build_boundary_sweep(features.roles, region.contract,
                                                   {options.seed, options.max_boundary_cases});
  r.t.propose_ns += ns_since(sweep_start);
  r.boundary_generated = sweep.generated;
  r.boundary_duplicates = sweep.duplicates;
  r.boundary_truncated = sweep.truncated;

  std::map<std::string, uint64_t> unsupported_exclusions;
  uint64_t undefined = 0;
  MachineState init_old, init_new;

  // Execute one validated sample on both sides and classify the outcome.
  auto run_case = [&](const Sample &sample, uint64_t case_index, PhaseCounts &phase,
                      const char *phase_name, const std::string &family) {
    ++r.accepted;
    ++phase.accepted;
    auto t1 = Clock::now();
    build_initial_states(region.contract, sample, frame, options.seed, case_index, init_old,
                         init_new);
    old_runner->load(init_old);
    new_runner->load(init_new);
    r.t.load_ns += ns_since(t1);

    auto t2 = Clock::now();
    const RunOutcome out_old = old_runner->run(region.old_side.insts.size());
    r.t.old_exec_ns += ns_since(t2);
    auto t3 = Clock::now();
    const RunOutcome out_new = new_runner->run(region.new_side.insts.size());
    r.t.new_exec_ns += ns_since(t3);
    ++r.executed_pairs;
    r.old_insts_executed += out_old.executed;
    r.new_insts_executed += out_new.executed;

    auto t4 = Clock::now();
    Mismatch m;
    m.case_index = case_index;
    m.phase = phase_name;
    m.out_old = out_old;
    m.out_new = out_new;
    std::string kind;
    if (out_old.status != RunStatus::Ok || out_new.status != RunStatus::Ok) {
      const std::string key = std::string("exit: old ") + run_status_name(out_old.status) +
                              " / new " + run_status_name(out_new.status);
      ++r.rejections[key];
      if (is_harness_status(out_old.status) || is_harness_status(out_new.status)) {
        ++r.harness_errors;
        kind = "harness_exit";
      } else if (is_unsupported_status(out_old.status) || is_unsupported_status(out_new.status)) {
        ++unsupported_exclusions[key];
      } else if (out_old.status == RunStatus::UndefinedBehavior ||
                 out_new.status == RunStatus::UndefinedBehavior) {
        ++undefined;
      }
      r.t.compare_ns += ns_since(t4);
      policy.observe({case_index, true, false, "excluded"});
      if (kind.empty())
        return;
      m.kind = kind;
      m.detail = key + ": " + out_old.detail + " " + out_new.detail;
    } else {
      ++r.valid_pairs;
      ++phase.valid;
      if (!family.empty())
        ++r.boundary_families[family];
      const MachineState final_old = old_runner->capture();
      const MachineState final_new = new_runner->capture();
      kind = compare_pair(region, init_old, init_new, final_old, final_new, m);
      r.t.compare_ns += ns_since(t4);
      policy.observe({case_index, true, !kind.empty(), kind});
      if (kind.empty())
        return;
      m.kind = kind;
      m.final_old = final_old;
      m.final_new = final_new;
      if (kind == "effect_model" || kind == "frame_preservation") {
        ++r.harness_errors;
      } else {
        ++r.mismatches;
        ++phase.mismatches;
      }
    }
    ++r.mismatch_kinds[kind];
    m.init_old = init_old;
    m.init_new = init_new;
    m.events = sample.events;
    if (r.examples.size() < options.max_counterexamples) {
      auto t5 = Clock::now();
      if (!options.counterexample_dir.empty()) {
        // Re-execute with tracing on the same runners; determinism makes this the same run.
        std::vector<TraceStep> trace_old, trace_new;
        old_runner->load(init_old);
        old_runner->run(region.old_side.insts.size(), &trace_old);
        new_runner->load(init_new);
        new_runner->run(region.new_side.insts.size(), &trace_new);
        const Json bundle =
            counterexample_json(region, m, options.seed, policy.name(), trace_old, trace_new);
        std::filesystem::create_directories(options.counterexample_dir);
        const std::string path = options.counterexample_dir + "/" + region.id + "-case" +
                                 std::to_string(case_index) + ".json";
        if (write_text_file(path, bundle.dump()))
          r.counterexample_files.push_back(path);
      }
      r.examples.push_back(std::move(m));
      r.t.triage_ns += ns_since(t5);
    }
  };

  // Phase 1: the boundary sweep always runs in full (exact values; never projected).
  uint64_t case_index = 0;
  for (const SweepCase &c : sweep.cases) {
    auto t0 = Clock::now();
    const Sample sample = sweep.materialize(c);
    ++r.attempted;
    ++r.boundary.cases;
    auto rejection = validate_sample(sample, region.contract);
    r.t.propose_ns += ns_since(t0);
    if (rejection) {
      ++r.rejections["boundary constraint: " + *rejection];
      policy.observe({case_index, false, false, "rejected"});
    } else {
      run_case(sample, case_index, r.boundary, "boundary", c.family);
    }
    ++case_index;
  }

  // Phase 2: random cases from the policy.
  const uint64_t max_attempts = options.max_attempts ? options.max_attempts : 4 * options.cases;
  while (r.random.accepted < options.cases && r.random.cases < max_attempts) {
    auto t0 = Clock::now();
    Sample sample = policy.propose(features, region.contract, case_index);
    project_onto_constraints(sample, region.contract, options.seed ^ (case_index << 1));
    ++r.attempted;
    ++r.random.cases;
    auto rejection = validate_sample(sample, region.contract);
    r.t.propose_ns += ns_since(t0);
    if (rejection) {
      ++r.rejections["constraint: " + *rejection];
      policy.observe({case_index, false, false, "rejected"});
    } else {
      run_case(sample, case_index, r.random, "random", "");
    }
    ++case_index;
  }

  // Status.
  if (r.harness_errors) {
    r.test_status = TestStatus::HarnessError;
    r.reason = "harness or model error during execution (see mismatch_kinds)";
  } else if (r.mismatches) {
    r.test_status = TestStatus::LocalMismatch;
    r.witness_scope = WitnessScope::LocalOnly;
  } else if (r.valid_pairs == 0) {
    if (!unsupported_exclusions.empty()) {
      r.test_status = TestStatus::Unsupported;
      r.reason = "no case could execute: " + unsupported_exclusions.begin()->first;
    } else {
      r.test_status = TestStatus::Inconclusive;
      r.reason = r.accepted == 0 ? "no sample satisfied the constraints (zero valid cases)"
                                 : "no case exited normally on both sides";
    }
  } else if (undefined || !unsupported_exclusions.empty()) {
    r.test_status = TestStatus::Inconclusive;
    r.reason = "some cases ended in undefined behaviour or unsupported execution";
  } else {
    r.test_status = TestStatus::NoMismatchFound;
  }
  if (r.valid_pairs > 0 && r.test_status != TestStatus::HarnessError)
    r.coverage_scope =
        r.boundary_truncated ? CoverageScope::Partial : CoverageScope::CompleteForDeclaredScope;
  if (r.boundary_truncated) {
    if (!r.reason.empty())
      r.reason += "; ";
    r.reason += "boundary sweep truncated at " + std::to_string(sweep.cases.size()) + " of " +
                std::to_string(sweep.generated) + " generated cases (--max-boundary-cases)";
  }
  return r;
}

ReplayOutcome replay_counterexample(const Json &bundle, std::string &error) {
  ReplayOutcome out;
  auto region = RegionPair::from_json(bundle["region"], error);
  if (!region)
    return out;
  if (compute_prepared_key(*region) != bundle["prepared_key"].as_string()) {
    error = "prepared key mismatch: the bundle's region or contract was edited";
    return out;
  }
  auto init_old = MachineState::from_json(bundle["initial_state"]["old"], error);
  auto init_new = MachineState::from_json(bundle["initial_state"]["new"], error);
  if (!init_old || !init_new)
    return out;
  auto old_runner = RegionRunner::create(region->old_side.insts, region->id + "_replay_old", error);
  auto new_runner =
      old_runner ? RegionRunner::create(region->new_side.insts, region->id + "_replay_new", error)
                 : nullptr;
  if (!old_runner || !new_runner)
    return out;
  std::vector<TraceStep> trace_old, trace_new;
  old_runner->load(*init_old);
  const RunOutcome oo = old_runner->run(region->old_side.insts.size(), &trace_old);
  new_runner->load(*init_new);
  const RunOutcome on = new_runner->run(region->new_side.insts.size(), &trace_new);
  Mismatch m;
  m.out_old = oo;
  m.out_new = on;
  std::string kind;
  if (oo.status != RunStatus::Ok || on.status != RunStatus::Ok) {
    kind = "harness_exit";
  } else {
    kind = compare_pair(*region, *init_old, *init_new, old_runner->capture(), new_runner->capture(),
                        m);
  }
  const std::string recorded_kind = bundle["kind"].as_string();
  const std::string recorded_locs = bundle["locations"].as_string();
  out.reproduced = kind == recorded_kind && format_locs(m.locs) == recorded_locs;
  if (kind != "harness_exit" && oo.status == RunStatus::Ok && on.status == RunStatus::Ok) {
    // Final values must match the bundle exactly, not just the kind.
    auto fo = MachineState::from_json(bundle["final_state"]["old"], error);
    auto fn = MachineState::from_json(bundle["final_state"]["new"], error);
    if (!fo || !fn || !(old_runner->capture() == *fo) || !(new_runner->capture() == *fn))
      out.reproduced = false;
  }
  out.summary = std::string(out.reproduced ? "REPRODUCED" : "NOT_REPRODUCED") +
                ": kind=" + (kind.empty() ? "none" : kind) + " locations=" + format_locs(m.locs) +
                " (recorded kind=" + recorded_kind + " locations=" + recorded_locs + ")";
  out.detail = Json::object();
  out.detail.set("reproduced", out.reproduced);
  out.detail.set("kind", kind);
  out.detail.set("locations", format_locs(m.locs));
  out.detail.set("detail", m.detail);
  out.detail.set("synthetic", region->synthetic);
  Json tr = Json::object();
  tr.set("old", trace_json(trace_old));
  tr.set("new", trace_json(trace_new));
  out.detail.set("trace", std::move(tr));
  return out;
}

} // namespace rocjitsu::isa_diff
