// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "differential.h"

#include "boundary.h"

#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <unordered_set>

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
  const auto fwd = k.exit_forward();
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
  // R_out: equality on the observable set (old naming; the new side is read at map(l)).
  LocSet diff;
  for (unsigned l = 0; l < kNumLocs; ++l)
    if (k.observable.test(l) && final_old.get(l) != final_new.get(fwd[l]))
      diff.set(l);
  if (diff.any()) {
    m.locs = diff;
    for (unsigned l = 0; l < kNumLocs; ++l)
      if (diff.test(l)) {
        m.detail = "first observable difference " + loc_name(l) + ": old " +
                   hex32(final_old.get(l)) + " new " +
                   (fwd[l] == l ? "" : "(" + loc_name(fwd[l]) + ") ") +
                   hex32(final_new.get(fwd[l]));
        break;
      }
    return "observable";
  }
  return {};
}

// ---------------------------------------------------------------------------
// Novelty (handoff 8): a bounded classification of an outcome. Raw register values never
// form a class; only the outcome kind and, per written location, a coarse value class
// and whether old and new agree.

unsigned value_class(uint32_t v) {
  if (v == 0)
    return 0;
  if (v == 1)
    return 1;
  if (v == 0xffffffffu)
    return 2;
  if (v & 0x80000000u)
    return 3;
  return 4;
}

uint64_t novelty_bucket(const RegionContract &k, const std::string &category,
                        const MachineState *final_old, const MachineState *final_new) {
  uint64_t h = splitmix64(0x6e6f76656c747900ull);
  for (char c : category)
    h = splitmix64(h ^ static_cast<uint8_t>(c));
  if (!final_old || !final_new)
    return h;
  const auto fwd = k.exit_forward();
  const auto inv = k.exit_inverse();
  LocSet watched = k.declared_writes_old;
  for (unsigned l = 0; l < kNumLocs; ++l)
    if (k.declared_writes_new.test(l))
      watched.set(inv[l]);
  watched.set(kLocScc);
  for (unsigned l = 0; l < kNumLocs; ++l) {
    if (!watched.test(l))
      continue;
    const unsigned eq = final_old->get(l) == final_new->get(fwd[l]) ? 1u : 0u;
    h = splitmix64(h ^ (uint64_t{l} << 8) ^ (value_class(final_old->get(l)) << 1) ^ eq);
  }
  return h;
}

// ---------------------------------------------------------------------------
// Counterexample reduction (handoff 10). Shrink the inputs while the same kind of
// mismatch persists under the contract. The program is never reduced here: changing it
// would require re-planning the region and contract.

bool simpler(uint32_t candidate, uint32_t current) {
  const int pc = std::popcount(candidate), pv = std::popcount(current);
  return pc < pv || (pc == pv && candidate < current);
}

Json reduce_mismatch(const RegionPair &region, RegionRunner &old_runner, RegionRunner &new_runner,
                     Mismatch &m, size_t budget) {
  const RegionContract &k = region.contract;
  const auto fwd = k.entry_forward();
  const MachineState original_old = m.init_old, original_new = m.init_new;
  const LocSet original_locs = m.locs;
  MachineState cur_o = m.init_old, cur_n = m.init_new;
  size_t runs = 0, steps = 0;
  bool seed_independent = false;
  auto attempt = [&](const MachineState &a, const MachineState &b) {
    if (runs >= budget)
      return false;
    ++runs;
    Sample s;
    for (unsigned l = 0; l < kNumLocs; ++l)
      s.value[l] = a.get(l);
    if (ConstraintChecker::validate(s, k))
      return false; // never leave the contract
    old_runner.load(a);
    const RunOutcome oo = old_runner.run(region.old_side.insts.size());
    new_runner.load(b);
    const RunOutcome on = new_runner.run(region.new_side.insts.size());
    if (oo.status != RunStatus::Ok || on.status != RunStatus::Ok)
      return false;
    const MachineState fa = old_runner.capture(), fb = new_runner.capture();
    Mismatch t;
    const std::string kind = compare_pair(region, a, b, fa, fb, t);
    if (kind != m.kind)
      return false;
    m.locs = t.locs;
    m.detail = t.detail;
    m.frame_fields = t.frame_fields;
    m.init_old = a;
    m.init_new = b;
    m.final_old = fa;
    m.final_new = fb;
    m.out_old = oo;
    m.out_new = on;
    return true;
  };
  // 1. Seed independence: replace the seed-derived canaries with fixed distinct values.
  {
    MachineState a = cur_o, b = cur_n;
    for (unsigned l = 0; l < kNumLocs; ++l)
      if (k.canary.test(l)) {
        a.set(l, 0);
        b.set(fwd[l], l == kLocScc ? 1u : 0xffffffffu);
      }
    if (!(a == cur_o && b == cur_n) && attempt(a, b)) {
      cur_o = a;
      cur_n = b;
      seed_independent = true;
      ++steps;
    } else if (a == cur_o && b == cur_n) {
      seed_independent = true;
    }
  }
  // 2. Shrink each shared entry value: 0, 1, halvings, then clearing single bits.
  bool progress = true;
  for (int pass = 0; progress && pass < 8 && runs < budget; ++pass) {
    progress = false;
    for (unsigned l = 0; l < kNumLocs; ++l) {
      if (!k.entry_equal.test(l))
        continue;
      const uint32_t v = cur_o.get(l);
      if (v == 0)
        continue;
      std::vector<uint32_t> cands = {0u, 1u};
      for (uint32_t h = v >> 1; h; h >>= 1)
        cands.push_back(h);
      for (int bit = 31; bit >= 0; --bit)
        if (v & (1u << bit))
          cands.push_back(v & ~(1u << bit));
      for (uint32_t c : cands) {
        if (c == v || !simpler(c, v))
          continue;
        MachineState a = cur_o, b = cur_n;
        a.set(l, c);
        b.set(fwd[l], c);
        if (attempt(a, b)) {
          cur_o = a;
          cur_n = b;
          ++steps;
          progress = true;
          break;
        }
      }
    }
  }
  // attempt() leaves m at the last accepted state; make sure that is the current one.
  if (steps == 0) {
    m.init_old = original_old;
    m.init_new = original_new;
  }
  Json red = Json::object();
  red.set("runs", static_cast<uint64_t>(runs));
  red.set("accepted_steps", static_cast<uint64_t>(steps));
  red.set("budget_exhausted", runs >= budget);
  red.set("seed_independent", seed_independent);
  red.set("program_reduced", false);
  red.set("original_locations", format_locs(original_locs));
  Json orig = Json::object();
  orig.set("old", original_old.to_json());
  orig.set("new", original_new.to_json());
  red.set("original_initial_state", std::move(orig));
  Json nonzero = Json::object();
  for (unsigned l = 0; l < kNumLocs; ++l)
    if (k.entry_equal.test(l) && cur_o.get(l) != 0)
      nonzero.set(loc_name(l), hex32(cur_o.get(l)));
  red.set("reduced_entry_values", std::move(nonzero));
  return red;
}

// ---------------------------------------------------------------------------
// Per-case learning log (handoff 8).

Json summary_json(const std::vector<InstSummary> &insts) {
  Json arr = Json::array();
  for (const auto &i : insts) {
    Json o = Json::object();
    o.set("pc", hex64(i.pc));
    o.set("mnemonic", i.mnemonic);
    o.set("asm", i.text);
    o.set("reads", format_locs(i.reads));
    o.set("writes", format_locs(i.writes));
    arr.push(std::move(o));
  }
  return arr;
}

Json log_header(const RegionPair &region, const RegionFeatures &f, const TestOptions &options,
                const std::string &policy, size_t sweep_cases) {
  Json h = Json::object();
  h.set("type", "header");
  h.set("schema", "rj_isa_diff.case_log.v1");
  h.set("region", region.id);
  if (!region.parent.empty())
    h.set("parent", region.parent);
  h.set("synthetic", region.synthetic);
  h.set("prepared_key", region.prepared_key);
  h.set("seed", options.seed);
  h.set("policy", policy);
  Json k = Json::object();
  k.set("status", contract_status_name(region.contract.status));
  k.set("entry_equal", format_locs(region.contract.entry_equal));
  k.set("observable", format_locs(region.contract.observable));
  k.set("pending_effects", region.contract.pending_effects);
  k.set("composition", region.contract.composition);
  h.set("contract", std::move(k));
  h.set("old", summary_json(f.roles.old_summary));
  h.set("new", summary_json(f.roles.new_summary));
  Json roles = Json::object();
  for (unsigned l = 0; l < kNumLocs; ++l)
    if (f.roles.roles[l])
      roles.set(loc_name(l), role_names(f.roles.roles[l]));
  h.set("roles", std::move(roles));
  Json pairs = Json::array();
  for (const auto &p : f.roles.pairs)
    pairs.push(p.mnemonic + " " + loc_name(p.a) + "," + loc_name(p.b) +
               (p.wide ? " (64-bit)" : ""));
  h.set("pairs", std::move(pairs));
  Json rel = Json::array();
  for (const auto &c : f.roles.const_relations)
    rel.push(c.mnemonic + " " + loc_name(c.loc) + " vs " + hex32(c.constant));
  h.set("const_relations", std::move(rel));
  h.set("boundary_cases", static_cast<uint64_t>(sweep_cases));
  return h;
}

Json feedback_json(const Feedback &fb) {
  Json o = Json::object();
  o.set("type", "case");
  o.set("case", fb.case_index);
  o.set("phase", fb.phase);
  o.set("accepted", fb.accepted);
  if (!fb.rejection.empty())
    o.set("rejection", fb.rejection);
  Json ev = Json::array();
  for (const auto &e : fb.events)
    ev.push(e);
  o.set("events", std::move(ev));
  if (fb.accepted) {
    Json ex = Json::array();
    ex.push(fb.exit_old);
    ex.push(fb.exit_new);
    o.set("exit", std::move(ex));
    o.set("category", fb.category);
    if (fb.locs.any())
      o.set("locs", format_locs(fb.locs));
    o.set("bucket", hex64(fb.bucket));
    o.set("novel", fb.novel);
    Json cost = Json::array();
    for (uint64_t c : {fb.propose_ns, fb.load_ns, fb.old_ns, fb.new_ns, fb.compare_ns})
      cost.push(c);
    o.set("cost_ns", std::move(cost)); // propose, load, old, new, compare
    Json insts = Json::array();
    insts.push(fb.insts_old);
    insts.push(fb.insts_new);
    o.set("insts", std::move(insts));
  }
  return o;
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
  // Sample and contract sets use old naming; the new side receives each value at map(l)
  // (identity without a register map). entry_forward() is a permutation, so every new
  // location is initialised exactly once.
  const auto fwd = contract.entry_forward();
  for (unsigned l = 0; l < kNumLocs; ++l) {
    if (contract.entry_equal.test(l)) {
      old_state.set(l, sample.value[l]);
      new_state.set(fwd[l], sample.value[l]);
    } else {
      old_state.set(l, canary_value(seed, case_index, l, /*new_side=*/false));
      new_state.set(fwd[l], canary_value(seed, case_index, l, /*new_side=*/true));
    }
  }
}

Json RegionResult::to_json() const {
  if (!cached.is_null())
    return cached;
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
  if (!pending_effects.empty())
    j.set("pending_effects", pending_effects);
  if (!composition.empty())
    j.set("composition", composition);
  if (!resolution.empty()) {
    Json e = Json::object();
    e.set("resolution", resolution);
    if (!resolved_by.empty()) {
      e.set("resolved_by", resolved_by);
      e.set("resolved_contract_status", contract_status_name(resolved_contract));
      e.set("resolved_coverage_scope", coverage_scope_name(resolved_coverage));
    }
    e.set("tried", expansion_tried);
    j.set("expansion", std::move(e));
  }
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
  Json nov = Json::object();
  nov.set("distinct_buckets", novelty_buckets);
  nov.set("first_seen_in_boundary", novel_boundary);
  nov.set("first_seen_in_random", novel_random);
  j.set("novelty", std::move(nov));
  if (!case_log.empty())
    j.set("case_log", case_log);
  if (!unrestricted.is_null())
    j.set("unrestricted", unrestricted);
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
  // Observable locations use old naming (the new side is read at map(l)); effect-model
  // locations are physical on their own side.
  const auto fwd = region.contract.exit_forward();
  const bool observable = m.kind == "observable";
  Json firsts = Json::array();
  for (unsigned l = 0; l < kNumLocs; ++l) {
    if (!m.locs.test(l))
      continue;
    const unsigned nl = observable ? fwd[l] : l;
    Json d = Json::object();
    d.set("loc", loc_name(l));
    if (nl != l)
      d.set("new_loc", loc_name(nl));
    d.set("old", hex32(m.final_old.get(l)));
    d.set("new", hex32(m.final_new.get(nl)));
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
  if (!m.reduction.is_null())
    j.set("reduction", m.reduction);
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
  r.pending_effects = region.contract.pending_effects;
  r.composition = region.contract.composition;
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

  const auto new_to_old = region.contract.entry_inverse();
  const RegionFeatures features =
      make_region_features(region.old_side.insts, region.new_side.insts,
                           region.contract.entry_map.empty() ? nullptr : &new_to_old);
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
  History history;
  std::unordered_set<uint64_t> buckets;

  // Optional per-case learning log (handoff 8).
  std::ofstream log;
  if (!options.case_log_dir.empty()) {
    std::filesystem::create_directories(options.case_log_dir);
    r.case_log = options.case_log_dir + "/" + region.id + ".jsonl";
    log.open(r.case_log, std::ios::trunc);
    if (log)
      log << log_header(region, features, options, policy.name(), sweep.cases.size()).dump(-1)
          << '\n';
    else
      r.case_log.clear();
  }
  auto finish = [&](Feedback &fb) {
    policy.observe(fb);
    history.record(fb);
    if (log)
      log << feedback_json(fb).dump(-1) << '\n';
  };

  // Execute one validated sample on both sides and classify the outcome.
  auto run_case = [&](const Sample &sample, uint64_t case_index, PhaseCounts &phase,
                      const char *phase_name, const std::string &family, Feedback &fb) {
    ++r.accepted;
    ++phase.accepted;
    fb.accepted = true;
    auto t1 = Clock::now();
    build_initial_states(region.contract, sample, frame, options.seed, case_index, init_old,
                         init_new);
    old_runner->load(init_old);
    new_runner->load(init_new);
    fb.load_ns = ns_since(t1);
    r.t.load_ns += fb.load_ns;

    auto t2 = Clock::now();
    const RunOutcome out_old = old_runner->run(region.old_side.insts.size());
    fb.old_ns = ns_since(t2);
    r.t.old_exec_ns += fb.old_ns;
    auto t3 = Clock::now();
    const RunOutcome out_new = new_runner->run(region.new_side.insts.size());
    fb.new_ns = ns_since(t3);
    r.t.new_exec_ns += fb.new_ns;
    ++r.executed_pairs;
    r.old_insts_executed += out_old.executed;
    r.new_insts_executed += out_new.executed;
    fb.insts_old = out_old.executed;
    fb.insts_new = out_new.executed;
    fb.exit_old = run_status_name(out_old.status);
    fb.exit_new = run_status_name(out_new.status);

    auto t4 = Clock::now();
    Mismatch m;
    m.case_index = case_index;
    m.phase = phase_name;
    m.out_old = out_old;
    m.out_new = out_new;
    std::string kind;
    auto note_novelty = [&](const MachineState *fo, const MachineState *fn) {
      fb.bucket = novelty_bucket(region.contract, fb.category, fo, fn);
      fb.novel = buckets.insert(fb.bucket).second;
      if (fb.novel)
        ++(std::string_view(phase_name) == "boundary" ? r.novel_boundary : r.novel_random);
    };
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
      fb.compare_ns = ns_since(t4);
      r.t.compare_ns += fb.compare_ns;
      fb.category = kind.empty() ? "excluded" : kind;
      note_novelty(nullptr, nullptr);
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
      fb.compare_ns = ns_since(t4);
      r.t.compare_ns += fb.compare_ns;
      fb.mismatch = !kind.empty();
      fb.category = kind.empty() ? "ok" : kind;
      fb.locs = m.locs;
      note_novelty(&final_old, &final_new);
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
      // Shrink the inputs first (handoff 10), so bundles and traces show the reduced case.
      if (options.reduce && kind == "observable")
        m.reduction =
            reduce_mismatch(region, *old_runner, *new_runner, m, options.reduction_budget);
      if (!options.counterexample_dir.empty()) {
        // Re-execute with tracing on the same runners; determinism makes this the same run.
        std::vector<TraceStep> trace_old, trace_new;
        old_runner->load(m.init_old);
        old_runner->run(region.old_side.insts.size(), &trace_old);
        new_runner->load(m.init_new);
        new_runner->run(region.new_side.insts.size(), &trace_new);
        Json bundle =
            counterexample_json(region, m, options.seed, policy.name(), trace_old, trace_new);
        if (options.artifact_info.has("artifacts"))
          bundle.set("artifacts", options.artifact_info["artifacts"]);
        if (options.artifact_info.has("provenance"))
          bundle.set("provenance", options.artifact_info["provenance"]);
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
    Feedback fb;
    fb.case_index = case_index;
    fb.phase = "boundary";
    fb.events = sample.events;
    auto rejection = ConstraintChecker::validate(sample, region.contract);
    fb.propose_ns = ns_since(t0);
    r.t.propose_ns += fb.propose_ns;
    if (rejection) {
      ++r.rejections["boundary constraint: " + *rejection];
      fb.category = "rejected";
      fb.rejection = *rejection;
    } else {
      run_case(sample, case_index, r.boundary, "boundary", c.family, fb);
    }
    finish(fb);
    ++case_index;
  }

  // Phase 2: random cases from the policy.
  const uint64_t max_attempts = options.max_attempts ? options.max_attempts : 4 * options.cases;
  while (r.random.accepted < options.cases && r.random.cases < max_attempts) {
    auto t0 = Clock::now();
    Sample sample = policy.propose(features, region.contract, history, case_index);
    project_onto_constraints(sample, region.contract, options.seed ^ (case_index << 1));
    ++r.attempted;
    ++r.random.cases;
    Feedback fb;
    fb.case_index = case_index;
    fb.phase = "random";
    fb.events = sample.events;
    auto rejection = ConstraintChecker::validate(sample, region.contract);
    fb.propose_ns = ns_since(t0);
    r.t.propose_ns += fb.propose_ns;
    if (rejection) {
      ++r.rejections["constraint: " + *rejection];
      fb.category = "rejected";
      fb.rejection = *rejection;
    } else {
      run_case(sample, case_index, r.random, "random", "", fb);
    }
    finish(fb);
    ++case_index;
  }
  r.novelty_buckets = buckets.size();
  if (log) {
    Json s = Json::object();
    s.set("type", "summary");
    s.set("cases", history.cases);
    s.set("accepted", history.accepted);
    s.set("valid", history.valid);
    s.set("mismatches", history.mismatches);
    s.set("novelty_buckets", static_cast<uint64_t>(buckets.size()));
    Json cats = Json::object();
    for (const auto &[k, v] : history.categories)
      cats.set(k, v);
    s.set("categories", std::move(cats));
    log << s.dump(-1) << '\n';
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
