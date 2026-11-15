// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "case_io.h"

#include <chrono>
#include <filesystem>

namespace rocjitsu::isa_diff {
namespace {

using Clock = std::chrono::steady_clock;

uint64_t ns_since(Clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

Json artifact_json(const Artifact &a, size_t num_insts) {
  Json j = Json::object();
  j.set("label", a.label);
  j.set("kind", a.kind);
  j.set("sha256", a.sha256);
  j.set("target", a.target);
  j.set("kernel", a.kernel_name);
  j.set("text_vaddr", hex64(a.text_vaddr));
  j.set("kernel_begin", hex64(a.kernel_begin));
  j.set("kernel_end", hex64(a.kernel_end));
  j.set("kernel_bytes", static_cast<uint64_t>(a.kernel_end - a.kernel_begin));
  j.set("reachable_insts", static_cast<uint64_t>(num_insts));
  if (!a.metadata_sha256.empty())
    j.set("metadata_sha256", a.metadata_sha256);
  return j;
}

int contract_rank(ContractStatus s) {
  switch (s) {
  case ContractStatus::Established:
    return 0;
  case ContractStatus::Assumed:
    return 1;
  case ContractStatus::Unresolved:
    return 2;
  }
  return 2;
}

} // namespace

Json CaseData::to_json() const {
  Json j = Json::object();
  j.set("schema_version", kSchemaVersion);
  j.set("identity", identity);
  j.set("command", command);
  j.set("artifacts", artifacts);
  j.set("changes", changes);
  Json un = Json::array();
  for (const auto &u : unresolved) {
    Json uj = Json::object();
    uj.set("kind", u.kind);
    uj.set("what", u.what);
    if (!u.region.empty())
      uj.set("region", u.region);
    un.push(std::move(uj));
  }
  j.set("unresolved", std::move(un));
  Json nj = Json::array();
  for (const auto &n : notes) {
    Json o = Json::object();
    o.set("kind", change_kind_name(n.kind));
    o.set("what", n.what);
    o.set("semantic", n.semantic);
    nj.push(std::move(o));
  }
  j.set("notes", std::move(nj));
  Json rj = Json::array();
  for (const auto &r : regions)
    rj.push(r.to_json());
  j.set("regions", std::move(rj));
  Json vj = Json::array();
  for (const auto &v : variants)
    vj.push(v.to_json());
  j.set("expansion_variants", std::move(vj));
  j.set("analysis", analysis);
  return j;
}

std::optional<CaseData> CaseData::from_json(const Json &j, std::string &error) {
  if (j["schema_version"].as_string() != kSchemaVersion) {
    error = "unsupported case schema '" + j["schema_version"].as_string() + "'";
    return std::nullopt;
  }
  CaseData d;
  d.identity = j["identity"];
  d.command = j["command"];
  d.artifacts = j["artifacts"];
  d.changes = j["changes"];
  for (const Json &u : j["unresolved"].items())
    d.unresolved.push_back({u["kind"].as_string(), u["what"].as_string(), u["region"].as_string()});
  for (const Json &n : j["notes"].items())
    d.notes.push_back({ChangeKind::Layout, n["what"].as_string(), n["semantic"].as_bool()});
  auto read_regions = [&](const Json &list, std::vector<RegionPair> &out) {
    for (const Json &r : list.items()) {
      auto region = RegionPair::from_json(r, error);
      if (!region)
        return false;
      // A hand-edited contract is an assumption, never an established fact.
      if (compute_prepared_key(*region) != region->prepared_key) {
        if (region->contract.status == ContractStatus::Established)
          region->contract.status = ContractStatus::Assumed;
        region->contract.sources.push_back(
            {"region or contract edited after analysis (prepared key changed)", "user_assumption"});
        region->prepared_key = compute_prepared_key(*region);
      }
      out.push_back(std::move(*region));
    }
    return true;
  };
  if (!read_regions(j["regions"], d.regions) || !read_regions(j["expansion_variants"], d.variants))
    return std::nullopt;
  d.analysis = j["analysis"];
  return d;
}

std::optional<CaseData> analyze_artifacts(const Artifact &old_artifact,
                                          const Artifact &new_artifact, const PlanOptions &options,
                                          std::string &error) {
  auto t0 = Clock::now();
  auto old_model = KernelModel::build(old_artifact, error);
  if (!old_model) {
    error = "old: " + error;
    return std::nullopt;
  }
  auto new_model = KernelModel::build(new_artifact, error);
  if (!new_model) {
    error = "new: " + error;
    return std::nullopt;
  }
  const uint64_t decode_ns = ns_since(t0);
  auto t1 = Clock::now();
  const DiffResult diff = diff_kernels(*old_model, *new_model);
  const uint64_t diff_ns = ns_since(t1);
  auto t2 = Clock::now();
  PlanResult plan = plan_regions(*old_model, *new_model, diff, options);
  const uint64_t plan_ns = ns_since(t2);

  CaseData d;
  d.identity = build_identity();
  d.command = Json::object();
  d.artifacts = Json::object();
  d.artifacts.set("old", artifact_json(old_artifact, old_model->insts().size()));
  d.artifacts.set("new", artifact_json(new_artifact, new_model->insts().size()));
  d.changes = Json::array();
  for (size_t h = 0; h < diff.hunks.size(); ++h) {
    const Hunk &hunk = diff.hunks[h];
    Json hj = Json::object();
    hj.set("hunk", static_cast<uint64_t>(h));
    hj.set("kind", change_kind_name(hunk.kind));
    hj.set("old_insts", static_cast<uint64_t>(hunk.old_end - hunk.old_begin));
    hj.set("new_insts", static_cast<uint64_t>(hunk.new_end - hunk.new_begin));
    Json o = Json::array(), n = Json::array();
    for (size_t i = hunk.old_begin; i < hunk.old_end; ++i)
      o.push(hex64(old_model->insts()[i].pc) + "  " + old_model->insts()[i].text);
    for (size_t i = hunk.new_begin; i < hunk.new_end; ++i)
      n.push(hex64(new_model->insts()[i].pc) + "  " + new_model->insts()[i].text);
    hj.set("old", std::move(o));
    hj.set("new", std::move(n));
    if (!hunk.note.empty())
      hj.set("note", hunk.note);
    d.changes.push(std::move(hj));
  }
  d.unresolved = std::move(plan.unresolved);
  d.notes = std::move(plan.notes);
  d.regions = std::move(plan.regions);
  d.variants = std::move(plan.variants);
  d.analysis = Json::object();
  Json opts = Json::object();
  opts.set("max_region_insts", static_cast<uint64_t>(options.max_region_insts));
  opts.set("max_expansions", static_cast<uint64_t>(options.max_expansions));
  opts.set("context_insts", static_cast<uint64_t>(options.context_insts));
  opts.set("merge_same_block", options.merge_same_block);
  d.analysis.set("plan_options", std::move(opts));
  d.analysis.set("alignment_complete", diff.alignment_complete);
  d.analysis.set("layout_shift", diff.layout_shift);
  Json timing = Json::object();
  timing.set("decode_cfg_liveness_ns", decode_ns);
  timing.set("align_classify_ns", diff_ns);
  timing.set("plan_ns", plan_ns);
  d.analysis.set("timing", std::move(timing));
  return d;
}

Aggregate aggregate_results(const CaseData &data, const std::vector<RegionResult> &results) {
  Aggregate a;
  bool harness = false, mismatch = false, all_pass = true, any_tested = false;
  bool all_unsupported = true;
  int worst = -1;
  bool any_witness = false;
  for (const auto &r : results) {
    // A local mismatch that a larger region (with common context) does not reproduce
    // was compensated within the block: the larger region's verdict stands.
    const bool resolved = r.test_status == TestStatus::LocalMismatch && !r.resolved_by.empty();
    const TestStatus status = resolved ? TestStatus::NoMismatchFound : r.test_status;
    const ContractStatus contract = resolved ? r.resolved_contract : r.contract_status;
    all_unsupported &= status == TestStatus::Unsupported;
    harness |= status == TestStatus::HarnessError;
    any_witness |= status == TestStatus::LocalMismatch;
    // A mismatch under an unresolved entry relation may come from the relation itself
    // (wrong mapping, unreachable inputs); it stays a witness but not a verdict.
    mismatch |= status == TestStatus::LocalMismatch && contract != ContractStatus::Unresolved;
    all_pass &= status == TestStatus::NoMismatchFound;
    any_tested |= r.valid_pairs > 0;
    if (r.valid_pairs > 0)
      worst = std::max(worst, contract_rank(contract));
  }
  const bool no_changes =
      data.regions.empty() && data.unresolved.empty() && data.changes.size() == 0;
  if (no_changes) {
    a.test_status = TestStatus::NoSemanticDiff;
    a.contract_status = "NOT_APPLICABLE";
    a.coverage_scope = CoverageScope::CompleteForDeclaredScope;
    return a;
  }
  if (harness)
    a.test_status = TestStatus::HarnessError;
  else if (mismatch)
    a.test_status = TestStatus::LocalMismatch;
  else if (!any_tested && all_unsupported && !results.empty())
    a.test_status = TestStatus::Unsupported;
  else if (all_pass && data.unresolved.empty() && !results.empty())
    a.test_status = TestStatus::NoMismatchFound;
  else
    a.test_status = TestStatus::Inconclusive;
  a.witness_scope = any_witness ? WitnessScope::LocalOnly : WitnessScope::None;
  a.contract_status = worst < 0 ? "NOT_APPLICABLE"
                                : contract_status_name(worst == 0   ? ContractStatus::Established
                                                       : worst == 1 ? ContractStatus::Assumed
                                                                    : ContractStatus::Unresolved);
  bool complete = data.unresolved.empty() && !results.empty();
  for (const auto &r : results) {
    const bool resolved = r.test_status == TestStatus::LocalMismatch && !r.resolved_by.empty();
    complete &= (resolved ? r.resolved_coverage : r.coverage_scope) ==
                CoverageScope::CompleteForDeclaredScope;
  }
  a.coverage_scope = complete     ? CoverageScope::CompleteForDeclaredScope
                     : any_tested ? CoverageScope::Partial
                                  : CoverageScope::NotEstablished;
  return a;
}

Json test_case(const CaseData &data, const TestOptions &options, const std::string &region_filter,
               std::vector<RegionResult> *results_out) {
  std::vector<RegionResult> results;
  for (const auto &region : data.regions) {
    if (!region_filter.empty() && region.id != region_filter)
      continue;
    BaselinePolicy policy(options.seed);
    RegionResult r = test_region(region, options, policy);
    if (r.test_status == TestStatus::LocalMismatch) {
      // Handoff 5.2: a local mismatch may be compensated by common code around the
      // change. Try the precomputed larger regions, smallest first.
      TestOptions vopt = options;
      vopt.max_counterexamples = std::min<size_t>(options.max_counterexamples, 1);
      bool tried = false;
      for (const auto &variant : data.variants) {
        if (variant.parent != region.id || variant.status != RegionStatus::Ready)
          continue;
        tried = true;
        BaselinePolicy vpolicy(options.seed);
        const RegionResult vr = test_region(variant, vopt, vpolicy);
        Json t = Json::object();
        t.set("variant", variant.id);
        t.set("context_before", variant.context_before);
        t.set("context_after", variant.context_after);
        t.set("test_status", test_status_name(vr.test_status));
        t.set("contract_status", contract_status_name(vr.contract_status));
        t.set("valid_pairs", vr.valid_pairs);
        t.set("mismatches", vr.mismatches);
        if (!vr.examples.empty())
          t.set("first_mismatch", vr.examples.front().detail);
        r.expansion_tried.push(std::move(t));
        if (vr.test_status == TestStatus::NoMismatchFound) {
          r.resolved_by = variant.id;
          r.resolved_contract = vr.contract_status;
          r.resolved_coverage = vr.coverage_scope;
          break;
        }
      }
      r.resolution = !r.resolved_by.empty() ? "RESOLVED_BY_CONTEXT"
                     : tried                ? "PERSISTS_WITH_CONTEXT"
                                            : "NO_EXPANSION_AVAILABLE";
    }
    results.push_back(std::move(r));
  }
  // A filtered run cannot speak for the whole change set.
  CaseData scoped = data;
  if (!region_filter.empty())
    scoped.unresolved.push_back({"scope", "region filter '" + region_filter + "' applied", ""});
  const Aggregate agg = aggregate_results(scoped, results);
  Json j = Json::object();
  j.set("schema_version", kSchemaVersion);
  j.set("identity", build_identity());
  j.set("artifacts", data.artifacts);
  Json overall = Json::object();
  overall.set("test_status", test_status_name(agg.test_status));
  overall.set("contract_status", agg.contract_status);
  overall.set("witness_scope", witness_scope_name(agg.witness_scope));
  overall.set("coverage_scope", coverage_scope_name(agg.coverage_scope));
  overall.set("proof_status", "NOT_RUN");
  overall.set("note", "NO_MISMATCH_FOUND means no counterexample under the given seed, budget and "
                      "contract; it is not proven equivalence");
  j.set("overall", std::move(overall));
  Json opts = Json::object();
  opts.set("seed", options.seed);
  opts.set("cases", options.cases);
  opts.set("boundary_sweep", "mandatory");
  opts.set("max_boundary_cases", static_cast<uint64_t>(options.max_boundary_cases));
  opts.set("policy", "baseline-v1");
  j.set("options", std::move(opts));
  Json un = Json::array();
  for (const auto &u : scoped.unresolved) {
    Json uj = Json::object();
    uj.set("kind", u.kind);
    uj.set("what", u.what);
    un.push(std::move(uj));
  }
  j.set("unresolved", std::move(un));
  Json rj = Json::array();
  for (const auto &r : results)
    rj.push(r.to_json());
  j.set("regions", std::move(rj));
  if (results_out)
    *results_out = std::move(results);
  return j;
}

std::optional<CaseData> read_case_dir(const std::string &dir, std::string &error) {
  auto text = read_text_file(dir + "/case.json");
  if (!text) {
    error = "cannot read " + dir + "/case.json";
    return std::nullopt;
  }
  auto json = Json::parse(*text, &error);
  if (!json)
    return std::nullopt;
  return CaseData::from_json(*json, error);
}

bool write_case_dir(const std::string &dir, const CaseData &data, std::string &error) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    error = "cannot create " + dir + ": " + ec.message();
    return false;
  }
  if (!write_text_file(dir + "/case.json", data.to_json().dump())) {
    error = "cannot write " + dir + "/case.json";
    return false;
  }
  return true;
}

} // namespace rocjitsu::isa_diff
