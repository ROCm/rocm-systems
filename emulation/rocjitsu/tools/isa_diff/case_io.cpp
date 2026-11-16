// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "case_io.h"

#include "result_cache.h"

#include <algorithm>
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
  j.set("target", a.target.name);
  j.set("wave_size", static_cast<uint64_t>(a.target.wave_size));
  j.set("kernel", a.kernel_name);
  j.set("text_vaddr", hex64(a.text_vaddr));
  j.set("kernel_begin", hex64(a.kernel_begin));
  j.set("kernel_end", hex64(a.kernel_end));
  j.set("kernel_bytes", static_cast<uint64_t>(a.kernel_end - a.kernel_begin));
  j.set("reachable_insts", static_cast<uint64_t>(num_insts));
  if (!a.metadata_sha256.empty())
    j.set("metadata_sha256", a.metadata_sha256);
  if (a.kind == "code_object") {
    j.set("text_relocations", static_cast<uint64_t>(a.text_relocations.size()));
    j.set("metadata_parsed", a.metadata_parsed);
  }
  if (a.callable) {
    Json f = Json::object();
    f.set("calling_convention", a.calling_convention == "shader" ? "graphics shader part"
                                : a.calling_convention == "gfx"  ? "amdgpu_gfx (CC_SI_Gfx)"
                                                                 : "AMDGPU C (CC_AMDGPU_Func)");
    f.set("return", a.calling_convention == "shader" ? "s_endpgm (to the shader epilog)"
                                                     : "s_setpc_b64 s[30:31]");
    if (a.return_sgprs)
      f.set("return_sgprs", static_cast<uint64_t>(a.return_sgprs));
    f.set("return_vgprs", static_cast<uint64_t>(a.return_vgprs));
    f.set("entry_mode", hex32(a.function_mode));
    if (!a.return_masks.empty()) {
      Json masks = Json::array();
      for (uint32_t m : a.return_masks)
        masks.push(hex32(m));
      f.set("return_masks", std::move(masks));
    }
    j.set("callable_function", std::move(f));
  }
  j.set("entry_vgprs", static_cast<int64_t>(a.entry_vgprs));
  j.set("entry_sgprs", static_cast<int64_t>(a.entry_sgprs));
  if (!a.inline_asm.empty()) {
    Json blocks = Json::array();
    for (const InlineAsmBlock &b : a.inline_asm) {
      Json bj = Json::object();
      bj.set("offset", hex64(b.offset));
      bj.set("text", b.text);
      if (b.before_label)
        bj.set("before_label", true);
      blocks.push(std::move(bj));
    }
    j.set("inline_asm", std::move(blocks));
  }
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
  if (!provenance.is_null())
    j.set("provenance", provenance);
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
  if (!diagnostics.empty()) {
    Json dj = Json::array();
    for (const auto &d : diagnostics)
      dj.push(d.to_json());
    j.set("diagnostic_regions", std::move(dj));
  }
  if (whole_function)
    j.set("whole_function", whole_function->to_json());
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
  d.provenance = j["provenance"];
  d.changes = j["changes"];
  for (const Json &u : j["unresolved"].items())
    d.unresolved.push_back({u["kind"].as_string(), u["what"].as_string(), u["region"].as_string()});
  for (const Json &n : j["notes"].items()) {
    ChangeKind kind = ChangeKind::Unexplained;
    for (auto k : {ChangeKind::Instruction, ChangeKind::ControlFlow, ChangeKind::Layout,
                   ChangeKind::Metadata, ChangeKind::Unexplained, ChangeKind::Hint})
      if (n["kind"].as_string() == change_kind_name(k))
        kind = k;
    d.notes.push_back({kind, n["what"].as_string(), n["semantic"].as_bool()});
  }
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
  if (!read_regions(j["regions"], d.regions) ||
      !read_regions(j["expansion_variants"], d.variants) ||
      !read_regions(j["diagnostic_regions"], d.diagnostics))
    return std::nullopt;
  if (j["whole_function"].is_object()) {
    Json one = Json::array();
    one.push(j["whole_function"]);
    std::vector<RegionPair> f;
    if (!read_regions(one, f))
      return std::nullopt;
    d.whole_function = std::move(f.front());
  }
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
  PlanResult plan;
  if (!options.manual_region.empty()) {
    auto manual = parse_manual_region(options.manual_region, *old_model, *new_model, error);
    if (!manual)
      return std::nullopt;
    if (!options.register_map.empty() &&
        (!options.entry_map.empty() || !options.exit_map.empty())) {
      error = "--map sets both maps; use either --map or --entry-map/--exit-map";
      return std::nullopt;
    }
    for (auto [text, out] : {std::pair{&options.entry_map, &manual->entry_map},
                             std::pair{&options.exit_map, &manual->exit_map}}) {
      const std::string &spec = text->empty() ? options.register_map : *text;
      if (spec.empty())
        continue;
      auto map = parse_register_map(spec, error);
      if (!map)
        return std::nullopt;
      *out = std::move(*map);
    }
    plan = plan_manual_region(*old_model, *new_model, diff, *manual, options);
  } else if (!options.register_map.empty() || !options.entry_map.empty() ||
             !options.exit_map.empty()) {
    error = "a register map applies only to a manual region (--manual-region)";
    return std::nullopt;
  } else {
    plan = plan_regions(*old_model, *new_model, diff, options);
  }
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
  d.diagnostics = std::move(plan.diagnostics);
  d.whole_function = std::move(plan.whole_function);
  d.analysis = Json::object();
  Json opts = Json::object();
  opts.set("max_region_insts", static_cast<uint64_t>(options.max_region_insts));
  opts.set("max_expansions", static_cast<uint64_t>(options.max_expansions));
  opts.set("context_insts", static_cast<uint64_t>(options.context_insts));
  opts.set("merge_same_block", options.merge_same_block);
  opts.set("derive_constraints", options.derive_constraints);
  opts.set("max_analysis_us", options.max_analysis_us);
  opts.set("diagnostic_entry_prefix", options.diagnostic_entry_prefix);
  if (!options.manual_region.empty())
    opts.set("manual_region", options.manual_region);
  if (!options.register_map.empty())
    opts.set("register_map", options.register_map);
  if (!options.entry_map.empty())
    opts.set("entry_map", options.entry_map);
  if (!options.exit_map.empty())
    opts.set("exit_map", options.exit_map);
  d.analysis.set("plan_options", std::move(opts));
  if (!plan.skipped_for_time.empty()) {
    Json skipped = Json::array();
    for (const auto &s : plan.skipped_for_time)
      skipped.push(s);
    d.analysis.set("skipped_for_time", std::move(skipped));
  }
  d.analysis.set("alignment_complete", diff.alignment_complete);
  d.analysis.set("layout_shift", diff.layout_shift);
  // Handoff 11 cost split. compile_ns is added by the CLI when the LLVM plugin ran.
  Json timing = Json::object();
  timing.set("parse_ns", old_artifact.load_ns + new_artifact.load_ns);
  timing.set("decode_cfg_ns", old_model->decode_cfg_ns() + new_model->decode_cfg_ns());
  timing.set("dataflow_ns", old_model->dataflow_ns() + new_model->dataflow_ns());
  timing.set("model_total_ns", decode_ns);
  timing.set("align_classify_ns", diff_ns);
  timing.set("plan_ns", plan_ns);
  d.analysis.set("timing", std::move(timing));
  return d;
}

std::optional<CaseData> analyze_cached(const Artifact &old_artifact, const Artifact &new_artifact,
                                       const PlanOptions &options, const std::string &cache_dir,
                                       bool &hit, std::string &error) {
  hit = false;
  if (cache_dir.empty() || build_fingerprint() == "unavailable")
    return analyze_artifacts(old_artifact, new_artifact, options, error);
  const std::string key = analysis_cache_key(old_artifact, new_artifact, options);
  const std::string dir = cache_dir + "/analysis/" + key;
  std::string ignored;
  if (auto cached = read_case_dir(dir, ignored)) {
    hit = true;
    // Same bytes, possibly another path: keep this run's labels.
    for (const auto *a : {&old_artifact, &new_artifact}) {
      const char *side = a == &old_artifact ? "old" : "new";
      Json aj = cached->artifacts[side];
      aj.set("label", a->label);
      cached->artifacts.set(side, std::move(aj));
    }
    Json c = Json::object();
    c.set("hit", true);
    c.set("key", key);
    c.set("build_fingerprint", build_fingerprint());
    cached->analysis.set("cache", std::move(c));
    return cached;
  }
  auto data = analyze_artifacts(old_artifact, new_artifact, options, error);
  // A time-limited plan depends on machine load: never cached.
  if (data && !data->analysis.has("skipped_for_time")) {
    std::string werr;
    if (!write_case_dir(dir, *data, werr))
      data->analysis.set("cache_store_error", werr);
  }
  return data;
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
    mismatch |= status == TestStatus::LocalMismatch && contract != ContractStatus::Unresolved &&
                !r.compensable;
    all_pass &= status == TestStatus::NoMismatchFound;
    any_tested |= r.valid_pairs > 0;
    if (r.valid_pairs > 0)
      worst = std::max(worst, contract_rank(contract));
  }
  // Hint hunks (scheduling hints, dependency waits) alone are no semantic difference.
  bool only_hints = true;
  for (const Json &c : data.changes.items())
    only_hints = only_hints && c["kind"].as_string() == "hint";
  const bool no_changes = data.regions.empty() && data.unresolved.empty() && only_hints;
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
  // Counterexample bundles carry the artifacts and their provenance (handoff 10).
  TestOptions with_info = options;
  with_info.artifact_info = Json::object();
  with_info.artifact_info.set("artifacts", data.artifacts);
  if (!data.provenance.is_null())
    with_info.artifact_info.set("provenance", data.provenance);
  const TestOptions &topt = with_info;
  uint64_t cache_hits = 0, cache_misses = 0;
  for (const auto &region : data.regions) {
    if (!region_filter.empty() && region.id != region_filter)
      continue;
    std::string cache_key;
    if (!topt.cache_dir.empty()) {
      cache_key = result_cache_key(region, data.variants, topt, "baseline-v1");
      if (auto cached = load_cached_result(topt.cache_dir, cache_key)) {
        ++cache_hits;
        results.push_back(std::move(*cached));
        continue;
      }
      ++cache_misses;
    }
    BaselinePolicy policy(topt.seed);
    RegionResult r = test_region(region, topt, policy);
    // Handoff 3.3/13: with constraints derived from the common prefix, also report the
    // unrestricted verdict so that a constraint-dependent rewrite is visible as such.
    const bool derived = std::ranges::any_of(region.contract.constraints, [](const Constraint &c) {
      return c.source == "common_prefix_analysis";
    });
    if (derived && region.status == RegionStatus::Ready) {
      RegionPair open = region;
      std::erase_if(open.contract.constraints,
                    [](const Constraint &c) { return c.source == "common_prefix_analysis"; });
      open.id = region.id + ".unrestricted";
      open.prepared_key = compute_prepared_key(open);
      TestOptions uopt = topt;
      uopt.max_counterexamples = 1;
      uopt.case_log_dir.clear();
      BaselinePolicy upolicy(topt.seed);
      const RegionResult ur = test_region(open, uopt, upolicy);
      Json u = Json::object();
      u.set("test_status", test_status_name(ur.test_status));
      u.set("valid_pairs", ur.valid_pairs);
      u.set("mismatches", ur.mismatches);
      if (!ur.examples.empty())
        u.set("first_mismatch", ur.examples.front().detail);
      u.set("constraint_dependent", ur.test_status == TestStatus::LocalMismatch &&
                                        r.test_status == TestStatus::NoMismatchFound);
      u.set("note", "same region without the constraints derived from the common prefix; the "
                    "constrained result above is the verdict");
      r.unrestricted = std::move(u);
    }
    if (r.test_status == TestStatus::LocalMismatch) {
      // Handoff 5.2: a local mismatch may be compensated by common code around the
      // change. Try the precomputed larger regions, smallest first.
      TestOptions vopt = topt;
      vopt.max_counterexamples = std::min<size_t>(topt.max_counterexamples, 1);
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
    if (!cache_key.empty())
      (void)store_cached_result(topt.cache_dir, cache_key, r);
    results.push_back(std::move(r));
  }
  // A local mismatch that changed code downstream may compensate is a witness, not a
  // verdict (handoff 5.2/13): every mismatching location is read by that code.
  std::vector<UnresolvedItem> compensable;
  for (auto &r : results) {
    if (r.test_status != TestStatus::LocalMismatch || !r.resolved_by.empty())
      continue;
    for (const auto &region : data.regions) {
      if (region.id != r.region_id)
        continue;
      const RegionContract &k = region.contract;
      const bool any = r.mismatch_locs.any() || r.mismatch_vlocs.any() || r.mismatch_memory;
      if (any && (r.mismatch_locs & ~k.downstream_reads).none() &&
          (r.mismatch_vlocs & ~k.vdownstream_reads).none() &&
          (!r.mismatch_memory || k.downstream_memory_reads) && !k.downstream_changed.empty()) {
        r.compensable = true;
        std::string ids;
        for (const auto &d : k.downstream_changed)
          ids += (ids.empty() ? "" : ", ") + d;
        std::string locs = format_locs(r.mismatch_locs);
        if (r.mismatch_vlocs.any())
          locs += (locs.empty() ? "" : ",") + format_vgprs(r.mismatch_vlocs);
        compensable.push_back({"context",
                               "local mismatch of " + r.region_id + " on {" + locs +
                                   "} may be compensated by changed code downstream (" + ids + ")",
                               r.region_id});
        if (!r.cached.is_null())
          r.cached.set("compensable_downstream", true);
      }
    }
  }
  // A region whose entry relation relies on an upstream region's R_out (depends_on) has
  // no justified entry relation when that region itself shows an unresolved mismatch:
  // its mismatch is a witness, not a verdict (transitively).
  for (bool changed = true; changed;) {
    changed = false;
    for (auto &r : results) {
      if (r.test_status != TestStatus::LocalMismatch || !r.resolved_by.empty() || r.compensable)
        continue;
      for (const auto &region : data.regions) {
        if (region.id != r.region_id)
          continue;
        for (const auto &up : region.contract.depends_on)
          for (const auto &u : results)
            if (u.region_id == up && u.test_status == TestStatus::LocalMismatch &&
                u.resolved_by.empty() && !r.compensable) {
              r.compensable = true;
              changed = true;
              compensable.push_back({"context",
                                     "local mismatch of " + r.region_id +
                                         " relies on the R_out of " + up +
                                         ", which itself differs: its entry relation is not "
                                         "justified",
                                     r.region_id});
            }
      }
    }
  }
  // Handoff 5.3: diagnostic regions were requested explicitly at analysis time. They run
  // after the verdict regions and are reported apart from them.
  Json diagnostics = Json::array();
  for (const auto &dr : data.diagnostics) {
    if (!region_filter.empty() && dr.parent != region_filter)
      continue;
    TestOptions dopt = topt;
    dopt.max_counterexamples = std::min<size_t>(topt.max_counterexamples, 1);
    dopt.case_log_dir.clear();
    BaselinePolicy dpolicy(topt.seed);
    const RegionResult dres = test_region(dr, dopt, dpolicy);
    Json dj = dres.to_json();
    dj.set("diagnostic", "entry_prefix");
    dj.set("parent", dr.parent);
    diagnostics.push(std::move(dj));
  }
  // A filtered run cannot speak for the whole change set.
  CaseData scoped = data;
  scoped.unresolved.insert(scoped.unresolved.end(), compensable.begin(), compensable.end());
  if (!region_filter.empty())
    scoped.unresolved.push_back({"scope", "region filter '" + region_filter + "' applied", ""});
  Aggregate agg = aggregate_results(scoped, results);
  // The whole function settles what the regions leave open (composition, exit mapping,
  // compensation, control flow): run it unless the regions already give a verdict.
  // Unresolved items about the artifact (metadata, layout, relocations, unknown code) are
  // outside it and keep a pass from being declared.
  Json whole = Json::object();
  bool resolved_by_function = false;
  const bool open = agg.test_status == TestStatus::Inconclusive ||
                    agg.test_status == TestStatus::Unsupported ||
                    (agg.test_status == TestStatus::NoMismatchFound &&
                     agg.coverage_scope != CoverageScope::CompleteForDeclaredScope);
  if (data.whole_function && region_filter.empty() && open) {
    const RegionPair &f = *data.whole_function;
    std::string blocking;
    for (const auto &u : scoped.unresolved)
      if (u.kind != "region" && u.kind != "context" && u.kind != "instruction" &&
          u.kind != "control_flow" && blocking.empty())
        blocking = u.kind + ": " + u.what;
    if (f.status != RegionStatus::Ready) {
      whole.set("decides", false);
      whole.set("reason", f.reason);
    } else {
      BaselinePolicy fpolicy(topt.seed);
      const RegionResult fr = test_region(f, topt, fpolicy);
      const bool established = fr.contract_status == ContractStatus::Established;
      // A pass needs enough valid cases (loops past the step limit and inputs that fault
      // on both sides are excluded); a mismatch needs one witness.
      constexpr uint64_t kMinValid = 100;
      const bool pass = fr.test_status == TestStatus::NoMismatchFound && established &&
                        fr.coverage_scope == CoverageScope::CompleteForDeclaredScope &&
                        blocking.empty() && fr.valid_pairs >= kMinValid;
      const bool mismatch = fr.test_status == TestStatus::LocalMismatch && established;
      whole.set("result", fr.to_json());
      whole.set("decides", pass || mismatch);
      if (pass) {
        agg.test_status = TestStatus::NoMismatchFound;
        agg.coverage_scope = CoverageScope::CompleteForDeclaredScope;
        agg.contract_status = contract_status_name(ContractStatus::Established);
        whole.set("reason", "the whole function passes: it covers every change of the "
                            "function from its entry to its return");
      } else if (mismatch) {
        agg.test_status = TestStatus::LocalMismatch;
        agg.witness_scope = WitnessScope::LocalOnly;
        agg.contract_status = contract_status_name(ContractStatus::Established);
        whole.set("reason", "the whole function differs at its return, from an entry state "
                            "both sides receive alike");
      } else {
        whole.set("reason",
                  !blocking.empty() ? "the whole function cannot decide: " + blocking
                  : fr.test_status == TestStatus::NoMismatchFound && fr.valid_pairs < kMinValid
                      ? "the whole function passes on only " + std::to_string(fr.valid_pairs) +
                            " valid cases (fewer than " + std::to_string(kMinValid) +
                            "): not decisive"
                      : std::string("the whole function is ") + test_status_name(fr.test_status) +
                            " (contract " + contract_status_name(fr.contract_status) + ")");
      }
      resolved_by_function = pass || mismatch;
      if (resolved_by_function)
        results.push_back(fr);
    }
  }
  Json j = Json::object();
  j.set("schema_version", kSchemaVersion);
  j.set("identity", build_identity());
  j.set("artifacts", data.artifacts);
  if (!data.provenance.is_null())
    j.set("provenance", data.provenance);
  Json overall = Json::object();
  overall.set("test_status", test_status_name(agg.test_status));
  // Passes that needed the per-bit don't-care mask are not plain passes (README "NaNs,
  // undefined values and poison").
  uint64_t masked = 0;
  for (const auto &r : results)
    masked += r.to_json()["counts"]["dont_care_masked"].as_uint();
  overall.set("dont_care_masked_cases", masked);
  if (resolved_by_function)
    overall.set("resolved_by", "whole_function");
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
  if (whole.has("decides"))
    j.set("whole_function", std::move(whole));
  if (!data.diagnostics.empty()) {
    j.set("diagnostics", std::move(diagnostics));
    j.set("diagnostics_note", "explicitly enabled diagnostic path (entry prefix executed with "
                              "arbitrary entry values); not part of the overall verdict");
  }
  if (!topt.cache_dir.empty()) {
    Json c = Json::object();
    c.set("dir", topt.cache_dir);
    c.set("hits", cache_hits);
    c.set("misses", cache_misses);
    c.set("build_fingerprint", build_fingerprint());
    j.set("cache", std::move(c));
  }
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
