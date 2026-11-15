// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "result_cache.h"

#include "util/sha256.h"

#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

namespace rocjitsu::isa_diff {
namespace {

namespace fs = std::filesystem;

std::string compute_fingerprint() {
  std::set<std::string> files;
  std::error_code ec;
  const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
  if (ec)
    return "unavailable";
  files.insert(exe.string());
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    // address perms offset dev inode path
    std::istringstream in(line);
    std::string addr, perms, offset, dev, inode, path;
    in >> addr >> perms >> offset >> dev >> inode;
    std::getline(in >> std::ws, path);
    if (perms.size() >= 3 && perms[2] == 'x' && !path.empty() && path[0] == '/')
      files.insert(path);
  }
  util::Sha256 sha;
  std::vector<char> buf(1 << 20);
  for (const auto &f : files) {
    sha.update(reinterpret_cast<const uint8_t *>(f.data()), f.size() + 1);
    std::ifstream in(f, std::ios::binary);
    if (!in)
      return "unavailable";
    while (in.read(buf.data(), static_cast<std::streamsize>(buf.size())) || in.gcount() > 0)
      sha.update(reinterpret_cast<const uint8_t *>(buf.data()), static_cast<size_t>(in.gcount()));
  }
  const auto digest = sha.finish();
  return sha256_hex(std::span<const uint8_t>(digest.data(), digest.size()));
}

template <typename E, size_t N>
std::optional<E> parse_enum(const std::string &s, const E (&values)[N], const char *(*name)(E)) {
  for (E v : values)
    if (s == name(v))
      return v;
  return std::nullopt;
}

std::string entry_dir(const std::string &cache_dir, const std::string &key) {
  return cache_dir + "/results/" + key;
}

} // namespace

const std::string &build_fingerprint() {
  static const std::string fp = compute_fingerprint();
  return fp;
}

std::string result_cache_key(const RegionPair &region, const std::vector<RegionPair> &variants,
                             const TestOptions &options, const std::string &policy) {
  Json k = Json::object();
  k.set("layer", "rj_isa_diff.result_cache.v1");
  k.set("tool", kToolVersion);
  k.set("schema", kSchemaVersion);
  k.set("build", build_fingerprint());
  k.set("prepared_key", compute_prepared_key(region));
  Json vs = Json::array();
  for (const auto &v : variants)
    if (v.parent == region.id)
      vs.push(compute_prepared_key(v));
  k.set("variants", std::move(vs));
  k.set("policy", policy);
  k.set("seed", options.seed);
  k.set("cases", options.cases);
  k.set("max_attempts", options.max_attempts);
  k.set("max_boundary_cases", static_cast<uint64_t>(options.max_boundary_cases));
  k.set("max_counterexamples", static_cast<uint64_t>(options.max_counterexamples));
  k.set("reduce", options.reduce);
  k.set("reduction_budget", static_cast<uint64_t>(options.reduction_budget));
  k.set("case_log", !options.case_log_dir.empty());
  return sha256_hex(k.dump(-1));
}

std::string analysis_cache_key(const Artifact &old_artifact, const Artifact &new_artifact,
                               const PlanOptions &options) {
  Json k = Json::object();
  k.set("layer", "rj_isa_diff.analysis_cache.v1");
  k.set("tool", kToolVersion);
  k.set("schema", kSchemaVersion);
  k.set("build", build_fingerprint());
  for (const auto *a : {&old_artifact, &new_artifact}) {
    Json aj = Json::object();
    aj.set("sha256", a->sha256);
    aj.set("kind", a->kind);
    aj.set("kernel", a->kernel_name);
    aj.set("text_vaddr", a->text_vaddr);
    k.set(a == &old_artifact ? "old" : "new", std::move(aj));
  }
  Json o = Json::object();
  o.set("max_region_insts", static_cast<uint64_t>(options.max_region_insts));
  o.set("max_expansions", static_cast<uint64_t>(options.max_expansions));
  o.set("context_insts", static_cast<uint64_t>(options.context_insts));
  o.set("merge_same_block", options.merge_same_block);
  o.set("auto_expand", options.auto_expand);
  o.set("max_variants", static_cast<uint64_t>(options.max_variants));
  o.set("derive_constraints", options.derive_constraints);
  o.set("manual_region", options.manual_region);
  o.set("register_map", options.register_map);
  o.set("entry_map", options.entry_map);
  o.set("exit_map", options.exit_map);
  o.set("diagnostic_entry_prefix", options.diagnostic_entry_prefix);
  k.set("plan", std::move(o));
  return sha256_hex(k.dump(-1));
}

std::optional<RegionResult> load_cached_result(const std::string &cache_dir,
                                               const std::string &key) {
  if (build_fingerprint() == "unavailable")
    return std::nullopt;
  const std::string dir = entry_dir(cache_dir, key);
  auto text = read_text_file(dir + "/result.json");
  if (!text)
    return std::nullopt;
  std::string error;
  auto j = Json::parse(*text, &error);
  if (!j || !j->is_object())
    return std::nullopt;
  static constexpr TestStatus kTest[] = {TestStatus::NoMismatchFound, TestStatus::LocalMismatch,
                                         TestStatus::Inconclusive,    TestStatus::Unsupported,
                                         TestStatus::HarnessError,    TestStatus::NoSemanticDiff};
  static constexpr ContractStatus kContract[] = {
      ContractStatus::Established, ContractStatus::Assumed, ContractStatus::Unresolved};
  static constexpr WitnessScope kWitness[] = {WitnessScope::None, WitnessScope::LocalOnly,
                                              WitnessScope::ReachableAtCut,
                                              WitnessScope::ObservableKernelDivergence};
  static constexpr CoverageScope kCoverage[] = {CoverageScope::CompleteForDeclaredScope,
                                                CoverageScope::Partial,
                                                CoverageScope::NotEstablished};
  const Json &rj = *j;
  auto ts = parse_enum(rj["test_status"].as_string(), kTest, test_status_name);
  auto cs = parse_enum(rj["contract_status"].as_string(), kContract, contract_status_name);
  auto ws = parse_enum(rj["witness_scope"].as_string(), kWitness, witness_scope_name);
  auto cov = parse_enum(rj["coverage_scope"].as_string(), kCoverage, coverage_scope_name);
  if (!ts || !cs || !ws || !cov)
    return std::nullopt;
  RegionResult r;
  r.region_id = rj["region"].as_string();
  r.synthetic = rj["synthetic"].as_bool();
  r.test_status = *ts;
  r.contract_status = *cs;
  r.witness_scope = *ws;
  r.coverage_scope = *cov;
  r.reason = rj["reason"].as_string();
  r.prepared_key = rj["prepared_key"].as_string();
  const Json &counts = rj["counts"];
  r.attempted = counts["attempted"].as_uint();
  r.accepted = counts["accepted"].as_uint();
  r.executed_pairs = counts["executed_pairs"].as_uint();
  r.valid_pairs = counts["valid_pairs"].as_uint();
  r.mismatches = counts["mismatches"].as_uint();
  r.harness_errors = counts["harness_errors"].as_uint();
  auto phase = [](const Json &p) {
    return PhaseCounts{p["cases"].as_uint(), p["accepted"].as_uint(), p["valid"].as_uint(),
                       p["mismatches"].as_uint()};
  };
  r.boundary = phase(rj["boundary_sweep"]);
  r.random = phase(rj["random_phase"]);
  r.boundary_truncated = rj["boundary_sweep"]["truncated"].as_bool();
  for (const auto &[family, n] : rj["boundary_sweep"]["valid_by_family"].members())
    r.boundary_families[family] = n.as_uint();
  r.novelty_buckets = rj["novelty"]["distinct_buckets"].as_uint();
  r.novel_boundary = rj["novelty"]["first_seen_in_boundary"].as_uint();
  r.novel_random = rj["novelty"]["first_seen_in_random"].as_uint();
  r.old_len = rj["execution"]["old_region_insts"].as_uint();
  r.new_len = rj["execution"]["new_region_insts"].as_uint();
  r.old_insts_executed = rj["execution"]["old_insts_executed"].as_uint();
  r.new_insts_executed = rj["execution"]["new_insts_executed"].as_uint();
  r.policy = rj["policy"].as_string();
  if (auto ml = parse_locs(rj["mismatch_locations"].as_string()))
    r.mismatch_locs = *ml;
  if (auto mv = parse_vgprs(rj["vgpr_mismatch_locations"].as_string()))
    r.mismatch_vlocs = *mv;
  r.seed = rj["seed"].as_uint();
  const Json &e = rj["expansion"];
  if (e.is_object()) {
    r.resolution = e["resolution"].as_string();
    r.resolved_by = e["resolved_by"].as_string();
    if (auto rc =
            parse_enum(e["resolved_contract_status"].as_string(), kContract, contract_status_name))
      r.resolved_contract = *rc;
    if (auto rcov =
            parse_enum(e["resolved_coverage_scope"].as_string(), kCoverage, coverage_scope_name))
      r.resolved_coverage = *rcov;
  }
  // Counterexamples point at the copies inside the cache entry.
  Json out = rj;
  Json files = Json::array();
  for (const Json &f : rj["counterexamples"].items()) {
    const std::string copy = dir + "/" + fs::path(f.as_string()).filename().string();
    if (!fs::exists(copy))
      return std::nullopt; // incomplete entry: rerun
    r.counterexample_files.push_back(copy);
    files.push(copy);
  }
  out.set("counterexamples", std::move(files));
  Json c = Json::object();
  c.set("hit", true);
  c.set("key", key);
  c.set("build_fingerprint", build_fingerprint());
  c.set("note", "result reused from the cache: same prepared region and contract, variants, "
                "policy, seed, budgets and build; timings and case_log are from the original run");
  out.set("cache", std::move(c));
  r.cached = std::move(out);
  return r;
}

bool store_cached_result(const std::string &cache_dir, const std::string &key,
                         const RegionResult &result) {
  if (build_fingerprint() == "unavailable" || !result.cached.is_null())
    return false;
  // Harness errors describe this run's environment, not the region: never cached.
  if (result.test_status == TestStatus::HarnessError)
    return false;
  const std::string dir = entry_dir(cache_dir, key);
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec)
    return false;
  for (const auto &f : result.counterexample_files) {
    fs::copy_file(f, dir + "/" + fs::path(f).filename().string(),
                  fs::copy_options::overwrite_existing, ec);
    if (ec)
      return false;
  }
  // Write then rename, so a concurrent reader never sees a partial entry.
  const std::string tmp = dir + "/result.json.tmp." + std::to_string(::getpid());
  if (!write_text_file(tmp, result.to_json().dump()))
    return false;
  fs::rename(tmp, dir + "/result.json", ec);
  return !ec;
}

} // namespace rocjitsu::isa_diff
