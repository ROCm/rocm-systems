// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_isa_diff_main.cpp
/// @brief CLI: rj_isa_diff analyze | test | replay. See tools/isa_diff/README.md.

#include "case_io.h"
#include "mutation.h"

#include <charconv>
#include <cstdio>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace rocjitsu::isa_diff;

constexpr int kExitPass = 0;
constexpr int kExitMismatch = 1;
constexpr int kExitInconclusive = 2;
constexpr int kExitError = 3;

const char *kUsage =
    "usage:\n"
    "  rj_isa_diff analyze --old FILE --new FILE --out DIR [--kernel NAME]\n"
    "                      [--old-kernel NAME] [--new-kernel NAME] [--context N]\n"
    "                      [--max-region N] [--max-expansions N] [--no-merge]\n"
    "  rj_isa_diff test    --case DIR [--seed N] [--cases N] [--region ID]\n"
    "                      [--max-counterexamples N] [--max-boundary-cases N]\n"
    "      --cases counts random cases; the boundary/NaN/corner sweep always runs first.\n"
    "  rj_isa_diff replay  --counterexample FILE [--case DIR]\n"
    "  rj_isa_diff inject  --case DIR --region ID --xor MASK --out DIR [--side old|new]\n"
    "                      [--inst N] [--word N] [--label NAME]   (synthetic self-test)\n"
    "FILE is a gfx950 AMDGPU code object or a raw little-endian instruction stream.\n"
    "exit: 0 no mismatch found with complete declared scope, 1 local mismatch,\n"
    "      2 inconclusive/unsupported/partial scope, 3 error.\n";

struct Args {
  std::map<std::string, std::string> values;
  std::vector<std::string> switches;
  std::string error;

  bool has(const std::string &k) const { return values.count(k) || is_switch(k); }
  bool is_switch(const std::string &k) const {
    for (const auto &s : switches)
      if (s == k)
        return true;
    return false;
  }
  std::string get(const std::string &k, const std::string &fallback = "") const {
    auto it = values.find(k);
    return it == values.end() ? fallback : it->second;
  }
  std::optional<uint64_t> number(const std::string &k, uint64_t fallback) const {
    auto it = values.find(k);
    if (it == values.end())
      return fallback;
    uint64_t v = 0;
    const std::string &s = it->second;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || ptr != s.data() + s.size())
      return std::nullopt;
    return v;
  }
};

Args parse(int argc, char **argv, int first, const std::vector<std::string> &value_flags,
           const std::vector<std::string> &switch_flags) {
  Args a;
  for (int i = first; i < argc; ++i) {
    std::string arg = argv[i];
    bool known = false;
    for (const auto &f : switch_flags)
      if (arg == f) {
        a.switches.push_back(f);
        known = true;
      }
    if (known)
      continue;
    std::string value;
    std::string key = arg;
    if (auto eq = arg.find('='); eq != std::string::npos) {
      key = arg.substr(0, eq);
      value = arg.substr(eq + 1);
    }
    for (const auto &f : value_flags)
      if (key == f) {
        known = true;
        if (value.empty() && arg.find('=') == std::string::npos) {
          if (i + 1 >= argc) {
            a.error = "missing value for " + f;
            return a;
          }
          value = argv[++i];
        }
        a.values[f] = value;
      }
    if (!known) {
      a.error = "unknown argument '" + arg + "'";
      return a;
    }
  }
  return a;
}

std::string command_line(int argc, char **argv) {
  std::string out;
  for (int i = 0; i < argc; ++i) {
    if (i)
      out += ' ';
    out += argv[i];
  }
  return out;
}

int exit_code_for(const Json &overall) {
  const std::string &status = overall["test_status"].as_string();
  const std::string &coverage = overall["coverage_scope"].as_string();
  if (status == "LOCAL_MISMATCH")
    return kExitMismatch;
  if (status == "HARNESS_ERROR")
    return kExitError;
  if ((status == "NO_MISMATCH_FOUND" || status == "NO_SEMANTIC_DIFF") &&
      coverage == "COMPLETE_FOR_DECLARED_SCOPE")
    return kExitPass;
  return kExitInconclusive;
}

int cmd_analyze(int argc, char **argv) {
  Args a = parse(argc, argv, 2,
                 {"--old", "--new", "--out", "--kernel", "--old-kernel", "--new-kernel",
                  "--context", "--max-region", "--max-expansions"},
                 {"--no-merge"});
  if (!a.error.empty() || !a.has("--old") || !a.has("--new") || !a.has("--out")) {
    std::cerr << (a.error.empty() ? "analyze needs --old, --new and --out" : a.error) << "\n"
              << kUsage;
    return kExitError;
  }
  PlanOptions plan;
  auto context = a.number("--context", 0);
  auto max_region = a.number("--max-region", plan.max_region_insts);
  auto max_exp = a.number("--max-expansions", plan.max_expansions);
  if (!context || !max_region || !max_exp) {
    std::cerr << "bad numeric option\n";
    return kExitError;
  }
  plan.context_insts = *context;
  plan.max_region_insts = *max_region;
  plan.max_expansions = *max_exp;
  plan.merge_same_block = !a.is_switch("--no-merge");
  std::string error;
  Artifact old_a, new_a;
  const std::string kernel = a.get("--kernel");
  if (!load_artifact(a.get("--old"), a.get("--old-kernel", kernel), old_a, error)) {
    std::cerr << "old: " << error << "\n";
    return kExitError;
  }
  if (!load_artifact(a.get("--new"), a.get("--new-kernel", kernel), new_a, error)) {
    std::cerr << "new: " << error << "\n";
    return kExitError;
  }
  auto data = analyze_artifacts(old_a, new_a, plan, error);
  if (!data) {
    std::cerr << error << "\n";
    return kExitError;
  }
  data->command.set("argv", command_line(argc, argv));
  if (!write_case_dir(a.get("--out"), *data, error)) {
    std::cerr << error << "\n";
    return kExitError;
  }
  std::cout << "kernel: old " << old_a.kernel_name << " ("
            << data->artifacts["old"]["reachable_insts"].as_uint() << " insts), new "
            << new_a.kernel_name << " (" << data->artifacts["new"]["reachable_insts"].as_uint()
            << " insts)\n";
  std::cout << "changes: " << data->changes.size() << " hunk(s), " << data->regions.size()
            << " region(s), " << data->unresolved.size() << " unresolved item(s)\n";
  for (const auto &r : data->regions)
    std::cout << "  " << r.id << " " << region_status_name(r.status) << " old "
              << r.old_side.insts.size() << " / new " << r.new_side.insts.size()
              << " insts, contract " << contract_status_name(r.contract.status)
              << (r.reason.empty() ? "" : " : " + r.reason) << "\n";
  for (const auto &u : data->unresolved)
    std::cout << "  unresolved [" << u.kind << "] " << u.what << "\n";
  std::cout << "wrote " << a.get("--out") << "/case.json\n";
  return kExitPass;
}

int cmd_test(int argc, char **argv) {
  Args a = parse(
      argc, argv, 2,
      {"--case", "--seed", "--cases", "--region", "--max-counterexamples", "--max-boundary-cases"},
      {});
  if (!a.error.empty() || !a.has("--case")) {
    std::cerr << (a.error.empty() ? "test needs --case" : a.error) << "\n" << kUsage;
    return kExitError;
  }
  TestOptions opt;
  auto seed = a.number("--seed", 1);
  auto cases = a.number("--cases", 1000);
  auto maxcex = a.number("--max-counterexamples", 3);
  auto maxbound = a.number("--max-boundary-cases", opt.max_boundary_cases);
  if (!seed || !cases || !maxcex || !maxbound || *maxbound == 0) {
    std::cerr << "bad numeric option\n";
    return kExitError;
  }
  opt.seed = *seed;
  opt.cases = *cases;
  opt.max_counterexamples = *maxcex;
  opt.max_boundary_cases = *maxbound;
  const std::string dir = a.get("--case");
  opt.counterexample_dir = dir + "/counterexamples";
  std::string error;
  auto data = read_case_dir(dir, error);
  if (!data) {
    std::cerr << error << "\n";
    return kExitError;
  }
  std::vector<RegionResult> results;
  Json report = test_case(*data, opt, a.get("--region"), &results);
  if (!write_text_file(dir + "/results.json", report.dump())) {
    std::cerr << "cannot write results.json\n";
    return kExitError;
  }
  for (const auto &r : results) {
    std::cout << r.region_id << ": " << test_status_name(r.test_status) << " (contract "
              << contract_status_name(r.contract_status) << ", valid " << r.valid_pairs << "/"
              << r.attempted << " attempted, executed insts old " << r.old_insts_executed << " new "
              << r.new_insts_executed << ")";
    if (!r.reason.empty())
      std::cout << " : " << r.reason;
    std::cout << "\n";
    if (r.boundary.cases || r.random.cases) {
      std::cout << "  boundary sweep: " << r.boundary.valid << "/" << r.boundary.cases << " valid, "
                << r.boundary.mismatches << " mismatching"
                << (r.boundary_truncated ? ", TRUNCATED" : "") << "; random: " << r.random.valid
                << "/" << r.random.cases << " valid, " << r.random.mismatches << " mismatching\n";
      std::cout << "  boundary families:";
      for (const auto &[family, count] : r.boundary_families)
        std::cout << " " << family << "=" << count;
      std::cout << "\n";
    }
    if (!r.examples.empty())
      std::cout << "  first mismatch case " << r.examples.front().case_index << " ("
                << r.examples.front().phase << ") [" << r.examples.front().kind << "] "
                << r.examples.front().detail << "\n";
    for (const auto &f : r.counterexample_files)
      std::cout << "  counterexample " << f << "\n";
  }
  const Json &overall = report["overall"];
  std::cout << "overall: " << overall["test_status"].as_string() << ", contract "
            << overall["contract_status"].as_string() << ", witness "
            << overall["witness_scope"].as_string() << ", coverage "
            << overall["coverage_scope"].as_string() << ", proof NOT_RUN\n";
  std::cout << "wrote " << dir << "/results.json\n";
  return exit_code_for(overall);
}

int cmd_inject(int argc, char **argv) {
  Args a =
      parse(argc, argv, 2,
            {"--case", "--region", "--side", "--inst", "--word", "--xor", "--label", "--out"}, {});
  if (!a.error.empty() || !a.has("--case") || !a.has("--region") || !a.has("--xor") ||
      !a.has("--out")) {
    std::cerr << (a.error.empty() ? "inject needs --case, --region, --xor and --out" : a.error)
              << "\n"
              << kUsage;
    return kExitError;
  }
  std::string error;
  auto data = read_case_dir(a.get("--case"), error);
  if (!data) {
    std::cerr << error << "\n";
    return kExitError;
  }
  const RegionPair *base = nullptr;
  for (const auto &r : data->regions)
    if (r.id == a.get("--region"))
      base = &r;
  if (!base) {
    std::cerr << "no region " << a.get("--region") << "\n";
    return kExitError;
  }
  MutationSpec spec;
  spec.new_side = a.get("--side", "new") != "old";
  auto inst = a.number("--inst", 0);
  auto word = a.number("--word", 0);
  auto mask = json_u64(Json(a.get("--xor")));
  if (!inst || !word || !mask || *mask > 0xffffffffu) {
    std::cerr << "bad --inst/--word/--xor\n";
    return kExitError;
  }
  spec.inst = *inst;
  spec.word = *word;
  spec.xor_mask = static_cast<uint32_t>(*mask);
  spec.label = a.get("--label", "mutation");
  auto mutated = make_synthetic_mutation(*base, spec, error);
  if (!mutated) {
    std::cerr << error << "\n";
    return kExitError;
  }
  CaseData out = *data;
  out.regions = {*mutated};
  out.command.set("argv", command_line(argc, argv));
  out.command.set("synthetic", true);
  if (!write_case_dir(a.get("--out"), out, error)) {
    std::cerr << error << "\n";
    return kExitError;
  }
  const RegionSide &side = spec.new_side ? mutated->new_side : mutated->old_side;
  std::cout << "synthetic region " << mutated->id << ": " << (spec.new_side ? "new" : "old")
            << " inst " << spec.inst << " is now '" << side.insts[spec.inst].text << "'\n"
            << "wrote " << a.get("--out") << "/case.json (synthetic; not an LLVM finding)\n";
  return kExitPass;
}

int cmd_replay(int argc, char **argv) {
  Args a = parse(argc, argv, 2, {"--counterexample", "--case"}, {});
  if (!a.error.empty() || !a.has("--counterexample")) {
    std::cerr << (a.error.empty() ? "replay needs --counterexample" : a.error) << "\n" << kUsage;
    return kExitError;
  }
  std::string error;
  auto text = read_text_file(a.get("--counterexample"));
  if (!text) {
    std::cerr << "cannot read " << a.get("--counterexample") << "\n";
    return kExitError;
  }
  auto bundle = Json::parse(*text, &error);
  if (!bundle) {
    std::cerr << error << "\n";
    return kExitError;
  }
  if (a.has("--case")) {
    auto data = read_case_dir(a.get("--case"), error);
    if (!data) {
      std::cerr << error << "\n";
      return kExitError;
    }
    bool found = false;
    for (const auto &r : data->regions)
      found |= r.prepared_key == (*bundle)["prepared_key"].as_string();
    if (!found) {
      std::cerr << "counterexample does not belong to any region of this case\n";
      return kExitError;
    }
  }
  ReplayOutcome out = replay_counterexample(*bundle, error);
  if (!error.empty()) {
    std::cerr << error << "\n";
    return kExitError;
  }
  std::cout << out.summary << "\n";
  std::cout << out.detail.dump();
  return out.reproduced ? kExitPass : kExitMismatch;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << kUsage;
    return kExitError;
  }
  const std::string_view cmd = argv[1];
  if (cmd == "analyze")
    return cmd_analyze(argc, argv);
  if (cmd == "test")
    return cmd_test(argc, argv);
  if (cmd == "replay")
    return cmd_replay(argc, argv);
  if (cmd == "inject")
    return cmd_inject(argc, argv);
  if (cmd == "--help" || cmd == "-h" || cmd == "help") {
    std::cout << kUsage;
    return kExitPass;
  }
  std::cerr << "unknown command '" << cmd << "'\n" << kUsage;
  return kExitError;
}
