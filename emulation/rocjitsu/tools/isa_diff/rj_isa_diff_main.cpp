// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_isa_diff_main.cpp
/// @brief CLI: rj_isa_diff analyze | test | replay. See tools/isa_diff/README.md.

#include "case_io.h"
#include "llvm_frontend.h"
#include "mutation.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
    "                      [--llvm-bin DIR] [--llc-args S] [--old-llc-args S] [--new-llc-args S]\n"
    "                      [--old-kernel NAME] [--new-kernel NAME] [--context N]\n"
    "                      [--max-region N] [--max-expansions N] [--no-merge]\n"
    "                      [--no-derive-constraints] [--max-analysis-ms N]\n"
    "                      [--manual-region old:B-E,new:B-E [--map s4=s8,...]\n"
    "                       [--entry-map s4=s8,...] [--exit-map s4=s8,...]]\n"
    "                      [--diagnostic-entry-prefix] [--cache-dir DIR]\n"
    "                      [--return-vgprs N] [--return-masks M,...] [--function-mode HEX]\n"
    "                      [--entry-vgprs N] [--entry-sgprs N]  (entry and return ABI)\n"
    "                      [--target NAME[:w32|w64]] [--wave-size 32|64] [--mattr FEATURES]\n"
    "                      [--calling-convention c|gfx|shader] [--return-sgprs N]\n"
    "                      [--return-sgpr-masks M,...]  (shader parts)\n"
    "                      [--old-mattr F] [--new-mattr F]  (per-side target features)\n"
    "      --target: the GPU of raw words and .s/.ll inputs (code objects name their own;\n"
    "      default gfx950). --wave-size: wave32/wave64 on GFX10+ (functions; a kernel's\n"
    "      descriptor decides its own). --mattr: target features for llc/llvm-mc.\n"
    "      --manual-region: one hand-written region (instruction indices or 0x PCs, E\n"
    "      exclusive); --map: old=new register correspondence. Both stay ASSUMED.\n"
    "  rj_isa_diff test    --case DIR [--seed N] [--cases N] [--region ID]\n"
    "                      [--max-counterexamples N] [--max-boundary-cases N]\n"
    "                      [--reduction-budget N] [--no-reduce] [--no-case-log]\n"
    "                      [--cache-dir DIR] [--strict-nan] [--poison-filter KINDS]\n"
    "      --cases counts random cases (default 5000); the boundary/NaN/corner sweep always\n"
    "      runs first.\n"
    "      --strict-nan: NaNs compare bit for bit (default: any NaN equals any NaN of the\n"
    "      same format, as LLVM allows). --poison-filter shift,fptoint,ctz,nsw,nuw,\n"
    "      disjoint,exact,nnan,ninf|all: exclude inputs that are poison in the IR.\n"
    "  rj_isa_diff replay  --counterexample FILE [--case DIR]\n"
    "  rj_isa_diff inject  --case DIR --region ID --xor MASK --out DIR [--side old|new]\n"
    "                      [--inst N] [--word N] [--label NAME]   (synthetic self-test)\n"
    "  rj_isa_diff llvm-pair --input K.ll --old-args S --new-args S --out DIR\n"
    "                      [--common-args S] [--kernel NAME] [--llvm-bin DIR]  (LLVM plugin)\n"
    "FILE is a code object (CDNA1-4: gfx908/90a/940-942/950; RDNA1-4: gfx1010-1201;\n"
    "gfx1250/1251), a raw instruction stream, or .s/.ll (LLVM plugin:\n"
    "llc -> llvm-mc -> ld.lld from --llvm-bin, $RJ_ISA_DIFF_LLVM_BIN or /opt/rocm/llvm/bin).\n"
    "exit: 0 no mismatch found with complete declared scope (no don't-care masking),\n"
    "      1 local mismatch, 2 inconclusive/unsupported/partial scope or a pass that needed\n"
    "      the per-bit don't-care mask, 3 error.\n";

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
  // A pass that needed the per-bit don't-care mask is not a plain pass: per-bit masking
  // can accept values old cannot produce.
  if ((status == "NO_MISMATCH_FOUND" || status == "NO_SEMANTIC_DIFF") &&
      coverage == "COMPLETE_FOR_DECLARED_SCOPE" && overall["dont_care_masked_cases"].as_uint() == 0)
    return kExitPass;
  return kExitInconclusive;
}

const std::vector<std::string> kPlanFlags = {
    "--context",      "--max-region",         "--max-expansions", "--manual-region",
    "--map",          "--entry-map",          "--exit-map",       "--max-analysis-ms",
    "--cache-dir",    "--return-vgprs",       "--function-mode",  "--entry-vgprs",
    "--return-masks", "--entry-sgprs",        "--target",         "--wave-size",
    "--mattr",        "--calling-convention", "--return-sgprs",   "--return-sgpr-masks",
    "--old-mattr",    "--new-mattr"};
const std::vector<std::string> kPlanSwitches = {"--no-merge", "--no-derive-constraints",
                                                "--diagnostic-entry-prefix"};

std::optional<PlanOptions> parse_plan(const Args &a) {
  PlanOptions plan;
  auto context = a.number("--context", 0);
  auto max_region = a.number("--max-region", plan.max_region_insts);
  auto max_exp = a.number("--max-expansions", plan.max_expansions);
  auto max_ms = a.number("--max-analysis-ms", 0);
  if (!context || !max_region || !max_exp || !max_ms)
    return std::nullopt;
  plan.max_analysis_us = *max_ms * 1000;
  plan.context_insts = *context;
  plan.max_region_insts = *max_region;
  plan.max_expansions = *max_exp;
  plan.merge_same_block = !a.is_switch("--no-merge");
  plan.derive_constraints = !a.is_switch("--no-derive-constraints");
  plan.manual_region = a.get("--manual-region");
  plan.register_map = a.get("--map");
  plan.entry_map = a.get("--entry-map");
  plan.exit_map = a.get("--exit-map");
  plan.diagnostic_entry_prefix = a.is_switch("--diagnostic-entry-prefix");
  return plan;
}

bool has_suffix(const std::string &s, const std::string &suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// The target asked for on the command line: --target NAME[:w32|w64], --wave-size 32|64.
struct TargetRequest {
  std::optional<Target> target; ///< With its default wave size unless one was given.
  unsigned wave_size = 0;       ///< Explicit wave size, 0 if none.
  std::string mattr;            ///< Target features for the LLVM plugin.
};

/// Apply a request to a loaded artifact: raw words take the target; a code object must be
/// for it; an explicit wave size sets a callable function's (a kernel's descriptor decides
/// its own and must agree).
bool apply_target(const TargetRequest &req, Artifact &a, std::string &error) {
  if (req.target) {
    if (a.kind == "raw_words") {
      a.target = *req.target;
    } else if (a.target.name != req.target->name) {
      error = "the code object is for " + a.target.name + ", not " + req.target->name;
      return false;
    }
  }
  if (req.wave_size && req.wave_size != a.target.wave_size) {
    if (a.kind == "code_object" && !a.callable) {
      error = "the kernel descriptor asks for wave" + std::to_string(a.target.wave_size) +
              ", not wave" + std::to_string(req.wave_size);
      return false;
    }
    auto t = with_wave_size(a.target, req.wave_size);
    if (!t) {
      error = a.target.name + " cannot run wave" + std::to_string(req.wave_size);
      return false;
    }
    a.target = *t;
  }
  return true;
}

/// Load one side: a code object or raw words directly, `.s`/`.ll` through the LLVM plugin.
bool load_side(const std::string &path, const std::string &kernel, const std::string &side,
               const std::string &out_dir, const std::string &llvm_bin,
               const std::vector<std::string> &common_args,
               const std::vector<std::string> &side_args, const TargetRequest &req, Artifact &out,
               Json &provenance, std::string &error) {
  const bool is_ll = has_suffix(path, ".ll");
  const bool is_s = has_suffix(path, ".s");
  if (!is_ll && !is_s)
    return load_artifact(path, kernel, out, error) && apply_target(req, out, error);
  auto tc = llvm::Toolchain::discover(llvm_bin, error);
  if (!tc)
    return false;
  std::error_code ec;
  std::filesystem::create_directories(out_dir + "/artifacts", ec);
  const std::string prefix = out_dir + "/artifacts/" + side;
  llvm::Provenance prov;
  if (req.target)
    prov.mcpu = req.target->name;
  prov.wave_size = req.wave_size;
  prov.mattr = req.mattr;
  const bool ok =
      is_ll ? llvm::compile(*tc, path, common_args, side_args, prefix, kernel, out, prov, error)
            : llvm::assemble(*tc, path, prefix, kernel, out, prov, error);
  provenance = prov.to_json();
  if (!write_text_file(prefix + ".provenance.json", provenance.dump()))
    std::cerr << "warning: cannot write " << prefix << ".provenance.json\n";
  return ok && apply_target(req, out, error);
}

int run_analyze(const std::string &old_path, const std::string &new_path, const Args &a,
                const std::vector<std::string> &common_args,
                const std::vector<std::string> &old_args, const std::vector<std::string> &new_args,
                int argc, char **argv) {
  auto plan = parse_plan(a);
  if (!plan) {
    std::cerr << "bad numeric option\n";
    return kExitError;
  }
  const std::string out_dir = a.get("--out");
  const std::string kernel = a.get("--kernel");
  std::string error;
  TargetRequest req;
  if (a.has("--target")) {
    req.target = parse_target(a.get("--target"));
    if (!req.target) {
      std::cerr << "--target takes NAME[:w32|w64] with NAME one of " << supported_targets() << "\n";
      return kExitError;
    }
    if (a.get("--target").find(':') != std::string::npos)
      req.wave_size = req.target->wave_size;
  }
  if (a.has("--wave-size")) {
    const auto w = a.number("--wave-size", 0);
    if (!w || (*w != 32 && *w != 64)) {
      std::cerr << "--wave-size takes 32 or 64\n";
      return kExitError;
    }
    req.wave_size = static_cast<unsigned>(*w);
    if (req.target && !with_wave_size(*req.target, req.wave_size)) {
      std::cerr << req.target->name << " cannot run wave" << req.wave_size << "\n";
      return kExitError;
    }
  }
  req.mattr = a.get("--mattr");
  // A RUN line whose features changed: each side assembles with its own.
  TargetRequest old_req = req, new_req = req;
  if (a.has("--old-mattr"))
    old_req.mattr = a.get("--old-mattr");
  if (a.has("--new-mattr"))
    new_req.mattr = a.get("--new-mattr");
  Artifact old_a, new_a;
  Json old_prov, new_prov;
  if (!load_side(old_path, a.get("--old-kernel", kernel), "old", out_dir, a.get("--llvm-bin"),
                 common_args, old_args, old_req, old_a, old_prov, error)) {
    std::cerr << "old: " << error << "\n";
    return kExitError;
  }
  if (!load_side(new_path, a.get("--new-kernel", kernel), "new", out_dir, a.get("--llvm-bin"),
                 common_args, new_args, new_req, new_a, new_prov, error)) {
    std::cerr << "new: " << error << "\n";
    return kExitError;
  }
  if (!(old_a.target == new_a.target)) {
    std::cerr << "old and new run on different targets (" << old_a.target.label() << ", "
              << new_a.target.label() << ")\n";
    return kExitError;
  }
  // Callable functions: the caller-visible part of the calling convention.
  const auto ret_vgprs = a.number("--return-vgprs", 32);
  std::optional<uint32_t> fmode = default_function_mode(old_a.target);
  if (a.has("--function-mode")) {
    const std::string &s = a.get("--function-mode");
    char *end = nullptr;
    const unsigned long v = std::strtoul(s.c_str(), &end, 0);
    fmode = end && *end == '\0' && !s.empty() && v <= 0xffffffffu
                ? std::optional<uint32_t>(static_cast<uint32_t>(v))
                : std::nullopt;
  }
  const std::string cc = a.get("--calling-convention", "c");
  const auto ret_sgprs = a.number("--return-sgprs", 0);
  const unsigned max_ret_vgprs = cc == "shader" ? 136 : 32;
  if (!ret_vgprs || *ret_vgprs > max_ret_vgprs || !fmode || !ret_sgprs || *ret_sgprs > 44 ||
      (cc != "c" && cc != "gfx" && cc != "shader")) {
    std::cerr << "--return-vgprs takes 0..32 (0..136 for shaders), --return-sgprs 0..44, "
                 "--calling-convention c|gfx|shader; --function-mode takes a MODE register "
                 "value\n";
    return kExitError;
  }
  std::vector<uint32_t> ret_masks, ret_sgpr_masks;
  for (const std::string &tok : llvm::split_args([&] {
         std::string t = a.get("--return-sgpr-masks");
         std::replace(t.begin(), t.end(), ',', ' ');
         return t;
       }())) {
    char *end = nullptr;
    const unsigned long v = std::strtoul(tok.c_str(), &end, 0);
    if (!end || *end != '\0' || v > 0xffffffffu) {
      std::cerr << "--return-sgpr-masks takes comma-separated 32-bit masks\n";
      return kExitError;
    }
    ret_sgpr_masks.push_back(static_cast<uint32_t>(v));
  }
  for (const std::string &tok : llvm::split_args([&] {
         std::string t = a.get("--return-masks");
         std::replace(t.begin(), t.end(), ',', ' ');
         return t;
       }())) {
    char *end = nullptr;
    const unsigned long v = std::strtoul(tok.c_str(), &end, 0);
    if (!end || *end != '\0' || v > 0xffffffffu) {
      std::cerr << "--return-masks takes comma-separated 32-bit masks\n";
      return kExitError;
    }
    ret_masks.push_back(static_cast<uint32_t>(v));
  }
  const auto entry_vgprs = a.number("--entry-vgprs", 0);
  const auto entry_sgprs = a.number("--entry-sgprs", 0);
  if (!entry_vgprs || *entry_vgprs > 256 || !entry_sgprs || *entry_sgprs > kNumSgprs) {
    std::cerr << "--entry-vgprs takes 0..256, --entry-sgprs 0..106\n";
    return kExitError;
  }
  for (Artifact *art : {&old_a, &new_a}) {
    art->return_vgprs = static_cast<unsigned>(*ret_vgprs);
    art->calling_convention = cc;
    art->return_sgprs = static_cast<unsigned>(*ret_sgprs);
    art->return_sgpr_masks = ret_sgpr_masks;
    art->function_mode = *fmode;
    art->return_masks = ret_masks;
    if (a.has("--entry-vgprs") && art->entry_vgprs >= 0)
      art->entry_vgprs = static_cast<int>(*entry_vgprs);
    if (a.has("--entry-sgprs") && art->entry_sgprs >= 0)
      art->entry_sgprs = static_cast<int>(*entry_sgprs);
  }
  bool cache_hit = false;
  auto data = analyze_cached(old_a, new_a, *plan, a.get("--cache-dir"), cache_hit, error);
  if (!data) {
    std::cerr << error << "\n";
    return kExitError;
  }
  data->command.set("argv", command_line(argc, argv));
  if (!old_prov.is_null() || !new_prov.is_null()) {
    data->provenance = Json::object();
    data->provenance.set("old", old_prov);
    data->provenance.set("new", new_prov);
    const bool same_input =
        old_prov["input_sha256"].as_string() == new_prov["input_sha256"].as_string();
    data->provenance.set("same_input", same_input);
    data->provenance.set("same_toolchain",
                         old_prov["toolchain"].dump(-1) == new_prov["toolchain"].dump(-1));
    Json timing = data->analysis["timing"];
    timing.set("compile_ns", old_prov["compile_ns"].as_uint() + new_prov["compile_ns"].as_uint());
    data->analysis.set("timing", std::move(timing));
  }
  if (!write_case_dir(out_dir, *data, error)) {
    std::cerr << error << "\n";
    return kExitError;
  }
  std::cout << "kernel: old " << old_a.kernel_name << " ("
            << data->artifacts["old"]["reachable_insts"].as_uint() << " insts), new "
            << new_a.kernel_name << " (" << data->artifacts["new"]["reachable_insts"].as_uint()
            << " insts)\n";
  if (!data->provenance.is_null())
    std::cout << "provenance: " << data->provenance["old"]["input_kind"].as_string() << " via "
              << data->provenance["old"]["toolchain"]["version"].as_string()
              << (data->provenance["same_input"].as_bool() ? ", same input" : ", different inputs")
              << "\n";
  std::cout << "changes: " << data->changes.size() << " hunk(s), " << data->regions.size()
            << " region(s), " << data->unresolved.size() << " unresolved item(s)\n";
  for (const auto &r : data->regions)
    std::cout << "  " << r.id << " " << region_status_name(r.status) << " old "
              << r.old_side.insts.size() << " / new " << r.new_side.insts.size()
              << " insts, contract " << contract_status_name(r.contract.status) << ", composition "
              << r.contract.composition << ", pending " << r.contract.pending_effects
              << (r.reason.empty() ? "" : " : " + r.reason) << "\n";
  if (!data->variants.empty())
    std::cout << "  " << data->variants.size()
              << " expansion variant(s) precomputed (tried automatically after a local mismatch)\n";
  for (const auto &d : data->diagnostics)
    std::cout << "  diagnostic " << d.id << " " << region_status_name(d.status) << " old "
              << d.old_side.insts.size() << " / new " << d.new_side.insts.size() << " insts"
              << (d.reason.empty() ? "" : " : " + d.reason) << "\n";
  if (cache_hit)
    std::cout << "analysis cache hit\n";
  for (const auto &u : data->unresolved)
    std::cout << "  unresolved [" << u.kind << "] " << u.what << "\n";
  std::cout << "wrote " << out_dir << "/case.json\n";
  return kExitPass;
}

int cmd_analyze(int argc, char **argv) {
  std::vector<std::string> flags = {"--old",          "--new",         "--out",      "--kernel",
                                    "--old-kernel",   "--new-kernel",  "--llvm-bin", "--llc-args",
                                    "--old-llc-args", "--new-llc-args"};
  flags.insert(flags.end(), kPlanFlags.begin(), kPlanFlags.end());
  Args a = parse(argc, argv, 2, flags, kPlanSwitches);
  if (!a.error.empty() || !a.has("--old") || !a.has("--new") || !a.has("--out")) {
    std::cerr << (a.error.empty() ? "analyze needs --old, --new and --out" : a.error) << "\n"
              << kUsage;
    return kExitError;
  }
  return run_analyze(a.get("--old"), a.get("--new"), a, llvm::split_args(a.get("--llc-args")),
                     llvm::split_args(a.get("--old-llc-args")),
                     llvm::split_args(a.get("--new-llc-args")), argc, argv);
}

int cmd_llvm_pair(int argc, char **argv) {
  std::vector<std::string> flags = {"--input",       "--out",      "--kernel",  "--llvm-bin",
                                    "--common-args", "--old-args", "--new-args"};
  flags.insert(flags.end(), kPlanFlags.begin(), kPlanFlags.end());
  Args a = parse(argc, argv, 2, flags, kPlanSwitches);
  if (!a.error.empty() || !a.has("--input") || !a.has("--out")) {
    std::cerr << (a.error.empty() ? "llvm-pair needs --input and --out" : a.error) << "\n"
              << kUsage;
    return kExitError;
  }
  const std::string input = a.get("--input");
  if (!has_suffix(input, ".ll")) {
    std::cerr << "llvm-pair compiles one LLVM IR (.ll) input twice\n";
    return kExitError;
  }
  const auto old_args = llvm::split_args(a.get("--old-args"));
  const auto new_args = llvm::split_args(a.get("--new-args"));
  if (old_args == new_args) {
    std::cerr << "--old-args and --new-args are identical: nothing differs between the sides\n";
    return kExitError;
  }
  return run_analyze(input, input, a, llvm::split_args(a.get("--common-args")), old_args, new_args,
                     argc, argv);
}

int cmd_test(int argc, char **argv) {
  Args a = parse(argc, argv, 2,
                 {"--case", "--seed", "--cases", "--region", "--max-counterexamples",
                  "--max-boundary-cases", "--reduction-budget", "--cache-dir", "--poison-filter"},
                 {"--no-reduce", "--no-case-log", "--strict-nan"});
  if (!a.error.empty() || !a.has("--case")) {
    std::cerr << (a.error.empty() ? "test needs --case" : a.error) << "\n" << kUsage;
    return kExitError;
  }
  TestOptions opt;
  auto seed = a.number("--seed", 1);
  auto cases = a.number("--cases", 5000);
  auto maxcex = a.number("--max-counterexamples", 3);
  auto maxbound = a.number("--max-boundary-cases", opt.max_boundary_cases);
  auto redbudget = a.number("--reduction-budget", opt.reduction_budget);
  if (!seed || !cases || !maxcex || !maxbound || *maxbound == 0 || !redbudget) {
    std::cerr << "bad numeric option\n";
    return kExitError;
  }
  opt.seed = *seed;
  opt.cases = *cases;
  opt.max_counterexamples = *maxcex;
  opt.max_boundary_cases = *maxbound;
  opt.reduction_budget = *redbudget;
  opt.reduce = !a.is_switch("--no-reduce");
  opt.strict_nan = a.is_switch("--strict-nan");
  if (a.has("--poison-filter")) {
    const auto kinds = parse_poison_kinds(a.get("--poison-filter"));
    if (!kinds) {
      std::cerr << "--poison-filter takes shift,fptoint,ctz,nsw,nuw,disjoint,exact,nnan,ninf "
                   "or all\n";
      return kExitError;
    }
    opt.poison_filter = *kinds;
  }
  const std::string dir = a.get("--case");
  opt.counterexample_dir = dir + "/counterexamples";
  if (!a.is_switch("--no-case-log"))
    opt.case_log_dir = dir + "/logs";
  opt.cache_dir = a.get("--cache-dir");
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
    if (!r.cached.is_null())
      std::cout << "  reused from the result cache (key " << r.cached["cache"]["key"].as_string()
                << ")\n";
    if (r.boundary.cases || r.random.cases) {
      std::cout << "  boundary sweep: " << r.boundary.valid << "/" << r.boundary.cases << " valid, "
                << r.boundary.mismatches << " mismatching"
                << (r.boundary_truncated ? ", TRUNCATED" : "") << "; random: " << r.random.valid
                << "/" << r.random.cases << " valid, " << r.random.mismatches << " mismatching\n";
      std::cout << "  boundary families:";
      for (const auto &[family, count] : r.boundary_families)
        std::cout << " " << family << "=" << count;
      std::cout << "\n";
      std::cout << "  novelty: " << r.novelty_buckets << " outcome bucket(s) (" << r.novel_boundary
                << " first seen in the sweep, " << r.novel_random << " in the random phase)\n";
    }
    if (!r.case_log.empty())
      std::cout << "  case log " << r.case_log << "\n";
    if (!r.examples.empty() && !r.examples.front().reduction.is_null()) {
      const Json &red = r.examples.front().reduction;
      std::cout << "  reduced first counterexample in " << red["runs"].as_uint() << " run(s), "
                << red["accepted_steps"].as_uint() << " step(s)"
                << (red["seed_independent"].as_bool() ? ", seed-independent" : "")
                << "; non-zero inputs:";
      for (const auto &[loc, value] : red["reduced_entry_values"].members())
        std::cout << " " << loc << "=" << value.as_string();
      std::cout << "\n";
    }
    if (!r.examples.empty())
      std::cout << "  first mismatch case " << r.examples.front().case_index << " ("
                << r.examples.front().phase << ") [" << r.examples.front().kind << "] "
                << r.examples.front().detail << "\n";
    for (const auto &f : r.counterexample_files)
      std::cout << "  counterexample " << f << "\n";
    if (!r.resolution.empty()) {
      std::cout << "  expansion: " << r.resolution;
      if (!r.resolved_by.empty())
        std::cout << " by " << r.resolved_by << " (contract "
                  << contract_status_name(r.resolved_contract) << ")";
      std::cout << ", " << r.expansion_tried.size() << " variant(s) tried\n";
    }
  }
  for (const Json &d : report["diagnostics"].items())
    std::cout << "diagnostic " << d["region"].as_string() << " (for " << d["parent"].as_string()
              << "): " << d["test_status"].as_string() << ", valid "
              << d["counts"]["valid_pairs"].as_uint()
              << (d["reason"].as_string().empty() ? "" : " : " + d["reason"].as_string())
              << " [not part of the verdict]\n";
  if (report.has("cache"))
    std::cout << "result cache: " << report["cache"]["hits"].as_uint() << " hit(s), "
              << report["cache"]["misses"].as_uint() << " miss(es)\n";
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
  out.variants.clear(); // variants belong to the unmutated base region
  out.diagnostics.clear();
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
  if (cmd == "llvm-pair")
    return cmd_llvm_pair(argc, argv);
  if (cmd == "--help" || cmd == "-h" || cmd == "help") {
    std::cout << kUsage;
    return kExitPass;
  }
  std::cerr << "unknown command '" << cmd << "'\n" << kUsage;
  return kExitError;
}
