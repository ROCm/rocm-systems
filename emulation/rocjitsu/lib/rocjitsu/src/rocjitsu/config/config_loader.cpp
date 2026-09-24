// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/config_loader.h"

#include "rocjitsu/vm/amdgpu/pci/gpu_generation_registry.h"

#include "rocjitsu/config/config_common.h"
#include "rocjitsu/isa/target_registry.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/hbm_controller.h"
#include "rocjitsu/vm/amdgpu/iod.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"
#include "rocjitsu/vm/amdgpu/memory_side_cache.h"
#include "rocjitsu/vm/amdgpu/partitioning.h"
#include "rocjitsu/vm/amdgpu/shader_engine.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/soc.h"

#include "flatbuffers/idl.h"
#include "simdojo/sim/exec_mode.h"
#include "simdojo/sim/topology.h"
#include "simulation_config_generated.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace config {

ExecutionThreadAllocation resolve_execution_threads(const ExecutionThreadRequest &request,
                                                    uint32_t host_threads, uint32_t xcds,
                                                    std::span<const uint32_t> dispatch_capacities,
                                                    bool async_supported,
                                                    std::span<const ExecutionThreadChoice> choices,
                                                    bool clocked) {
  if (request.helpers < -1 || request.helpers > 128)
    throw std::invalid_argument("async_helper_threads must be -1 or between 0 and 128");
  const uint32_t budget = request.budget ? request.budget : std::max(host_threads, 1u);
  const uint32_t cpu_budget =
      request.budget ? request.budget : std::min(budget, kDefaultExecutionThreadCap);
  xcds = std::max(xcds, 1u);
  if (clocked)
    return {std::min(request.engines ? request.engines : cpu_budget, xcds),
            std::vector<uint32_t>(dispatch_capacities.size(), 1), 0};
  auto effective = [&](ExecutionThreadChoice choice) {
    ExecutionThreadAllocation result;
    result.engines = std::min(request.engines ? request.engines : choice.engines, xcds);
    result.helpers = request.helpers >= 0 ? static_cast<uint32_t>(request.helpers)
                                          : (async_supported ? choice.helpers : 0);
    for (uint32_t capacity : dispatch_capacities)
      result.dispatch.push_back(
          std::min(request.dispatch ? request.dispatch : choice.dispatch, std::max(capacity, 1u)));
    return result;
  };
  // Keep explicit overrides even when no table entry fits the budget.
  auto result = effective({});
  uint64_t best_cost = 0;
  for (auto choice : choices) {
    if (!choice.engines || !choice.dispatch || choice.helpers > 128)
      throw std::invalid_argument("thread_allocations require num_threads >= 1, "
                                  "cpu_dispatch_threads >= 1 and async_helper_threads <= 128");
    auto candidate = effective(choice);
    uint64_t cpu_cost = candidate.engines;
    for (uint32_t width : candidate.dispatch)
      cpu_cost += width - 1;
    // Extra affinity can supply helpers without increasing the automatic
    // engine/dispatch allocation beyond its existing cap. Explicit CPU knobs
    // retain their override behavior.
    if (!request.engines && !request.dispatch && cpu_cost > cpu_budget)
      continue;
    const uint64_t cost = cpu_cost + candidate.helpers;
    if (cost <= budget && cost >= best_cost) {
      result = std::move(candidate);
      best_cost = cost;
    }
  }
  return result;
}

bool configured_async_mma_supported(rj_code_arch_t arch) {
  return amdgpu::async_mma_policy::supported(arch);
}

std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources>
make_async_execution_resources(uint32_t helpers) {
  return std::make_shared<amdgpu::matrix_coexecution::ExecutionResources>(helpers);
}

std::vector<uint32_t> resolve_cpu_dispatch_thread_budgets(uint32_t requested_threads,
                                                          uint32_t hardware_threads,
                                                          size_t soc_count,
                                                          uint32_t automatic_thread_cap) {
  if (soc_count == 0)
    return {};
  if (requested_threads != 0)
    return std::vector<uint32_t>(soc_count, requested_threads);

  const uint32_t host_width = hardware_threads ? hardware_threads : 1u;
  const uint32_t host_budget = std::min(host_width, std::max(automatic_thread_cap, 1u));
  if (host_budget < soc_count)
    return std::vector<uint32_t>(soc_count, 1);

  const uint32_t base = static_cast<uint32_t>(host_budget / soc_count);
  const size_t remainder = host_budget % soc_count;
  std::vector<uint32_t> budgets(soc_count, base);
  for (size_t i = 0; i < remainder; ++i)
    ++budgets[i];
  return budgets;
}

SoC *LoadedConfig::soc() { return dynamic_cast<SoC *>(build_result.root.get()); }

void LoadedConfig::apply_cpu_dispatch_threads() {
  if (execution_threads.dispatch.empty()) {
    override_cpu_dispatch_threads(amdgpu::available_host_threads());
    return;
  }
  if (auto *primary = soc())
    primary->set_dispatch_threads(execution_threads.dispatch.front());
  for (size_t i = 0; i < extra_gpu_builds.size(); ++i)
    if (auto *extra = dynamic_cast<SoC *>(extra_gpu_builds[i].root.get()))
      extra->set_dispatch_threads(execution_threads.dispatch.at(i + 1));
}

void LoadedConfig::override_cpu_dispatch_threads(uint32_t hardware_threads,
                                                 uint32_t automatic_thread_cap) {
  std::vector<SoC *> socs;
  socs.reserve(extra_gpu_builds.size() + 1);
  if (auto *primary = soc())
    socs.push_back(primary);
  if (num_gpus > 1) {
    for (auto &extra : extra_gpu_builds) {
      if (auto *extra_soc = dynamic_cast<SoC *>(extra.root.get()))
        socs.push_back(extra_soc);
    }
  }

  const uint32_t requested_threads =
      exec_mode == simdojo::ExecMode::FUNCTIONAL ? cpu_dispatch_threads : 1u;
  const auto budgets = resolve_cpu_dispatch_thread_budgets(requested_threads, hardware_threads,
                                                           socs.size(), automatic_thread_cap);
  for (size_t i = 0; i < socs.size(); ++i)
    socs[i]->set_dispatch_threads(budgets[i]);
}

namespace {

simdojo::SimulationEngine::Config
engine_config_from_fb(const rocjitsu::fb::SimulationConfig *fb_config) {
  simdojo::SimulationEngine::Config cfg{};
  cfg.max_ticks = fb_config->max_ticks();
  cfg.num_threads = fb_config->num_threads();
  return cfg;
}

simdojo::ExecMode exec_mode_from_fb(const rocjitsu::fb::SimulationConfig *fb_config) {
  return parse_exec_mode(fb_config->exec_mode() ? fb_config->exec_mode()->str() : "");
}

uint32_t config_u32(const std::unordered_map<std::string, std::string> &cfg, const std::string &key,
                    uint32_t def = 0) {
  auto it = cfg.find(key);
  if (it == cfg.end())
    return def;
  return static_cast<uint32_t>(std::stoul(it->second));
}

uint32_t default_sgprs_per_wf(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5)
    return 128;
  return 112;
}

uint32_t default_vgprs_per_wf(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    return 1024;
  return 256;
}

enum class TokType { NUMBER, IDENT, OP2, OP1, LPAREN, RPAREN, END_TOK };

struct Tok {
  TokType type;
  int32_t num = 0;
  std::string str;
  char ch = '\0';
};

class Lexer {
public:
  explicit Lexer(const std::string &e) : e_(e) {}

  Tok next() {
    skip();
    if (pos_ >= e_.size())
      return {TokType::END_TOK, 0, "", '\0'};
    char c = e_[pos_];
    if (c == '(') {
      ++pos_;
      return {TokType::LPAREN, 0, "", '('};
    }
    if (c == ')') {
      ++pos_;
      return {TokType::RPAREN, 0, "", ')'};
    }
    if (pos_ + 1 < e_.size()) {
      std::string s2 = e_.substr(pos_, 2);
      if (s2 == "==" || s2 == "!=" || s2 == "<=" || s2 == ">=" || s2 == "&&" || s2 == "||") {
        pos_ += 2;
        return {TokType::OP2, 0, s2, '\0'};
      }
    }
    if (c == '+' || c == '-' || c == '*' || c == '/' || c == '%' || c == '<' || c == '>') {
      ++pos_;
      return {TokType::OP1, 0, "", c};
    }
    if (std::isdigit(static_cast<unsigned char>(c))) {
      int32_t v = 0;
      while (pos_ < e_.size() && std::isdigit(static_cast<unsigned char>(e_[pos_])))
        v = v * 10 + (e_[pos_++] - '0');
      return {TokType::NUMBER, v, "", '\0'};
    }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      std::string id;
      while (pos_ < e_.size() &&
             (std::isalnum(static_cast<unsigned char>(e_[pos_])) || e_[pos_] == '_'))
        id += e_[pos_++];
      return {TokType::IDENT, 0, id, '\0'};
    }
    throw std::runtime_error("Bad char in expr: '" + std::string(1, c) + "'");
  }

private:
  void skip() {
    while (pos_ < e_.size() && std::isspace(static_cast<unsigned char>(e_[pos_])))
      ++pos_;
  }
  std::string e_;
  size_t pos_ = 0;
};

class Parser {
public:
  Parser(const std::string &e, const std::unordered_map<std::string, int32_t> &v)
      : lex_(e), vars_(v), e_(e) {
    adv();
  }
  int32_t run() {
    auto r = p_or();
    if (cur_.type != TokType::END_TOK)
      throw std::runtime_error("Extra tokens in: " + e_);
    return r;
  }

private:
  void adv() { cur_ = lex_.next(); }
  bool is1(char c) { return cur_.type == TokType::OP1 && cur_.ch == c; }
  bool is2(const std::string &s) { return cur_.type == TokType::OP2 && cur_.str == s; }

  int32_t p_or() {
    auto l = p_and();
    while (is2("||")) {
      adv();
      l = (l || p_and()) ? 1 : 0;
    }
    return l;
  }
  int32_t p_and() {
    auto l = p_cmp();
    while (is2("&&")) {
      adv();
      l = (l && p_cmp()) ? 1 : 0;
    }
    return l;
  }
  int32_t p_cmp() {
    auto l = p_add();
    if (is2("==")) {
      adv();
      return (l == p_add()) ? 1 : 0;
    }
    if (is2("!=")) {
      adv();
      return (l != p_add()) ? 1 : 0;
    }
    if (is2("<=")) {
      adv();
      return (l <= p_add()) ? 1 : 0;
    }
    if (is2(">=")) {
      adv();
      return (l >= p_add()) ? 1 : 0;
    }
    if (is1('<')) {
      adv();
      return (l < p_add()) ? 1 : 0;
    }
    if (is1('>')) {
      adv();
      return (l > p_add()) ? 1 : 0;
    }
    return l;
  }
  int32_t p_add() {
    auto l = p_mul();
    while (is1('+') || is1('-')) {
      char o = cur_.ch;
      adv();
      auto r = p_mul();
      l = (o == '+') ? l + r : l - r;
    }
    return l;
  }
  int32_t p_mul() {
    auto l = p_unary();
    while (is1('*') || is1('/') || is1('%')) {
      char o = cur_.ch;
      adv();
      auto r = p_unary();
      if (o == '*')
        l *= r;
      else if (o == '/')
        l = r ? l / r : 0;
      else
        l = r ? l % r : 0;
    }
    return l;
  }
  int32_t p_unary() {
    if (is1('-')) {
      adv();
      return -p_atom();
    }
    return p_atom();
  }
  int32_t p_atom() {
    if (cur_.type == TokType::NUMBER) {
      auto v = cur_.num;
      adv();
      return v;
    }
    if (cur_.type == TokType::IDENT) {
      auto it = vars_.find(cur_.str);
      if (it == vars_.end())
        throw std::runtime_error("Unknown var '" + cur_.str + "' in: " + e_);
      adv();
      return it->second;
    }
    if (cur_.type == TokType::LPAREN) {
      adv();
      auto v = p_or();
      if (cur_.type != TokType::RPAREN)
        throw std::runtime_error("Missing ')' in: " + e_);
      adv();
      return v;
    }
    throw std::runtime_error("Unexpected token in: " + e_);
  }

  Lexer lex_;
  const std::unordered_map<std::string, int32_t> &vars_;
  std::string e_;
  Tok cur_;
};

int32_t eval_expr(const std::string &expr, const std::unordered_map<std::string, int32_t> &vars) {
  Parser p(expr, vars);
  return p.run();
}

bool eval_where(const std::string &w, const std::unordered_map<std::string, int32_t> &vars) {
  if (w.empty())
    return true;
  return eval_expr(w, vars) != 0;
}

std::string subst_vars(const std::string &pat, const std::unordered_map<std::string, int32_t> &v) {
  std::string r;
  size_t i = 0;
  while (i < pat.size()) {
    if (pat[i] == '[') {
      auto j = pat.find(']', i);
      if (j == std::string::npos)
        throw std::runtime_error("Unmatched '[' in: " + pat);
      auto e = pat.substr(i + 1, j - i - 1);
      if (e.find(':') != std::string::npos)
        r += pat.substr(i, j - i + 1);
      else
        r += std::to_string(eval_expr(e, v));
      i = j + 1;
    } else {
      r += pat[i++];
    }
  }
  return r;
}

std::vector<std::string> expand_range(const std::string &pat) {
  static const std::regex re(R"(^(\w*)\[(\d+):(\d+)\](.*)$)");
  std::smatch m;
  if (!std::regex_match(pat, m, re))
    return {pat};
  auto pfx = m[1].str();
  auto s = static_cast<uint32_t>(std::stoul(m[2].str()));
  auto e = static_cast<uint32_t>(std::stoul(m[3].str()));
  auto sfx = m[4].str();
  std::vector<std::string> out;
  for (uint32_t i = s; i < e; ++i)
    out.push_back(pfx + std::to_string(i) + sfx);
  return out;
}

void iterate_for(const flatbuffers::Vector<flatbuffers::Offset<fb::ForRange>> *ranges,
                 const std::string &where,
                 std::function<void(const std::unordered_map<std::string, int32_t> &)> fn) {
  if (!ranges || ranges->size() == 0) {
    std::unordered_map<std::string, int32_t> empty;
    if (eval_where(where, empty))
      fn(empty);
    return;
  }
  struct R {
    std::string name;
    uint32_t lo, hi;
  };
  std::vector<R> rv;
  for (auto *fr : *ranges)
    rv.push_back({fr->var_name() ? fr->var_name()->str() : "", fr->start(), fr->end()});

  std::unordered_map<std::string, int32_t> vars;
  std::function<void(size_t)> go = [&](size_t d) {
    if (d == rv.size()) {
      if (eval_where(where, vars))
        fn(vars);
      return;
    }
    for (uint32_t v = rv[d].lo; v < rv[d].hi; ++v) {
      vars[rv[d].name] = static_cast<int32_t>(v);
      go(d + 1);
    }
    vars.erase(rv[d].name);
  };
  go(0);
}

std::vector<simdojo::LinkSpec> expand_link(const fb::LinkDef *ld) {
  std::vector<simdojo::LinkSpec> out;
  auto lat = ld->latency();
  auto wt = ld->weight();
  if (ld->pattern() && ld->pattern()->size() > 0) {
    auto p = ld->pattern()->str();
    auto ar = p.find(" -> ");
    if (ar == std::string::npos)
      throw std::runtime_error("Link pattern missing ' -> ': " + p);
    auto sp = p.substr(0, ar), dp = p.substr(ar + 4);
    std::string we = (ld->where_expr() && ld->where_expr()->size()) ? ld->where_expr()->str() : "";
    iterate_for(ld->for_ranges(), we, [&](const std::unordered_map<std::string, int32_t> &v) {
      out.push_back({subst_vars(sp, v), subst_vars(dp, v), lat, wt});
    });
  } else if (ld->src() && ld->dst()) {
    out.push_back({ld->src()->str(), ld->dst()->str(), lat, wt});
  }
  return out;
}

using AsyncResources = std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources>;
using CfgMap = std::unordered_map<std::string, std::string>;
using FactoryFn = std::function<std::unique_ptr<simdojo::Component>(
    const std::string &name, const CfgMap &cfg, simdojo::ExecMode mode, rj_code_arch_t arch,
    rj_code_target_id_t target, amdgpu::GpuMemory *mem, const AsyncResources &resources)>;

std::unordered_map<std::string, FactoryFn> &factories() {
  static std::unordered_map<std::string, FactoryFn> f;
  static bool init = false;
  if (!init) {
    init = true;
    f["composite"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode, rj_code_arch_t,
                        rj_code_target_id_t, amdgpu::GpuMemory *, const AsyncResources &) {
      auto c = std::make_unique<simdojo::CompositeComponent>(n);
      c->set_weight(0);
      return c;
    };
    f["soc"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode mode, rj_code_arch_t arch,
                  rj_code_target_id_t, amdgpu::GpuMemory *mem,
                  const AsyncResources &) -> std::unique_ptr<simdojo::Component> {
      auto soc = std::make_unique<SoC>(n, mem);
      soc->set_arch(arch);
      soc->set_exec_mode(mode);
      return soc;
    };
    f["xcd"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode, rj_code_arch_t,
                  rj_code_target_id_t, amdgpu::GpuMemory *,
                  const AsyncResources &) -> std::unique_ptr<simdojo::Component> {
      return std::make_unique<amdgpu::Xcd>(n);
    };
    f["shader_engine"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode, rj_code_arch_t,
                            rj_code_target_id_t, amdgpu::GpuMemory *,
                            const AsyncResources &) -> std::unique_ptr<simdojo::Component> {
      return std::make_unique<amdgpu::ShaderEngine>(n);
    };

    f["gpu_memory"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode, rj_code_arch_t,
                         rj_code_target_id_t, amdgpu::GpuMemory *,
                         const AsyncResources &) -> std::unique_ptr<simdojo::Component> {
      return std::make_unique<amdgpu::GpuMemory>(n);
    };

    f["iod"] = [](const std::string &n, const CfgMap &cfg, simdojo::ExecMode, rj_code_arch_t,
                  rj_code_target_id_t, amdgpu::GpuMemory *mem,
                  const AsyncResources &) -> std::unique_ptr<simdojo::Component> {
      if (!mem)
        throw std::runtime_error("IOD requires gpu_memory");
      amdgpu::Iod::Config ic{};
      ic.num_hbm_stacks = config_u32(cfg, "num_hbm_stacks", 4);
      return std::make_unique<amdgpu::Iod>(n, ic, mem);
    };

    f["l2_cache"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode, rj_code_arch_t,
                       rj_code_target_id_t, amdgpu::GpuMemory *mem,
                       const AsyncResources &) -> std::unique_ptr<simdojo::Component> {
      auto l2 = std::make_unique<amdgpu::L2Cache>(n);
      l2->set_backing_memory(mem);
      l2->set_legacy_maintenance_memory(mem);
      return l2;
    };

    f["memory_side_cache"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode,
                                rj_code_arch_t, rj_code_target_id_t, amdgpu::GpuMemory *mem,
                                const AsyncResources &) -> std::unique_ptr<simdojo::Component> {
      return std::make_unique<amdgpu::MemorySideCache>(
          n, std::make_shared<amdgpu::DeviceCacheCoherence>(), mem);
    };

    f["command_processor"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode mode,
                                rj_code_arch_t arch, rj_code_target_id_t, amdgpu::GpuMemory *,
                                const AsyncResources &) -> std::unique_ptr<simdojo::Component> {
      auto cp = std::make_unique<amdgpu::CommandProcessor>(n, mode);
      cp->configure_for_arch(arch);
      return cp;
    };

    f["compute_unit"] = [](const std::string &n, const CfgMap &cfg, simdojo::ExecMode mode,
                           rj_code_arch_t arch, rj_code_target_id_t target, amdgpu::GpuMemory *mem,
                           const AsyncResources &resources) -> std::unique_ptr<simdojo::Component> {
      amdgpu::ComputeUnitCore::Config cc{};
      cc.arch = arch;
      cc.async_resources = resources;
      cc.target = target;
      cc.num_wf_slots = config_u32(cfg, "num_wf_slots", 10);
      cc.sgprs_per_wf = config_u32(cfg, "sgprs_per_wf", default_sgprs_per_wf(arch));
      cc.vgprs_per_wf = config_u32(cfg, "vgprs_per_wf", default_vgprs_per_wf(arch));
      cc.lds_size_kb = config_u32(cfg, "lds_size_kb", 160);
      cc.functional_quantum =
          config_u32(cfg, "functional_quantum", amdgpu::ComputeUnitCore::kFunctionalQuantum);
      return amdgpu::ComputeUnitCore::create(n, cc, mem, nullptr, mode);
    };
  }
  return f;
}

void build_children(simdojo::CompositeComponent *parent,
                    const flatbuffers::Vector<flatbuffers::Offset<fb::ComponentDef>> *children,
                    simdojo::ExecMode mode, rj_code_arch_t arch, rj_code_target_id_t target,
                    amdgpu::GpuMemory *&mem, const AsyncResources &resources) {
  if (!children)
    return;
  for (auto *cd : *children) {
    auto np = cd->name() ? cd->name()->str() : "";
    auto tp = cd->type() ? cd->type()->str() : "composite";
    CfgMap cfg;
    if (cd->config())
      for (auto *e : *cd->config())
        if (e->key() && e->value())
          cfg[e->key()->str()] = e->value()->str();

    auto names = expand_range(np);
    for (auto &n : names) {
      auto &fmap = factories();
      auto it = fmap.find(tp);
      if (it == fmap.end())
        throw std::runtime_error("Unknown type: " + tp);
      auto comp = it->second(n, cfg, mode, arch, target, mem, resources);
      auto *raw = comp.get();
      if (auto *gm = dynamic_cast<amdgpu::GpuMemory *>(raw))
        mem = gm;
      parent->add_child(std::move(comp));
      if (cd->children() && cd->children()->size() > 0 && raw->is_composite())
        build_children(static_cast<simdojo::CompositeComponent *>(raw), cd->children(), mode, arch,
                       target, mem, resources);
    }
  }
}

void do_wire_cps(simdojo::CompositeComponent *root) {
  std::vector<simdojo::Component *> all;
  root->collect_components(all);
  for (auto *comp : all) {
    auto *cp = dynamic_cast<amdgpu::CommandProcessor *>(comp);
    if (!cp)
      continue;
    auto *par = static_cast<simdojo::CompositeComponent *>(cp->parent());
    if (!par)
      continue;

    // Preserve the physical SE/CU hierarchy when flattening the XCD's CUs into
    // the command processor. gfx12 CWSR records identify scratch with a shader
    // engine plus a per-SE scoreboard slot, so a flat CU ordinal is not enough.
    uint32_t shader_engine_id = 0;
    bool found_shader_engine = false;
    for (auto &child : par->children()) {
      auto *se = dynamic_cast<amdgpu::ShaderEngine *>(child.get());
      if (!se)
        continue;
      found_shader_engine = true;
      uint32_t cu_index = 0;
      for (auto &se_child : se->children()) {
        auto *cu = dynamic_cast<amdgpu::ComputeUnitCore *>(se_child.get());
        if (!cu)
          continue;
        cu->set_shader_engine_location(shader_engine_id, cu_index++);
        cp->add_compute_unit(cu);
      }
      ++shader_engine_id;
    }

    // Keep custom topologies with CUs directly below the XCD working. They
    // form one shader engine and retain their child order as scoreboard order.
    if (!found_shader_engine) {
      uint32_t cu_index = 0;
      for (auto &child : par->children()) {
        auto *cu = dynamic_cast<amdgpu::ComputeUnitCore *>(child.get());
        if (!cu)
          continue;
        cu->set_shader_engine_location(0, cu_index++);
        cp->add_compute_unit(cu);
      }
    }
  }
}

/// @brief Register each shader engine's SPI with its parent XCD's command
/// processor. Must be called AFTER CUs are added to SEs (so the lazily
/// created SPI captures the correct CU list).
void wire_spi_to_cp(simdojo::CompositeComponent *root) {
  std::vector<simdojo::Component *> all;
  root->collect_components(all);
  for (auto *comp : all) {
    auto *cp = dynamic_cast<amdgpu::CommandProcessor *>(comp);
    if (!cp)
      continue;
    auto *par = static_cast<simdojo::CompositeComponent *>(cp->parent());
    if (!par)
      continue;
    std::vector<simdojo::Component *> sub;
    par->collect_components(sub);
    for (auto *s : sub) {
      if (auto *se = dynamic_cast<amdgpu::ShaderEngine *>(s))
        cp->add_spi(&se->spi());
    }
  }
}

void set_cu_l2(simdojo::CompositeComponent *root) {
  std::vector<simdojo::Component *> all;
  root->collect_components(all);
  for (auto *comp : all) {
    auto *cu = dynamic_cast<amdgpu::ComputeUnitCore *>(comp);
    if (!cu)
      continue;
    auto *se = cu->parent();
    if (!se)
      continue;
    auto *xcd = se->parent();
    if (!xcd || !static_cast<simdojo::Component *>(xcd)->is_composite())
      continue;
    for (auto &ch : static_cast<simdojo::CompositeComponent *>(xcd)->children())
      if (auto *l2 = dynamic_cast<amdgpu::L2Cache *>(ch.get())) {
        cu->set_l2(l2);
        break;
      }
  }
}

TopologyBuildResult build_topology(const fb::TopologyDef *topology_def, simdojo::ExecMode mode,
                                   rj_code_arch_t arch, rj_code_target_id_t target,
                                   const AsyncResources &resources) {
  if (!topology_def || !topology_def->root())
    throw std::runtime_error("TopologyDef missing root ComponentDef");

  TopologyBuildResult result;
  auto *rd = topology_def->root();
  auto rn = rd->name() ? rd->name()->str() : "vm";
  auto rt = rd->type() ? rd->type()->str() : "composite";

  CfgMap root_cfg;
  if (rd->config())
    for (auto *e : *rd->config())
      if (e->key() && e->value())
        root_cfg[e->key()->str()] = e->value()->str();

  auto &f = factories();
  auto it = f.find(rt);
  if (it == f.end())
    throw std::runtime_error("Unknown component type for root: " + rt);
  auto root_comp = it->second(rn, root_cfg, mode, arch, target, nullptr, resources);
  auto *root = dynamic_cast<simdojo::CompositeComponent *>(root_comp.get());
  if (!root)
    throw std::runtime_error("Root component must be a CompositeComponent");
  auto root_owner = std::unique_ptr<simdojo::CompositeComponent>(
      static_cast<simdojo::CompositeComponent *>(root_comp.release()));

  amdgpu::GpuMemory *mem = nullptr;
  build_children(root, rd->children(), mode, arch, target, mem, resources);

  if (!mem) {
    std::vector<simdojo::Component *> all;
    root->collect_components(all);
    for (auto *c : all)
      if (auto *m = dynamic_cast<amdgpu::GpuMemory *>(c)) {
        mem = m;
        break;
      }
  }
  if (!mem)
    throw std::runtime_error("Topology must contain a 'gpu_memory' component");

  {
    std::vector<simdojo::Component *> all;
    root->collect_components(all);
    for (auto *c : all) {
      if (auto *cu = dynamic_cast<amdgpu::ComputeUnitCore *>(c))
        cu->set_memory(mem);
    }
  }

  set_cu_l2(root);
  do_wire_cps(root);

  if (topology_def->links())
    for (auto *ld : *topology_def->links())
      for (auto &ls : expand_link(ld))
        result.link_specs.push_back(std::move(ls));

  {
    std::vector<simdojo::Component *> all;
    root->collect_components(all);
    auto *soc = dynamic_cast<SoC *>(root);

    for (auto *c : all) {
      if (auto *xcd = dynamic_cast<amdgpu::Xcd *>(c)) {
        result.xcds.push_back(xcd);

        amdgpu::CommandProcessor *xcd_cp = nullptr;
        amdgpu::L2Cache *xcd_l2 = nullptr;
        for (auto &ch : xcd->children()) {
          if (auto *cp = dynamic_cast<amdgpu::CommandProcessor *>(ch.get())) {
            xcd->set_command_processor(cp);
            xcd_cp = cp;
          } else if (auto *l2 = dynamic_cast<amdgpu::L2Cache *>(ch.get())) {
            xcd->set_l2_cache(l2);
            xcd_l2 = l2;
          } else if (auto *se = dynamic_cast<amdgpu::ShaderEngine *>(ch.get())) {
            xcd->add_shader_engine(se);
            for (auto &sch : se->children())
              if (auto *cu = dynamic_cast<amdgpu::ComputeUnitCore *>(sch.get()))
                se->add_compute_unit(cu);
          }
        }
        // Register the XCD's L2 with its CP so SDMA cache maintenance
        // (flush_gpu_caches/invalidate_gpu_caches) operates on it. The Xcd full
        // constructor wires this, but the config-driven path builds children
        // from the topology and must wire it here.
        if (xcd_cp && xcd_l2)
          xcd_cp->add_l2_cache(xcd_l2);
        if (soc)
          soc->add_xcd(xcd);
      } else if (auto *iod = dynamic_cast<amdgpu::Iod *>(c)) {
        if (soc)
          soc->add_iod(iod);
      }
    }
    result.num_xcds = static_cast<uint32_t>(result.xcds.size());
    if (soc && !soc->set_memory(mem))
      throw std::logic_error("cannot replace GPU memory while address spaces are active");
  }

  // Wire SPIs after CUs are added to SEs (above), so the lazily created
  // SPI captures the correct CU list.
  wire_spi_to_cp(root);

  result.root = std::move(root_owner);
  result.memory = mem;
  return result;
}

struct ExecutionTopology {
  uint32_t xcds = 0;
  uint32_t dispatch_capacity = 1;
};

ExecutionTopology execution_topology(const fb::ComponentDef *root) {
  auto type = [](const fb::ComponentDef *c) -> std::string_view {
    return c->type() ? c->type()->string_view() : "composite";
  };
  auto count = [](const fb::ComponentDef *c) -> uint64_t {
    return expand_range(c->name() ? c->name()->str() : "").size();
  };
  auto checked = [](uint64_t n) {
    if (n > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument("Topology execution capacity exceeds uint32 range");
    return static_cast<uint32_t>(n);
  };
  ExecutionTopology result;
  auto visit = [&](auto &&self, const fb::ComponentDef *node, uint64_t copies) -> void {
    if (!copies)
      return;
    if (type(node) == "xcd")
      result.xcds = checked(result.xcds + copies);
    if (!node->children())
      return;
    bool has_cp = false, has_se = false;
    uint64_t direct_cus = 0, se_cus = 0;
    for (auto *child : *node->children()) {
      const uint64_t n = count(child);
      has_cp |= type(child) == "command_processor" && n != 0;
      if (type(child) == "compute_unit")
        direct_cus += n;
      if (type(child) == "shader_engine" && n) {
        has_se = true;
        if (child->children())
          for (auto *cu : *child->children())
            if (type(cu) == "compute_unit")
              se_cus += n * count(cu);
      }
      self(self, child, checked(copies * n));
    }
    // Match do_wire_cps(): a CP drains its sibling SEs, or direct sibling CUs
    // when the parent has no shader engines.
    if (has_cp)
      result.dispatch_capacity =
          std::max(result.dispatch_capacity, checked(has_se ? se_cus : direct_cus));
  };
  if (root)
    visit(visit, root, 1);
  return result;
}

ExecutionThreadSettings execution_thread_settings(const fb::SimulationConfig *config) {
  ExecutionThreadSettings settings;
  settings.request = {
      config->cpu_thread_budget(), config->num_threads(),
      flatbuffers::IsFieldPresent(config, fb::SimulationConfig::VT_CPU_DISPATCH_THREADS)
          ? config->cpu_dispatch_threads()
          : 0,
      config->async_helper_threads()};
  if (!config->vm() || !config->vm()->arch() || !config->topology())
    throw std::invalid_argument("Thread allocation requires vm.arch and topology");
  settings.arch = parse_arch(config->vm()->arch()->str());
  if (settings.arch == ROCJITSU_CODE_ARCH_INVALID)
    throw std::invalid_argument("Invalid vm.arch for thread allocation");
  const auto dimensions = execution_topology(config->topology()->root());
  // Match build_from_fb(): extra SoCs require a device identity to replicate.
  const auto *gpu = config->vm()->gpu();
  const uint32_t gpus = gpu && gpu->device() ? std::max(1u, gpu->num_gpus()) : 1u;
  if (uint64_t{dimensions.xcds} * gpus > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("Total XCD count exceeds uint32 range");
  settings.xcds = dimensions.xcds * gpus;
  if (config->thread_allocations())
    for (const auto *choice : *config->thread_allocations())
      settings.choices.push_back(
          {choice->num_threads(), choice->cpu_dispatch_threads(), choice->async_helper_threads()});
  settings.dispatch_capacities.assign(gpus, dimensions.dispatch_capacity);
  settings.async_supported = configured_async_mma_supported(settings.arch);
  settings.clocked = exec_mode_from_fb(config) == simdojo::ExecMode::CLOCKED;
  return settings;
}

LoadedConfig build_from_fb(const rocjitsu::fb::SimulationConfig *fb_config, uint32_t host_threads) {
  LoadedConfig result;
  result.engine_config = engine_config_from_fb(fb_config);
  result.exec_mode = exec_mode_from_fb(fb_config);
  result.cpu_dispatch_threads =
      flatbuffers::IsFieldPresent(fb_config, fb::SimulationConfig::VT_CPU_DISPATCH_THREADS)
          ? fb_config->cpu_dispatch_threads()
          : 0;
  result.cpu_thread_budget = fb_config->cpu_thread_budget();
  result.async_helper_threads = fb_config->async_helper_threads();
  result.requested_engine_threads = result.engine_config.num_threads;

  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  if (fb_config->vm() && fb_config->vm()->arch())
    arch = parse_arch(fb_config->vm()->arch()->str());
  if (arch == ROCJITSU_CODE_ARCH_INVALID)
    throw std::runtime_error("Missing or invalid 'arch' in vm configuration");

  const IsaTargetRegistry &registry = default_isa_target_registry();
  const IsaTargetDescriptor *architecture = registry.find(arch);
  const IsaGpuTargetDescription *target = nullptr;
  if (fb_config->vm()->target() && !fb_config->vm()->target()->str().empty()) {
    const std::string target_name = fb_config->vm()->target()->str();
    target = registry.find_gpu_target_by_code_object_id(target_name);
    if (target == nullptr)
      throw std::runtime_error("Missing or invalid 'target' in vm configuration: " + target_name);
  } else if (architecture != nullptr &&
             architecture->default_gpu_target != ROCJITSU_CODE_TARGET_INVALID) {
    target = registry.find_default_gpu_target(*architecture);
    if (target == nullptr)
      throw std::runtime_error("selected architecture has no concrete default target");
  }
  const IsaTargetDescriptor *target_architecture = nullptr;
  if (target != nullptr) {
    target_architecture = registry.find(target->public_id);
    if (target_architecture == nullptr || target_architecture->architecture_id != arch)
      throw std::runtime_error("vm target does not belong to the selected architecture");
    result.target = target->public_id;
  }

  // Extract KFD device identity before topology construction so target/version
  // mismatches fail before any simulator components are exposed.
  if (fb_config->vm()->gpu() && fb_config->vm()->gpu()->device())
    result.device = kfd_device_from_fb(fb_config->vm()->gpu()->device(), "vm.gpu.device");
  if (result.device.present)
    (void)resolve_gpu_generation_topology(result.device);
  if (target != nullptr && target->gfx_target_version != 0 && result.device.present &&
      result.device.gfx_target_version != 0 &&
      result.device.gfx_target_version != target->gfx_target_version)
    throw std::runtime_error("vm target does not match device.gfx_target_version");
  if (target != nullptr &&
      (!target->capabilities.execution_implemented || target_architecture == nullptr ||
       !target_architecture->supports_execution))
    throw std::runtime_error("selected GPU target does not implement simulator execution");

  auto *topo_def = fb_config->topology();
  if (!topo_def)
    throw std::runtime_error("Config missing 'topology' section");

  if (fb_config->vm()->gpu())
    result.num_gpus = std::max(1u, fb_config->vm()->gpu()->num_gpus());
  const auto thread_settings = execution_thread_settings(fb_config);
  result.thread_allocations = thread_settings.choices;
  result.execution_threads = thread_settings.resolve(host_threads);
  if (!result.engine_config.num_threads)
    result.engine_config.num_threads = result.execution_threads.engines;
  result.async_resources = make_async_execution_resources(result.execution_threads.helpers);
  result.build_result =
      build_topology(topo_def, result.exec_mode, arch, result.target, result.async_resources);

  // A config that describes no bus still yields usable defaults, so front ends
  // that attach the GPU to a VMM work without every config being updated.
  if (fb_config->vm() && fb_config->vm()->gpu()) {
    result.pci = pci_device_from_fb(fb_config->vm()->gpu()->pci());
  }

  result.dbt_guest = dbt_guest_from_fb(fb_config->dbt_guest());

  if (result.num_gpus > 1 && result.device.present) {
    result.devices.resize(result.num_gpus);
    for (uint32_t i = 0; i < result.num_gpus; ++i) {
      result.devices[i] = result.device;
      result.devices[i].gpu_id = result.device.gpu_id + i;
      result.devices[i].location_id = 0x0300 + (i << 8);
      result.devices[i].drm_render_minor = 128 + i;
      result.devices[i].unique_id = result.device.unique_id + i;
    }
    for (uint32_t i = 1; i < result.num_gpus; ++i)
      result.extra_gpu_builds.push_back(
          build_topology(topo_def, result.exec_mode, arch, result.target, result.async_resources));
  }

  return result;
}

} // namespace

simdojo::ExecMode parse_exec_mode(const std::string &mode_str) {
  return mode_str == "clocked" ? simdojo::ExecMode::CLOCKED : simdojo::ExecMode::FUNCTIONAL;
}

rj_code_arch_t parse_arch(const std::string &arch_str) {
  // \NPI new ISA family: add its "arch" string here and in arch_to_string().
  if (arch_str == "cdna1")
    return ROCJITSU_CODE_ARCH_CDNA1;
  if (arch_str == "cdna2")
    return ROCJITSU_CODE_ARCH_CDNA2;
  if (arch_str == "cdna3")
    return ROCJITSU_CODE_ARCH_CDNA3;
  if (arch_str == "cdna4")
    return ROCJITSU_CODE_ARCH_CDNA4;
  if (arch_str == "rdna1")
    return ROCJITSU_CODE_ARCH_RDNA1;
  if (arch_str == "rdna2")
    return ROCJITSU_CODE_ARCH_RDNA2;
  if (arch_str == "rdna3")
    return ROCJITSU_CODE_ARCH_RDNA3;
  if (arch_str == "rdna3_5" || arch_str == "rdna3.5")
    return ROCJITSU_CODE_ARCH_RDNA3_5;
  if (arch_str == "rdna4")
    return ROCJITSU_CODE_ARCH_RDNA4;
  if (arch_str == "cdna5")
    return ROCJITSU_CODE_ARCH_CDNA5;
  if (arch_str == "rv32i")
    return ROCJITSU_CODE_ARCH_RV32I;
  if (arch_str == "rv64i")
    return ROCJITSU_CODE_ARCH_RV64I;
  return ROCJITSU_CODE_ARCH_INVALID;
}

const char *arch_to_string(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return "cdna1";
  case ROCJITSU_CODE_ARCH_CDNA2:
    return "cdna2";
  case ROCJITSU_CODE_ARCH_CDNA3:
    return "cdna3";
  case ROCJITSU_CODE_ARCH_CDNA4:
    return "cdna4";
  case ROCJITSU_CODE_ARCH_RDNA1:
    return "rdna1";
  case ROCJITSU_CODE_ARCH_RDNA2:
    return "rdna2";
  case ROCJITSU_CODE_ARCH_RDNA3:
    return "rdna3";
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return "rdna3_5";
  case ROCJITSU_CODE_ARCH_RDNA4:
    return "rdna4";
  case ROCJITSU_CODE_ARCH_CDNA5:
    return "cdna5";
  case ROCJITSU_CODE_ARCH_RV32I:
    return "rv32i";
  case ROCJITSU_CODE_ARCH_RV64I:
    return "rv64i";
  default:
    return "invalid";
  }
}

DeviceIdentityConfig load_device_identity(const std::string &json_path,
                                          const std::string &schema_text) {
  const std::string json = read_config_file(json_path);
  return with_parsed_simulation_config_json(
      json, schema_text, [](const fb::SimulationConfig *config) {
        DeviceIdentityConfig identity;
        if (config->vm() == nullptr || config->vm()->gpu() == nullptr) {
          return identity;
        }
        identity.device = kfd_device_from_fb(config->vm()->gpu()->device(), "vm.gpu.device");
        if (identity.device.present)
          (void)resolve_gpu_generation_topology(identity.device);
        identity.pci = pci_device_from_fb(config->vm()->gpu()->pci());
        return identity;
      });
}

ExecutionThreadSettings load_execution_thread_settings(const std::string &json_path,
                                                       const std::string &schema_text) {
  return with_parsed_simulation_config_json(read_config_file(json_path), schema_text,
                                            execution_thread_settings);
}

LoadedConfig load_config(const std::string &json_path, const std::string &schema_text,
                         uint32_t host_threads) {
  std::string json_text = read_config_file(json_path);
  return with_parsed_simulation_config_json(json_text, schema_text,
                                            [host_threads](const fb::SimulationConfig *fb_config) {
                                              return build_from_fb(fb_config, host_threads);
                                            });
}

LoadedConfig load_config(const std::string &json_path, const std::string &schema_text) {
  return load_config(json_path, schema_text, amdgpu::available_host_threads());
}

LoadedConfig load_config_from_string(const std::string &json, const std::string &schema_text,
                                     uint32_t host_threads) {
  return with_parsed_simulation_config_json(json, schema_text,
                                            [host_threads](const fb::SimulationConfig *fb_config) {
                                              return build_from_fb(fb_config, host_threads);
                                            });
}

LoadedConfig load_config_from_string(const std::string &json, const std::string &schema_text) {
  return load_config_from_string(json, schema_text, amdgpu::available_host_threads());
}

} // namespace config
} // namespace rocjitsu
