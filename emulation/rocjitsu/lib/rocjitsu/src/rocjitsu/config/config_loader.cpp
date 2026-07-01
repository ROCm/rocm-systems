// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/config_loader.h"

#include "rocjitsu/vm/virtual_machine.h"

#include "rocjitsu/vm/amdgpu/command_processor.h"

rocjitsu::SoC *rocjitsu::config::LoadedConfig::soc() {
  return dynamic_cast<SoC *>(build_result.root.get());
}
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/hbm_controller.h"
#include "rocjitsu/vm/amdgpu/iod.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_side_cache.h"
#include "rocjitsu/vm/amdgpu/shader_engine.h"
#include "rocjitsu/vm/amdgpu/xcd.h"

#include "flatbuffers/idl.h"
#include "simdojo/sim/exec_mode.h"
#include "simdojo/sim/topology.h"
#include "simulation_config_generated.h"

#include <cassert>
#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace config {

namespace {

std::string read_file(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open file: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

simdojo::SimulationEngine::Config
engine_config_from_fb(const rocjitsu::fb::SimulationConfig *fb_config) {
  simdojo::SimulationEngine::Config cfg{};
  cfg.max_ticks = fb_config->max_ticks();
  cfg.num_threads = fb_config->num_threads();
  return cfg;
}

simdojo::ExecMode parse_exec_mode(const rocjitsu::fb::SimulationConfig *fb_config) {
  if (fb_config->exec_mode()) {
    std::string mode_str = fb_config->exec_mode()->str();
    if (mode_str == "clocked")
      return simdojo::ExecMode::CLOCKED;
  }
  return simdojo::ExecMode::FUNCTIONAL;
}

const rocjitsu::fb::SimulationConfig *
parse_json(const std::string &json, const std::string &schema_text, flatbuffers::Parser &parser) {
  parser.opts.skip_unexpected_fields_in_json = true;
  if (!parser.Parse(schema_text.c_str()))
    throw std::runtime_error("Failed to parse schema: " + std::string(parser.error_));
  if (!parser.Parse(json.c_str()))
    throw std::runtime_error("Failed to parse JSON config: " + std::string(parser.error_));
  return flatbuffers::GetRoot<rocjitsu::fb::SimulationConfig>(parser.builder_.GetBufferPointer());
}

uint32_t config_u32(const std::unordered_map<std::string, std::string> &cfg, const std::string &key,
                    uint32_t def = 0) {
  auto it = cfg.find(key);
  if (it == cfg.end())
    return def;
  return static_cast<uint32_t>(std::stoul(it->second));
}

uint32_t default_sgprs_per_wf(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250)
    return 128;
  return 112;
}

uint32_t default_vgprs_per_wf(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
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

using CfgMap = std::unordered_map<std::string, std::string>;
using FactoryFn = std::function<std::unique_ptr<simdojo::Component>(
    const std::string &name, const CfgMap &cfg, simdojo::ExecMode mode, rj_code_arch_t arch,
    amdgpu::GpuMemory *mem)>;

std::unordered_map<std::string, FactoryFn> &factories() {
  static std::unordered_map<std::string, FactoryFn> f;
  static bool init = false;
  if (!init) {
    init = true;
    f["composite"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode, rj_code_arch_t,
                        amdgpu::GpuMemory *) {
      auto c = std::make_unique<simdojo::CompositeComponent>(n);
      c->set_weight(0);
      return c;
    };
    f["soc"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode, rj_code_arch_t arch,
                  amdgpu::GpuMemory *mem) -> std::unique_ptr<simdojo::Component> {
      auto soc = std::make_unique<SoC>(n, mem);
      soc->set_arch(arch);
      return soc;
    };
    f["xcd"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode, rj_code_arch_t,
                  amdgpu::GpuMemory *) -> std::unique_ptr<simdojo::Component> {
      return std::make_unique<amdgpu::Xcd>(n);
    };
    f["shader_engine"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode, rj_code_arch_t,
                            amdgpu::GpuMemory *) -> std::unique_ptr<simdojo::Component> {
      return std::make_unique<amdgpu::ShaderEngine>(n);
    };

    f["gpu_memory"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode, rj_code_arch_t,
                         amdgpu::GpuMemory *) -> std::unique_ptr<simdojo::Component> {
      return std::make_unique<amdgpu::GpuMemory>(n);
    };

    f["iod"] = [](const std::string &n, const CfgMap &cfg, simdojo::ExecMode, rj_code_arch_t,
                  amdgpu::GpuMemory *mem) -> std::unique_ptr<simdojo::Component> {
      if (!mem)
        throw std::runtime_error("IOD requires gpu_memory");
      amdgpu::Iod::Config ic{};
      ic.num_hbm_stacks = config_u32(cfg, "num_hbm_stacks", 4);
      return std::make_unique<amdgpu::Iod>(n, ic, mem);
    };

    f["l2_cache"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode, rj_code_arch_t,
                       amdgpu::GpuMemory *mem) -> std::unique_ptr<simdojo::Component> {
      auto l2 = std::make_unique<amdgpu::L2Cache>(n);
      l2->set_backing_memory(mem);
      return l2;
    };

    f["memory_side_cache"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode,
                                rj_code_arch_t,
                                amdgpu::GpuMemory *) -> std::unique_ptr<simdojo::Component> {
      return std::make_unique<amdgpu::MemorySideCache>(n);
    };

    f["command_processor"] = [](const std::string &n, const CfgMap &, simdojo::ExecMode,
                                rj_code_arch_t arch,
                                amdgpu::GpuMemory *) -> std::unique_ptr<simdojo::Component> {
      auto cp = std::make_unique<amdgpu::CommandProcessor>(n);
      // Matches llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
      // AMDGPUBaseInfo::getVGPREncodingGranule():
      // gfx1250 has Feature1024AddressableVGPRs, so Wave32 descriptors encode
      // VGPR counts in 16-register blocks; other RDNA Wave32 targets use 8.
      // LLVM's AMDGPULowerVGPREncoding.cpp handles the separate gfx1250
      // s_set_vgpr_msb high-bank indexing needed to access VGPRs above v255.
      uint32_t gran = 4;
      if (arch == ROCJITSU_CODE_ARCH_GFX1250)
        gran = 16;
      else if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
               arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
               arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
               arch == ROCJITSU_CODE_ARCH_RDNA4)
        gran = 8;
      cp->set_vgpr_granularity(gran);
      bool packed = (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
                     arch == ROCJITSU_CODE_ARCH_GFX1250);
      cp->set_packed_tid(packed);
      const bool gfx11_plus_sdma =
          arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
          arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250;
      cp->set_sdma_packet_dialect(gfx11_plus_sdma ? amdgpu::SdmaPacketDialect::Gfx11Plus
                                                  : amdgpu::SdmaPacketDialect::Legacy);
      return cp;
    };

    f["compute_unit"] = [](const std::string &n, const CfgMap &cfg, simdojo::ExecMode mode,
                           rj_code_arch_t arch,
                           amdgpu::GpuMemory *mem) -> std::unique_ptr<simdojo::Component> {
      amdgpu::ComputeUnitCore::Config cc{};
      cc.arch = arch;
      cc.num_wf_slots = config_u32(cfg, "num_wf_slots", 10);
      cc.sgprs_per_wf = config_u32(cfg, "sgprs_per_wf", default_sgprs_per_wf(arch));
      cc.vgprs_per_wf = config_u32(cfg, "vgprs_per_wf", default_vgprs_per_wf(arch));
      cc.lds_size_kb = config_u32(cfg, "lds_size_kb", 160);
      return amdgpu::ComputeUnitCore::create(n, cc, mem, nullptr, mode);
    };
  }
  return f;
}

void build_children(simdojo::CompositeComponent *parent,
                    const flatbuffers::Vector<flatbuffers::Offset<fb::ComponentDef>> *children,
                    simdojo::ExecMode mode, rj_code_arch_t arch, amdgpu::GpuMemory *&mem) {
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
      auto comp = it->second(n, cfg, mode, arch, mem);
      auto *raw = comp.get();
      if (auto *gm = dynamic_cast<amdgpu::GpuMemory *>(raw))
        mem = gm;
      parent->add_child(std::move(comp));
      if (cd->children() && cd->children()->size() > 0 && raw->is_composite())
        build_children(static_cast<simdojo::CompositeComponent *>(raw), cd->children(), mode, arch,
                       mem);
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
    std::vector<simdojo::Component *> sub;
    par->collect_components(sub);
    for (auto *s : sub) {
      if (auto *cu = dynamic_cast<amdgpu::ComputeUnitCore *>(s))
        cp->add_compute_unit(cu);
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
                                   rj_code_arch_t arch) {
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
  auto root_comp = it->second(rn, root_cfg, mode, arch, nullptr);
  auto *root = dynamic_cast<simdojo::CompositeComponent *>(root_comp.get());
  if (!root)
    throw std::runtime_error("Root component must be a CompositeComponent");
  auto root_owner = std::unique_ptr<simdojo::CompositeComponent>(
      static_cast<simdojo::CompositeComponent *>(root_comp.release()));

  amdgpu::GpuMemory *mem = nullptr;
  build_children(root, rd->children(), mode, arch, mem);

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
      if (auto *cp = dynamic_cast<amdgpu::CommandProcessor *>(c))
        cp->set_memory(mem);
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
        if (soc)
          soc->add_xcd(xcd);

        for (auto &ch : xcd->children()) {
          if (auto *cp = dynamic_cast<amdgpu::CommandProcessor *>(ch.get()))
            xcd->set_command_processor(cp);
          else if (auto *l2 = dynamic_cast<amdgpu::L2Cache *>(ch.get()))
            xcd->set_l2_cache(l2);
          else if (auto *se = dynamic_cast<amdgpu::ShaderEngine *>(ch.get())) {
            xcd->add_shader_engine(se);
            for (auto &sch : se->children())
              if (auto *cu = dynamic_cast<amdgpu::ComputeUnitCore *>(sch.get()))
                se->add_compute_unit(cu);
          }
        }
      } else if (auto *iod = dynamic_cast<amdgpu::Iod *>(c)) {
        if (soc)
          soc->add_iod(iod);
      }
    }
    result.num_xcds = static_cast<uint32_t>(result.xcds.size());
    if (soc)
      soc->set_memory(mem);
  }

  // Wire SPIs after CUs are added to SEs (above), so the lazily created
  // SPI captures the correct CU list.
  wire_spi_to_cp(root);

  result.root = std::move(root_owner);
  result.memory = mem;
  return result;
}

LoadedConfig build_from_fb(const rocjitsu::fb::SimulationConfig *fb_config) {
  LoadedConfig result;
  result.engine_config = engine_config_from_fb(fb_config);
  result.exec_mode = parse_exec_mode(fb_config);

  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  if (fb_config->vm() && fb_config->vm()->arch())
    arch = parse_arch(fb_config->vm()->arch()->str());
  if (arch == ROCJITSU_CODE_ARCH_INVALID)
    throw std::runtime_error("Missing or invalid 'arch' in vm configuration");

  auto *topo_def = fb_config->topology();
  if (!topo_def)
    throw std::runtime_error("Config missing 'topology' section");

  result.build_result = build_topology(topo_def, result.exec_mode, arch);

  // Extract KFD device identity from vm.gpu.device if present.
  if (fb_config->vm() && fb_config->vm()->gpu() && fb_config->vm()->gpu()->device()) {
    auto *d = fb_config->vm()->gpu()->device();
    auto &dev = result.device;
    dev.present = true;
    dev.gpu_id = d->gpu_id();
    dev.gfx_target_version = d->gfx_target_version();
    dev.vendor_id = d->vendor_id();
    dev.device_id = d->device_id();
    dev.family_id = d->family_id();
    dev.unique_id = d->unique_id();
    if (d->marketing_name())
      dev.marketing_name = d->marketing_name()->str();
    dev.drm_render_minor = d->drm_render_minor();
    dev.revision_id = d->revision_id();
    dev.pci_revision_id = d->pci_revision_id();
    dev.simd_count = d->simd_count();
    dev.max_waves_per_simd = d->max_waves_per_simd();
    dev.num_shader_engines = d->num_shader_engines();
    dev.num_shader_arrays_per_engine = d->num_shader_arrays_per_engine();
    dev.num_cu_per_sh = d->num_cu_per_sh();
    dev.simd_per_cu = d->simd_per_cu();
    dev.wave_front_size = d->wave_front_size();
    dev.max_slots_scratch_cu = d->max_slots_scratch_cu();
    dev.local_mem_size = d->local_mem_size();
    dev.vram_type = d->vram_type();
    dev.lds_size_kb = d->lds_size_kb();
    dev.mem_width = d->mem_width();
    dev.mem_clk_max = d->mem_clk_max();
    dev.l1_size_kb = d->l1_size_kb();
    dev.l1_line_size = d->l1_line_size();
    dev.l1_assoc = d->l1_assoc();
    dev.l2_size_kb = d->l2_size_kb();
    dev.l2_line_size = d->l2_line_size();
    dev.l2_assoc = d->l2_assoc();
    dev.num_sdma_engines = d->num_sdma_engines();
    dev.num_sdma_xgmi_engines = d->num_sdma_xgmi_engines();
    dev.num_cp_queues = d->num_cp_queues();
    dev.max_engine_clk_fcompute = d->max_engine_clk_fcompute();
    dev.location_id = d->location_id();
    dev.hive_id = d->hive_id();
    dev.domain = d->domain();
  }

  if (fb_config->vm() && fb_config->vm()->gpu())
    result.num_gpus = std::max(1u, fb_config->vm()->gpu()->num_gpus());

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
      result.extra_gpu_builds.push_back(build_topology(topo_def, result.exec_mode, arch));
  }

  return result;
}

} // namespace

rj_code_arch_t parse_arch(const std::string &arch_str) {
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
  if (arch_str == "gfx1250")
    return ROCJITSU_CODE_ARCH_GFX1250;
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
  case ROCJITSU_CODE_ARCH_GFX1250:
    return "gfx1250";
  case ROCJITSU_CODE_ARCH_RV32I:
    return "rv32i";
  case ROCJITSU_CODE_ARCH_RV64I:
    return "rv64i";
  default:
    return "invalid";
  }
}

LoadedConfig load_config(const std::string &json_path, const std::string &schema_text) {
  std::string json_text = read_file(json_path);
  flatbuffers::Parser parser;
  auto *fb_config = parse_json(json_text, schema_text, parser);
  return build_from_fb(fb_config);
}

LoadedConfig load_config_from_string(const std::string &json, const std::string &schema_text) {
  flatbuffers::Parser parser;
  auto *fb_config = parse_json(json, schema_text, parser);
  return build_from_fb(fb_config);
}

} // namespace config
} // namespace rocjitsu
