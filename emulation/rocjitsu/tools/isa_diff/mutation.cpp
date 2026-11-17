// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mutation.h"

#include "boundary.h"
#include "effects.h"
#include "input_policy.h"
#include "json.h"
#include "kernel_model.h"
#include "region_runner.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/diagnostic.h"

#include <cctype>
#include <map>
#include <set>

namespace rocjitsu::isa_diff {

std::optional<RegionPair> make_synthetic_mutation(const RegionPair &base, const MutationSpec &spec,
                                                  std::string &error) {
  RegionPair r = base;
  RegionSide &side = spec.new_side ? r.new_side : r.old_side;
  if (spec.inst >= side.insts.size() || spec.word >= side.insts[spec.inst].words.size()) {
    error = "mutation target out of range";
    return std::nullopt;
  }
  if (spec.xor_mask == 0) {
    error = "mutation mask is zero";
    return std::nullopt;
  }
  const std::string before = side.insts[spec.inst].text;
  const bool was_branch = is_region_branch(before.substr(0, before.find(' ')));
  side.insts[spec.inst].words[spec.word] ^= spec.xor_mask;

  const Target target = r.contract.gpu().value_or(Target{});
  auto decoder = Decoder::create(target.arch);
  if (!decoder) {
    error = "no " + target.name + " decoder";
    return std::nullopt;
  }
  LocSet writes, wide;
  VSet vwrites, vnamed;
  for (auto &inst : side.insts) {
    if (inst.inline_asm >= 0) {
      const Effects fx = classify_inline_asm(inst.asm_text);
      writes |= fx.writes;
      vwrites |= fx.vwrites;
      continue;
    }
    std::vector<uint32_t> window = inst.words;
    window.resize(window.size() + 3, 0);
    util::StringDiagnostic diag;
    auto decoded = decoder->decode_window(window, inst.pc, diag.emitter());
    if (decoded.failed()) {
      error = "mutated region does not decode: " + diag.message();
      return std::nullopt;
    }
    const Instruction &d = *decoded.value();
    if (static_cast<size_t>(d.size()) != inst.words.size() * 4) {
      error = "mutation changes the instruction size";
      return std::nullopt;
    }
    const Effects fx = classify(d, target);
    if (!fx.p0) {
      error = "mutated instruction is outside P0: " + fx.reject_reason;
      return std::nullopt;
    }
    // The region's control flow was planned from the original code: a mutant that becomes
    // a branch would leave it through an edge nobody declared.
    if (&inst == &side.insts[spec.inst] && fx.control_flow && !was_branch) {
      error = "mutation turns the instruction into a branch";
      return std::nullopt;
    }
    inst.text = d.disassemble();
    writes |= fx.writes;
    vwrites |= fx.vwrites;
    vnamed |= fx.vreads | fx.vwrites;
    wide |= fx.wide_reads;
  }
  for (unsigned v = r.contract.vgpr_count; v < kNumVgprs; ++v)
    if (vnamed.test(v)) {
      error = "mutated region names v" + std::to_string(v) + ", outside the region's " +
              std::to_string(r.contract.vgpr_count) + " tracked VGPRs";
      return std::nullopt;
    }
  (spec.new_side ? r.contract.declared_writes_new : r.contract.declared_writes_old) = writes;
  (spec.new_side ? r.contract.vwrites_new : r.contract.vwrites_old) = vwrites;
  r.contract.wide_reads |= wide;
  r.synthetic = true;
  r.id = base.id + "-syn-" + spec.label;
  r.contract.sources.push_back(
      {std::string("synthetic mutation on ") + (spec.new_side ? "new" : "old") + " inst " +
           std::to_string(spec.inst) + " word " + std::to_string(spec.word) + " xor " +
           hex32(spec.xor_mask) + " -> '" + side.insts[spec.inst].text + "'",
       "synthetic_injection"});
  r.prepared_key = compute_prepared_key(r);
  return r;
}

namespace {

std::vector<std::string> tokens(const std::string &text) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : text) {
    if (c == ' ' || c == ',' || c == '\t') {
      if (!cur.empty())
        out.push_back(std::move(cur));
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty())
    out.push_back(std::move(cur));
  return out;
}

/// The token without neg/abs decoration ("-|v0|" -> "v0").
std::string bare(const std::string &t) {
  std::string b;
  for (char c : t)
    if (c != '|' && !(b.empty() && c == '-'))
      b.push_back(c);
  return b;
}

bool is_register(const std::string &t) {
  static constexpr std::string_view kNames[] = {
      "vcc", "exec", "m0", "null", "scc", "ttmp", "flat_scratch", "src_", "lds_direct", "off"};
  if (t.size() >= 2 && (t[0] == 'v' || t[0] == 's' || t[0] == 'a') &&
      (std::isdigit(static_cast<unsigned char>(t[1])) || t[1] == '['))
    return true;
  for (std::string_view n : kNames)
    if (t.rfind(n, 0) == 0)
      return true;
  return false;
}

/// The key of a "key:value" operand, or empty (a register range "s[8:9]" has none).
std::string key_of(const std::string &t) {
  const size_t colon = t.find(':');
  if (colon == std::string::npos || colon == 0)
    return {};
  const std::string key = t.substr(0, colon);
  for (char c : key)
    if (!std::islower(static_cast<unsigned char>(c)) &&
        !std::isdigit(static_cast<unsigned char>(c)) && c != '_')
      return {};
  return key;
}

bool is_constant(const std::string &t) {
  return !t.empty() && (std::isdigit(static_cast<unsigned char>(t[0])) || t[0] == '-' ||
                        t[0] == '.' || t.rfind("0x", 0) == 0);
}

std::string unit_of(const std::string &mnemonic) {
  for (std::string_view m :
       {"global_", "s_load", "s_buffer", "buffer_", "ds_", "flat_", "scratch_", "s_store"})
    if (mnemonic.rfind(m, 0) == 0)
      return "memory";
  return mnemonic.rfind("v_", 0) == 0 ? "valu" : "salu";
}

} // namespace

std::string classify_mutation(const std::string &original, const std::string &mutated) {
  const auto a = tokens(original), b = tokens(mutated);
  if (a.empty() || b.empty() || a[0] != b[0])
    return "opcode";
  if (a.size() != b.size())
    return "modifier"; // a flag (clamp, op_sel, ...) appeared or disappeared
  for (size_t i = 1; i < a.size(); ++i) {
    if (a[i] == b[i])
      continue;
    if (bare(a[i]) == bare(b[i]) && is_register(bare(a[i])))
      return "modifier"; // neg or abs of a register (a constant's sign is its value)
    // A "key:value" operand (offset:12, op_sel:[1,0], mul:2); a register range s[8:9] is not.
    const std::string ka = key_of(a[i]), kb = key_of(b[i]);
    if (!ka.empty() || !kb.empty()) {
      const std::string &key = ka.empty() ? kb : ka;
      return key == "offset" || key == "offset0" || key == "offset1" ? "constant" : "modifier";
    }
    // The original operand decides: a register that became a constant is a register
    // mutant, a constant that became a register a constant mutant.
    const std::string x = bare(a[i]), y = bare(b[i]);
    if (is_register(x))
      return "register";
    if (is_constant(x))
      return "constant";
    if (is_register(y))
      return "register";
    return is_constant(y) ? "constant" : "other";
  }
  return "other";
}

std::vector<Mutant> enumerate_mutants(const RegionPair &base) {
  std::vector<Mutant> out;
  std::set<std::string> seen;
  const RegionSide &side = base.new_side;
  for (size_t i = 0; i < side.insts.size(); ++i) {
    const RegionInst &inst = side.insts[i];
    if (inst.inline_asm >= 0 || inst.words.empty())
      continue;
    const auto toks = tokens(inst.text);
    const std::string mnemonic = toks.empty() ? "" : toks[0];
    if (is_region_noop(mnemonic) || is_region_branch(mnemonic))
      continue; // hints have no effect; a retargeted branch leaves the region's CFG
    for (size_t w = 0; w < inst.words.size(); ++w)
      for (unsigned bit = 0; bit < 32; ++bit) {
        MutationSpec spec;
        spec.new_side = true;
        spec.inst = i;
        spec.word = w;
        spec.xor_mask = 1u << bit;
        spec.label = "i" + std::to_string(i) + "w" + std::to_string(w) + "b" + std::to_string(bit);
        std::string error;
        auto r = make_synthetic_mutation(base, spec, error);
        if (!r)
          continue;
        const std::string &text = r->new_side.insts[i].text;
        if (text == inst.text || !seen.insert(std::to_string(i) + ":" + text).second)
          continue;
        Mutant m;
        m.spec = spec;
        m.original = inst.text;
        m.mutated = text;
        m.operator_class = classify_mutation(inst.text, text);
        m.unit = unit_of(mnemonic);
        out.push_back(std::move(m));
      }
  }
  return out;
}

std::vector<Mutant> sample_mutants(std::vector<Mutant> all, size_t max, uint64_t seed) {
  std::map<std::string, std::vector<Mutant>> by_class;
  for (auto &m : all)
    by_class[m.operator_class].push_back(std::move(m));
  for (auto &[name, list] : by_class) {
    // Fisher-Yates with splitmix64: the order depends on the seed and the class only.
    uint64_t state = seed;
    for (char c : name)
      state = splitmix64(state ^ static_cast<uint8_t>(c));
    for (size_t k = list.size(); k > 1; --k) {
      state = splitmix64(state);
      std::swap(list[k - 1], list[state % k]);
    }
  }
  std::vector<Mutant> out;
  for (size_t round = 0; out.size() < max; ++round) {
    bool any = false;
    for (auto &[name, list] : by_class)
      if (round < list.size() && out.size() < max) {
        out.push_back(list[round]);
        any = true;
      }
    if (!any)
      break;
  }
  return out;
}

namespace {

/// The verdict of one test of a mutant: killed (a mismatch, or a clobbered frame field),
/// survived, or invalid (the mutant left the model: undeclared effects, no valid case).
std::string verdict_of(const RegionResult &r) {
  if (r.test_status == TestStatus::LocalMismatch)
    return "killed";
  if (r.test_status == TestStatus::HarnessError)
    return r.mismatch_kinds.count("frame_preservation") ? "killed" : "invalid";
  if (r.test_status == TestStatus::NoMismatchFound)
    return "survived";
  return "invalid";
}

} // namespace

std::pair<LocSet, VSet> region_pending_conflicts(const RegionSide &side, const Target &target) {
  LocSet conflict;
  VSet vconflict;
  auto decoder = Decoder::create(target.arch);
  if (!decoder)
    return {conflict, vconflict};
  PendingState pending;
  for (const RegionInst &inst : side.insts) {
    InstRef ref;
    ref.pc = inst.pc;
    std::vector<uint32_t> window = inst.words; // outlives the decoded instruction below
    std::unique_ptr<Instruction> owned;
    if (inst.inline_asm >= 0) {
      ref.fx = classify_inline_asm(inst.asm_text);
    } else {
      window.resize(window.size() + 3, 0);
      util::StringDiagnostic diag;
      auto decoded = decoder->decode_window(window, inst.pc, diag.emitter());
      if (decoded.failed())
        continue;
      owned = std::move(decoded.value());
      ref.inst = owned.get();
      ref.fx = classify(*owned, target);
    }
    conflict |= pending.all_dests() & (ref.fx.reads | ref.fx.writes);
    vconflict |= pending.all_vdests() & (ref.fx.vreads | ref.fx.vwrites);
    pending = pending_transfer(ref, pending, target.arch);
  }
  return {conflict, vconflict};
}

std::string survivor_effect(const RegionPair &base, const RegionPair &mutant, uint64_t seed,
                            uint64_t cases, uint64_t *first_case, std::string *detail) {
  const RegionContract &k = base.contract;
  const auto target = k.gpu();
  if (!target)
    return "unknown";
  std::string error;
  auto orig = RegionRunner::create(base.new_side.insts, base.id + "_effect_new", error,
                                   k.vgpr_count, *target);
  auto mut = orig ? RegionRunner::create(mutant.new_side.insts, base.id + "_effect_mutant", error,
                                         k.vgpr_count, *target)
                  : nullptr;
  if (!orig || !mut)
    return "unknown";
  auto image = std::make_shared<MemoryImage>();
  image->set_small_scalars(orig->loops() || mut->loops());
  share_code_guard(*orig, *mut);
  orig->set_memory_image(image);
  mut->set_memory_image(image);
  const Frame frame = orig->baseline_frame();
  const auto new_to_old = k.entry_inverse();
  const RegionFeatures features =
      make_region_features(base.old_side.insts, base.new_side.insts,
                           k.entry_map.empty() ? nullptr : &new_to_old, *target);
  const BoundarySweep sweep = build_boundary_sweep(features.roles, k, {seed, 250000});
  // Live at any exit, in new naming (VGPRs are not renamed).
  const auto fwd = k.exit_forward();
  LocSet live;
  VSet vlive = k.vobservable, vactive = k.vactive_only;
  for (unsigned l = 0; l < kNumLocs; ++l)
    if (k.observable.test(l))
      live.set(fwd[l]);
  for (const RegionExit &e : k.exits) {
    for (unsigned l = 0; l < kNumLocs; ++l)
      if (e.observable.test(l))
        live.set(fwd[l]);
    vlive |= e.vobservable;
    vactive |= e.vactive_only;
  }
  // Narrow return values: only these bits are compared (straight regions, as in the test).
  std::array<uint32_t, kNumLocs> smask;
  smask.fill(0xffffffffu);
  std::array<uint32_t, kNumVgprs> vmask;
  vmask.fill(0xffffffffu);
  if (k.exits.empty()) {
    for (const auto &[l, m] : k.exit_masks)
      smask[fwd[l]] = m;
    for (const auto &[r, m] : k.vexit_masks)
      if (r < kNumVgprs)
        vmask[r] = m;
  }
  bool effect = false, undefined = false;
  uint64_t index = 0;
  // The effect of one input: 0 none, 1 dead only, 2 observable.
  auto one = [&](const Sample &sample) {
    if (ConstraintChecker::validate(sample, k))
      return 0;
    MachineState init_old, init_new;
    build_initial_states(k, sample, frame, seed, index, init_old, init_new);
    orig->load(init_new);
    const RunOutcome oa = orig->run(base.new_side.insts.size());
    mut->load(init_new);
    const RunOutcome ob = mut->run(mutant.new_side.insts.size());
    if (oa.status != ob.status)
      return 2;
    if (oa.status != RunStatus::Ok)
      return 0;
    const MachineState a = orig->capture(), b = mut->capture();
    const LocSet dl = differing_locs(a, b);
    const VSet dv = differing_vgprs(a, b);
    std::string why;
    if (a.exit_pc != b.exit_pc)
      why = "exit";
    else if (a.mem_written != b.mem_written)
      why = "stored bytes";
    else if (a.asm_uses != b.asm_uses)
      why = "inline asm use";
    else if (!differing_frame_fields(a.frame, b.frame).empty())
      why = "frame " + differing_frame_fields(a.frame, b.frame).front();
    // A live location that is undefined at the entry and that the original passes through
    // holds an undefined value: any value refines it (LLVM undef).
    bool undefined = false;
    for (unsigned l = 0; why.empty() && l < kNumLocs; ++l) {
      if (!dl.test(l) || !live.test(l) || !((a.get(l) ^ b.get(l)) & smask[l]))
        continue;
      if (k.undef_new.test(l) && !k.declared_writes_new.test(l)) {
        undefined = true;
        continue;
      }
      LocSet one;
      one.set(l);
      why = format_locs(one);
    }
    const uint64_t exec = uint64_t{a.get(kLocExecLo)} | uint64_t{a.get(kLocExecHi)} << 32;
    for (unsigned r = 0; why.empty() && r < kNumVgprs; ++r) {
      if (!dv.test(r) || !vlive.test(r))
        continue;
      uint64_t lanes = differing_lanes(a, b, r);
      if (vactive.test(r))
        lanes &= exec;
      for (unsigned lane = 0; lane < kWaveLanes && why.empty(); ++lane) {
        if (!(lanes >> lane & 1u) || !((a.vget(r, lane) ^ b.vget(r, lane)) & vmask[r]))
          continue;
        if (k.vundef_new.test(r) && !k.vwrites_new.test(r)) {
          undefined = true;
          break;
        }
        why = vgpr_name(r) + " lane " + std::to_string(lane);
      }
    }
    if (!why.empty()) {
      if (detail)
        *detail = why;
      return 2;
    }
    if (undefined)
      return 3;
    return dl.any() || dv.any() ? 1 : 0;
  };
  auto note = [&](int e) {
    effect = effect || e > 0;
    undefined = undefined || e == 3;
    if (e == 2 && first_case)
      *first_case = index;
    return e == 2;
  };
  for (const SweepCase &c : sweep.cases) {
    if (note(one(sweep.materialize(c))))
      return "observable_effect";
    ++index;
  }
  BaselinePolicy policy(seed);
  const History history;
  for (uint64_t n = 0; n < cases; ++n, ++index) {
    Sample sample = policy.propose(features, k, history, index);
    project_onto_constraints(sample, k, seed ^ (index << 1));
    if (note(one(sample)))
      return "observable_effect";
  }
  return undefined ? "undefined_effect" : effect ? "dead_effect" : "no_effect";
}

Json run_mutants(const CaseData &data, const MutantRunOptions &options) {
  TestOptions topt = options.test;
  topt.reduce = false;
  topt.max_counterexamples = 1;
  topt.counterexample_dir.clear();
  topt.case_log_dir.clear();
  topt.cache_dir.clear();
  topt.stop_at_first_mismatch = true;
  Json regions = Json::array();
  std::map<std::string, std::map<std::string, uint64_t>> by_class, by_unit;
  std::map<std::string, uint64_t> totals, effects;
  for (const RegionPair &region : data.regions) {
    if (region.status != RegionStatus::Ready ||
        (!options.region.empty() && region.id != options.region))
      continue;
    Json rj = Json::object();
    rj.set("id", region.id);
    BaselinePolicy policy(topt.seed);
    TestOptions base_opt = topt;
    base_opt.stop_at_first_mismatch = false;
    const RegionResult base = test_region(region, base_opt, policy);
    rj.set("baseline", test_status_name(base.test_status));
    const bool plain =
        base.test_status == TestStatus::NoMismatchFound && base.dont_care_masked == 0;
    rj.set("baseline_masked", base.dont_care_masked > 0);
    Json list = Json::array();
    if (plain) {
      const Target tgt = region.contract.gpu().value_or(Target{});
      const auto [base_conflict, base_vconflict] = region_pending_conflicts(region.new_side, tgt);
      std::vector<Mutant> all = enumerate_mutants(region);
      rj.set("candidates", static_cast<uint64_t>(all.size()));
      const auto chosen = sample_mutants(std::move(all), options.max_per_region, topt.seed);
      for (const Mutant &m : chosen) {
        std::string error;
        auto mutated = make_synthetic_mutation(region, m.spec, error);
        if (!mutated)
          continue;
        Json mj = Json::object();
        mj.set("inst", static_cast<uint64_t>(m.spec.inst));
        mj.set("word", static_cast<uint64_t>(m.spec.word));
        mj.set("xor", hex32(m.spec.xor_mask));
        mj.set("class", m.operator_class);
        mj.set("unit", m.unit);
        mj.set("original", m.original);
        mj.set("mutated", m.mutated);
        // Static first: a new pending-memory conflict leaves the region's context
        // unresolved in the real pipeline, so the mutant is never passed.
        const auto [conflict, vconflict] = region_pending_conflicts(mutated->new_side, tgt);
        if ((conflict & ~base_conflict).any() || (vconflict & ~base_vconflict).any()) {
          mj.set("verdict", "killed");
          mj.set("valid", uint64_t{0});
          mj.set("phase", "static");
          mj.set("kind", "pending_conflict");
          ++by_class[m.operator_class]["killed"];
          ++by_unit[m.unit]["killed"];
          ++totals["killed"];
          list.push(std::move(mj));
          continue;
        }
        BaselinePolicy mpolicy(topt.seed);
        RegionResult res = test_region(*mutated, topt, mpolicy);
        std::string verdict = verdict_of(res);
        if (verdict == "survived" && options.confirm_cases) {
          TestOptions copt = topt;
          copt.cases = options.confirm_cases;
          copt.seed = topt.seed ^ 0x636f6e6669726dull; // other random inputs than the first run
          BaselinePolicy cpolicy(copt.seed);
          RegionResult again = test_region(*mutated, copt, cpolicy);
          const std::string v2 = verdict_of(again);
          if (v2 == "killed") {
            verdict = "killed_late";
            res = std::move(again);
          } else if (v2 == "invalid") {
            verdict = "invalid";
            res = std::move(again);
          }
        }
        mj.set("verdict", verdict);
        mj.set("valid", res.valid_pairs);
        if (!res.examples.empty()) {
          mj.set("phase", res.examples[0].phase);
          mj.set("case", res.examples[0].case_index);
          mj.set("kind", res.examples[0].kind);
        } else if (verdict == "invalid") {
          std::string why = res.reason;
          if (why.empty() && !res.mismatch_kinds.empty())
            why = res.mismatch_kinds.begin()->first;
          mj.set("reason", why);
        }
        if (verdict == "survived" && res.dont_care_masked)
          mj.set("masked", true);
        if (verdict == "survived" && options.effect_cases) {
          uint64_t first = 0;
          std::string where;
          const std::string effect =
              survivor_effect(region, *mutated, topt.seed, options.effect_cases, &first, &where);
          mj.set("effect", effect);
          if (effect == "observable_effect") {
            mj.set("effect_case", first);
            mj.set("effect_detail", where);
          }
          ++effects[effect];
        }
        ++by_class[m.operator_class][verdict];
        ++by_unit[m.unit][verdict];
        ++totals[verdict];
        list.push(std::move(mj));
      }
    }
    rj.set("tested", static_cast<uint64_t>(list.size()));
    rj.set("mutants", std::move(list));
    regions.push(std::move(rj));
  }
  auto counts = [](const std::map<std::string, std::map<std::string, uint64_t>> &m) {
    Json o = Json::object();
    for (const auto &[k, v] : m) {
      Json c = Json::object();
      for (const auto &[verdict, n] : v)
        c.set(verdict, n);
      o.set(k, std::move(c));
    }
    return o;
  };
  Json out = Json::object();
  out.set("schema_version", "rj_isa_diff.mutants.v1");
  Json opts = Json::object();
  opts.set("seed", topt.seed);
  opts.set("cases", topt.cases);
  opts.set("confirm_cases", options.confirm_cases);
  opts.set("effect_cases", options.effect_cases);
  opts.set("max_per_region", static_cast<uint64_t>(options.max_per_region));
  out.set("options", std::move(opts));
  Json t = Json::object();
  for (const auto &[verdict, n] : totals)
    t.set(verdict, n);
  out.set("totals", std::move(t));
  Json e = Json::object();
  for (const auto &[effect, n] : effects)
    e.set(effect, n);
  out.set("survivor_effects", std::move(e));
  out.set("by_class", counts(by_class));
  out.set("by_unit", counts(by_unit));
  out.set("regions", std::move(regions));
  return out;
}

} // namespace rocjitsu::isa_diff
