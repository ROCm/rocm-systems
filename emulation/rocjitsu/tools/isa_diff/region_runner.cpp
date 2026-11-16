// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "region_runner.h"

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "util/diagnostic.h"

#include "effects.h"
#include "input_policy.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <exception>
#include <set>
#include <unordered_set>

namespace rocjitsu::isa_diff {
namespace {

bool f32_nan(uint32_t v) { return (v & 0x7f800000u) == 0x7f800000u && (v & 0x7fffffu); }
bool f32_inf(uint32_t v) { return (v & 0x7fffffffu) == 0x7f800000u; }
bool f16_nan(uint32_t h) { return (h & 0x7c00u) == 0x7c00u && (h & 0x3ffu); }
bool f16_inf(uint32_t h) { return (h & 0x7fffu) == 0x7c00u; }
bool bf16_nan(uint32_t h) { return (h & 0x7f80u) == 0x7f80u && (h & 0x7fu); }
bool bf16_inf(uint32_t h) { return (h & 0x7fffu) == 0x7f80u; }
bool f64_nan(uint64_t v) {
  return (v & 0x7ff0000000000000ull) == 0x7ff0000000000000ull && (v & 0xfffffffffffffull);
}
bool f64_inf(uint64_t v) { return (v & 0x7fffffffffffffffull) == 0x7ff0000000000000ull; }

double f16_value(uint32_t h) {
  const int e = static_cast<int>((h >> 10) & 0x1f);
  const double m = static_cast<double>(h & 0x3ff);
  const double mag = e == 0 ? std::ldexp(m, -24) : std::ldexp(1024.0 + m, e - 25);
  return (h & 0x8000u) ? -mag : mag;
}

std::string hex_value(uint64_t v) {
  char buf[24];
  std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(v));
  return buf;
}
constexpr uint32_t kVgprFile = 256;   ///< Per-wave VGPR file the CU is configured with.
constexpr uint32_t kMinVgprAlloc = 8; ///< Allocation for a region with no VGPR operand.

std::array<uint8_t, 9> pack_counters(const amdgpu::WaitCounters &c) {
  return {c.vmcnt, c.lgkmcnt,   c.expcnt,   c.vscnt,    c.dscnt,
          c.kmcnt, c.tensorcnt, c.asynccnt, c.samplecnt};
}

void unpack_counters(const std::array<uint8_t, 9> &p, amdgpu::WaitCounters &c) {
  c.vmcnt = p[0];
  c.lgkmcnt = p[1];
  c.expcnt = p[2];
  c.vscnt = p[3];
  c.dscnt = p[4];
  c.kmcnt = p[5];
  c.tensorcnt = p[6];
  c.asynccnt = p[7];
  c.samplecnt = p[8];
}

RunStatus map_error(amdgpu::InstructionExecutionError e) {
  using E = amdgpu::InstructionExecutionError;
  switch (e) {
  case E::None:
    return RunStatus::ExecutionError;
  case E::InvalidInstructionEncoding:
  case E::DecodeFailure:
    return RunStatus::DecodeFailure;
  case E::UnsupportedOperandValue:
    return RunStatus::UnsupportedOperand;
  case E::MissingExecutionBackend:
    return RunStatus::MissingExecutionBackend;
  case E::InvalidRegisterAccess:
    return RunStatus::InvalidRegisterAccess;
  case E::UnimplementedInstruction:
    return RunStatus::Unimplemented;
  case E::IllegalInstruction:
    return RunStatus::IllegalInstruction;
  case E::PluginFailure:
    return RunStatus::ExecutionError;
  }
  return RunStatus::ExecutionError;
}

} // namespace

void MemoryImage::begin(uint64_t seed, const std::vector<std::pair<uint64_t, uint32_t>> &preset) {
  if (!started_ || seed != seed_) {
    words_.clear();
    seed_ = seed;
    started_ = true;
  }
  for (const auto &[w, v] : preset)
    words_.try_emplace(w, v);
}

uint32_t MemoryImage::word(uint64_t w, int pointer_half) {
  auto it = words_.find(w);
  if (it != words_.end())
    return it->second;
  uint32_t v = initial_memory_word(seed_, w);
  if (pointer_half >= 0) {
    // Canonical and far above the code: high half in [0x10, 0x7f], low half 64 KiB aligned.
    const uint64_t h = splitmix64(seed_ ^ (w * 0x9e3779b97f4a7c15ull) ^ 0x7074725f);
    v = pointer_half == 0 ? static_cast<uint32_t>(h) & 0xffff0000u
                          : 0x10u + static_cast<uint32_t>((h >> 32) % 0x70u);
  }
  words_.emplace(w, v);
  return v;
}

std::vector<std::pair<uint64_t, uint32_t>> MemoryImage::snapshot() const {
  std::vector<std::pair<uint64_t, uint32_t>> out(words_.begin(), words_.end());
  std::sort(out.begin(), out.end());
  return out;
}

/// Seeds load bytes from the initial memory content before the pipeline reads them, and
/// records the bytes stores write. Observations fire after routing, before the access.
class RegionRunner::MemoryModel final : public ExecutionPlugin {
public:
  MemoryModel(amdgpu::GpuMemory *mem, uint64_t code_lo, uint64_t code_hi)
      : ExecutionPlugin("rj_isa_diff_memory"), mem_(mem), code_lo_(code_lo), code_hi_(code_hi) {}
  bool observes_before_execute_instruction() const override { return false; }
  bool observes_after_execute_instruction() const override { return false; }
  bool observes_async_instruction_issued() const override { return false; }
  bool observes_memory_instruction_routing() const override { return false; }
  bool observes_vgpr_reads() const override { return false; }
  bool observes_vgpr_writes() const override { return false; }
  bool observes_scalar_register_writes() const override { return false; }
  bool observes_sgpr_reads() const override { return false; }
  bool observes_memory_routing() const override { return true; }

  void set_access(MemoryImage *image, uint32_t pointer_lo, uint32_t pointer_hi) {
    image_ = image;
    pointer_lo_ = pointer_lo;
    pointer_hi_ = pointer_hi;
  }

  void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &a) override {
    if (a.atomic_op != amdgpu::AtomicOp::NONE || !a.secondary_addresses.empty()) {
      unsupported_ = true;
      return;
    }
    const uint64_t bytes = uint64_t{a.element_size_bytes} * a.elements_per_lane;
    for (size_t lane = 0; lane < a.addresses.size() && lane < 64; ++lane) {
      if (!(a.request_lane_mask >> lane & 1u))
        continue;
      const uint64_t base = a.addresses[lane];
      if (bytes && base + (bytes - 1) < base) {
        // The access wraps around the 64-bit address space: hardware faults (the address
        // is far outside any GPUVM range); there are no bytes to seed or record.
        wraps_ = true;
        continue;
      }
      if (a.is_load) {
        for (uint64_t w = base >> 2; w <= (base + bytes - 1) >> 2; ++w) {
          const uint64_t k = w - (base >> 2); // dword of this lane's access
          const int half = k < 32 ? (pointer_lo_ >> k & 1u   ? 0
                                     : pointer_hi_ >> k & 1u ? 1
                                                             : -1)
                                  : -1;
          seed_word(w, half);
        }
      } else {
        for (uint64_t b = base; b < base + bytes; ++b)
          stored_.insert(b);
        record_fp_units(lane, base, bytes);
      }
    }
  }

  /// Undo the previous case (touched bytes back to 0, the unwritten state) and set the
  /// seed of the next one.
  void reset(uint64_t seed) {
    static constexpr uint8_t kZero[4] = {};
    for (uint64_t w : seeded_)
      mem_->write_block(w << 2, std::span<const uint8_t>(kZero, 4));
    for (uint64_t b : stored_)
      mem_->write_block(b, std::span<const uint8_t>(kZero, 1));
    seeded_.clear();
    stored_.clear();
    fp_units_.clear();
    store_fp_ = false;
    seed_ = seed;
    unsupported_ = false;
    wraps_ = false;
  }
  [[nodiscard]] uint64_t seed() const { return seed_; }
  [[nodiscard]] bool unsupported() const { return unsupported_; }
  [[nodiscard]] bool wraps() const { return wraps_; }
  /// Floating-point provenance of the next store's data: per lane, the FpTag of each data
  /// dword; @p hi16 for a `_d16_hi` store. Off for every other access.
  void set_store_fp(bool on, bool hi16,
                    const std::array<std::array<uint8_t, 4>, kWaveLanes> &tags) {
    store_fp_ = on;
    store_hi16_ = hi16;
    if (on)
      store_tags_ = tags;
  }
  [[nodiscard]] std::vector<FpUnit> fp_units() const {
    std::vector<FpUnit> out = fp_units_;
    std::sort(out.begin(), out.end());
    return out;
  }
  [[nodiscard]] std::vector<std::pair<uint64_t, uint8_t>> written() const {
    std::vector<std::pair<uint64_t, uint8_t>> out;
    for (uint64_t b : std::set<uint64_t>(stored_.begin(), stored_.end())) {
      uint8_t v = 0;
      mem_->read_block(b, std::span<uint8_t>(&v, 1));
      out.emplace_back(b, v);
    }
    return out;
  }
  [[nodiscard]] bool touches_code() const {
    for (uint64_t b : stored_)
      if (b >= code_lo_ && b < code_hi_)
        return true;
    return false;
  }

private:
  /// A store replaces the floating-point units it overlaps, and adds those its data
  /// carries: f32 dwords, f64 dword pairs, and f16/bf16 halves.
  void record_fp_units(size_t lane, uint64_t base, uint64_t bytes) {
    std::erase_if(fp_units_,
                  [&](const FpUnit &u) { return u.addr < base + bytes && base < u.addr + u.size; });
    if (!store_fp_ || lane >= kWaveLanes)
      return;
    const auto &t = store_tags_[lane];
    auto half = [&](uint64_t addr, uint8_t tag, bool hi) {
      const uint8_t f16 = hi ? kFpF16Hi : kFpF16Lo, bf16 = hi ? kFpBF16Hi : kFpBF16Lo;
      if (tag & f16)
        fp_units_.push_back({addr, 2, kFpF16Lo});
      else if (tag & bf16)
        fp_units_.push_back({addr, 2, kFpBF16Lo});
    };
    if (bytes == 2) {
      half(base, t[0], store_hi16_);
      return;
    }
    const uint64_t dwords = std::min<uint64_t>(bytes / 4, 4);
    for (uint64_t k = 0; k < dwords; ++k) {
      const uint64_t addr = base + 4 * k;
      if (t[k] & kFpF32) {
        fp_units_.push_back({addr, 4, kFpF32});
      } else if ((t[k] & kFpF64Lo) && k + 1 < dwords && (t[k + 1] & kFpF64Hi)) {
        fp_units_.push_back({addr, 8, kFpF64Lo});
        ++k;
      } else {
        half(addr, t[k], false);
        half(addr + 2, t[k], true);
      }
    }
  }

  void seed_word(uint64_t w, int pointer_half) {
    if (!seeded_.insert(w).second)
      return;
    const uint32_t value = image_ ? image_->word(w, pointer_half) : initial_memory_word(seed_, w);
    for (unsigned k = 0; k < 4; ++k) {
      const uint64_t addr = (w << 2) + k;
      if (stored_.count(addr) || (addr >= code_lo_ && addr < code_hi_))
        continue; // a byte this run stored, or the region's own code
      const uint8_t byte = static_cast<uint8_t>(value >> (8 * k));
      mem_->write_block(addr, std::span<const uint8_t>(&byte, 1));
    }
  }

  amdgpu::GpuMemory *mem_;
  MemoryImage *image_ = nullptr;
  uint32_t pointer_lo_ = 0, pointer_hi_ = 0;
  uint64_t code_lo_, code_hi_;
  uint64_t seed_ = 0;
  bool unsupported_ = false;
  bool wraps_ = false;
  std::unordered_set<uint64_t> seeded_; ///< 4-byte words seeded this run.
  std::unordered_set<uint64_t> stored_; ///< Bytes stored this run.
  std::vector<FpUnit> fp_units_;        ///< Stored floating-point values this run.
  bool store_fp_ = false, store_hi16_ = false;
  std::array<std::array<uint8_t, 4>, kWaveLanes> store_tags_{};
};

const char *run_status_name(RunStatus s) {
  switch (s) {
  case RunStatus::Ok:
    return "OK";
  case RunStatus::DecodeFailure:
    return "DECODE_FAILURE";
  case RunStatus::MissingExecutionBackend:
    return "MISSING_EXECUTION_BACKEND";
  case RunStatus::Unimplemented:
    return "UNIMPLEMENTED";
  case RunStatus::UnsupportedOperand:
    return "UNSUPPORTED_OPERAND";
  case RunStatus::InvalidRegisterAccess:
    return "INVALID_REGISTER_ACCESS";
  case RunStatus::MemoryFault:
    return "MEMORY_FAULT";
  case RunStatus::IllegalInstruction:
    return "ILLEGAL_INSTRUCTION";
  case RunStatus::ExecutionError:
    return "EXECUTION_ERROR";
  case RunStatus::HostException:
    return "HOST_EXCEPTION";
  case RunStatus::UndefinedBehavior:
    return "UNDEFINED_BEHAVIOR";
  case RunStatus::PcWritten:
    return "PC_WRITTEN";
  case RunStatus::WaveStateChanged:
    return "WAVE_STATE_CHANGED";
  case RunStatus::StepLimit:
    return "STEP_LIMIT";
  case RunStatus::BudgetExceeded:
    return "BUDGET_EXCEEDED";
  }
  return "?";
}

RegionRunner::RegionRunner() = default;

RegionRunner::~RegionRunner() {
  // Release the wave's register blocks before the CU goes away.
  if (wave_ && !wave_->is_halted())
    wave_->halt(amdgpu::Wavefront::CpCompletionNotice::Suppress);
  insts_.clear();
}

std::unique_ptr<RegionRunner> RegionRunner::create(const std::vector<RegionInst> &insts,
                                                   const std::string &label, std::string &error,
                                                   unsigned vgpr_count, const Target &target) {
  std::unique_ptr<RegionRunner> r(new RegionRunner());
  r->target_ = target;
  r->lane_mask_ = target.wave32() ? uint64_t{0xffffffffu} : ~uint64_t{0};
  if (vgpr_count > kVgprFile) {
    error = "region needs more than 256 VGPRs";
    return nullptr;
  }
  r->vgpr_count_ = vgpr_count;
  for (const RegionInst &ri : insts) {
    // Classification needs a decode; the memory decision only needs the mnemonic.
    const size_t sp = ri.text.find(' ');
    const std::string mnemonic = sp == std::string::npos ? ri.text : ri.text.substr(0, sp);
    r->is_mem_.push_back(is_region_memory(mnemonic));
    r->has_memory_ = r->has_memory_ || r->is_mem_.back();
  }
  r->memory_ = std::make_unique<amdgpu::GpuMemory>(label + "_mem");
  r->l2_ = std::make_unique<amdgpu::L2Cache>(label + "_l2");
  r->l2_->set_backing_memory(r->memory_.get()); // global accesses reach this memory
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = target.arch;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = target.wave_sgprs();
  config.vgprs_per_wf = kVgprFile;
  config.lds_size_kb = 64;
  // Waits are modelled by the region planner (pending effects); a region replays the same
  // PCs for every case, which the CU's own wait scoreboard would report as hazards.
  config.memory_wait_diagnostics = amdgpu::MemoryWaitDiagnostics::Off;
  util::StringDiagnostic diag;
  r->cu_ = amdgpu::ComputeUnitCore::create(label, config, r->memory_.get(), r->l2_.get(),
                                           simdojo::ExecMode::FUNCTIONAL, diag.emitter());
  if (!r->cu_) {
    error = "cannot create a " + target.name + " compute unit: " + diag.message();
    return nullptr;
  }
  const uint64_t entry = insts.empty() ? 0 : insts.front().pc;
  if (r->has_memory_) {
    // step() fetches from memory: place the region's words at their PCs, and observe
    // memory accesses to seed loads and record stores.
    uint64_t lo = ~uint64_t{0}, hi = 0;
    for (const RegionInst &ri : insts) {
      r->memory_->write_block(
          ri.pc, std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(ri.words.data()),
                                          ri.words.size() * 4));
      lo = std::min(lo, ri.pc);
      hi = std::max<uint64_t>(hi, ri.pc + ri.words.size() * 4);
    }
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto *model = new MemoryModel(r->memory_.get(), lo, hi);
    if (!group->add(OwnedPlugin(model, PluginDeleter{&delete_execution_plugin}))) {
      error = "cannot attach the memory-model plugin";
      return nullptr;
    }
    r->mem_ = model;
    r->cu_->set_plugin_group(group);
    r->plugin_group_ = group;
  }
  r->entry_pc_ = entry;
  r->vgpr_alloc_ = vgpr_count ? vgpr_count : kMinVgprAlloc;
  r->wave_ = r->cu_->dispatch_wf(0, entry, target.wave_sgprs(), r->vgpr_alloc_, target.wave_size);
  if (!r->wave_) {
    error = "cannot dispatch a " + target.label() + " wave";
    return nullptr;
  }
  r->decoder_ = Decoder::create(target.arch);
  if (!r->decoder_) {
    error = "no " + target.name + " decoder";
    return nullptr;
  }
  r->asm_.assign(insts.size(), std::nullopt);
  r->asm_block_.assign(insts.size(), 0);
  r->branch_.assign(insts.size(), false);
  for (size_t i = 0; i < insts.size(); ++i)
    r->pc_index_.emplace(insts[i].pc, i);
  for (size_t i = 0; i < insts.size(); ++i) {
    const RegionInst &ri = insts[i];
    r->pcs_.push_back(ri.pc);
    r->texts_.push_back(ri.text);
    if (ri.inline_asm >= 0) {
      // Comment-only inline asm: no machine code; run_inline_asm() gives it its meaning.
      r->asm_[i] = parse_inline_asm(ri.asm_text);
      r->asm_block_[i] = static_cast<uint32_t>(ri.inline_asm);
      r->insts_.push_back(nullptr);
      r->model_only_.push_back(false);
      continue;
    }
    util::StringDiagnostic ddiag;
    std::vector<uint32_t> window = ri.words;
    window.resize(window.size() + 3, 0); // decoder may look ahead for literals
    auto decoded = r->decoder_->decode_window(window, ri.pc, ddiag.emitter());
    if (decoded.failed() || static_cast<size_t>(decoded.value()->size()) != ri.words.size() * 4) {
      if (!r->decode_failed_) {
        r->decode_failed_ = true;
        r->decode_failed_index_ = i;
        r->decode_error_ = decoded.failed()
                               ? "decode failed: " + ddiag.message()
                               : "decoded size does not match the recorded instruction words";
      }
      r->insts_.push_back(nullptr);
      r->model_only_.push_back(false);
      continue;
    }
    r->insts_.push_back(std::move(decoded).value());
    r->model_only_.push_back(false);
  }
  // Per-instruction plans: source operands, floating-point provenance, poison checks.
  r->srcs_.assign(insts.size(), {});
  r->fp_plan_.assign(insts.size(), {});
  r->poison_kind_.assign(insts.size(), 0);
  r->mnemonics_.assign(insts.size(), {});
  for (size_t i = 0; i < r->insts_.size(); ++i) {
    const Instruction *inst = r->insts_[i].get();
    if (!inst)
      continue;
    const std::string mn(inst->mnemonic());
    r->mnemonics_[i] = mn;
    const Effects fx = classify(*inst, target);
    if (fx.branch) {
      r->branch_[i] = true;
      // SOPP simm16 (bits 15:0): the target is PC + 4 + 4 * simm16; a negative offset goes
      // back (at or before the branch), so the region may loop.
      if (static_cast<int16_t>(insts[i].words[0] & 0xffffu) < 0)
        r->back_edge_ = true;
    }
    const std::vector<SourceOperand> so = source_operands(*inst, target);
    for (const SourceOperand &o : so) {
      if (o.fieldless)
        continue;
      OperandRef ref;
      if (o.constant) {
        ref.kind = OperandRef::Constant;
        ref.value = *o.constant;
      } else if (o.vlocs.any()) {
        ref.kind = OperandRef::Vgpr;
        ref.count = static_cast<unsigned>(o.vlocs.count());
        for (unsigned v = 0; v < kNumVgprs; ++v)
          if (o.vlocs.test(v)) {
            ref.reg = v;
            break;
          }
      } else if (o.locs.any()) {
        ref.kind = OperandRef::Scalar;
        ref.count = static_cast<unsigned>(o.locs.count());
        for (unsigned l = 0; l < kNumLocs; ++l)
          if (o.locs.test(l)) {
            ref.reg = l;
            break;
          }
      }
      r->srcs_[i].push_back(ref);
    }
    FpPlan &plan = r->fp_plan_[i];
    plan.kind = fp_result_kind(mn, insts[i].text);
    for (unsigned v = 0; v < kNumVgprs; ++v)
      if (fx.vwrites.test(v))
        plan.dsts.push_back(v);
    if (plan.kind == FpResult::CndMask) {
      // The lane-select mask: VCC (fieldless, VOP2) or the last wave-mask source (VOP3): an
      // SGPR pair on wave64, one SGPR on wave32 (where a scalar data source is one SGPR too,
      // so the last one wins).
      const size_t mask_regs = target.wave32() ? 1 : 2;
      for (const SourceOperand &o : so)
        if (o.locs.count() == mask_regs && o.vlocs.none() && !o.constant)
          for (unsigned l = 0; l < kNumLocs; ++l)
            if (o.locs.test(l)) {
              plan.mask_loc = static_cast<int>(l);
              break;
            }
      if (plan.mask_loc < 0)
        plan.kind = FpResult::None;
    }
    if (plan.kind == FpResult::PkMov) {
      const std::string &t = insts[i].text;
      const size_t at = t.find("op_sel:[");
      if (at != std::string::npos && at + 11 <= t.size()) {
        plan.op_sel_lo = t[at + 8] == '1';
        plan.op_sel_hi = t[at + 10] == '1';
      }
    }
    plan.store = r->is_mem_[i] && fx.mem_write;
    plan.skind = fp_scalar_result_kind(mn);
    for (unsigned l = 0; l < kNumLocs; ++l)
      if (fx.writes.test(l) && l != kLocScc)
        plan.sdsts.push_back(l);
    plan.store_hi16 = mn.find("_d16_hi") != std::string::npos;
    uint32_t pk = poison_check_kind(mn);
    if (insts[i].text.find("clamp") != std::string::npos)
      pk &= ~(kPoisonNsw | kPoisonNuw); // saturating: no overflow
    r->poison_kind_[i] = pk;
  }

  // Pointer dataflow (memory model): a scalar load whose destination later serves as a
  // memory base (directly or through SALU moves/adds) gets pointer-shaped content there.
  r->pointer_lo_.assign(insts.size(), 0);
  r->pointer_hi_.assign(insts.size(), 0);
  if (r->has_memory_) {
    std::vector<std::pair<int, int>> origin(kNumSgprs, {-1, -1}); // (load index, dword)
    for (size_t i = 0; i < r->insts_.size(); ++i) {
      const Instruction *inst = r->insts_[i].get();
      if (!inst && r->asm_[i]) {
        // An inline-asm def is an opaque producer, like a scalar load: if its SGPRs later
        // serve as a memory base, it produces pointer-shaped values there.
        const auto &defs = r->asm_[i]->defs;
        for (size_t j = 0; j < defs.size(); ++j)
          if (!defs[j].vgpr && defs[j].index < kNumSgprs)
            origin[defs[j].index] =
                j < 32 ? std::pair<int, int>{static_cast<int>(i), static_cast<int>(j)}
                       : std::pair<int, int>{-1, -1};
        continue;
      }
      if (!inst)
        break;
      const Effects fx = classify(*inst, target);
      const std::vector<SourceOperand> src = source_operands(*inst, target);
      std::vector<unsigned> writes;
      for (unsigned l = 0; l < kNumSgprs; ++l)
        if (fx.writes.test(l))
          writes.push_back(l);
      if (r->is_mem_[i]) {
        for (const SourceOperand &so : src) {
          if (so.locs.count() != 2)
            continue;
          unsigned lo = kNumSgprs;
          for (unsigned l = 0; l < kNumSgprs; ++l)
            if (so.locs.test(l)) {
              lo = l;
              break;
            }
          for (unsigned p : {lo, lo + 1})
            if (p < kNumSgprs && origin[p].first >= 0) {
              const auto [li, k] = origin[p];
              (p == lo ? r->pointer_lo_ : r->pointer_hi_)[static_cast<size_t>(li)] |= 1u << k;
            }
          break; // the first SGPR pair is the base (sbase / saddr)
        }
        const bool scalar_load =
            fx.mem_read && std::string_view(inst->mnemonic()).substr(0, 6) == "s_load";
        for (size_t idx = 0; idx < writes.size(); ++idx)
          origin[writes[idx]] = scalar_load && idx < 32 ? std::pair<int, int>{static_cast<int>(i),
                                                                              static_cast<int>(idx)}
                                                        : std::pair<int, int>{-1, -1};
        continue;
      }
      const SourceOperand *first = nullptr;
      for (const SourceOperand &so : src)
        if (so.locs.any()) {
          first = &so;
          break;
        }
      unsigned first_lo = kNumSgprs;
      if (first)
        for (unsigned l = 0; l < kNumSgprs; ++l)
          if (first->locs.test(l)) {
            first_lo = l;
            break;
          }
      for (size_t idx = 0; idx < writes.size(); ++idx) {
        const unsigned from = first_lo + static_cast<unsigned>(idx);
        origin[writes[idx]] = first && from < kNumSgprs && first->locs.test(from)
                                  ? origin[from]
                                  : std::pair<int, int>{-1, -1};
      }
    }
  }
  // Fix the frame to the post-dispatch values.
  r->baseline_ = r->capture().frame;
  return r;
}

void RegionRunner::simulate_model_only_for_test(size_t index) {
  if (index < model_only_.size())
    model_only_[index] = true;
}

void RegionRunner::load(const MachineState &s) {
  if (wave_->state() != amdgpu::WfState::RUNNING) // a memory fault halted it: re-dispatch
    wave_ = cu_->dispatch_wf(0, entry_pc_, target_.wave_sgprs(), vgpr_alloc_, target_.wave_size);
  if (has_memory_)
    image_->begin(s.mem_seed, s.mem_init);
  amdgpu::Wavefront &wf = *wave_;
  const uint32_t base = wf.sgpr_alloc().base;
  for (unsigned i = 0; i < target_.sgprs(); ++i)
    cu_->write_sgpr(base + i, s.loc[i]);
  if (target_.major == 9) { // the two allocated SGPRs above s101
    cu_->write_sgpr(base + 102, s.frame.s102);
    cu_->write_sgpr(base + 103, s.frame.s103);
  }
  wf.set_vcc_raw(uint64_t{s.loc[kLocVccLo]} | (uint64_t{s.loc[kLocVccHi]} << 32));
  wf.set_exec_raw(uint64_t{s.loc[kLocExecLo]} | (uint64_t{s.loc[kLocExecHi]} << 32));
  wf.set_m0(s.loc[kLocM0]);
  wf.set_status_raw((s.frame.status_except_scc & ~1u) | (s.loc[kLocScc] & 1u));
  wf.set_mode_raw(s.frame.mode);
  wf.set_trapsts(s.frame.trapsts);
  for (uint32_t i = 0; i < 16; ++i)
    wf.set_ttmp(i, s.frame.ttmp[i]);
  wf.set_scratch_base(s.frame.scratch_base);
  unpack_counters(s.frame.wait_counters, wf.wait_counters());
  wf.set_in_trap_handler(s.frame.in_trap_handler);
  if (mem_) {
    // Write back the previous case's stores, undo its memory, and drop every cached line
    // (L1 scalar/vector caches and the write-back L2), so this case reads its own input.
    cu_->flush_all();
    mem_->reset(s.mem_seed);
    l2_->invalidate_all();
    cu_->flush_l1();
  }
  const uint32_t vbase = wf.vgpr_alloc().base;
  for (unsigned r = 0; r < vgpr_count_ && r < s.vgpr_count; ++r)
    for (unsigned lane = 0; lane < target_.wave_size; ++lane)
      cu_->write_vgpr(vbase + r, lane, s.vget(r, lane));
  wf.begin_instruction_execution();
  wf.pc = pcs_.empty() ? 0 : pcs_.front();
  asm_seed_ = s.asm_seed;
  asm_uses_.clear();
  vfp_.assign(size_t{vgpr_count_} * kWaveLanes, 0); // inputs carry no provenance
  sfp_.fill(0);
  poison_hit_.clear();
  nan_produced_ = false;
}

MachineState RegionRunner::capture() const {
  const amdgpu::Wavefront &wf = *wave_;
  MachineState s;
  const uint32_t base = wf.sgpr_alloc().base;
  for (unsigned i = 0; i < target_.sgprs(); ++i)
    s.loc[i] = cu_->read_sgpr_storage(base + i);
  if (target_.major == 9) {
    s.frame.s102 = cu_->read_sgpr_storage(base + 102);
    s.frame.s103 = cu_->read_sgpr_storage(base + 103);
  }
  const uint64_t vcc = wf.vcc();
  const uint64_t exec = wf.exec_raw();
  s.loc[kLocVccLo] = static_cast<uint32_t>(vcc);
  s.loc[kLocVccHi] = static_cast<uint32_t>(vcc >> 32);
  s.loc[kLocExecLo] = static_cast<uint32_t>(exec);
  s.loc[kLocExecHi] = static_cast<uint32_t>(exec >> 32);
  s.loc[kLocM0] = wf.m0();
  s.loc[kLocScc] = wf.read_scc() ? 1u : 0u;
  s.frame.status_except_scc = wf.status_raw() & ~1u;
  s.frame.mode = wf.mode_raw();
  s.frame.trapsts = wf.trapsts();
  for (uint32_t i = 0; i < 16; ++i)
    s.frame.ttmp[i] = wf.ttmp(i);
  s.frame.scratch_base = wf.scratch_base();
  s.frame.wait_counters = pack_counters(wf.wait_counters());
  s.frame.in_trap_handler = wf.in_trap_handler();
  s.frame.wave_state = static_cast<uint8_t>(wf.state());
  s.frame.sleep_cycles = wf.sleep_cycles();
  s.resize_vgprs(vgpr_count_);
  const uint32_t vbase = wf.vgpr_alloc().base;
  for (unsigned r = 0; r < vgpr_count_; ++r)
    for (unsigned lane = 0; lane < target_.wave_size; ++lane)
      s.vset(r, lane, cu_->read_vgpr_storage(vbase + r, lane));
  if (mem_) {
    cu_->flush_all(); // stores may still sit in the write-back L2
    s.mem_seed = mem_->seed();
    s.mem_written = mem_->written();
    s.mem_init = image_->snapshot();
  }
  s.asm_seed = asm_seed_;
  s.asm_uses = asm_uses_;
  s.exit_pc = exit_pc_;
  if (std::ranges::any_of(vfp_, [](uint8_t t) { return t != 0; }))
    s.vfp = vfp_;
  if (std::ranges::any_of(sfp_, [](uint8_t t) { return t != 0; }))
    s.sfp.assign(sfp_.begin(), sfp_.end());
  if (mem_)
    s.mem_fp = mem_->fp_units();
  return s;
}

uint64_t RegionRunner::exec_lanes() const { return wave_->exec_raw() & lane_mask_; }

uint32_t RegionRunner::read_loc(unsigned l) const {
  const amdgpu::Wavefront &wf = *wave_;
  if (l < kNumSgprs) // s102-s105 do not exist on GFX9
    return l < target_.sgprs() ? cu_->read_sgpr_storage(wf.sgpr_alloc().base + l) : 0u;
  switch (l) {
  case kLocVccLo:
    return static_cast<uint32_t>(wf.vcc());
  case kLocVccHi:
    return static_cast<uint32_t>(wf.vcc() >> 32);
  case kLocExecLo:
    return static_cast<uint32_t>(wf.exec_raw());
  case kLocExecHi:
    return static_cast<uint32_t>(wf.exec_raw() >> 32);
  case kLocM0:
    return wf.m0();
  default:
    return wf.read_scc() ? 1u : 0u;
  }
}

uint32_t RegionRunner::read_vgpr(unsigned r, unsigned lane) const {
  return cu_->read_vgpr_storage(wave_->vgpr_alloc().base + r, lane);
}

std::optional<uint64_t> RegionRunner::operand_value(const OperandRef &op, unsigned lane,
                                                    bool wide) const {
  switch (op.kind) {
  case OperandRef::Constant:
    return wide ? std::nullopt : std::optional<uint64_t>(op.value);
  case OperandRef::Vgpr:
    if (op.reg + std::max(op.count, 1u) > vgpr_count_)
      return std::nullopt;
    if (!wide)
      return read_vgpr(op.reg, lane);
    if (op.count < 2)
      return std::nullopt;
    return uint64_t{read_vgpr(op.reg, lane)} | (uint64_t{read_vgpr(op.reg + 1, lane)} << 32);
  case OperandRef::Scalar:
    if (!wide)
      return read_loc(op.reg);
    if (op.count < 2)
      return std::nullopt;
    return uint64_t{read_loc(op.reg)} | (uint64_t{read_loc(op.reg + 1)} << 32);
  default:
    return std::nullopt;
  }
}

void RegionRunner::plan_tags(size_t i, std::vector<std::array<uint8_t, kWaveLanes>> &out) const {
  const FpPlan &p = fp_plan_[i];
  out.resize(p.dsts.size());
  const auto &src = srcs_[i];
  auto src_tag = [&](size_t s, unsigned k, unsigned lane) -> uint8_t {
    if (s >= src.size())
      return 0;
    if (src[s].kind == OperandRef::Scalar) // an SGPR source: the same value in every lane
      return src[s].reg + k < kNumLocs ? sfp_[src[s].reg + k] : 0;
    if (src[s].kind != OperandRef::Vgpr || src[s].reg + k >= vgpr_count_)
      return 0;
    return vfp_[size_t{src[s].reg + k} * kWaveLanes + lane];
  };
  uint64_t mask = 0;
  if (p.kind == FpResult::CndMask)
    mask = uint64_t{read_loc(static_cast<unsigned>(p.mask_loc))} |
           (target_.wave32() ? 0 : uint64_t{read_loc(static_cast<unsigned>(p.mask_loc) + 1)} << 32);
  for (size_t k = 0; k < p.dsts.size(); ++k) {
    const unsigned d = p.dsts[k];
    for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
      const uint8_t cur = d < vgpr_count_ ? vfp_[size_t{d} * kWaveLanes + lane] : 0;
      uint8_t t = 0;
      switch (p.kind) {
      case FpResult::None:
        break;
      case FpResult::F32:
        t = kFpF32;
        break;
      case FpResult::F64:
        t = k % 2 == 0 ? kFpF64Lo : kFpF64Hi;
        break;
      case FpResult::F16Lo:
        t = (cur & (kFpF16Hi | kFpBF16Hi)) | kFpF16Lo;
        break;
      case FpResult::F16Hi:
        t = (cur & (kFpF16Lo | kFpBF16Lo)) | kFpF16Hi;
        break;
      case FpResult::BF16Lo:
        t = (cur & (kFpF16Hi | kFpBF16Hi)) | kFpBF16Lo;
        break;
      case FpResult::BF16Hi:
        t = (cur & (kFpF16Lo | kFpBF16Lo)) | kFpBF16Hi;
        break;
      case FpResult::F16x2:
        t = kFpF16Lo | kFpF16Hi;
        break;
      case FpResult::BF16x2:
        t = kFpBF16Lo | kFpBF16Hi;
        break;
      case FpResult::Move:
        t = src_tag(0, static_cast<unsigned>(k), lane);
        break;
      case FpResult::CndMask:
        t = src_tag((mask >> lane) & 1u ? 1 : 0, 0, lane);
        break;
      case FpResult::PkMov:
        t = k == 0 ? src_tag(0, p.op_sel_lo, lane) : src_tag(1, p.op_sel_hi, lane);
        break;
      case FpResult::Select:
      case FpResult::CMov:
      case FpResult::ReadFirstLane:
      case FpResult::ReadLane:
        break; // scalar-result kinds (plan_stags)
      }
      out[k][lane] = t;
    }
  }
}

std::vector<uint8_t> RegionRunner::plan_stags(size_t i) const {
  const FpPlan &p = fp_plan_[i];
  std::vector<uint8_t> out(p.sdsts.size(), 0);
  if (p.sdsts.empty() || p.skind == FpResult::None)
    return out;
  const auto &src = srcs_[i];
  auto stag = [&](size_t s, unsigned k) -> uint8_t {
    if (s >= src.size() || src[s].kind != OperandRef::Scalar || src[s].reg + k >= kNumLocs)
      return 0;
    return sfp_[src[s].reg + k];
  };
  auto lane_tag = [&](unsigned lane) -> uint8_t {
    if (src.empty() || src[0].kind != OperandRef::Vgpr || src[0].reg >= vgpr_count_)
      return 0;
    return vfp_[size_t{src[0].reg} * kWaveLanes + (lane % kWaveLanes)];
  };
  const bool scc = read_loc(kLocScc) & 1u;
  for (size_t k = 0; k < p.sdsts.size(); ++k) {
    const auto kk = static_cast<unsigned>(k);
    switch (p.skind) {
    case FpResult::F32:
      out[k] = kFpF32;
      break;
    case FpResult::F16Lo:
      out[k] = kFpF16Lo;
      break;
    case FpResult::F16x2:
      out[k] = kFpF16Lo | kFpF16Hi;
      break;
    case FpResult::Move:
      out[k] = stag(0, kk);
      break;
    case FpResult::Select:
      out[k] = scc ? stag(0, kk) : stag(1, kk);
      break;
    case FpResult::CMov:
      out[k] = scc ? stag(0, kk) : sfp_[p.sdsts[k]];
      break;
    case FpResult::ReadFirstLane: {
      const uint64_t exec = exec_lanes();
      out[k] = lane_tag(exec ? static_cast<unsigned>(std::countr_zero(exec)) : 0u);
      break;
    }
    case FpResult::ReadLane: {
      const auto lane = src.size() > 1 ? operand_value(src[1], 0, false) : std::nullopt;
      out[k] = lane ? lane_tag(static_cast<unsigned>(*lane) % target_.wave_size) : 0;
      break;
    }
    default:
      break;
    }
  }
  return out;
}

void RegionRunner::apply_stags(size_t i, const std::vector<uint8_t> &tags) {
  const FpPlan &p = fp_plan_[i];
  for (size_t k = 0; k < p.sdsts.size() && k < tags.size(); ++k)
    sfp_[p.sdsts[k]] = tags[k];
}

void RegionRunner::write_loc(unsigned l, uint32_t v) {
  amdgpu::Wavefront &wf = *wave_;
  if (l < kNumSgprs) {
    if (l < target_.sgprs())
      cu_->write_sgpr(wf.sgpr_alloc().base + l, v);
    return;
  }
  switch (l) {
  case kLocVccLo:
    wf.set_vcc_raw((wf.vcc() & ~uint64_t{0xffffffffu}) | v);
    break;
  case kLocVccHi:
    wf.set_vcc_raw((wf.vcc() & uint64_t{0xffffffffu}) | (uint64_t{v} << 32));
    break;
  case kLocExecLo:
    wf.set_exec_raw((wf.exec_raw() & ~uint64_t{0xffffffffu}) | v);
    break;
  case kLocExecHi:
    wf.set_exec_raw((wf.exec_raw() & uint64_t{0xffffffffu}) | (uint64_t{v} << 32));
    break;
  case kLocM0:
    wf.set_m0(v);
    break;
  default:
    wf.set_status_raw((wf.status_raw() & ~1u) | (v & 1u)); // SCC
    break;
  }
}

void RegionRunner::apply_tags(size_t i, uint64_t exec,
                              const std::vector<std::array<uint8_t, kWaveLanes>> &tags) {
  const FpPlan &p = fp_plan_[i];
  for (size_t k = 0; k < p.dsts.size() && k < tags.size(); ++k) {
    if (p.dsts[k] >= vgpr_count_)
      continue;
    for (unsigned lane = 0; lane < kWaveLanes; ++lane)
      if ((exec >> lane) & 1u)
        vfp_[size_t{p.dsts[k]} * kWaveLanes + lane] = tags[k][lane];
  }
}

std::string RegionRunner::poison_before(size_t i) const {
  const uint32_t kinds = poison_kind_[i] & poison_filter_;
  if (!kinds)
    return {};
  const std::string &mn = mnemonics_[i];
  std::string_view b = mn;
  for (std::string_view suffix : {"_e32", "_e64"})
    if (b.size() > suffix.size() && b.substr(b.size() - suffix.size()) == suffix)
      b = b.substr(0, b.size() - suffix.size());
  const auto &src = srcs_[i];
  const bool scalar = b.substr(0, 2) == "s_";
  const uint64_t exec = scalar ? 1u : exec_lanes();
  const bool b64 = b.find("b64") != b.npos || b.find("i64") != b.npos || b.find("_f64") != b.npos;
  const unsigned width = b64 ? 64 : (b.find("b16") != b.npos || b.find("i16") != b.npos) ? 16 : 32;
  auto hit = [&](const char *kind, const std::string &what, unsigned lane) {
    return std::string(kind) + ": " + what + " at '" + texts_[i] + "'" +
           (scalar ? std::string() : " lane " + std::to_string(lane));
  };
  for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
    if (!((exec >> lane) & 1u))
      continue;
    auto val = [&](size_t s, bool wide = false) -> std::optional<uint64_t> {
      return s < src.size() ? operand_value(src[s], lane, wide) : std::nullopt;
    };
    if (kinds & (kPoisonShift | kPoisonExact)) {
      const size_t amount_i = scalar ? 1 : 0, value_i = scalar ? 0 : 1;
      const bool right = b.find("lshr") != b.npos || b.find("ashr") != b.npos;
      const bool packed = b.substr(0, 5) == "v_pk_";
      const auto amount = val(amount_i);
      const auto value = val(value_i, width == 64);
      if (amount) {
        const unsigned parts = packed ? 2 : 1;
        for (unsigned h = 0; h < parts; ++h) {
          const uint64_t a = packed        ? (*amount >> (16 * h)) & 0xffffu
                             : width == 16 ? *amount & 0xffffu
                                           : *amount;
          if ((kinds & kPoisonShift) && a >= width)
            return hit("shift", "amount " + std::to_string(a) + " >= " + std::to_string(width),
                       lane);
          if ((kinds & kPoisonExact) && right && value && a > 0 && a < width) {
            const uint64_t v = packed ? (*value >> (16 * h)) & 0xffffu : *value;
            if (v & ((uint64_t{1} << a) - 1))
              return hit("exact", "non-zero bits shifted out", lane);
          }
        }
      }
    }
    if (kinds & kPoisonFpToInt) {
      const bool f64 = b.find("_f64") != b.npos, f16 = b.find("_f16") != b.npos;
      if (const auto v = val(0, f64)) {
        double d = 0;
        bool special = false;
        if (f64) {
          special = f64_nan(*v) || f64_inf(*v);
          d = std::bit_cast<double>(*v);
        } else if (f16) {
          special = f16_nan(*v & 0xffffu) || f16_inf(*v & 0xffffu);
          d = f16_value(*v & 0xffffu);
        } else {
          special = f32_nan(static_cast<uint32_t>(*v)) || f32_inf(static_cast<uint32_t>(*v));
          d = static_cast<double>(std::bit_cast<float>(static_cast<uint32_t>(*v)));
        }
        if (special)
          return hit("fptoint", "NaN or infinite input " + hex_value(*v), lane);
        const double r = b.find("flr") != b.npos   ? std::floor(d)
                         : b.find("rpi") != b.npos ? std::floor(d + 0.5)
                                                   : std::trunc(d);
        const bool is_unsigned = b.find("_u32_") != b.npos || b.find("_u16_") != b.npos;
        const bool narrow = b.find("_i16_") != b.npos || b.find("_u16_") != b.npos;
        const double lo = is_unsigned ? 0.0 : narrow ? -32768.0 : -2147483648.0;
        const double hi =
            is_unsigned ? (narrow ? 65535.0 : 4294967295.0) : (narrow ? 32767.0 : 2147483647.0);
        if (r < lo || r > hi)
          return hit("fptoint", "input " + hex_value(*v) + " out of the integer range", lane);
      }
    }
    if (kinds & kPoisonCtz) {
      if (const auto v = val(0, width == 64); v && *v == 0)
        return hit("ctz", "zero input", lane);
    }
    if (kinds & (kPoisonNsw | kPoisonNuw | kPoisonDisjoint)) {
      const auto x = val(0), y = val(1);
      if (x && y) {
        const uint32_t a = static_cast<uint32_t>(*x), c = static_cast<uint32_t>(*y);
        if (kinds & kPoisonDisjoint) {
          if (a & c)
            return hit("disjoint", "operands share bits", lane);
        } else {
          const bool sub = b.find("sub") != b.npos, rev = b.find("subrev") != b.npos;
          const bool mul = b.find("mul") != b.npos;
          const int64_t sa = static_cast<int32_t>(a), sc = static_cast<int32_t>(c);
          const int64_t sres = mul ? sa * sc : sub ? (rev ? sc - sa : sa - sc) : sa + sc;
          const uint64_t ures = mul   ? uint64_t{a} * c
                                : sub ? (rev ? uint64_t{c} - a : uint64_t{a} - c)
                                      : uint64_t{a} + c;
          const bool uoverflow = sub ? (rev ? c < a : a < c) : ures > 0xffffffffull;
          if ((kinds & kPoisonNsw) && (sres < INT32_MIN || sres > INT32_MAX))
            return hit("nsw", "signed overflow", lane);
          if ((kinds & kPoisonNuw) && uoverflow)
            return hit("nuw", "unsigned overflow", lane);
        }
      }
    }
    if (kinds & (kPoisonNnan | kPoisonNinf)) {
      const FpResult f = fp_source_kind(mn);
      for (size_t s = 0; s < src.size(); ++s) {
        const bool wide = f == FpResult::F64;
        const auto v = val(s, wide);
        if (!v)
          continue;
        bool nan = false, inf = false;
        const uint32_t w = static_cast<uint32_t>(*v);
        switch (f) {
        case FpResult::F64:
          nan = f64_nan(*v);
          inf = f64_inf(*v);
          break;
        case FpResult::F16Lo:
          nan = f16_nan(w & 0xffffu);
          inf = f16_inf(w & 0xffffu);
          break;
        case FpResult::F16x2:
          nan = f16_nan(w & 0xffffu) || f16_nan(w >> 16);
          inf = f16_inf(w & 0xffffu) || f16_inf(w >> 16);
          break;
        case FpResult::BF16Lo:
          nan = bf16_nan(w & 0xffffu);
          inf = bf16_inf(w & 0xffffu);
          break;
        case FpResult::BF16x2:
          nan = bf16_nan(w & 0xffffu) || bf16_nan(w >> 16);
          inf = bf16_inf(w & 0xffffu) || bf16_inf(w >> 16);
          break;
        default:
          nan = f32_nan(w);
          inf = f32_inf(w);
          break;
        }
        if ((kinds & kPoisonNnan) && nan)
          return hit("nnan", "NaN input " + hex_value(*v), lane);
        if ((kinds & kPoisonNinf) && inf)
          return hit("ninf", "infinite input " + hex_value(*v), lane);
      }
    }
    if (scalar)
      break;
  }
  return {};
}

void RegionRunner::choose_nans(size_t i, uint64_t exec) {
  const FpPlan &p = fp_plan_[i];
  if (p.skind == FpResult::F32 || p.skind == FpResult::F16Lo || p.skind == FpResult::F16x2) {
    for (unsigned l : p.sdsts) {
      const uint32_t w = read_loc(l);
      const uint64_t r = splitmix64(nan_seed_ ^ (uint64_t{i} << 20) ^ (uint64_t{l} << 8) ^ 0xffu);
      uint32_t out = w;
      bool any = false;
      if (p.skind == FpResult::F32) {
        if (is_nan_f32(w)) {
          any = true;
          out = nan_choice_ == 1   ? 0x7fc00000u
                : nan_choice_ == 2 ? 0xffffffffu
                                   : static_cast<uint32_t>((r & 0x803fffffu) | 0x7fc00000u);
        }
      } else {
        for (unsigned h = 0; h < (p.skind == FpResult::F16x2 ? 2u : 1u); ++h) {
          const uint32_t half = (w >> (16 * h)) & 0xffffu;
          if (!is_nan_f16(half))
            continue;
          any = true;
          const uint32_t n = nan_choice_ == 1 ? 0x7e00u
                             : nan_choice_ == 2
                                 ? 0xffffu
                                 : static_cast<uint32_t>(((r >> (16 * h)) & 0x81ffu) | 0x7e00u);
          out = (out & ~(0xffffu << (16 * h))) | (n << (16 * h));
        }
      }
      if (!any)
        continue;
      nan_produced_ = true;
      if (nan_choice_)
        write_loc(l, out);
    }
  }
  switch (p.kind) {
  case FpResult::None:
  case FpResult::Move:
  case FpResult::CndMask:
  case FpResult::PkMov:
  case FpResult::Select:
  case FpResult::CMov:
  case FpResult::ReadFirstLane:
  case FpResult::ReadLane:
    return; // not a floating-point operation: NaN bits pass through unchanged
  default:
    break;
  }
  const uint32_t vbase = wave_->vgpr_alloc().base;
  for (size_t k = 0; k < p.dsts.size(); ++k) {
    const unsigned d = p.dsts[k];
    if (d >= vgpr_count_)
      continue;
    for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
      if (!((exec >> lane) & 1u))
        continue;
      const uint8_t t = vfp_[size_t{d} * kWaveLanes + lane];
      const uint32_t w = read_vgpr(d, lane);
      const uint64_t r = splitmix64(nan_seed_ ^ (uint64_t{i} << 20) ^ (uint64_t{d} << 8) ^ lane);
      auto pick = [&](uint64_t canonical, uint64_t ones, uint64_t keep, uint64_t quiet) {
        return nan_choice_ == 1 ? canonical : nan_choice_ == 2 ? ones : ((r & keep) | quiet);
      };
      if (t & kFpF32) {
        if (!is_nan_f32(w))
          continue;
        nan_produced_ = true;
        if (nan_choice_)
          cu_->write_vgpr(
              vbase + d, lane,
              static_cast<uint32_t>(pick(0x7fc00000u, 0xffffffffu, 0x803fffffu, 0x7fc00000u)));
      } else if ((t & kFpF64Lo) && d + 1 < vgpr_count_) {
        const uint64_t v = uint64_t{w} | (uint64_t{read_vgpr(d + 1, lane)} << 32);
        if (!is_nan_f64(v))
          continue;
        nan_produced_ = true;
        if (nan_choice_) {
          const uint64_t n = pick(0x7ff8000000000000ull, ~uint64_t{0}, 0x8007ffffffffffffull,
                                  0x7ff8000000000000ull);
          cu_->write_vgpr(vbase + d, lane, static_cast<uint32_t>(n));
          cu_->write_vgpr(vbase + d + 1, lane, static_cast<uint32_t>(n >> 32));
        }
      } else {
        uint32_t out = w;
        bool any = false;
        for (unsigned h = 0; h < 2; ++h) {
          const uint32_t half = (w >> (16 * h)) & 0xffffu;
          const bool f16 = t & (h ? kFpF16Hi : kFpF16Lo), bf16 = t & (h ? kFpBF16Hi : kFpBF16Lo);
          if (!((f16 && is_nan_f16(half)) || (bf16 && is_nan_bf16(half))))
            continue;
          any = true;
          const uint64_t rh = r >> (16 * h);
          const uint32_t n =
              f16 ? static_cast<uint32_t>(nan_choice_ == 1   ? 0x7e00u
                                          : nan_choice_ == 2 ? 0xffffu
                                                             : ((rh & 0x81ffu) | 0x7e00u))
                  : static_cast<uint32_t>(nan_choice_ == 1   ? 0x7fc0u
                                          : nan_choice_ == 2 ? 0xffffu
                                                             : ((rh & 0x803fu) | 0x7fc0u));
          out = (out & ~(0xffffu << (16 * h))) | (n << (16 * h));
        }
        if (!any)
          continue;
        nan_produced_ = true;
        if (nan_choice_)
          cu_->write_vgpr(vbase + d, lane, out);
      }
    }
  }
}

std::string RegionRunner::poison_after(size_t i, uint64_t exec) const {
  const uint32_t kinds = poison_kind_[i] & poison_filter_ & (kPoisonNnan | kPoisonNinf);
  if (!kinds)
    return {};
  const FpPlan &p = fp_plan_[i];
  if (p.skind == FpResult::F32 || p.skind == FpResult::F16Lo || p.skind == FpResult::F16x2)
    for (unsigned l : p.sdsts) {
      const uint32_t w = read_loc(l);
      const bool f32 = p.skind == FpResult::F32;
      const bool nan =
          f32 ? f32_nan(w)
              : f16_nan(w & 0xffffu) || (p.skind == FpResult::F16x2 && f16_nan(w >> 16));
      const bool inf =
          f32 ? f32_inf(w)
              : f16_inf(w & 0xffffu) || (p.skind == FpResult::F16x2 && f16_inf(w >> 16));
      if ((nan && (kinds & kPoisonNnan)) || (inf && (kinds & kPoisonNinf)))
        return std::string(nan ? "nnan" : "ninf") + ": a " + (nan ? "NaN" : "infinite") +
               " result at '" + texts_[i] + "'";
    }
  for (unsigned d : p.dsts) {
    if (d >= vgpr_count_)
      continue;
    for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
      if (!((exec >> lane) & 1u))
        continue;
      const uint8_t t = vfp_[size_t{d} * kWaveLanes + lane];
      const uint32_t w = read_vgpr(d, lane);
      bool nan = false, inf = false;
      if (t & kFpF32) {
        nan = f32_nan(w);
        inf = f32_inf(w);
      } else if ((t & kFpF64Lo) && d + 1 < vgpr_count_) {
        const uint64_t v = uint64_t{w} | (uint64_t{read_vgpr(d + 1, lane)} << 32);
        nan = f64_nan(v);
        inf = f64_inf(v);
      } else {
        if (t & kFpF16Lo) {
          nan = nan || f16_nan(w & 0xffffu);
          inf = inf || f16_inf(w & 0xffffu);
        }
        if (t & kFpF16Hi) {
          nan = nan || f16_nan(w >> 16);
          inf = inf || f16_inf(w >> 16);
        }
        if (t & kFpBF16Lo) {
          nan = nan || bf16_nan(w & 0xffffu);
          inf = inf || bf16_inf(w & 0xffffu);
        }
        if (t & kFpBF16Hi) {
          nan = nan || bf16_nan(w >> 16);
          inf = inf || bf16_inf(w >> 16);
        }
      }
      if ((kinds & kPoisonNnan) && nan)
        return "nnan: NaN result at '" + texts_[i] + "' lane " + std::to_string(lane);
      if ((kinds & kPoisonNinf) && inf)
        return "ninf: infinite result at '" + texts_[i] + "' lane " + std::to_string(lane);
    }
  }
  return {};
}

uint32_t inline_asm_word(uint64_t seed, uint32_t block, uint32_t slot, uint32_t lane) {
  return initial_memory_word(seed ^ 0x61736d0000000000ull,
                             (uint64_t{block} << 40) | (uint64_t{slot} << 8) | lane);
}

RunOutcome RegionRunner::run_inline_asm(size_t i) {
  RunOutcome out;
  const InlineAsmEffect &e = *asm_[i];
  const uint32_t block = asm_block_[i];
  amdgpu::Wavefront &wf = *wave_;
  const uint32_t vbase = wf.vgpr_alloc().base;
  if (e.opaque || e.untracked) {
    out.status = RunStatus::UnsupportedOperand;
    out.failed_index = i;
    out.detail = e.reason;
    return out;
  }
  for (const auto *slots : {&e.uses, &e.defs})
    for (const AsmSlot &s : *slots)
      if (s.vgpr && s.index >= vgpr_count_) {
        out.status = RunStatus::InvalidRegisterAccess;
        out.failed_index = i;
        out.detail = "inline asm names v" + std::to_string(s.index) + " beyond the tracked VGPRs";
        return out;
      }
  if (!e.uses.empty()) {
    AsmUse u;
    u.block = block;
    const uint64_t exec = exec_lanes();
    bool vector = false;
    for (const AsmSlot &s : e.uses) {
      if (!s.vgpr) {
        u.values.push_back(read_loc(s.index));
        u.fp.push_back(s.index < kNumLocs ? sfp_[s.index] : 0);
        continue;
      }
      vector = true;
      for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
        const bool on = (exec >> lane) & 1u;
        u.values.push_back(on ? cu_->read_vgpr_storage(vbase + s.index, lane) : 0u);
        u.fp.push_back(on ? vfp_[size_t{s.index} * kWaveLanes + lane] : 0);
      }
    }
    if (vector) {
      u.values.push_back(static_cast<uint32_t>(exec));
      u.values.push_back(static_cast<uint32_t>(exec >> 32));
      u.fp.push_back(0);
      u.fp.push_back(0);
    }
    asm_uses_.push_back(std::move(u));
  }
  for (size_t j = 0; j < e.defs.size(); ++j) {
    const AsmSlot &s = e.defs[j];
    const auto slot = static_cast<uint32_t>(j);
    if (s.vgpr) {
      // Per-thread semantics: the lanes in EXEC get the value, the others keep theirs (as
      // for a VALU def; LLVM may keep another thread's value in an inactive lane).
      const uint64_t exec = exec_lanes();
      for (unsigned lane = 0; lane < kWaveLanes; ++lane)
        if ((exec >> lane) & 1u) {
          cu_->write_vgpr(vbase + s.index, lane, inline_asm_word(asm_seed_, block, slot, lane));
          vfp_[size_t{s.index} * kWaveLanes + lane] = 0; // an opaque value
        }
    } else {
      uint32_t v = inline_asm_word(asm_seed_, block, slot, 0);
      // Pointer-shaped where the region uses it as a memory base (see MemoryImage::word).
      const uint64_t h = splitmix64(asm_seed_ ^ (uint64_t{block} << 32) ^ slot ^ 0x7074725f);
      if (j < 32 && ((pointer_lo_[i] >> j) & 1u))
        v = static_cast<uint32_t>(h) & 0xffff0000u;
      else if (j < 32 && ((pointer_hi_[i] >> j) & 1u))
        v = 0x10u + static_cast<uint32_t>((h >> 32) % 0x70u);
      write_loc(s.index, v & loc_value_mask(s.index));
      if (s.index < kNumLocs)
        sfp_[s.index] = 0; // an opaque value
    }
  }
  return out;
}

RunOutcome RegionRunner::run(size_t budget, std::vector<TraceStep> *trace) {
  RunOutcome out;
  amdgpu::Wavefront &wf = *wave_;
  MachineState prev;
  if (trace)
    prev = capture();
  // Without a backward branch every instruction runs at most once; a loop gets a step
  // limit, past which the case is excluded.
  constexpr size_t kLoopStepFactor = 64, kMaxLoopSteps = 1u << 16;
  const size_t limit = back_edge_ ? std::min(std::max<size_t>(budget, 1) * kLoopStepFactor,
                                             std::max<size_t>(budget, kMaxLoopSteps))
                                  : budget;
  exit_pc_ = 0;
  bool left = false;
  size_t next = 0;
  for (size_t i = 0; i < insts_.size() && !left; i = next) {
    next = i + 1;
    if (out.executed >= limit) {
      out.status = back_edge_ ? RunStatus::StepLimit : RunStatus::BudgetExceeded;
      out.failed_index = i;
      return out;
    }
    if (decode_failed_ && i >= decode_failed_index_) {
      out.status = RunStatus::DecodeFailure;
      out.failed_index = decode_failed_index_;
      out.detail = decode_error_;
      return out;
    }
    if (asm_[i]) {
      RunOutcome a = run_inline_asm(i);
      if (a.status != RunStatus::Ok) {
        a.executed = out.executed;
        return a;
      }
      ++out.executed;
      ++total_executed_;
      if (trace) {
        MachineState now = capture();
        TraceStep step;
        step.index = i;
        step.pc = pcs_[i];
        step.text = texts_[i];
        step.changed = differing_locs(prev, now);
        step.vchanged = differing_vgprs(prev, now);
        step.frame_changed = differing_frame_fields(prev.frame, now.frame);
        step.after = now;
        trace->push_back(std::move(step));
        prev = std::move(now);
      }
      continue;
    }
    Instruction *inst = insts_[i].get();
    const uint64_t exec_before = exec_lanes();
    plan_tags(i, scratch_tags_);
    scratch_stags_ = plan_stags(i);
    if (is_mem_[i] && inst && inst->execute && !model_only_[i]) {
      // The full issue path: fetch at the PC, execute, route, complete (functional).
      wf.pc = pcs_[i];
      if (mem_) {
        mem_->set_access(image_.get(), pointer_lo_[i], pointer_hi_[i]);
        // A store's data carries its floating-point provenance into memory: the data
        // operand is the second VGPR operand (after the address).
        std::array<std::array<uint8_t, 4>, kWaveLanes> lane_tags{};
        const FpPlan &plan = fp_plan_[i];
        if (plan.store) {
          const OperandRef *data = nullptr;
          int vgprs_seen = 0;
          for (const OperandRef &o : srcs_[i])
            if (o.kind == OperandRef::Vgpr && ++vgprs_seen == 2) {
              data = &o;
              break;
            }
          if (data)
            for (unsigned lane = 0; lane < kWaveLanes; ++lane)
              for (unsigned k = 0; k < 4 && k < std::max(data->count, 1u); ++k)
                if (data->reg + k < vgpr_count_)
                  lane_tags[lane][k] = vfp_[size_t{data->reg + k} * kWaveLanes + lane];
        }
        mem_->set_store_fp(plan.store, plan.store_hi16, lane_tags);
        // Seeding writes backing memory when the access is observed (before the pipeline
        // reads it). Write back and drop every cached line first, so the access fills
        // its lines from the seeded memory instead of an earlier, partially seeded fill.
        cu_->flush_all();
        l2_->invalidate_all();
        cu_->flush_l1();
      }
      const uint64_t ub_before = cu_->isa_diagnostic_count();
      try {
        (void)cu_->step();
      } catch (const std::exception &e) {
        out.status = RunStatus::HostException;
        out.failed_index = i;
        out.detail = e.what();
        return out;
      }
      ++out.executed;
      ++total_executed_;
      if (wf.instruction_execution_failed()) {
        out.status = map_error(wf.instruction_execution_error());
        out.failed_index = i;
        out.detail = "memory instruction failed";
        return out;
      }
      if (mem_ && (mem_->unsupported() || mem_->touches_code())) {
        out.status = RunStatus::UnsupportedOperand;
        out.failed_index = i;
        out.detail = mem_->unsupported() ? "atomic or dual-address memory access"
                                         : "store into the region's own code";
        return out;
      }
      if (cu_->isa_diagnostic_count() != ub_before) {
        out.status = RunStatus::UndefinedBehavior;
        out.failed_index = i;
        out.detail = "RocJITsu reported ISA undefined behaviour (see isa-undefined log)";
        return out;
      }
      if (mem_ && mem_->wraps()) {
        out.status = RunStatus::MemoryFault;
        out.failed_index = i;
        out.detail = "memory access wraps around the 64-bit address space at '" + texts_[i] + "'";
        return out;
      }
      if (wf.state() == amdgpu::WfState::HALTED || wf.state() == amdgpu::WfState::MEMORY_TRAP) {
        out.status = RunStatus::MemoryFault;
        out.failed_index = i;
        out.detail = "memory access fault at '" + texts_[i] + "'";
        return out;
      }
      if (wf.state() != amdgpu::WfState::RUNNING) {
        out.status = RunStatus::WaveStateChanged;
        out.failed_index = i;
        out.detail = "the wave stopped running at a memory instruction (state " +
                     std::to_string(static_cast<unsigned>(wf.state())) + ")";
        return out;
      }
      if (wf.pc != pcs_[i] + inst->size()) {
        out.status = RunStatus::PcWritten;
        out.failed_index = i;
        return out;
      }
      apply_tags(i, exec_before, scratch_tags_); // load destinations: no provenance
      apply_stags(i, scratch_stags_);
      if (trace) {
        MachineState now = capture();
        TraceStep step;
        step.index = i;
        step.pc = pcs_[i];
        step.text = texts_[i];
        step.changed = differing_locs(prev, now);
        step.vchanged = differing_vgprs(prev, now);
        step.frame_changed = differing_frame_fields(prev.frame, now.frame);
        step.after = now;
        trace->push_back(std::move(step));
        prev = std::move(now);
      }
      continue;
    }
    if (!inst->execute || model_only_[i]) {
      out.status = RunStatus::MissingExecutionBackend;
      out.failed_index = i;
      out.detail = "instruction has no execution callback";
      return out;
    }
    wf.pc = pcs_[i];
    if (poison_filter_ && poison_hit_.empty())
      poison_hit_ = poison_before(i);
    const uint64_t ub_before = cu_->isa_diagnostic_count();
    std::string detail;
    util::Result result = util::Result::failure();
    try {
      result = cu_->execute_instruction(inst, wf, &detail);
    } catch (const std::exception &e) {
      out.status = RunStatus::HostException;
      out.failed_index = i;
      out.detail = e.what();
      return out;
    } catch (...) {
      out.status = RunStatus::HostException;
      out.failed_index = i;
      out.detail = "unknown exception";
      return out;
    }
    ++out.executed;
    ++total_executed_;
    if (result.failed()) {
      out.status = map_error(wf.instruction_execution_error());
      out.failed_index = i;
      out.detail = detail;
      return out;
    }
    if (cu_->isa_diagnostic_count() != ub_before) {
      out.status = RunStatus::UndefinedBehavior;
      out.failed_index = i;
      out.detail = "RocJITsu reported ISA undefined behaviour (see isa-undefined log)";
      return out;
    }
    if (branch_[i]) {
      // Taken: RocJITsu leaves PC at target - size; not taken: unchanged. Either way the
      // next PC is PC + size. Falling through past the end, or into the region's own
      // fallthrough PC, is the fallthrough exit (0).
      const uint64_t to = wf.pc + static_cast<uint64_t>(inst->size());
      wf.pc = pcs_[i];
      const uint64_t end_pc = pcs_.back() + (insts_.back() ? insts_.back()->size() : 0);
      if (auto it = pc_index_.find(to); it != pc_index_.end()) {
        next = it->second;
      } else {
        exit_pc_ = to == end_pc ? 0 : to;
        left = true;
      }
    } else if (wf.pc != pcs_[i]) {
      out.status = RunStatus::PcWritten;
      out.failed_index = i;
      return out;
    }
    if (wf.state() != amdgpu::WfState::RUNNING) {
      out.status = RunStatus::WaveStateChanged;
      out.failed_index = i;
      return out;
    }
    apply_tags(i, exec_before, scratch_tags_);
    apply_stags(i, scratch_stags_);
    choose_nans(i, exec_before);
    if (poison_filter_ && poison_hit_.empty())
      poison_hit_ = poison_after(i, exec_before);
    if (trace) {
      MachineState now = capture();
      TraceStep step;
      step.index = i;
      step.pc = pcs_[i];
      step.text = texts_[i];
      step.changed = differing_locs(prev, now);
      step.vchanged = differing_vgprs(prev, now);
      step.frame_changed = differing_frame_fields(prev.frame, now.frame);
      step.after = now;
      trace->push_back(std::move(step));
      prev = std::move(now);
    }
  }
  out.exit_pc = exit_pc_;
  return out;
}

} // namespace rocjitsu::isa_diff
