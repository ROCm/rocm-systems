// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_math_kernels.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/amdgpu_exp.h"
#include "util/amdgpu_log.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <utility>

namespace rocjitsu::amdgpu::fp_math {

enum class Operation : uint8_t { Exp, Log };
enum class InstructionForm : uint8_t { Vop1, Vop3 };
enum class F32Format : uint8_t { F32 };
enum class HostCpuTier : uint8_t { Scalar, X86V3, X86V4 };
enum class MathKernelKind : uint8_t { Scalar, Avx2x8, Avx512x8, Avx512x16 };

namespace detail {
// EXP/LOG have not met the default-adoption performance gate. The cheap
// request flag lets ordinary auto execution bypass qualification entirely.
inline constexpr bool auto_adopt_exp_log = false;
extern const bool explicit_backend_requested;
} // namespace detail

// No sticky exception/status state belongs in this key. A new lookup is made
// for each instruction, so a wave's MODE change cannot retain the old entry.
struct FpQualificationKey {
  rj_code_target_id_t gfx;
  Operation operation;
  InstructionForm form;
  F32Format source_format;
  F32Format destination_format;
  uint8_t source_modifiers;
  uint8_t destination_modifiers;
  uint8_t round_mode;
  uint8_t denorm_mode;
  bool ieee_mode;
};

template <unsigned Width> class LaneMask {
public:
  static_assert(Width == 8 || Width == 16);
  static LaneMask from_low_bits(uint64_t bits) noexcept {
    return LaneMask{bits & ((uint64_t{1} << Width) - 1)};
  }
  bool empty() const noexcept { return bits_ == 0; }
  bool contains(unsigned lane) const noexcept { return (bits_ >> lane) & 1; }
  uint64_t native_bits() const noexcept { return bits_; }

private:
  explicit LaneMask(uint64_t bits) : bits_(bits) {}
  uint64_t bits_;
};

class ActiveLanes {
public:
  static ActiveLanes make(uint64_t exec, unsigned wave_size) noexcept {
    assert((wave_size == 32 || wave_size == 64) && "AMDGPU wave size must be 32 or 64");
    return ActiveLanes{wave_size == 32 ? exec & 0xffffffffu : exec, wave_size};
  }
  template <unsigned Width> LaneMask<Width> chunk(unsigned base) const noexcept {
    return LaneMask<Width>::from_low_bits(bits_ >> base);
  }
  unsigned size() const noexcept { return size_; }
  unsigned count() const noexcept { return std::popcount(bits_); }
  template <typename Fn> void for_each_active(Fn &&fn) const {
    uint64_t remaining = bits_;
    while (remaining) {
      fn(std::countr_zero(remaining));
      remaining &= remaining - 1;
    }
  }

private:
  ActiveLanes(uint64_t bits, unsigned size) : bits_(bits), size_(size) {}
  uint64_t bits_;
  unsigned size_;
};

struct F32SourcePolicy {
  bool absolute = false;
  bool negate = false;
};
struct F32DestinationPolicy {
  bool clamp = false;
};
struct F32ExecutionMode {
  bool quiet_snan = true;
};

// The register-access views, not raw VGPR pointers, are the only way through
// this adapter. Native SIMD types stay inside this baseline-compiled bridge.
class F32WaveSource {
public:
  F32WaveSource(const RegisterAccess::OperandReadView &view, F32SourcePolicy policy)
      : view_(&view), policy_(policy) {}

  uint32_t lane(unsigned index) const {
    uint32_t bits = view_->lane(index);
    if (policy_.absolute)
      bits &= 0x7fffffffu;
    if (policy_.negate)
      bits ^= 0x80000000u;
    return bits;
  }

  template <unsigned Width>
  std::array<uint32_t, Width> load(unsigned base, LaneMask<Width> active,
                                   uint32_t inactive_input) const {
    std::array<uint32_t, Width> bits{};
    constexpr unsigned native_width = util::native_width_v<uint32_t>;
    if constexpr (native_width <= Width && Width % native_width == 0) {
      for (unsigned offset = 0; offset < Width; offset += native_width)
        view_->load_native<uint32_t>(base + offset)
            .copy_to(bits.data() + offset, util::stdx::element_aligned);
    } else if constexpr (Width == 8 && util::native_width64 == 8) {
      view_->load_narrow<uint32_t>(base).copy_to(bits.data(), util::stdx::element_aligned);
    } else {
      for (unsigned lane = 0; lane < Width; ++lane)
        bits[lane] = view_->lane(base + lane);
    }
    for (unsigned lane = 0; lane < Width; ++lane) {
      if (!active.contains(lane)) {
        bits[lane] = inactive_input;
        continue;
      }
      if (policy_.absolute)
        bits[lane] &= 0x7fffffffu;
      if (policy_.negate)
        bits[lane] ^= 0x80000000u;
    }
    return bits;
  }

private:
  const RegisterAccess::OperandReadView *view_;
  F32SourcePolicy policy_;
};

class F32WaveDestination {
public:
  F32WaveDestination(const RegisterAccess::OperandWriteView &view, F32DestinationPolicy policy)
      : view_(&view), policy_(policy) {}

  void store_lane(unsigned lane, uint32_t bits) const {
    (void)policy_;
    constexpr unsigned native_width = util::native_width_v<uint32_t>;
    const unsigned base = (lane / native_width) * native_width;
    view_->store_native<uint32_t>(base, util::broadcast<uint32_t>(bits),
                                  uint64_t{1} << (lane - base));
  }

  template <unsigned Width>
  void store(unsigned base, std::array<uint32_t, Width> bits, LaneMask<Width> active) const {
    // CLAMP needs its own qualified semantics before it can be enabled.
    (void)policy_;
    constexpr unsigned native_width = util::native_width_v<uint32_t>;
    if constexpr (native_width <= Width && Width % native_width == 0) {
      for (unsigned offset = 0; offset < Width; offset += native_width) {
        const uint64_t mask =
            (active.native_bits() >> offset) & ((uint64_t{1} << native_width) - 1);
        if (mask != 0)
          view_->store_native<uint32_t>(
              base + offset,
              util::native<uint32_t>(bits.data() + offset, util::stdx::element_aligned), mask);
      }
    } else if constexpr (Width == 8 && util::native_width64 == 8) {
      view_->store_narrow<uint32_t>(
          base, util::narrow32<uint32_t>(bits.data(), util::stdx::element_aligned),
          active.native_bits());
    } else {
      static_assert(native_width <= Width || (Width == 8 && util::native_width64 == 8),
                    "FP math bridge requires a native or narrow eight-lane store");
    }
  }

private:
  const RegisterAccess::OperandWriteView *view_;
  F32DestinationPolicy policy_;
};

class F32UnaryWave {
public:
  F32UnaryWave(F32WaveSource source, F32WaveDestination destination, ActiveLanes active,
               F32ExecutionMode mode)
      : source_(source), destination_(destination), active_(active), mode_(mode) {}
  const F32WaveSource &source() const noexcept { return source_; }
  const F32WaveDestination &destination() const noexcept { return destination_; }
  const ActiveLanes &active() const noexcept { return active_; }
  F32ExecutionMode mode() const noexcept { return mode_; }
  unsigned size() const noexcept { return active_.size(); }

private:
  F32WaveSource source_;
  F32WaveDestination destination_;
  ActiveLanes active_;
  F32ExecutionMode mode_;
};

// The owner outlives wave(); the wave contains only borrowed references to its
// views and copies of small policies. No wave-sized register buffer is copied.
class F32UnaryExecution {
public:
  F32UnaryExecution(RegisterAccess::OperandReadView source,
                    RegisterAccess::OperandWriteView destination, ActiveLanes active,
                    F32SourcePolicy source_policy, F32DestinationPolicy destination_policy,
                    F32ExecutionMode mode)
      : source_(std::move(source)), destination_(std::move(destination)), active_(active),
        source_policy_(source_policy), destination_policy_(destination_policy), mode_(mode) {}

  F32UnaryWave wave() const {
    return {F32WaveSource{source_, source_policy_},
            F32WaveDestination{destination_, destination_policy_}, active_, mode_};
  }

private:
  RegisterAccess::OperandReadView source_;
  RegisterAccess::OperandWriteView destination_;
  ActiveLanes active_;
  F32SourcePolicy source_policy_;
  F32DestinationPolicy destination_policy_;
  F32ExecutionMode mode_;
};

struct UnaryF32Kernel {
  void (*execute)(const F32UnaryWave &);
  MathKernelKind kind;
  uint8_t vector_lanes;
  uint8_t scalar_threshold;
};

class FpProvider {
public:
  FpProvider(HostCpuTier tier, bool enable_exp_log, const UnaryF32Kernel *exp = nullptr,
             const UnaryF32Kernel *log = nullptr)
      : tier_(tier), enable_exp_log_(enable_exp_log), exp_(exp), log_(log) {}
  HostCpuTier selected_tier() const noexcept { return tier_; }
  // Null also means a physically modeled case has not passed the auto-adoption
  // performance gate. An explicit backend request may still exercise it.
  const UnaryF32Kernel *unary_f32(FpQualificationKey key) const noexcept;

private:
  HostCpuTier tier_;
  bool enable_exp_log_;
  const UnaryF32Kernel *exp_;
  const UnaryF32Kernel *log_;
};

const FpProvider &fp_provider();
bool is_qualified(FpQualificationKey key) noexcept;

// These adapters are invoked only after a non-null qualified entry is found.
// Force-scalar and any ineligible view retain the generated scalar handler's
// original per-lane operand access and writeback behavior.
template <Operation Op, typename Inst>
void execute_qualified_scalar_instruction(Inst &inst, Wavefront &wf) {
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if ((exec & (uint64_t{1} << lane)) == 0)
      continue;
    const uint32_t input = RegisterAccess(wf).read_lane(inst.src0, lane);
    const uint32_t output = Op == Operation::Exp ? util::detail::exp::evaluate(input)
                                                 : util::detail::log::evaluate(input);
    sdwa::write_lane<sdwa::ResultFormat::F32>(inst, wf, inst.vdst, lane, output);
  }
}

template <Operation Op> void execute_sparse_scalar_wave(const F32UnaryWave &wave) {
  wave.active().for_each_active([&](unsigned lane) {
    const uint32_t input = wave.source().lane(lane);
    const uint32_t output = Op == Operation::Exp
                                ? util::detail::exp::evaluate(input, wave.mode().quiet_snan)
                                : util::detail::log::evaluate(input, wave.mode().quiet_snan);
    wave.destination().store_lane(lane, output);
  });
}

template <Operation Op, InstructionForm Form, typename Inst>
[[nodiscard]] bool try_execute_qualified_f32_math(Inst &inst, Wavefront &wf) {
  if (!util::force_scalar() && !detail::auto_adopt_exp_log && !detail::explicit_backend_requested)
    return false;
  uint8_t source_modifiers = 0;
  uint8_t destination_modifiers = 0;
  if constexpr (Form == InstructionForm::Vop3) {
    source_modifiers = static_cast<uint8_t>((inst.inst_.abs & 1u) | ((inst.inst_.neg & 1u) << 1));
    destination_modifiers =
        static_cast<uint8_t>((inst.inst_.omod & 3u) | ((inst.inst_.clamp & 1u) << 2));
  } else if constexpr (requires { inst.inst_.src0; }) {
    if (inst.inst_.src0 == SRC_SDWA || inst.inst_.src0 == SRC_DPP ||
        dpp::is_src_dpp8(inst.inst_.src0))
      return false;
  }
  const FpQualificationKey key{wf.cu().target(),
                               Op,
                               Form,
                               F32Format::F32,
                               F32Format::F32,
                               source_modifiers,
                               destination_modifiers,
                               static_cast<uint8_t>(wf.fp_round_mode_f32()),
                               static_cast<uint8_t>(wf.fp_denorm_mode_f32()),
                               wf.ieee_mode()};
  if (!is_qualified(key))
    return false;
  if (util::force_scalar()) {
    execute_qualified_scalar_instruction<Op>(inst, wf);
    return true;
  }
  const UnaryF32Kernel *kernel = fp_provider().unary_f32(key);
  if (!kernel)
    return false;
  if (kernel->kind == MathKernelKind::Scalar || !sdwa::supports_direct_simd_store(inst) ||
      !inst.src0.simd_capable() || !inst.vdst.simd_capable()) {
    execute_qualified_scalar_instruction<Op>(inst, wf);
    return true;
  }
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto source = regs.read_operand(inst.src0, exec);
  auto destination = regs.write_operand(inst.vdst, exec);
  F32UnaryExecution execution{std::move(source),
                              std::move(destination),
                              ActiveLanes::make(exec, wf.wf_size()),
                              {},
                              {},
                              {true}};
  const F32UnaryWave wave = execution.wave();
  if (kernel->scalar_threshold != 0 && wave.active().count() != 0 &&
      wave.active().count() <= kernel->scalar_threshold)
    execute_sparse_scalar_wave<Op>(wave);
  else
    kernel->execute(wave);
  return true;
}

} // namespace rocjitsu::amdgpu::fp_math
