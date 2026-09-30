// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/fp_math_provider.h"

#include <cstdlib>
#include <stdexcept>
#include <string_view>

#ifndef ROCJITSU_FP_MATH_X86_BACKENDS
#define ROCJITSU_FP_MATH_X86_BACKENDS 0
#endif

#ifndef ROCJITSU_FP_MATH_EXTERNAL_V4
#define ROCJITSU_FP_MATH_EXTERNAL_V4 0
#endif

#if ROCJITSU_FP_MATH_EXTERNAL_V4
#include "rocjitsu/isa/arch/amdgpu/shared/x86_v4_provider.h"
#endif

#if ROCJITSU_FP_MATH_X86_BACKENDS
#include <cpuid.h>
#endif

namespace rocjitsu::amdgpu::fp_math {
namespace detail {
extern const bool explicit_backend_requested = [] {
  const char *raw = std::getenv("RJ_MATH_BACKEND");
  return raw && std::string_view(raw) != "auto";
}();
} // namespace detail

namespace {

enum class BackendRequest : uint8_t { Auto, Scalar, X86V3, X86V4 };

struct BackendSelection {
  HostCpuTier tier;
  bool enable_exp_log;
};

BackendRequest read_backend_request() {
  const char *raw = std::getenv("RJ_MATH_BACKEND");
  if (!raw || std::string_view(raw) == "auto")
    return BackendRequest::Auto;
  if (std::string_view(raw) == "scalar")
    return BackendRequest::Scalar;
  if (std::string_view(raw) == "v3")
    return BackendRequest::X86V3;
  if (std::string_view(raw) == "v4")
    return BackendRequest::X86V4;
  throw std::invalid_argument("RJ_MATH_BACKEND must be auto, scalar, v3, or v4");
}

#if ROCJITSU_FP_MATH_X86_BACKENDS
uint64_t xcr0() noexcept {
  uint32_t low = 0, high = 0;
  __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
  return (uint64_t{high} << 32) | low;
}

bool cpu_and_os_support_v3() noexcept {
  unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
  constexpr unsigned leaf1_required =
      (1u << 12) | (1u << 22) | (1u << 27) | (1u << 28) | (1u << 29);
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx) || (ecx & leaf1_required) != leaf1_required)
    return false;
  if ((xcr0() & 0x6u) != 0x6u)
    return false;
  constexpr unsigned leaf7_required = (1u << 3) | (1u << 5) | (1u << 8);
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) || (ebx & leaf7_required) != leaf7_required)
    return false;
  return __get_cpuid(0x80000001u, &eax, &ebx, &ecx, &edx) && (ecx & (1u << 5));
}

bool cpu_and_os_support_v4() noexcept {
#if ROCJITSU_FP_MATH_EXTERNAL_V4
  return cpu_and_os_support_x86_v4();
#else
  if (!cpu_and_os_support_v3())
    return false;
  unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
    return false;
  constexpr unsigned avx512_required =
      (1u << 16) | (1u << 17) | (1u << 28) | (1u << 30) | (1u << 31);
  return (ebx & avx512_required) == avx512_required && (xcr0() & 0xe6u) == 0xe6u;
#endif
}
#endif

BackendSelection select_backend() {
  if (util::force_scalar())
    return {HostCpuTier::Scalar, true};
  const BackendRequest request = read_backend_request();
  if (request == BackendRequest::Scalar)
    return {HostCpuTier::Scalar, true};
#if ROCJITSU_FP_MATH_X86_BACKENDS
  const bool v3 = cpu_and_os_support_v3();
  const bool v4 = v3 && cpu_and_os_support_v4();
  if (request == BackendRequest::X86V3) {
    if (!v3)
      throw std::runtime_error("RJ_MATH_BACKEND=v3 requires AVX2 and OS-enabled AVX state");
    return {HostCpuTier::X86V3, true};
  }
  if (request == BackendRequest::X86V4) {
    if (!v4)
      throw std::runtime_error("RJ_MATH_BACKEND=v4 requires AVX-512F/BW/CD/DQ/VL and OS state");
    return {HostCpuTier::X86V4, true};
  }
  // The matched Release dispatch benchmark did not meet the RFC's 5% gain
  // gate for EXP/LOG. Auto keeps the original handler until it does; explicit
  // selections remain available for qualification and profiling.
  return {v4   ? HostCpuTier::X86V4
          : v3 ? HostCpuTier::X86V3
               : HostCpuTier::Scalar,
          detail::auto_adopt_exp_log};
#else
  if (request == BackendRequest::X86V3 || request == BackendRequest::X86V4)
    throw std::runtime_error("RJ_MATH_BACKEND requests an unavailable x86 backend");
  return {HostCpuTier::Scalar, detail::auto_adopt_exp_log};
#endif
}

#if ROCJITSU_FP_MATH_X86_BACKENDS
template <Operation Op, F32Bits8 (*Kernel)(F32Bits8, bool) noexcept>
void execute_eight(const F32UnaryWave &wave) {
  constexpr uint32_t inactive_input = Op == Operation::Exp ? 0u : 0x3f800000u;
  for (unsigned base = 0; base < wave.size(); base += 8) {
    const LaneMask<8> active = wave.active().chunk<8>(base);
    if (active.empty())
      continue;
    const F32Bits8 input{wave.source().load<8>(base, active, inactive_input)};
    const F32Bits8 result = Kernel(input, wave.mode().quiet_snan);
    wave.destination().store<8>(base, result.lane, active);
  }
}

constexpr UnaryF32Kernel v3_exp{&execute_eight<Operation::Exp, &exp_v3>, MathKernelKind::Avx2x8, 8,
                                1};
constexpr UnaryF32Kernel v3_log{&execute_eight<Operation::Log, &log_v3>, MathKernelKind::Avx2x8, 8,
                                1};
#if !ROCJITSU_FP_MATH_EXTERNAL_V4
constexpr UnaryF32Kernel v4_exp{&execute_eight<Operation::Exp, &exp_v4x8>, MathKernelKind::Avx512x8,
                                8, 1};
constexpr UnaryF32Kernel v4_log{&execute_eight<Operation::Log, &log_v4x8>, MathKernelKind::Avx512x8,
                                8, 1};
#endif
#endif

constexpr UnaryF32Kernel scalar{nullptr, MathKernelKind::Scalar, 1, 0};

} // namespace

bool is_qualified(FpQualificationKey key) noexcept {
  // The retained physical captures are gfx1201, default MODE, ordinary FP32
  // VOP1. gfx1100 is not yet a concrete RocJITsu target. Do not extrapolate
  // to gfx1200, CDNA, VOP3, SDWA/DPP, or changed guest FP controls.
  return key.gfx == ROCJITSU_CODE_TARGET_GFX1201 && key.form == InstructionForm::Vop1 &&
         (key.operation == Operation::Exp || key.operation == Operation::Log) &&
         key.source_format == F32Format::F32 && key.destination_format == F32Format::F32 &&
         key.source_modifiers == 0 && key.destination_modifiers == 0 && key.round_mode == 0 &&
         key.denorm_mode == 0 && !key.ieee_mode;
}

const FpProvider &fp_provider() {
  static const FpProvider provider = [] {
    const BackendSelection selected = select_backend();
#if ROCJITSU_FP_MATH_X86_BACKENDS
    if (selected.tier == HostCpuTier::X86V3)
      return FpProvider{selected.tier, selected.enable_exp_log, &v3_exp, &v3_log};
#if ROCJITSU_FP_MATH_EXTERNAL_V4
    if (selected.tier == HostCpuTier::X86V4 && selected.enable_exp_log) {
      const auto *v4 = resolve_x86_v4_provider(true);
      return FpProvider{selected.tier, selected.enable_exp_log, &v4->exp, &v4->log};
    }
#else
    if (selected.tier == HostCpuTier::X86V4)
      return FpProvider{selected.tier, selected.enable_exp_log, &v4_exp, &v4_log};
#endif
#endif
    return FpProvider{selected.tier, selected.enable_exp_log};
  }();
  return provider;
}

const UnaryF32Kernel *FpProvider::unary_f32(FpQualificationKey key) const noexcept {
  if (!is_qualified(key) || !enable_exp_log_)
    return nullptr;
  if (tier_ == HostCpuTier::Scalar)
    return &scalar;
  return key.operation == Operation::Exp ? exp_ : log_;
}

} // namespace rocjitsu::amdgpu::fp_math
