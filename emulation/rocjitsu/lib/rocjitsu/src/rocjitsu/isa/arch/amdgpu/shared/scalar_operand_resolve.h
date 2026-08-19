// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_OPERAND_RESOLVE_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_OPERAND_RESOLVE_H_

#include "rocjitsu/isa/arch/amdgpu/shared/scalar_selector_layout.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_static_resolve.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/except.h"
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace rocjitsu {
namespace amdgpu {

// resolve_src_scalar_statically() (the wavefront-free inline-constant subset) is
// defined in scalar_static_resolve.h, included above, so model-side code can use
// it without depending on the vm/ simulator layer. Keep the encoding-value
// handling in resolve_src_scalar() below in sync with it.

[[noreturn]] inline void throw_unimplemented_xnack_selector(int selector) {
  throw util::UnimplementedInst("XNACK scalar selector " + std::to_string(selector));
}

// The value of a scalar source operand resolvable from wavefront state alone:
// ordinary SGPRs, architecture-defined special scalar state, and inline
// constants. The ISA profile-generated property map is the single source of
// truth for selector layouts that differ between CDNA and RDNA generations.
inline uint32_t resolve_src_scalar(const Wavefront &wf, int ev) {
  const auto arch = wf.cu().arch();
  const auto properties = isa_properties(arch);
  if (selector_in_pair(ev, properties.scalar_flat_scratch_base_selector))
    return ev == properties.scalar_flat_scratch_base_selector
               ? static_cast<uint32_t>(wf.scratch_base())
               : static_cast<uint32_t>(wf.scratch_base() >> 32);
  if (is_xnack_scalar_selector(arch, ev))
    throw_unimplemented_xnack_selector(ev);
  if (is_ordinary_sgpr_selector(arch, ev))
    return RegisterAccess(wf).read_sgpr_or_trap_register(static_cast<uint32_t>(ev));
  if (ev == 106)
    return static_cast<uint32_t>(wf.vcc());
  if (ev == 107)
    return static_cast<uint32_t>(wf.vcc() >> 32);
  if (ev >= 108 && ev <= 123)
    return wf.read_trap_register(static_cast<uint32_t>(ev));
  if (is_null_scalar_selector(arch, ev))
    return 0u; // NULL
  if (ev == properties.scalar_m0_selector)
    return wf.m0();
  if (ev == 126)
    return static_cast<uint32_t>(wf.exec());
  if (ev == 127)
    return static_cast<uint32_t>(wf.exec_raw() >> 32);
  if (ev >= 128 && ev <= 192)
    return static_cast<uint32_t>(ev - 128);
  if (ev >= 193 && ev <= 208)
    return static_cast<uint32_t>(static_cast<int32_t>(-(ev - 192)));
  if (ev == 230)
    return static_cast<uint32_t>(wf.scratch_base()); // SRC_FLAT_SCRATCH_BASE_LO
  if (ev == 231)
    return static_cast<uint32_t>(wf.scratch_base() >> 32); // SRC_FLAT_SCRATCH_BASE_HI
  if (ev == 240)
    return 0x3F000000u; // 0.5f
  if (ev == 241)
    return 0xBF000000u; // -0.5f
  if (ev == 242)
    return 0x3F800000u; // 1.0f
  if (ev == 243)
    return 0xBF800000u; // -1.0f
  if (ev == 244)
    return 0x40000000u; // 2.0f
  if (ev == 245)
    return 0xC0000000u; // -2.0f
  if (ev == 246)
    return 0x40800000u; // 4.0f
  if (ev == 247)
    return 0xC0800000u; // -4.0f
  if (ev == 248)
    return 0x3E22F983u; // 1/(2*pi)
  if (ev == 235)
    return static_cast<uint32_t>(wf.shared_aperture_base() >> 32); // SRC_SHARED_BASE
  if (ev == 236)
    return static_cast<uint32_t>(wf.shared_aperture_limit() >> 32); // SRC_SHARED_LIMIT
  if (ev == 237)
    return static_cast<uint32_t>(wf.private_aperture_base() >> 32); // SRC_PRIVATE_BASE
  if (ev == 238)
    return static_cast<uint32_t>(wf.private_aperture_limit() >> 32); // SRC_PRIVATE_LIMIT
  if (ev == 239)
    return 0u; // SRC_POPS_EXITING_WAVE_ID (not used in compute)
  if (ev == 251)
    return wf.vcc_mask() == 0 ? 1u : 0u; // VCCZ
  if (ev == 252)
    return wf.exec() == 0 ? 1u : 0u; // EXECZ
  if (ev == 253)
    return wf.read_scc() ? 1u : 0u; // SCC
  throw std::logic_error("Unsupported encoding value for scalar read: " + std::to_string(ev));
}

// 16-bit reads of the inline float constants use the half-precision bit
// patterns rather than the single-precision ones; every other encoding value
// resolves identically to the 32-bit path.
inline uint32_t resolve_src_scalar16(const Wavefront &wf, int ev) {
  switch (ev) {
  case 240:
    return 0x3800u; // 0.5h
  case 241:
    return 0xB800u; // -0.5h
  case 242:
    return 0x3C00u; // 1.0h
  case 243:
    return 0xBC00u; // -1.0h
  case 244:
    return 0x4000u; // 2.0h
  case 245:
    return 0xC000u; // -2.0h
  case 246:
    return 0x4400u; // 4.0h
  case 247:
    return 0xC400u; // -4.0h
  case 248:
    return 0x3118u; // f16 1/(2*pi)
  default:
    return resolve_src_scalar(wf, ev);
  }
}

// Must stay in sync with resolve_src_scalar above -- returns true for exactly
// the encoding values that resolve_src_scalar handles without throwing. Used by
// Isa::simd_capable_value() to keep the SIMD fast path off operands whose
// scalar broadcast would throw at runtime.
inline bool can_resolve_src_scalar(rj_code_arch_t arch, int ev) {
  const auto properties = isa_properties(arch);
  return selector_in_pair(ev, properties.scalar_flat_scratch_base_selector) ||
         is_ordinary_sgpr_selector(arch, ev) || ev == 106 || ev == 107 ||
         (ev >= 108 && ev <= 123) || is_null_scalar_selector(arch, ev) ||
         ev == properties.scalar_m0_selector || ev == 126 || ev == 127 ||
         (ev >= 128 && ev <= 208) || ev == 230 || ev == 231 || (ev >= 235 && ev <= 248) ||
         (ev >= 251 && ev <= 253);
}

inline uint64_t resolve_src_scalar64(const Wavefront &wf, int ev) {
  const auto arch = wf.cu().arch();
  const auto properties = isa_properties(arch);
  if (ev == properties.scalar_flat_scratch_base_selector)
    return wf.scratch_base();
  if (is_xnack_scalar_selector(arch, ev))
    throw_unimplemented_xnack_selector(ev);
  if (is_ordinary_sgpr_selector_range(arch, ev, 2))
    return RegisterAccess(wf).read_sgpr_or_trap_register64(static_cast<uint32_t>(ev));
  if (ev == 106)
    return wf.vcc();
  if (ev >= 108 && ev <= 122) {
    uint32_t lo = wf.read_trap_register(static_cast<uint32_t>(ev));
    uint32_t hi = wf.read_trap_register(static_cast<uint32_t>(ev + 1));
    return static_cast<uint64_t>(hi) << 32 | lo;
  }
  if (is_null_scalar_selector(arch, ev))
    return 0u; // NULL
  if (ev == properties.scalar_m0_selector)
    return wf.m0();
  if (ev == 126)
    return wf.exec_raw();
  if (ev >= 128 && ev <= 192)
    return static_cast<uint64_t>(ev - 128);
  if (ev >= 193 && ev <= 208)
    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(-(ev - 192))));
  if (ev == 230)
    return wf.scratch_base(); // SRC_FLAT_SCRATCH_BASE
  if (ev == 240)
    return 0x3FE0000000000000ULL; // 0.5
  if (ev == 241)
    return 0xBFE0000000000000ULL; // -0.5
  if (ev == 242)
    return 0x3FF0000000000000ULL; // 1.0
  if (ev == 243)
    return 0xBFF0000000000000ULL; // -1.0
  if (ev == 244)
    return 0x4000000000000000ULL; // 2.0
  if (ev == 245)
    return 0xC000000000000000ULL; // -2.0
  if (ev == 246)
    return 0x4010000000000000ULL; // 4.0
  if (ev == 247)
    return 0xC010000000000000ULL; // -4.0
  if (ev == 248)
    return 0x3FC45F306DC9C883ULL; // 1/(2*pi)
  if (ev == 235)
    return wf.shared_aperture_base(); // SRC_SHARED_BASE
  if (ev == 236)
    return wf.shared_aperture_limit(); // SRC_SHARED_LIMIT
  if (ev == 237)
    return wf.private_aperture_base(); // SRC_PRIVATE_BASE
  if (ev == 238)
    return wf.private_aperture_limit(); // SRC_PRIVATE_LIMIT
  throw std::logic_error("Unsupported encoding value for scalar64 read: " + std::to_string(ev));
}

inline void resolve_dst_write(Wavefront &wf, int ev, uint32_t val) {
  const auto arch = wf.cu().arch();
  const auto properties = isa_properties(arch);
  if (ev == properties.scalar_flat_scratch_base_selector) {
    uint64_t sb = wf.scratch_base();
    wf.set_scratch_base((sb & 0xFFFFFFFF00000000ULL) | val);
    return;
  }
  if (properties.scalar_flat_scratch_base_selector >= 0 &&
      ev == properties.scalar_flat_scratch_base_selector + 1) {
    uint64_t sb = wf.scratch_base();
    wf.set_scratch_base((sb & 0x00000000FFFFFFFFULL) | (static_cast<uint64_t>(val) << 32));
    return;
  }
  if (is_xnack_scalar_selector(arch, ev))
    throw_unimplemented_xnack_selector(ev);
  if (is_ordinary_sgpr_selector(arch, ev)) {
    RegisterAccess(wf).write_sgpr_or_trap_register(static_cast<uint32_t>(ev), val);
    return;
  }
  if (ev == 106) {
    wf.set_vcc((wf.vcc() & 0xFFFFFFFF00000000ULL) | val);
    return;
  }
  if (ev == 107) {
    wf.set_vcc((wf.vcc() & 0x00000000FFFFFFFFULL) | (static_cast<uint64_t>(val) << 32));
    return;
  }
  if (ev >= 108 && ev <= 123) {
    wf.write_trap_register(static_cast<uint32_t>(ev), val);
    return;
  }
  if (is_null_scalar_selector(arch, ev))
    return; // NULL
  if (ev == properties.scalar_m0_selector) {
    wf.set_m0(val);
    return;
  }
  if (ev == 126) {
    wf.set_exec((wf.exec() & 0xFFFFFFFF00000000ULL) | val);
    return;
  }
  if (ev == 127) {
    wf.set_exec_raw((wf.exec_raw() & 0x00000000FFFFFFFFULL) | (static_cast<uint64_t>(val) << 32));
    return;
  }
  throw std::logic_error("Unsupported encoding value for scalar write: " + std::to_string(ev));
}

inline void resolve_dst_write64(Wavefront &wf, int ev, uint64_t val) {
  const auto arch = wf.cu().arch();
  const auto properties = isa_properties(arch);
  if (ev == properties.scalar_flat_scratch_base_selector) {
    wf.set_scratch_base(val);
    return;
  }
  if (is_xnack_scalar_selector(arch, ev))
    throw_unimplemented_xnack_selector(ev);
  if (is_ordinary_sgpr_selector_range(arch, ev, 2)) {
    RegisterAccess(wf).write_sgpr_or_trap_register64(static_cast<uint32_t>(ev), val);
    return;
  }
  if (ev == 106) {
    wf.set_vcc(val);
    return;
  }
  if (ev >= 108 && ev <= 122) {
    wf.write_trap_register(static_cast<uint32_t>(ev), static_cast<uint32_t>(val));
    wf.write_trap_register(static_cast<uint32_t>(ev + 1), static_cast<uint32_t>(val >> 32));
    return;
  }
  if (is_null_scalar_selector(arch, ev))
    return;
  if (ev == properties.scalar_m0_selector)
    throw util::UnimplementedInst("64-bit M0 scalar destination");
  if (ev == 126) {
    wf.set_exec_raw(val);
    return;
  }
  throw std::logic_error("Unsupported encoding value for scalar64 write: " + std::to_string(ev));
}

inline void resolve_dst_write_span(Wavefront &wf, int ev, std::span<const uint32_t> values) {
  if (values.empty())
    return;

  const auto arch = wf.cu().arch();
  if (is_null_scalar_selector(arch, ev))
    return;

  if (values.size() == 1) {
    resolve_dst_write(wf, ev, values.front());
    return;
  }

  if (values.size() == 2) {
    const uint64_t value =
        static_cast<uint64_t>(values[0]) | (static_cast<uint64_t>(values[1]) << 32);
    resolve_dst_write64(wf, ev, value);
    return;
  }

  const bool ordinary_range =
      is_ordinary_sgpr_selector_range(arch, ev, static_cast<uint32_t>(values.size()));
  const bool trap_range =
      ev >= static_cast<int>(Wavefront::kTrapRegisterSelectorBase) &&
      ev + static_cast<int>(values.size()) <=
          static_cast<int>(Wavefront::kTrapRegisterSelectorBase + Wavefront::kTrapRegisterCount);
  if (!ordinary_range && !trap_range) {
    throw util::UnimplementedInst("Unsupported " + std::to_string(values.size() * 32) +
                                  "-bit scalar destination starting at selector " +
                                  std::to_string(ev));
  }

  RegisterAccess(wf).write_sgpr_or_trap_registers(static_cast<uint32_t>(ev), values);
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_OPERAND_RESOLVE_H_
