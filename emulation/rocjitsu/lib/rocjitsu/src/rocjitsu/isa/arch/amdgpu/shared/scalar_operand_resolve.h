// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_OPERAND_RESOLVE_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_OPERAND_RESOLVE_H_

#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_static_resolve.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace rocjitsu {
namespace amdgpu {

// resolve_src_scalar_statically() (the wavefront-free inline-constant subset) is
// defined in scalar_static_resolve.h, included above, so model-side code can use
// it without depending on the vm/ simulator layer. Keep the encoding-value
// handling in resolve_src_scalar() below in sync with it.

// The value of a scalar source operand resolvable from wavefront state alone
// (SGPR/VCC/EXEC/M0 reads plus inline constants). `m0_ev` is the M0 encoding
// value for this arch (124 on most arches; 125 on RDNA 3 / RDNA 3.5 / RDNA4
// / GFX1250, where 124 is the NULL slot).
inline uint32_t resolve_src_scalar(const Wavefront &wf, int ev, int m0_ev) {
  if (arch_uses_legacy_flat_scratch_sgprs(wf.cu().arch()) &&
      ev == static_cast<int>(kFlatScratchSelectorFirst))
    return static_cast<uint32_t>(wf.scratch_base());
  if (arch_uses_legacy_flat_scratch_sgprs(wf.cu().arch()) &&
      ev == static_cast<int>(kFlatScratchSelectorLast))
    return static_cast<uint32_t>(wf.scratch_base() >> 32);
  if (ev <= static_cast<int>(kScalarSgprSelectorLast))
    return RegisterAccess(wf).read_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev));
  if (ev == static_cast<int>(kVccSelectorFirst))
    return static_cast<uint32_t>(wf.vcc());
  if (ev == static_cast<int>(kVccSelectorLast))
    return static_cast<uint32_t>(wf.vcc() >> 32);
  // TTMP0-15 are the trap handler's private scratch registers. They are NOT
  // part of the wave's SGPR allocation: hardware banks them separately, the CP
  // seeds them with the dispatch/queue identity that rocm-dbgapi reads back out
  // of the CWSR area, and a shader that never enters a trap must not be able to
  // clobber them through an SGPR write.
  if (ev >= static_cast<int>(kTtmpSelectorFirst) && ev <= static_cast<int>(kTtmpSelectorLast))
    return RegisterAccess(wf).read_ttmp(static_cast<uint32_t>(ev) - kTtmpSelectorFirst);
  if (m0_ev == static_cast<int>(kModernM0Selector) && ev == static_cast<int>(kModernNullSelector))
    return 0u; // NULL
  if (ev == m0_ev)
    return wf.m0();
  if (ev == static_cast<int>(kExecSelectorFirst))
    return static_cast<uint32_t>(wf.exec());
  if (ev == static_cast<int>(kExecSelectorLast))
    return static_cast<uint32_t>(wf.exec_raw() >> 32);
  if (ev >= 128 && ev <= 192)
    return static_cast<uint32_t>(ev - 128);
  if (ev >= 193 && ev <= 208)
    return static_cast<uint32_t>(static_cast<int32_t>(-(ev - 192)));
  if (ev == static_cast<int>(kFlatScratchBaseSelectorFirst))
    return static_cast<uint32_t>(wf.scratch_base()); // SRC_FLAT_SCRATCH_BASE_LO
  if (ev == static_cast<int>(kFlatScratchBaseSelectorLast))
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
  if (ev == 249)
    return 0u; // SRC_POPS_EXITING_WAVE_ID (not used in compute)
  if (ev == 250)
    return 0u; // NULL
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
inline uint32_t resolve_src_scalar16(const Wavefront &wf, int ev, int m0_ev) {
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
    return resolve_src_scalar(wf, ev, m0_ev);
  }
}

// Must stay in sync with resolve_src_scalar above -- returns true for exactly
// the encoding values that resolve_src_scalar handles without throwing. Used by
// Isa::simd_capable_value() to keep the SIMD fast path off operands whose
// scalar broadcast would throw at runtime.
inline bool can_resolve_src_scalar(int ev, int m0_ev) {
  bool ok =
      (ev >= 0 && ev <= static_cast<int>(kVccSelectorLast)) ||
      (ev >= static_cast<int>(kTtmpSelectorFirst) && ev <= static_cast<int>(kTtmpSelectorLast)) ||
      ev == static_cast<int>(kLegacyM0Selector) ||
      (ev >= static_cast<int>(kExecSelectorFirst) && ev <= static_cast<int>(kExecSelectorLast)) ||
      (ev >= 128 && ev <= 208) || (ev >= 235 && ev <= 238) || (ev >= 240 && ev <= 253);
  if (m0_ev == static_cast<int>(kModernM0Selector))
    ok = ok || ev == static_cast<int>(kModernM0Selector) ||
         ev == static_cast<int>(kFlatScratchBaseSelectorFirst) ||
         ev == static_cast<int>(kFlatScratchBaseSelectorLast);
  return ok;
}

inline uint64_t resolve_src_scalar64(const Wavefront &wf, int ev, int m0_ev) {
  if (is_src_scalar_register_pair(ev)) {
    if (arch_uses_legacy_flat_scratch_sgprs(wf.cu().arch()) &&
        ev == static_cast<int>(kFlatScratchSelectorFirst))
      return wf.scratch_base();
    if (ev <= static_cast<int>(kScalarSgprSelectorLast))
      return RegisterAccess(wf).read_sgpr64(wf.sgpr_alloc().base + static_cast<uint32_t>(ev));
    if (ev == static_cast<int>(kVccSelectorFirst))
      return wf.vcc();
    if (ev >= static_cast<int>(kTtmpSelectorFirst) &&
        ev < static_cast<int>(kTtmpSelectorLast)) { // TTMP pair; see resolve_src_scalar()
      return RegisterAccess(wf).read_ttmp64(static_cast<uint32_t>(ev) - kTtmpSelectorFirst);
    }
    if (ev == static_cast<int>(kExecSelectorFirst))
      return wf.exec_raw();
    if (ev == static_cast<int>(kFlatScratchBaseSelectorFirst))
      return wf.scratch_base(); // SRC_FLAT_SCRATCH_BASE
    throw std::logic_error("Scalar register-pair selector is not resolved: " + std::to_string(ev));
  }
  if (m0_ev == static_cast<int>(kModernM0Selector) && ev == static_cast<int>(kModernNullSelector))
    return 0u; // NULL
  if (ev == m0_ev)
    return wf.m0();
  if (ev >= 128 && ev <= 192)
    return static_cast<uint64_t>(ev - 128);
  if (ev >= 193 && ev <= 208)
    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(-(ev - 192))));
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

inline void resolve_dst_write(Wavefront &wf, int ev, uint32_t val, int m0_ev) {
  if (arch_uses_legacy_flat_scratch_sgprs(wf.cu().arch()) &&
      ev == static_cast<int>(kFlatScratchSelectorFirst)) {
    uint64_t sb = wf.scratch_base();
    wf.set_scratch_base((sb & 0xFFFFFFFF00000000ULL) | val);
    return;
  }
  if (arch_uses_legacy_flat_scratch_sgprs(wf.cu().arch()) &&
      ev == static_cast<int>(kFlatScratchSelectorLast)) {
    uint64_t sb = wf.scratch_base();
    wf.set_scratch_base((sb & 0x00000000FFFFFFFFULL) | (static_cast<uint64_t>(val) << 32));
    return;
  }
  if (ev <= static_cast<int>(kScalarSgprSelectorLast)) {
    RegisterAccess(wf).write_sgpr(wf.sgpr_alloc().base + static_cast<uint32_t>(ev), val);
    return;
  }
  if (ev == static_cast<int>(kVccSelectorFirst)) {
    wf.set_vcc_raw((wf.vcc() & 0xFFFFFFFF00000000ULL) | val);
    return;
  }
  if (ev == static_cast<int>(kVccSelectorLast)) {
    wf.set_vcc_raw((wf.vcc() & 0x00000000FFFFFFFFULL) | (static_cast<uint64_t>(val) << 32));
    return;
  }
  if (ev >= static_cast<int>(kTtmpSelectorFirst) &&
      ev <= static_cast<int>(kTtmpSelectorLast)) { // see resolve_src_scalar()
    RegisterAccess(wf).write_ttmp(static_cast<uint32_t>(ev) - kTtmpSelectorFirst, val);
    return;
  }
  if (m0_ev == static_cast<int>(kModernM0Selector) && ev == static_cast<int>(kModernNullSelector))
    return; // NULL
  if (ev == m0_ev) {
    wf.set_m0(val);
    return;
  }
  if (ev == static_cast<int>(kExecSelectorFirst)) {
    wf.set_exec((wf.exec() & 0xFFFFFFFF00000000ULL) | val);
    return;
  }
  if (ev == static_cast<int>(kExecSelectorLast)) {
    wf.set_exec_raw((wf.exec_raw() & 0x00000000FFFFFFFFULL) | (static_cast<uint64_t>(val) << 32));
    return;
  }
  throw std::logic_error("Unsupported encoding value for scalar write: " + std::to_string(ev));
}

inline void resolve_dst_write64(Wavefront &wf, int ev, uint64_t val) {
  if (arch_uses_legacy_flat_scratch_sgprs(wf.cu().arch()) &&
      ev == static_cast<int>(kFlatScratchSelectorFirst)) {
    wf.set_scratch_base(val);
    return;
  }
  if (ev <= static_cast<int>(kScalarSgprSelectorLast)) {
    RegisterAccess(wf).write_sgpr64(wf.sgpr_alloc().base + static_cast<uint32_t>(ev), val);
    return;
  }
  if (ev == static_cast<int>(kVccSelectorFirst)) {
    wf.set_vcc_raw(val);
    return;
  }
  if (ev >= static_cast<int>(kTtmpSelectorFirst) &&
      ev < static_cast<int>(kTtmpSelectorLast)) { // TTMP pair; see resolve_src_scalar()
    RegisterAccess(wf).write_ttmp64(static_cast<uint32_t>(ev) - kTtmpSelectorFirst, val);
    return;
  }
  if (ev == static_cast<int>(kModernNullSelector))
    return;
  if (ev == static_cast<int>(kExecSelectorFirst)) {
    wf.set_exec_raw(val);
    return;
  }
  throw std::logic_error("Unsupported encoding value for scalar64 write: " + std::to_string(ev));
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_SCALAR_OPERAND_RESOLVE_H_
