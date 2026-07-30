// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/shader_engine.h"
#include "rocjitsu/vm/amdgpu/xcd.h"

#include <stdexcept>

namespace rocjitsu {
namespace amdgpu {

namespace {

struct Gfx12TopologyLocation {
  uint32_t compute_unit = 0;
  uint32_t shader_engine = 0;
};

[[nodiscard]] Gfx12TopologyLocation topology_location(const ComputeUnitCore &cu) {
  Gfx12TopologyLocation location;
  const auto *shader_engine = dynamic_cast<const ShaderEngine *>(cu.parent());
  if (!shader_engine)
    return location;

  bool found_compute_unit = false;
  for (uint32_t i = 0; i < shader_engine->num_compute_units(); ++i) {
    if (shader_engine->compute_unit(i) == &cu) {
      location.compute_unit = i;
      found_compute_unit = true;
      break;
    }
  }
  if (!found_compute_unit)
    throw std::runtime_error("compute unit is missing from its parent shader engine");

  const auto *xcd = dynamic_cast<const Xcd *>(shader_engine->parent());
  if (!xcd)
    return location;
  bool found_shader_engine = false;
  for (uint32_t i = 0; i < xcd->num_shader_engines(); ++i) {
    if (xcd->shader_engine(i) == shader_engine) {
      location.shader_engine = i;
      found_shader_engine = true;
      break;
    }
  }
  if (!found_shader_engine)
    throw std::runtime_error("shader engine is missing from its parent XCD");
  return location;
}

} // namespace

Lds &Wavefront::lds() { return lds_ ? *lds_ : cu_.lds(); }

const Lds &Wavefront::lds() const { return lds_ ? *lds_ : cu_.lds(); }

uint32_t Wavefront::hw_id1_raw() const {
  constexpr uint32_t kSimdIdShift = 8u;
  constexpr uint32_t kWgpIdShift = 10u;
  constexpr uint32_t kShaderArrayIdShift = 16u;
  constexpr uint32_t kShaderEngineIdShift = 18u;
  constexpr uint32_t kWaveIdLimit = 32u;
  constexpr uint32_t kSimdIdLimit = 4u;
  constexpr uint32_t kWgpIdLimit = 16u;
  constexpr uint32_t kShaderArrayIdLimit = 2u;
  constexpr uint32_t kShaderEngineIdLimit = 8u;

  const rj_code_arch_t arch = cu_.arch();
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_GFX1250)
    throw std::runtime_error("HW_ID1 is only modeled for GFX12-family targets");
  const Gfx12TopologyLocation location = topology_location(cu_);

  uint32_t wave_id = 0;
  uint32_t simd_id = 0;
  uint32_t wgp_id = 0;
  uint32_t shader_array_id = 0;
  uint32_t shader_engine_id = 0;
  if (arch == ROCJITSU_CODE_ARCH_GFX1250) {
    // GFX12.1 models four SIMDs in each CU and four CUs in each shader
    // array. Its HW_ID1 layout has no SE_ID field.
    constexpr uint32_t kSimdsPerCu = 4u;
    constexpr uint32_t kCusPerShaderArray = 4u;
    wave_id = wf_id_ / kSimdsPerCu;
    simd_id = wf_id_ % kSimdsPerCu;
    shader_array_id = location.compute_unit / kCusPerShaderArray;
    wgp_id = location.compute_unit % kCusPerShaderArray;
  } else {
    // GFX12.0 SIMD_ID is within the WGP: bit 0 selects the CU and bit 1
    // selects one of the two SIMDs in that CU. Each shader array has four
    // two-CU WGPs.
    constexpr uint32_t kSimdsPerCu = 2u;
    constexpr uint32_t kCusPerShaderArray = 8u;
    const uint32_t cu_within_shader_array = location.compute_unit % kCusPerShaderArray;
    wave_id = wf_id_ / kSimdsPerCu;
    simd_id = (cu_within_shader_array % 2u) | ((wf_id_ % kSimdsPerCu) << 1u);
    wgp_id = cu_within_shader_array / 2u;
    shader_array_id = location.compute_unit / kCusPerShaderArray;
    shader_engine_id = location.shader_engine;
  }

  if (wave_id >= kWaveIdLimit || simd_id >= kSimdIdLimit || wgp_id >= kWgpIdLimit ||
      shader_array_id >= kShaderArrayIdLimit || shader_engine_id >= kShaderEngineIdLimit) {
    throw std::runtime_error("simulator topology cannot be represented in HW_ID1");
  }
  return wave_id | (simd_id << kSimdIdShift) | (wgp_id << kWgpIdShift) |
         (shader_array_id << kShaderArrayIdShift) | (shader_engine_id << kShaderEngineIdShift);
}

uint32_t Wavefront::hw_id2_raw() const {
  constexpr uint32_t kWorkgroupIdShift = 16u;
  constexpr uint32_t kVmIdShift = 24u;
  constexpr uint32_t kQueueIdMask = 0xfu;
  constexpr uint32_t kWorkgroupIdMask = 0x1fu;
  constexpr uint32_t kVmIdMask = 0xfu;
  return (queue_id_ & kQueueIdMask) | ((wg_id_ & kWorkgroupIdMask) << kWorkgroupIdShift) |
         ((process_id_ & kVmIdMask) << kVmIdShift);
}

void Wavefront::halt() {
  // s_endpgm terminates the wave, frees its resources, and notifies the CP as one
  // action, mirroring hardware. Order matters:
  //   (1) fire the halt hook while registers are still live so observers snapshot
  //       final state before it is freed,
  //   (2) free SGPR/VGPR and reset the slot (sets state HALTED); capture the WG ids
  //       first because reset() zeroes them,
  //   (3) notify the CU/CP of workgroup completion. Freeing before release_wf keeps
  //       has_active_wfs() accurate so the last wave triggers LDS reclaim.
  cu_.plugin_group().onAmdgpuWavefrontHalted(*this);
  const uint32_t dispatch_id = dispatch_id_;
  const uint32_t wg_id = wg_id_;
  cu_.free_wavefront_resources(*this);
  cu_.release_wf(dispatch_id, wg_id);
}

void Wavefront::release_wait_counter(WaitCounterType type) {
  wait_counters_.decrement(type);
  if (state_ == WfState::WAITCNT && wait_satisfied())
    state_ = WfState::RUNNING;
  if (state_ == WfState::ENDING && wait_counters_.empty())
    halt();
}

} // namespace amdgpu
} // namespace rocjitsu
