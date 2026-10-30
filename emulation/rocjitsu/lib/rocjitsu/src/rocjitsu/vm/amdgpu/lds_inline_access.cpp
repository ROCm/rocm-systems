// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/lds_inline_access.h"

#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/isa/arch/amdgpu/shared/ds_transpose.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <span>

namespace rocjitsu {
namespace amdgpu {
namespace {

/// Largest per-lane footprint of a plain DS access (b128).
constexpr uint32_t kMaxLaneBytes = 16;

/// Gather a store payload lane-major, with the register reads the generated
/// ds_write handlers make: one region for whole dwords, per-lane reads for a
/// region the wave's physical storage splits, and byte-masked per-lane reads
/// for narrow (and D16_HI) sources.
void gather_payload(const LdsAccessRequest &req, Wavefront &wf, uint64_t exec, uint32_t stride,
                    uint8_t *payload) {
  auto &cu = wf.cu();
  const uint32_t wave = wf.wf_size();
  if (req.elem_size >= sizeof(uint32_t)) {
    const uint32_t dwords = stride / sizeof(uint32_t);
    if (exec) {
      auto data = RegisterAccess(wf).read_vgpr_region(req.data_vgpr, dwords, exec);
      if (data.valid()) {
        data.copy_dwords_lane_major(std::span<uint8_t>(payload, wave * stride), exec);
        return;
      }
    }
    for (uint32_t lane = 0; lane < wave; ++lane) {
      if (!(exec & (1ULL << lane)))
        continue;
      for (uint32_t i = 0; i < dwords; ++i) {
        const uint32_t value = RegisterAccess(cu).read_vgpr(req.data_vgpr + i, lane);
        std::memcpy(&payload[lane * stride + i * sizeof(uint32_t)], &value, sizeof(uint32_t));
      }
    }
    return;
  }
  assert(req.num_elems == 1 && "narrow DS stores carry one element");
  const uint8_t byte_mask =
      static_cast<uint8_t>(((1u << req.elem_size) - 1u) << (req.d16_hi ? 2 : 0));
  for (uint32_t lane = 0; lane < wave; ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t value = RegisterAccess(cu).read_vgpr(req.data_vgpr, lane, byte_mask);
    if (req.d16_hi)
      value >>= 16;
    std::memcpy(&payload[lane * stride], &value, req.elem_size);
  }
}

/// Write a load's lane-major response to its destination VGPRs: the non-atomic
/// part of the routed path's vector_complete(). DS lanes are never dropped as
/// out of bounds, so there is no OOB zeroing.
void write_back(const LdsAccessRequest &req, Wavefront &wf, ComputeUnitCore &cu, uint64_t exec,
                uint32_t stride, const uint8_t *response) {
  const uint32_t vgpr_count = std::max(1u, (stride + 3u) / 4u);
  if (!cu.owns_vgpr_range(wf, req.dst_vgpr, vgpr_count))
    return;
  if (req.elem_size == sizeof(uint32_t) && !req.sign_extend && !req.d16_hi && !req.d16_lo) {
    for (uint32_t i = 0; i < vgpr_count; ++i) {
      auto *destination = reinterpret_cast<uint32_t *>(cu.raw_vgpr_data(req.dst_vgpr + i));
      for (uint64_t lanes = exec; lanes; lanes &= lanes - 1) {
        const uint32_t lane = std::countr_zero(lanes);
        std::memcpy(&destination[lane], &response[lane * stride + i * sizeof(uint32_t)],
                    sizeof(uint32_t));
      }
    }
    return;
  }
  const uint32_t copy_size = std::min<uint32_t>(req.elem_size, 4u);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    for (uint32_t i = 0; i < vgpr_count; ++i) {
      uint32_t value = 0;
      const uint8_t *source = &response[lane * stride + i * 4];
      if (copy_size == 4)
        std::memcpy(&value, source, 4);
      else if (copy_size == 2)
        std::memcpy(&value, source, 2);
      else
        std::memcpy(&value, source, copy_size);
      if (req.sign_extend && i == 0 && req.elem_size < 4) {
        if (req.elem_size == 1)
          value = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(value)));
        else
          value = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(value)));
      }
      if (copy_size <= 2 && (req.d16_hi || req.d16_lo)) {
        if (cu.sram_ecc()) {
          value = req.d16_hi ? value << 16 : value & 0xFFFFu;
        } else {
          const uint32_t old = cu.read_vgpr_storage(req.dst_vgpr + i, lane);
          value = req.d16_hi ? (old & 0xFFFFu) | (value << 16)
                             : (old & 0xFFFF0000u) | (value & 0xFFFFu);
        }
      }
      cu.write_vgpr(req.dst_vgpr + i, lane, value);
    }
  }
}

} // namespace

bool LdsInlineAccess::complete(const Instruction &inst, const LdsAccessRequest &req,
                               Wavefront &wf) {
  ComputeUnitCore &cu = wf.raw_cu();
  if (!cu.completes_lds_inline())
    return false;
  const uint32_t wave = wf.wf_size();
  const uint64_t exec = wf.exec();
  const uint32_t stride = req.num_elems * req.elem_size;
  assert(stride != 0 && stride <= kMaxLaneBytes);

  // Addresses, read as ds_calculate_addresses() reads them. Inactive lanes are
  // never read by the LDS model, so the array is not initialized.
  alignas(32) uint64_t addresses[64];
  {
    auto region = RegisterAccess(wf.cu()).read_vgpr_region(req.addr_vgpr, 1, exec);
    addr_calc::ds_lane_addresses(region.lanes(), exec, wave, req.offset + wf.lds_base(), addresses);
  }

  // Per-lane staging on the stack; only active lanes are written and read.
  alignas(64) uint8_t lanes[64 * kMaxLaneBytes];
  Lds &lds = wf.lds();
  if (req.is_load) {
    lds.vector_load(addresses, exec, req.elem_size, req.num_elems, lanes);
    write_back(req, wf, cu, exec, stride, lanes);
  } else {
    gather_payload(req, wf, exec, stride, lanes);
    lds.vector_store(addresses, exec, req.elem_size, req.num_elems, lanes);
  }
  cu.note_lds_inline_completion(inst);
  return true;
}

bool LdsInlineAccess::complete(const Instruction &inst, VectorMemState &d, Wavefront &wf) {
  ComputeUnitCore &cu = wf.raw_cu();
  if (!cu.completes_lds_inline() || d.lds_dst)
    return false;
  d.wf_size = wf.wf_size();
  LocalMemPipeline::initiate_lds_access(d, wf, transpose_request_lane_mask(d, d.wf_size));
  const MemoryAccessCompletion completion =
      LocalMemPipeline::complete_lds_access(d, wf, MemoryAccessDeferredCompletion{});
  assert(completion == MemoryAccessCompletion::Complete);
  (void)completion;
  cu.note_lds_inline_completion(inst);
  return true;
}

} // namespace amdgpu
} // namespace rocjitsu
