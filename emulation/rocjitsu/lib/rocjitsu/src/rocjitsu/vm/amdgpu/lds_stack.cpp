// Copyright (c) 2020-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocjitsu/vm/amdgpu/lds_stack.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <algorithm>
#include <array>
#include <cstring>

namespace rocjitsu::amdgpu {
namespace {
// Stack slots are 32 DWORDs apart in both wave32 and wave64; callers choose each lane's base.
constexpr uint32_t stack_slot_bytes = 128;
constexpr uint32_t invalid = 0xffffffffu, terminal = 0xfffffffeu;
bool node_valid(uint32_t node) { return (node >> 3) != 0x1fffffffu; }
// RDNA4 control state follows GPURT rtip3_ds_stack_emulation.hlsl, qualified against ISA probes.
// https://github.com/GPUOpen-Drivers/gpurt/blob/7b226d48b46b7e92fec3b9ecc5712e5bf2bf3dd9/src/shaders/rtip3_ds_stack_emulation.hlsl
struct Stack {
  Lds &lds;
  uint32_t allocation, size, base, ring, valid, to_parent;
  bool parent, overflow, transition;
  void push(uint32_t node) {
    overflow |= valid == size;
    lds.write32(allocation + base + ring * stack_slot_bytes, node);
    ring = ring == size - 1 ? 0 : (ring + 1) & 31;
    if (to_parent == size)
      parent = false;
    to_parent = std::min(to_parent + 1, size);
    valid = std::min(valid + 1, size);
  }
  uint32_t pop(bool second = false) {
    if (second && (!valid || (parent && to_parent == 0)))
      return invalid;
    if (!second) {
      transition = parent && to_parent == 0;
      if (transition)
        parent = false;
      if (!valid) {
        transition = parent = false;
        return overflow ? invalid : terminal;
      }
    }
    const uint32_t next = ring ? ring - 1 : size - 1;
    const uint32_t node = lds.read32(allocation + base + next * stack_slot_bytes);
    if (second && (node & 7) == 6)
      return invalid;
    ring = next;
    to_parent -= to_parent != 0;
    --valid;
    if ((node & 7) == 6) {
      if (parent) {
        valid = to_parent;
        overflow = true;
      }
      parent = true;
      to_parent = 0;
    }
    return node;
  }
  uint32_t pack() const {
    return (uint32_t{transition} << 31) | (uint32_t{overflow} << 30) | (uint32_t{parent} << 29) |
           ((base / 4) << 15) | ((ring & 31) << 10) | ((to_parent & 31) << 5) | (valid & 31);
  }
};
} // namespace

void prepare_lds_stack(Wavefront &wf, VectorMemState &d, uint32_t addr, uint32_t last,
                       uint32_t nodes, uint32_t dst, uint32_t push, uint32_t pop, uint32_t size,
                       uint32_t flags) {
  RegisterAccess regs(wf);
  const uint32_t base = wf.vgpr_alloc().base;
  d.lds_stack_inputs = push;
  d.lds_stack_size = size;
  d.lds_stack_flags = flags;
  d.wf_size = wf.wf_size();
  d.exec_mask = d.lane_mask = wf.exec();
  d.elem_size = 4;
  d.num_elems = pop;
  d.is_load = true;
  d.dst_reg_base = base + dst;
  d.ds2_active = true;
  d.ds2_dst_reg_base = base + addr;
  if (!regs.owns_vgpr_range(base + addr, 1) || !regs.owns_vgpr_range(base + dst, pop) ||
      !regs.owns_vgpr_range(base + last, 1) || !regs.owns_vgpr_range(base + nodes, push)) {
    d.exec_mask = d.lane_mask = 0;
    return;
  }
  auto pointers = regs.read_vgpr_region(base + addr, 1, wf.exec());
  auto visited = regs.read_vgpr_region(base + last, 1, wf.exec());
  auto children = regs.read_vgpr_region(base + nodes, push, wf.exec());
  d.store_data.resize(wf.wf_size() * (push + 2) * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(wf.exec() & (uint64_t{1} << lane)))
      continue;
    std::array<uint32_t, 10> data{};
    data[0] = pointers.lane(0, lane);
    data[1] = visited.lane(0, lane);
    for (uint32_t i = 0; i < push; ++i)
      data[i + 2] = children.lane(i, lane);
    std::memcpy(d.store_data.data() + lane * (push + 2) * 4, data.data(), (push + 2) * 4);
    d.per_lane_addr[lane] =
        wf.lds_base() + (flags & 1 ? ((data[0] >> 15) & 0x3fff) * 4 : (data[0] >> 18) * 4);
  }
}

void execute_lds_stack(Wavefront &wf, VectorMemState &d) {
  d.response_data.resize(d.wf_size * d.num_elems * 4);
  d.ds2_response_data.resize(d.wf_size * 4);
  const bool modern = d.lds_stack_flags & 1;
  const bool pairs = d.lds_stack_flags & 2;
  const bool primitive_range = d.lds_stack_flags & 4;
  const uint32_t count = d.lds_stack_inputs, size = d.lds_stack_size;
  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(d.lane_mask & (uint64_t{1} << lane)))
      continue;
    std::array<uint32_t, 10> data{};
    std::memcpy(data.data(), d.store_data.data() + lane * (count + 2) * 4, (count + 2) * 4);
    uint32_t ptr = data[0];
    const uint32_t last = data[1];
    // RDNA3 reserves only the all-ones pointer; RDNA4 has eight traversal-control values.
    const bool walkback = modern ? node_valid(last) : last != invalid;
    bool found = !walkback;
    for (uint32_t i = 0; i < count; ++i)
      found |= data[i + 2] == last || (primitive_range && (data[i + 2] & 7) <= 3);
    std::array<uint32_t, 2> result{invalid, invalid};
    if (!modern) {
      uint32_t index = ptr & 0xffff;
      // RDNA3 returns the first eligible child directly, leaving all ring slots available.
      uint32_t first = count;
      if (found) {
        uint32_t after = 0;
        if (walkback)
          for (uint32_t i = 0; i < count; ++i)
            if (data[i + 2] == last)
              after = i + 1;
        for (uint32_t i = after; i < count; ++i)
          if (data[i + 2] != invalid) {
            first = i;
            break;
          }
      }
      if (first != count) {
        for (uint32_t i = count; i-- > first + 1;)
          if (data[i + 2] != invalid) {
            wf.lds().write32(static_cast<uint32_t>(d.per_lane_addr[lane]) +
                                 (index % size) * stack_slot_bytes,
                             data[i + 2]);
            index = (index + 1) & 0xffff;
          }
        result[0] = data[first + 2];
      } else if (index) {
        --index;
        const uint32_t address =
            static_cast<uint32_t>(d.per_lane_addr[lane]) + (index % size) * stack_slot_bytes;
        result[0] = wf.lds().read32(address);
        wf.lds().write32(address, invalid);
      } else
        result[0] = terminal;
      ptr = (ptr & 0xffff0000u) | index;
    } else {
      Stack s{wf.lds(),
              wf.lds_base(),
              size,
              static_cast<uint32_t>(d.per_lane_addr[lane]) - wf.lds_base(),
              (ptr >> 10) & 31,
              ptr & 31,
              (ptr >> 5) & 31,
              bool(ptr & (1u << 29)),
              bool(ptr & (1u << 30)),
              false};
      const bool update = primitive_range && !walkback && !(last & 4);
      if (primitive_range)
        found = (found || (last & 7) <= 3) && !(update && (last & 2));
      bool skip_low = false, skip_high = false;
      for (uint32_t i = count; i-- > 0;) {
        uint32_t node = data[i + 2];
        bool &skip = i >= 4 ? skip_high : skip_low;
        if (!node_valid(node) && found && !skip) {
          skip_low |= !(node & 2);
          if (i >= 4)
            skip_high |= !(node & 4);
        }
        if (update && count == 8 && i != count - 1)
          skip_low = skip_high = true;
        if (walkback && (node == last || (primitive_range && (node & 7) <= 3)))
          found = false;
        if (!found || !node_valid(node) || skip)
          continue;
        if (update && count == 8)
          node = last & 1 ? (node & ~15u) + 16 : (node & 15) == 3 ? node ^ 11 : node + 1;
        s.push(node);
      }
      result[0] = s.pop();
      if (d.num_elems == 2) {
        result[1] = pairs && (result[0] & 7) == 1 ? result[0] & ~7u : s.pop(true);
      }
      ptr = s.pack();
    }
    std::memcpy(d.response_data.data() + lane * d.num_elems * 4, result.data(), d.num_elems * 4);
    std::memcpy(d.ds2_response_data.data() + lane * 4, &ptr, 4);
  }
}
} // namespace rocjitsu::amdgpu
