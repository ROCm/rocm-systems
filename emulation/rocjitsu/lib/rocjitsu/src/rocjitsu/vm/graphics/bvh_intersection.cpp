// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/graphics/bvh_intersection.h"

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "util/data_types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace rocjitsu::amdgpu {
namespace {

using V3 = std::array<float, 3>;
using Result = std::array<uint32_t, 4>;
using Node = std::array<uint32_t, 32>;
using Ray = BvhIntersectionAccess::Ray;
constexpr auto invalid = std::numeric_limits<uint32_t>::max();
constexpr auto inf = std::numeric_limits<float>::infinity();

// RTIP2 pointer flags occupy bits 54..63 of a tagged 64-bit node pointer.
enum RayFlag {
  ForceOpaque = 1,
  ForceNonOpaque = 2,
  DoubleSided = 4,
  Flip = 8,
  CullBack = 16,
  CullFront = 32,
  CullOpaque = 64,
  CullNonOpaque = 128,
  SkipTriangles = 256,
  SkipProcedural = 512
};

bool culled_box(uint32_t flags, uint32_t child) {
  const bool opaque = ((child & 1) || (flags & ForceOpaque)) && !(flags & ForceNonOpaque);
  const bool nonopaque = ((child & 2) || (flags & ForceNonOpaque)) && !(flags & ForceOpaque);
  return (opaque && (flags & CullOpaque)) || (nonopaque && (flags & CullNonOpaque)) ||
         ((child & 4) && (flags & SkipTriangles)) || ((child & 8) && (flags & SkipProcedural));
}

Result box(const Node &node, const BvhIntersectionAccess &resource, const Ray &ray) {
  const bool half = ray.node_type == 4;
  Result children{};
  std::array<float, 4> keys{};
  for (unsigned i = 0; i < children.size(); ++i) {
    children[i] = invalid;
    if (node[i] == invalid || culled_box(ray.flags, half ? 0 : node[28] >> (8 * i)))
      continue;
    // Bounding boxes store min.xyz, max.xyz starting at byte 16.
    const auto bound = [&](unsigned j) {
      const unsigned k = 6 * i + j;
      return half ? util::f16_to_f32(uint16_t(node[4 + k / 2] >> ((k & 1) * 16)))
                  : std::bit_cast<float>(node[4 + k]);
    };
    float near = -inf, far = inf;
    for (unsigned axis = 0; axis < 3; ++axis) {
      const float lo = bound(axis), hi = bound(axis + 3);
      if (!(lo <= hi)) {
        near = inf;
        far = -inf;
        break;
      }
      if (std::isinf(ray.inverse[axis])) {
        if (!(lo <= ray.origin[axis] && ray.origin[axis] <= hi)) {
          near = inf;
          far = -inf;
          break;
        }
        continue;
      }
      const float a = (lo - ray.origin[axis]) * ray.inverse[axis];
      const float b = (hi - ray.origin[axis]) * ray.inverse[axis];
      near = std::max(near, std::min(a, b));
      far = std::min(far, std::max(a, b));
    }
    const float first = std::max(0.0f, near), last = std::min(ray.extent, far);
    if (first <= last * (1.0f + resource.grow * 0x1p-24f)) {
      children[i] = node[i];
      keys[i] = resource.sort == 1 ? first - last : resource.sort == 2 ? near + far : first;
    }
  }
  if (resource.sort_enabled) {
    // GPURT's four-input sorting network defines tie order, including missed children.
    const auto order = [&](unsigned a, unsigned b) {
      if (children[a] == invalid || (children[b] != invalid && keys[b] < keys[a])) {
        std::swap(children[a], children[b]);
        std::swap(keys[a], keys[b]);
      }
    };
    order(0, 2);
    order(1, 3);
    order(0, 1);
    order(2, 3);
    order(1, 2);
  }
  return children;
}

V3 subtract(const V3 &a, const V3 &b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
V3 cross(const V3 &a, const V3 &b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
float dot(const V3 &a, const V3 &b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// In ray space, a shared edge belongs to the triangle on its top or left side.
// Keep the projected coordinates scaled by the dominant ray component: division
// would round an exact edge crossing to opposite sides of the ray.
bool covers_triangle(const V3 &a, const V3 &b, const V3 &c, const Ray &ray, unsigned z, bool ccw) {
  const unsigned x = (z + 1) % 3, y = (z + 2) % 3;
  const double dominant = ray.direction[z];
  using Point = std::array<double, 2>;
  const auto project = [&](const V3 &vertex) -> Point {
    const double depth = double(vertex[z]) - ray.origin[z];
    const double u = (double(vertex[x]) - ray.origin[x]) * dominant - depth * ray.direction[x];
    const double v = (double(vertex[y]) - ray.origin[y]) * dominant - depth * ray.direction[y];
    return {dominant < 0 ? -u : u, v};
  };
  const auto inside = [&](const Point &first, const Point &second) {
    // Compare the rounded products directly. A fused multiply-subtract would
    // round only one product, breaking symmetry when an edge is reversed.
    const double lhs = first[0] * second[1], rhs = first[1] * second[0];
    if (lhs != rhs)
      return ccw ? lhs > rhs : lhs < rhs;
    const double direction = second[1] == first[1] ? second[0] - first[0] : second[1] - first[1];
    return ccw ? direction < 0 : direction > 0;
  };
  const Point p = project(a), q = project(b), r = project(c);
  return inside(p, q) && inside(q, r) && inside(r, p);
}

Result triangle(const Node &node, bool barycentrics, const Ray &ray) {
  if (ray.flags & SkipTriangles)
    return {invalid, invalid, invalid, invalid};
  // RRA TriangleNode::GetTriangle maps each tag onto three of the five vertices.
  static constexpr unsigned vertices[][3] = {{0, 1, 2}, {1, 3, 2}, {2, 3, 4}, {2, 4, 0}};
  const auto vertex = [&](unsigned i) {
    const auto k = 3 * vertices[ray.node_type][i];
    return V3{std::bit_cast<float>(node[k]), std::bit_cast<float>(node[k + 1]),
              std::bit_cast<float>(node[k + 2])};
  };
  const V3 a = vertex(0), b = vertex(1), c = vertex(2);
  unsigned z = 0;
  for (unsigned axis : {1u, 2u})
    if (std::abs(ray.direction[axis]) > std::abs(ray.direction[z]))
      z = axis;
  const V3 e1 = subtract(b, a), e2 = subtract(c, a);
  const V3 relative = subtract(ray.origin, a);
  const V3 p = cross(ray.direction, e2), q = cross(relative, e1);
  float denominator = dot(p, e1), numerator = dot(e2, q);
  const float i = dot(relative, p), j = dot(ray.direction, q);
  const bool intersect = denominator != 0 && numerator / denominator >= 0 &&
                         covers_triangle(a, b, c, ray, z, denominator < 0);
  const unsigned id = node[15], control = (id >> (8 * ray.node_type)) & 255;
  const bool opaque =
      (ray.flags & ForceOpaque) || ((control & 128) && !(ray.flags & ForceNonOpaque));
  const bool flip = ray.flags & Flip;
  // The returned numerators and denominator use the same unnormalized ray space.
  const double scale = double(ray.direction[z]) * ray.direction[z];
  const auto orient = [&](float value) {
    // The intersection unit returns positive geometric zeros before applying FLIP.
    const float scaled = value == 0 ? 0.f : float(value * scale);
    return flip ? -scaled : scaled;
  };
  const std::array<float, 4> bary{orient(denominator - i - j), orient(i), orient(j), 0};
  numerator = orient(numerator);
  denominator = orient(denominator);
  const bool cull_face =
      !(ray.flags & DoubleSided) && (ray.flags & (denominator >= 0 ? CullFront : CullBack));
  const bool culled = cull_face || (ray.flags & (opaque ? CullOpaque : CullNonOpaque));
  const bool hit = intersect && !culled;
  if (barycentrics) {
    if (!hit)
      numerator = std::copysign(inf, denominator);
    return {std::bit_cast<uint32_t>(numerator), std::bit_cast<uint32_t>(denominator),
            std::bit_cast<uint32_t>(bary[control & 3]),
            std::bit_cast<uint32_t>(bary[(control >> 2) & 3])};
  }
  if (!hit)
    return {std::bit_cast<uint32_t>(inf), std::bit_cast<uint32_t>(1.0f), id, 0};
  return {std::bit_cast<uint32_t>(numerator), std::bit_cast<uint32_t>(denominator), id, 1};
}
} // namespace

bool prepare_gfx115_bvh(Wavefront &wf, VectorMemState &state, uint32_t descriptor,
                        uint32_t destination, uint32_t address, const uint32_t *nsa, bool wide,
                        bool a16, bool valid_encoding) {
  const auto fail = [&](const char *detail) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue,
                                          detail);
    return false;
  };
  if (!valid_encoding || !scalar_selector_range_is_backed(wf, descriptor, 4))
    return fail("invalid image_bvh encoding or resource tuple");
  uint32_t word[4];
  for (unsigned i = 0; i < 4; ++i)
    word[i] = read_scalar_selector(wf, descriptor + i);
  if ((word[3] >> 28) != 8)
    return fail("invalid BVH resource type");
  const uint64_t base = (word[0] | (uint64_t(word[1] & 255) << 32)) << 8;
  const uint64_t last_node = word[2] | (uint64_t(word[3] & 0x3ff) << 32);
  const bool pointer_flags = word[3] & (1u << 23);
  auto access = std::make_unique<BvhIntersectionAccess>();
  access->grow = uint8_t(word[1] >> 23);
  access->sort = (word[1] >> 21) & 3;
  access->sort_enabled = word[1] >> 31;
  access->barycentrics = word[3] & (1u << 24);

  const auto registers = gfx11_bvh_registers(wide, a16, address, nsa);
  const unsigned extent = 1u + wide, origin = extent + 1, direction = origin + 3;
  state.wf_size = wf.wf_size();
  state.wg_id = wf.wg_id();
  state.wf_id = wf.wf_id();
  state.dst_reg_base = wf.vgpr_alloc().base + destination;
  state.exec_mask = RegisterAccess(wf).owns_vgpr_range(state.dst_reg_base, 4) ? wf.exec() : 0;
  // A box32 spans two 64-byte nodes; the other defined fetches span one.
  state.elem_size = 64;
  uint64_t box32 = 0;
  RegisterAccess regs(wf);
  for (uint64_t mask = state.exec_mask; mask; mask &= mask - 1) {
    const unsigned lane = std::countr_zero(mask);
    auto &ray = access->rays[lane];
    const auto read = [&](unsigned word_index) {
      const unsigned reg = gfx115_source_vgpr(registers[word_index], wf.num_vgprs());
      return regs.read_vgpr(wf.vgpr_alloc().base + reg, lane);
    };
    uint64_t pointer = read(0);
    if (wide)
      pointer |= uint64_t(read(1)) << 32;
    ray.node_type = pointer & 7;
    ray.flags = pointer_flags ? uint16_t(pointer >> 54) : 0;
    if (!bvh_node_can_load(pointer, last_node))
      continue;
    const uint64_t index = (pointer >> 3) & ((uint64_t{1} << 42) - 1);
    state.per_lane_addr[lane] = (base + index * 64) & 0xffffffffffffULL;
    state.lane_mask |= uint64_t{1} << lane;
    if (ray.node_type == 5)
      box32 |= uint64_t{1} << lane;
    ray.extent = std::bit_cast<float>(read(extent));
    for (unsigned ax = 0; ax < 3; ++ax) {
      ray.origin[ax] = std::bit_cast<float>(read(origin + ax));
      const auto dir = read(direction + ax);
      ray.direction[ax] = a16 ? util::f16_to_f32(uint16_t(dir)) : std::bit_cast<float>(dir);
      ray.inverse[ax] = a16 ? util::f16_to_f32(uint16_t(dir >> 16))
                            : std::bit_cast<float>(read(direction + 3 + ax));
    }
  }
  state.num_elems = box32 ? 2 : 1;
  if (box32 && box32 != state.lane_mask) {
    state.element_lane_masks.assign(2, state.lane_mask);
    state.element_lane_masks[1] = box32;
  }
  state.bvh = std::move(access);
  return true;
}

void complete_bvh_intersection(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &state) {
  if (!cu.owns_vgpr_range(wf, state.dst_reg_base, 4))
    return;
  const fp_mode::detail::ScopedFenv environment(0);
  for (uint64_t mask = state.exec_mask; mask; mask &= mask - 1) {
    const unsigned lane = std::countr_zero(mask);
    Result result{invalid, invalid, invalid, invalid};
    const auto &ray = state.bvh->rays[lane];
    if ((state.lane_mask >> lane) & 1) {
      Node node{};
      const auto stride = state.elem_size * state.num_elems;
      std::memcpy(node.data(), state.response_data.data() + lane * stride,
                  ray.node_type == 5 ? 128 : 64);
      result = ray.node_type < 4 ? triangle(node, state.bvh->barycentrics, ray)
                                 : box(node, *state.bvh, ray);
    }
    for (unsigned i = 0; i < result.size(); ++i)
      cu.write_vgpr(state.dst_reg_base + i, lane, result[i]);
  }
}

} // namespace rocjitsu::amdgpu
