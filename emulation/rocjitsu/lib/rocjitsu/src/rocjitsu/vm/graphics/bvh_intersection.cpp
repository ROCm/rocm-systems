// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/graphics/bvh_intersection.h"

#include "rocjitsu/isa/arch/amdgpu/shared/buffer_address.h"
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

using Vector3 = std::array<float, 3>;
using BvhResultWords = std::array<uint32_t, 4>;
using Node = std::array<uint32_t, 32>;
using Ray = BvhIntersectionAccess::Ray;
constexpr auto kInvalidNode = std::numeric_limits<uint32_t>::max();
constexpr auto kInfinity = std::numeric_limits<float>::infinity();

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

// On W7900, forcing and culling the same opacity discards every node type before intersection.
// Setting both force flags leaves opacity unchanged; both cull flags discard either opacity.
bool discards_all_opacities(uint32_t flags) {
  const unsigned force = flags & (ForceOpaque | ForceNonOpaque);
  const unsigned cull = flags & (CullOpaque | CullNonOpaque);
  return cull == (CullOpaque | CullNonOpaque) || (force == ForceOpaque && (cull & CullOpaque)) ||
         (force == ForceNonOpaque && (cull & CullNonOpaque));
}

bool culled_child(uint32_t flags, uint32_t pointer, uint32_t metadata) {
  const unsigned force = flags & (ForceOpaque | ForceNonOpaque);
  const auto type = gfx11_bvh_node_type(pointer);
  return ((metadata & 1) && force != ForceNonOpaque && (flags & CullOpaque)) ||
         ((metadata & 2) && force != ForceOpaque && (flags & CullNonOpaque)) ||
         (((metadata & 4) || type <= Gfx11BvhNodeType::Triangle3) && (flags & SkipTriangles)) ||
         (((metadata & 8) || type == Gfx11BvhNodeType::Procedural) && (flags & SkipProcedural));
}

BvhResultWords box(const Node &node, const BvhIntersectionAccess &resource, const Ray &ray) {
  const bool half = ray.node_type == Gfx11BvhNodeType::Box16;
  BvhResultWords children{};
  std::array<float, 4> keys{};
  for (unsigned child_index = 0; child_index < children.size(); ++child_index) {
    children[child_index] = kInvalidNode;
    if (node[child_index] == kInvalidNode ||
        culled_child(ray.flags, node[child_index], half ? 0 : node[28] >> (8 * child_index)))
      continue;
    // Bounding boxes store min.xyz, max.xyz starting at byte 16.
    const auto bound = [&](unsigned bound_component) {
      const unsigned bound_index = 6 * child_index + bound_component;
      return half
                 ? util::f16_to_f32(uint16_t(node[4 + bound_index / 2] >> ((bound_index & 1) * 16)))
                 : std::bit_cast<float>(node[4 + bound_index]);
    };
    float near = -kInfinity, far = kInfinity;
    for (unsigned axis = 0; axis < 3; ++axis) {
      const float lo = bound(axis), hi = bound(axis + 3);
      if (!(lo <= hi)) {
        near = kInfinity;
        far = -kInfinity;
        break;
      }
      if (std::isinf(ray.inverse[axis])) {
        if (!(lo <= ray.origin[axis] && ray.origin[axis] <= hi)) {
          near = kInfinity;
          far = -kInfinity;
          break;
        }
        continue;
      }
      const float lower_bound_distance = (lo - ray.origin[axis]) * ray.inverse[axis];
      const float upper_bound_distance = (hi - ray.origin[axis]) * ray.inverse[axis];
      near = std::max(near, std::min(lower_bound_distance, upper_bound_distance));
      far = std::min(far, std::max(lower_bound_distance, upper_bound_distance));
    }
    const float first = std::max(0.0f, near), last = std::min(ray.extent, far);
    if (first <= last * (1.0f + resource.grow * 0x1p-24f)) {
      children[child_index] = node[child_index];
      keys[child_index] = resource.sort == 1   ? first - last
                          : resource.sort == 2 ? near + far
                                               : first;
    }
  }
  if (resource.sort_enabled) {
    // GPURT's four-input sorting network defines tie order, including missed children.
    const auto order = [&](unsigned first_child, unsigned second_child) {
      if (children[first_child] == kInvalidNode ||
          (children[second_child] != kInvalidNode && keys[second_child] < keys[first_child])) {
        std::swap(children[first_child], children[second_child]);
        std::swap(keys[first_child], keys[second_child]);
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

BvhResultWords triangle(const Node &node, const BvhIntersectionAccess &resource, const Ray &ray) {
  // RRA TriangleNode::GetTriangle maps each tag onto three of the five vertices.
  static constexpr unsigned vertices[][3] = {{0, 1, 2}, {1, 3, 2}, {2, 3, 4}, {2, 4, 0}};
  const auto vertex = [&](unsigned vertex_index) {
    const auto offset = 3 * vertices[static_cast<unsigned>(ray.node_type)][vertex_index];
    return Vector3{std::bit_cast<float>(node[offset]), std::bit_cast<float>(node[offset + 1]),
                   std::bit_cast<float>(node[offset + 2])};
  };
  const Vector3 vertex0 = vertex(0), vertex1 = vertex(1), vertex2 = vertex(2);
  // W7900 breaks equal-magnitude ties in X, Y, Z order.
  unsigned dominant_axis = 0;
  for (unsigned axis : {1u, 2u})
    if (std::abs(ray.direction[axis]) > std::abs(ray.direction[dominant_axis]))
      dominant_axis = axis;
  // W7900 returns the three projected edge functions in unnormalized ray space.
  // Each subtraction, product and ordered sum rounds toward zero at FP32 precision.
  std::array<float, 4> bary{};
  float numerator, denominator;
  std::array<Vector3, 3> projected;
  {
    const fp_mode::detail::ScopedFenv toward_zero(3);
    const unsigned first_axis = (dominant_axis + 1) % 3, second_axis = (dominant_axis + 2) % 3;
    const auto project = [&](const Vector3 &point) -> Vector3 {
      const float depth = point[dominant_axis] - ray.origin[dominant_axis];
      return {(point[first_axis] - ray.origin[first_axis]) * ray.direction[dominant_axis] -
                  depth * ray.direction[first_axis],
              (point[second_axis] - ray.origin[second_axis]) * ray.direction[dominant_axis] -
                  depth * ray.direction[second_axis],
              depth};
    };
    projected = {project(vertex0), project(vertex1), project(vertex2)};
    std::array<float, 3> distances;
    for (unsigned vertex_index = 0; vertex_index < 3; ++vertex_index) {
      const auto &first = projected[(vertex_index + 2) % 3];
      const auto &second = projected[(vertex_index + 1) % 3];
      const float edge = first[0] * second[1] - first[1] * second[0];
      bary[vertex_index] = edge * ray.direction[dominant_axis];
      distances[vertex_index] = edge * projected[vertex_index][2];
    }
    numerator = (distances[0] + distances[1]) + distances[2];
    denominator = (bary[0] + bary[1]) + bary[2];
  }
  // W7900 uses FP32 edge signs for coverage. When an edge rounds to zero,
  // the top-left rule uses the same projected endpoints. For a horizontal
  // edge the dominant ray direction determines the orientation of X.
  const auto covers_edge = [&](unsigned edge_index) {
    const float edge = bary[edge_index];
    if (edge != 0)
      return denominator < 0 ? edge < 0 : edge > 0;
    const auto &first = projected[(edge_index + 1) % 3];
    const auto &second = projected[(edge_index + 2) % 3];
    if (first[1] != second[1])
      return denominator < 0 ? second[1] < first[1] : second[1] > first[1];
    const bool decreasing_x = (denominator < 0) != (ray.direction[dominant_axis] < 0);
    return decreasing_x ? second[0] < first[0] : second[0] > first[0];
  };
  const bool intersect = denominator != 0 && numerator / denominator >= 0 && covers_edge(0) &&
                         covers_edge(1) && covers_edge(2);
  const unsigned triangle_index = static_cast<unsigned>(ray.node_type);
  const unsigned control = (node[15] >> (8 * triangle_index)) & 255;
  const unsigned force = ray.flags & (ForceOpaque | ForceNonOpaque);
  const bool opaque = force == ForceOpaque || ((control & 128) && force != ForceNonOpaque);
  // In pointer-flag mode ID bits 4 and 5 disable face culling and reverse winding, respectively.
  const bool double_sided =
      (ray.flags & DoubleSided) || (resource.pointer_flags && (control & (1u << 4)));
  const bool triangle_flip = resource.pointer_flags && (control & (1u << 5));
  const bool flip = bool(ray.flags & Flip) != triangle_flip;
  const auto orient = [&](float value) {
    // The intersection unit returns positive geometric zeros before applying FLIP.
    value = value == 0 ? 0.f : value;
    return flip ? -value : value;
  };
  for (unsigned vertex_index = 0; vertex_index < 3; ++vertex_index)
    bary[vertex_index] = orient(bary[vertex_index]);
  numerator = orient(numerator);
  denominator = orient(denominator);
  const bool cull_face = !double_sided && (ray.flags & (denominator >= 0 ? CullFront : CullBack));
  const bool culled = cull_face || (ray.flags & (opaque ? CullOpaque : CullNonOpaque));
  const bool hit = intersect && !culled;
  if (resource.barycentrics) {
    if (!hit)
      numerator = std::copysign(kInfinity, denominator);
    // W7900 aliases selector 3 to the oriented weight of vertex 0.
    bary[3] = bary[0];
    return {std::bit_cast<uint32_t>(numerator), std::bit_cast<uint32_t>(denominator),
            std::bit_cast<uint32_t>(bary[control & 3]),
            std::bit_cast<uint32_t>(bary[(control >> 2) & 3])};
  }
  // With pointer flags disabled the node holds a base primitive ID shared by its triangles.
  const uint32_t id = resource.pointer_flags ? control : node[15] + triangle_index;
  if (!hit)
    return {std::bit_cast<uint32_t>(kInfinity), std::bit_cast<uint32_t>(1.0f), id, 0};
  return {std::bit_cast<uint32_t>(numerator), std::bit_cast<uint32_t>(denominator), id, 1};
}
} // namespace

util::Result prepare_gfx11_bvh_intersection(Wavefront &wf, VectorMemState &state,
                                            uint32_t descriptor, uint32_t destination,
                                            uint32_t address, const uint32_t *nsa, bool wide,
                                            bool a16, bool valid_encoding) {
  const auto fail = [&](const char *detail) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue,
                                          detail);
    return util::Result::failure();
  };
  if (!valid_encoding || !scalar_selector_range_is_backed(wf, descriptor, 4))
    return fail("invalid image_bvh encoding or resource tuple");
  uint32_t word[4];
  for (unsigned i = 0; i < 4; ++i)
    word[i] = read_scalar_selector(wf, descriptor + i);
  const auto last_node = gfx11_bvh_last_node(word[2], word[3]);
  if (!last_node)
    return fail("invalid BVH resource type");
  const uint64_t base = (word[0] | (uint64_t(word[1] & 255) << 32)) << 8;
  auto access = std::make_unique<BvhIntersectionAccess>();
  access->pointer_flags = gfx11_bvh_pointer_flags_enabled(word[3]);
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
    ray.node_type = gfx11_bvh_node_type(pointer);
    ray.flags = access->pointer_flags ? uint16_t(pointer >> 54) : 0;
    if (!bvh_node_can_load(pointer, *last_node, access->pointer_flags))
      continue;
    const uint64_t index = gfx11_bvh_node_index(pointer);
    state.per_lane_addr[lane] = addr_calc::buffer_virtual_address(base + index * 64);
    state.lane_mask |= uint64_t{1} << lane;
    if (ray.node_type == Gfx11BvhNodeType::Box32)
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
  return util::Result::success();
}

void complete_bvh_intersection(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &state) {
  if (!cu.owns_vgpr_range(wf, state.dst_reg_base, 4))
    return;
  const fp_mode::detail::ScopedFenv environment(0);
  for (uint64_t mask = state.exec_mask; mask; mask &= mask - 1) {
    const unsigned lane = std::countr_zero(mask);
    BvhResultWords result{kInvalidNode, kInvalidNode, kInvalidNode, kInvalidNode};
    const auto &ray = state.bvh->rays[lane];
    const bool triangle_node = ray.node_type <= Gfx11BvhNodeType::Triangle3;
    // SkipTriangles returns four invalid words in either result mode, unlike a geometric miss.
    if (((state.lane_mask >> lane) & 1) && !discards_all_opacities(ray.flags) &&
        !(triangle_node && (ray.flags & SkipTriangles))) {
      Node node{};
      const auto stride = state.elem_size * state.num_elems;
      std::memcpy(node.data(), state.response_data.data() + lane * stride,
                  ray.node_type == Gfx11BvhNodeType::Box32 ? 128 : 64);
      result = triangle_node ? triangle(node, *state.bvh, ray) : box(node, *state.bvh, ray);
    }
    for (unsigned i = 0; i < result.size(); ++i)
      cu.write_vgpr(state.dst_reg_base + i, lane, result[i]);
  }
}

} // namespace rocjitsu::amdgpu
