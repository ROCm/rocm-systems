// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/graphics/bvh_intersection.h"
#include "rocjitsu/vm/graphics/gfx12_bvh_node.h"

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
#include <tuple>

namespace rocjitsu::amdgpu {
namespace {

using Vector3 = std::array<float, 3>;
using BvhResultWords = std::array<uint32_t, 4>;
using Node = gfx12_bvh::Node;
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

bool bvh_child_is_triangle(uint32_t pointer, BvhForm form) {
  const unsigned type = pointer & 7;
  return type < (form == BvhForm::Bvh8 ? 2u : 4u);
}

bool culled_child(uint32_t flags, uint32_t pointer, uint32_t metadata, BvhForm form) {
  const unsigned force = flags & (ForceOpaque | ForceNonOpaque);
  const bool triangle = bvh_child_is_triangle(pointer, form);
  return ((metadata & 1) && force != ForceNonOpaque && (flags & CullOpaque)) ||
         ((metadata & 2) && force != ForceOpaque && (flags & CullNonOpaque)) ||
         (((metadata & 4) || (form != BvhForm::Bvh8 && triangle)) && (flags & SkipTriangles)) ||
         (((metadata & 8) || (pointer & 7) == 7) && (flags & SkipProcedural));
}

BvhResultWords box(const Node &node, const BvhIntersectionAccess &resource, const Ray &ray,
                   std::array<float, 4> *distances = nullptr) {
  const bool half = ray.node_type == Gfx11BvhNodeType::Box16;
  BvhResultWords children{};
  std::array<float, 4> keys{};
  for (unsigned child_index = 0; child_index < children.size(); ++child_index) {
    children[child_index] = kInvalidNode;
    if (node[child_index] == kInvalidNode ||
        culled_child(ray.flags, node[child_index], half ? 0 : node[28] >> (8 * child_index),
                     resource.form))
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
    const auto priority = [&](uint32_t child) {
      if (child == kInvalidNode)
        return std::pair{2u, 0u};
      if (resource.triangles_first && bvh_child_is_triangle(child, resource.form)) {
        const unsigned type = child & 7;
        return std::pair{0u, type == 1 ? 0u : 1u};
      }
      return std::pair{1u, 0u};
    };
    const auto key = [&](unsigned child) {
      return std::tuple{priority(children[child]), keys[child]};
    };
    const auto order = [&](unsigned first_child, unsigned second_child) {
      if (children[first_child] == kInvalidNode ||
          (children[second_child] != kInvalidNode && key(second_child) < key(first_child))) {
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
  if (distances)
    *distances = keys;
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
// Expand the quantized 128-byte IP 3.1 box into the two four-child groups used by the
// intersection unit. The child ranges advance independent internal and leaf pointers.
std::array<Node, 2> expand_bvh8(const Node &node, const Ray &ray) {
  std::array<Node, 2> groups{};
  for (auto &g : groups)
    std::fill_n(g.begin(), 4, kInvalidNode);
  uint32_t internal = node[0] | 5u, leaf = node[1];
  const unsigned count = ((node[6] >> 28) & 7) + 1;
  for (unsigned i = 0; i < count; ++i) {
    const auto *child = &node[8 + 3 * i];
    const unsigned type = (child[2] >> 24) & 15, range = child[2] >> 28;
    auto &next = type == 5 ? internal : leaf;
    const uint32_t pointer = next | type;
    next += range << 4;
    if (!((child[1] >> 24) & ray.instance_mask))
      continue;
    auto &group = groups[i / 4];
    const unsigned j = i % 4;
    group[j] = pointer;
    group[28] |= ((child[0] >> 24) & 15) << (8 * j);
    const unsigned planes[] = {child[0],       child[0] >> 12, child[1],
                               child[1] >> 12, child[2],       child[2] >> 12};
    for (unsigned plane = 0; plane < 6; ++plane) {
      const unsigned axis = plane % 3;
      const int exponent = ((node[6] >> (8 * axis)) & 255) - 139;
      const unsigned q = (planes[plane] & 4095) + (plane >= 3);
      group[4 + 6 * j + plane] = std::bit_cast<uint32_t>(std::bit_cast<float>(node[3 + axis]) +
                                                         std::ldexp(float(q), exponent));
    }
  }
  return groups;
}

// GFX12 returns packed normalized coordinates and IDs for compressed triangle pairs.
std::array<uint32_t, 10> intersect_primitive(const Node &node, const BvhIntersectionAccess &access,
                                             const Ray &input, unsigned type) {
  const gfx12_bvh::Primitive primitive{node, (type & 3) | ((type & 8) >> 1)};
  std::array<uint32_t, 10> result{};
  const unsigned descriptor = primitive.descriptor();
  for (unsigned slot = 0; slot < 2; ++slot) {
    const unsigned indices = (descriptor >> (slot ? 3 : 17)) & 0xfff;
    const bool procedural = (indices & 255) == 255;
    const bool opaque = (input.flags & ForceOpaque) ||
                        (!(input.flags & ForceNonOpaque) && (descriptor & (1u << (slot ? 2 : 16))));
    const unsigned prim_id = primitive.id(slot, false);
    auto *out = &result[4 * slot];
    if (procedural) {
      const bool culled = input.flags & (SkipProcedural | (opaque ? CullOpaque : CullNonOpaque));
      out[0] = culled ? std::bit_cast<uint32_t>(kInfinity) : 0;
      out[1] = std::bit_cast<uint32_t>(-1.f);
      out[2] = std::bit_cast<uint32_t>(opaque ? 1.f : -1.f);
      out[3] = prim_id << 1;
    } else {
      Node vertices{};
      for (unsigned vertex = 0; vertex < 3; ++vertex) {
        const Vector3 value = primitive.vertex((indices >> (4 * vertex)) & 15);
        for (unsigned axis = 0; axis < 3; ++axis)
          vertices[3 * vertex + axis] = std::bit_cast<uint32_t>(value[axis]);
      }
      // The compressed return tuple always selects weights for vertices 1 and 2.
      vertices[15] = 9 | (opaque ? 128u : 0u);
      Ray ray = input;
      ray.node_type = Gfx11BvhNodeType::Triangle0;
      if (descriptor & (1u << (slot ? 1 : 15)))
        ray.flags |= DoubleSided;
      auto projected = triangle(vertices, access, ray);
      const float det = std::bit_cast<float>(projected[1]);
      out[0] = det != 0 && !(ray.flags & SkipTriangles)
                   ? std::bit_cast<uint32_t>(std::bit_cast<float>(projected[0]) / det)
                   : 0x7f800000;
      const float u = std::bit_cast<float>(projected[2]) / det;
      const float v = std::bit_cast<float>(projected[3]) / det;
      out[1] = std::bit_cast<uint32_t>(access.pointer_flags ? std::abs(u) : u);
      out[2] = std::bit_cast<uint32_t>(
          access.pointer_flags ? std::copysign(std::abs(v), opaque ? 1.f : -1.f) : v);
      out[3] = (prim_id << 1) | (projected[1] >> 31);
    }
    result[8 + slot] = (primitive.id(slot, true) << 2) | primitive.navigation();
  }
  return result;
}

void complete_instance(ComputeUnitCore &cu, const VectorMemState &state, unsigned lane,
                       const Node &node, Ray ray) {
  const auto &access = *state.bvh;
  const Vector3 original_origin = ray.origin, original_direction = ray.direction;
  std::array<float, 12> matrix{};
  for (unsigned i = 0; i < matrix.size(); ++i)
    matrix[i] = std::bit_cast<float>(node[i]);
  ray.origin = gfx12_bvh::transform(matrix, original_origin, true);
  ray.direction = gfx12_bvh::transform(matrix, original_direction, false);
  for (unsigned axis = 0; axis < 3; ++axis) {
    ray.inverse[axis] = 1.f / ray.direction[axis];
    cu.write_vgpr(access.transform_registers[axis], lane,
                  std::bit_cast<uint32_t>(ray.origin[axis]));
    cu.write_vgpr(access.transform_registers[3 + axis], lane,
                  std::bit_cast<uint32_t>(ray.direction[axis]));
  }
  const uint64_t child = node[12] | (uint64_t{node[13]} << 32);
  const unsigned flags_mask = ray.flags & (ForceOpaque | ForceNonOpaque) ? 0x3fc : 0x3ff;
  const uint64_t next =
      (child & 0x3ffffffffffff8ull) | (uint64_t{((child >> 54) & flags_mask) | ray.flags} << 54);
  const bool visible = (ray.instance_mask & (node[15] >> 24)) &&
                       !(ray.flags & ((node[13] & (1u << 30)) ? SkipProcedural : SkipTriangles));
  uint32_t pointer = visible ? node[14] : kInvalidNode;
  if (access.compressed && visible) {
    Node embedded{};
    std::copy_n(node.begin() + 16, 4, embedded.begin() + 3);
    embedded[6] &= 0x3fffffff;
    std::copy_n(node.begin() + 20, 12, embedded.begin() + 8);
    ray.node_type = Gfx11BvhNodeType::Box32;
    const auto hits = box(expand_bvh8(embedded, ray)[0], access, ray);
    pointer = 0;
    for (unsigned i = 0; i < 4; ++i)
      pointer |= (hits[i] & 255) << (8 * i);
  }
  cu.write_vgpr(state.dst_reg_base + 2, lane, static_cast<uint32_t>(next));
  cu.write_vgpr(state.dst_reg_base + 3, lane, static_cast<uint32_t>(next >> 32));
  cu.write_vgpr(state.dst_reg_base + 6, lane, node[15] & 0xffffff);
  cu.write_vgpr(state.dst_reg_base + 7, lane, pointer);
}
void complete_extended_bvh(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &state,
                           unsigned lane) {
  const auto &access = *state.bvh;
  Ray ray = access.rays[lane];
  std::array<uint32_t, 10> result;
  result.fill(kInvalidNode);
  std::array<std::array<float, 4>, 2> distances{};
  bool box_group[2]{};
  for (unsigned slot = 0; slot < 2; ++slot) {
    if (!ray.blocks[slot] || discards_all_opacities(ray.flags))
      continue;
    Node node{};
    for (unsigned part = 0; part < ray.blocks[slot]; ++part) {
      const auto offset = ((2 * slot + part) * state.wf_size + lane) * 64;
      std::memcpy(node.data() + 16 * part, state.response_data.data() + offset, 64);
    }
    ray.node_type = gfx11_bvh_node_type(ray.types[slot]);
    if (ray.node_type == Gfx11BvhNodeType::Instance) {
      complete_instance(cu, state, lane, node, ray);
      return;
    }
    if (access.compressed && ray.node_type <= Gfx11BvhNodeType::Triangle3) {
      result = intersect_primitive(node, access, ray, ray.types[slot]);
      break;
    }
    if (ray.node_type == Gfx11BvhNodeType::Box16 && (access.hp64 || ray.hp64_pair)) {
      node = gfx12_bvh::expand_hp64(node, access.form == BvhForm::Bvh8);
      ray.node_type = Gfx11BvhNodeType::Box32;
    }
    if (access.compressed) {
      if ((node[7] & 127) < 104) {
        const auto matrix = gfx12_bvh::orientation(node[7] & 127);
        ray.origin = gfx12_bvh::transform(matrix, ray.origin, false);
        ray.direction = gfx12_bvh::transform(matrix, ray.direction, false);
        for (unsigned axis = 0; axis < 3; ++axis)
          ray.inverse[axis] = 1.f / ray.direction[axis];
      } else if ((node[7] & 127) != 127) {
        wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue,
                                              "reserved BVH8 oriented box matrix index");
        return;
      }
      const auto groups = expand_bvh8(node, ray);
      for (unsigned i = 0; i < groups.size(); ++i) {
        const auto hits = box(groups[i], access, ray, &distances[i]);
        std::copy(hits.begin(), hits.end(), result.begin() + i * 4);
        box_group[i] = true;
      }
    } else if (ray.node_type == Gfx11BvhNodeType::Box32 ||
               ray.node_type == Gfx11BvhNodeType::Box16) {
      const auto hits = box(node, access, ray, &distances[slot]);
      std::copy(hits.begin(), hits.end(), result.begin() + slot * 4);
      box_group[slot] = true;
    } else if (!(ray.flags & SkipTriangles)) {
      const auto hits = triangle(node, access, ray);
      std::copy(hits.begin(), hits.end(), result.begin() + slot * 4);
      result[8 + slot] = node[14];
    }
  }
  if (box_group[0] && box_group[1] && access.wide_sort && access.sort_enabled) {
    std::array<unsigned, 8> order{0, 1, 2, 3, 4, 5, 6, 7};
    const auto key = [&](unsigned i) {
      const auto child = result[i];
      const bool triangle = access.triangles_first && bvh_child_is_triangle(child, access.form);
      const unsigned type = child & 7;
      return std::tuple{child == kInvalidNode, !triangle, triangle && type != 1,
                        distances[i / 4][i % 4]};
    };
    std::stable_sort(order.begin(), order.end(),
                     [&](unsigned a, unsigned b) { return key(a) < key(b); });
    const auto original = result;
    for (unsigned i = 0; i < order.size(); ++i)
      result[i] = original[order[i]];
  }
  for (unsigned i = 0; i < result.size(); ++i)
    cu.write_vgpr(state.dst_reg_base + i, lane, result[i]);
}
} // namespace

util::Result prepare_bvh(Wavefront &wf, VectorMemState &state, uint32_t descriptor,
                         uint32_t destination, const ImageAddressRegisters &registers, BvhForm form,
                         bool a16, bool valid_encoding) {
  const bool extended = bvh_two_results(form);
  const bool wide = form == BvhForm::Bvh64;
  const bool gfx12 = wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4;
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

  access->form = form;
  access->compressed = gfx12 && (word[3] & (1u << 19));
  access->hp64 = gfx12 && (word[3] & (1u << 20));
  access->wide_sort = gfx12 && (word[3] & (1u << 21));
  access->triangles_first = gfx12 && (word[1] & (1u << 20));
  if (extended) {
    for (unsigned i = 0; i < 6; ++i)
      access->transform_registers[i] =
          wf.vgpr_alloc().base + gfx115_source_vgpr(registers[4 + i], wf.num_vgprs());
    state.graphics_sample = std::make_unique<GraphicsSampleAccess>();
    state.graphics_sample->tap_count = 4; // Two 64-byte blocks for each independent node.
    state.graphics_sample->taps.resize(4);
  }
  const unsigned extent = 1u + wide, origin = extent + 1, direction = origin + 3;
  state.wf_size = wf.wf_size();
  state.wg_id = wf.wg_id();
  state.wf_id = wf.wf_id();
  state.dst_reg_base = wf.vgpr_alloc().base + destination;
  state.exec_mask =
      RegisterAccess(wf).owns_vgpr_range(state.dst_reg_base, extended ? 10 : 4) ? wf.exec() : 0;
  if (access->compressed && (form != BvhForm::Bvh8 || !access->wide_sort ||
                             !access->triangles_first || !access->barycentrics))
    state.exec_mask = 0;
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
    if (extended) {
      const uint64_t base_pointer = read(0) | (uint64_t{read(1)} << 32);
      const uint64_t base_index = gfx11_bvh_node_index(base_pointer);
      ray.flags = access->pointer_flags ? uint16_t(base_pointer >> 54) : 0;
      ray.instance_mask = uint8_t(read(3));
      ray.extent = std::bit_cast<float>(read(2));
      for (unsigned axis = 0; axis < 3; ++axis) {
        ray.origin[axis] = std::bit_cast<float>(read(4 + axis));
        ray.direction[axis] = std::bit_cast<float>(read(7 + axis));
        ray.inverse[axis] = 1.f / ray.direction[axis];
      }
      const uint32_t first = read(10);
      for (unsigned slot = 0; slot < 2; ++slot) {
        if (slot && ray.blocks[0] && ray.types[0] == 6)
          break; // An instance owns both result slots.
        uint32_t pointer = slot == 0 ? first : form == BvhForm::Dual ? read(11) : kInvalidNode;
        if (form == BvhForm::Bvh8 && !access->compressed && (first & 7) == 2)
          pointer = (first & ~7u) + 16 * slot + 5;
        if (form == BvhForm::Bvh8 && !access->compressed && (first & 7) == 3) {
          ray.hp64_pair = true;
          pointer = (first & ~7u) + 8 * slot + 4;
        }
        if (form == BvhForm::Bvh8 && !access->compressed && (first & 7) == 1)
          pointer = first - 1 + slot;
        if (pointer == kInvalidNode)
          continue;
        const unsigned type = pointer & 7;
        const uint64_t index = base_index + ((access->compressed ? pointer & ~15u : pointer) >> 3);
        if (index > *last_node)
          continue;
        const bool instance = type == 6 && (word[3] & (1u << 22));
        if (type == 6 && (!instance || slot != 0))
          continue;
        const unsigned blocks = type == 5 || access->compressed ? 2 : 1;
        if (blocks - 1 > *last_node - index)
          continue;
        if (type == 7)
          continue;
        if (instance && !discards_all_opacities(ray.flags))
          access->instance_lanes |= uint64_t{1} << lane;
        ray.types[slot] = access->compressed ? pointer & 15 : type;
        ray.blocks[slot] = blocks;
        for (unsigned part = 0; part < blocks; ++part) {
          auto &tap = state.graphics_sample->taps[2 * slot + part];
          tap.addresses[lane] = addr_calc::buffer_virtual_address(base + (index + part) * 64);
          tap.lane_mask |= uint64_t{1} << lane;
        }
        state.lane_mask |= uint64_t{1} << lane;
      }
      continue;
    }
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
      if (gfx12 && a16) {
        const auto half = [&](unsigned component) {
          return util::f16_to_f32(
              uint16_t(read(direction + component / 2) >> (16 * (component % 2))));
        };
        ray.direction[ax] = half(ax);
        ray.inverse[ax] = half(3 + ax);
        continue;
      }
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
  if (!cu.owns_vgpr_range(wf, state.dst_reg_base, state.destination_vgpr_count()))
    return;
  const fp_mode::detail::ScopedFenv environment(0);
  for (uint64_t mask = state.exec_mask; mask; mask &= mask - 1) {
    const unsigned lane = std::countr_zero(mask);
    if (bvh_two_results(state.bvh->form)) {
      complete_extended_bvh(wf, cu, state, lane);
      continue;
    }
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
      if (state.bvh->hp64 && ray.node_type == Gfx11BvhNodeType::Box16) {
        Ray expanded = ray;
        expanded.node_type = Gfx11BvhNodeType::Box32;
        result = box(gfx12_bvh::expand_hp64(node, false), *state.bvh, expanded);
      } else {
        result = triangle_node ? triangle(node, *state.bvh, ray) : box(node, *state.bvh, ray);
      }
    }
    for (unsigned i = 0; i < result.size(); ++i)
      cu.write_vgpr(state.dst_reg_base + i, lane, result[i]);
  }
}

} // namespace rocjitsu::amdgpu
