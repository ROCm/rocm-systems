// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited gfx11 box/triangle arithmetic, instantiated in isolated ISA domains.
// Fetching, address validation and register publication belong to rocjitsu.
#pragma once
#include "common/bits.h"
#include "common/compiler.h"
#include "common/float_encoding.h"
#include "graphics/rounding.h"
#include "rj_arith/detail/rj_arith_graphics.h"
#if defined(RJ_ARITH_BVH_SIMD)
#include "graphics/x86_64/bvh_box.h"
#include "graphics/x86_64/bvh_rounding.h"
#endif
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::bvh {
enum BvhNodeType { Triangle0, Triangle1, Triangle2, Triangle3, Box16, Box32, Instance, Procedural };
// Exchange values without standard-library symbols shared by different ISA objects.
template <typename T> inline void swap(T &a, T &b) {
  T copy = a;
  a = b;
  b = copy;
}
// Fixed-size geometry storage whose methods are local to this CPU compilation domain.
template <typename T, int Count> struct Array {
  T values[Count];
  // Access a component; i is within the fixed extent.
  constexpr T &operator[](int i) { return values[i]; }
  // Access a read-only component within the extent.
  constexpr const T &operator[](int i) const { return values[i]; }
  // Return the compile-time extent.
  constexpr int size() const { return Count; }
  // Expose contiguous component storage.
  constexpr T *data() { return values; }
  // Expose read-only contiguous storage.
  constexpr const T *data() const { return values; }
};
using Vector3 = Array<float, 3>;
using BvhResultWords = Array<uint32_t, 4>;
using Node = const uint32_t *;
using Ray = rj_arith_bvh_ray;
constexpr auto kInvalidNode = UINT32_MAX;
constexpr auto kInfinity = __builtin_inff();

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
inline bool discards_all_opacities(uint32_t flags) {
  const unsigned force = flags & (ForceOpaque | ForceNonOpaque);
  const unsigned cull = flags & (CullOpaque | CullNonOpaque);
  return cull == (CullOpaque | CullNonOpaque) || (force == ForceOpaque && (cull & CullOpaque)) ||
         (force == ForceNonOpaque && (cull & CullNonOpaque));
}

// Test opacity and primitive-type culling before evaluating a child box.
inline bool culled_child(uint32_t flags, uint32_t pointer, uint32_t metadata) {
  const unsigned force = flags & (ForceOpaque | ForceNonOpaque);
  const auto type = BvhNodeType(pointer & 7);
  return ((metadata & 1) && force != ForceNonOpaque && (flags & CullOpaque)) ||
         ((metadata & 2) && force != ForceOpaque && (flags & CullNonOpaque)) ||
         (((metadata & 4) || type <= BvhNodeType::Triangle3) && (flags & SkipTriangles)) ||
         (((metadata & 8) || type == BvhNodeType::Procedural) && (flags & SkipProcedural));
}

// Intersect four decoded boxes and apply the resource sorting/tie policy.
inline BvhResultWords box(const Node &node, const rj_arith_bvh_controls &resource, const Ray &ray) {
  const bool half = ray.node_type == BvhNodeType::Box16;
  BvhResultWords children{};
  Array<float, 4> keys{};
#if defined(RJ_ARITH_BVH_SIMD)
  float near_values[4], far_values[4];
  const unsigned valid = box_distances(node, half, ray, near_values, far_values);
#endif
  for (int child_index = 0; child_index < children.size(); ++child_index) {
    children[child_index] = kInvalidNode;
    if (node[child_index] == kInvalidNode ||
        culled_child(ray.flags, node[child_index], half ? 0 : node[28] >> (8 * child_index))) {
      continue;
    }
#if defined(RJ_ARITH_BVH_SIMD)
    if (!((valid >> child_index) & 1)) {
      continue;
    }
    const float near = near_values[child_index], far = far_values[child_index];
#else
    // Bounding boxes store min.xyz, max.xyz starting at byte 16.
    const auto bound = [&](int bound_component) {
      const int bound_index = 6 * child_index + bound_component;
      return half ? f16_to_float(uint16_t(node[4 + bound_index / 2] >> ((bound_index & 1) * 16)))
                  : bit_cast<float>(node[4 + bound_index]);
    };
    float near = -kInfinity, far = kInfinity;
    for (int axis = 0; axis < 3; ++axis) {
      const float lo = bound(axis), hi = bound(axis + 3);
      if (!(lo <= hi)) {
        near = kInfinity;
        far = -kInfinity;
        break;
      }
      if (__builtin_isinf(ray.inverse[axis])) {
        if (!(lo <= ray.origin[axis] && ray.origin[axis] <= hi)) {
          near = kInfinity;
          far = -kInfinity;
          break;
        }
        continue;
      }
      const float lower_bound_distance = (lo - ray.origin[axis]) * ray.inverse[axis];
      const float upper_bound_distance = (hi - ray.origin[axis]) * ray.inverse[axis];
      near = maximum(near, minimum(lower_bound_distance, upper_bound_distance));
      far = minimum(far, maximum(lower_bound_distance, upper_bound_distance));
    }
#endif
    const float first = maximum(0.0f, near), last = minimum(ray.extent, far);
    if (first <= last * (1.0f + resource.grow * 0x1p-24f)) {
      children[child_index] = node[child_index];
      keys[child_index] = resource.sort == 1   ? first - last
                          : resource.sort == 2 ? near + far
                                               : first;
    }
  }
  if (resource.sort_enabled) {
    // GPURT's four-input sorting network defines tie order, including missed children.
    const auto order = [&](int first_child, int second_child) {
      if (children[first_child] == kInvalidNode ||
          (children[second_child] != kInvalidNode && keys[second_child] < keys[first_child])) {
        swap(children[first_child], children[second_child]);
        swap(keys[first_child], keys[second_child]);
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

// Intersect a tagged triangle with inherited stepwise RTZ and edge coverage.
inline BvhResultWords triangle(const Node &node, const rj_arith_bvh_controls &resource,
                               const Ray &ray) {
  // RRA TriangleNode::GetTriangle maps each tag onto three of the five vertices.
  static constexpr int vertices[][3] = {{0, 1, 2}, {1, 3, 2}, {2, 3, 4}, {2, 4, 0}};
  const auto vertex = [&](int vertex_index) {
    const auto offset = 3 * vertices[static_cast<int>(ray.node_type)][vertex_index];
    return Vector3{bit_cast<float>(node[offset]), bit_cast<float>(node[offset + 1]),
                   bit_cast<float>(node[offset + 2])};
  };
  const Vector3 vertex0 = vertex(0), vertex1 = vertex(1), vertex2 = vertex(2);
  // W7900 breaks equal-magnitude ties in X, Y, Z order.
  int dominant_axis = 0;
  for (int axis = 1; axis < 3; ++axis) {
    if (__builtin_fabsf(ray.direction[axis]) > __builtin_fabsf(ray.direction[dominant_axis])) {
      dominant_axis = axis;
    }
  }
  // W7900 returns the three projected edge functions in unnormalized ray space.
  // Each subtraction, product and ordered sum rounds toward zero at FP32 precision.
  Array<float, 4> bary{};
  float numerator, denominator;
  Array<Vector3, 3> projected;
#if defined(RJ_ARITH_BVH_SIMD)
  {
    using namespace bvh_rounding;
    // Process the three vertices together, retaining each FP32 rounding point.
    const int first = (dominant_axis + 1) % 3, second = (dominant_axis + 2) % 3;
    const auto coordinate = [&](int axis) {
      return subtract(pack(vertex0[axis], vertex1[axis], vertex2[axis]),
                      broadcast(ray.origin[axis]));
    };
    const auto depth = coordinate(dominant_axis);
    const auto direction = broadcast(ray.direction[dominant_axis]);
    const auto x = subtract(multiply(coordinate(first), direction),
                            multiply(depth, broadcast(ray.direction[first])));
    const auto y = subtract(multiply(coordinate(second), direction),
                            multiply(depth, broadcast(ray.direction[second])));
    const auto edge = subtract(
        multiply(permute<_MM_SHUFFLE(3, 1, 0, 2)>(x), permute<_MM_SHUFFLE(3, 0, 2, 1)>(y)),
        multiply(permute<_MM_SHUFFLE(3, 1, 0, 2)>(y), permute<_MM_SHUFFLE(3, 0, 2, 1)>(x)));
    const auto weights = multiply(edge, direction);
    const auto distances = multiply(edge, depth);
    const auto pairs = interleave_low(distances, weights);
    const auto totals = add(add(pairs, permute<_MM_SHUFFLE(3, 2, 3, 2)>(pairs)),
                            interleave_high(distances, weights));
    float result[4], px[4], py[4], pz[4];
    store(result, totals);
    numerator = result[0];
    denominator = result[1];
    store(bary.data(), weights);
    store(px, x);
    store(py, y);
    store(pz, depth);
    for (int i = 0; i < 3; ++i) {
      projected[i] = {px[i], py[i], pz[i]};
    }
  }
#else
  {
    // This unit rounds each FP32 step toward zero, independently of MODE.
    const auto add = [](float a, float b) { return graphics_rounding::add_f32(a, b, 3); };
    const auto sub = [](float a, float b) { return graphics_rounding::subtract_f32(a, b, 3); };
    const auto mul = [](float a, float b) { return graphics_rounding::multiply_f32(a, b, 3); };
    const int first_axis = (dominant_axis + 1) % 3, second_axis = (dominant_axis + 2) % 3;
    const auto project = [&](const Vector3 &point) -> Vector3 {
      const float depth = sub(point[dominant_axis], ray.origin[dominant_axis]);
      return {
          sub(mul(sub(point[first_axis], ray.origin[first_axis]), ray.direction[dominant_axis]),
              mul(depth, ray.direction[first_axis])),
          sub(mul(sub(point[second_axis], ray.origin[second_axis]), ray.direction[dominant_axis]),
              mul(depth, ray.direction[second_axis])),
          depth};
    };
    projected = {project(vertex0), project(vertex1), project(vertex2)};
    Array<float, 3> distances;
    for (int vertex_index = 0; vertex_index < 3; ++vertex_index) {
      const auto &first = projected[(vertex_index + 2) % 3];
      const auto &second = projected[(vertex_index + 1) % 3];
      const float edge = sub(mul(first[0], second[1]), mul(first[1], second[0]));
      bary[vertex_index] = mul(edge, ray.direction[dominant_axis]);
      distances[vertex_index] = mul(edge, projected[vertex_index][2]);
    }
    numerator = add(add(distances[0], distances[1]), distances[2]);
    denominator = add(add(bary[0], bary[1]), bary[2]);
  }
#endif
  // W7900 uses FP32 edge signs for coverage. When an edge rounds to zero,
  // the top-left rule uses the same projected endpoints. For a horizontal
  // edge the dominant ray direction determines the orientation of X.
  const auto covers_edge = [&](int edge_index) {
    const float edge = bary[edge_index];
    if (edge != 0) {
      return denominator < 0 ? edge < 0 : edge > 0;
    }
    const auto &first = projected[(edge_index + 1) % 3];
    const auto &second = projected[(edge_index + 2) % 3];
    if (first[1] != second[1]) {
      return denominator < 0 ? second[1] < first[1] : second[1] > first[1];
    }
    const bool decreasing_x = (denominator < 0) != (ray.direction[dominant_axis] < 0);
    return decreasing_x ? second[0] < first[0] : second[0] > first[0];
  };
  const bool intersect = denominator != 0 && numerator / denominator >= 0 && covers_edge(0) &&
                         covers_edge(1) && covers_edge(2);
  const int triangle_index = static_cast<int>(ray.node_type);
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
  for (int vertex_index = 0; vertex_index < 3; ++vertex_index) {
    bary[vertex_index] = orient(bary[vertex_index]);
  }
  numerator = orient(numerator);
  denominator = orient(denominator);
  const bool cull_face = !double_sided && (ray.flags & (denominator >= 0 ? CullFront : CullBack));
  const bool culled = cull_face || (ray.flags & (opaque ? CullOpaque : CullNonOpaque));
  const bool hit = intersect && !culled;
  if (resource.barycentrics) {
    if (!hit) {
      numerator =
          __builtin_isnan(denominator) ? kInfinity : __builtin_copysignf(kInfinity, denominator);
    }
    // W7900 aliases selector 3 to the oriented weight of vertex 0.
    bary[3] = bary[0];
    // The intersection unit canonicalizes NaNs independently of host propagation order.
    const auto encode = [](float value) {
      return __builtin_isnan(value) ? 0xffc00000u : bit_cast<uint32_t>(value);
    };
    return {bit_cast<uint32_t>(numerator), encode(denominator), encode(bary[control & 3]),
            encode(bary[(control >> 2) & 3])};
  }
  // With pointer flags disabled the node holds a base primitive ID shared by its triangles.
  const uint32_t id = resource.pointer_flags ? control : node[15] + triangle_index;
  if (!hit) {
    return {bit_cast<uint32_t>(kInfinity), bit_cast<uint32_t>(1.0f), id, 0};
  }
  return {bit_cast<uint32_t>(numerator), bit_cast<uint32_t>(denominator), id, 1};
}

// Evaluate only live lanes, including null-fetch and primitive-culling results.
inline void execute(int lanes, uint64_t exec_mask, uint64_t fetch_mask, const void *nodes,
                    uint32_t node_stride, const rj_arith_bvh_ray *rays,
                    const rj_arith_bvh_controls &controls, uint32_t *result) {
  // Failed or inactive fetches need no valid ray data; their results stay private.
  fetch_mask &= exec_mask;
  for (int lane = 0; lane < lanes; ++lane) {
    BvhResultWords value{kInvalidNode, kInvalidNode, kInvalidNode, kInvalidNode};
    if ((fetch_mask >> lane) & 1) {
      const auto &ray = rays[lane];
      const bool is_triangle = ray.node_type <= Triangle3;
      if (ray.node_type <= Box32 && !discards_all_opacities(ray.flags) &&
          !(is_triangle && (ray.flags & SkipTriangles))) {
        // Each node kind reads only the words supplied by its complete fetch.
        uint32_t node[32];
        __builtin_memcpy(node,
                         static_cast<const unsigned char *>(nodes) + uint64_t(lane) * node_stride,
                         ray.node_type == Box32 ? 128 : 64);
        value = is_triangle ? triangle(node, controls, ray) : box(node, controls, ray);
      }
    }
    for (int i = 0; i < 4; ++i) {
      result[i * lanes + lane] = value[i];
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::bvh
