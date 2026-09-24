// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_IMAGE_CUBE_H_
#define ROCJITSU_VM_AMDGPU_IMAGE_CUBE_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

/// Cube faces use +X, -X, +Y, -Y, +Z, -Z ordering. Texture coordinates here
/// are normalized to [0, 1]; the ISA supplies them with an additional +1.
inline std::array<double, 3> image_cube_direction(uint32_t face, double u, double v) {
  const double s = 2 * u - 1, t = 2 * v - 1;
  switch (face) {
  case 0:
    return {1, -t, -s};
  case 1:
    return {-1, -t, s};
  case 2:
    return {s, 1, t};
  case 3:
    return {s, -1, -t};
  case 4:
    return {s, -t, 1};
  default:
    return {-s, -t, -1};
  }
}

/// Project a direction onto a face, or return null at its projection plane.
inline std::optional<std::array<double, 2>>
image_cube_project(uint32_t face, const std::array<double, 3> &direction) {
  const auto [x, y, z] = direction;
  const double major = 2 * std::abs(direction[face / 2]);
  if (major == 0)
    return std::nullopt;
  const double s = face == 0 ? -z : face == 1 ? z : face == 5 ? -x : x;
  const double t = face == 2 ? z : face == 3 ? -z : -y;
  return std::array{s / major + 0.5, t / major + 0.5};
}

struct ImageCubeDerivativePolicy {
  std::array<int, 2> rounding_directions{};
  std::array<bool, 2> reflected_coordinates{};
};

/// Rounding for derivatives across adjacent cube faces. The edge-crossing
/// coordinate retains its source orientation; an unchanged parallel coordinate
/// uses ordinary magnitude rounding. A reflected parallel coordinate first
/// forms a sum of the ISA-biased coordinates, retaining 24 significant bits.
inline ImageCubeDerivativePolicy image_cube_derivative_policy(uint32_t face, uint32_t source_face) {
  // At (1, 1), each non-major direction component is its coordinate's sign.
  auto direction = image_cube_direction(source_face, 1, 1);
  const double sign = face & 1 ? -1 : 1;
  direction[source_face / 2] *= -sign * direction[face / 2];
  direction[face / 2] = sign;
  const auto projected = *image_cube_project(face, direction);
  const std::array axes{face / 2 == 0 ? 2u : 0u, face / 2 == 1 ? 2u : 1u};
  ImageCubeDerivativePolicy policy;
  for (uint32_t i = 0; i < 2; ++i) {
    const int orientation = static_cast<int>(2 * projected[i] - 1);
    const bool crosses_edge = axes[i] == source_face / 2;
    policy.rounding_directions[i] = crosses_edge || orientation < 0 ? orientation : 0;
    policy.reflected_coordinates[i] = !crosses_edge && orientation < 0;
  }
  return policy;
}

/// Biased cube coordinates lie in [1, 2], so their sum retains 22 fractional
/// bits. After subtracting the face offset, halfway derivatives round away from zero.
inline double image_cube_reflected_derivative(double value) {
  return std::copysign(std::floor(std::abs(value) * 0x1p22 + 0.5) * 0x1p-22, value);
}

/// An integer texel coordinate and its physical cube face.
struct ImageCubeTexel {
  uint32_t face, x, y;
};

/// Remap an out-of-face texel onto the neighboring face.
inline ImageCubeTexel image_cube_texel(uint32_t face, double x, double y, uint32_t size) {
  const auto direction = image_cube_direction(face, (x + 0.5) / size, (y + 0.5) / size);
  const double ax = std::abs(direction[0]), ay = std::abs(direction[1]),
               az = std::abs(direction[2]);
  // Z wins ties, followed by Y, matching the cube coordinate instructions.
  const uint32_t axis = az >= ax && az >= ay ? 2 : ay >= ax ? 1 : 0;
  const uint32_t selected = 2 * axis + (direction[axis] < 0);
  const auto projected = *image_cube_project(selected, direction);
  return {
      selected,
      static_cast<uint32_t>(std::clamp(std::floor(projected[0] * size), 0.0, double(size - 1))),
      static_cast<uint32_t>(std::clamp(std::floor(projected[1] * size), 0.0, double(size - 1)))};
}

} // namespace rocjitsu::amdgpu

#endif
