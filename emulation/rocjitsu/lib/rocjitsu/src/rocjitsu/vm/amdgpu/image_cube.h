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
