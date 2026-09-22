// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <cstdint>

namespace rocjitsu::plugins::race_detector {

/// 3D integer coordinate for workgroup IDs and block dimensions.
class Dim3d {
public:
  Dim3d() = delete;
  explicit Dim3d(int x) : x(x), y(0), z(0) {}
  Dim3d(int x, int y) : x(x), y(y), z(0) {}
  Dim3d(int x, int y, int z) : x(x), y(y), z(z) {}
  int x = 0;
  int y = 0;
  int z = 0;
};

/// Iterate over active lanes of an exec mask, calling func(lane) for each.
/// waveSize must be 32 or 64 (the two GPU wave sizes). When all lanes within
/// the wave are active, a fast path avoids per-bit checking.
template <typename F> void forEachActiveLane(uint64_t execMask, int waveSize, F func) {
  uint64_t fullMask = (waveSize == 64) ? ~0ULL : ((1ULL << waveSize) - 1);

  if ((execMask & fullMask) == fullMask) {
    for (int lane = 0; lane < waveSize; ++lane) {
      func(lane);
    }
  } else {
    for (int lane = 0; lane < waveSize; ++lane) {
      if ((execMask >> lane) & 1) {
        func(lane);
      }
    }
  }
}

} // namespace rocjitsu::plugins::race_detector
