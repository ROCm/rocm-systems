/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_LAUNCH_VALIDATION_HPP_
#define HIP_LAUNCH_VALIDATION_HPP_

#include <hip/hip_runtime.h>
#include <limits>
#include <type_traits>

#include "device/device.hpp"
#include "platform/ndrange.hpp"

namespace hip {

//! Bitmask of launch-configuration violations detected while building a launch NDRange.
enum LaunchViolation : uint16_t {
  kLaunchOk            = 0,
  kZeroGlobal          = 1u << 0,  //!< global has a zero dim (grid*block == 0)
  kZeroBlock           = 1u << 1,  //!< local has a zero dim
  kGridOverflow        = 1u << 2,  //!< global does not fit uint32_t
  kBlockOverflow       = 1u << 3,  //!< local does not fit uint16_t
  kClusterOverflow     = 1u << 4,  //!< cluster does not fit uint8_t
  kClusterIndivisible  = 1u << 5,  //!< grid % cluster != 0
  kBlockExceedsMaxWG   = 1u << 6,  //!< (local[0] * local[1] * local[2]) exceeds maxWorkGroupSize_
  kSharedMemExceedsMax = 1u << 7,  //!< sharedMemBytes exceeds localMemSizePerCU_
  kSharedMemOverflow   = 1u << 8,  //!< sharedMemBytes does not fit uint32_t
};

//! Storage type for a combination of LaunchViolation bits.
using LaunchViolationBits = std::underlying_type_t<LaunchViolation>;

//! Helper specifying most used bits to be checked
static constexpr LaunchViolationBits kCommonRulesBits =
    kGridOverflow | kBlockOverflow | kClusterOverflow | kClusterIndivisible;

//! The bits that make a config outright unlaunchable, regardless of entry point.
static constexpr LaunchViolationBits kInvalidConfigBits =
    kGridOverflow | kBlockOverflow | kClusterOverflow | kClusterIndivisible | kZeroBlock;

//! One entry in a launch-error rule table: the violation bits it matches and the error to return.
struct LaunchErrorRule {
  LaunchViolationBits violations;  //!< bits this rule matches
  hipError_t error;                //!< code to return if violation occurred
};

//! Return the error for the FIRST rule matching any violation bit; hipSuccess if none. Callers
//! list rules in their original check order so the reported error code does not change.
template <size_t N>
hipError_t MapLaunchViolations(LaunchViolationBits violations, const LaunchErrorRule (&rules)[N]) {
  if (violations == kLaunchOk) return hipSuccess;
  for (const auto& rule : rules) {
    if (violations & rule.violations) return rule.error;
  }
  return hipSuccess;
}

//! A cluster launch splits the same blocks across the CU/WGPs of one shader engine, so the grid
//! dims have to be divisible by the cluster dims.
inline bool IsClusterDivisible(const amd::NDRange32& grid, uint32_t clusterX, uint32_t clusterY,
                               uint32_t clusterZ) {
  return (grid[0] % clusterX == 0) && (grid[1] % clusterY == 0) && (grid[2] % clusterZ == 0);
}

//! Narrow the app supplied index space into \a ndrange and report every configuration violation
//! detected while doing so. Shared by both entry styles; \a deduceGrid tells the two apart. When
//! it is false \a grid must already hold the app supplied grid (total blocks); when it is true the
//! grid is deduced here from global / local, which only the cluster checks need.
inline LaunchViolationBits BuildLaunchNDRange(amd::NDRangeContainer& ndrange, amd::NDRange32& grid,
                                              size_t globalX, size_t globalY, size_t globalZ,
                                              uint32_t localX, uint32_t localY, uint32_t localZ,
                                              uint32_t clusterX, uint32_t clusterY,
                                              uint32_t clusterZ, size_t sharedMemBytes,
                                              const amd::Device& device, bool deduceGrid) {
  const amd::NDRange32 global(static_cast<uint32_t>(globalX), static_cast<uint32_t>(globalY),
                              static_cast<uint32_t>(globalZ));
  const amd::NDRange16 local(static_cast<uint16_t>(localX), static_cast<uint16_t>(localY),
                             static_cast<uint16_t>(localZ));
  const amd::NDRange8 cluster(static_cast<uint8_t>(clusterX), static_cast<uint8_t>(clusterY),
                              static_cast<uint8_t>(clusterZ));
  ndrange = amd::NDRangeContainer(3, amd::NDRange(0, 0, 0), global, local, cluster);

  LaunchViolationBits violations = kLaunchOk;
  const device::Info& info = device.info();

  if (local.product() > info.maxWorkGroupSize_) {
    violations |= kBlockExceedsMaxWG;
  }
  if (static_cast<uint32_t>(sharedMemBytes) > info.localMemSizePerCU_) {
    violations |= kSharedMemExceedsMax;
  }
  if (sharedMemBytes > std::numeric_limits<uint32_t>::max()) {
    violations |= kSharedMemOverflow;
  }

  if (global[0] == 0 || global[1] == 0 || global[2] == 0) {
    violations |= kZeroGlobal;
  }

  if (!amd::NDRange8::CanSafelyNarrow(clusterX, clusterY, clusterZ)) {
    violations |= kClusterOverflow;
  }

  if (!amd::NDRange16::CanSafelyNarrow(localX, localY, localZ)) {
    violations |= kBlockOverflow;
  }

  if (local[0] == 0 || local[1] == 0 || local[2] == 0) {
    violations |= kZeroBlock;
  }

  if (!amd::NDRange32::CanSafelyNarrow(globalX, globalY, globalZ)) {
    violations |= kGridOverflow;
  }

  if (deduceGrid) {
    if (violations & kZeroBlock) {
      // Avoid divide by 0 — the grid cannot be deduced, so the cluster check is skipped too.
      return violations;
    }
    grid[0] = global[0] / local[0];
    grid[1] = global[1] / local[1];
    grid[2] = global[2] / local[2];
  }

  if (clusterX > 1 || clusterY > 1 || clusterZ > 1) {
    if (!IsClusterDivisible(grid, clusterX, clusterY, clusterZ)) {
      violations |= kClusterIndivisible;
    }
  }
  return violations;
}

//! Build the launch NDRange from an app supplied global (total work-items) and local size. \a grid
//! receives the deduced total number of workgroups. The global dims are already uint32_t here, so
//! kGridOverflow can never fire on this path.
inline LaunchViolationBits BuildLaunchNDRangeFromGlobal(
    amd::NDRangeContainer& ndrange, amd::NDRange32& grid, uint32_t globalX, uint32_t globalY,
    uint32_t globalZ, uint32_t localX, uint32_t localY, uint32_t localZ, uint32_t clusterX,
    uint32_t clusterY, uint32_t clusterZ, size_t sharedMemBytes, const amd::Device& device) {
  grid = amd::NDRange32(1, 1, 1);
  return BuildLaunchNDRange(ndrange, grid, globalX, globalY, globalZ, localX, localY, localZ,
                            clusterX, clusterY, clusterZ, sharedMemBytes, device,
                            true /*deduceGrid*/);
}

//! Build the launch NDRange in HIP style, where the app supplies a grid (total blocks) and a block
//! size and the global size needs computation. \a grid receives the app supplied grid.
inline LaunchViolationBits BuildLaunchNDRangeFromGrid(
    amd::NDRangeContainer& ndrange, amd::NDRange32& grid, uint32_t gridX, uint32_t gridY,
    uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t remainderX,
    uint32_t remainderY, uint32_t remainderZ, uint32_t clusterX, uint32_t clusterY,
    uint32_t clusterZ, size_t sharedMemBytes, const amd::Device& device) {
  grid = amd::NDRange32(gridX, gridY, gridZ);
  return BuildLaunchNDRange(ndrange, grid, static_cast<size_t>(gridX) * blockX + remainderX,
                            static_cast<size_t>(gridY) * blockY + remainderY,
                            static_cast<size_t>(gridZ) * blockZ + remainderZ, blockX, blockY,
                            blockZ, clusterX, clusterY, clusterZ, sharedMemBytes, device,
                            false /*deduceGrid*/);
}

//! Build the NDRange and map any violation to this entry point's error code.
template <size_t N>
hipError_t MakeLaunchNDRangeFromGlobal(amd::NDRangeContainer& ndrange, amd::NDRange32& grid,
                                       uint32_t globalX, uint32_t globalY, uint32_t globalZ,
                                       uint32_t localX, uint32_t localY, uint32_t localZ,
                                       uint32_t clusterX, uint32_t clusterY, uint32_t clusterZ,
                                       size_t sharedMemBytes, const amd::Device& device,
                                       const LaunchErrorRule (&rules)[N]) {
  return MapLaunchViolations(
      BuildLaunchNDRangeFromGlobal(ndrange, grid, globalX, globalY, globalZ, localX, localY, localZ,
                                   clusterX, clusterY, clusterZ, sharedMemBytes, device),
      rules);
}

//! Build the NDRange without mapping violations — for the entry points that historically deferred
//! every check to ihipLaunchKernel_validate.
inline void MakeLaunchNDRangeFromGlobal(amd::NDRangeContainer& ndrange, amd::NDRange32& grid,
                                        uint32_t globalX, uint32_t globalY, uint32_t globalZ,
                                        uint32_t localX, uint32_t localY, uint32_t localZ,
                                        uint32_t clusterX, uint32_t clusterY, uint32_t clusterZ,
                                        size_t sharedMemBytes, const amd::Device& device) {
  BuildLaunchNDRangeFromGlobal(ndrange, grid, globalX, globalY, globalZ, localX, localY, localZ,
                               clusterX, clusterY, clusterZ, sharedMemBytes, device);
}

//! Build the NDRange and map any violation to this entry point's error code.
template <size_t N>
hipError_t MakeLaunchNDRangeFromGrid(amd::NDRangeContainer& ndrange, amd::NDRange32& grid,
                                     uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                                     uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                                     uint32_t remainderX, uint32_t remainderY, uint32_t remainderZ,
                                     uint32_t clusterX, uint32_t clusterY, uint32_t clusterZ,
                                     size_t sharedMemBytes, const amd::Device& device,
                                     const LaunchErrorRule (&rules)[N]) {
  return MapLaunchViolations(
      BuildLaunchNDRangeFromGrid(ndrange, grid, gridX, gridY, gridZ, blockX, blockY, blockZ,
                                 remainderX, remainderY, remainderZ, clusterX, clusterY, clusterZ,
                                 sharedMemBytes, device),
      rules);
}

//! Build the NDRange without mapping violations — see the FromGlobal overload above.
inline void MakeLaunchNDRangeFromGrid(amd::NDRangeContainer& ndrange, amd::NDRange32& grid,
                                      uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                                      uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                                      uint32_t remainderX, uint32_t remainderY, uint32_t remainderZ,
                                      uint32_t clusterX, uint32_t clusterY, uint32_t clusterZ,
                                      size_t sharedMemBytes, const amd::Device& device) {
  BuildLaunchNDRangeFromGrid(ndrange, grid, gridX, gridY, gridZ, blockX, blockY, blockZ, remainderX,
                             remainderY, remainderZ, clusterX, clusterY, clusterZ, sharedMemBytes,
                             device);
}

}  // namespace hip

#endif  // HIP_LAUNCH_VALIDATION_HPP_
