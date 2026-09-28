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

//! Dims that cannot be encoded in the AQL packet, or that contradict each other.
static constexpr LaunchViolationBits kMalformedDimsBits =
    kGridOverflow | kBlockOverflow | kClusterOverflow | kClusterIndivisible;

//! The bits that make a config outright unlaunchable, regardless of entry point.
static constexpr LaunchViolationBits kUnlaunchableBits = kMalformedDimsBits | kZeroBlock;

//! Every violation bit, spelled out so a new LaunchViolation forces a decision about this set.
static constexpr LaunchViolationBits kAnyViolationBits = kZeroGlobal | kZeroBlock | kGridOverflow |
    kBlockOverflow | kClusterOverflow | kClusterIndivisible | kBlockExceedsMaxWG |
    kSharedMemExceedsMax | kSharedMemOverflow;

//! One entry in a launch-error rule table: the violation bits it matches and the error to return.
struct LaunchErrorRule {
  LaunchViolationBits violations;  //!< bits this rule matches
  hipError_t error;                //!< code to return if violation occurred
};

//! Return the error for the FIRST rule matching any violation bit; hipSuccess if none.
template <size_t N>
hipError_t MapLaunchViolations(LaunchViolationBits violations, const LaunchErrorRule (&rules)[N]) {
  if (violations == kLaunchOk) return hipSuccess;
  for (const auto& rule : rules) {
    if (violations & rule.violations) return rule.error;
  }
  return hipSuccess;
}

//! Reject anything unlaunchable and leave the rest to ihipLaunchKernel_validate.
static constexpr LaunchErrorRule kUnlaunchableConfigRules[] = {
    {kUnlaunchableBits, hipErrorInvalidConfiguration},
};

//! As above but tolerating a zero block, which these entry points leave to a later check.
static constexpr LaunchErrorRule kMalformedDimsRules[] = {
    {kMalformedDimsBits, hipErrorInvalidConfiguration},
};

//! The classic hipModuleLaunchKernel contract: every violation reports hipErrorInvalidValue.
static constexpr LaunchErrorRule kModuleLaunchRules[] = {
    {kAnyViolationBits, hipErrorInvalidValue},
};

//! No cluster requested — one block per cluster in every dim.
static constexpr dim3 kNoCluster{1, 1, 1};

//! No leftover work-items beyond grid * block.
static constexpr dim3 kNoRemainder{0, 0, 0};

//! A cluster splits the grid across one shader engine, so grid must divide by cluster.
inline bool IsClusterDivisible(const amd::NDRange32& grid, const dim3& cluster) {
  return (grid[0] % cluster.x == 0) && (grid[1] % cluster.y == 0) && (grid[2] % cluster.z == 0);
}

//! The violations derivable from a narrowed index space plus a device, so validate can re-run them.
inline LaunchViolationBits CheckNDRangeAgainstDevice(const amd::NDRange32& global,
                                                     const amd::NDRange16& local,
                                                     size_t sharedMemBytes,
                                                     const amd::Device& device) {
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
  if (local[0] == 0 || local[1] == 0 || local[2] == 0) {
    violations |= kZeroBlock;
  }
  return violations;
}

//! Narrow into \a ndrange and report the violations; \a grid is deduced iff \a deduceGrid.
inline LaunchViolationBits BuildLaunchNDRange(amd::NDRangeContainer& ndrange, amd::NDRange32& grid,
                                             const amd::NDRange& globalDim, const dim3& localDim,
                                             const dim3& clusterDim, size_t sharedMemBytes,
                                             const amd::Device& device, bool deduceGrid) {
  const amd::NDRange32 global(
      static_cast<uint32_t>(globalDim[0]), static_cast<uint32_t>(globalDim[1]),
      static_cast<uint32_t>(globalDim[2]));
  const amd::NDRange16 local(static_cast<uint16_t>(localDim.x), static_cast<uint16_t>(localDim.y),
                             static_cast<uint16_t>(localDim.z));
  const amd::NDRange8 cluster(static_cast<uint8_t>(clusterDim.x),
                              static_cast<uint8_t>(clusterDim.y),
                              static_cast<uint8_t>(clusterDim.z));
  ndrange = amd::NDRangeContainer(3, amd::NDRange(0, 0, 0), global, local, cluster);

  LaunchViolationBits violations =
      CheckNDRangeAgainstDevice(global, local, sharedMemBytes, device);

  // The remaining violations need the un-narrowed inputs, so they can only be spotted here.
  if (!amd::NDRange8::CanSafelyNarrow(clusterDim.x, clusterDim.y, clusterDim.z)) {
    violations |= kClusterOverflow;
  }

  if (!amd::NDRange16::CanSafelyNarrow(localDim.x, localDim.y, localDim.z)) {
    violations |= kBlockOverflow;
  }

  if (!amd::NDRange32::CanSafelyNarrow(globalDim[0], globalDim[1], globalDim[2])) {
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

  if (clusterDim.x > 1 || clusterDim.y > 1 || clusterDim.z > 1) {
    if (!IsClusterDivisible(grid, clusterDim)) {
      violations |= kClusterIndivisible;
    }
  }
  return violations;
}

//! Build from an app supplied global and local size; \a grid receives the deduced blocks.
inline LaunchViolationBits BuildLaunchNDRangeFromGlobal(amd::NDRangeContainer& ndrange,
                                                       amd::NDRange32& grid, const dim3& globalDim,
                                                       const dim3& localDim, const dim3& clusterDim,
                                                       size_t sharedMemBytes,
                                                       const amd::Device& device) {
  grid = amd::NDRange32(1, 1, 1);
  return BuildLaunchNDRange(ndrange, grid, amd::NDRange(globalDim.x, globalDim.y, globalDim.z),
                            localDim, clusterDim, sharedMemBytes, device, true /*deduceGrid*/);
}

//! Build HIP style, where the app supplies the grid and block and the global is computed.
inline LaunchViolationBits BuildLaunchNDRangeFromGrid(amd::NDRangeContainer& ndrange,
                                                     amd::NDRange32& grid, const dim3& gridDim,
                                                     const dim3& blockDim, const dim3& remainder,
                                                     const dim3& clusterDim, size_t sharedMemBytes,
                                                     const amd::Device& device) {
  grid = amd::NDRange32(gridDim.x, gridDim.y, gridDim.z);
  const amd::NDRange globalDim(static_cast<size_t>(gridDim.x) * blockDim.x + remainder.x,
                               static_cast<size_t>(gridDim.y) * blockDim.y + remainder.y,
                               static_cast<size_t>(gridDim.z) * blockDim.z + remainder.z);
  return BuildLaunchNDRange(ndrange, grid, globalDim, blockDim, clusterDim, sharedMemBytes, device,
                            false /*deduceGrid*/);
}

//! Build the NDRange and map any violation to this entry point's error code.
template <size_t N>
hipError_t MakeLaunchNDRangeFromGlobal(amd::NDRangeContainer& ndrange, amd::NDRange32& grid,
                                       const dim3& globalDim, const dim3& localDim,
                                       const dim3& clusterDim, size_t sharedMemBytes,
                                       const amd::Device& device,
                                       const LaunchErrorRule (&rules)[N]) {
  return MapLaunchViolations(BuildLaunchNDRangeFromGlobal(ndrange, grid, globalDim, localDim,
                                                          clusterDim, sharedMemBytes, device),
                             rules);
}

//! Build without mapping violations, for entry points that defer to ihipLaunchKernel_validate.
inline void MakeLaunchNDRangeFromGlobal(amd::NDRangeContainer& ndrange, amd::NDRange32& grid,
                                        const dim3& globalDim, const dim3& localDim,
                                        const dim3& clusterDim, size_t sharedMemBytes,
                                        const amd::Device& device) {
  BuildLaunchNDRangeFromGlobal(ndrange, grid, globalDim, localDim, clusterDim, sharedMemBytes,
                               device);
}

//! Build the NDRange and map any violation to this entry point's error code.
template <size_t N>
hipError_t MakeLaunchNDRangeFromGrid(amd::NDRangeContainer& ndrange, amd::NDRange32& grid,
                                     const dim3& gridDim, const dim3& blockDim,
                                     const dim3& remainder, const dim3& clusterDim,
                                     size_t sharedMemBytes, const amd::Device& device,
                                     const LaunchErrorRule (&rules)[N]) {
  return MapLaunchViolations(BuildLaunchNDRangeFromGrid(ndrange, grid, gridDim, blockDim, remainder,
                                                        clusterDim, sharedMemBytes, device),
                             rules);
}

//! Build the NDRange without mapping violations — see the FromGlobal overload above.
inline void MakeLaunchNDRangeFromGrid(amd::NDRangeContainer& ndrange, amd::NDRange32& grid,
                                      const dim3& gridDim, const dim3& blockDim,
                                      const dim3& remainder, const dim3& clusterDim,
                                      size_t sharedMemBytes, const amd::Device& device) {
  BuildLaunchNDRangeFromGrid(ndrange, grid, gridDim, blockDim, remainder, clusterDim,
                             sharedMemBytes, device);
}

}  // namespace hip

#endif  // HIP_LAUNCH_VALIDATION_HPP_
