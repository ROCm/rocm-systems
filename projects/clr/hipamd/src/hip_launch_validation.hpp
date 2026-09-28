/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_LAUNCH_VALIDATION_HPP_
#define HIP_LAUNCH_VALIDATION_HPP_

#include <hip/hip_runtime.h>
#include <limits>

#include "device/device.hpp"
#include "platform/ndrange.hpp"

namespace hip {

//! Bitmask of launch-configuration violations detected for a given LaunchParams instance.
enum LaunchViolation : uint16_t {
  kLaunchOk            = 0,
  kZeroGlobal          = 1u << 0,  //!< global_ has a zero dim (grid*block == 0)
  kZeroBlock           = 1u << 1,  //!< local_ has a zero dim
  kGridOverflow        = 1u << 2,  //!< global_ does not fit uint32_t
  kBlockOverflow       = 1u << 3,  //!< local_ does not fit uint16_t
  kClusterOverflow     = 1u << 4,  //!< cluster_ does not fit uint8_t
  kClusterIndivisible  = 1u << 5,  //!< grid_ % cluster_ != 0
  kBlockExceedsMaxWG   = 1u << 6,  //!< (local_[0] * local[1] * local[2]) exceeds maxWorkGroupSize_
  kSharedMemExceedsMax = 1u << 7,  //!< sharedMemBytes exceeds localMemSizePerCU_
  kSharedMemOverflow   = 1u << 8,  //!< sharedMemBytes does not fit uint32_t
};

//! Structure to store launch parameters. Lives here rather than in platform/ndrange.hpp so the
//! ctor can read amd::Device::info() without an include cycle.
struct LaunchParams {
  amd::NDRange32 global_;    //!< Total number of work-items in N-dims (matches AQL grid_size)
  amd::NDRange16 local_;     //!< Number of work-items per workgroup (matches AQL workgroup_size)
  amd::NDRange8 cluster_;    //!< Cluster dims (matches AQL cluster_size, max 255)
  amd::NDRange32 grid_;      //!< Total number of workgroups in grid in N-dims
  uint32_t sharedMemBytes_;  //!< Shared Memory bytes
  bool hipParams_;           //!< If this is launched through hipParams_
  uint16_t violations_;      //!< Bitmask of LaunchViolation bits detected for this config.

  LaunchParams(size_t globalX, size_t globalY, size_t globalZ, uint32_t localX, uint32_t localY,
               uint32_t localZ, size_t sharedMemBytes, const amd::Device& device,
               uint32_t clusterX = 1, uint32_t clusterY = 1, uint32_t clusterZ = 1,
               uint32_t gridX = 1, uint32_t gridY = 1, uint32_t gridZ = 1, bool hipParams = false)
      : global_(static_cast<uint32_t>(globalX), static_cast<uint32_t>(globalY),
                static_cast<uint32_t>(globalZ)),
        local_(static_cast<uint16_t>(localX), static_cast<uint16_t>(localY),
               static_cast<uint16_t>(localZ)),
        cluster_(static_cast<uint8_t>(clusterX), static_cast<uint8_t>(clusterY),
                 static_cast<uint8_t>(clusterZ)),
        grid_(gridX, gridY, gridZ),
        sharedMemBytes_(static_cast<uint32_t>(sharedMemBytes)),
        hipParams_(hipParams),
        violations_(kLaunchOk) {
    const device::Info& info = device.info();

    if (local_.product() > info.maxWorkGroupSize_) {
      violations_ |= kBlockExceedsMaxWG;
    }
    if (sharedMemBytes_ > info.localMemSizePerCU_) {
      violations_ |= kSharedMemExceedsMax;
    }
    if (sharedMemBytes > std::numeric_limits<uint32_t>::max()) {
      violations_ |= kSharedMemOverflow;
    }

    if (global_[0] == 0 || global_[1] == 0 || global_[2] == 0) {
      violations_ |= kZeroGlobal;
    }

    if (!amd::NDRange8::CanSafelyNarrow(clusterX, clusterY, clusterZ)) {
      violations_ |= kClusterOverflow;
    }

    if (!amd::NDRange16::CanSafelyNarrow(localX, localY, localZ)) {
      violations_ |= kBlockOverflow;
    }

    if (local_[0] == 0 || local_[1] == 0 || local_[2] == 0) {
      violations_ |= kZeroBlock;
    }

    if (hipParams_) {
      if (!amd::NDRange32::CanSafelyNarrow(globalX, globalY, globalZ)) {
        violations_ |= kGridOverflow;
      }
    } else {
      // The app supplied the global and local size directly, so deduce the grid (total blocks).
      if ((violations_ & kZeroBlock) != 0) {
        // Avoid divide by 0
        return;
      }
      grid_[0] = global_[0] / local_[0];
      grid_[1] = global_[1] / local_[1];
      grid_[2] = global_[2] / local_[2];
    }

    if (clusterX > 1 || clusterY > 1 || clusterZ > 1) {
      if (!CheckClusterDivisibility(clusterX, clusterY, clusterZ)) {
        violations_ |= kClusterIndivisible;
      }
    }
  }

  bool CheckClusterDivisibility(uint32_t clusterX, uint32_t clusterY, uint32_t clusterZ) {
    // A cluster launch splits the same blocks across the CU/WGPs of one shader engine, so the
    // grid dims have to be divisible by the cluster dims.
    if ((grid_[0] % clusterX != 0) || (grid_[1] % clusterY != 0) || (grid_[2] % clusterZ != 0)) {
      return false;
    }
    return true;
  }

  //! Sometimes we receive cluster launch info from kernel, not through HIP launch kernel APIs.
  bool UpdateClusterLaunchParams(uint32_t clusterX, uint32_t clusterY, uint32_t clusterZ) {
    if (clusterX > 1 || clusterY > 1 || clusterZ > 1) {
      if (!CheckClusterDivisibility(clusterX, clusterY, clusterZ)) {
        violations_ |= kClusterIndivisible;
        return false;
      }
      violations_ &= ~static_cast<uint16_t>(kClusterIndivisible);
      cluster_[0] = static_cast<uint8_t>(clusterX);
      cluster_[1] = static_cast<uint8_t>(clusterY);
      cluster_[2] = static_cast<uint8_t>(clusterZ);
    }
    return true;
  }

  static constexpr uint16_t kInvalidConfigBits =
      kGridOverflow | kBlockOverflow | kClusterOverflow | kClusterIndivisible | kZeroBlock;

  bool IsValidConfig() const { return (violations_ & kInvalidConfigBits) == 0; }
};

//! Structure to store launch parameters in HIP Style (global and local size needs computation).
struct HIPLaunchParams : public LaunchParams {

  HIPLaunchParams(uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                  uint32_t blockY, uint32_t blockZ, size_t sharedMemBytes,
                  const amd::Device& device, uint32_t globalX_remainder = 0,
                  uint32_t globalY_remainder = 0, uint32_t globalZ_remainder = 0,
                  uint32_t clusterX = 1, uint32_t clusterY = 1, uint32_t clusterZ = 1)
                  : LaunchParams(static_cast<size_t>(gridX) * blockX + globalX_remainder,
                                 static_cast<size_t>(gridY) * blockY + globalY_remainder,
                                 static_cast<size_t>(gridZ) * blockZ + globalZ_remainder,
                                 blockX, blockY, blockZ, sharedMemBytes, device, clusterX,
                                 clusterY, clusterZ, gridX, gridY, gridZ, true /*hipParams*/) {}
};

//! One entry in a launch-error rule table: the violation bits it matches and the error to return.
struct LaunchErrorRule {
  uint16_t violations;   //!< bits this rule matches
  hipError_t error;      //!< code to return if violation occurred
};

//! Return the error for the FIRST rule matching any set bit; hipSuccess if none. Callers list
//! rules in their original check order so the reported error code does not change.
template <size_t N>
inline hipError_t MapLaunchViolations(uint16_t violations, const LaunchErrorRule (&rules)[N]) {
  if (violations == kLaunchOk) return hipSuccess;
  for (const auto& rule : rules) {
    if (violations & rule.violations) return rule.error;
  }
  return hipSuccess;
}

//! Helper specifying most used bits to be checked
static constexpr uint16_t kConfigBits =
    kGridOverflow | kBlockOverflow | kClusterOverflow | kClusterIndivisible;

}  // namespace hip

#endif  // HIP_LAUNCH_VALIDATION_HPP_
