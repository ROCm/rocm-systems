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

//! Bitmask of launch-configuration violations detected while building a LaunchConfig.
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

//! The bits CheckNDRangeAgainstDevice derives, i.e. the ones that depend on the target device.
static constexpr LaunchViolationBits kDeviceDependentBits =
    kZeroGlobal | kZeroBlock | kBlockExceedsMaxWG | kSharedMemExceedsMax | kSharedMemOverflow;

//! The bits that make a config outright unlaunchable, regardless of entry point.
static constexpr LaunchViolationBits kUnlaunchableBits = kMalformedDimsBits | kZeroBlock;

//! Every violation bit, spelled out so a new LaunchViolation forces a decision about this set.
static constexpr LaunchViolationBits kAnyViolationBits = kMalformedDimsBits | kDeviceDependentBits;

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

// =================================================================================================
// Per entry point rule tables. Every launch API's error-code contract lives here, so the
// differences between them are diffable against each other rather than spread across call sites.
// An entry point that names no table defers every check to ihipLaunchKernel_validate.
// =================================================================================================

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

//! hipExtModuleLaunchKernel: a bad block is hipErrorInvalidConfiguration, unlike everywhere else.
static constexpr LaunchErrorRule kExtModuleLaunchRules[] = {
    {kUnlaunchableBits | kBlockExceedsMaxWG,    hipErrorInvalidConfiguration},
    {kSharedMemExceedsMax | kSharedMemOverflow, hipErrorInvalidValue},
    {kZeroGlobal,                               hipErrorInvalidValue},
};

//! hipLaunchCooperativeKernel: too much shared memory is reported as a too-large launch.
static constexpr LaunchErrorRule kCooperativeLaunchRules[] = {
    {kMalformedDimsBits | kBlockExceedsMaxWG,   hipErrorInvalidConfiguration},
    {kSharedMemExceedsMax | kSharedMemOverflow, hipErrorCooperativeLaunchTooLarge},
};

//! hipLaunchByPtr: the <<<>>> shim, which has only ever reported hipErrorInvalidValue.
static constexpr LaunchErrorRule kLaunchByPtrRules[] = {
    {kMalformedDimsBits | kBlockExceedsMaxWG, hipErrorInvalidValue},
};

//! The checks ihipLaunchKernel_validate applies on behalf of every entry point.
static constexpr LaunchErrorRule kValidateRules[] = {
    {kZeroGlobal | kZeroBlock,                  hipErrorInvalidConfiguration},
    {kSharedMemExceedsMax | kSharedMemOverflow, hipErrorInvalidValue},
    {kBlockExceedsMaxWG,                        hipErrorInvalidConfiguration},
};

//! No cluster requested — one block per cluster in every dim.
static constexpr dim3 kNoCluster{1, 1, 1};

//! No leftover work-items beyond grid * block.
static constexpr dim3 kNoRemainder{0, 0, 0};

//! A cluster splits the grid across one shader engine, so grid must divide by cluster.
inline bool IsClusterDivisible(const amd::NDRange32& grid, const dim3& cluster) {
  return (grid[0] % cluster.x == 0) && (grid[1] % cluster.y == 0) && (grid[2] % cluster.z == 0);
}

//! The violations derivable from a narrowed index space plus a device. See kDeviceDependentBits.
inline LaunchViolationBits CheckNDRangeAgainstDevice(const amd::NDRange32& global,
                                                     const amd::NDRange16& local,
                                                     size_t sharedMemBytes,
                                                     const amd::Device& device) {
  LaunchViolationBits violations = kLaunchOk;
  const auto& info = device.info();

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

//! Everything the launch path needs to dispatch a kernel, plus the violations found building it.
//!
//! Built once per launch by MakeLaunchConfigFromGrid/FromGlobal and threaded down through
//! ihipModuleLaunchKernel. The violations are computed during construction and never recomputed,
//! so a launch costs exactly one pass over the device limits.
struct LaunchConfig {
  // Declaration order matters: violations_ is initialised from ndrange_ in the member init list.
  amd::NDRangeContainer ndrange_;   //!< offset / global / local / cluster, ready for dispatch
  amd::NDRange32 grid_;             //!< blocks in the grid (global / local is lossy with remainder)
  const amd::Device* device_;       //!< the device the device-dependent bits were derived against
  size_t sharedMemBytes_;           //!< dynamic shared memory, un-narrowed
  LaunchViolationBits violations_;  //!< everything wrong with this config

  //! Narrow \a globalDim / \a localDim / \a clusterDim and record every violation found doing so.
  //! \a grid is taken from \a gridIn, or deduced as global / local when \a deduceGrid.
  LaunchConfig(const amd::NDRange& globalDim, const amd::NDRange32& gridIn, const dim3& localDim,
               const dim3& clusterDim, size_t sharedMemBytes, const amd::Device& device,
               bool deduceGrid)
      : ndrange_(3, amd::NDRange(0, 0, 0),
                 amd::NDRange32(static_cast<uint32_t>(globalDim[0]),
                                static_cast<uint32_t>(globalDim[1]),
                                static_cast<uint32_t>(globalDim[2])),
                 amd::NDRange16(static_cast<uint16_t>(localDim.x),
                                static_cast<uint16_t>(localDim.y),
                                static_cast<uint16_t>(localDim.z)),
                 amd::NDRange8(static_cast<uint8_t>(clusterDim.x),
                               static_cast<uint8_t>(clusterDim.y),
                               static_cast<uint8_t>(clusterDim.z))),
        grid_(gridIn),
        device_(&device),
        sharedMemBytes_(sharedMemBytes),
        violations_(CheckNDRangeAgainstDevice(ndrange_.global(), ndrange_.local(), sharedMemBytes,
                                              device)) {
    // The remaining violations need the un-narrowed inputs, so they can only be spotted here.
    if (!amd::NDRange8::CanSafelyNarrow(clusterDim.x, clusterDim.y, clusterDim.z)) {
      violations_ |= kClusterOverflow;
    }

    if (!amd::NDRange16::CanSafelyNarrow(localDim.x, localDim.y, localDim.z)) {
      violations_ |= kBlockOverflow;
    }

    if (!amd::NDRange32::CanSafelyNarrow(globalDim[0], globalDim[1], globalDim[2])) {
      violations_ |= kGridOverflow;
    }

    if (deduceGrid) {
      if (violations_ & kZeroBlock) {
        // Avoid divide by 0 — the grid cannot be deduced, so the cluster check is skipped too.
        return;
      }
      const amd::NDRange32& global = ndrange_.global();
      const amd::NDRange16& local = ndrange_.local();
      grid_ = amd::NDRange32(global[0] / local[0], global[1] / local[1], global[2] / local[2]);
    }

    if (clusterDim.x > 1 || clusterDim.y > 1 || clusterDim.z > 1) {
      if (!IsClusterDivisible(grid_, clusterDim)) {
        violations_ |= kClusterIndivisible;
      }
    }
  }

  //! Map the violations found at construction through this entry point's rule table.
  template <size_t N> hipError_t Status(const LaunchErrorRule (&rules)[N]) const {
    return MapLaunchViolations(violations_, rules);
  }

  //! As Status(), but re-derives the device-dependent bits if \a device is not the build device.
  template <size_t N>
  hipError_t StatusForDevice(const amd::Device& device, const LaunchErrorRule (&rules)[N]) const {
    LaunchViolationBits violations = violations_;
    if (&device != device_) {
      violations = (violations & ~kDeviceDependentBits) |
          CheckNDRangeAgainstDevice(ndrange_.global(), ndrange_.local(), sharedMemBytes_, device);
    }
    return MapLaunchViolations(violations, rules);
  }

  //! Shared memory narrowed for the AQL packet. Only valid once kSharedMemOverflow is ruled out.
  uint32_t sharedMemBytes32() const { return static_cast<uint32_t>(sharedMemBytes_); }
};

//! Build HIP style, where the app supplies the grid and block and the global is computed.
inline LaunchConfig MakeLaunchConfigFromGrid(const dim3& gridDim, const dim3& blockDim,
                                             size_t sharedMemBytes, const amd::Device& device,
                                             const dim3& clusterDim = kNoCluster,
                                             const dim3& remainder = kNoRemainder) {
  return LaunchConfig(amd::NDRange(static_cast<size_t>(gridDim.x) * blockDim.x + remainder.x,
                                   static_cast<size_t>(gridDim.y) * blockDim.y + remainder.y,
                                   static_cast<size_t>(gridDim.z) * blockDim.z + remainder.z),
                      amd::NDRange32(gridDim.x, gridDim.y, gridDim.z), blockDim, clusterDim,
                      sharedMemBytes, device, false /*deduceGrid*/);
}

//! Build from an app supplied global and local size; the grid (total blocks) is deduced.
inline LaunchConfig MakeLaunchConfigFromGlobal(const dim3& globalDim, const dim3& localDim,
                                               size_t sharedMemBytes, const amd::Device& device,
                                               const dim3& clusterDim = kNoCluster) {
  return LaunchConfig(amd::NDRange(globalDim.x, globalDim.y, globalDim.z),
                      amd::NDRange32(1, 1, 1), localDim, clusterDim, sharedMemBytes, device,
                      true /*deduceGrid*/);
}

}  // namespace hip

#endif  // HIP_LAUNCH_VALIDATION_HPP_
