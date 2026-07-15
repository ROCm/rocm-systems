/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef NDRANGE_CONTAINER_HPP_
#define NDRANGE_CONTAINER_HPP_

#include "top.hpp"
#include "device/device.hpp"

#include <cassert>
#include <limits>
#include <type_traits>

namespace amd {

/*! \addtogroup Runtime
 *  @{
 *
 *  \addtogroup Program Programs and Kernel functions
 *  @{
 */

//! A container for the local and global worksizes.
class NDRangeContainer {
 private:
  // Types directly map to AQL dispatch size, do not alter unless AQL changes types
  size_t offset_[3];
  // uint32 to match AQL grid_size; callers must bound inputs to uint32
  // enforced by MakeLaunchFrom* factories and the OpenCL boundary check.
  uint32_t global_[3];
  uint16_t local_[3];
  uint16_t dimensions_;
  // AQL cluster_size_{x,y,z} fields are uint8_t, so a cluster dimension can never
  // exceed 255 workgroups. Store narrow; UpdateCluster rejects anything larger.
  uint8_t cluster_[3];

 public:
  /*! \brief Construct a new nd-range container with the given local
   *  and global worksizes in \a dimensions dimensions.
   */
  NDRangeContainer(size_t dimensions, const size_t* globalWorkOffset, const size_t* globalWorkSize,
                   const size_t* localWorkSize, const size_t* clusterWorkSize = nullptr) noexcept
      : offset_{0, 0, 0},
        global_{1, 1, 1},
        local_{1, 1, 1},
        dimensions_(static_cast<uint16_t>(dimensions)),
        cluster_{0, 0, 0} {
    for (uint16_t i = 0; i < dimensions_; ++i) {
      offset_[i] = globalWorkOffset != nullptr ? globalWorkOffset[i] : 0;
      global_[i] = static_cast<uint32_t>(globalWorkSize[i]);
      assert(localWorkSize[i] <= std::numeric_limits<uint16_t>::max());
      local_[i] = static_cast<uint16_t>(localWorkSize[i]);
      cluster_[i] = clusterWorkSize != nullptr ? static_cast<uint8_t>(clusterWorkSize[i]) : 1;
    }
  }

  //! Return the number of dimensions.
  uint16_t get_dimensions() const { return dimensions_; }

  //! Per-element getters (no NDRange copies).
  size_t get_offset(uint16_t dim) const { return offset_[dim]; }
  uint32_t get_global(uint16_t dim) const { return global_[dim]; }
  uint16_t get_local(uint16_t dim) const { return local_[dim]; }
  uint32_t get_cluster(uint16_t dim) const { return cluster_[dim]; }
  uint32_t get_block_count(uint16_t dim) const {
    assert(local_[dim] > 0 && "NDRangeContainer must be in a valid state prior to this query");
    return global_[dim] / local_[dim];
  }
  uint16_t get_block_remainder(uint16_t dim) const {
    assert(local_[dim] > 0 && "NDRangeContainer must be in a valid state prior to this query");
    return static_cast<uint16_t>(global_[dim] % local_[dim]);
  }

  //! Set the local work size for a dimension. Used in OpenCL
  //! to replace null local sizes and apply required work groups
  void set_local(uint16_t dim, uint16_t value) { local_[dim] = value; }

  //! Product of local work sizes across all dimensions, accumulated in size_t to avoid wrap.
  size_t get_local_group_size() const {
    size_t result = 1;
    for (uint16_t i = 0; i < dimensions_; ++i) {
      result *= static_cast<size_t>(local_[i]);
    }
    return result;
  }

  //! Product of global work sizes across all dimensions, accumulated in size_t to avoid wrap.
  size_t get_global_group_size() const {
    size_t result = 1;
    for (uint16_t i = 0; i < dimensions_; ++i) {
      result *= static_cast<size_t>(global_[i]);
    }
    return result;
  }

  //! Shared cluster helper used by factories and UpdateNumClustersFromKernel.
  //! For each dim where the cluster extent > 1, derives grid = global/local and
  //! requires grid % cluster == 0. No-op (returns true) when all cluster extents are 1.
  //! On success stores the cluster extents and returns true; returns false on bad
  //! divisibility or when a cluster extent exceeds the uint8_t AQL packet field width.
  bool UpdateCluster(uint32_t cx, uint32_t cy, uint32_t cz) {
    if (cx <= 1 && cy <= 1 && cz <= 1) {
      return true;
    }
    // AQL cluster_size fields are uint8_t; reject anything the packet cannot represent.
    constexpr uint32_t kMaxClusterDim = std::numeric_limits<uint8_t>::max();
    if (cx > kMaxClusterDim || cy > kMaxClusterDim || cz > kMaxClusterDim) {
      return false;
    }
    const uint32_t c[3] = {cx, cy, cz};
    for (uint16_t i = 0; i < dimensions_; ++i) {
      if (c[i] <= 1) {
        continue;
      }
      // local_ is non-zero here: MakeLaunchFromGlobalLocal rejects zero local before
      // calling UpdateCluster, and MakeLaunchFromGrid constructs global from grid*block
      // so block (==local_) is the caller-supplied blockDim which is trusted non-zero
      // when a cluster dim > 1 is requested.
      uint32_t grid = global_[i] / static_cast<uint32_t>(local_[i]);
      if (grid % c[i] != 0) {
        return false;
      }
    }
    cluster_[0] = static_cast<uint8_t>(cx);
    cluster_[1] = static_cast<uint8_t>(cy);
    cluster_[2] = static_cast<uint8_t>(cz);
    return true;
  }
};

static_assert(sizeof(NDRangeContainer) <= 64,
              "Aim to keep NDRangeContainer under half a cache line sizes");
static_assert(std::is_trivially_copyable<NDRangeContainer>::value,
              "NDRangeContainer should stay trivially copyable");

//! Factory: HIP grid+block convention.  global = grid*block + remainder.
//! Sets valid=false if any global dim exceeds uint32 max (checked in size_t before
//! narrowing) or a supplied cluster dim does not evenly divide the block-count grid.
inline NDRangeContainer MakeLaunchFromGrid(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                                           uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                                           uint32_t remX, uint32_t remY, uint32_t remZ,
                                           uint32_t clusterX, uint32_t clusterY, uint32_t clusterZ,
                                           bool& valid) {
  valid = true;

  const size_t gX = static_cast<size_t>(gridX) * blockX + remX;
  const size_t gY = static_cast<size_t>(gridY) * blockY + remY;
  const size_t gZ = static_cast<size_t>(gridZ) * blockZ + remZ;

  if (gX > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      gY > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      gZ > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    valid = false;
  }

  const size_t globalWorkSize[3] = {gX, gY, gZ};
  const size_t localWorkSize[3] = {blockX, blockY, blockZ};
  NDRangeContainer c(3, nullptr, globalWorkSize, localWorkSize, nullptr);

  if (valid) {
    if (!c.UpdateCluster(clusterX, clusterY, clusterZ)) {
      valid = false;
    }
  }
  return c;
}

//! Factory: direct global+local convention.  Inputs are uint32 so global cannot overflow.
//! Sets valid=false if any local dim is zero or a supplied cluster dim does not evenly
//! divide the block-count grid (= global/local).
inline NDRangeContainer MakeLaunchFromGlobalLocal(uint32_t globalX, uint32_t globalY,
                                                  uint32_t globalZ, uint32_t localX,
                                                  uint32_t localY, uint32_t localZ,
                                                  uint32_t clusterX, uint32_t clusterY,
                                                  uint32_t clusterZ, bool& valid) {
  valid = true;

  if (localX == 0 || localY == 0 || localZ == 0) {
    valid = false;
  }

  const size_t globalWorkSize[3] = {globalX, globalY, globalZ};
  const size_t localWorkSize[3] = {localX, localY, localZ};
  NDRangeContainer c(3, nullptr, globalWorkSize, localWorkSize, nullptr);

  if (valid) {
    if (!c.UpdateCluster(clusterX, clusterY, clusterZ)) {
      valid = false;
    }
  }
  return c;
}

/*! @}\
 *  @}
 */

}  // namespace amd

#endif /*NDRANGE_CONTAINER_HPP_*/
