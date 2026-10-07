/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef NDRANGE_HPP_
#define NDRANGE_HPP_

#include "top.hpp"

#include <limits>

namespace amd {
class Device;
}  // namespace amd

namespace amd {

/*! \addtogroup Runtime
 *  @{
 *
 *  \addtogroup Program Programs and Kernel functions
 *  @{
 */

//! A fixed 3-element typed index space (no dimension tracking — NDRangeContainer owns dims).
template <typename T = size_t> class NDRangeImpl : public EmbeddedObject {
 private:
  T data_[3];  //!< indexes array

 public:
  NDRangeImpl(T dataX, T dataY, T dataZ) {
    data_[0] = dataX;
    data_[1] = dataY;
    data_[2] = dataZ;
  }

  //! Copy constructor.
  NDRangeImpl(const NDRangeImpl& space) { *this = space; }

  //! Converting copy constructor — widens or narrows element type.
  template <typename U> NDRangeImpl(const NDRangeImpl<U>& space) {
    for (size_t i = 0; i < 3; ++i) data_[i] = static_cast<T>(space[i]);
  }

  //! Copy operator.
  NDRangeImpl& operator=(const NDRangeImpl& space) {
    data_[0] = space.data_[0];
    data_[1] = space.data_[1];
    data_[2] = space.data_[2];
    return *this;
  }

  //! Return the element at the given \a index.
  T& operator[](size_t index) {
    assert(index < 3 && "Index overflows data_");
    return data_[index];
  }

  //! Return the element at the given \a index.
  T operator[](size_t index) const {
    assert(index < 3 && "Index overflows data_");
    return data_[index];
  }

  //! Return the product of all three elements (unused dims must be set to 1 by caller).
  //! product returns size_t to avoid overflows with narrow dtypes
  size_t product() const {
    return static_cast<size_t>(data_[0]) * static_cast<size_t>(data_[1]) *
           static_cast<size_t>(data_[2]);
  }

  //! Return true if this index space is identical to \a x.
  bool operator==(const NDRangeImpl& x) const {
    return data_[0] == x.data_[0] && data_[1] == x.data_[1] && data_[2] == x.data_[2];
  }

  //! Return true if this index space and \a x are different.
  bool operator!=(const NDRangeImpl& x) const { return !(*this == x); }

  const T* Data() const { return data_; }

  static bool CanSafelyNarrow(size_t x, size_t y, size_t z) {
    return (x <= std::numeric_limits<T>::max() && y <= std::numeric_limits<T>::max() &&
            z <= std::numeric_limits<T>::max());
  }
};

using NDRange = NDRangeImpl<size_t>;      //!< Default index space (size_t elements)
using NDRange32 = NDRangeImpl<uint32_t>;  //!< AQL grid_size_{x,y,z}
using NDRange16 = NDRangeImpl<uint16_t>;  //!< AQL workgroup_size_{x,y,z}
using NDRange8 = NDRangeImpl<uint8_t>;    //!< AQL cluster_size_{x,y,z}

//! Stucture to store launch parameters.
struct LaunchParams {
  NDRange32 global_;         //!< Total number of work-items in N-dims (matches AQL grid_size)
  NDRange16 local_;          //!< Number of work-items per workgroup (matches AQL workgroup_size)
  NDRange8 cluster_;         //!< Cluster dims (matches AQL cluster_size, max 255)
  NDRange32 grid_;           //!< Total number of workgroups in grid in N-dims
  uint32_t sharedMemBytes_;  //!< Shared Memory bytes
  bool hipParams_;           //!< If this is launched through hipParams_
  bool validConfig_;         //!< Flag will be set to false when config is not correct.

  LaunchParams(size_t globalX, size_t globalY, size_t globalZ, uint32_t localX, uint32_t localY,
               uint32_t localZ, uint32_t sharedMemBytes, const Device& device,
               uint32_t clusterX = 1, uint32_t clusterY = 1, uint32_t clusterZ = 1,
               uint32_t gridX = 1, uint32_t gridY = 1, uint32_t gridZ = 1, bool hipParams = false)
      : global_(static_cast<uint32_t>(globalX), static_cast<uint32_t>(globalY),
                static_cast<uint32_t>(globalZ)),
        local_(static_cast<uint16_t>(localX), static_cast<uint16_t>(localY),
               static_cast<uint16_t>(localZ)),
        cluster_(static_cast<uint8_t>(clusterX), static_cast<uint8_t>(clusterY),
                 static_cast<uint8_t>(clusterZ)),
        grid_(gridX, gridY, gridZ),
        sharedMemBytes_(sharedMemBytes),
        hipParams_(hipParams),
        validConfig_(true) {
    if (!NDRange8::CanSafelyNarrow(clusterX, clusterY, clusterZ)) {
      validConfig_ = false;
    }

    if (!NDRange16::CanSafelyNarrow(localX, localY, localZ)) {
      validConfig_ = false;
    }

    if (hipParams_) {
      // Check that the size_t globals fit in uint32_t before the narrowing cast above.
      if (!NDRange32::CanSafelyNarrow(globalX, globalY, globalZ)) {
        validConfig_ = false;
      }
    } else {
      // Non HIPLaunchParams, App directly calculated the global and local size,
      // manually deduce the grid (total blocks) size.
      if (local_[0] == 0 || local_[1] == 0 || local_[2] == 0) {
        validConfig_ = false;
        return;
      }
      grid_[0] = global_[0] / local_[0];
      grid_[1] = global_[1] / local_[1];
      grid_[2] = global_[2] / local_[2];
    }

    // If cluster parameters is set, then check if it is divisble by grid (total blocks).
    if (clusterX > 1 || clusterY > 1 || clusterZ > 1) {
      if (!CheckClusterDivisibility(clusterX, clusterY, clusterZ)) {
        validConfig_ = false;
      }
    }
  }

  bool CheckClusterDivisibility(uint32_t clusterX, uint32_t clusterY, uint32_t clusterZ) {
    // With cluster launch, the total number of blocks or threads the work is launched doesnt
    // change, except that the work is launch into different CU/WGP's under the same shader engine.
    // So, the grid values are basically split among different CUs based on cluster dims, hence
    // grid dims has to be divisble by cluster dims.
    if ((grid_[0] % clusterX != 0) || (grid_[1] % clusterY != 0) || (grid_[2] % clusterZ != 0)) {
      return false;
    }
    return true;
  }

  //! Sometimes we receive cluster launch info from kernel, not through HIP launch kernel APIs.
  bool UpdateClusterLaunchParams(uint32_t clusterX, uint32_t clusterY, uint32_t clusterZ) {
    // If cluster parameters are not > 1, we dont need to update since it is the default value set.
    if (clusterX > 1 || clusterY > 1 || clusterZ > 1) {
      if (!CheckClusterDivisibility(clusterX, clusterY, clusterZ)) {
        return false;
      }
      cluster_[0] = static_cast<uint8_t>(clusterX);
      cluster_[1] = static_cast<uint8_t>(clusterY);
      cluster_[2] = static_cast<uint8_t>(clusterZ);
    }
    return true;
  }

  bool IsValidConfig() const { return validConfig_; }
};

//! Structure to store launch parameters in HIP Style (global and local size needs computation).
struct HIPLaunchParams : public LaunchParams {

  HIPLaunchParams(uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                  uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes, const Device& device,
                  uint32_t globalX_remainder = 0, uint32_t globalY_remainder = 0,
                  uint32_t globalZ_remainder = 0, uint32_t clusterX = 1,
                  uint32_t clusterY = 1, uint32_t clusterZ = 1)
                  : LaunchParams(static_cast<size_t>(gridX) * blockX + globalX_remainder,
                                 static_cast<size_t>(gridY) * blockY + globalY_remainder,
                                 static_cast<size_t>(gridZ) * blockZ + globalZ_remainder,
                                 blockX, blockY, blockZ, sharedMemBytes, device, clusterX, clusterY,
                                 clusterZ, gridX, gridY, gridZ, true /*hipParams*/) {}
};

//! A container for the local and global worksizes.
class NDRangeContainer {
 private:
  NDRange offset_;       //!< Global work-item offset (size_t — passed as-is to hidden args).
  NDRange32 global_;     //!< Total number of work-items in N-dims (AQL grid_size).
  NDRange16 local_;      //!< Number of work-items per workgroup (AQL workgroup_size).
  NDRange8 cluster_;     //!< Cluster dims (AQL cluster_size, max 255 per dim).
  uint8_t dimensions_;   //!< Number of dimensions (1, 2, or 3).

 public:
  //! From size_t arrays (blit, OCL, devprogram callers — no cluster).
  NDRangeContainer(size_t dimensions, const size_t* globalWorkOffset, const size_t* globalWorkSize,
                   const size_t* localWorkSize)
      : offset_(0, 0, 0),
        global_(1, 1, 1),
        local_(1, 1, 1),
        cluster_(1, 1, 1),
        dimensions_(static_cast<uint8_t>(dimensions)) {
    assert(dimensions_ >= 1 && dimensions_ <= 3 && "Dimensions must be 1, 2, or 3");
    for (size_t i = 0; i < dimensions; ++i) {
      offset_[i] = globalWorkOffset != nullptr ? globalWorkOffset[i] : 0;
      global_[i] = static_cast<uint32_t>(globalWorkSize[i]);
      local_[i] = static_cast<uint16_t>(localWorkSize[i]);
    }
  }

  //! From typed NDRange arrays (LaunchParams::Data() callers — includes cluster).
  NDRangeContainer(size_t dimensions, const size_t* globalWorkOffset,
                   const uint32_t* globalWorkSize, const uint16_t* localWorkSize,
                   const uint8_t* clusterWorkSize)
      : offset_(0, 0, 0),
        global_(1, 1, 1),
        local_(1, 1, 1),
        cluster_(1, 1, 1),
        dimensions_(static_cast<uint8_t>(dimensions)) {
    assert(dimensions_ >= 1 && dimensions_ <= 3 && "Dimensions must be 1, 2, or 3");
    for (size_t i = 0; i < dimensions; ++i) {
      offset_[i] = globalWorkOffset != nullptr ? globalWorkOffset[i] : 0;
      global_[i] = globalWorkSize[i];
      local_[i] = localWorkSize[i];
      cluster_[i] = clusterWorkSize[i];
    }
  }

  //! Return the number of dimensions.
  size_t dimensions() const { return dimensions_; }

  const NDRange& offset() const { return offset_; }
  const NDRange32& global() const { return global_; }
  const NDRange16& local() const { return local_; }
  const NDRange8& cluster() const { return cluster_; }
};

static_assert(sizeof(NDRangeContainer) <= 64,
              "Aim to keep NDRangeContainer under half a cache line sizes");


/*! @}\
 *  @}
 */

}  // namespace amd

#endif /*NDRANGE_HPP_*/
