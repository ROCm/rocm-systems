/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef NDRANGE_HPP_
#define NDRANGE_HPP_

#include "top.hpp"

#include <valarray>

#ifdef DEBUG
#include <cstdio>
#endif  // DEBUG

namespace amd {

/*! \addtogroup Runtime
 *  @{
 *
 *  \addtogroup Program Programs and Kernel functions
 *  @{
 */

//! An N-dimensions index space.
class NDRange : public EmbeddedObject {
 private:
  const size_t dimensions_ : 2;  //!< Number of dimensions [0-3]
  size_t data_[3];               //!< indexes array

 private:
  //! Construct a new index space for an array of elements (no-copy)
  NDRange(size_t dimensions, size_t* elements) : dimensions_(dimensions) {
    for (uint i = 0; i < dimensions_; ++i) {
      data_[i] = elements[i];
    }
  }

 public:
  //! Construct a new index space of the given dimensions.
  explicit NDRange(size_t dimensions);

  NDRange(size_t dataX, size_t dataY, size_t dataZ) : dimensions_(3) {
    data_[0] = dataX;
    data_[1] = dataY;
    data_[2] = dataZ;
  }

  //! Copy constructor.
  NDRange(const NDRange& space);

  //! Destroy the index space.
  ~NDRange();

  //! Copy operator
  inline NDRange& operator=(const NDRange& space);

  //! Make all elements of this space equal to x.
  NDRange& operator=(size_t x);

  //! Return the number of dimensions.
  size_t dimensions() const { return dimensions_; }

  //! Return the element at the given \a index.
  size_t& operator[](size_t index) {
    assert(index < dimensions_ && "index is out of bounds");
    return data_[index];
  }

  //! Return the element at the given \a index.
  size_t operator[](size_t index) const {
    assert(index < dimensions_ && "index is out of bounds");
    return data_[index];
  }

  //! Return the sum of this index space elements.
  inline size_t sum() const;

  //! Return the product of this index space elements (size)
  inline size_t product() const;

  // Binary operators:
  inline friend NDRange operator+(const NDRange& x, const NDRange& y);
  inline friend NDRange operator-(const NDRange& x, const NDRange& y);
  inline friend NDRange operator*(const NDRange& x, const NDRange& y);
  inline friend NDRange operator/(const NDRange& x, const NDRange& y);
  inline friend NDRange operator%(const NDRange& x, const NDRange& y);

  //! Return true if this index space is identical to \a x.
  bool operator==(const NDRange& x) const;

  //! Return true if this index space and \a x are different.
  bool operator!=(const NDRange& x) const { return !(*this == x); }

  //! Return true if all elements are equal to \a x.
  bool operator==(size_t x) const;

  //! Return true if one element of this space is not equal to \a x.
  bool operator!=(size_t x) const { return !(*this == x); }

#ifdef DEBUG
  //! Print this index space on the given stream.
  void printOn(FILE* file) const;
#endif  // DEBUG

  const size_t* Data() const { return data_; }
};

//! Stucture to store launch parameters.
struct LaunchParams {
  NDRange global_;           //!< Total number of work-items in N-dims
  NDRange local_;            //!< Number of work-items in N-dims in a workgroup.
  uint32_t sharedMemBytes_;  //!< Shared Memory bytes
  NDRange cluster_;          //!< Total number of clusters in N-dims
  NDRange grid_;             //!< Total number of workgroups in grid in N-dims.
  bool hipParams_;           //!< If this is launched through hipParams_
  bool validConfig_;         //!< Flag will be set to false when config is not correct.

  LaunchParams(size_t globalX, size_t globalY, size_t globalZ, uint32_t localX,
               uint32_t localY, uint32_t localZ, uint32_t sharedMemBytes, uint32_t clusterX = 1,
               uint32_t clusterY = 1, uint32_t clusterZ = 1, uint32_t gridX = 1, uint32_t gridY = 1,
               uint32_t gridZ = 1, bool hipParams = false) : global_(globalX, globalY, globalZ),
               local_(localX, localY, localZ), sharedMemBytes_ (sharedMemBytes),
               cluster_(clusterX, clusterY, clusterZ), grid_(gridX, gridY, gridZ),
               hipParams_(hipParams), validConfig_(true) {

    if (hipParams_) {
      // if this is launched through HIPLaunchParams, then we need to check the global does not
      // take up more than 32 bits, the max we are allowed to launch in the backend.
      if (global_[0] > std::numeric_limits<uint32_t>::max()
          || global_[1] > std::numeric_limits<uint32_t>::max()
          || global_[2] > std::numeric_limits<uint32_t>::max()) {
          validConfig_ = false;
      }
    } else {
      // Non HIPLaunchParams, App directly calculated the global and local size,
      // manually deduce the grid (total blocks) size.
      if (local_[0] == 0 || local_[1] == 0 ||local_[2] == 0) {
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
      cluster_[0] = clusterX; cluster_[1] = clusterY; cluster_[2] = clusterZ;
    }
    return true;
  }

  bool IsValidConfig() const { return validConfig_; }
};

//! Structure to store launch parameters in HIP Style (global and local size needs computation).
struct HIPLaunchParams : public LaunchParams {

  HIPLaunchParams(uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                  uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
                  uint32_t globalX_remainder = 0, uint32_t globalY_remainder = 0,
                  uint32_t globalZ_remainder = 0, uint32_t clusterX = 1,
                  uint32_t clusterY = 1, uint32_t clusterZ = 1)
                  : LaunchParams(static_cast<uint32_t>(gridX) * blockX + globalX_remainder,
                                 static_cast<uint32_t>(gridY) * blockY + globalY_remainder,
                                 static_cast<uint32_t>(gridZ) * blockZ + globalZ_remainder,
                                 blockX, blockY, blockZ, sharedMemBytes, clusterX, clusterY,
                                 clusterZ, gridX, gridY, gridZ, true /*hipParams*/) {}
};

//! A container for the local and global worksizes.
class NDRangeContainer : public HeapObject {
 private:
  const size_t dimensions_;  //!< Number of dimensions.
  NDRange offset_;           //!< Global work-item offset.
  NDRange global_;           //!< Total number of work-items in N-dims
  NDRange local_;            //!< Number of work-items in N-dims in a workgroup.
  NDRange cluster_;          //!< Number of Cluster in N-dims across work group.

 public:
  /*! \brief Construct a new nd-range container with the given local
   *  and global worksizes in \a nDimensions dimensions.
   */
  NDRangeContainer(size_t dimensions, const size_t* globalWorkOffset, const size_t* globalWorkSize,
                   const size_t* localWorkSize, const size_t* clusterWorkSize = nullptr)
      : dimensions_(dimensions), offset_(dimensions), global_(dimensions), local_(dimensions), 
        cluster_(dimensions) {
    for (size_t i = 0; i < dimensions; ++i) {
      offset_[i] = globalWorkOffset != NULL ? globalWorkOffset[i] : 0;
      global_[i] = globalWorkSize[i];
      local_[i] = localWorkSize[i];
      cluster_[i] = clusterWorkSize != nullptr ? clusterWorkSize[i] : 1;
    }
  }

  //! updates nd-range container
  void update(size_t dimensions, const size_t* globalWorkOffset, const size_t* globalWorkSize,
              const size_t* localWorkSize) {
    for (size_t i = 0; i < dimensions; ++i) {
      offset_[i] = globalWorkOffset != NULL ? globalWorkOffset[i] : 0;
      global_[i] = globalWorkSize[i];
      local_[i] = localWorkSize[i];
    }
  }

  //! Return the number of dimensions.
  size_t dimensions() const { return dimensions_; }

  //! Return the global workoffset.
  const NDRange& offset() const { return offset_; }
  NDRange& offset() { return offset_; }
  //! Return the global worksize.
  const NDRange& global() const { return global_; }
  NDRange& global() { return global_; }
  //! Return the local worksize.
  const NDRange& local() const { return local_; }
  NDRange& local() { return local_; }
  //! Return the cluster worksize.
  const NDRange& cluster() const { return cluster_; }
  NDRange& cluster() { return cluster_; }
};


/*! @}\
 *  @}
 */

inline size_t NDRange::sum() const {
  size_t result = data_[0];
  for (size_t i = 1; i < dimensions_; ++i) {
    result += data_[i];
  }
  return result;
}

inline size_t NDRange::product() const {
  size_t result = data_[0];
  for (size_t i = 1; i < dimensions_; ++i) {
    result *= data_[i];
  }
  return result;
}

// This function is in this header file for performance improvements:
inline NDRange& NDRange::operator=(const NDRange& space) {
  assert(dimensions_ == space.dimensions_ && "dimensions mismatch");
  for (size_t i = 0; i < sizeof(data_) / sizeof(*data_); ++i) {
    data_[i] = space.data_[i];
  }
  return *this;
}

#define DEFINE_NDRANGE_BINARY_OP(op)                                                               \
  inline NDRange operator op(const NDRange& x, const NDRange& y) {                                 \
    assert(x.dimensions_ == y.dimensions_ && "dimensions mismatch");                               \
                                                                                                   \
    size_t dimensions = x.dimensions_;                                                             \
    size_t result[3] = {0};                                                                        \
    for (size_t i = 0; i < dimensions; ++i) {                                                      \
      result[i] = x.data_[i] op y.data_[i];                                                        \
    }                                                                                              \
                                                                                                   \
    return NDRange(dimensions, &result[0]);                                                        \
  }

DEFINE_NDRANGE_BINARY_OP(+);
DEFINE_NDRANGE_BINARY_OP(-);
DEFINE_NDRANGE_BINARY_OP(*);
DEFINE_NDRANGE_BINARY_OP(/);
DEFINE_NDRANGE_BINARY_OP(%);

#undef DEFINE_NDRANGE_BINARY_OP

}  // namespace amd

#endif /*NDRANGE_HPP_*/
