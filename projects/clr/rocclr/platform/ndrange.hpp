/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef NDRANGE_HPP_
#define NDRANGE_HPP_

#include "top.hpp"
#include "device/device.hpp"

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

//! Reason a launch configuration was rejected. Detection is centralized, but the
//! hipError_t a given reason maps to is a property of the calling entry point, so
//! callers translate this reason into their own error code (see reintroduce_pr3151_plan.md).
enum class LaunchConfigStatus : uint32_t {
  kOk = 0,
  kZeroGrid,            //!< a global/grid dimension is zero
  kZeroBlock,           //!< a local/block dimension is zero
  kGlobalOverflow,      //!< HIP style: grid*block exceeds the 32-bit backend limit
  kBlockExceedsMaxWG,   //!< block work-item count exceeds the device max workgroup size
  kSharedMemExceeded,   //!< requested shared memory exceeds the device per-CU limit
  kClusterIndivisible,  //!< grid is not divisible by the requested cluster dimensions
};

//! A container for the local and global worksizes.
class NDRangeContainer {
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

  //! Fold cluster dimensions carried by the kernel's metadata into this container.
  //! When the kernel specifies a cluster size (> 1), the grid (global/local) must be
  //! divisible by it; on success the cluster dims are updated. Returns false if the
  //! grid is not divisible by the kernel's cluster dimensions (invalid launch).
  bool UpdateNumClustersFromKernel(const device::Kernel* devKernel);
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
