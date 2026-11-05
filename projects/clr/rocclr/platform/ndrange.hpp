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

  static bool CanSafelyNarrow(size_t x, size_t y, size_t z) {
    return (x <= std::numeric_limits<T>::max() && y <= std::numeric_limits<T>::max() &&
            z <= std::numeric_limits<T>::max());
  }
};

using NDRange = NDRangeImpl<size_t>;      //!< Default index space (size_t elements)
using NDRange32 = NDRangeImpl<uint32_t>;  //!< AQL grid_size_{x,y,z}
using NDRange16 = NDRangeImpl<uint16_t>;  //!< AQL workgroup_size_{x,y,z}
using NDRange8 = NDRangeImpl<uint8_t>;    //!< AQL cluster_size_{x,y,z}

//! A container for the local and global worksizes.
class NDRangeContainer {
 private:
  NDRange offset_;       //!< Global work-item offset (size_t — passed as-is to hidden args).
  NDRange32 global_;     //!< Total number of work-items in N-dims (AQL grid_size).
  NDRange16 local_;      //!< Number of work-items per workgroup (AQL workgroup_size).
  NDRange8 cluster_;     //!< Cluster dims (AQL cluster_size, max 255 per dim).
  uint8_t dimensions_;   //!< Number of dimensions (1, 2, or 3).

 public:
  //! From already narrowed typed index spaces.
  NDRangeContainer(size_t dimensions, const NDRange& globalWorkOffset, const NDRange32& global,
                   const NDRange16& local, const NDRange8& cluster)
      : offset_(globalWorkOffset),
        global_(global),
        local_(local),
        cluster_(cluster),
        dimensions_(static_cast<uint8_t>(dimensions)) {
    assert(dimensions_ >= 1 && dimensions_ <= 3 && "Dimensions must be 1, 2, or 3");
  }

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

  //! Return the number of dimensions.
  size_t dimensions() const { return dimensions_; }

  const NDRange& offset() const { return offset_; }
  const NDRange32& global() const { return global_; }
  const NDRange16& local() const { return local_; }
  const NDRange8& cluster() const { return cluster_; }

  //! Shrink the workgroup in one dim (used to clamp local to global on a HIP launch).
  void setLocal(size_t dim, uint16_t value) { local_[dim] = value; }

  //! Override the cluster dims (used when the kernel metadata carries its own cluster size).
  void setCluster(const NDRange8& cluster) { cluster_ = cluster; }
};

static_assert(sizeof(NDRangeContainer) <= 64,
              "Aim to keep NDRangeContainer under half a cache line sizes");


/*! @}\
 *  @}
 */

}  // namespace amd

#endif /*NDRANGE_HPP_*/
