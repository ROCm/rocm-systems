/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "platform/ndrange.hpp"

#include "device/devkernel.hpp"

namespace amd {

NDRange::NDRange(size_t dimensions) : dimensions_(dimensions) { *this = 0; }

NDRange::NDRange(const NDRange& space) : dimensions_(space.dimensions_) { *this = space; }

NDRange& NDRange::operator=(size_t x) {
  for (size_t i = 0; i < dimensions_; ++i) {
    data_[i] = x;
  }
  return *this;
}

NDRange::~NDRange() {}

bool NDRangeContainer::UpdateNumClustersFromKernel(const device::Kernel* devKernel) {
  // Cluster dimensions may be specified in the kernel's metadata rather than through the
  // launch API. Only act when the kernel requests a real cluster (> 1 in any dimension).
  const size_t clusterX = devKernel->getClusterSize(0);
  const size_t clusterY = devKernel->getClusterSize(1);
  const size_t clusterZ = devKernel->getClusterSize(2);
  if (clusterX <= 1 && clusterY <= 1 && clusterZ <= 1) {
    return true;
  }
  // The grid (total number of workgroups) is global / local. It must be divisible by the
  // cluster dimensions, otherwise the work cannot be split evenly across the cluster.
  const size_t gridX = global_[0] / local_[0];
  const size_t gridY = global_[1] / local_[1];
  const size_t gridZ = global_[2] / local_[2];
  if ((gridX % clusterX != 0) || (gridY % clusterY != 0) || (gridZ % clusterZ != 0)) {
    return false;
  }
  cluster_[0] = clusterX;
  cluster_[1] = clusterY;
  cluster_[2] = clusterZ;
  return true;
}

bool NDRange::operator==(const NDRange& x) const {
  assert(dimensions_ == x.dimensions_ && "dimensions mismatch");

  for (size_t i = 0; i < dimensions_; ++i) {
    if (data_[i] != x.data_[i]) {
      return false;
    }
  }
  return true;
}

bool NDRange::operator==(size_t x) const {
  for (size_t i = 0; i < dimensions_; ++i) {
    if (data_[i] != x) {
      return false;
    }
  }
  return true;
}

#ifdef DEBUG
void NDRange::printOn(FILE* file) const {
  fprintf(file, "[");
  for (size_t i = dimensions_ - 1; i > 0; --i) {
    fprintf(file, SIZE_T_FMT ", ", data_[i]);
  }
  fprintf(file, SIZE_T_FMT "]", data_[0]);
}
#endif  // DEBUG

}  // namespace amd
