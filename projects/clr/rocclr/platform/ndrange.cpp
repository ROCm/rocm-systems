/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "platform/ndrange.hpp"

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
