/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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
