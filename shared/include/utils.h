////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef SHARED_UTILS_H
#define SHARED_UTILS_H

#include <cstddef>
#include <cstdint>

#define DISABLE_COPY_AND_ASSIGN(Cls)    \
  Cls(const Cls&) = delete;             \
  Cls& operator=(const Cls&) = delete;  \
  Cls(Cls &&) = delete;                 \
  Cls& operator=(Cls &&) = delete;

template<typename T>
inline const T *ptr_inc(const T* ptr, size_t sz) {
  return reinterpret_cast<const T *>(reinterpret_cast<const std::byte *>(ptr) + sz);
}

template<typename T>
inline T *ptr_inc(T* ptr, size_t sz) {
  return reinterpret_cast<T *>(reinterpret_cast<std::byte *>(ptr) + sz);
}


namespace wsl {
namespace thunk {

struct GfxipTable {
  uint16_t device_id;
  uint8_t  major;
  uint8_t  minor;
  uint8_t  stepping;
};


bool QueryAdapterSupported(unsigned int device_id);
bool LookupGfxipEntry(uint16_t device_id, GfxipTable *out);

} // namespace thunk
} // namespace wsl
#endif
