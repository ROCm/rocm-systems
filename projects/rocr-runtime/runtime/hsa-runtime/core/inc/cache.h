////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HSA_RUNTIME_CORE_INC_CACHE_H
#define HSA_RUNTIME_CORE_INC_CACHE_H

#include "core/inc/hsa_internal.h"
#include "core/inc/checked.h"
#include "core/util/utils.h"
#include <utility>
#include <string>

namespace rocr {
namespace core {

class Cache : public Checked<0x39A6C7AD3F135B06> {
 public:
  static __forceinline hsa_cache_t Convert(const Cache* cache) {
    const hsa_cache_t handle = {static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cache))};
    return handle;
  }
  static __forceinline Cache* Convert(const hsa_cache_t cache) {
    return reinterpret_cast<Cache*>(static_cast<uintptr_t>(cache.handle));
  }

  Cache(const std::string& name, uint8_t level, uint32_t size)
      : name_(name), level_(level), size_(size) {}

  Cache(std::string&& name, uint8_t level, uint32_t size)
      : name_(std::move(name)), level_(level), size_(size) {}

  hsa_status_t GetInfo(hsa_cache_info_t attribute, void* value);

 private:
  std::string name_;
  uint32_t level_;
  uint32_t size_;

  // Forbid copying and moving of this object
  DISALLOW_COPY_AND_ASSIGN(Cache);
};

}   // namespace core
}   // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_CACHE_H
