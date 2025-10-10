////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
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

#ifndef HSA_RUNTIME_CORE_UTIL_MEMPOOL_HPP_
#define HSA_RUNTIME_CORE_UTIL_MEMPOOL_HPP_

#include <atomic>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>
#include <cassert>

#include "core/util/locks.h"

namespace rocr {

template<typename T, size_t PoolSize>
class GenericMemPool {
    public:
      GenericMemPool() : block_size_(preallocblocks_ * minblock_) {}
      ~GenericMemPool() { clear(); }

      T* alloc() {
        ScopedAcquire<HybridMutex> lock(&lock_);
        if (free_list_.empty()) {
          allocate_block(block_size_);
          if (block_size_ < maxblocksize_) {
            block_size_ *= 2;
          }
        }
        T* item = free_list_.back();
        free_list_.pop_back();
        return item;
      }
      void free(T* item) {
        if (item == nullptr) return;
        ScopedAcquire<HybridMutex> lock(&lock_);
        free_list_.push_back(item);
      }
      void clear() {
        for (auto& blk : block_list_) {
            std::free(blk.first);
        }
        block_list_.clear();
        free_list_.clear();
      }

    private:
      void allocate_block(size_t count) {
        size_t bytes = count * sizeof(T);
        void* block = std::aligned_alloc(alignof(T), bytes);
        if (block == nullptr) throw std::bad_alloc();
        block_list_.push_back({block, count});
        auto *base = reinterpret_cast<T*>(block);
        for (size_t i = 0; i < count; ++i) {
          new (&base[i]) T();
          free_list_.push_back(&base[i]);
        }

      }
      static const size_t minblock_ = 4096 / sizeof(T);
      static const size_t preallocblocks_ = PoolSize;
      static const size_t maxblocksize_ = 1ULL << 28;
      HybridMutex lock_;
      std::vector<T*> free_list_;
      std::vector<std::pair<void*, size_t>> block_list_;
      size_t block_size_;
};
}
#endif //HSA_RUNTIME_CORE_UTIL_MEMPOOL_HPP_