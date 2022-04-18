/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

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

#ifndef SRC_CORE_JOURNAL_H_
#define SRC_CORE_JOURNAL_H_

#include "ext/prof_protocol.h"

#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace roctracer {

template <typename Data> class Journal {
 public:
  /* Insert { domain, op } into the journal. Return false if the insertion failed.  */
  template <typename T = Data, std::enable_if_t<std::is_constructible_v<Data, T>, int> = 0>
  bool Insert(roctracer_domain_t domain, uint32_t op, T&& data) {
    std::lock_guard lock(mutex_);
    return map_[domain].emplace(op, std::forward<T>(data)).second;
  }

  /* Remove { domain, op } from the journal. Return false if the entry did not exist.  */
  bool Remove(roctracer_domain_t domain, uint32_t op) {
    std::lock_guard lock(mutex_);
    return map_[domain].erase(op) == 1;
  }

  template <typename Functor> void ForEach(Functor&& func) {
    std::lock_guard lock(mutex_);
    for (auto&& domain : map_)
      for (auto&& operation : domain.second)
        if (!func(domain.first /* domain */, operation.first /* op */, operation.second /* data */))
          break; /* FIXME: what are we breaking out of? */
  }

 private:
  std::mutex mutex_;
  std::unordered_map<roctracer_domain_t, std::unordered_map<uint32_t, Data>> map_;
};

}  // namespace roctracer

#endif  // SRC_CORE_JOURNAL_H_
