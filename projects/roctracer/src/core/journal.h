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

#include <map>
#include <mutex>

namespace roctracer {

template <class Data>
class Journal {
  public:
  typedef std::mutex mutex_t;
  typedef std::map<uint32_t, Data> domain_map_t;
  typedef std::map<uint32_t, domain_map_t*> journal_map_t;

  struct record_t {
    uint32_t domain;
    uint32_t op;
    Data data;
  };

  Journal() {
    domain_mask_ = 0;
    map_ = new journal_map_t;
  }

  ~Journal() {
    for (auto& val : *map_) delete val.second;
    delete map_;
  }

  void registr(const record_t& record) {
    std::lock_guard<mutex_t> lck(mutex_);
    auto* map = get_domain_map(record.domain);
    map->insert({record.op, record.data});
  }

  void remove(const record_t& record) {
    std::lock_guard<mutex_t> lck(mutex_);
    auto* map = get_domain_map(record.domain);
    map->erase(record.op);
  }

  template <class F>
  F foreach(const F& f_i) {
    std::lock_guard<mutex_t> lck(mutex_);
    F f = f_i;
    for (uint32_t domain = 0, mask = domain_mask_; mask != 0; ++domain, mask >>= 1) {
      if (mask & 1) {
        auto map = get_domain_map(domain);
        auto begin = map->begin();
        auto end = map->end();
        for (auto it = begin; it != end; ++it) {
          if (f.fun({domain, it->first, it->second}) == false) break;
        }
      }
    }
    return f;
  }

  private:
  domain_map_t* get_domain_map(const uint32_t& domain) {
    domain_mask_ |= 1u << domain;
    auto domain_it = map_->find(domain);
    if (domain_it == map_->end()) {
      auto* domain_map = new domain_map_t;
      auto ret = map_->insert({domain, domain_map});
      domain_it = ret.first;
    }
    return domain_it->second;
  }

  mutex_t mutex_;
  journal_map_t* map_;
  uint32_t domain_mask_;
};

}  // namespace roctracer

#endif  // SRC_CORE_JOURNAL_H_
