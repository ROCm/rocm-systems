/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

#ifndef SRC_CORE_CONTEXT_POOL_H_
#define SRC_CORE_CONTEXT_POOL_H_

#include "rocprofiler.h"

#include <thread>

#include "core/context.h"

namespace rocprofiler {
class ContextPool {
 public:
  typedef uint64_t index_t;
  typedef std::mutex mutex_t;

  struct entry_t {
    ContextPool* pool;
    Context* context;
    std::atomic<bool> completed;
  };

  static ContextPool* Create(uint32_t num_entries, uint32_t payload_bytes,
                             const util::AgentInfo* agent_info, rocprofiler_feature_t* info,
                             const uint32_t info_count, rocprofiler_pool_handler_t handler,
                             void* handler_arg) {
    ContextPool* obj = new ContextPool(num_entries, payload_bytes, agent_info, info, info_count,
                                       handler, handler_arg);
    if (obj == NULL) EXC_RAISING(HSA_STATUS_ERROR, "allocation error");
    return obj;
  }

  static void Destroy(ContextPool* pool) { delete pool; }

  void Fetch(rocprofiler_pool_entry_t* pool_entry) {
    if (constructed_ == false) {
      Construct(agent_info_, info_, info_count_);
    }
    const index_t write_index =
        write_index_.fetch_add(entry_size_bytes_, std::memory_order_relaxed);
    while (write_index >= (read_index_.load(std::memory_order_acquire) + array_size_bytes_)) {
      check_completed();
      std::this_thread::yield();
    }
    entry_t* entry = GetPoolEntry(write_index, pool_entry);
    if (entry->completed.load(std::memory_order_relaxed) != false)
      EXC_RAISING(HSA_STATUS_ERROR, "Corrupted pool entry");
  }

  void Flush() { check_completed(); }
#if 0
  template <class F>
  F for_each(const F& f_p) {
    F f = f_p;
    while (sync_flag_.test_and_set(std::memory_order_acquire) != false) {
      std::this_thread::yield();
    }

    index_t read_index = read_index_.load(std::memory_order_relaxed);
    const index_t write_index = write_index_.load(std::memory_order_relaxed);
    while(read_index < write_index) {
      rocprofiler_pool_entry_t pool_entry{};
      entry_t* entry = GetPoolEntry(read_index, &pool_entry);
      const bool completed = entry->completed.load(std::memory_order_acquire);
      if (completed == false) {
        f(entry->context, entry->payload);
      }
    }

    return f;
  }
#endif
 private:
  static unsigned aligned64(const unsigned& size) { return (size + 0x3f) & ~0x3fu; }

  static bool context_handler(rocprofiler_group_t group, void* arg) {
    entry_t* entry = reinterpret_cast<entry_t*>(arg);
    entry->completed.store(true, std::memory_order_release);
    entry->pool->check_completed();
    return true;
  }

  ContextPool(uint32_t num_entries, uint32_t payload_bytes, const util::AgentInfo* agent_info,
              rocprofiler_feature_t* info, const uint32_t info_count,
              rocprofiler_pool_handler_t pool_handler, void* pool_handler_arg)
      : payload_off_(aligned64(sizeof(entry_t))),
        entry_size_bytes_(payload_off_ + aligned64(payload_bytes)),
        array_size_bytes_(entry_size_bytes_ * num_entries),
        array_(NULL),
        read_index_(0),
        write_index_(0),
        sync_flag_(false),

        agent_info_(agent_info),
        info_(info),
        info_count_(info_count),
        pool_handler_(pool_handler),
        pool_handler_arg_(pool_handler_arg),
        constructed_(false) {}

  void Construct(const util::AgentInfo* agent_info, rocprofiler_feature_t* info,
                 const uint32_t info_count) {
    std::lock_guard<mutex_t> lck(mutex_);

    if (constructed_ == false) {
      array_data_ = (char*)malloc(array_size_bytes_ + 0x3f);
      array_ = reinterpret_cast<char*>(((intptr_t)array_data_ + 0x3f) >> 6 << 6);
      if (((intptr_t)array_ & 0x3f) != 0)
        EXC_RAISING(HSA_STATUS_ERROR, "Pool array is not aligned");
      memset(array_, 0, array_size_bytes_);

      const char* end = array_ + array_size_bytes_;
      for (char* ptr = array_; ptr < end; ptr += entry_size_bytes_) {
        entry_t* entry = reinterpret_cast<entry_t*>(ptr);
        entry->pool = this;
        entry->context =
            Context::Create(agent_info, NULL, info, info_count, ContextPool::context_handler, ptr);
      }

      constructed_ = true;
    }
  }

  ~ContextPool() {
    const char* end = array_ + array_size_bytes_;
    for (char* ptr = array_; ptr < end; ptr += entry_size_bytes_) {
      entry_t* entry = reinterpret_cast<entry_t*>(ptr);
      Context::Destroy(entry->context);
    }
    free(array_);
  }

  char* GetArrayPtr(const uint32_t& index) { return array_ + (index % array_size_bytes_); }

  entry_t* GetPoolEntry(const uint32_t& index, rocprofiler_pool_entry_t* pool_entry) {
    char* ptr = GetArrayPtr(index);
    entry_t* entry = reinterpret_cast<entry_t*>(ptr);
    void* payload = ptr + payload_off_;
    *pool_entry = rocprofiler_pool_entry_t{};
    pool_entry->context = reinterpret_cast<rocprofiler_t*>(entry->context);
    pool_entry->payload = payload;
    return entry;
  }

  void check_completed() {
    if (sync_flag_.test_and_set(std::memory_order_acquire) == false) {
      index_t read_index = read_index_.load(std::memory_order_relaxed);
      const index_t write_index = write_index_.load(std::memory_order_relaxed);
      while (read_index < write_index) {
        rocprofiler_pool_entry_t pool_entry{};
        entry_t* entry = GetPoolEntry(read_index, &pool_entry);
        if (entry->completed.load(std::memory_order_acquire) == true) {
          pool_handler_(&pool_entry, pool_handler_arg_);
          entry->completed.store(false, std::memory_order_relaxed);
          read_index += entry_size_bytes_;
          read_index_.store(read_index, std::memory_order_release);
        } else {
          break;
        }
      }
      sync_flag_.clear(std::memory_order_release);
    }
  }

  const uint32_t payload_off_;
  const uint32_t entry_size_bytes_;
  const uint32_t array_size_bytes_;
  char* array_data_;
  char* array_;
  volatile std::atomic<index_t> read_index_;
  volatile std::atomic<index_t> write_index_;
  volatile std::atomic_flag sync_flag_;

  const util::AgentInfo* agent_info_;
  rocprofiler_feature_t* info_;
  const uint32_t info_count_;
  rocprofiler_pool_handler_t pool_handler_;
  void* pool_handler_arg_;

  bool constructed_;
  mutex_t mutex_;
};
}  // namespace rocprofiler

#endif  // SRC_CORE_CONTEXT_POOL_H_
