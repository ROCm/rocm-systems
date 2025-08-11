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

#ifndef MEMORY_POOL_H_
#define MEMORY_POOL_H_

#include "roctracer.h"

#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <future>
#include <mutex>
#include <type_traits>

namespace roctracer {

class MemoryPool {
 public:
  MemoryPool(const roctracer_properties_t& properties) : properties_(properties) {
    // Pool definition: The memory pool is split in 2 buffers of equal size. When first initialized,
    // the write pointer points to the first element of the first buffer. When a buffer is full,  or
    // when Flush() is called, the write pointer moves to the other buffer.
    // Each buffer should be large enough to hold at least 2 activity records, as record pairs may
    // be written when external correlation ids are used.
    const size_t allocation_size =
        2 * std::max(2 * sizeof(roctracer_record_t), properties_.buffer_size);
    pool_begin_ = nullptr;
    AllocateMemory(&pool_begin_, allocation_size);
    assert(pool_begin_ != nullptr && "pool allocator failed");

    pool_end_ = pool_begin_ + allocation_size;
    buffer_begin_ = pool_begin_;
    buffer_end_ = buffer_begin_ + properties_.buffer_size;
    record_ptr_ = buffer_begin_;
    data_ptr_ = buffer_end_;

    // Create a consumer thread and wait for it to be ready to accept work.
    std::promise<void> ready;
    std::future<void> future = ready.get_future();
    consumer_thread_ = std::thread(&MemoryPool::ConsumerThreadLoop, this, std::move(ready));
    future.wait();
  }

  ~MemoryPool() {
    Flush();

    // Wait for the previous flush to complete, then send the exit signal.
    NotifyConsumerThread(nullptr, nullptr);
    consumer_thread_.join();

    // Free the pool's buffer memory.
    AllocateMemory(&pool_begin_, 0);
  }

  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;

  template <typename Record, typename Functor = std::function<void(Record& record, const void*)>>
  void Write(Record&& record, const void* data, size_t data_size, Functor&& store_data = {}) {
    assert(data != nullptr || data_size == 0);  // If data is null, then data_size must be 0

    std::lock_guard producer_lock(producer_mutex_);

    // The amount of memory reserved in the buffer to store data. If the data cannot fit because it
    // is larger than the buffer size minus one record, then the data won't be copied into the
    // buffer.
    size_t reserve_data_size =
        data_size <= (properties_.buffer_size - sizeof(Record)) ? data_size : 0;

    std::byte* next_record = record_ptr_ + sizeof(Record);
    if (next_record > (data_ptr_ - reserve_data_size)) {
      NotifyConsumerThread(buffer_begin_, record_ptr_);
      SwitchBuffers();
      next_record = record_ptr_ + sizeof(Record);
      assert(next_record <= buffer_end_ && "buffer size is less then the record size");
    }

    // Store data in the record. Copy the data first if it fits in the buffer
    // (reserve_data_size != 0).
    if (reserve_data_size) {
      data_ptr_ -= data_size;
      ::memcpy(data_ptr_, data, data_size);
      store_data(record, data_ptr_);
    } else if (data != nullptr) {
      store_data(record, data);
    }

    // Store the record into the buffer, and increment the write pointer.
    ::memcpy(record_ptr_, &record, sizeof(Record));
    record_ptr_ = next_record;

    // If the data does not fit in the buffer, flush the buffer with the record as is. We don't copy
    // the data so we make sure that the record and its data are processed by waiting until the
    // flush is complete.
    if (data != nullptr && reserve_data_size == 0) {
      NotifyConsumerThread(buffer_begin_, record_ptr_);
      SwitchBuffers();
      {
        std::unique_lock consumer_lock(consumer_mutex_);
        consumer_cond_.wait(consumer_lock, [this]() { return !consumer_arg_.valid; });
      }
    }
  }
  template <typename Record> void Write(Record&& record) {
    using DataPtr = void*;
    Write(std::forward<Record>(record), DataPtr(nullptr), 0, {});
  }

  // Flush the records and block until they are all made visible to the client.
  void Flush() {
    {
      std::lock_guard producer_lock(producer_mutex_);
      if (record_ptr_ == buffer_begin_) return;

      NotifyConsumerThread(buffer_begin_, record_ptr_);
      SwitchBuffers();
    }
    {
      // Wait for the current operation to complete.
      std::unique_lock consumer_lock(consumer_mutex_);
      consumer_cond_.wait(consumer_lock, [this]() { return !consumer_arg_.valid; });
    }
  }

 private:
  void SwitchBuffers() {
    buffer_begin_ = (buffer_end_ == pool_end_) ? pool_begin_ : buffer_end_;
    buffer_end_ = buffer_begin_ + properties_.buffer_size;
    record_ptr_ = buffer_begin_;
    data_ptr_ = buffer_end_;
  }

  void ConsumerThreadLoop(std::promise<void> ready) {
    std::unique_lock consumer_lock(consumer_mutex_);

    // This consumer is now ready to accept work.
    ready.set_value();

    while (true) {
      consumer_cond_.wait(consumer_lock, [this]() { return consumer_arg_.valid; });

      // begin == end == nullptr means the thread needs to exit.
      if (consumer_arg_.begin == nullptr && consumer_arg_.end == nullptr) break;

      properties_.buffer_callback_fun(reinterpret_cast<const char*>(consumer_arg_.begin),
                                      reinterpret_cast<const char*>(consumer_arg_.end),
                                      properties_.buffer_callback_arg);

      // Mark this operation as complete (valid=false) and notify all producers that may be
      // waiting for this operation to finish, or to start a new operation. See comment below in
      // NotifyConsumerThread().
      consumer_arg_.valid = false;
      consumer_cond_.notify_all();
    }
  }

  void NotifyConsumerThread(const std::byte* data_begin, const std::byte* data_end) {
    std::unique_lock consumer_lock(consumer_mutex_);

    // If consumer_arg_ is still in use (valid=true), then wait for the consumer thread to finish
    // processing the current operation. Multiple producers may wait here, one will be allowed to
    // continue once the consumer thread is idle and valid=false. This prevents a race condition
    // where operations would be lost if multiple producers could enter this critical section
    // (sequentially) before the consumer thread could re-acquire the consumer_mutex_ lock.
    consumer_cond_.wait(consumer_lock, [this]() { return !consumer_arg_.valid; });

    consumer_arg_.begin = data_begin;
    consumer_arg_.end = data_end;

    consumer_arg_.valid = true;
    consumer_cond_.notify_all();
  }

  void AllocateMemory(std::byte** ptr, size_t size) const {
    if (properties_.alloc_fun != nullptr) {
      // Use the custom allocator provided in the properties.
      properties_.alloc_fun(reinterpret_cast<char**>(ptr), size, properties_.alloc_arg);
      return;
    }

    // No custom allocator was provided so use the default malloc/realloc/free allocator.
    if (*ptr == nullptr) {
      *ptr = static_cast<std::byte*>(malloc(size));
    } else if (size != 0) {
      *ptr = static_cast<std::byte*>(realloc(*ptr, size));
    } else {
      free(*ptr);
      *ptr = nullptr;
    }
  }

  // Properties used to create the memory pool.
  const roctracer_properties_t properties_;

  // Pool definition
  std::byte* pool_begin_;
  std::byte* pool_end_;
  std::byte* buffer_begin_;
  std::byte* buffer_end_;
  std::byte* record_ptr_;
  std::byte* data_ptr_;
  std::mutex producer_mutex_;

  // Consumer thread
  std::thread consumer_thread_;
  struct {
    const std::byte* begin;
    const std::byte* end;
    bool valid = false;
  } consumer_arg_;

  std::mutex consumer_mutex_;
  std::condition_variable consumer_cond_;
};

}  // namespace roctracer

#endif  // MEMORY_POOL_H_
