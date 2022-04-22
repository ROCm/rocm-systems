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

#include "util/exception.h"

#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <future>
#include <mutex>

namespace roctracer {

class MemoryPool {
 public:
  MemoryPool(const roctracer_properties_t& properties) : properties_(properties) {
    // Pool definition: The memory pool is split in 2 buffers of equal size. When first initialized,
    // the write pointer points to the first element of the first buffer. When a buffer is full,  or
    // when Flush() is called, the write pointer moves to the other buffer.
    const size_t allocation_size = 2 * properties_.buffer_size;
    pool_begin_ = nullptr;
    AllocateMemory(&pool_begin_, allocation_size);
    if (pool_begin_ == nullptr) EXC_ABORT(ROCTRACER_STATUS_ERROR, "pool allocator failed");

    pool_end_ = pool_begin_ + allocation_size;
    buffer_begin_ = pool_begin_;
    buffer_end_ = buffer_begin_ + properties_.buffer_size;
    write_ptr_ = buffer_begin_;

    // Create a consumer thread and wait for it to be ready to accept work.
    std::promise<void> ready;
    std::future<void> future = ready.get_future();
    consumer_thread_ = std::thread(ConsumerThreadLoop, this, std::move(ready));
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

  template <typename Record> void Write(Record&& record) {
    std::lock_guard producer_lock(producer_mutex_);
    char* next = write_ptr_ + sizeof(record);
    if (next > buffer_end_) {
      NotifyConsumerThread(buffer_begin_, write_ptr_);

      // Switch buffers
      buffer_begin_ = (buffer_end_ == pool_end_) ? pool_begin_ : buffer_end_;
      buffer_end_ = buffer_begin_ + properties_.buffer_size;
      write_ptr_ = buffer_begin_;

      next = write_ptr_ + sizeof(record);
      if (next > buffer_end_)
        EXC_ABORT(ROCTRACER_STATUS_ERROR,
                  "buffer size(" << properties_.buffer_size << ") is less then the record("
                                 << sizeof(record) << ")");
    }

    // Store the record into the buffer, and increment the write pointer.
    ::memcpy(write_ptr_, &record, sizeof(record));
    write_ptr_ = next;
  }

  // Flush the records and block until they are all made visible to the client.
  void Flush() {
    {
      std::lock_guard producer_lock(producer_mutex_);
      if (write_ptr_ == buffer_begin_) return;

      NotifyConsumerThread(buffer_begin_, write_ptr_);

      // Switch buffers
      buffer_begin_ = (buffer_end_ == pool_end_) ? pool_begin_ : buffer_end_;
      buffer_end_ = buffer_begin_ + properties_.buffer_size;
      write_ptr_ = buffer_begin_;
    }
    {
      // Wait for the current operation to complete.
      std::unique_lock consumer_lock(consumer_mutex_);
      consumer_cond_.wait(consumer_lock, [this]() { return !consumer_arg_.valid; });
    }
  }

 private:
  void ConsumerThreadLoop(std::promise<void> ready) {
    std::unique_lock consumer_lock(consumer_mutex_);

    // This consumer is now ready to accept work.
    ready.set_value();

    while (true) {
      consumer_cond_.wait(consumer_lock, [this]() { return consumer_arg_.valid; });

      // begin == end == nullptr means the thread needs to exit.
      if (consumer_arg_.begin == nullptr && consumer_arg_.end == nullptr) break;

      properties_.buffer_callback_fun(consumer_arg_.begin, consumer_arg_.end,
                                      properties_.buffer_callback_arg);

      // Mark this operation as complete (valid=false) and notify all producers that may be
      // waiting for this operation to finish, or to start a new operation. See comment below in
      // NotifyConsumerThread().
      consumer_arg_.valid = false;
      consumer_cond_.notify_all();
    }
  }

  void NotifyConsumerThread(const char* data_begin, const char* data_end) {
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

  void AllocateMemory(char** ptr, size_t size) const {
    if (properties_.alloc_fun != nullptr) {
      // Use the custom allocator provided in the properties.
      properties_.alloc_fun(ptr, size, properties_.alloc_arg);
      return;
    }

    // No custom allocator was provided so use the default malloc/realloc/free allocator.
    if (*ptr == nullptr) {
      *ptr = reinterpret_cast<char*>(malloc(size));
    } else if (size != 0) {
      *ptr = reinterpret_cast<char*>(realloc(*ptr, size));
    } else {
      free(*ptr);
      *ptr = nullptr;
    }
  }

  // Properties used to create the memory pool.
  const roctracer_properties_t properties_;

  // Pool definition
  char* pool_begin_;  // FIXME: shouldn't these be void*?
  char* pool_end_;
  char* buffer_begin_;
  char* buffer_end_;
  char* write_ptr_;
  std::mutex producer_mutex_;

  // Consumer thread
  std::thread consumer_thread_;
  struct {
    const char* begin;
    const char* end;
    bool valid = false;
  } consumer_arg_;

  std::mutex consumer_mutex_;
  std::condition_variable consumer_cond_;
};

}  // namespace roctracer

#endif  // MEMORY_POOL_H_
