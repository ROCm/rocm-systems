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

#ifndef TOOL_TRACE_BUFFER_H_
#define TOOL_TRACE_BUFFER_H_

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <list>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

namespace roctracer {

class TraceBufferBase {
 public:
  static void FlushAll() {
    std::lock_guard lock(mutex_);

    for (auto* trace_buffer = head_; trace_buffer != nullptr; trace_buffer = trace_buffer->next_)
      trace_buffer->Flush();
  }

  static void Register(TraceBufferBase* elem) {
    std::lock_guard lock(mutex_);

    auto** prev_ptr = &head_;
    while (*prev_ptr != nullptr && elem->priority_ > (*prev_ptr)->priority_)
      prev_ptr = &(*prev_ptr)->next_;

    elem->next_ = *prev_ptr;
    *prev_ptr = elem;
  }

  static void Unregister(TraceBufferBase* elem) {
    std::lock_guard lock(mutex_);

    auto** prev_ptr = &head_;
    while (*prev_ptr != nullptr && *prev_ptr != elem) prev_ptr = &(*prev_ptr)->next_;

    assert(*prev_ptr != nullptr && "elem is not in the list");
    *prev_ptr = elem->next_;
  }

  TraceBufferBase(std::string name, int priority)
      : name_(std::move(name)), priority_(priority), next_(nullptr) {}

  TraceBufferBase(const TraceBufferBase&) = delete;
  TraceBufferBase& operator=(const TraceBufferBase&) = delete;

  virtual ~TraceBufferBase() { Unregister(this); }

  virtual void Flush() = 0;

  std::string name() && { return std::move(name_); }
  const std::string& name() const& { return name_; }

 private:
  const std::string name_;
  const int priority_;
  TraceBufferBase* next_;

  static TraceBufferBase* head_;
  static std::mutex mutex_;
};

enum TraceEntryState { TRACE_ENTRY_INVALID = 0, TRACE_ENTRY_INIT = 1, TRACE_ENTRY_COMPLETE = 2 };

template <typename Entry, typename Allocator = std::allocator<Entry>>
class TraceBuffer : protected TraceBufferBase {
 public:
  using callback_t = std::function<void(Entry*)>;

  TraceBuffer(std::string name, uint64_t size, callback_t flush_callback, int priority = 0)
      : TraceBufferBase(std::move(name), priority),
        flush_callback_(std::move(flush_callback)),
        size_(size) {
    assert(size_ != 0 && "cannot create an empty trace buffer");

    Entry* write_buffer = allocator_.allocate(size_);
    assert(write_buffer != nullptr);
    buffer_list_.push_back(write_buffer);

    read_index_ = 0;
    write_index_ = {0, write_buffer};

    AllocateFreeBuffer();

    // Add this instance to the link list of all trace buffers in the process.
    Register(this);
  }

  ~TraceBuffer() override {
    // Flush the remaining records. After flushing, there should not be any records left in the
    // trace buffer.
    Flush();
    assert(read_index_ == write_index_.load().index);

    // Acquire both the writer and worker lock as we are accessing shared variables they protect.
    std::unique_lock writer_lock(write_mutex_, std::defer_lock);
    std::unique_lock worker_lock(worker_mutex_, std::defer_lock);
    std::lock(writer_lock, worker_lock);

    // Deallocate the buffers.
    allocator_.deallocate(write_index_.load().buffer, size_);
    allocator_.deallocate(free_buffer_, size_);

    // Stop the worker thread. The worker thread loop checks the 'worker_thread_' std::optional
    // after waking up, and exits if it does not have a value.
    if (worker_thread_) {
      std::thread worker_thread = std::move(worker_thread_.value());
      {
        // Tell the worker thread loop to exit.
        worker_thread_.reset();
        free_buffer_ = nullptr;
        worker_cond_.notify_one();
      }
      // Release the worker lock to allow the worker thread to exit.
      worker_lock.unlock();
      worker_thread.join();
    }
  }

  // Flush all entries between read_pointer and write_pointer. read_pointer and write_pointer are
  // monotonically increasing indices, with read_pointer % size always indexing inside the first
  // buffer in the list. Stop flushing if an incomplete entry is found, it will be flushed with
  // the next invocation after changing its state to 'complete'.
  void Flush() override {
    std::lock_guard lock(write_mutex_);
    auto write_index = write_index_.load(std::memory_order_relaxed);

    for (auto it = buffer_list_.begin(); it != buffer_list_.end();) {
      auto end_of_buffer = read_index_ - read_index_ % size_ + size_;

      while (read_index_ < std::min(write_index.index, end_of_buffer)) {
        Entry* entry = &(*it)[read_index_ % size_];

        // The entry is not yet complete, stop flushing here.
        if (entry->valid.load(std::memory_order_acquire) != TRACE_ENTRY_COMPLETE) return;

        flush_callback_(entry);
        entry->~Entry();

        ++read_index_;
      }

      // The buffer is still in use or the read pointer did not reach the end of the buffer.
      if (*it == write_index.buffer || read_index_ != end_of_buffer) return;

      // All entries in the current buffer are now processed. Destroy the buffer and move onto the
      // next buffer in the list.
      allocator_.deallocate(*it, size_);
      it = buffer_list_.erase(it);
    }
  }

  template <typename... Args> Entry& Emplace(Args... args) {
    return *new (GetEntry()) Entry(std::forward<Args>(args)...);
  }

 private:
  Entry* GetEntry() {
    auto current = write_index_.load(std::memory_order_relaxed);

    while (true) {
      // If the pointer is at the end of the current buffer, switch to the available free buffer and
      // notify the worker thread to allocate a new buffer.
      if (current.index != 0 && current.index % size_ == 0) {
        std::lock_guard lock(write_mutex_);

        // If the worker thread wasn't already started, start it now. This avoids starting a new
        // thread when the trace buffer is created.
        if (!worker_thread_) {
          std::promise<void> ready;
          auto future = ready.get_future();
          {
            std::lock_guard worker_lock(worker_mutex_);
            worker_thread_.emplace(&TraceBuffer::WorkerThreadLoop, this, std::move(ready));
          }
          future.wait();
        }

        // Re-check the pointer overflow under the writer lock, another thread could have beaten us
        // to it and already bumped the write_index_.
        current = write_index_.load(std::memory_order_relaxed);
        if (current.index % size_ == 0) {
          std::unique_lock worker_lock(worker_mutex_);

          // Wait for the free buffer to become available.
          worker_cond_.wait(worker_lock, [this]() { return free_buffer_ != nullptr; });

          current.buffer = free_buffer_;
          buffer_list_.push_back(current.buffer);
          write_index_.store({current.index + 1, current.buffer}, std::memory_order_relaxed);

          // Tell the worker thread to allocate a new free buffer.
          free_buffer_ = nullptr;
          worker_cond_.notify_one();

          // We successfully allocated a new buffer, return the first element.
          return &current.buffer[0];
        }
      }

      if (write_index_.compare_exchange_weak(current, {current.index + 1, current.buffer},
                                             std::memory_order_relaxed))
        return &current.buffer[current.index % size_];
    }
  }

  void AllocateFreeBuffer() {
    assert(free_buffer_ == nullptr);

    free_buffer_ = allocator_.allocate(size_);
    assert(free_buffer_ != nullptr);

    for (size_t i = 0; i < size_; ++i)
      free_buffer_[i].valid.store(TRACE_ENTRY_INVALID, std::memory_order_relaxed);
  }

  void WorkerThreadLoop(std::promise<void> ready) {
    std::unique_lock lock(worker_mutex_);

    // This worker thread is now ready to accept work.
    ready.set_value();

    while (true) {
      worker_cond_.wait(lock, [this]() { return free_buffer_ == nullptr; });
      if (!worker_thread_) break;
      AllocateFreeBuffer();
      worker_cond_.notify_one();
    }
  }

  // The WriteIndex is used to store both the index and the buffer associated with that index (the
  // buffer contains the trace buffer records at [index - index % size, index - index % size_t +
  // size_ - 1]) in a single atomic variable.
  struct WriteIndex {
    uint64_t index;
    Entry* buffer;
  };

  const callback_t flush_callback_;
  const uint64_t size_;

  uint64_t read_index_;                  // The index of the next record to flush.
  std::atomic<WriteIndex> write_index_;  // The index of the next record that could be written.
  Entry* free_buffer_{nullptr};          // The next available free buffer.

  std::optional<std::thread> worker_thread_;
  std::mutex worker_mutex_;
  std::condition_variable worker_cond_;

  std::mutex write_mutex_;
  std::list<Entry*> buffer_list_;
  Allocator allocator_;
};
}  // namespace roctracer

#define TRACE_BUFFER_INSTANTIATE()                                                                 \
  roctracer::TraceBufferBase* roctracer::TraceBufferBase::head_ = nullptr;                         \
  std::mutex roctracer::TraceBufferBase::mutex_;

#endif  // TOOL_TRACE_BUFFER_H_
