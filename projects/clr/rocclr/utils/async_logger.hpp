/* Copyright (c) 2026 Advanced Micro Devices, Inc.

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

#ifndef ASYNC_LOGGER_HPP_
#define ASYNC_LOGGER_HPP_

#include "top.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdarg>
#include <memory>

namespace amd {
namespace logging {

//! \brief Configuration for async logger
struct AsyncLoggerConfig {
  bool enabled = true;
  size_t buffer_size = 8 * Mi;
  size_t flush_interval_ms = 50;
  // Redirect to stderr by default
  const char* log_file_path = nullptr;
};

//! \brief Simple async logger with single shared ring buffer
class AsyncLogger {
 public:
  static AsyncLogger& GetInstance();

  void Initialize(const AsyncLoggerConfig& config);
  void Shutdown();
  void WriteLog(uint16_t level, uint32_t mask, const char* file, uint32_t line,
                const char* format, va_list args);
  void FlushSync();

  //! \brief Signal-safe flush for crash handlers (best effort)
  void FlushOnCrash();

 private:
  AsyncLogger();
  ~AsyncLogger();
  AsyncLogger(const AsyncLogger&) = delete;
  AsyncLogger& operator=(const AsyncLogger&) = delete;

  void FlusherThreadLoop();
  void FlushBuffer();
  void InstallSignalHandlers();

  // Ring buffer entry header
  struct EntryHeader {
    uint32_t total_size;
    uint32_t thread_id;
    uint32_t pid;
    uint32_t line;
    uint64_t timestamp_us;
    uint16_t level;
    uint16_t file_len;
    uint16_t msg_len;
    std::atomic<uint16_t> ready;  // Set to 1 when write complete
  };

  AsyncLoggerConfig config_;
  std::atomic<bool> initialized_{false};
  std::atomic<bool> shutdown_{false};

  // Single shared ring buffer
  std::unique_ptr<char[]> buffer_;
  size_t capacity_;
  std::atomic<size_t> write_pos_{0};
  std::atomic<size_t> read_pos_{0};

  // Flusher thread (using std::thread for early init reliability)
  std::thread flusher_thread_;
  std::mutex flush_mutex_;
  std::condition_variable flush_cv_;

  // Cached values
  uint32_t cached_pid_;
  int output_fd_;
};

}  // namespace logging
}  // namespace amd

#endif  // ASYNC_LOGGER_HPP_
