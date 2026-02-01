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

#include "async_logger.hpp"
#include "os/os.hpp"
#include <cstring>
#include <cstdio>
#include <inttypes.h>
#include <algorithm>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#define STDERR_FILENO 2
#define open _open
#define close _close
#define write _write
#define lseek _lseek
#define fsync _commit
typedef int ssize_t;
typedef long off_t;
#else
#include <unistd.h>
#include <fcntl.h>
#endif

namespace amd {

extern const size_t maxLogSize;

namespace logging {

// Static callback for crash signal handler
static void crashFlushCallback() {
  AsyncLogger::GetInstance().FlushOnCrash();
}

AsyncLogger& AsyncLogger::GetInstance() {
  static AsyncLogger instance;
  return instance;
}

AsyncLogger::AsyncLogger()
    : capacity_(0), cached_pid_(0), output_fd_(-1) {}

AsyncLogger::~AsyncLogger() {
  Shutdown();
}

void AsyncLogger::Initialize(const AsyncLoggerConfig& config) {
  if (initialized_.exchange(true)) {
    return;
  }

  config_ = config;
  cached_pid_ = Os::getProcessId();

  if (!config_.enabled) {
    return;
  }

  // Validate and allocate ring buffer (round up to power of 2)
  // Minimum size: 64KB to hold at least a few entries
  constexpr size_t kMinBufferSize = 64 * 1024;
  capacity_ = std::max(config_.buffer_size, kMinBufferSize);
  if ((capacity_ & (capacity_ - 1)) != 0) {
    size_t pow2 = 1;
    while (pow2 < capacity_) pow2 <<= 1;
    capacity_ = pow2;
  }
  buffer_ = std::make_unique<char[]>(capacity_);

  // Open output file
  if (config_.log_file_path) {
    output_fd_ = open(config_.log_file_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (output_fd_ < 0) output_fd_ = STDERR_FILENO;
  } else {
    output_fd_ = STDERR_FILENO;
  }

  // Start flusher thread
  shutdown_.store(false);
  flusher_thread_ = std::thread(&AsyncLogger::FlusherThreadLoop, this);

#ifdef __linux__
  pthread_setname_np(flusher_thread_.native_handle(), "amd_log_flush");
#endif

  // Install crash signal handlers
  InstallSignalHandlers();
}

void AsyncLogger::Shutdown() {
  if (!initialized_.load() || shutdown_.exchange(true)) {
    return;
  }

  flush_cv_.notify_one();
  if (flusher_thread_.joinable()) {
    flusher_thread_.join();
  }

  FlushBuffer();

  if (output_fd_ >= 0 && output_fd_ != STDERR_FILENO) {
    close(output_fd_);
    output_fd_ = -1;
  }
}

void AsyncLogger::WriteLog(uint16_t level, uint32_t mask, const char* file,
                           uint32_t line, const char* format, va_list args) {
  if (!config_.enabled || shutdown_.load(std::memory_order_acquire)) {
    return;
  }

  // Format message
  char message[4096];
  vsnprintf(message, sizeof(message), format, args);

  uint16_t file_len = file ? static_cast<uint16_t>(std::min(strlen(file), size_t(255))) : 0;
  uint16_t msg_len = static_cast<uint16_t>(std::min(strlen(message), size_t(4095)));
  uint32_t total_size = sizeof(EntryHeader) + file_len + msg_len;

  // Align to 8 bytes
  total_size = alignUp(total_size, 8);

  // Atomically claim space using CAS loop to avoid double-advance on retry
  size_t claimed;
  for (;;) {
    size_t write = write_pos_.load(std::memory_order_relaxed);
    size_t read = read_pos_.load(std::memory_order_acquire);
    size_t pos = write & (capacity_ - 1);

    // Check if entry would wrap around buffer boundary
    // If so, pad to end and try again (entries must be contiguous)
    if (pos + total_size > capacity_) {
      size_t padding = capacity_ - pos;
      if (write_pos_.compare_exchange_weak(write, write + padding,
                                            std::memory_order_relaxed)) {
        // Padding added, retry to get new position
        continue;
      }
      continue;  // CAS failed, retry
    }

    // Check if buffer has enough space
    if (write + total_size - read > capacity_) {
      FlushSync();
      continue;  // Retry after flush
    }

    // Try to claim space
    if (write_pos_.compare_exchange_weak(write, write + total_size,
                                          std::memory_order_acq_rel)) {
      claimed = write;
      break;
    }
  }

  // Calculate position in buffer (guaranteed contiguous now)
  size_t pos = claimed & (capacity_ - 1);

  // Write header directly to buffer, set ready to 0 initially
  EntryHeader* hdr = reinterpret_cast<EntryHeader*>(buffer_.get() + pos);
  hdr->total_size = total_size;
  hdr->thread_id =
      static_cast<uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
  hdr->pid = cached_pid_;
  hdr->line = line;
  hdr->timestamp_us = Os::timeNanos() / 1000ULL;
  hdr->level = level;
  hdr->file_len = file_len;
  hdr->msg_len = msg_len;
  hdr->ready.store(0, std::memory_order_relaxed);

  // Write file and message (contiguous, no wrap)
  char* data_ptr = buffer_.get() + pos + sizeof(EntryHeader);
  if (file_len > 0) {
    memcpy(data_ptr, file, file_len);
    data_ptr += file_len;
  }
  if (msg_len > 0) {
    memcpy(data_ptr, message, msg_len);
  }

  // Mark ready (release ensures all prior writes are visible to reader)
  hdr->ready.store(1, std::memory_order_release);
}

void AsyncLogger::FlushSync() {
  std::lock_guard<std::mutex> lock(flush_mutex_);
  FlushBuffer();
}

void AsyncLogger::FlusherThreadLoop() {
  std::unique_lock<std::mutex> lock(flush_mutex_);
  while (!shutdown_.load(std::memory_order_acquire)) {
    flush_cv_.wait_for(lock, std::chrono::milliseconds(config_.flush_interval_ms),
                      [this] { return shutdown_.load(std::memory_order_acquire); });
    lock.unlock();
    FlushBuffer();
    lock.lock();
  }
}

void AsyncLogger::FlushBuffer() {
  size_t read_idx = read_pos_.load(std::memory_order_acquire);
  size_t write_idx = write_pos_.load(std::memory_order_acquire);

  char output[8192];

  while (read_idx < write_idx) {
    size_t pos = read_idx & (capacity_ - 1);

    // Entry header is guaranteed contiguous (WriteLog pads to avoid wrap)
    EntryHeader* hdr = reinterpret_cast<EntryHeader*>(buffer_.get() + pos);

    // Wait for ready FIRST (acquire ensures we see all writes from producer)
    uint16_t ready_val = hdr->ready.load(std::memory_order_acquire);
    if (ready_val == 0) {
      constexpr int kSpinIter = 100;
      for (int i = 0; i < kSpinIter && ready_val == 0; ++i) {
        Os::spinPause();
        ready_val = hdr->ready.load(std::memory_order_acquire);
      }
      if (ready_val == 0) {
        std::this_thread::yield();
        continue;
      }
    }

    // Now safe to read header fields (producer has finished writing)
    uint32_t total_size = hdr->total_size;
    uint16_t file_len = hdr->file_len;
    uint16_t msg_len = hdr->msg_len;

    // Validate
    if (total_size == 0 || total_size > 8192) {
      read_idx += 8;
      continue;
    }

    // Read data (contiguous, no wrap needed)
    char filename[256] = {0};
    char message[4096] = {0};
    const char* data_ptr = buffer_.get() + pos + sizeof(EntryHeader);

    if (file_len > 0 && file_len < 256) {
      memcpy(filename, data_ptr, file_len);
      data_ptr += file_len;
    }
    if (msg_len > 0 && msg_len < 4096) {
      memcpy(message, data_ptr, msg_len);
    }

    // Format and write
    int len = snprintf(output, sizeof(output),
                      ":%d:%-25s:%-4d: %010" PRIu64 " us: [pid:%d tid:0x%x] %s\n",
                      hdr->level, filename, hdr->line, hdr->timestamp_us,
                      hdr->pid, hdr->thread_id, message);

    if (len > 0 && output_fd_ >= 0) {
      [[maybe_unused]] ssize_t r = ::write(output_fd_, output, len);
    }

    read_idx += total_size;
  }

  read_pos_.store(read_idx, std::memory_order_release);

  // Truncate if needed
  if (output_fd_ > STDERR_FILENO && config_.log_file_path) {
    off_t size = lseek(output_fd_, 0, SEEK_END);
    if (size > static_cast<off_t>(maxLogSize)) {
      close(output_fd_);
      output_fd_ = open(config_.log_file_path, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
      if (output_fd_ < 0) output_fd_ = STDERR_FILENO;
    }
  }
}

void AsyncLogger::InstallSignalHandlers() {
  Os::installExceptionHandlers(crashFlushCallback);
}

void AsyncLogger::FlushOnCrash() {
  // Async-signal-safe: only use write() - fsync is NOT async-signal-safe
  [[maybe_unused]] ssize_t r;

  static const char crash_header[] = "\n=== CRASH - Flushing buffered logs ===\n";
  r = ::write(output_fd_, crash_header, sizeof(crash_header) - 1);

  // Read positions atomically (async-signal-safe on most platforms)
  size_t read_idx = read_pos_.load(std::memory_order_relaxed);
  size_t write_idx = write_pos_.load(std::memory_order_relaxed);

  // Simple approach: write raw buffer content between read and write positions
  // This won't be formatted but preserves the data
  while (read_idx < write_idx) {
    size_t pos = read_idx & (capacity_ - 1);
    size_t available = write_idx - read_idx;
    size_t chunk = (pos + available <= capacity_) ? available : (capacity_ - pos);

    // Limit chunk size to avoid very long writes
    if (chunk > 4096) chunk = 4096;

    r = ::write(output_fd_, buffer_.get() + pos, chunk);
    read_idx += chunk;
  }

  static const char crash_footer[] = "\n=== END CRASH FLUSH ===\n";
  r = ::write(output_fd_, crash_footer, sizeof(crash_footer) - 1);

  // Note: fsync() is NOT async-signal-safe, omitted for crash context
}

}  // namespace logging
}  // namespace amd
