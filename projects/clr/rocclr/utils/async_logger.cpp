/* Copyright (c) 2025 Advanced Micro Devices, Inc.

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
#include "flags.hpp"
#include "debug.hpp"
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <sys/uio.h>
#include <inttypes.h>
#include <signal.h>
#include <atomic>

namespace amd {

// Defined in debug.cpp: const size_t maxLogSize = AMD_LOG_LEVEL_SIZE * Mi
extern const size_t maxLogSize;

namespace logging {

// Signal handler support for crash recovery
static std::atomic<bool> signal_handlers_installed{false};

static void crash_signal_handler(int signum) {
  // Best-effort flush without blocking - may lose some logs but better than nothing
  AsyncLogger::GetInstance().FlushUnsafe();

  // Re-raise signal with default handler to get proper crash behavior (core dump, etc.)
  signal(signum, SIG_DFL);
  raise(signum);
}

static void install_crash_signal_handlers() {
  if (signal_handlers_installed.exchange(true, std::memory_order_acq_rel)) {
    return;  // Already installed
  }

  signal(SIGSEGV, crash_signal_handler);
  signal(SIGABRT, crash_signal_handler);
  signal(SIGBUS, crash_signal_handler);
  signal(SIGFPE, crash_signal_handler);
  signal(SIGILL, crash_signal_handler);
}

// Thread-local storage for ring buffer
thread_local RingBuffer* AsyncLogger::tls_buffer_ = nullptr;
thread_local uint32_t AsyncLogger::tls_thread_id_ = 0;
thread_local bool AsyncLogger::tls_thread_id_cached_ = false;
thread_local std::unique_ptr<ThreadLocalBufferGuard> AsyncLogger::tls_guard_ = nullptr;

// Singleton instance
AsyncLogger& AsyncLogger::GetInstance() {
  static AsyncLogger instance;
  return instance;
}

AsyncLogger::AsyncLogger()
    : initialized_(false),
      shutdown_(false),
      cached_pid_(0),
      output_fd_(-1) {
}

AsyncLogger::~AsyncLogger() {
  Shutdown();
}

void AsyncLogger::Initialize(const AsyncLoggerConfig& config) {
  if (initialized_.exchange(true)) {
    return;  // Already initialized
  }

  config_ = config;

  // Cache PID - never changes
  cached_pid_ = Os::getProcessId();

  if (!config_.enabled) {
    return;
  }

  // Open output file
  if (config_.log_file_path) {
    output_fd_ = open(config_.log_file_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (output_fd_ < 0) {
      output_fd_ = STDERR_FILENO;
    }
  } else {
    output_fd_ = STDERR_FILENO;
  }

  // Start flusher thread
  shutdown_.store(false);
  flusher_thread_ = std::thread(&AsyncLogger::FlusherThread, this);

  // Set thread name for debugging
#ifdef __linux__
  pthread_setname_np(flusher_thread_.native_handle(), "amd_log_flush");
#endif

  // Install crash signal handlers for emergency flush
  install_crash_signal_handlers();
}

void AsyncLogger::Shutdown() {
  if (!initialized_.load() || shutdown_.exchange(true)) {
    return;
  }

  // Wake up flusher thread
  flush_cv_.notify_one();

  // Wait for flusher thread
  if (flusher_thread_.joinable()) {
    flusher_thread_.join();
  }

  // Final flush
  FlushAllBuffers();

  // Close output file
  if (output_fd_ >= 0 && output_fd_ != STDERR_FILENO) {
    close(output_fd_);
    output_fd_ = -1;
  }

  // Note: Don't clear thread_buffers_ here - threads may still hold tls_buffer_
  // pointers. Buffers are already flushed and will be destroyed with the singleton.
}

RingBuffer* AsyncLogger::GetThreadBuffer() {
  if (tls_buffer_) {
    return tls_buffer_;
  }

  // Check if we're shutting down - don't create new buffers
  if (shutdown_.load(std::memory_order_acquire)) {
    return nullptr;
  }

  // Create new buffer for this thread
  auto buffer = std::make_unique<RingBuffer>(config_.buffer_size_per_thread);
  RingBuffer* raw_ptr = buffer.get();

  // Register buffer (transfer ownership to thread_buffers_)
  {
    std::lock_guard<std::mutex> lock(buffers_mutex_);
    thread_buffers_.push_back(std::move(buffer));
  }

  // Set up thread-local pointer and guard
  tls_buffer_ = raw_ptr;
  tls_guard_ = std::make_unique<ThreadLocalBufferGuard>(raw_ptr);

  return raw_ptr;
}

void AsyncLogger::WriteLog(uint16_t level, uint32_t mask, const char* file,
                           uint32_t line, const char* format, va_list args) {
  // Early exit if disabled or shutting down
  if (!config_.enabled || shutdown_.load(std::memory_order_acquire)) {
    return;
  }

  // Get thread-local buffer first (fast path check)
  RingBuffer* buffer = GetThreadBuffer();
  if (!buffer) {
    return;
  }

  // Get cached thread ID (computed once per thread)
  if (!tls_thread_id_cached_) {
    tls_thread_id_ = static_cast<uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()));
    tls_thread_id_cached_ = true;
  }

  // Format message - use return value to get length
  char message[4096];
  vsnprintf(message, sizeof(message), format, args);

  // Get timestamp (still needed per call)
  uint64_t timestamp_us = Os::timeNanos() / 1000ULL;

  // Use cached values
  uint32_t thread_id = tls_thread_id_;
  uint32_t pid = cached_pid_;

  // Try to write to buffer
  if (!buffer->TryWrite(timestamp_us, thread_id, pid, level, mask,
                        file, line, message)) {
    // Buffer full - do synchronous flush and retry
    FlushSync();
    buffer->TryWrite(timestamp_us, thread_id, pid, level, mask, file, line, message);
  }
}

void AsyncLogger::FlushSync() {
  FlushAllBuffers();
}

void AsyncLogger::FlusherThread() {
  std::unique_lock<std::mutex> lock(flush_mutex_);
  while (!shutdown_.load(std::memory_order_acquire)) {
    flush_cv_.wait_for(lock, std::chrono::milliseconds(config_.flush_interval_ms),
                      [this] { return shutdown_.load(std::memory_order_acquire); });
    lock.unlock();
    FlushAllBuffers();
    lock.lock();
  }
}

void AsyncLogger::FlushAllBuffers() {
  std::vector<char> batch;
  batch.reserve(1024 * 1024);  // 1MB batch buffer

  std::vector<RingBuffer*> buffers_snapshot;
  {
    std::lock_guard<std::mutex> lock(buffers_mutex_);
    buffers_snapshot.reserve(thread_buffers_.size());
    for (const auto& buf : thread_buffers_) {
      buffers_snapshot.push_back(buf.get());
    }
  }

  // Read from all thread buffers
  for (auto* buffer : buffers_snapshot) {
    buffer->ReadAvailable([this, &batch](uint64_t timestamp_us, uint32_t thread_id,
                                        uint32_t pid, uint16_t level, uint32_t mask,
                                        const char* file, uint32_t line,
                                        const char* message) {
      FormatLogEntry(batch, timestamp_us, thread_id, pid, level, mask, file, line, message);
    });
  }

  // Write batch
  if (!batch.empty()) {
    WriteBatch(batch);
  }

  // Periodically clean up orphaned buffers
  CleanupOrphanedBuffers();
}

void AsyncLogger::FlushUnsafe() {
  // Signal-safe emergency flush - uses only async-signal-safe functions
  // Note: write() and fsync() are async-signal-safe per POSIX
  // We can't safely read from ring buffers here (would need locks/allocations)
  // Just sync what's already been written and add a crash marker

  if (output_fd_ < 0) {
    return;
  }

  // Static buffer - no allocation. Use fixed-size string literals.
  static const char crash_msg[] = "\n--- [CRASH - buffered logs may be lost] ---\n";
  write(output_fd_, crash_msg, sizeof(crash_msg) - 1);
  fsync(output_fd_);
}

void AsyncLogger::WriteBatch(const std::vector<char>& batch) {
  if (batch.empty() || output_fd_ < 0) {
    return;
  }

  // Check if log file needs truncation (O_APPEND handles write positioning)
  if (output_fd_ > STDERR_FILENO && config_.log_file_path) {
    off_t file_size = lseek(output_fd_, 0, SEEK_END);
    if (file_size > static_cast<off_t>(amd::maxLogSize)) {
      close(output_fd_);
      output_fd_ = open(config_.log_file_path, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
      if (output_fd_ < 0) {
        output_fd_ = STDERR_FILENO;
      }
    }
  }

  struct iovec iov;
  iov.iov_base = const_cast<char*>(batch.data());
  iov.iov_len = batch.size();
  writev(output_fd_, &iov, 1);
}
void AsyncLogger::FormatLogEntry(std::vector<char>& output, uint64_t timestamp_us,
                                uint32_t thread_id, uint32_t pid, uint16_t level,
                                uint32_t mask, const char* file, uint32_t line,
                                const char* message) {
  char buffer[8192];
  int len;

  // Format log entry - always include pid/tid for consistency
  len = snprintf(buffer, sizeof(buffer),
                ":%d:%-25s:%-4d: %010" PRIu64 " us: [pid:%d tid:0x%x] %s\n",
                level, file, line, timestamp_us, pid, thread_id, message);

  if (len > 0 && len < (int)sizeof(buffer)) {
    output.insert(output.end(), buffer, buffer + len);
  }
}

void AsyncLogger::CleanupOrphanedBuffers() {
  std::lock_guard<std::mutex> lock(buffers_mutex_);

  // Remove buffers that are orphaned AND empty
  // We can safely delete them since no thread is writing to them
  auto it = std::remove_if(thread_buffers_.begin(), thread_buffers_.end(),
                          [](const std::unique_ptr<RingBuffer>& buf) {
                            return buf->IsOrphaned() && buf->IsEmpty();
                          });

  thread_buffers_.erase(it, thread_buffers_.end());
}

void async_log_printf(uint16_t level, uint32_t mask, const char* file,
                     uint32_t line, const char* format, ...) {
  va_list args;
  va_start(args, format);
  AsyncLogger::GetInstance().WriteLog(level, mask, file, line, format, args);
  va_end(args);
}

}  // namespace logging
}  // namespace amd

