/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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

// Include Quill headers BEFORE top.hpp to ensure Logger is fully defined
// before any forward declarations
#ifdef ROCCLR_USE_QUILL_LOGGING
#include "quill/Backend.h"
#include "quill/Frontend.h"
#include "quill/LogMacros.h"
#include "quill/Logger.h"
#include "quill/sinks/ConsoleSink.h"
#include "quill/sinks/FileSink.h"
#endif  // ROCCLR_USE_QUILL_LOGGING

#include "top.hpp"
#include "utils/debug.hpp"
#include "os/os.hpp"

#if !defined(AMD_LOG_LEVEL)
#include "utils/flags.hpp"
#endif

#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <thread>
#include <sstream>
#include <iomanip>
#include <inttypes.h>
#ifdef _WIN32
#include <windows.h>
#endif  // _WIN32

namespace amd {

// Legacy file output (used when Quill is disabled or as fallback)
FILE* outFile = stderr;

#ifdef ROCCLR_USE_QUILL_LOGGING
// Quill logger instance (static since it's only used in this file)
static ::quill::Logger* g_logger = nullptr;
static bool g_quill_initialized = false;

// Custom pattern formatter - just output the message since formatting is done in log calls
static constexpr char kLogPattern[] = "%(message)s";

// ================================================================================================
void init_logging() {
  if (g_quill_initialized) {
    return;
  }

  // Configure backend options
  quill::BackendOptions backend_options;
  // Use a shorter sleep duration for lower latency
  backend_options.sleep_duration = std::chrono::microseconds{100};
  // Disable Quill's internal logging to stderr
  backend_options.error_notifier = [](std::string const&) {};

  // Start the backend thread
  quill::Backend::start(backend_options);

  // Create sink based on AMD_LOG_LEVEL_FILE environment variable
  std::shared_ptr<quill::Sink> sink;

  // Check if a log file is specified
  bool use_file = false;
#if !defined(AMD_LOG_LEVEL)
  if (!flagIsDefault(AMD_LOG_LEVEL_FILE) && AMD_LOG_LEVEL_FILE[0] != '\0') {
    use_file = true;
  }
#endif

  if (use_file) {
#if !defined(AMD_LOG_LEVEL)
    // Create filename with process ID appended
    std::string filename = AMD_LOG_LEVEL_FILE;
    filename += "_" + std::to_string(Os::getProcessId());

    sink = quill::Frontend::create_or_get_sink<quill::FileSink>(
        filename,
        []() {
          quill::FileSinkConfig cfg;
          cfg.set_open_mode('a');  // Append mode
          return cfg;
        }(),
        quill::FileEventNotifier{});
#endif
  } else {
    // Use console (stderr) sink
    sink = quill::Frontend::create_or_get_sink<quill::ConsoleSink>("amd_console");
  }

  // Create pattern formatter options
  quill::PatternFormatterOptions formatter_options;
  formatter_options.format_pattern = kLogPattern;
  formatter_options.timestamp_pattern = "%H:%M:%S.%Qus";  // Time with microseconds

  // Create or get the logger
  g_logger = quill::Frontend::create_or_get_logger("amd", std::move(sink), formatter_options);

  // Set log level based on AMD_LOG_LEVEL
  quill::LogLevel quill_level = quill::LogLevel::Info;
  if (AMD_LOG_LEVEL >= LOG_EXTRA_DEBUG) {
    quill_level = quill::LogLevel::TraceL3;
  } else if (AMD_LOG_LEVEL >= LOG_DETAIL_DEBUG) {
    quill_level = quill::LogLevel::TraceL2;
  } else if (AMD_LOG_LEVEL >= LOG_DEBUG) {
    quill_level = quill::LogLevel::TraceL1;
  } else if (AMD_LOG_LEVEL >= LOG_INFO) {
    quill_level = quill::LogLevel::Info;
  } else if (AMD_LOG_LEVEL >= LOG_WARNING) {
    quill_level = quill::LogLevel::Warning;
  } else if (AMD_LOG_LEVEL >= LOG_ERROR) {
    quill_level = quill::LogLevel::Error;
  } else {
    quill_level = quill::LogLevel::Critical;
  }
  g_logger->set_log_level(quill_level);

  // Register an atexit handler to shut down logging before global destructors run
  // This prevents crashes from Quill's backend thread being destroyed during process exit
  static bool atexit_registered = false;
  if (!atexit_registered) {
    std::atexit([]() {
      if (g_quill_initialized && g_logger) {
        // Just flush and mark as uninitialized, let the process exit handle cleanup
        g_logger->flush_log();
        g_logger = nullptr;
        g_quill_initialized = false;
      }
    });
    atexit_registered = true;
  }

  g_quill_initialized = true;
}

// ================================================================================================
void shutdown_logging() {
  if (!g_quill_initialized) {
    return;
  }

  // During process shutdown, don't try to flush or stop the backend as this can cause
  // segfaults due to race conditions with threads being torn down or file handles being closed.
  // The OS will handle closing file handles and cleaning up resources.
  // Just mark as not initialized so future log calls won't try to use the logger.
  g_logger = nullptr;
  g_quill_initialized = false;
}

// ================================================================================================
bool is_quill_initialized() { return g_quill_initialized; }

// ================================================================================================
quill::Logger* get_logger() { return g_logger; }

#else  // !ROCCLR_USE_QUILL_LOGGING

// Stub implementations when Quill is not enabled
void init_logging() {
  // No-op: use legacy fprintf-based logging
}

void shutdown_logging() {
  // No-op
}

bool is_quill_initialized() { return false; }

#endif  // ROCCLR_USE_QUILL_LOGGING

// ================================================================================================
void truncate_log_file() {
  if (outFile != stderr) {
    fseek(outFile, 0, SEEK_END);
    long size = ftell(outFile);

    const size_t maxLogSize = AMD_LOG_LEVEL_SIZE * Mi;
    if (size > maxLogSize) {
      if (nullptr == freopen(NULL, "w", outFile)) {
        outFile = stderr;
      }
    }
  }
}

// ================================================================================================
void report_warning(const char* message) {
#ifdef ROCCLR_USE_QUILL_LOGGING
  if (g_quill_initialized && g_logger) {
    LOG_WARNING(g_logger, "Warning: {}", message);
    return;
  }
#endif
  truncate_log_file();
  fprintf(outFile, "Warning: %s\n", message);
}

// ================================================================================================
void log_entry(LogLevel level, const char* file, int line, const char* message) {
  if (level == LOG_NONE) {
    return;
  }

#ifdef ROCCLR_USE_QUILL_LOGGING
  if (g_quill_initialized && g_logger) {
    switch (level) {
      case LOG_ERROR:
        LOG_ERROR(g_logger, "{}:{}: {}", file, line, message);
        break;
      case LOG_WARNING:
        LOG_WARNING(g_logger, "{}:{}: {}", file, line, message);
        break;
      case LOG_INFO:
        LOG_INFO(g_logger, "{}:{}: {}", file, line, message);
        break;
      default:
        LOG_DEBUG(g_logger, "{}:{}: {}", file, line, message);
        break;
    }
    return;
  }
#endif

  truncate_log_file();
  fprintf(outFile, ":%d:%s:%d: %s\n", level, file, line, message);
  fflush(outFile);
}

// ================================================================================================
void log_timestamped(LogLevel level, const char* file, int line, const char* message) {
  static bool gotstart = false;  // not thread-safe, but not scary if fails
  static uint64_t start;

  if (!gotstart) {
    start = Os::timeNanos();
    gotstart = true;
  }

  uint64_t time = Os::timeNanos() - start;
  if (level == LOG_NONE) {
    return;
  }

#ifdef ROCCLR_USE_QUILL_LOGGING
  if (g_quill_initialized && g_logger) {
    uint64_t timeUs = time / 1000ULL;
    switch (level) {
      case LOG_ERROR:
        LOG_ERROR(g_logger, "{}:{}: ({:010}) us {}", file, line, timeUs, message);
        break;
      case LOG_WARNING:
        LOG_WARNING(g_logger, "{}:{}: ({:010}) us {}", file, line, timeUs, message);
        break;
      case LOG_INFO:
        LOG_INFO(g_logger, "{}:{}: ({:010}) us {}", file, line, timeUs, message);
        break;
      default:
        LOG_DEBUG(g_logger, "{}:{}: ({:010}) us {}", file, line, timeUs, message);
        break;
    }
    return;
  }
#endif

  truncate_log_file();
  fprintf(outFile, ":% 2d:%15s:% 5d: (%010lld) us %s\n", level, file, line,
          static_cast<long long>(time / 1000ULL), message);
  fflush(outFile);
}

// ================================================================================================
void log_printf(LogLevel level, const char* file, int line, const char* format, ...) {
  va_list ap;
  std::stringstream pidtid;
  if (AMD_LOG_LEVEL >= 4) {
    pidtid << "[pid:" << Os::getProcessId() << " tid: 0x";
    pidtid << std::hex << std::setw(5) << std::this_thread::get_id() << "]";
  }

  va_start(ap, format);
  char message[4096];
  vsnprintf(message, sizeof(message), format, ap);
  va_end(ap);
  uint64_t timeUs = Os::timeNanos() / 1000ULL;

#ifdef ROCCLR_USE_QUILL_LOGGING
  if (g_quill_initialized && g_logger) {
    // Log the pre-formatted message through Quill
    // Note: We use the pre-formatted message to maintain compatibility with printf-style calls
    // Once all call sites are migrated to {fmt} style, this can be simplified
    switch (level) {
      case LOG_NONE:
        // LOG_NONE is used by guarantee() before abort - use critical level
        LOG_CRITICAL(g_logger, "{:<25}:{:<4}: {:010} us: {} {}", file, line, timeUs,
                     pidtid.str(), message);
        g_logger->flush_log();  // Ensure message is written before abort
        break;
      case LOG_ERROR:
        LOG_ERROR(g_logger, "{:<25}:{:<4}: {:010} us: {} {}", file, line, timeUs, pidtid.str(),
                  message);
        break;
      case LOG_WARNING:
        LOG_WARNING(g_logger, "{:<25}:{:<4}: {:010} us: {} {}", file, line, timeUs, pidtid.str(),
                    message);
        break;
      case LOG_INFO:
        LOG_INFO(g_logger, "{:<25}:{:<4}: {:010} us: {} {}", file, line, timeUs, pidtid.str(),
                 message);
        break;
      default:
        LOG_DEBUG(g_logger, "{:<25}:{:<4}: {:010} us: {} {}", file, line, timeUs, pidtid.str(),
                  message);
        break;
    }
    return;
  }
#endif

  truncate_log_file();

  fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s\n", level, file, line, timeUs,
          pidtid.str().c_str(), message);

  fflush(outFile);
}

// ================================================================================================
void log_printf(LogLevel level, const char* file, int line, uint64_t* start, const char* format,
                ...) {
  va_list ap;
  std::stringstream pidtid;
  if (AMD_LOG_LEVEL >= 4) {
    pidtid << "[pid:" << Os::getProcessId() << " tid: 0x";
    pidtid << std::hex << std::setw(5) << std::this_thread::get_id() << "]";
  }
  va_start(ap, format);
  char message[4096];
  vsnprintf(message, sizeof(message), format, ap);
  va_end(ap);
  uint64_t timeUs = Os::timeNanos() / 1000ULL;

#ifdef ROCCLR_USE_QUILL_LOGGING
  if (g_quill_initialized && g_logger) {
    uint64_t duration = (start != nullptr && *start != 0) ? (timeUs - *start) : 0;
    bool has_duration = (start != nullptr && *start != 0);

    auto log_with_duration = [&](auto log_fn) {
      if (has_duration) {
        log_fn(g_logger, "{:<25}:{:<4}: {:010} us: {} {}: duration: {} us", file, line, timeUs,
               pidtid.str(), message, duration);
      } else {
        log_fn(g_logger, "{:<25}:{:<4}: {:010} us: {} {}", file, line, timeUs, pidtid.str(),
               message);
      }
    };

    switch (level) {
      case LOG_NONE:
        if (has_duration) {
          LOG_CRITICAL(g_logger, "{:<25}:{:<4}: {:010} us: {} {}: duration: {} us", file, line,
                       timeUs, pidtid.str(), message, duration);
        } else {
          LOG_CRITICAL(g_logger, "{:<25}:{:<4}: {:010} us: {} {}", file, line, timeUs,
                       pidtid.str(), message);
        }
        g_logger->flush_log();
        break;
      case LOG_ERROR:
        if (has_duration) {
          LOG_ERROR(g_logger, "{:<25}:{:<4}: {:010} us: {} {}: duration: {} us", file, line,
                    timeUs, pidtid.str(), message, duration);
        } else {
          LOG_ERROR(g_logger, "{:<25}:{:<4}: {:010} us: {} {}", file, line, timeUs, pidtid.str(),
                    message);
        }
        break;
      case LOG_WARNING:
        if (has_duration) {
          LOG_WARNING(g_logger, "{:<25}:{:<4}: {:010} us: {} {}: duration: {} us", file, line,
                      timeUs, pidtid.str(), message, duration);
        } else {
          LOG_WARNING(g_logger, "{:<25}:{:<4}: {:010} us: {} {}", file, line, timeUs,
                      pidtid.str(), message);
        }
        break;
      case LOG_INFO:
        if (has_duration) {
          LOG_INFO(g_logger, "{:<25}:{:<4}: {:010} us: {} {}: duration: {} us", file, line,
                   timeUs, pidtid.str(), message, duration);
        } else {
          LOG_INFO(g_logger, "{:<25}:{:<4}: {:010} us: {} {}", file, line, timeUs, pidtid.str(),
                   message);
        }
        break;
      default:
        if (has_duration) {
          LOG_DEBUG(g_logger, "{:<25}:{:<4}: {:010} us: {} {}: duration: {} us", file, line,
                    timeUs, pidtid.str(), message, duration);
        } else {
          LOG_DEBUG(g_logger, "{:<25}:{:<4}: {:010} us: {} {}", file, line, timeUs, pidtid.str(),
                    message);
        }
        break;
    }

    if (start != nullptr && *start == 0) {
      *start = timeUs;
    }
    return;
  }
#endif

  truncate_log_file();

  if (start == 0 || *start == 0) {
    fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s\n", level, file, line, timeUs,
            pidtid.str().c_str(), message);
  } else {
    fprintf(outFile, ":%d:%-25s:%-4d: %010" PRIu64 " us: %s %s: duration: %" PRIu64 " us\n", level,
            file, line, timeUs, pidtid.str().c_str(), message, timeUs - *start);
  }
  fflush(outFile);
  if (*start == 0) {
    *start = timeUs;
  }
}

}  // namespace amd
