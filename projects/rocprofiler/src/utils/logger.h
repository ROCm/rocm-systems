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

#ifndef SRC_UTILS_LOGGER_H_
#define SRC_UTILS_LOGGER_H_

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace rocprofiler {

class Logger {
 public:
  template <typename T> Logger& operator<<(T&& m) {
    std::ostringstream oss;
    oss << std::forward<T>(m);
    if (!streaming_)
      Log(oss.str());
    else
      Put(oss.str());
    streaming_ = true;
    return *this;
  }

  using manip_t = void (*)();
  Logger& operator<<(manip_t f) {
    f();
    return *this;
  }

  static void begm() { Instance().ResetStreaming(true); }
  static void endl() { Instance().ResetStreaming(false); }

  const std::string& LastMessage() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return message_[GetTid()];
  }

  static Logger& Instance() {
    static Logger instance;
    return instance;
  }

  static uint32_t GetPid() { return syscall(__NR_getpid); }
  static uint32_t GetTid() { return syscall(__NR_gettid); }

 private:
  Logger() : file_(nullptr), dirty_(false), streaming_(false), messaging_(false) {
    const char* var = getenv("ROCPROFILER_LOG");
    if (var != nullptr) file_ = fopen("/tmp/rocprofiler_log.txt", "a");
    ResetStreaming(false);
  }

  ~Logger() {
    if (file_ != nullptr) {
      if (dirty_) Put("\n");
      fclose(file_);
    }
  }

  void ResetStreaming(const bool messaging) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (messaging) {
      message_[GetTid()] = "";
    } else if (streaming_) {
      Put("\n");
      dirty_ = false;
    }
    messaging_ = messaging;
    streaming_ = messaging;
  }

  void Put(const std::string& m) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (messaging_) {
      message_[GetTid()] += m;
    }
    if (file_ != nullptr) {
      dirty_ = true;
      flock(fileno(file_), LOCK_EX);
      fprintf(file_, "%s", m.c_str());
      fflush(file_);
      flock(fileno(file_), LOCK_UN);
    }
  }

  void Log(const std::string& m) {
    const time_t rawtime = time(nullptr);
    tm tm_info;
    localtime_r(&rawtime, &tm_info);
    char tm_str[26];
    strftime(tm_str, 26, "%Y-%m-%d %H:%M:%S", &tm_info);
    std::ostringstream oss;
    oss << "<" << tm_str << std::dec << " pid" << GetPid() << " tid" << GetTid() << "> " << m;
    Put(oss.str());
  }

  FILE* file_;
  bool dirty_;
  bool streaming_;
  bool messaging_;

  std::recursive_mutex mutex_;
  std::map<uint32_t, std::string> message_;
};

}  // namespace rocprofiler

#define FATAL_LOGGING(stream)                                                                      \
  do {                                                                                             \
    rocprofiler::Logger::Instance()                                                                \
        << "fatal: " << rocprofiler::Logger::begm << stream << rocprofiler::Logger::endl;          \
    throw(ROCPROFILER_STATUS_ERROR, "Error: " << stream);                                          \
  } while (false)

#define ERR_LOGGING(stream)                                                                        \
  do {                                                                                             \
    rocprofiler::Logger::Instance()                                                                \
        << "error: " << rocprofiler::Logger::begm << stream << rocprofiler::Logger::endl;          \
  } while (false)

#define INFO_LOGGING(stream)                                                                       \
  do {                                                                                             \
    rocprofiler::Logger::Instance()                                                                \
        << "info: " << rocprofiler::Logger::begm << stream << rocprofiler::Logger::endl;           \
  } while (false)

#define WARN_LOGGING(stream)                                                                       \
  do {                                                                                             \
    std::cerr << "ROCProfiler: " << stream << std::endl;                                           \
    rocprofiler::Logger::Instance()                                                                \
        << "warning: " << rocprofiler::Logger::begm << stream << rocprofiler::Logger::endl;        \
  } while (false)

#endif  // SRC_UTILS_LOGGER_H_
