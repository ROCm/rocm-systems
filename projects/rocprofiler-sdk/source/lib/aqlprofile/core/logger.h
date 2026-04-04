// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_CORE_LOGGER_H_
#define SRC_CORE_LOGGER_H_

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

namespace aql_profile {

class Logger {
 public:
  typedef std::recursive_mutex mutex_t;

  template <typename T>
  Logger& operator<<(const T& m) {
    std::ostringstream oss;
    oss << m;
    if (!streaming_)
      Log(oss.str());
    else
      Put(oss.str());
    streaming_ = true;
    return *this;
  }

  typedef void (*manip_t)();
  Logger& operator<<(manip_t f) {
    f();
    return *this;
  }

  static void begm() { Instance().messaging_ = true; }
  static void endl() { Instance().ResetStreaming(); }

  static const std::string& LastMessage() {
    Logger& logger = Instance();
    std::lock_guard<mutex_t> lck(mutex_);
    return logger.message_[GetTid()];
  }

  static Logger& Instance() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (instance_ == NULL) instance_ = new Logger();
    return *instance_;
  }

  static void Destroy() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (instance_ != NULL) delete instance_;
    instance_ = NULL;
  }

 private:
  static uint32_t GetPid() { return syscall(__NR_getpid); }
  static uint32_t GetTid() { return syscall(__NR_gettid); }

  Logger() : file_(NULL), dirty_(false), streaming_(false), messaging_(false) {
    const char* path = getenv("HSA_VEN_AMD_AQLPROFILE_LOG");
    if (path != NULL) {
      file_ = fopen("/tmp/aql_profile_log.txt", "a");
    }
    ResetStreaming();
  }

  ~Logger() {
    if (file_ != NULL) {
      if (dirty_) Put("\n");
      fclose(file_);
    }
  }

  void ResetStreaming() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (messaging_) {
      message_[GetTid()] = "";
    }
    messaging_ = false;
    streaming_ = false;
  }

  void Put(const std::string& m) {
    std::lock_guard<mutex_t> lck(mutex_);
    if (messaging_) {
      message_[GetTid()] += m;
    }
    if (file_ != NULL) {
      dirty_ = true;
      flock(fileno(file_), LOCK_EX);
      fprintf(file_, "%s", m.c_str());
      fflush(file_);
      flock(fileno(file_), LOCK_UN);
    }
  }

  void Log(const std::string& m) {
    const time_t rawtime = time(NULL);
    tm tm_info;
    localtime_r(&rawtime, &tm_info);
    char tm_str[26];
    strftime(tm_str, 26, "%Y-%m-%d %H:%M:%S", &tm_info);
    std::ostringstream oss;
    oss << "\n<" << tm_str << std::dec << " pid" << GetPid() << " tid" << GetTid() << "> " << m;
    Put(oss.str());
  }

  FILE* file_;
  bool dirty_;
  bool streaming_;
  bool messaging_;

  static mutex_t mutex_;
  static Logger* instance_;
  std::map<uint32_t, std::string> message_;
};

}  // namespace aql_profile

#define ERR_LOGGING                \
  (aql_profile::Logger::Instance() \
   << aql_profile::Logger::endl    \
   << "Error: " << __FUNCTION__ << "(): " << aql_profile::Logger::begm)
#define ERR2_LOGGING                                            \
  (aql_profile::Logger::Instance() << aql_profile::Logger::endl \
                                   << "Error: " << __FUNCTION__ << "(): ")
#define INFO_LOGGING               \
  (aql_profile::Logger::Instance() \
   << aql_profile::Logger::endl    \
   << "Info: " << __FUNCTION__ << "(): " << aql_profile::Logger::begm)

#define WARN_LOGGING               \
  (aql_profile::Logger::Instance() \
   << aql_profile::Logger::endl    \
   << "Warning: " << __FUNCTION__ << "(): " << aql_profile::Logger::begm)

#ifdef DEBUG
#define DBG_LOGGING                                                                      \
  (aql_profile::Logger::Instance() << aql_profile::Logger::endl                          \
                                   << "Debug: in " << __FUNCTION__ << " at " << __FILE__ \
                                   << " line " << __LINE__ << aql_profile::Logger::begm)
#endif

#endif  // SRC_CORE_LOGGER_H_
