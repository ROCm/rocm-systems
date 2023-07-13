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

#ifndef SRC_UTIL_LOGGER_H_
#define SRC_UTIL_LOGGER_H_

#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/file.h>
#include <stdarg.h>
#include <stdlib.h>

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <exception>
#include <mutex>
#include <map>

namespace rocprofiler {
namespace util {

class Logger {
 public:
  typedef std::recursive_mutex mutex_t;

  template <typename T> Logger& operator<<(const T& m) {
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

  static void begm() { Instance().ResetStreaming(true); }
  static void endl() { Instance().ResetStreaming(false); }
  static void errm() { Instance().SetError(); }

  static const std::string& LastMessage() {
    Logger& logger = Instance();
    std::lock_guard<mutex_t> lck(mutex_);
    return logger.message_[GetTid()];
  }

  static Logger* Create() {
    std::lock_guard<mutex_t> lck(mutex_);
    Logger* obj = instance_.load(std::memory_order_relaxed);
    if (obj == NULL) {
      obj = new Logger();
      if (obj == NULL) {
        std::cerr << "ROCProfiler: log object creation failed" << std::endl << std::flush;
        abort();
      }
      instance_.store(obj, std::memory_order_release);
    }
    return obj;
  }

  static void Destroy() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (instance_ != NULL) delete instance_.load();
    instance_ = NULL;
  }

  static Logger& Instance() {
    Logger* obj = instance_.load(std::memory_order_acquire);
    if (obj == NULL) obj = Create();
    return *obj;
  }

 private:
  static uint32_t GetPid() { return syscall(__NR_getpid); }
  static uint32_t GetTid() { return syscall(__NR_gettid); }

  Logger()
      : file_(NULL),
        session_file_(NULL),
        dirty_(false),
        streaming_(false),
        messaging_(false),
        error_(false) {
    const char* var = getenv("ROCPROFILER_LOG");
    if (var != NULL) file_ = fopen("/tmp/rocprofiler_log.txt", "a");

    var = getenv("ROCPROFILER_SESS");
    if (var != NULL) {
      std::string dir = var;
      if (dir.back() != '/') dir.push_back('/');
      std::string name = dir + "log.txt";
      session_file_ = fopen(name.c_str(), "a");
      if (session_file_ != NULL)
        session_dir_ = dir;
      else
        std::cerr << "ROCProfiler: cannot create session log '" << name << "'" << std::endl
                  << std::flush;
    }

    ResetStreaming(false);
  }

  ~Logger() {
    if (dirty_) Put("\n");
    if (file_ != NULL) fclose(file_);
    if (session_file_ != NULL) fclose(session_file_);
  }

  void ResetStreaming(const bool messaging) {
    std::lock_guard<mutex_t> lck(mutex_);
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
    std::lock_guard<mutex_t> lck(mutex_);
    if (messaging_) {
      message_[GetTid()] += m;
    }
    if (file_ != NULL) {
      dirty_ = true;
      flock(fileno(file_), LOCK_EX);

      fprintf(file_, "%s", m.c_str());
      fflush(file_);

      if (session_file_ != NULL) {
        fprintf(session_file_, "%s", m.c_str());
        fflush(session_file_);
      }

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
    oss << "<" << tm_str << std::dec << " pid" << GetPid() << " tid" << GetTid() << "> " << m;
    Put(oss.str());
  }

  void SetError() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (error_ == false) {
      error_ = true;
      if (session_dir_.empty() == false) {
        auto x = fopen(std::string(session_dir_ + "error").c_str(), "w");
        (void)x;
      }
    }
  }

  FILE* file_;
  FILE* session_file_;
  bool dirty_;
  bool streaming_;
  bool messaging_;
  bool error_;
  std::string session_dir_;
  std::map<uint32_t, std::string> message_;

  static mutex_t mutex_;
  static std::atomic<Logger*> instance_;
};

}  // namespace util
}  // namespace rocprofiler

#define ERR_LOGGING(stream)                                                                        \
  do {                                                                                             \
    rocprofiler::util::Logger::Instance()                                                          \
        << rocprofiler::util::Logger::errm << "error: " << rocprofiler::util::Logger::begm         \
        << stream << rocprofiler::util::Logger::endl;                                              \
  } while (0)

#define INFO_LOGGING(stream)                                                                       \
  do {                                                                                             \
    rocprofiler::util::Logger::Instance() << "info: " << rocprofiler::util::Logger::begm << stream \
                                          << rocprofiler::util::Logger::endl;                      \
  } while (0)

#define WARN_LOGGING(stream)                                                                       \
  do {                                                                                             \
    std::cerr << "ROCProfiler: " << stream << std::endl;                                           \
    rocprofiler::util::Logger::Instance() << "warning: " << rocprofiler::util::Logger::begm        \
                                          << stream << rocprofiler::util::Logger::endl;            \
  } while (0)

#ifdef DEBUG
#define DBG_LOGGING(stream)                                                                        \
  do {                                                                                             \
    rocprofiler::util::Logger::Instance()                                                          \
            << rocprofiler::util::Logger::begm << "debug: \"" << stream << "\"" < < < <            \
        " in " << __FUNCTION__ << " at " << __FILE__ << " line " << __LINE__                       \
               << rocprofiler::util::Logger::endl;                                                 \
  } while (0)
#endif

#endif  // SRC_UTIL_LOGGER_H_
