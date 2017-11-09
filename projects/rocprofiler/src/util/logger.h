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

  static const std::string& LastMessage() {
    Logger& logger = Instance();
    std::lock_guard<mutex_t> lck(mutex_);
    return logger.message_[GetTid()];
  }

  static Logger* Create() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (instance_ == NULL) instance_ = new Logger();
    return instance_;
  }

  static void Destroy() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (instance_ != NULL) delete instance_;
    instance_ = NULL;
  }

  static Logger& Instance() {
    Create();
    return *instance_;
  }

 private:
  static uint32_t GetPid() { return syscall(__NR_getpid); }
  static uint32_t GetTid() { return syscall(__NR_gettid); }

  Logger() : file_(NULL), dirty_(false), streaming_(false), messaging_(false) {
    const char* path = getenv("ROCPROFILER_LOG");
    if (path != NULL) {
      file_ = fopen("/tmp/rocprofiler_log.txt", "a");
    }
    ResetStreaming(false);
  }

  ~Logger() {
    if (file_ != NULL) {
      if (dirty_) Put("\n");
      fclose(file_);
    }
  }

  void ResetStreaming(const bool messaging) {
    std::lock_guard<mutex_t> lck(mutex_);
    if (messaging) {
      message_[GetTid()] = "";
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

  FILE* file_;
  bool dirty_;
  bool streaming_;
  bool messaging_;

  static mutex_t mutex_;
  static Logger* instance_;
  std::map<uint32_t, std::string> message_;
};

}  // namespace util
}  // namespace rocprofiler

#define ERR_LOGGING(stream) {                                                                      \
  rocprofiler::util::Logger::Instance()                                                            \
                                   << "error: "                                                    \
                                   << rocprofiler::util::Logger::begm                              \
                                   << stream                                                       \
                                   << rocprofiler::util::Logger::endl;                             \
}

#define INFO_LOGGING(stream) {                                                                     \
  rocprofiler::util::Logger::Instance()                                                            \
                                   << "info: "                                                     \
                                   << rocprofiler::util::Logger::begm                              \
                                   << stream                                                       \
                                   << rocprofiler::util::Logger::endl;                             \
}

#ifdef DEBUG
# define DBG_LOGGING(stream) {                                                                     \
  rocprofiler::util::Logger::Instance() << rocprofiler::util::Logger::begm                         \
                                   << "debug: \"" << stream << "\"" <<                             \
                                   << " in " << __FUNCTION__                                       \
                                   << " at " << __FILE__ << " line " << __LINE__                   \
                                   << rocprofiler::util::Logger::endl;                             \
}
#endif

#endif  // SRC_UTIL_LOGGER_H_
