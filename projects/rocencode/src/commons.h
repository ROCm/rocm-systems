/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
*/

#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <sys/syscall.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>

#define ROCENC_TOSTR(X) std::to_string(X)
#define ROCENC_STR(X) std::string(X)

// Logging control.  Mirrors rocDecode's scheme; the environment variable is
// ROCENC_LOG_LEVEL.
enum RocEncLogLevel {
    kRocEncLogCritical       = 0,  // Only output critical messages
    kRocEncLogError          = 1,  // Output critical and error messages
    kRocEncLogWarning        = 2,  // Output critical, error and warning messages
    kRocEncLogInfo           = 3,  // Output critical, error, warning and info messages
    kRocEncLogDebug          = 4,  // Output critical, error, warning, info and debug messages
    kRocEncLogLevelMax       = 4
};

#define GET_TIME_NS() ([]() -> uint64_t { struct timespec ts_; clock_gettime(CLOCK_MONOTONIC, &ts_); return static_cast<uint64_t>(ts_.tv_sec) * 1000000000LL + ts_.tv_nsec; }())
#define FILENAME_ONLY (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define GET_THREAD_ID() (static_cast<pid_t>(syscall(SYS_gettid)))
#define GET_PID() (getpid())
#define GET_HASHED_THREAD_ID() ([]() -> std::string { std::ostringstream oss; oss << "0x" << std::hex << std::setw(5) << std::setfill('0') << (std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFF); return oss.str(); }())
#define MakeMsg(msg) ROCENC_STR(FILENAME_ONLY) + ":" + ROCENC_TOSTR(__LINE__) + ": " + ROCENC_TOSTR(GET_TIME_NS() / 1000ULL) + ROCENC_STR(" us: ") + ROCENC_STR("[pid:") + ROCENC_TOSTR(GET_PID()) + ROCENC_STR(" tid:") + ROCENC_TOSTR(GET_THREAD_ID()) + ROCENC_STR(" hashid:") + GET_HASHED_THREAD_ID() + ROCENC_STR("] ") + ROCENC_STR(__func__) + "(): " + msg

#define OutputMsg(msg) std::cout << msg << std::endl
#define OutputErrMsg(msg) std::cerr << msg << std::endl

class RocEncLogger {
public:
    RocEncLogger() : log_level_(kRocEncLogCritical) {
        char *env_log_level = std::getenv("ROCENC_LOG_LEVEL");
        if (env_log_level != nullptr) {
            log_level_ = std::clamp(std::atoi(env_log_level), 0, static_cast<int>(kRocEncLogLevelMax));
        }
    }
    explicit RocEncLogger(int log_level) : log_level_(log_level) {}
    ~RocEncLogger() {}
    void SetLogLevel(int log_level) { log_level_ = std::clamp(log_level, 0, static_cast<int>(kRocEncLogLevelMax)); }
    int GetLogLevel() const { return log_level_; }

private:
    int log_level_ = kRocEncLogCritical;
};

// Single global logger instance shared across all components.  Log level is
// controlled via the ROCENC_LOG_LEVEL environment variable (default: critical).
// Meyer's singleton: initialized on first use, thread-safe by C++11 6.7.
inline RocEncLogger& RocEncGetLogger() {
    static RocEncLogger instance;
    return instance;
}
#define g_rocenc_logger (RocEncGetLogger())

// RAII helper for function-scope entry/exit logging.  The start timestamp is a
// stack variable, so nesting and concurrent threads are safe.
class RocEncFuncScopeLog {
public:
    RocEncFuncScopeLog(RocEncLogger& logger, const char* filename, int line, const char* func,
                       const std::string& args = "")
        : logger_(logger), filename_(filename), line_(line), func_(func), args_(args), start_time_(0) {
        if (logger_.GetLogLevel() >= kRocEncLogInfo) {
            start_time_ = GET_TIME_NS() / 1000ULL;
            OutputMsg("[" + ROCENC_TOSTR(static_cast<int>(kRocEncLogInfo)) + ", Info] " + ROCENC_STR(filename_) + ":" + ROCENC_TOSTR(line_) + ": " +
                      ROCENC_TOSTR(start_time_) + ROCENC_STR(" us: ") + ROCENC_STR("[pid:") + ROCENC_TOSTR(GET_PID()) + ROCENC_STR(" tid:") +
                      ROCENC_TOSTR(GET_THREAD_ID()) + ROCENC_STR(" hashid:") + GET_HASHED_THREAD_ID() + ROCENC_STR("] ") + ROCENC_STR(func_) +
                      "( " + args_ + " ): entry ...");
        }
    }
    ~RocEncFuncScopeLog() {
        if (start_time_ != 0 && logger_.GetLogLevel() >= kRocEncLogInfo) {
            uint64_t end_time = GET_TIME_NS() / 1000ULL;
            OutputMsg("[" + ROCENC_TOSTR(static_cast<int>(kRocEncLogInfo)) + ", Info] " + ROCENC_STR(filename_) + ":" + ROCENC_TOSTR(line_) + ": " +
                      ROCENC_TOSTR(end_time) + ROCENC_STR(" us: ") + ROCENC_STR("[pid:") + ROCENC_TOSTR(GET_PID()) + ROCENC_STR(" tid:") +
                      ROCENC_TOSTR(GET_THREAD_ID()) + ROCENC_STR(" hashid:") + GET_HASHED_THREAD_ID() + ROCENC_STR("] ") + ROCENC_STR(func_) +
                      "( " + args_ + " ): exit (" + ROCENC_TOSTR(end_time - start_time_) + " us) ...");
        }
    }
    RocEncFuncScopeLog(const RocEncFuncScopeLog&) = delete;
    RocEncFuncScopeLog& operator=(const RocEncFuncScopeLog&) = delete;
    RocEncFuncScopeLog(RocEncFuncScopeLog&&) = delete;
    RocEncFuncScopeLog& operator=(RocEncFuncScopeLog&&) = delete;

private:
    RocEncLogger& logger_;
    const char* filename_;
    int line_;
    const char* func_;
    std::string args_;
    uint64_t start_time_;
};

#define CriticalLog(logger, msg) \
    do { \
        if ((logger).GetLogLevel() >= kRocEncLogCritical) { \
            OutputErrMsg("[" + ROCENC_TOSTR(static_cast<int>(kRocEncLogCritical)) + ", Critical] " + MakeMsg(msg)); \
        } \
    } while (0)

#define ErrorLog(logger, msg) \
    do { \
        if ((logger).GetLogLevel() >= kRocEncLogError) { \
            OutputErrMsg("[" + ROCENC_TOSTR(static_cast<int>(kRocEncLogError)) + ", Error] " + MakeMsg(msg)); \
        } \
    } while (0)

#define WarningLog(logger, msg) \
    do { \
        if ((logger).GetLogLevel() >= kRocEncLogWarning) { \
            OutputErrMsg("[" + ROCENC_TOSTR(static_cast<int>(kRocEncLogWarning)) + ", Warning] " + MakeMsg(msg)); \
        } \
    } while (0)

#define InfoLog(logger, msg) \
    do { \
        if ((logger).GetLogLevel() >= kRocEncLogInfo) { \
            OutputErrMsg("[" + ROCENC_TOSTR(static_cast<int>(kRocEncLogInfo)) + ", Info] " + MakeMsg(msg)); \
        } \
    } while (0)

#define DebugLog(logger, msg) \
    do { \
        if ((logger).GetLogLevel() >= kRocEncLogDebug) { \
            OutputErrMsg("[" + ROCENC_TOSTR(static_cast<int>(kRocEncLogDebug)) + ", Debug] " + MakeMsg(msg)); \
        } \
    } while (0)

// Format a pointer argument as hex for API argument logging.
template<typename T>
static inline std::string RocEncFmtPtr(T* p) {
    if (p == nullptr) return "nullptr";
    std::ostringstream oss;
    oss << "0x" << std::hex << reinterpret_cast<uintptr_t>(p);
    return oss.str();
}

#define FunctionEntryLog(logger) \
    RocEncFuncScopeLog _rocenc_func_scope_log_(logger, FILENAME_ONLY, __LINE__, __func__)

// Use this variant at API boundaries to include argument values in the entry log line.
#define FunctionEntryLogWithArgs(logger, args) \
    RocEncFuncScopeLog _rocenc_func_scope_log_(logger, FILENAME_ONLY, __LINE__, __func__, \
        ((logger).GetLogLevel() >= kRocEncLogInfo) ? std::string(args) : std::string())

// FunctionExitLog is a no-op: exit is logged when the RAII object goes out of scope.
#define FunctionExitLog(logger)

class rocEncodeException : public std::exception {
public:
    explicit rocEncodeException(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }

private:
    std::string message_;
};

#define THROW(X) throw rocEncodeException(" { " + std::string(__func__) + " } " + X)

// Parse a comma-separated list of device indices, e.g. the value of
// ROCR_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES.  The argument is typically the
// result of std::getenv(); that pointer aliases the process environment and
// must not be written through.  The input is copied into a stream before
// tokenising so the caller's buffer is never modified.  The result is not
// sorted; callers apply their own ordering.
inline std::vector<int> ParseVisibleDevicesCsv(const char* env) {
    std::vector<int> devices;
    if (env == nullptr) {
        return devices;
    }
    std::istringstream csv{std::string(env)};
    std::string token;
    while (std::getline(csv, token, ',')) {
        if (!token.empty()) {
            devices.push_back(std::atoi(token.c_str()));
        }
    }
    return devices;
}
