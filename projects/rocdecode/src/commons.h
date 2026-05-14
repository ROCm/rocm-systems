/*
Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include <stdexcept>
#include <exception>
#include <string>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <thread>
#include <sstream>
#include <iomanip>
#include <sys/syscall.h>

#define MSG(X) std::clog << X << std::endl;
#define MSG_NO_NEWLINE(X) std::clog << X;

#define ROCDEC_TOSTR(X) std::to_string(X)
#define ROCDEC_STR(X) std::string(X)

// Logging control
enum RocDecLogLevel {
    kRocDecLogCritical       = 0,  // Only output critical messages
    kRocDecLogError          = 1,  // Output critical and error messages
    kRocDecLogWarning        = 2,  // Output critical, error and warning messages
    kRocDecLogInfo           = 3,  // Output critical, error, warning and info messages
    kRocDecLogDebug          = 4,  // Output critical, error, warning, info and debug messages
    kRocDecLogLevelMax       = 4
};

#define GET_TIME_NS() ([]() -> uint64_t { struct timespec ts_; clock_gettime(CLOCK_MONOTONIC, &ts_); return static_cast<uint64_t>(ts_.tv_sec) * 1000000000LL + ts_.tv_nsec; }())
#define FILENAME_ONLY (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define GET_HASHED_THREAD_ID() ([]() -> std::string { std::ostringstream oss; oss << "0x" << std::hex << std::setw(5) << std::setfill('0') << (std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFF); return oss.str(); }())
#define GET_THREAD_ID() (static_cast<pid_t>(syscall(SYS_gettid)))
#define MakeMsg(msg) ROCDEC_STR(FILENAME_ONLY) + ":" + ROCDEC_TOSTR(__LINE__) + ": " + ROCDEC_TOSTR(GET_TIME_NS() / 1000ULL) + ROCDEC_STR(" us: ") + ROCDEC_STR("[pid:") + ROCDEC_TOSTR(getpid()) + ROCDEC_STR(" tid:") + ROCDEC_TOSTR(GET_THREAD_ID()) + ROCDEC_STR(" hashid:") + GET_HASHED_THREAD_ID() + ROCDEC_STR("] ") + ROCDEC_STR(__func__) + "(): " + msg

#define OutputMsg(msg) std::cout << msg << std::endl
#define OutputErrMsg(msg) std::cerr << msg << std::endl

class RocDecLogger {
public:
    RocDecLogger() : log_level_(kRocDecLogCritical) {
        char *env_log_level = std::getenv("ROCDEC_LOG_LEVEL");
        if (env_log_level != nullptr) {
            log_level_ = std::clamp(std::atoi(env_log_level), 0, static_cast<int>(kRocDecLogLevelMax));
        }
    }
    RocDecLogger(int log_level) : log_level_(log_level) {};
    ~RocDecLogger() {};
    void SetLogLevel(int log_level) {log_level_ = std::clamp(log_level, 0, static_cast<int>(kRocDecLogLevelMax));};
    int GetLogLevel() {return log_level_;};
    void AlwaysLog(std::string msg) {
        OutputMsg(msg);
    };
private:
    int log_level_ = kRocDecLogCritical;
};

// Single global logger instance shared across all components. Log level is
// controlled via the ROCDEC_LOG_LEVEL environment variable (default: critical).
// Meyer's singleton: initialized on first use (avoids static init order fiasco),
// thread-safe by C++11 §6.7.
inline RocDecLogger& RocDecGetLogger() {
    static RocDecLogger instance;
    return instance;
}
#define g_rocdec_logger (RocDecGetLogger())

// RAII helper for function-scope entry/exit logging.
// Keeps the start timestamp per call-scope (stack variable), making it
// safe for nested calls and concurrent threads sharing the same logger.
class RocDecFuncScopeLog {
public:
    RocDecFuncScopeLog(RocDecLogger& logger, const char* filename, int line, const char* func)
        : logger_(logger), filename_(filename), line_(line), func_(func), start_time_(0) {
        if (logger_.GetLogLevel() >= kRocDecLogInfo) {
            start_time_ = GET_TIME_NS() / 1000ULL;
            OutputMsg("[" + ROCDEC_TOSTR(kRocDecLogInfo) + ", Info] " + ROCDEC_STR(filename_) + ":" + ROCDEC_TOSTR(line_) + ": " +
                      ROCDEC_TOSTR(start_time_) + ROCDEC_STR(" us: ") + ROCDEC_STR("[pid:") + ROCDEC_TOSTR(getpid()) + ROCDEC_STR(" tid:") +
                      ROCDEC_TOSTR(GET_THREAD_ID()) + ROCDEC_STR(" hashid:") + GET_HASHED_THREAD_ID() + ROCDEC_STR("] ") + ROCDEC_STR(func_) + "(): entry ...");
        }
    }
    ~RocDecFuncScopeLog() {
        if (logger_.GetLogLevel() >= kRocDecLogInfo) {
            uint64_t end_time = GET_TIME_NS() / 1000ULL;
            OutputMsg("[" + ROCDEC_TOSTR(kRocDecLogInfo) + ", Info] " + ROCDEC_STR(filename_) + ":" + ROCDEC_TOSTR(line_) + ": " +
                      ROCDEC_TOSTR(end_time) + ROCDEC_STR(" us: ") + ROCDEC_STR("[pid:") + ROCDEC_TOSTR(getpid()) + ROCDEC_STR(" tid:") +
                      ROCDEC_TOSTR(GET_THREAD_ID()) + ROCDEC_STR(" hashid:") + GET_HASHED_THREAD_ID() + ROCDEC_STR("] ") + ROCDEC_STR(func_) + "(): exit (" +
                      ROCDEC_TOSTR(end_time - start_time_) + " us) ...");
        }
    }
    RocDecFuncScopeLog(const RocDecFuncScopeLog&) = delete;
    RocDecFuncScopeLog& operator=(const RocDecFuncScopeLog&) = delete;
    RocDecFuncScopeLog(RocDecFuncScopeLog&&) = delete;
    RocDecFuncScopeLog& operator=(RocDecFuncScopeLog&&) = delete;
private:
    RocDecLogger& logger_;
    const char* filename_;
    int line_;
    const char* func_;
    uint64_t start_time_;
};

#define CriticalLog(logger, msg) \
    do { \
        if (logger.GetLogLevel() >= kRocDecLogCritical) { \
            OutputErrMsg("[" + ROCDEC_TOSTR(kRocDecLogCritical) + ", Critical] " + MakeMsg(msg)); \
        } \
    } while (0)

#define ErrorLog(logger, msg) \
    do { \
        if (logger.GetLogLevel() >= kRocDecLogError) { \
            OutputErrMsg("[" + ROCDEC_TOSTR(kRocDecLogError) + ", Error] " + MakeMsg(msg)); \
        } \
    } while (0)

#define WarningLog(logger, msg) \
    do { \
        if (logger.GetLogLevel() >= kRocDecLogWarning) { \
            OutputErrMsg("[" + ROCDEC_TOSTR(kRocDecLogWarning) + ", Warning] " + MakeMsg(msg)); \
        } \
    } while (0)

#define InfoLog(logger, msg) \
    do { \
        if (logger.GetLogLevel() >= kRocDecLogInfo) { \
            OutputErrMsg("[" + ROCDEC_TOSTR(kRocDecLogInfo) + ", Info] " + MakeMsg(msg)); \
        } \
    } while (0)

#define DebugLog(logger, msg) \
    do { \
        if (logger.GetLogLevel() >= kRocDecLogDebug) { \
            OutputErrMsg("[" + ROCDEC_TOSTR(kRocDecLogDebug) + ", Debug] " + MakeMsg(msg)); \
        } \
    } while (0)

#define FunctionEntryLog(logger) \
    RocDecFuncScopeLog _rocdec_func_scope_log_(logger, FILENAME_ONLY, __LINE__, __func__)

// FunctionExitLog is a no-op: exit is logged automatically when the
// RocDecFuncScopeLog RAII object created by FunctionEntryLog goes out of scope.
#define FunctionExitLog(logger)

class rocDecodeException : public std::exception {
public:
    explicit rocDecodeException(const std::string& OutputMsg):_message(OutputMsg){}
    virtual const char* what() const throw() override {
        return _message.c_str();
    }
private:
    std::string _message;
};

#define THROW(X) throw rocDecodeException(" { "+std::string(__func__)+" } " + X);
