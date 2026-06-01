/*
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef ROC_JPEG_COMMON_H_
#define ROC_JPEG_COMMON_H_

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

#define ROCJPEG_TOSTR(X) std::to_string(X)
#define ROCJPEG_STR(X) std::string(X)

// Logging control
enum RocJpegLogLevel {
    kRocJpegLogCritical       = 0,  // Only output critical messages
    kRocJpegLogError          = 1,  // Output critical and error messages
    kRocJpegLogWarning        = 2,  // Output critical, error and warning messages
    kRocJpegLogInfo           = 3,  // Output critical, error, warning and info messages
    kRocJpegLogDebug          = 4,  // Output critical, error, warning, info and debug messages
    kRocJpegLogLevelMax       = 4
};

#define GET_TIME_NS() ([]() -> uint64_t { struct timespec ts_; clock_gettime(CLOCK_MONOTONIC, &ts_); return static_cast<uint64_t>(ts_.tv_sec) * 1000000000LL + ts_.tv_nsec; }())
#define FILENAME_ONLY (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define GET_HASHED_THREAD_ID() ([]() -> std::string { std::ostringstream oss; oss << "0x" << std::hex << std::setw(5) << std::setfill('0') << (std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFF); return oss.str(); }())
#define GET_THREAD_ID() (static_cast<pid_t>(syscall(SYS_gettid)))
#define MakeMsg(msg) ROCJPEG_STR(FILENAME_ONLY) + ":" + ROCJPEG_TOSTR(__LINE__) + ": " + ROCJPEG_TOSTR(GET_TIME_NS() / 1000ULL) + ROCJPEG_STR(" us: ") + ROCJPEG_STR("[pid:") + ROCJPEG_TOSTR(getpid()) + ROCJPEG_STR(" tid:") + ROCJPEG_TOSTR(GET_THREAD_ID()) + ROCJPEG_STR(" hashid:") + GET_HASHED_THREAD_ID() + ROCJPEG_STR("] ") + ROCJPEG_STR(__func__) + "(): " + msg

#define OutputMsg(msg) std::cout << msg << std::endl
#define OutputErrMsg(msg) std::cerr << msg << std::endl

class RocJpegLogger {
public:
    RocJpegLogger() : log_level_(kRocJpegLogCritical) {
        char *env_log_level = std::getenv("ROCJPEG_LOG_LEVEL");
        if (env_log_level != nullptr) {
            log_level_ = std::clamp(std::atoi(env_log_level), 0, static_cast<int>(kRocJpegLogLevelMax));
        }
    }
    RocJpegLogger(int log_level) : log_level_(std::clamp(log_level, 0, static_cast<int>(kRocJpegLogLevelMax))) {};
    ~RocJpegLogger() {};
    void SetLogLevel(int log_level) {log_level_ = std::clamp(log_level, 0, static_cast<int>(kRocJpegLogLevelMax));};
    int GetLogLevel() {return log_level_;};
    void AlwaysLog(std::string msg) {
        OutputMsg(msg);
    };
private:
    int log_level_ = kRocJpegLogCritical;
};

// Single global logger instance shared across all components. Log level is
// controlled via the ROCJPEG_LOG_LEVEL environment variable (default: critical).
inline RocJpegLogger& RocJpegGetLogger() {
    static RocJpegLogger instance;
    return instance;
}
#define g_rocjpeg_logger (RocJpegGetLogger())

// RAII helper for function-scope entry/exit logging.
// Keeps the start timestamp per call-scope (stack variable), making it
// safe for nested calls and concurrent threads sharing the same logger.
class RocJpegFuncScopeLog {
public:
    RocJpegFuncScopeLog(RocJpegLogger& logger, const char* filename, int line, const char* func,
                        const std::string& args = "")
        : logger_(logger), filename_(filename), line_(line), func_(func), args_(args), start_time_(0) {
        if (logger_.GetLogLevel() >= kRocJpegLogInfo) {
            start_time_ = GET_TIME_NS() / 1000ULL;
            OutputMsg("[" + ROCJPEG_TOSTR(kRocJpegLogInfo) + ", Info] " + ROCJPEG_STR(filename_) + ":" + ROCJPEG_TOSTR(line_) + ": " +
                      ROCJPEG_TOSTR(start_time_) + ROCJPEG_STR(" us: ") + ROCJPEG_STR("[pid:") + ROCJPEG_TOSTR(getpid()) + ROCJPEG_STR(" tid:") +
                      ROCJPEG_TOSTR(GET_THREAD_ID()) + ROCJPEG_STR(" hashid:") + GET_HASHED_THREAD_ID() + ROCJPEG_STR("] ") + ROCJPEG_STR(func_) +
                      "( " + args_ + " ): entry ...");
        }
    }
    ~RocJpegFuncScopeLog() {
        if (logger_.GetLogLevel() >= kRocJpegLogInfo) {
            uint64_t end_time = GET_TIME_NS() / 1000ULL;
            OutputMsg("[" + ROCJPEG_TOSTR(kRocJpegLogInfo) + ", Info] " + ROCJPEG_STR(filename_) + ":" + ROCJPEG_TOSTR(line_) + ": " +
                      ROCJPEG_TOSTR(end_time) + ROCJPEG_STR(" us: ") + ROCJPEG_STR("[pid:") + ROCJPEG_TOSTR(getpid()) + ROCJPEG_STR(" tid:") +
                      ROCJPEG_TOSTR(GET_THREAD_ID()) + ROCJPEG_STR(" hashid:") + GET_HASHED_THREAD_ID() + ROCJPEG_STR("] ") + ROCJPEG_STR(func_) +
                      "( " + args_ + " ): exit (" + ROCJPEG_TOSTR(end_time - start_time_) + " us) ...");
        }
    }
    RocJpegFuncScopeLog(const RocJpegFuncScopeLog&) = delete;
    RocJpegFuncScopeLog& operator=(const RocJpegFuncScopeLog&) = delete;
    RocJpegFuncScopeLog(RocJpegFuncScopeLog&&) = delete;
    RocJpegFuncScopeLog& operator=(RocJpegFuncScopeLog&&) = delete;
private:
    RocJpegLogger& logger_;
    const char* filename_;
    int line_;
    const char* func_;
    std::string args_;
    uint64_t start_time_;
};

#define CriticalLog(logger, msg) \
    do { \
        if (logger.GetLogLevel() >= kRocJpegLogCritical) { \
            OutputErrMsg("[" + ROCJPEG_TOSTR(kRocJpegLogCritical) + ", Critical] " + MakeMsg(msg)); \
        } \
    } while (0)

#define ErrorLog(logger, msg) \
    do { \
        if (logger.GetLogLevel() >= kRocJpegLogError) { \
            OutputErrMsg("[" + ROCJPEG_TOSTR(kRocJpegLogError) + ", Error] " + MakeMsg(msg)); \
        } \
    } while (0)

#define WarningLog(logger, msg) \
    do { \
        if (logger.GetLogLevel() >= kRocJpegLogWarning) { \
            OutputErrMsg("[" + ROCJPEG_TOSTR(kRocJpegLogWarning) + ", Warning] " + MakeMsg(msg)); \
        } \
    } while (0)

#define InfoLog(logger, msg) \
    do { \
        if (logger.GetLogLevel() >= kRocJpegLogInfo) { \
            OutputErrMsg("[" + ROCJPEG_TOSTR(kRocJpegLogInfo) + ", Info] " + MakeMsg(msg)); \
        } \
    } while (0)

#define DebugLog(logger, msg) \
    do { \
        if (logger.GetLogLevel() >= kRocJpegLogDebug) { \
            OutputErrMsg("[" + ROCJPEG_TOSTR(kRocJpegLogDebug) + ", Debug] " + MakeMsg(msg)); \
        } \
    } while (0)

// Format a pointer argument as hex for API argument logging.
template<typename T>
static inline std::string RocJpegFmtPtr(T* p) {
    if (p == nullptr) return "nullptr";
    std::ostringstream oss;
    oss << "0x" << std::hex << reinterpret_cast<uintptr_t>(p);
    return oss.str();
}

#define FunctionEntryLog(logger) \
    RocJpegFuncScopeLog _rocjpeg_func_scope_log_(logger, FILENAME_ONLY, __LINE__, __func__)

// Use this variant at API boundaries to include argument values in the entry log line.
// Pass a string built with RocJpegFmtPtr() / ROCJPEG_TOSTR() for each argument, e.g.:
//   FunctionEntryLogWithArgs(g_rocjpeg_logger, RocJpegFmtPtr(handle) + ", " + RocJpegFmtPtr(dst))
#define FunctionEntryLogWithArgs(logger, args) \
    RocJpegFuncScopeLog _rocjpeg_func_scope_log_(logger, FILENAME_ONLY, __LINE__, __func__, (args))

// FunctionExitLog is a no-op: exit is logged automatically when the
// RocJpegFuncScopeLog RAII object created by FunctionEntryLog goes out of scope.
#define FunctionExitLog(logger)

#define CHECK_VAAPI(call) {                                               \
    VAStatus va_status = (call);                                          \
    if (va_status != VA_STATUS_SUCCESS) {                                 \
        std::cerr << "VAAPI failure: " << #call << " failed with status: " << std::hex << "0x" << va_status << std::dec << " = '" << vaErrorStr(va_status) << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
        return ROCJPEG_STATUS_EXECUTION_FAILED;                           \
    }                                                                     \
}

#define CHECK_HIP(call) {                                             \
    hipError_t hip_status = (call);                                   \
    if (hip_status != hipSuccess) {                                   \
        std::cerr << "HIP failure: 'status: " << hipGetErrorName(hip_status) << "' at " << __FILE__ << ":" << __LINE__ << std::endl;\
        return ROCJPEG_STATUS_EXECUTION_FAILED;                       \
    }                                                                 \
}

#define CHECK_ROCJPEG(call) {                                             \
    RocJpegStatus rocjpeg_status = (call);                                \
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {                       \
        std::cerr << #call << " returned " << rocJpegGetErrorName(rocjpeg_status) << " at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
        return rocjpeg_status;                                                          \
    }                                                                     \
}

static inline int align(int value, int alignment) {
   return (value + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Custom exception class for RocJpeg.
 *
 * This exception class is used to handle errors and exceptions that occur during RocJpeg operations.
 * It inherits from the std::exception class and provides an implementation for the what() function.
 */
class RocJpegException : public std::exception {
    public:
        /**
         * @brief Constructs a RocJpegException object with the specified error message.
         *
         * @param message The error message associated with the exception.
         */
        explicit RocJpegException(const std::string& message):message_(message){}

        /**
         * @brief Returns a C-style string describing the exception.
         *
         * This function overrides the what() function from the std::exception class and returns
         * the error message associated with the exception.
         *
         * @return A C-style string describing the exception.
         */
        virtual const char* what() const throw() override {
            return message_.c_str();
        }

    private:
        std::string message_;
};

#define THROW(X) throw RocJpegException(" { "+std::string(__func__)+" } " + X);

#endif //ROC_JPEG_COMMON_H_