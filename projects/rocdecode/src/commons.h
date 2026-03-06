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

#define TOSTR(X) std::to_string(static_cast<int>(X))
#define STR(X) std::string(X)

#if DBGINFO
#define INFO(X) std::clog << "[INF] " << " {" << __func__ <<"} " << " " << X << std::endl;
#define MSG(X) std::clog << X << std::endl;
#define MSG_NO_NEWLINE(X) std::clog << X;
#else
#define INFO(X) ;
#define MSG(X) ;
#define MSG_NO_NEWLINE(X) ;
#endif
#define ERR(X) std::cerr << "[ERR] "  << " {" << __func__ <<"} " << " " << X << std::endl;

// Logging control
enum RocDecLogLevel {
    kRocDecLogCritical       = 0,  // Only output critical messages
    kRocDecLogError          = 1,  // Output critical and error messages
    kRocDecLogWarning        = 2,  // Output critical, error and warning messages
    kRocDecLogInfo           = 3,  // Output critical, error, warning and info messages
    kRocDecLogDebug          = 4,  // Output critical, error, warning, info and debug messages
    kRocDecLogLevelMax       = 4
};

// Use full function signature so member-function logs show class name (e.g. RocDecoder::InitializeDecoder)
#if defined(__GNUC__) || defined(__clang__)
#define ROCDEC_FUNC_DISPLAY __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define ROCDEC_FUNC_DISPLAY __FUNCSIG__
#else
#define ROCDEC_FUNC_DISPLAY __func__
#endif
#define MakeMsg(msg) STR(ROCDEC_FUNC_DISPLAY) + ", Line " + TOSTR(__LINE__) + ": " + msg
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

    static void AlwaysLog(std::string msg) {
        OutputMsg(msg);
    };

    void CriticalLog(std::string msg) {
        if (log_level_ >= kRocDecLogCritical) {
            OutputErrMsg("[Critical] " + msg);
        }
    };

    void ErrorLog(std::string msg) {
        if (log_level_ >= kRocDecLogError) {
            OutputErrMsg("[Error] " + msg);
        }
    };

    void WarningLog(std::string msg) {
        if (log_level_ >= kRocDecLogWarning) {
            OutputErrMsg("[Warning] " + msg);
        }
    };

    void InfoLog(std::string msg) {
        if (log_level_ >= kRocDecLogInfo) {
            OutputMsg("[Info] " + msg);
        }
    };

    void DebugLog(std::string msg) {
        if (log_level_ >= kRocDecLogDebug) {
            OutputMsg("[Debug] " + msg);
        }
    };

    void FunctionStartLog(std::string msg) {
        if (log_level_ >= kRocDecLogInfo) {
            OutputMsg("[Info] " + msg + " entry ...");
        }
    };

    void FunctionEndLog(std::string msg) {
        if (log_level_ >= kRocDecLogInfo) {
            OutputMsg("[Info] " + msg + " exit ...");
        }
    };

private:
    int log_level_ = kRocDecLogCritical;
};

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
