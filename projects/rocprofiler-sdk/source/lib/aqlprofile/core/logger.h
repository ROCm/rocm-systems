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

#include <fmt/format.h>

#include "lib/common/utility.hpp"

namespace aql_profile
{
class Logger
{
public:
    typedef std::recursive_mutex mutex_t;

    template <typename T>
    Logger& operator<<(const T& m)
    {
        std::ostringstream oss;
        oss << m;
        std::lock_guard<mutex_t> lck(mutex_);
        if(!streaming_)
            Log(oss.str());
        else
            Put(oss.str());
        streaming_ = true;
        return *this;
    }

    typedef void (*manip_t)();
    Logger& operator<<(manip_t f)
    {
        std::lock_guard<mutex_t> lck(mutex_);
        f();
        return *this;
    }

    static void begm() { Instance().messaging_ = true; }
    static void endl() { Instance().ResetStreaming(); }

    static const std::string& LastMessage()
    {
        Logger&                  logger = Instance();
        std::lock_guard<mutex_t> lck(mutex_);
        return logger.message_[rocprofiler::common::get_tid()];
    }

    static Logger& Instance()
    {
        std::lock_guard<mutex_t> lck(mutex_);
        if(instance_ == nullptr) instance_ = new Logger();
        return *instance_;
    }

    static void Destroy()
    {
        std::lock_guard<mutex_t> lck(mutex_);
        delete instance_;
        instance_ = nullptr;
    }

private:
    Logger()
    {
        const char* path = getenv("ROCPROFILER_AQLPROFILE_LOGFILE");
        if(path != nullptr)
        {
            auto log_path = fmt::format(
                "/tmp/aql_profile_log_{}_{}.txt", getppid(), rocprofiler::common::get_pid());
            file_ = fopen(log_path.c_str(), "a");
        }
        ResetStreaming();
    }

    ~Logger()
    {
        if(file_ != nullptr)
        {
            if(dirty_) Put("\n");
            fclose(file_);
        }
    }

    void ResetStreaming()
    {
        std::lock_guard<mutex_t> lck(mutex_);
        if(messaging_)
        {
            message_[rocprofiler::common::get_tid()] = "";
        }
        messaging_ = false;
        streaming_ = false;
    }

    void Put(const std::string& m)
    {
        std::lock_guard<mutex_t> lck(mutex_);
        if(messaging_)
        {
            message_[rocprofiler::common::get_tid()] += m;
        }
        if(file_ != nullptr)
        {
            dirty_ = true;
            flock(fileno(file_), LOCK_EX);
            fprintf(file_, "%s", m.c_str());
            fflush(file_);
            flock(fileno(file_), LOCK_UN);
        }
    }

    void Log(const std::string& m)
    {
        const time_t rawtime = time(nullptr);
        tm           tm_info;
        localtime_r(&rawtime, &tm_info);
        char tm_str[26];
        strftime(tm_str, 26, "%Y-%m-%d %H:%M:%S", &tm_info);
        std::ostringstream oss;
        oss << "\n<" << tm_str << std::dec << " pid" << rocprofiler::common::get_pid() << " tid"
            << rocprofiler::common::get_tid() << "> " << m;
        Put(oss.str());
    }

    bool                            dirty_     = false;
    bool                            streaming_ = false;
    bool                            messaging_ = false;
    FILE*                           file_      = nullptr;
    std::map<uint64_t, std::string> message_   = {};

    static mutex_t mutex_;
    static Logger* instance_;
};

}  // namespace aql_profile

#define ERR_LOGGING                                                                                \
    (aql_profile::Logger::Instance()                                                               \
     << aql_profile::Logger::endl                                                                  \
     << "Error: " << __FUNCTION__ << "(): " << aql_profile::Logger::begm)
#define ERR2_LOGGING                                                                               \
    (aql_profile::Logger::Instance() << aql_profile::Logger::endl                                  \
                                     << "Error: " << __FUNCTION__ << "(): ")
#define INFO_LOGGING                                                                               \
    (aql_profile::Logger::Instance()                                                               \
     << aql_profile::Logger::endl                                                                  \
     << "Info: " << __FUNCTION__ << "(): " << aql_profile::Logger::begm)

#define WARN_LOGGING                                                                               \
    (aql_profile::Logger::Instance()                                                               \
     << aql_profile::Logger::endl                                                                  \
     << "Warning: " << __FUNCTION__ << "(): " << aql_profile::Logger::begm)

#ifdef DEBUG
#    define DBG_LOGGING                                                                            \
        (aql_profile::Logger::Instance() << aql_profile::Logger::endl                              \
                                         << "Debug: in " << __FUNCTION__ << " at " << __FILE__     \
                                         << " line " << __LINE__ << aql_profile::Logger::begm)
#endif

#endif  // SRC_CORE_LOGGER_H_
