// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdlib>
#include <sstream>

// Error category tags for quick diagnosis
#define ERROR_TAG_SETUP "[SETUP] "
#define ERROR_TAG_PERMISSIONS "[PERMISSIONS] "
#define ERROR_TAG_INFRASTRUCTURE "[INFRASTRUCTURE] "
#define ERROR_TAG_DATA "[DATA] "

// Check if running in CI environment
inline bool is_ci_environment() {
    return std::getenv("CI") != nullptr ||
           std::getenv("CONTINUOUS_INTEGRATION") != nullptr ||
           std::getenv("JENKINS_HOME") != nullptr ||
           std::getenv("GITLAB_CI") != nullptr;
}

// Determine if verbose logging is enabled
inline bool is_verbose_logging() {
    const char* level = std::getenv("ROCPROFILER_TEST_LOG_LEVEL");
    if(level != nullptr && std::string(level) == "VERBOSE") return true;
    return !is_ci_environment();
}

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_name = rocprofiler_get_status_name(CHECKSTATUS);                    \
            std::string status_msg  = rocprofiler_get_status_string(CHECKSTATUS);                  \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << status_name << " (" << CHECKSTATUS          \
                      << "): " << status_msg << std::endl;                                         \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg               \
                   << " failure (" << status_name << ": " << status_msg << ")";                    \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

// Enhanced ROCPROFILER_CALL with error category and diagnostics
#define ROCPROFILER_CALL_DIAG(result, msg, category, diagnostic)                                   \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_name = rocprofiler_get_status_name(CHECKSTATUS);                    \
            std::string status_msg  = rocprofiler_get_status_string(CHECKSTATUS);                  \
            std::cerr << category << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] "       \
                      << msg << " failed with error code " << status_name << " (" << CHECKSTATUS   \
                      << "): " << status_msg << std::endl;                                         \
            if(is_verbose_logging()) {                                                             \
                std::cerr << "Diagnostic: " << diagnostic << std::endl;                            \
            }                                                                                      \
            std::stringstream errmsg{};                                                            \
            errmsg << category << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg   \
                   << " failure (" << status_name << ": " << status_msg << ")";                    \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

#if HIP_VERSION >= 60300000
#    define HIP_HOST_ALLOC_FUNC hipHostMalloc
#    define HIP_HOST_FREE_FUNC  hipHostFree
#else
#    define HIP_HOST_ALLOC_FUNC hipHostMalloc
#    define HIP_HOST_FREE_FUNC  hipHostFree
#endif
