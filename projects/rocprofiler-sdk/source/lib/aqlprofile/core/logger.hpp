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

#ifndef AQLPROFILE_SRC_CORE_LOGGER_H
#define AQLPROFILE_SRC_CORE_LOGGER_H

#include "lib/aqlprofile/core/last_error.hpp"
#include "lib/common/logging.hpp"

#include <fmt/core.h>

#define AQL_INFO          ROCP_INFO << "[aqlprofile] " << __FUNCTION__ << "(): "
#define AQL_ERROR         ROCP_ERROR << "[aqlprofile] " << __FUNCTION__ << "(): "
#define AQL_WARNING       ROCP_WARNING << "[aqlprofile] " << __FUNCTION__ << "(): "
#define AQL_WARNING_IF(X) ROCP_WARNING_IF(X) << "[aqlprofile] " << __FUNCTION__ << "(): "
#define AQL_FATAL_IF(X)   ROCP_FATAL_IF(X) << "[aqlprofile] " << __FUNCTION__ << "(): "
#define AQL_CI_LOG(LEVEL) ROCP_CI_LOG(LEVEL) << "[aqlprofile] " << __FUNCTION__ << "(): "

#define ERR_LOGGING(...)                                                                           \
    do                                                                                             \
    {                                                                                              \
        aql_profile::set_last_error(fmt::format(__VA_ARGS__));                                     \
        AQL_ERROR << aql_profile::get_last_error();                                                \
    } while(0)

#define WARN_LOGGING(...)                                                                          \
    do                                                                                             \
    {                                                                                              \
        aql_profile::set_last_error(fmt::format(__VA_ARGS__));                                     \
        AQL_WARNING << aql_profile::get_last_error();                                              \
    } while(0)

#define INFO_LOGGING(...)                                                                          \
    do                                                                                             \
    {                                                                                              \
        aql_profile::set_last_error(fmt::format(__VA_ARGS__));                                     \
        AQL_INFO << aql_profile::get_last_error();                                                 \
    } while(0)
#endif  // AQLPROFILE_SRC_CORE_LOGGER_H
