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

#include "lib/common/logging.hpp"

#include <array>
#include <cstring>
#include <string_view>

#include <fmt/format.h>

#define ERR_LOGGING (aql_profile::ErrorStream{__FILE__, __LINE__}) << __FUNCTION__ << "(): "

#define AQL_INFO          ROCP_INFO << "[aqlprofile]"
#define AQL_ERROR         ROCP_ERROR << "[aqlprofile]"
#define AQL_WARNING       ROCP_WARNING << "[aqlprofile]"
#define AQL_WARNING_IF(X) ROCP_WARNING_IF(X) << "[aqlprofile]"
#define AQL_FATAL_IF(X)   ROCP_FATAL_IF(X) << "[aqlprofile]"
#define AQL_CI_LOG(LEVEL) ROCP_CI_LOG(LEVEL) << "[aqlprofile]"

namespace aql_profile
{
static constexpr size_t MSG_BUF_LEN = 1024;

std::array<char, MSG_BUF_LEN>&
last_error_msg();

struct ErrorStream
{
    const char*                    file;
    int                            line;
    std::array<char, MSG_BUF_LEN>& buf = last_error_msg();
    size_t                         len = 0;

    ErrorStream(const char* f, int l)
    : file(f)
    , line(l)
    {}

    ErrorStream(const ErrorStream&) = delete;
    ErrorStream& operator=(const ErrorStream&) = delete;

    ErrorStream& operator<<(std::string_view v)
    {
        auto n = std::min(v.size(), MSG_BUF_LEN - 1 - len);
        std::memcpy(buf.data() + len, v.data(), n);
        len += n;
        return *this;
    }

    ~ErrorStream()
    {
        buf[len] = '\0';
        // manual call to retain caller source location
        google::LogMessage(file, line, google::GLOG_ERROR).stream()
            << "[aqlprofile] " << std::string_view{buf.data(), len};
    }
};

}  // namespace aql_profile

#endif  // AQLPROFILE_SRC_CORE_LOGGER_H
