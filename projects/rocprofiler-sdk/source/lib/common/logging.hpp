// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#pragma once

#include <glog/logging.h>

#include <cstdint>
#include <string_view>

#define ROCP_LEVEL_TRACE   12
#define ROCP_LEVEL_INFO    11
#define ROCP_LEVEL_WARNING 10
#define ROCP_NO_VLOG       -1

#define ROCP_TRACE   VLOG(ROCP_LEVEL_TRACE)
#define ROCP_INFO    VLOG(ROCP_LEVEL_INFO)
#define ROCP_WARNING VLOG(ROCP_LEVEL_WARNING)
#define ROCP_ERROR   LOG(ERROR)
#define ROCP_FATAL   LOG(FATAL)
#define ROCP_DFATAL  DLOG(FATAL)

namespace rocprofiler
{
namespace common
{
struct logging_config
{
    bool    install_failure_handler = false;
    bool    logtostderr             = true;
    bool    alsologtostderr         = false;
    int32_t vlog_level              = ROCP_NO_VLOG;
    int32_t loglevel                = google::WARNING;
};

void
init_logging(std::string_view env_var, logging_config cfg = logging_config{});

void
update_logging(const logging_config& cfg, bool setup_env = false, int env_override = 0);
}  // namespace common
}  // namespace rocprofiler
