// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/common/logging.hpp"

#define KFD_EVENT_PARSE_EVENTS(X, HANDLER)                                                         \
    do                                                                                             \
    {                                                                                              \
        const auto find_newline = [&](auto b) { return std::find(b, X.cend(), '\n'); };            \
                                                                                                   \
        const auto* cursor = X.cbegin();                                                           \
                                                                                                   \
        for(const auto* pos = find_newline(cursor); pos != X.cend(); pos = find_newline(cursor))   \
        {                                                                                          \
            size_t char_count = pos - cursor;                                                      \
            assert(char_count > 0);                                                                \
            std::string_view event_str{cursor, char_count};                                        \
                                                                                                   \
            ROCP_INFO << fmt::format("KFD event: [{}]", event_str);                                \
            HANDLER(event_str);                                                                    \
                                                                                                   \
            cursor = pos + 1;                                                                      \
        }                                                                                          \
    } while(0)

#define SPECIALIZE_PAGE_MIGRATION_INFO(ROCPROF_NAME, KFD_NAME, FORMAT_STRING)                        \
    template <>                                                                                      \
    struct page_migration_info<ROCPROFILER_PAGE_MIGRATION_##ROCPROF_NAME>                            \
    {                                                                                                \
        static constexpr auto             name          = "PAGE_MIGRATION_" #ROCPROF_NAME;           \
        static constexpr size_t           operation     = ROCPROFILER_PAGE_MIGRATION_##ROCPROF_NAME; \
        static constexpr size_t           kfd_operation = KFD_SMI_EVENT_##KFD_NAME;                  \
        static constexpr size_t           kfd_bitmask   = bitmask(KFD_SMI_EVENT_##KFD_NAME);         \
        static constexpr std::string_view format_str    = FORMAT_STRING;                             \
    };

#define SPECIALIZE_KFD_IOC_IOCTL(STRUCT, ARG_IOC)                                                  \
    template <>                                                                                    \
    struct IOC_event<STRUCT>                                                                       \
    {                                                                                              \
        static constexpr auto value = ARG_IOC;                                                     \
    }
