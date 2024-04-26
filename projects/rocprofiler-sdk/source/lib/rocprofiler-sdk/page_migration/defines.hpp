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
            LOG(INFO) << fmt::format("KFD event: [{}]", event_str);                                \
            HANDLER(event_str);                                                                    \
                                                                                                   \
            cursor = pos + 1;                                                                      \
        }                                                                                          \
    } while(0)

#define APPEND_UVM_EVENT(X)          ROCPROFILER_UVM_EVENT_##X
#define APPEND_1(X)                  APPEND_UVM_EVENT(X)
#define CONCAT(X, Y)                 X##Y
#define APPEND_2(A1, A2)             APPEND_1(A1), APPEND_1(A2)
#define APPEND_3(A1, A2, A3)         APPEND_2(A1, A2), APPEND_1(A3)
#define APPEND_4(A1, A2, A3, A4)     APPEND_3(A1, A2, A3), APPEND_1(A4)
#define APPEND_5(A1, A2, A3, A4, A5) APPEND_4(A1, A2, A3, A4), APPEND_1(A5)

#define MACRO_N(MACRO, N, ...) CONCAT(MACRO, N)(__VA_ARGS__)
#define APPLY_N(MACRO, ...)    MACRO_N(MACRO, IMPL_DETAIL_FOR_EACH_NARG(__VA_ARGS__), __VA_ARGS__)

#define GET_UVM_ENUMS(...) APPLY_N(APPEND_, __VA_ARGS__)

// static constexpr size_t           uvm_event = UVM_ENUM;
#define SPECIALIZE_UVM_KFD_EVENT(UVM_ENUM, KFD_ENUM, FORMAT_STRING)                                \
    template <>                                                                                    \
    struct uvm_event_info<UVM_ENUM>                                                                \
    {                                                                                              \
        static constexpr size_t           kfd_event = KFD_ENUM;                                    \
        static constexpr std::string_view format_str{FORMAT_STRING};                               \
    };

#define SPECIALIZE_PAGE_MIGRATION_INFO(TYPE, ...)                                                  \
    template <>                                                                                    \
    struct page_migration_info<ROCPROFILER_PAGE_MIGRATION_##TYPE>                                  \
    {                                                                                              \
        static constexpr auto   operation_idx = ROCPROFILER_PAGE_MIGRATION_##TYPE;                 \
        static constexpr auto   name          = "PAGE_MIGRATION_" #TYPE;                           \
        static constexpr size_t uvm_bitmask =                                                      \
            bitmask(std::index_sequence<GET_UVM_ENUMS(__VA_ARGS__)>());                            \
        static constexpr size_t kfd_bitmask =                                                      \
            to_kfd_bitmask(std::index_sequence<GET_UVM_ENUMS(__VA_ARGS__)>());                     \
    }

#define COPY_FROM_START_1(MEMBER)             end.MEMBER = start.MEMBER;
#define COPY_FROM_START_2(UNION_TYPE, MEMBER) end.UNION_TYPE.MEMBER = start.UNION_TYPE.MEMBER;

#define SPECIALIZE_KFD_IOC_IOCTL(STRUCT, ARG_IOC)                                                  \
    template <>                                                                                    \
    struct IOC_event<STRUCT>                                                                       \
    {                                                                                              \
        static constexpr auto value = ARG_IOC;                                                     \
    }
