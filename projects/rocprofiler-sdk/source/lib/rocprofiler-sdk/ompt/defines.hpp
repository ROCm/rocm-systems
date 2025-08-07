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

#include "lib/common/defines.hpp"

#define OMPT_INFO_DEFINITION_0(                                                                    \
    OMPT_NAME, BACKEND_OMPT_ID, OMPT_ID, OMPT_FUNC_TYPE, OMPT_ARG, OMPT_FUNC_PTR)                  \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace ompt                                                                                 \
    {                                                                                              \
    template <>                                                                                    \
    struct ompt_info<OMPT_ID> : ompt_domain_info                                                   \
    {                                                                                              \
        static constexpr auto ompt_idx      = BACKEND_OMPT_ID;                                     \
        static constexpr auto operation_idx = OMPT_ID;                                             \
        static constexpr auto name          = OMPT_NAME;                                           \
        static constexpr auto unsupported   = false;                                               \
                                                                                                   \
        using domain_type = ompt_domain_info;                                                      \
        using this_type   = ompt_info<operation_idx>;                                              \
                                                                                                   \
        static constexpr auto callback_domain_idx = domain_type::callback_domain_idx;              \
        static constexpr auto buffered_domain_idx = domain_type::buffered_domain_idx;              \
                                                                                                   \
        using domain_type::args_type;                                                              \
        using domain_type::retval_type;                                                            \
        using domain_type::callback_data_type;                                                     \
                                                                                                   \
        static constexpr auto offset()                                                             \
        {                                                                                          \
            return offsetof(ompt_table_lookup::type, OMPT_FUNC_PTR);                               \
        }                                                                                          \
                                                                                                   \
        static auto& get_table() { return ompt_table_lookup{}(); }                                 \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table_func(TableT& _table)                                                \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                return CHECK_NOTNULL(_table)->OMPT_FUNC_PTR;                                       \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.OMPT_FUNC_PTR;                                                       \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        static auto& get_table_func() { return get_table_func(get_table()); }                      \
                                                                                                   \
        template <typename DataT>                                                                  \
        static auto& get_api_data_args(DataT& _data)                                               \
        {                                                                                          \
            return _data.OMPT_ARG;                                                                 \
        }                                                                                          \
                                                                                                   \
        static std::vector<void*> as_arg_addr(callback_data_type) { return std::vector<void*>{}; } \
                                                                                                   \
        static std::vector<common::stringified_argument> as_arg_list(callback_data_type, int32_t)  \
        {                                                                                          \
            return {};                                                                             \
        }                                                                                          \
    };                                                                                             \
    }                                                                                              \
    }

#define OMPT_INFO_DEFINITION_V(                                                                    \
    OMPT_NAME, BACKEND_OMPT_ID, OMPT_ID, OMPT_FUNC_TYPE, OMPT_ARG, OMPT_FUNC_PTR, ...)             \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace ompt                                                                                 \
    {                                                                                              \
    template <>                                                                                    \
    struct ompt_info<OMPT_ID> : ompt_domain_info                                                   \
    {                                                                                              \
        static constexpr auto ompt_idx      = BACKEND_OMPT_ID;                                     \
        static constexpr auto operation_idx = OMPT_ID;                                             \
        static constexpr auto name          = OMPT_NAME;                                           \
        static constexpr auto unsupported   = false;                                               \
                                                                                                   \
        using domain_type = ompt_domain_info;                                                      \
        using this_type   = ompt_info<operation_idx>;                                              \
                                                                                                   \
        static constexpr auto callback_domain_idx = domain_type::callback_domain_idx;              \
        static constexpr auto buffered_domain_idx = domain_type::buffered_domain_idx;              \
                                                                                                   \
        using domain_type::args_type;                                                              \
        using domain_type::retval_type;                                                            \
        using domain_type::callback_data_type;                                                     \
                                                                                                   \
        static constexpr auto offset()                                                             \
        {                                                                                          \
            return offsetof(ompt_table_lookup::type, OMPT_FUNC_PTR);                               \
        }                                                                                          \
                                                                                                   \
        static auto& get_table() { return ompt_table_lookup{}(); }                                 \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table_func(TableT& _table)                                                \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                return CHECK_NOTNULL(_table)->OMPT_FUNC_PTR;                                       \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.OMPT_FUNC_PTR;                                                       \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        static auto& get_table_func() { return get_table_func(get_table()); }                      \
                                                                                                   \
        template <typename DataT>                                                                  \
        static auto& get_api_data_args(DataT& _data)                                               \
        {                                                                                          \
            return _data.OMPT_ARG;                                                                 \
        }                                                                                          \
                                                                                                   \
        static std::vector<void*> as_arg_addr(callback_data_type trace_data)                       \
        {                                                                                          \
            return std::vector<void*>{                                                             \
                GET_ADDR_MEMBER_FIELDS(get_api_data_args(trace_data.args), __VA_ARGS__)};          \
        }                                                                                          \
                                                                                                   \
        static auto as_arg_list(callback_data_type trace_data, int32_t max_deref)                  \
        {                                                                                          \
            return utils::stringize(                                                               \
                max_deref,                                                                         \
                GET_NAMED_MEMBER_FIELDS(get_api_data_args(trace_data.args), __VA_ARGS__));         \
        }                                                                                          \
    };                                                                                             \
    }                                                                                              \
    }
