// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#define ROCJPEG_API_INFO_DEFINITION_0(                                                             \
    ROCJPEG_TABLE, ROCJPEG_API_ID, ROCJPEG_FUNC, ROCJPEG_FUNC_PTR)                                 \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace rocjpeg                                                                              \
    {                                                                                              \
    template <>                                                                                    \
    struct rocjpeg_api_info<ROCJPEG_TABLE, ROCJPEG_API_ID> : rocjpeg_domain_info<ROCJPEG_TABLE>    \
    {                                                                                              \
        static constexpr auto table_idx     = ROCJPEG_TABLE;                                       \
        static constexpr auto operation_idx = ROCJPEG_API_ID;                                      \
        static constexpr auto name          = #ROCJPEG_FUNC;                                       \
                                                                                                   \
        using domain_type = rocjpeg_domain_info<table_idx>;                                        \
        using this_type   = rocjpeg_api_info<table_idx, operation_idx>;                            \
        using base_type   = rocjpeg_api_impl<table_idx, operation_idx>;                            \
                                                                                                   \
        using domain_type::callback_domain_idx;                                                    \
        using domain_type::buffered_domain_idx;                                                    \
        using domain_type::args_type;                                                              \
        using domain_type::retval_type;                                                            \
        using domain_type::callback_data_type;                                                     \
                                                                                                   \
        static constexpr auto offset()                                                             \
        {                                                                                          \
            return offsetof(rocjpeg_table_lookup<table_idx>::type, ROCJPEG_FUNC_PTR);              \
        }                                                                                          \
                                                                                                   \
        static_assert(offsetof(rocjpeg_table_lookup<table_idx>::type, ROCJPEG_FUNC_PTR) ==         \
                          (sizeof(size_t) + (operation_idx * sizeof(void*))),                      \
                      "ABI error for " #ROCJPEG_FUNC);                                             \
                                                                                                   \
        static auto& get_table() { return rocjpeg_table_lookup<table_idx>{}(); }                   \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table(TableT& _v)                                                         \
        {                                                                                          \
            return rocjpeg_table_lookup<table_idx>{}(_v);                                          \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table_func(TableT& _table)                                                \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                assert(_table != nullptr && "nullptr to MARKER table for " #ROCJPEG_FUNC           \
                                            " function");                                          \
                return _table->ROCJPEG_FUNC_PTR;                                                   \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.ROCJPEG_FUNC_PTR;                                                    \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        static auto& get_table_func() { return get_table_func(get_table()); }                      \
                                                                                                   \
        template <typename DataT>                                                                  \
        static auto& get_api_data_args(DataT& _data)                                               \
        {                                                                                          \
            return _data.ROCJPEG_FUNC;                                                             \
        }                                                                                          \
                                                                                                   \
        template <typename RetT, typename... Args>                                                 \
        static auto get_functor(RetT (*)(Args...))                                                 \
        {                                                                                          \
            return &base_type::functor<RetT, Args...>;                                             \
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

#define ROCJPEG_API_INFO_DEFINITION_V(                                                             \
    ROCJPEG_TABLE, ROCJPEG_API_ID, ROCJPEG_FUNC, ROCJPEG_FUNC_PTR, ...)                            \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace rocjpeg                                                                              \
    {                                                                                              \
    template <>                                                                                    \
    struct rocjpeg_api_info<ROCJPEG_TABLE, ROCJPEG_API_ID> : rocjpeg_domain_info<ROCJPEG_TABLE>    \
    {                                                                                              \
        static constexpr auto table_idx     = ROCJPEG_TABLE;                                       \
        static constexpr auto operation_idx = ROCJPEG_API_ID;                                      \
        static constexpr auto name          = #ROCJPEG_FUNC;                                       \
                                                                                                   \
        using domain_type = rocjpeg_domain_info<table_idx>;                                        \
        using this_type   = rocjpeg_api_info<table_idx, operation_idx>;                            \
        using base_type   = rocjpeg_api_impl<table_idx, operation_idx>;                            \
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
            return offsetof(rocjpeg_table_lookup<table_idx>::type, ROCJPEG_FUNC_PTR);              \
        }                                                                                          \
                                                                                                   \
        static_assert(offsetof(rocjpeg_table_lookup<table_idx>::type, ROCJPEG_FUNC_PTR) ==         \
                          (sizeof(size_t) + (operation_idx * sizeof(void*))),                      \
                      "ABI error for " #ROCJPEG_FUNC);                                             \
                                                                                                   \
        static auto& get_table() { return rocjpeg_table_lookup<table_idx>{}(); }                   \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table(TableT& _v)                                                         \
        {                                                                                          \
            return rocjpeg_table_lookup<table_idx>{}(_v);                                          \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table_func(TableT& _table)                                                \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                assert(_table != nullptr && "nullptr to MARKER table for " #ROCJPEG_FUNC           \
                                            " function");                                          \
                return _table->ROCJPEG_FUNC_PTR;                                                   \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.ROCJPEG_FUNC_PTR;                                                    \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        static auto& get_table_func() { return get_table_func(get_table()); }                      \
                                                                                                   \
        template <typename DataT>                                                                  \
        static auto& get_api_data_args(DataT& _data)                                               \
        {                                                                                          \
            return _data.ROCJPEG_FUNC;                                                             \
        }                                                                                          \
                                                                                                   \
        template <typename RetT, typename... Args>                                                 \
        static auto get_functor(RetT (*)(Args...))                                                 \
        {                                                                                          \
            return &base_type::functor<RetT, Args...>;                                             \
        }                                                                                          \
                                                                                                   \
        static std::vector<void*> as_arg_addr(callback_data_type trace_data)                       \
        {                                                                                          \
            return std::vector<void*>{                                                             \
                GET_ADDR_MEMBER_FIELDS(get_api_data_args(trace_data.args), __VA_ARGS__)};          \
        }                                                                                          \
    };                                                                                             \
    }                                                                                              \
    }

#define ROCJPEG_API_TABLE_LOOKUP_DEFINITION(TABLE_ID, TYPE)                                        \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace rocjpeg                                                                              \
    {                                                                                              \
    namespace                                                                                      \
    {                                                                                              \
    template <>                                                                                    \
    auto* get_table<TABLE_ID>()                                                                    \
    {                                                                                              \
        return get_table_impl<TYPE>();                                                             \
    }                                                                                              \
    }                                                                                              \
                                                                                                   \
    template <>                                                                                    \
    struct rocjpeg_table_lookup<TABLE_ID>                                                          \
    {                                                                                              \
        using type = TYPE;                                                                         \
        auto& operator()(type& _v) const { return _v; }                                            \
        auto& operator()(type* _v) const { return *_v; }                                           \
        auto& operator()() const { return (*this)(get_table<TABLE_ID>()); }                        \
    };                                                                                             \
                                                                                                   \
    template <>                                                                                    \
    struct rocjpeg_table_id_lookup<TYPE>                                                           \
    {                                                                                              \
        static constexpr auto value = TABLE_ID;                                                    \
    };                                                                                             \
    }                                                                                              \
    }
