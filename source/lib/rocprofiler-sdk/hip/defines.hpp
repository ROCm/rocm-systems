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

#define HIP_API_INFO_DEFINITION_0(HIP_TABLE, HIP_API_ID, HIP_FUNC, HIP_FUNC_PTR)                   \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace hip                                                                                  \
    {                                                                                              \
    template <>                                                                                    \
    struct hip_api_info<HIP_TABLE, HIP_API_ID> : hip_domain_info<HIP_TABLE>                        \
    {                                                                                              \
        static constexpr auto table_idx     = HIP_TABLE;                                           \
        static constexpr auto operation_idx = HIP_API_ID;                                          \
        static constexpr auto name          = #HIP_FUNC;                                           \
                                                                                                   \
        using domain_type = hip_domain_info<table_idx>;                                            \
        using this_type   = hip_api_info<table_idx, operation_idx>;                                \
        using base_type   = hip_api_impl<table_idx, operation_idx>;                                \
                                                                                                   \
        using domain_type::callback_domain_idx;                                                    \
        using domain_type::buffered_domain_idx;                                                    \
        using domain_type::args_type;                                                              \
        using domain_type::retval_type;                                                            \
        using domain_type::callback_data_type;                                                     \
                                                                                                   \
        static constexpr auto get_args_type() { return common::mpl::type_list<>{}; }               \
                                                                                                   \
        static constexpr auto offset()                                                             \
        {                                                                                          \
            return offsetof(hip_table_lookup<table_idx>::type, HIP_FUNC_PTR);                      \
        }                                                                                          \
                                                                                                   \
        static_assert(offsetof(hip_table_lookup<table_idx>::type, HIP_FUNC_PTR) ==                 \
                          (sizeof(size_t) + (operation_idx * sizeof(void*))),                      \
                      "ABI error for " #HIP_FUNC);                                                 \
                                                                                                   \
        static auto& get_table() { return hip_table_lookup<table_idx>{}(); }                       \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table(TableT& _v)                                                         \
        {                                                                                          \
            return hip_table_lookup<table_idx>{}(_v);                                              \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table_func(TableT& _table)                                                \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                assert(_table != nullptr && "nullptr to HIP table for " #HIP_FUNC " function");    \
                return _table->HIP_FUNC_PTR;                                                       \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.HIP_FUNC_PTR;                                                        \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        static auto& get_table_func() { return get_table_func(get_table()); }                      \
                                                                                                   \
        template <typename DataT>                                                                  \
        static auto& get_api_data_args(DataT& _data)                                               \
        {                                                                                          \
            return _data.HIP_FUNC;                                                                 \
        }                                                                                          \
                                                                                                   \
        template <typename RetT, typename... Args>                                                 \
        static auto get_functor(RetT (*)(Args...))                                                 \
        {                                                                                          \
            return &base_type::functor<RetT, Args...>;                                             \
        }                                                                                          \
                                                                                                   \
        static std::vector<void*> as_arg_addr(rocprofiler_hip_api_args_t)                          \
        {                                                                                          \
            return std::vector<void*>{};                                                           \
        }                                                                                          \
                                                                                                   \
        static std::vector<common::stringified_argument> as_arg_list(rocprofiler_hip_api_args_t,   \
                                                                     int32_t)                      \
        {                                                                                          \
            return {};                                                                             \
        }                                                                                          \
    };                                                                                             \
    }                                                                                              \
    }

#define HIP_API_INFO_DEFINITION_V(HIP_TABLE, HIP_API_ID, HIP_FUNC, HIP_FUNC_PTR, ...)              \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace hip                                                                                  \
    {                                                                                              \
    template <>                                                                                    \
    struct hip_api_info<HIP_TABLE, HIP_API_ID> : hip_domain_info<HIP_TABLE>                        \
    {                                                                                              \
        static constexpr auto table_idx     = HIP_TABLE;                                           \
        static constexpr auto operation_idx = HIP_API_ID;                                          \
        static constexpr auto name          = #HIP_FUNC;                                           \
                                                                                                   \
        using domain_type = hip_domain_info<table_idx>;                                            \
        using this_type   = hip_api_info<table_idx, operation_idx>;                                \
        using base_type   = hip_api_impl<table_idx, operation_idx>;                                \
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
            return offsetof(hip_table_lookup<table_idx>::type, HIP_FUNC_PTR);                      \
        }                                                                                          \
                                                                                                   \
        static_assert(offsetof(hip_table_lookup<table_idx>::type, HIP_FUNC_PTR) ==                 \
                          (sizeof(size_t) + (operation_idx * sizeof(void*))),                      \
                      "ABI error for " #HIP_FUNC);                                                 \
                                                                                                   \
        static auto& get_table() { return hip_table_lookup<table_idx>{}(); }                       \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table(TableT& _v)                                                         \
        {                                                                                          \
            return hip_table_lookup<table_idx>{}(_v);                                              \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table_func(TableT& _table)                                                \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                assert(_table != nullptr && "nullptr to HIP table for " #HIP_FUNC " function");    \
                return _table->HIP_FUNC_PTR;                                                       \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.HIP_FUNC_PTR;                                                        \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        static auto& get_table_func() { return get_table_func(get_table()); }                      \
                                                                                                   \
        template <typename DataT>                                                                  \
        static auto& get_api_data_args(DataT& _data)                                               \
        {                                                                                          \
            return _data.HIP_FUNC;                                                                 \
        }                                                                                          \
                                                                                                   \
        template <typename RetT, typename... Args>                                                 \
        static auto get_functor(RetT (*)(Args...))                                                 \
        {                                                                                          \
            return &base_type::functor<RetT, Args...>;                                             \
        }                                                                                          \
                                                                                                   \
        static constexpr auto get_args_type()                                                      \
        {                                                                                          \
            using func_t = decltype(get_table_func());                                             \
            return common::mpl::function_args_t<func_t>{};                                         \
        }                                                                                          \
                                                                                                   \
        static std::vector<void*> as_arg_addr(rocprofiler_hip_api_args_t args)                     \
        {                                                                                          \
            return std::vector<void*>{                                                             \
                GET_ADDR_MEMBER_FIELDS(get_api_data_args(args), __VA_ARGS__)};                     \
        }                                                                                          \
                                                                                                   \
        static auto as_arg_list(rocprofiler_hip_api_args_t args, int32_t max_deref)                \
        {                                                                                          \
            return utils::stringize(                                                               \
                max_deref, GET_NAMED_MEMBER_FIELDS(get_api_data_args(args), __VA_ARGS__));         \
        }                                                                                          \
    };                                                                                             \
    }                                                                                              \
    }

#define HIP_API_TABLE_LOOKUP_DEFINITION(TABLE_ID, TYPE, MEMBER)                                    \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace hip                                                                                  \
    {                                                                                              \
    template <>                                                                                    \
    struct hip_table_lookup<TABLE_ID>                                                              \
    {                                                                                              \
        using type = TYPE;                                                                         \
        auto& operator()(hip_api_table_t& _v) const { return _v.MEMBER; }                          \
        auto& operator()(hip_api_table_t* _v) const { return _v->MEMBER; }                         \
        auto& operator()(type& _v) const { return _v; }                                            \
        auto& operator()(type* _v) const { return *_v; }                                           \
        auto& operator()() const { return (*this)(get_table()); }                                  \
    };                                                                                             \
                                                                                                   \
    template <>                                                                                    \
    struct hip_table_id_lookup<TYPE>                                                               \
    {                                                                                              \
        static constexpr auto value = TABLE_ID;                                                    \
    };                                                                                             \
    }                                                                                              \
    }
