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

#define HSA_API_META_DEFINITION(HSA_TABLE, HSA_API_ID, HSA_FUNC, HSA_FUNC_PTR)                     \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace hsa                                                                                  \
    {                                                                                              \
    template <>                                                                                    \
    struct hsa_api_meta<HSA_TABLE, HSA_API_ID>                                                     \
    {                                                                                              \
        static constexpr auto table_idx     = HSA_TABLE;                                           \
        static constexpr auto operation_idx = HSA_API_ID;                                          \
        static constexpr auto name          = #HSA_FUNC;                                           \
                                                                                                   \
        using this_type     = hsa_api_meta<table_idx, operation_idx>;                              \
        using function_type = hsa_api_func<decltype(::HSA_FUNC)*>::function_type;                  \
                                                                                                   \
        static constexpr auto offset()                                                             \
        {                                                                                          \
            return offsetof(hsa_table_lookup<table_idx>::type, HSA_FUNC_PTR);                      \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table(TableT& _v)                                                         \
        {                                                                                          \
            return hsa_table_lookup<table_idx>{}(_v);                                              \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table_func(TableT& _table)                                                \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                assert(_table != nullptr && "nullptr to HSA table for " #HSA_FUNC " function");    \
                return _table->HSA_FUNC_PTR;                                                       \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.HSA_FUNC_PTR;                                                        \
            }                                                                                      \
        }                                                                                          \
    };                                                                                             \
    }                                                                                              \
    }

// meta definition for non-public APIs (i.e. only in table)
#define HSA_API_META_DEFINITION_NP(HSA_TABLE, HSA_API_ID, HSA_FUNC, HSA_FUNC_PTR)                  \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace hsa                                                                                  \
    {                                                                                              \
    template <>                                                                                    \
    struct hsa_api_meta<HSA_TABLE, HSA_API_ID>                                                     \
    {                                                                                              \
        static constexpr auto table_idx     = HSA_TABLE;                                           \
        static constexpr auto operation_idx = HSA_API_ID;                                          \
        static constexpr auto name          = #HSA_FUNC;                                           \
                                                                                                   \
        using this_type     = hsa_api_meta<table_idx, operation_idx>;                              \
        using function_type = hsa_api_func<decltype(                                               \
            std::declval<hsa_table_lookup<table_idx>::type>().HSA_FUNC_PTR)>::function_type;       \
                                                                                                   \
        static constexpr auto offset()                                                             \
        {                                                                                          \
            return offsetof(hsa_table_lookup<table_idx>::type, HSA_FUNC_PTR);                      \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table(TableT& _v)                                                         \
        {                                                                                          \
            return hsa_table_lookup<table_idx>{}(_v);                                              \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table_func(TableT& _table)                                                \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                assert(_table != nullptr && "nullptr to HSA table for " #HSA_FUNC " function");    \
                return _table->HSA_FUNC_PTR;                                                       \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.HSA_FUNC_PTR;                                                        \
            }                                                                                      \
        }                                                                                          \
    };                                                                                             \
    }                                                                                              \
    }

#define HSA_API_INFO_DEFINITION_0(HSA_TABLE, HSA_API_ID, HSA_FUNC, HSA_FUNC_PTR)                   \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace hsa                                                                                  \
    {                                                                                              \
    template <>                                                                                    \
    struct hsa_api_info<HSA_TABLE, HSA_API_ID>                                                     \
    {                                                                                              \
        static constexpr auto callback_domain_idx =                                                \
            hsa_domain_info<HSA_TABLE>::callback_domain_idx;                                       \
        static constexpr auto buffered_domain_idx =                                                \
            hsa_domain_info<HSA_TABLE>::buffered_domain_idx;                                       \
        static constexpr auto table_idx     = HSA_TABLE;                                           \
        static constexpr auto operation_idx = HSA_API_ID;                                          \
        static constexpr auto name          = #HSA_FUNC;                                           \
                                                                                                   \
        using this_type = hsa_api_info<table_idx, operation_idx>;                                  \
        using base_type = hsa_api_impl<table_idx, operation_idx>;                                  \
                                                                                                   \
        static constexpr auto offset()                                                             \
        {                                                                                          \
            return offsetof(hsa_table_lookup<table_idx>::type, HSA_FUNC_PTR);                      \
        }                                                                                          \
                                                                                                   \
        static auto& get_table(tracing_table)                                                      \
        {                                                                                          \
            return hsa_table_lookup<table_idx>{}(tracing_table{});                                 \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table(TableT& _v)                                                         \
        {                                                                                          \
            return hsa_table_lookup<table_idx>{}(_v);                                              \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table_func(TableT& _table)                                                \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                assert(_table != nullptr && "nullptr to HSA table for " #HSA_FUNC " function");    \
                return _table->HSA_FUNC_PTR;                                                       \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.HSA_FUNC_PTR;                                                        \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        static auto& get_table_func() { return get_table_func(get_table(tracing_table{})); }       \
                                                                                                   \
        template <typename DataT>                                                                  \
        static auto& get_api_data_args(DataT& _data)                                               \
        {                                                                                          \
            return _data.HSA_FUNC;                                                                 \
        }                                                                                          \
                                                                                                   \
        template <typename RetT, typename... Args>                                                 \
        static auto get_functor(RetT (*)(Args...))                                                 \
        {                                                                                          \
            return &base_type::functor<RetT, Args...>;                                             \
        }                                                                                          \
                                                                                                   \
        static std::vector<void*> as_arg_addr(rocprofiler_callback_tracing_hsa_api_data_t)         \
        {                                                                                          \
            return std::vector<void*>{};                                                           \
        }                                                                                          \
                                                                                                   \
        static std::vector<common::stringified_argument> as_arg_list(                              \
            rocprofiler_callback_tracing_hsa_api_data_t,                                           \
            int32_t)                                                                               \
        {                                                                                          \
            return {};                                                                             \
        }                                                                                          \
    };                                                                                             \
    }                                                                                              \
    }

#define HSA_API_INFO_DEFINITION_V(HSA_TABLE, HSA_API_ID, HSA_FUNC, HSA_FUNC_PTR, ...)              \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace hsa                                                                                  \
    {                                                                                              \
    template <>                                                                                    \
    struct hsa_api_info<HSA_TABLE, HSA_API_ID>                                                     \
    {                                                                                              \
        static constexpr auto callback_domain_idx =                                                \
            hsa_domain_info<HSA_TABLE>::callback_domain_idx;                                       \
        static constexpr auto buffered_domain_idx =                                                \
            hsa_domain_info<HSA_TABLE>::buffered_domain_idx;                                       \
        static constexpr auto table_idx     = HSA_TABLE;                                           \
        static constexpr auto operation_idx = HSA_API_ID;                                          \
        static constexpr auto name          = #HSA_FUNC;                                           \
                                                                                                   \
        using this_type = hsa_api_info<table_idx, operation_idx>;                                  \
        using base_type = hsa_api_impl<table_idx, operation_idx>;                                  \
                                                                                                   \
        static constexpr auto offset()                                                             \
        {                                                                                          \
            return offsetof(hsa_table_lookup<table_idx>::type, HSA_FUNC_PTR);                      \
        }                                                                                          \
                                                                                                   \
        static auto& get_table(tracing_table)                                                      \
        {                                                                                          \
            return hsa_table_lookup<table_idx>{}(tracing_table{});                                 \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table(TableT& _v)                                                         \
        {                                                                                          \
            return hsa_table_lookup<table_idx>{}(_v);                                              \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table_func(TableT& _table)                                                \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                assert(_table != nullptr && "nullptr to HSA table for " #HSA_FUNC " function");    \
                return _table->HSA_FUNC_PTR;                                                       \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.HSA_FUNC_PTR;                                                        \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        static auto& get_table_func() { return get_table_func(get_table(tracing_table{})); }       \
                                                                                                   \
        template <typename DataT>                                                                  \
        static auto& get_api_data_args(DataT& _data)                                               \
        {                                                                                          \
            return _data.HSA_FUNC;                                                                 \
        }                                                                                          \
                                                                                                   \
        template <typename RetT, typename... Args>                                                 \
        static auto get_functor(RetT (*)(Args...))                                                 \
        {                                                                                          \
            return &base_type::functor<RetT, Args...>;                                             \
        }                                                                                          \
                                                                                                   \
        static std::vector<void*> as_arg_addr(                                                     \
            rocprofiler_callback_tracing_hsa_api_data_t trace_data)                                \
        {                                                                                          \
            return std::vector<void*>{                                                             \
                GET_ADDR_MEMBER_FIELDS(get_api_data_args(trace_data.args), __VA_ARGS__)};          \
        }                                                                                          \
                                                                                                   \
        static auto as_arg_list(rocprofiler_callback_tracing_hsa_api_data_t trace_data,            \
                                int32_t                                     max_deref)             \
        {                                                                                          \
            return utils::stringize(                                                               \
                max_deref,                                                                         \
                GET_NAMED_MEMBER_FIELDS(get_api_data_args(trace_data.args), __VA_ARGS__));         \
        }                                                                                          \
    };                                                                                             \
    }                                                                                              \
    }

#define HSA_API_TABLE_LOOKUP_DEFINITION(TABLE_ID, TYPE, NAME)                                      \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace hsa                                                                                  \
    {                                                                                              \
    struct tracing_table;                                                                          \
    struct internal_table;                                                                         \
                                                                                                   \
    template <>                                                                                    \
    struct hsa_table_lookup<TABLE_ID>                                                              \
    {                                                                                              \
        using type = TYPE;                                                                         \
        auto& operator()(type& _v) const { return _v; }                                            \
        auto& operator()(type* _v) const { return *_v; }                                           \
        auto& operator()(tracing_table) const { return (*this)(get_tracing_##NAME##_table()); }    \
        auto& operator()(internal_table) const { return (*this)(get_##NAME##_table()); }           \
    };                                                                                             \
                                                                                                   \
    template <>                                                                                    \
    struct hsa_table_id_lookup<TYPE>                                                               \
    {                                                                                              \
        static constexpr auto value = TABLE_ID;                                                    \
    };                                                                                             \
    }                                                                                              \
    }
