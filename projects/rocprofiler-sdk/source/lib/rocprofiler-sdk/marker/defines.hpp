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

#define MARKER_API_INFO_DEFINITION_0(MARKER_TABLE, MARKER_API_ID, MARKER_FUNC, MARKER_FUNC_PTR)    \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace marker                                                                               \
    {                                                                                              \
    template <>                                                                                    \
    struct roctx_api_info<MARKER_TABLE, MARKER_API_ID> : roctx_domain_info<MARKER_TABLE>           \
    {                                                                                              \
        static constexpr auto is_range      = false;                                               \
        static constexpr auto table_idx     = MARKER_TABLE;                                        \
        static constexpr auto operation_idx = MARKER_API_ID;                                       \
        static constexpr auto name          = #MARKER_FUNC;                                        \
                                                                                                   \
        using domain_type = roctx_domain_info<table_idx>;                                          \
        using this_type   = roctx_api_info<table_idx, operation_idx>;                              \
        using base_type   = roctx_api_impl<table_idx, operation_idx>;                              \
                                                                                                   \
        using domain_type::callback_domain_idx;                                                    \
        using domain_type::buffered_domain_idx;                                                    \
        using domain_type::args_type;                                                              \
        using domain_type::retval_type;                                                            \
        using domain_type::callback_data_type;                                                     \
                                                                                                   \
        static constexpr auto offset()                                                             \
        {                                                                                          \
            return offsetof(roctx_table_lookup<table_idx>::type, MARKER_FUNC_PTR);                 \
        }                                                                                          \
                                                                                                   \
        static_assert(offsetof(roctx_table_lookup<table_idx>::type, MARKER_FUNC_PTR) ==            \
                          (sizeof(size_t) + (operation_idx * sizeof(void*))),                      \
                      "ABI error for " #MARKER_FUNC);                                              \
                                                                                                   \
        static auto& get_table() { return roctx_table_lookup<table_idx>{}(); }                     \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table(TableT& _v)                                                         \
        {                                                                                          \
            return roctx_table_lookup<table_idx>{}(_v);                                            \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table_func(TableT& _table)                                                \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                assert(_table != nullptr && "nullptr to MARKER table for " #MARKER_FUNC            \
                                            " function");                                          \
                return _table->MARKER_FUNC_PTR;                                                    \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.MARKER_FUNC_PTR;                                                     \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        static auto& get_table_func() { return get_table_func(get_table()); }                      \
                                                                                                   \
        template <typename DataT>                                                                  \
        static auto& get_api_data_args(DataT& _data)                                               \
        {                                                                                          \
            return _data.MARKER_FUNC;                                                              \
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

#define MARKER_API_INFO_DEFINITION_V(                                                              \
    MARKER_TABLE, MARKER_API_ID, MARKER_FUNC, MARKER_FUNC_PTR, ...)                                \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace marker                                                                               \
    {                                                                                              \
    template <>                                                                                    \
    struct roctx_api_info<MARKER_TABLE, MARKER_API_ID> : roctx_domain_info<MARKER_TABLE>           \
    {                                                                                              \
        static constexpr auto is_range      = false;                                               \
        static constexpr auto table_idx     = MARKER_TABLE;                                        \
        static constexpr auto operation_idx = MARKER_API_ID;                                       \
        static constexpr auto name          = #MARKER_FUNC;                                        \
                                                                                                   \
        using domain_type = roctx_domain_info<table_idx>;                                          \
        using this_type   = roctx_api_info<table_idx, operation_idx>;                              \
        using base_type   = roctx_api_impl<table_idx, operation_idx>;                              \
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
            return offsetof(roctx_table_lookup<table_idx>::type, MARKER_FUNC_PTR);                 \
        }                                                                                          \
                                                                                                   \
        static_assert(offsetof(roctx_table_lookup<table_idx>::type, MARKER_FUNC_PTR) ==            \
                          (sizeof(size_t) + (operation_idx * sizeof(void*))),                      \
                      "ABI error for " #MARKER_FUNC);                                              \
                                                                                                   \
        static auto& get_table() { return roctx_table_lookup<table_idx>{}(); }                     \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table(TableT& _v)                                                         \
        {                                                                                          \
            return roctx_table_lookup<table_idx>{}(_v);                                            \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table_func(TableT& _table)                                                \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                assert(_table != nullptr && "nullptr to MARKER table for " #MARKER_FUNC            \
                                            " function");                                          \
                return _table->MARKER_FUNC_PTR;                                                    \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.MARKER_FUNC_PTR;                                                     \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        static auto& get_table_func() { return get_table_func(get_table()); }                      \
                                                                                                   \
        template <typename DataT>                                                                  \
        static auto& get_api_data_args(DataT& _data)                                               \
        {                                                                                          \
            return _data.MARKER_FUNC;                                                              \
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

#define MARKER_EVENT_API_INFO_DEFINITION_V(                                                        \
    MARKER_TABLE, MARKER_API_ID, MARKER_NAME, MARKER_FUNC_PTR, ...)                                \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace marker                                                                               \
    {                                                                                              \
    template <>                                                                                    \
    struct roctx_api_info<MARKER_TABLE, MARKER_API_ID> : roctx_domain_info<MARKER_TABLE>           \
    {                                                                                              \
        static constexpr auto is_range      = false;                                               \
        static constexpr auto table_idx     = MARKER_TABLE;                                        \
        static constexpr auto operation_idx = MARKER_API_ID;                                       \
        static constexpr auto name          = #MARKER_NAME;                                        \
                                                                                                   \
        using domain_type = roctx_domain_info<table_idx>;                                          \
        using this_type   = roctx_api_info<table_idx, operation_idx>;                              \
        using base_type   = roctx_api_impl<table_idx, operation_idx>;                              \
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
            return offsetof(roctx_table_lookup<table_idx>::type, MARKER_FUNC_PTR);                 \
        }                                                                                          \
                                                                                                   \
        static auto& get_table() { return roctx_table_lookup<table_idx>{}(); }                     \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table(TableT& _v)                                                         \
        {                                                                                          \
            return roctx_table_lookup<table_idx>{}(_v);                                            \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table_func(TableT& _table)                                                \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                assert(_table != nullptr && "nullptr to MARKER table for " #MARKER_NAME            \
                                            " function");                                          \
                return _table->MARKER_FUNC_PTR;                                                    \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.MARKER_FUNC_PTR;                                                     \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        static auto& get_table_func() { return get_table_func(get_table()); }                      \
                                                                                                   \
        template <typename DataT>                                                                  \
        static auto& get_api_data_args(DataT& _data)                                               \
        {                                                                                          \
            return _data.MARKER_NAME;                                                              \
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

#define MARKER_RANGE_API_INFO_DEFINITION_V(                                                        \
    MARKER_TABLE, MARKER_API_ID, MARKER_NAME, MARKER_PUSH_FUNC_PTR, MARKER_POP_FUNC_PTR, ...)      \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace marker                                                                               \
    {                                                                                              \
    template <>                                                                                    \
    struct roctx_api_info<MARKER_TABLE, MARKER_API_ID> : roctx_domain_info<MARKER_TABLE>           \
    {                                                                                              \
        static constexpr auto is_range      = true;                                                \
        static constexpr auto table_idx     = MARKER_TABLE;                                        \
        static constexpr auto operation_idx = MARKER_API_ID;                                       \
        static constexpr auto name          = #MARKER_NAME;                                        \
                                                                                                   \
        using domain_type = roctx_domain_info<table_idx>;                                          \
        using this_type   = roctx_api_info<table_idx, operation_idx>;                              \
        using base_type   = roctx_api_impl<table_idx, operation_idx>;                              \
                                                                                                   \
        static constexpr auto callback_domain_idx = domain_type::callback_domain_idx;              \
        static constexpr auto buffered_domain_idx = domain_type::buffered_domain_idx;              \
                                                                                                   \
        using domain_type::args_type;                                                              \
        using domain_type::retval_type;                                                            \
        using domain_type::callback_data_type;                                                     \
                                                                                                   \
        static constexpr auto push_offset()                                                        \
        {                                                                                          \
            return offsetof(roctx_table_lookup<table_idx>::type, MARKER_PUSH_FUNC_PTR);            \
        }                                                                                          \
                                                                                                   \
        static constexpr auto pop_offset()                                                         \
        {                                                                                          \
            return offsetof(roctx_table_lookup<table_idx>::type, MARKER_POP_FUNC_PTR);             \
        }                                                                                          \
                                                                                                   \
        static auto& get_table() { return roctx_table_lookup<table_idx>{}(); }                     \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_table(TableT& _v)                                                         \
        {                                                                                          \
            return roctx_table_lookup<table_idx>{}(_v);                                            \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_push_table_func(TableT& _table)                                           \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                assert(_table != nullptr && "nullptr to MARKER table for " #MARKER_NAME            \
                                            " function");                                          \
                return _table->MARKER_PUSH_FUNC_PTR;                                               \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.MARKER_PUSH_FUNC_PTR;                                                \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        template <typename TableT>                                                                 \
        static auto& get_pop_table_func(TableT& _table)                                            \
        {                                                                                          \
            if constexpr(std::is_pointer<TableT>::value)                                           \
            {                                                                                      \
                assert(_table != nullptr && "nullptr to MARKER table for " #MARKER_NAME            \
                                            " function");                                          \
                return _table->MARKER_POP_FUNC_PTR;                                                \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                return _table.MARKER_POP_FUNC_PTR;                                                 \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        static auto& get_push_table_func() { return get_push_table_func(get_table()); }            \
        static auto& get_pop_table_func() { return get_pop_table_func(get_table()); }              \
                                                                                                   \
        template <typename DataT>                                                                  \
        static auto& get_api_data_args(DataT& _data)                                               \
        {                                                                                          \
            return _data.MARKER_NAME;                                                              \
        }                                                                                          \
                                                                                                   \
        template <typename RetT, typename... Args>                                                 \
        static auto get_push_functor(RetT (*)(Args...))                                            \
        {                                                                                          \
            return &base_type::push_functor<RetT, Args...>;                                        \
        }                                                                                          \
                                                                                                   \
        template <typename RetT, typename... Args>                                                 \
        static auto get_pop_functor(RetT (*)(Args...))                                             \
        {                                                                                          \
            return &base_type::pop_functor<RetT, Args...>;                                         \
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

#define MARKER_API_TABLE_LOOKUP_DEFINITION(TABLE_ID, TYPE)                                         \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace marker                                                                               \
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
    struct roctx_table_lookup<TABLE_ID>                                                            \
    {                                                                                              \
        using type = TYPE;                                                                         \
        auto& operator()(type& _v) const { return _v; }                                            \
        auto& operator()(type* _v) const { return *_v; }                                           \
        auto& operator()() const { return (*this)(get_table<TABLE_ID>()); }                        \
    };                                                                                             \
                                                                                                   \
    template <>                                                                                    \
    struct roctx_table_id_lookup<TYPE>                                                             \
    {                                                                                              \
        static constexpr auto value = TABLE_ID;                                                    \
    };                                                                                             \
    }                                                                                              \
    }

#define MARKER_API_TABLE_LOOKUP_DEFINITION_ALT(TABLE_ID, TYPE)                                     \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace marker                                                                               \
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
    struct roctx_table_lookup<TABLE_ID>                                                            \
    {                                                                                              \
        using type = TYPE;                                                                         \
        auto& operator()(type& _v) const { return _v; }                                            \
        auto& operator()(type* _v) const { return *_v; }                                           \
        auto& operator()() const { return (*this)(get_table<TABLE_ID>()); }                        \
    };                                                                                             \
    }                                                                                              \
    }
