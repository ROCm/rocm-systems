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

#define IMPL_DETAIL_EXPAND(X) X
#define IMPL_DETAIL_FOR_EACH_NARG(...)                                                             \
    IMPL_DETAIL_FOR_EACH_NARG_(__VA_ARGS__, IMPL_DETAIL_FOR_EACH_RSEQ_N())
#define IMPL_DETAIL_FOR_EACH_NARG_(...)                                                       IMPL_DETAIL_EXPAND(IMPL_DETAIL_FOR_EACH_ARG_N(__VA_ARGS__))
#define IMPL_DETAIL_FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, N, ...) N
#define IMPL_DETAIL_FOR_EACH_RSEQ_N()                                                         12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
#define IMPL_DETAIL_CONCATENATE(X, Y)                                                         X##Y
#define IMPL_DETAIL_FOR_EACH_(N, MACRO, PREFIX, ...)                                               \
    IMPL_DETAIL_EXPAND(IMPL_DETAIL_CONCATENATE(MACRO, N)(PREFIX, __VA_ARGS__))
#define IMPL_DETAIL_FOR_EACH(MACRO, PREFIX, ...)                                                   \
    IMPL_DETAIL_FOR_EACH_(IMPL_DETAIL_FOR_EACH_NARG(__VA_ARGS__), MACRO, PREFIX, __VA_ARGS__)

#define ADDR_MEMBER_0(...)
#define ADDR_MEMBER_1(PREFIX, FIELD)      static_cast<void*>(&PREFIX.FIELD)
#define ADDR_MEMBER_2(PREFIX, A, B)       ADDR_MEMBER_1(PREFIX, A), ADDR_MEMBER_1(PREFIX, B)
#define ADDR_MEMBER_3(PREFIX, A, B, C)    ADDR_MEMBER_2(PREFIX, A, B), ADDR_MEMBER_1(PREFIX, C)
#define ADDR_MEMBER_4(PREFIX, A, B, C, D) ADDR_MEMBER_3(PREFIX, A, B, C), ADDR_MEMBER_1(PREFIX, D)
#define ADDR_MEMBER_5(PREFIX, A, B, C, D, E)                                                       \
    ADDR_MEMBER_4(PREFIX, A, B, C, D), ADDR_MEMBER_1(PREFIX, E)
#define ADDR_MEMBER_6(PREFIX, A, B, C, D, E, F)                                                    \
    ADDR_MEMBER_5(PREFIX, A, B, C, D, E), ADDR_MEMBER_1(PREFIX, F)
#define ADDR_MEMBER_7(PREFIX, A, B, C, D, E, F, G)                                                 \
    ADDR_MEMBER_6(PREFIX, A, B, C, D, E, F), ADDR_MEMBER_1(PREFIX, G)
#define ADDR_MEMBER_8(PREFIX, A, B, C, D, E, F, G, H)                                              \
    ADDR_MEMBER_7(PREFIX, A, B, C, D, E, F, G), ADDR_MEMBER_1(PREFIX, H)
#define ADDR_MEMBER_9(PREFIX, A, B, C, D, E, F, G, H, I)                                           \
    ADDR_MEMBER_8(PREFIX, A, B, C, D, E, F, G, H), ADDR_MEMBER_1(PREFIX, I)
#define ADDR_MEMBER_10(PREFIX, A, B, C, D, E, F, G, H, I, J)                                       \
    ADDR_MEMBER_9(PREFIX, A, B, C, D, E, F, G, H, I), ADDR_MEMBER_1(PREFIX, J)
#define ADDR_MEMBER_11(PREFIX, A, B, C, D, E, F, G, H, I, J, K)                                    \
    ADDR_MEMBER_10(PREFIX, A, B, C, D, E, F, G, H, I, J), ADDR_MEMBER_1(PREFIX, K)
#define ADDR_MEMBER_12(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L)                                 \
    ADDR_MEMBER_11(PREFIX, A, B, C, D, E, F, G, H, I, J, K), ADDR_MEMBER_1(PREFIX, L)

#define NAMED_MEMBER_0(...)
#define NAMED_MEMBER_1(PREFIX, FIELD)   std::make_pair(#FIELD, PREFIX.FIELD)
#define NAMED_MEMBER_2(PREFIX, A, B)    NAMED_MEMBER_1(PREFIX, A), NAMED_MEMBER_1(PREFIX, B)
#define NAMED_MEMBER_3(PREFIX, A, B, C) NAMED_MEMBER_2(PREFIX, A, B), NAMED_MEMBER_1(PREFIX, C)
#define NAMED_MEMBER_4(PREFIX, A, B, C, D)                                                         \
    NAMED_MEMBER_3(PREFIX, A, B, C), NAMED_MEMBER_1(PREFIX, D)
#define NAMED_MEMBER_5(PREFIX, A, B, C, D, E)                                                      \
    NAMED_MEMBER_4(PREFIX, A, B, C, D), NAMED_MEMBER_1(PREFIX, E)
#define NAMED_MEMBER_6(PREFIX, A, B, C, D, E, F)                                                   \
    NAMED_MEMBER_5(PREFIX, A, B, C, D, E), NAMED_MEMBER_1(PREFIX, F)
#define NAMED_MEMBER_7(PREFIX, A, B, C, D, E, F, G)                                                \
    NAMED_MEMBER_6(PREFIX, A, B, C, D, E, F), NAMED_MEMBER_1(PREFIX, G)
#define NAMED_MEMBER_8(PREFIX, A, B, C, D, E, F, G, H)                                             \
    NAMED_MEMBER_7(PREFIX, A, B, C, D, E, F, G), NAMED_MEMBER_1(PREFIX, H)
#define NAMED_MEMBER_9(PREFIX, A, B, C, D, E, F, G, H, I)                                          \
    NAMED_MEMBER_8(PREFIX, A, B, C, D, E, F, G, H), NAMED_MEMBER_1(PREFIX, I)
#define NAMED_MEMBER_10(PREFIX, A, B, C, D, E, F, G, H, I, J)                                      \
    NAMED_MEMBER_9(PREFIX, A, B, C, D, E, F, G, H, I), NAMED_MEMBER_1(PREFIX, J)
#define NAMED_MEMBER_11(PREFIX, A, B, C, D, E, F, G, H, I, J, K)                                   \
    NAMED_MEMBER_10(PREFIX, A, B, C, D, E, F, G, H, I, J), NAMED_MEMBER_1(PREFIX, K)
#define NAMED_MEMBER_12(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L)                                \
    NAMED_MEMBER_11(PREFIX, A, B, C, D, E, F, G, H, I, J, K), NAMED_MEMBER_1(PREFIX, L)

#define GET_ADDR_MEMBER_FIELDS(VAR, ...)  IMPL_DETAIL_FOR_EACH(ADDR_MEMBER_, VAR, __VA_ARGS__)
#define GET_NAMED_MEMBER_FIELDS(VAR, ...) IMPL_DETAIL_FOR_EACH(NAMED_MEMBER_, VAR, __VA_ARGS__)

#define HSA_API_META_DEFINITION(HSA_TABLE, HSA_API_ID, HSA_FUNC, HSA_FUNC_PTR)                     \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace hsa                                                                                  \
    {                                                                                              \
    template <>                                                                                    \
    struct hsa_api_meta<HSA_API_ID>                                                                \
    {                                                                                              \
        static constexpr auto table_idx     = HSA_TABLE;                                           \
        static constexpr auto operation_idx = HSA_API_ID;                                          \
        static constexpr auto name          = #HSA_FUNC;                                           \
                                                                                                   \
        using this_type     = hsa_api_meta<operation_idx>;                                         \
        using function_type = hsa_api_func<decltype(::HSA_FUNC)*>::function_type;                  \
                                                                                                   \
        static auto& get_table() { return hsa_table_lookup<table_idx>{}(); }                       \
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
        static auto& get_table_func() { return get_table_func(get_table()); }                      \
    };                                                                                             \
    }                                                                                              \
    }

#define HSA_API_INFO_DEFINITION_0(HSA_TABLE, HSA_API_ID, HSA_FUNC, HSA_FUNC_PTR)                   \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace hsa                                                                                  \
    {                                                                                              \
    template <>                                                                                    \
    struct hsa_api_info<HSA_API_ID>                                                                \
    {                                                                                              \
        static constexpr auto callback_domain_idx = ROCPROFILER_CALLBACK_TRACING_HSA_API;          \
        static constexpr auto buffered_domain_idx = ROCPROFILER_BUFFER_TRACING_HSA_API;            \
        static constexpr auto table_idx           = HSA_TABLE;                                     \
        static constexpr auto operation_idx       = HSA_API_ID;                                    \
        static constexpr auto name                = #HSA_FUNC;                                     \
                                                                                                   \
        using this_type = hsa_api_info<operation_idx>;                                             \
        using base_type = hsa_api_impl<operation_idx>;                                             \
                                                                                                   \
        static auto& get_table() { return hsa_table_lookup<table_idx>{}(); }                       \
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
        static auto& get_table_func() { return get_table_func(get_table()); }                      \
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
            if constexpr(std::is_void<RetT>::value)                                                \
                return [](Args... args) -> RetT { base_type::functor(args...); };                  \
            else                                                                                   \
                return [](Args... args) -> RetT { return base_type::functor(args...); };           \
        }                                                                                          \
                                                                                                   \
        static auto get_functor() { return get_functor(get_table_func()); }                        \
                                                                                                   \
        static std::vector<void*> as_arg_addr(rocprofiler_callback_tracing_hsa_api_data_t)         \
        {                                                                                          \
            return std::vector<void*>{};                                                           \
        }                                                                                          \
                                                                                                   \
        static std::vector<std::pair<std::string, std::string>> as_arg_list(                       \
            rocprofiler_callback_tracing_hsa_api_data_t)                                           \
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
    struct hsa_api_info<HSA_API_ID>                                                                \
    {                                                                                              \
        static constexpr auto callback_domain_idx = ROCPROFILER_CALLBACK_TRACING_HSA_API;          \
        static constexpr auto buffered_domain_idx = ROCPROFILER_BUFFER_TRACING_HSA_API;            \
        static constexpr auto table_idx           = HSA_TABLE;                                     \
        static constexpr auto operation_idx       = HSA_API_ID;                                    \
        static constexpr auto name                = #HSA_FUNC;                                     \
                                                                                                   \
        using this_type = hsa_api_info<operation_idx>;                                             \
        using base_type = hsa_api_impl<operation_idx>;                                             \
                                                                                                   \
        static auto& get_table() { return hsa_table_lookup<table_idx>{}(); }                       \
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
        static auto& get_table_func() { return get_table_func(get_table()); }                      \
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
            if constexpr(std::is_void<RetT>::value)                                                \
                return [](Args... args) -> RetT { base_type::functor(args...); };                  \
            else                                                                                   \
                return [](Args... args) -> RetT { return base_type::functor(args...); };           \
        }                                                                                          \
                                                                                                   \
        static auto get_functor() { return get_functor(get_table_func()); }                        \
                                                                                                   \
        static std::vector<void*> as_arg_addr(                                                     \
            rocprofiler_callback_tracing_hsa_api_data_t trace_data)                                \
        {                                                                                          \
            return std::vector<void*>{                                                             \
                GET_ADDR_MEMBER_FIELDS(get_api_data_args(trace_data.args), __VA_ARGS__)};          \
        }                                                                                          \
                                                                                                   \
        static auto as_arg_list(rocprofiler_callback_tracing_hsa_api_data_t trace_data)            \
        {                                                                                          \
            return utils::stringize(                                                               \
                GET_NAMED_MEMBER_FIELDS(get_api_data_args(trace_data.args), __VA_ARGS__));         \
        }                                                                                          \
    };                                                                                             \
    }                                                                                              \
    }

#define HSA_API_TABLE_LOOKUP_DEFINITION(TABLE_ID, MEMBER)                                          \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace hsa                                                                                  \
    {                                                                                              \
    template <>                                                                                    \
    struct hsa_table_lookup<TABLE_ID>                                                              \
    {                                                                                              \
        auto& operator()(hsa_api_table_t& _v) const { return _v.MEMBER; }                          \
        auto& operator()(hsa_api_table_t* _v) const { return _v->MEMBER; }                         \
        auto& operator()() const { return (*this)(get_table()); }                                  \
    };                                                                                             \
    }                                                                                              \
    }
