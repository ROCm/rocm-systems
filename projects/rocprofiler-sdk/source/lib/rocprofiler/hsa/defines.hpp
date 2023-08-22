// Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
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

#define MEMBER_0(...)
#define MEMBER_1(PREFIX, FIELD)            PREFIX.FIELD
#define MEMBER_2(PREFIX, A, B)             MEMBER_1(PREFIX, A), MEMBER_1(PREFIX, B)
#define MEMBER_3(PREFIX, A, B, C)          MEMBER_2(PREFIX, A, B), MEMBER_1(PREFIX, C)
#define MEMBER_4(PREFIX, A, B, C, D)       MEMBER_3(PREFIX, A, B, C), MEMBER_1(PREFIX, D)
#define MEMBER_5(PREFIX, A, B, C, D, E)    MEMBER_4(PREFIX, A, B, C, D), MEMBER_1(PREFIX, E)
#define MEMBER_6(PREFIX, A, B, C, D, E, F) MEMBER_5(PREFIX, A, B, C, D, E), MEMBER_1(PREFIX, F)
#define MEMBER_7(PREFIX, A, B, C, D, E, F, G)                                                      \
    MEMBER_6(PREFIX, A, B, C, D, E, F), MEMBER_1(PREFIX, G)

#define MEMBER_8(PREFIX, A, B, C, D, E, F, G, H)                                                   \
    MEMBER_7(PREFIX, A, B, C, D, E, F, G), MEMBER_1(PREFIX, H)

#define MEMBER_9(PREFIX, A, B, C, D, E, F, G, H, I)                                                \
    MEMBER_8(PREFIX, A, B, C, D, E, F, G, H), MEMBER_1(PREFIX, I)

#define MEMBER_10(PREFIX, A, B, C, D, E, F, G, H, I, J)                                            \
    MEMBER_9(PREFIX, A, B, C, D, E, F, G, H, I), MEMBER_1(PREFIX, J)

#define MEMBER_11(PREFIX, A, B, C, D, E, F, G, H, I, J, K)                                         \
    MEMBER_10(PREFIX, A, B, C, D, E, F, G, H, I, J), MEMBER_1(PREFIX, K)

#define MEMBER_12(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L)                                      \
    MEMBER_11(PREFIX, A, B, C, D, E, F, G, H, I, J, K), MEMBER_1(PREFIX, L)

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

/// @def GET_MEMBER_FIELDS
/// @param VAR some struct instance
/// @param ... The member fields of the struct
///
/// @brief this macro is used to expand one variable (VAR) + one or more member fields (FIELDS)
/// into a sequence of something like: `(VAR.FIELD, ...)`
/// For example, `GET_MEMBER_FIELDS(foo, a, b, c)` would transform into `foo.a, foo.b, foo.c`:
///
/// @code{.cpp}
///
///     struct Foo
///     {
///         int    a;
///         float  b;
///         double c;
///     };
///
///     // some function taking int, float, and double
///     void some_function(int, float, double);
///
///     // overload to some_function accepting Foo instance and using
///     // the args to invoke "real" function
///     void some_function(Foo _foo_v)
///     {
///         some_function(GET_MEMBER_FIELDS(_foo_v, a, b, c));
///     }
///
///     int main()
///     {
///         Foo _foo_v = {-1, 0.5f, 2.0};
///         invoke_some_function(_foo_v);
///     }
///
/// @code
#define GET_MEMBER_FIELDS(VAR, ...)       IMPL_DETAIL_FOR_EACH(MEMBER_, VAR, __VA_ARGS__)
#define GET_NAMED_MEMBER_FIELDS(VAR, ...) IMPL_DETAIL_FOR_EACH(NAMED_MEMBER_, VAR, __VA_ARGS__)

#define HSA_API_INFO_DEFINITION_0(HSA_DOMAIN, HSA_TABLE, HSA_API_ID, HSA_FUNC, HSA_FUNC_PTR)       \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace hsa                                                                                  \
    {                                                                                              \
    template <>                                                                                    \
    struct hsa_api_info<HSA_API_ID>                                                                \
    {                                                                                              \
        static constexpr auto domain_idx    = HSA_DOMAIN;                                          \
        static constexpr auto table_idx     = HSA_TABLE;                                           \
        static constexpr auto operation_idx = HSA_API_ID;                                          \
        static constexpr auto name          = #HSA_FUNC;                                           \
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
            return _data.api_data.args.HSA_FUNC;                                                   \
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
        static std::string as_string(rocprofiler_hsa_trace_data_t)                                 \
        {                                                                                          \
            return std::string{name} + "()";                                                       \
        }                                                                                          \
                                                                                                   \
        static std::string as_named_string(rocprofiler_hsa_trace_data_t)                           \
        {                                                                                          \
            return std::string{name} + "()";                                                       \
        }                                                                                          \
                                                                                                   \
        static std::vector<std::pair<std::string, std::string>> as_arg_list(                       \
            rocprofiler_hsa_trace_data_t)                                                          \
        {                                                                                          \
            return {};                                                                             \
        }                                                                                          \
    };                                                                                             \
    }                                                                                              \
    }

#define HSA_API_INFO_DEFINITION_V(HSA_DOMAIN, HSA_TABLE, HSA_API_ID, HSA_FUNC, HSA_FUNC_PTR, ...)  \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace hsa                                                                                  \
    {                                                                                              \
    template <>                                                                                    \
    struct hsa_api_info<HSA_API_ID>                                                                \
    {                                                                                              \
        static constexpr auto domain_idx    = HSA_DOMAIN;                                          \
        static constexpr auto table_idx     = HSA_TABLE;                                           \
        static constexpr auto operation_idx = HSA_API_ID;                                          \
        static constexpr auto name          = #HSA_FUNC;                                           \
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
            return _data.api_data.args.HSA_FUNC;                                                   \
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
        static std::string as_string(rocprofiler_hsa_trace_data_t trace_data)                      \
        {                                                                                          \
            return utils::join(utils::join_args{std::string{name} + "(", ")", ", "},               \
                               GET_MEMBER_FIELDS(get_api_data_args(trace_data), __VA_ARGS__));     \
        }                                                                                          \
                                                                                                   \
        static std::string as_named_string(rocprofiler_hsa_trace_data_t trace_data)                \
        {                                                                                          \
            return utils::join(                                                                    \
                utils::join_args{std::string{name} + "(", ")", ", "},                              \
                GET_NAMED_MEMBER_FIELDS(get_api_data_args(trace_data), __VA_ARGS__));              \
        }                                                                                          \
                                                                                                   \
        static auto as_arg_list(rocprofiler_hsa_trace_data_t trace_data)                           \
        {                                                                                          \
            return utils::stringize(                                                               \
                GET_NAMED_MEMBER_FIELDS(get_api_data_args(trace_data), __VA_ARGS__));              \
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
