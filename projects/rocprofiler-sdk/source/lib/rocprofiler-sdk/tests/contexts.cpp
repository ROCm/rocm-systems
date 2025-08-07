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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/demangle.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/context/domain.hpp"

#include <gtest/gtest.h>
#include <memory>

namespace context = ::rocprofiler::context;
namespace common  = ::rocprofiler::common;

namespace
{
#define EXPECT_ROCP_SUCCESS(...)                                                                   \
    EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, (__VA_ARGS__)) << #__VA_ARGS__

#define EXPECT_ROCP_SUCCESS_STREAM(_VAR_NAME, ...)                                                 \
    auto _VAR_NAME = (__VA_ARGS__);                                                                \
    EXPECT_EQ(ROCPROFILER_STATUS_SUCCESS, _VAR_NAME) << #__VA_ARGS__ << " :: "

template <typename Tp>
auto
get_operation_name_impl(Tp kind, uint32_t op)
{
    const char* opname = "(<unknown>)";

    if constexpr(std::is_same<Tp, rocprofiler_callback_tracing_kind_t>::value)
        EXPECT_ROCP_SUCCESS(
            rocprofiler_query_callback_tracing_kind_operation_name(kind, op, &opname, nullptr));
    else if constexpr(std::is_same<Tp, rocprofiler_buffer_tracing_kind_t>::value)
        EXPECT_ROCP_SUCCESS(
            rocprofiler_query_buffer_tracing_kind_operation_name(kind, op, &opname, nullptr));
    else
        static_assert(common::mpl::assert_false<Tp>::value, "invalid type");

    return std::string_view{opname};
}

#define get_operation_name(...) get_operation_name_impl(__VA_ARGS__)

template <typename Tp>
auto
get_operations_impl(Tp kind)
{
    using opvector_t = std::map<rocprofiler_tracing_operation_t, std::string_view>;

    auto iterate_operations = [](Tp _kind_v, rocprofiler_tracing_operation_t op, void* data) {
        auto* _data = static_cast<opvector_t*>(data);

        _data->emplace(op, get_operation_name(_kind_v, op));
        return 0;
    };

    auto opdata = opvector_t{};
    if constexpr(std::is_same<Tp, rocprofiler_callback_tracing_kind_t>::value)
        rocprofiler_iterate_callback_tracing_kind_operations(kind, iterate_operations, &opdata);
    else if constexpr(std::is_same<Tp, rocprofiler_buffer_tracing_kind_t>::value)
        rocprofiler_iterate_buffer_tracing_kind_operations(kind, iterate_operations, &opdata);
    else
        static_assert(common::mpl::assert_false<Tp>::value, "invalid type");

    return opdata;
}

#define get_operations(...) get_operations_impl(__VA_ARGS__)

template <typename Tp>
auto
get_domain_name(Tp idx_v)
{
    const char* _name = "(<unknown>)";

    if constexpr(std::is_same<Tp, rocprofiler_callback_tracing_kind_t>::value)
        EXPECT_ROCP_SUCCESS(rocprofiler_query_callback_tracing_kind_name(idx_v, &_name, nullptr));
    else if constexpr(std::is_same<Tp, rocprofiler_buffer_tracing_kind_t>::value)
        EXPECT_ROCP_SUCCESS(rocprofiler_query_buffer_tracing_kind_name(idx_v, &_name, nullptr));
    else
        static_assert(common::mpl::assert_false<Tp>::value, "invalid type");

    return std::string_view{_name};
}

template <typename Tp>
struct kind_info;

template <>
struct kind_info<context::callback_tracing_service>
{
    using type = rocprofiler_callback_tracing_kind_t;
};

template <>
struct kind_info<context::buffer_tracing_service>
{
    using type = rocprofiler_buffer_tracing_kind_t;
};

template <typename Tp>
using kind_info_t = typename kind_info<Tp>::type;

template <typename Tp>
auto
add_domain_impl(Tp* tracer, int val)
{
    using kind_type = kind_info_t<Tp>;

    static auto type_name = common::cxx_demangle(typeid(kind_type).name());

    auto idx = static_cast<kind_type>(val);

    auto loc_info = std::stringstream{};
    loc_info << type_name << " (kind=" << val << ") :: " << get_domain_name(idx);

    // should initially be false
    EXPECT_FALSE(tracer->domains(idx)) << loc_info.str();

    EXPECT_ROCP_SUCCESS_STREAM(_status, context::add_domain(tracer->domains, idx))
        << loc_info.str() << " returned " << _status
        << " :: " << rocprofiler_get_status_string(_status);
    EXPECT_TRUE(tracer->domains(idx)) << loc_info.str();
}

#define add_domain(...) add_domain_impl(__VA_ARGS__)

template <typename Tp>
auto
add_domain_op_impl(Tp* tracer, int val, uint32_t op)
{
    using kind_type = kind_info_t<Tp>;

    static auto type_name = common::cxx_demangle(typeid(kind_type).name());

    auto idx = static_cast<kind_type>(val);

    auto loc_info = std::stringstream{};
    loc_info << type_name << " (kind=" << val << ", op=" << op << ") :: " << get_domain_name(idx);

    // conditional enabling of domain
    if(!tracer->domains(idx)) add_domain(tracer, val);

    EXPECT_ROCP_SUCCESS_STREAM(_status, context::add_domain_op(tracer->domains, idx, op))
        << loc_info.str() << " returned " << _status
        << " :: " << rocprofiler_get_status_string(_status);
    EXPECT_TRUE(tracer->domains(idx, op)) << loc_info.str();
}

#define add_domain_op(...) add_domain_op_impl(__VA_ARGS__)

template <typename Tp, typename BoolT = std::true_type>
auto
check_operations_impl(Tp* tracer, int val, BoolT = {})
{
    using kind_type = kind_info_t<Tp>;

    auto idx = static_cast<kind_type>(val);

    auto operations = get_operations(idx);
    for(auto oitr : operations)
    {
        if constexpr(BoolT::value)
        {
            EXPECT_TRUE(tracer->domains(idx, oitr.first))
                << get_domain_name(idx) << " (operation=" << oitr.first << "/" << oitr.second
                << ")";
        }
        else
        {
            EXPECT_FALSE(tracer->domains(idx, oitr.first))
                << get_domain_name(idx) << " (operation=" << oitr.first << "/" << oitr.second
                << ")";
        }
    }
}

#define check_operations(...) check_operations_impl(__VA_ARGS__)

template <typename Tp, typename BoolT>
auto
check_operation_impl(Tp* tracer, int val, int op, BoolT)
{
    using kind_type = kind_info_t<Tp>;

    auto idx = static_cast<kind_type>(val);

    auto operations = get_operations(idx);
    auto opname     = operations.at(op);

    if constexpr(BoolT::value)
    {
        EXPECT_TRUE(tracer->domains(idx, op))
            << get_domain_name(idx) << " (operation=" << op << "/" << opname << ")";
    }
    else
    {
        EXPECT_FALSE(tracer->domains(idx, op))
            << get_domain_name(idx) << " (operation=" << op << "/" << opname << ")";
    }
}

#define check_operation(...) check_operation_impl(__VA_ARGS__)
}  // namespace

TEST(contexts, callback_tracing)
{
    constexpr auto none = ROCPROFILER_CALLBACK_TRACING_NONE;
    constexpr auto last = ROCPROFILER_CALLBACK_TRACING_LAST;

    auto get_tracer = []() -> auto*
    {
        static auto ctx = context::context{};
        ctx.callback_tracer.reset();
        ctx.callback_tracer = std::make_unique<context::callback_tracing_service>();
        return ctx.callback_tracer.get();
    };

    {
        auto* tracer = get_tracer();

        EXPECT_EQ(tracer->callback_data.size(), last);

        for(int i = none + 1; i < last; ++i)
        {
            auto idx = static_cast<rocprofiler_callback_tracing_kind_t>(i);
            EXPECT_FALSE(tracer->domains(idx)) << "i=" << i << " :: " << get_domain_name(idx);
        }

        for(int i = none + 1; i < last; ++i)
        {
            add_domain(tracer, i);
            check_operations(tracer, i);
        }

        check_operations(tracer, none, std::false_type{});
        check_operations(tracer, last, std::false_type{});
    }

    {
        auto* tracer = get_tracer();

        for(int i = last - 1; i > none; --i)
        {
            add_domain(tracer, i);
            check_operations(tracer, i);
        }
    }

    {
        auto* tracer = get_tracer();

        auto fully_enabled = std::set<int>{ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                           ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                           ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                                           ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                                           ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API};

        for(auto i : fully_enabled)
        {
            add_domain(tracer, i);
            check_operations(tracer, i);
        }

        for(int i = none + 1; i < last; ++i)
        {
            if(fully_enabled.count(i) == 0)
            {
                check_operations(tracer, i, std::false_type{});
            }
        }

        add_domain_op(tracer,
                      ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API,
                      ROCPROFILER_HIP_COMPILER_API_ID___hipPushCallConfiguration);

        auto extra_enabled = fully_enabled;
        extra_enabled.emplace(ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API);

        for(auto itrv : extra_enabled)
        {
            auto itr = static_cast<rocprofiler_callback_tracing_kind_t>(itrv);
            EXPECT_TRUE(tracer->domains(itr)) << get_domain_name(itr);
        }

        check_operation(tracer,
                        ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API,
                        ROCPROFILER_HIP_COMPILER_API_ID___hipPushCallConfiguration,
                        std::true_type{});

        auto operations = get_operations(ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API);
        operations.erase(ROCPROFILER_HIP_COMPILER_API_ID___hipPushCallConfiguration);

        for(auto itr : operations)
        {
            check_operation(tracer,
                            ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API,
                            itr.first,
                            std::false_type{});
        }
    }

    {
        auto* tracer = get_tracer();
        for(int i = none + 1; i < last; ++i)
        {
            check_operations(tracer, i, std::false_type{});
        }
    }
}

TEST(contexts, buffer_tracing)
{
    constexpr auto none = ROCPROFILER_BUFFER_TRACING_NONE;
    constexpr auto last = ROCPROFILER_BUFFER_TRACING_LAST;

    auto get_tracer = []() -> auto*
    {
        static auto ctx = context::context{};
        ctx.buffered_tracer.reset();
        ctx.buffered_tracer = std::make_unique<context::buffer_tracing_service>();
        return ctx.buffered_tracer.get();
    };

    {
        auto* tracer = get_tracer();

        EXPECT_EQ(tracer->buffer_data.size(), last);

        for(int i = none + 1; i < last; ++i)
        {
            auto idx = static_cast<rocprofiler_buffer_tracing_kind_t>(i);
            EXPECT_FALSE(tracer->domains(idx)) << "i=" << i << " :: " << get_domain_name(idx);
        }

        for(int i = none + 1; i < last; ++i)
        {
            add_domain(tracer, i);
            check_operations(tracer, i);
        }
    }

    {
        auto* tracer = get_tracer();
        for(int i = last - 1; i > none; --i)
        {
            add_domain(tracer, i);
            check_operations(tracer, i);
        }
    }

    {
        auto* tracer = get_tracer();
        for(int i = none + 1; i < last; ++i)
        {
            check_operations(tracer, i, std::false_type{});
        }
    }
}
