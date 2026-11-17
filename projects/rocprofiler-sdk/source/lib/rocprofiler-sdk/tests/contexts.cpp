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
#include "lib/common/logging.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/context/domain.hpp"

#include <gtest/gtest.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

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

TEST(contexts, correlation_id_parallel_allocation)
{
    constexpr size_t total = 65536;
    for(size_t num_threads : {1, 2, 4, 8})
    {
        SCOPED_TRACE(num_threads);
        std::vector<std::vector<context::correlation_id*>> ids(
            num_threads, std::vector<context::correlation_id*>(total / num_threads));
        std::vector<uint64_t>    tids(num_threads);
        std::vector<std::thread> threads;
        std::mutex               mutex;
        std::condition_variable  changed;
        size_t                   ready = 0, finished = 0;
        bool                     start = false;
        for(size_t thread = 0; thread < num_threads; ++thread)
        {
            threads.emplace_back([&, thread]() {
                tids[thread] = common::get_tid();
                {
                    std::unique_lock<std::mutex> lock{mutex};
                    ++ready;
                    changed.notify_all();
                    changed.wait(lock, [&]() { return start; });
                }
                for(auto& id : ids[thread])
                {
                    id = context::correlation_tracing_service::construct(1);
                    if(id) context::pop_latest_correlation_id(id);
                }
                {
                    std::lock_guard<std::mutex> lock{mutex};
                    ++finished;
                }
                changed.notify_all();
            });
        }
        std::unique_lock<std::mutex> lock{mutex};
        changed.wait(lock, [&]() { return ready == num_threads; });
        auto begin = std::chrono::steady_clock::now();
        start      = true;
        changed.notify_all();
        changed.wait(lock, [&]() { return finished == num_threads; });
        auto seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        lock.unlock();
        for(auto& thread : threads)
            thread.join();
        RecordProperty("ids_per_second_" + std::to_string(num_threads),
                       std::to_string(static_cast<uint64_t>(total / seconds)));

        std::unordered_set<uint64_t> internal_ids;
        for(size_t thread = 0; thread < num_threads; ++thread)
        {
            for(auto* id : ids[thread])
            {
                ASSERT_NE(id, nullptr);
                EXPECT_EQ(id->thread_idx, tids[thread]);
                EXPECT_EQ(id->ancestor, 0);
                EXPECT_TRUE(internal_ids.emplace(id->internal).second);
                EXPECT_EQ(id->add_ref_count(), 1);
                EXPECT_EQ(id->sub_ref_count(), 2);
                EXPECT_EQ(id->sub_ref_count(), 1);
            }
        }
    }
}

TEST(contexts, correlation_id_nested_growth)
{
    ASSERT_EQ(context::get_latest_correlation_id(), nullptr);
    auto* parent = context::correlation_tracing_service::construct(1);
    ASSERT_NE(parent, nullptr);
    for(size_t i = 0; i < 8193; ++i)
    {
        auto* child = context::correlation_tracing_service::construct(1);
        ASSERT_NE(child, nullptr);
        EXPECT_EQ(child->ancestor, parent->internal);
        EXPECT_EQ(context::get_latest_correlation_id(), child);
        EXPECT_EQ(context::pop_latest_correlation_id(child), parent);
        EXPECT_EQ(child->sub_ref_count(), 1);
    }
    EXPECT_EQ(context::get_latest_correlation_id(), parent);
    EXPECT_EQ(context::pop_latest_correlation_id(parent), nullptr);
    EXPECT_EQ(parent->sub_ref_count(), 1);
}

TEST(contextsDeathTest, correlation_id_finalize_races_allocation)
{
    ASSERT_EXIT(
        {
            absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
            constexpr size_t                                   num_threads = 8;
            std::vector<std::vector<context::correlation_id*>> ids(num_threads);
            std::vector<std::thread>                           threads;
            std::atomic<size_t>                                ready{0};
            std::atomic<size_t>                                errors{0};
            std::atomic<bool>                                  start{false};
            std::atomic<bool>                                  sealed{false};
            context::correlation_id*                           exited_id = nullptr;
            std::thread                                        exited_thread([&]() {
                exited_id = context::correlation_tracing_service::construct(2);
                if(exited_id) context::pop_latest_correlation_id(exited_id);
            });
            exited_thread.join();
            for(size_t thread = 0; thread < num_threads; ++thread)
            {
                threads.emplace_back([&, thread]() {
                    ids[thread].reserve(8193);
                    auto allocate = [&]() {
                        auto* id = context::correlation_tracing_service::construct(1);
                        if(id)
                        {
                            ids[thread].emplace_back(id);
                            context::pop_latest_correlation_id(id);
                        }
                        return id;
                    };
                    // Exercise both cached storage and first registration during the sweep.
                    if(thread % 2 == 0 && !allocate()) ++errors;
                    ++ready;
                    while(!start.load())
                        std::this_thread::yield();
                    for(size_t i = 0; i < 8192; ++i)
                        if(!allocate()) break;
                    while(!sealed.load())
                        std::this_thread::yield();
                    if(allocate()) ++errors;
                });
            }
            while(ready.load() != num_threads)
                std::this_thread::yield();
            start.store(true);
            // Invoke the sweep directly to test its sealing independently of the
            // registration status check, which a constructor may already have passed.
            context::correlation_id_finalize();
            sealed.store(true);
            for(auto& thread : threads)
                thread.join();
            for(const auto& storage : ids)
                for(const auto* id : storage)
                    if(id->get_ref_count() != 0 || id->internal == 0) ++errors;
            if(!exited_id || exited_id->get_ref_count() != 0) ++errors;
            std::thread late([&]() {
                if(context::correlation_tracing_service::construct(1)) ++errors;
            });
            late.join();
            context::correlation_id_finalize();
            _exit(errors.load() == 0 ? 0 : 1);
        },
        ::testing::ExitedWithCode(0),
        "");
}
