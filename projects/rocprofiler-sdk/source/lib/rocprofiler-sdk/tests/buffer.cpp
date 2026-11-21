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

#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/common/units.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include <gtest/gtest.h>

#include <pthread.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <limits>
#include <mutex>
#include <random>
#include <thread>
#include <typeinfo>
#include <vector>

TEST(rocprofiler_lib, buffer)
{
    namespace buffer = ::rocprofiler::buffer;
    namespace common = ::rocprofiler::common;

    ASSERT_EQ(buffer::get_buffers()->size(), 0)
        << "no buffers should have been created at this point";
    auto buffer_id = buffer::allocate_buffer();

    EXPECT_TRUE(buffer_id) << "failed to allocate buffer";
    EXPECT_GT(buffer_id->handle, 0);
    EXPECT_TRUE(buffer::is_valid_buffer_id(*buffer_id)) << "id=" << buffer_id->handle;
    ASSERT_EQ(buffer::get_buffers()->size(), 1) << "incorrect number of buffers created";

    // get pointer to buffer
    auto* buffer_v = buffer::get_buffer(*buffer_id);
    ASSERT_NE(buffer_v, nullptr) << "get_buffer returned a nullptr. id=" << buffer_id->handle;
    EXPECT_EQ(buffer_v->buffer_id, buffer_id->handle);

    buffer_v->watermark = common::units::get_page_size();
    EXPECT_EQ(buffer_v->get_internal_buffer().get_num_record_headers(), 0);

    EXPECT_TRUE(buffer_v->get_internal_buffer().allocate(sizeof(rocprofiler_buffer_id_t)));

    EXPECT_EQ(buffer_v->get_internal_buffer().capacity(), common::units::get_page_size());

    auto data = *buffer_id;
    buffer_v->emplace(1, 1, data);

    EXPECT_EQ(buffer_v->get_internal_buffer().get_num_record_headers(), 1);

    auto flush_status = buffer::flush(*buffer_id, true);
    EXPECT_EQ(flush_status, ROCPROFILER_STATUS_SUCCESS);

    auto destroy_status = rocprofiler_destroy_buffer(*buffer_id);
    EXPECT_EQ(destroy_status, ROCPROFILER_STATUS_SUCCESS);
}

namespace
{
namespace buffer = rocprofiler::buffer;

buffer::instance*
make_test_buffer(rocprofiler_buffer_policy_t policy, size_t bytes = 4096)
{
    auto id = buffer::allocate_buffer();
    if(!id) throw std::runtime_error{"Could not allocate test buffer"};
    auto* result      = buffer::get_buffer(*id);
    result->policy    = policy;
    result->watermark = std::numeric_limits<uint64_t>::max();
    for(size_t i = 0; i < (policy == ROCPROFILER_BUFFER_POLICY_LOSSLESS ? result->size : 1); ++i)
        result->buffers.at(i).allocate(bytes);
    return result;
}

struct copy_control
{
    std::mutex              mutex;
    std::condition_variable changed;
    size_t                  entered = 0;
    bool                    release = false;
};

struct held_record
{
    copy_control* control = nullptr;
    uint64_t      value   = 0;
    held_record(copy_control& state, uint64_t data)
    : control{&state}
    , value{data}
    {}
    held_record(const held_record& rhs)
    : control{rhs.control}
    {
        std::unique_lock<std::mutex> lock{control->mutex};
        ++control->entered;
        control->changed.notify_all();
        control->changed.wait(lock, [&]() { return control->release; });
        value = rhs.value;
    }
};
}  // namespace

TEST(buffer_safety, busy_producer_does_not_rotate_or_flush)
{
    auto*               target = make_test_buffer(ROCPROFILER_BUFFER_POLICY_LOSSLESS);
    std::atomic<size_t> seen{0};
    target->callback_data = &seen;
    target->callback =
        [](auto, auto, rocprofiler_record_header_t** headers, size_t count, void* data, auto) {
            for(size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(static_cast<held_record*>(headers[i]->payload)->value, 42);
            }
            static_cast<std::atomic<size_t>*>(data)->fetch_add(count);
        };
    copy_control state;
    held_record  input{state, 42};
    std::thread  first([&]() { EXPECT_TRUE(target->emplace(1, 1, input)); });
    bool         entered = false;
    {
        std::unique_lock<std::mutex> lock{state.mutex};
        entered = state.changed.wait_for(
            lock, std::chrono::seconds{2}, [&]() { return state.entered == 1; });
    }
    EXPECT_EQ(buffer::flush(target->buffer_id, false), ROCPROFILER_STATUS_ERROR_BUFFER_BUSY);
    EXPECT_EQ(seen.load(), 0);
    EXPECT_EQ(target->buffer_idx.load(), 0);
    {
        std::lock_guard<std::mutex> lock{state.mutex};
        state.release = true;
    }
    state.changed.notify_all();
    first.join();
    EXPECT_TRUE(entered);
    EXPECT_EQ(buffer::flush(target->buffer_id, true), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(seen.load(), 1);
    EXPECT_EQ(rocprofiler_destroy_buffer({target->buffer_id}), ROCPROFILER_STATUS_SUCCESS);
}

TEST(buffer_safety, busy_destruction_rolls_back_and_discard_slot_reopens)
{
    for(auto policy : {ROCPROFILER_BUFFER_POLICY_LOSSLESS, ROCPROFILER_BUFFER_POLICY_DISCARD})
    {
        auto* target = make_test_buffer(policy);
        if(policy == ROCPROFILER_BUFFER_POLICY_LOSSLESS)
        {
            ASSERT_EQ(buffer::flush(target->buffer_id, true), ROCPROFILER_STATUS_SUCCESS);
        }
        struct callback_state
        {
            std::promise<void>       entered;
            std::shared_future<void> release;
            size_t                   seen = 0;
        };
        std::promise<void> release;
        callback_state     state{{}, release.get_future().share(), 0};
        auto               entered = state.entered.get_future();
        target->callback_data      = &state;
        target->callback           = [](auto, auto, auto, size_t count, void* data, auto) {
            auto& current = *static_cast<callback_state*>(data);
            if(current.seen == 0)
            {
                current.entered.set_value();
                current.release.wait();
            }
            current.seen += count;
        };
        uint64_t value = 42;
        EXPECT_TRUE(target->emplace(1, 1, value));
        EXPECT_EQ(buffer::flush(target->buffer_id, false), ROCPROFILER_STATUS_SUCCESS);
        EXPECT_EQ(entered.wait_for(std::chrono::seconds{2}), std::future_status::ready);
        EXPECT_EQ(rocprofiler_destroy_buffer({target->buffer_id}),
                  ROCPROFILER_STATUS_ERROR_BUFFER_BUSY);
        if(policy == ROCPROFILER_BUFFER_POLICY_LOSSLESS)
        {
            // Slot zero was acquired before destruction encountered busy slot one.
            bool busy = target->syncer[0].test_and_set(std::memory_order_acquire);
            EXPECT_FALSE(busy);
            if(!busy) target->syncer[0].clear(std::memory_order_release);
            EXPECT_TRUE(target->emplace(1, 1, value));
        }
        else
        {
            auto generation = target->buffer_idx.load();
            EXPECT_EQ(buffer::flush(target->buffer_id, false),
                      ROCPROFILER_STATUS_ERROR_BUFFER_BUSY);
            EXPECT_EQ(target->buffer_idx.load(), generation);
        }
        release.set_value();
        EXPECT_EQ(buffer::flush(target->buffer_id, true), ROCPROFILER_STATUS_SUCCESS);
        if(policy == ROCPROFILER_BUFFER_POLICY_DISCARD)
        {
            EXPECT_TRUE(target->emplace(1, 1, value));
            EXPECT_EQ(buffer::flush(target->buffer_id, true), ROCPROFILER_STATUS_SUCCESS);
            EXPECT_EQ(target->buffer_idx.load(), 0);
        }
        EXPECT_EQ(state.seen, 2);
        EXPECT_EQ(rocprofiler_destroy_buffer({target->buffer_id}), ROCPROFILER_STATUS_SUCCESS);
    }
}

TEST(buffer_safety, rotation_and_concurrent_flush_preserve_accepted_records)
{
    constexpr size_t threads = 8, per_thread = 2048, total = threads * per_thread;
    for(auto policy : {ROCPROFILER_BUFFER_POLICY_LOSSLESS, ROCPROFILER_BUFFER_POLICY_DISCARD})
    {
        SCOPED_TRACE(policy);
        auto* target      = make_test_buffer(policy);
        target->watermark = 2048;
        struct record
        {
            uint64_t index;
            uint64_t checksum;
        };
        std::vector<unsigned> seen(total, 0);
        std::vector<uint8_t>  accepted(total, 0);
        target->callback_data = &seen;
        target->callback      = [](auto,
                              auto,
                              rocprofiler_record_header_t** headers,
                              size_t                        count,
                              void*                         data,
                              uint64_t) {
            auto& values = *static_cast<std::vector<unsigned>*>(data);
            for(size_t i = 0; i < count; ++i)
            {
                const auto& value = *static_cast<record*>(headers[i]->payload);
                ASSERT_LT(value.index, values.size());
                EXPECT_EQ(value.checksum, ~value.index);
                ++values.at(value.index);
            }
        };
        std::atomic<bool>        start{false}, done{false};
        std::vector<std::thread> writers;
        for(size_t thread = 0; thread < threads; ++thread)
            writers.emplace_back([&, thread]() {
                while(!start.load())
                    std::this_thread::yield();
                for(size_t i = 0; i < per_thread; ++i)
                {
                    uint64_t index = thread * per_thread + i;
                    record   value{index, ~index};
                    accepted[index] = target->emplace(1, 1, value) ? 1 : 0;
                    if(policy == ROCPROFILER_BUFFER_POLICY_LOSSLESS)
                    {
                        EXPECT_EQ(accepted[index], 1);
                    }
                }
            });
        std::thread flusher([&]() {
            while(!start.load())
                std::this_thread::yield();
            while(!done.load())
            {
                auto status = buffer::flush(target->buffer_id, false);
                EXPECT_TRUE(status == ROCPROFILER_STATUS_SUCCESS ||
                            status == ROCPROFILER_STATUS_ERROR_BUFFER_BUSY);
                std::this_thread::yield();
            }
        });
        start.store(true);
        for(auto& thread : writers)
            thread.join();
        done.store(true);
        flusher.join();
        for(int i = 0; i < target->size; ++i)
            EXPECT_EQ(buffer::flush(target->buffer_id, true), ROCPROFILER_STATUS_SUCCESS);
        uint64_t drops = 0;
        for(size_t i = 0; i < total; ++i)
        {
            EXPECT_EQ(seen[i], accepted[i]);
            if(accepted[i] == 0) ++drops;
        }
        EXPECT_EQ(target->drop_count.load(), drops);
        EXPECT_EQ(rocprofiler_destroy_buffer({target->buffer_id}), ROCPROFILER_STATUS_SUCCESS);
    }
}
