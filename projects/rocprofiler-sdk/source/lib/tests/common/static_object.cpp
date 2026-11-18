// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/static_object.hpp"
#include "lib/common/static_tl_object.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <thread>

namespace
{
struct alignas(4096) payload
{
    explicit payload(uint64_t v)
    : value{v}
    {}
    uint64_t value;
};

// Read the actual address even when optimization assumes typed pointers are
// aligned. The singleton's byte buffer must honor payload's alignment itself.
void
check_address(const payload* ptr)
{
    volatile uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(address % alignof(payload), 0u);
}

struct regular_tag
{};
struct factory_tag
{};
struct thread_tag
{};
}  // namespace

TEST(static_object, respects_type_alignment)
{
    using object = rocprofiler::common::static_object<payload, regular_tag>;
    auto* ptr    = object::construct(uint64_t{17});
    check_address(ptr);
    EXPECT_EQ(ptr->value, 17);
    EXPECT_EQ(object::get(), ptr);
}

TEST(static_object, factory_receives_aligned_storage)
{
    using object = rocprofiler::common::static_object<payload, factory_tag>;
    auto* ptr    = object::construct_via_function(+[](void* storage) {
        volatile uintptr_t address = reinterpret_cast<uintptr_t>(storage);
        EXPECT_EQ(address % alignof(payload), 0u);
        return new(storage) payload{23};
    });
    check_address(ptr);
    EXPECT_EQ(ptr->value, 23);
}

TEST(static_tl_object, respects_type_alignment_on_worker_thread)
{
    std::thread worker{[] {
        using object = rocprofiler::common::static_tl_object<payload, thread_tag>;
        auto* ptr    = object::construct(uint64_t{31});
        check_address(ptr);
        EXPECT_EQ(ptr->value, 31);
        EXPECT_EQ(object::get(), ptr);
    }};
    worker.join();
}
