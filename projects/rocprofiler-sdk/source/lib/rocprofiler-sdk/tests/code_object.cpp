// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/code_object/hsa/code_object.hpp"
#include "lib/rocprofiler-sdk/code_object/hsa/kernel_symbol.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"

#include <gtest/gtest.h>

#include <utility>

namespace
{
namespace hsa = rocprofiler::code_object::hsa;

template <typename Tp>
class code_object_notifications : public ::testing::Test
{};

using notification_types = ::testing::Types<hsa::code_object, hsa::kernel_symbol>;
TYPED_TEST_SUITE(code_object_notifications, notification_types);

TYPED_TEST(code_object_notifications, per_context)
{
    auto first  = rocprofiler::context::context{};
    auto second = rocprofiler::context::context{};
    auto object = TypeParam{};

    auto notify = [](auto& notified, const auto* ctx) {
        return notified.wlock([ctx](auto& contexts) { return contexts.emplace(ctx).second; });
    };

    EXPECT_TRUE(notify(object.beg_notified, &first));
    EXPECT_FALSE(notify(object.beg_notified, &first));
    EXPECT_TRUE(notify(object.beg_notified, &second));
    EXPECT_FALSE(notify(object.beg_notified, &second));

    EXPECT_TRUE(notify(object.end_notified, &first));
    EXPECT_FALSE(notify(object.end_notified, &first));
    EXPECT_TRUE(notify(object.end_notified, &second));
    EXPECT_FALSE(notify(object.end_notified, &second));
}

TYPED_TEST(code_object_notifications, move_preserves_context_state)
{
    auto first  = rocprofiler::context::context{};
    auto second = rocprofiler::context::context{};
    auto source = TypeParam{};

    source.beg_notified.wlock([&](auto& contexts) {
        contexts.emplace(&first);
        contexts.emplace(&second);
    });
    source.end_notified.wlock([&](auto& contexts) { contexts.emplace(&first); });
    source.user_data.wlock([&](auto& data) {
        data[&first].value  = 11;
        data[&second].value = 22;
    });

    auto check_state = [&](const auto& object) {
        object.beg_notified.rlock([&](const auto& contexts) {
            EXPECT_EQ(contexts.size(), 2);
            EXPECT_EQ(contexts.count(&first), 1);
            EXPECT_EQ(contexts.count(&second), 1);
        });
        object.end_notified.rlock([&](const auto& contexts) {
            EXPECT_EQ(contexts.size(), 1);
            EXPECT_EQ(contexts.count(&first), 1);
            EXPECT_EQ(contexts.count(&second), 0);
        });
        object.user_data.rlock([&](const auto& data) {
            EXPECT_EQ(data.at(&first).value, 11);
            EXPECT_EQ(data.at(&second).value, 22);
        });
    };

    auto constructed = TypeParam{std::move(source)};
    check_state(constructed);

    auto assigned = TypeParam{};
    assigned.end_notified.wlock([&](auto& contexts) { contexts.emplace(&second); });
    assigned = std::move(constructed);
    check_state(assigned);
}

TEST(code_object_notifications, move_preserves_unload_recipients)
{
    auto first  = rocprofiler::context::context{};
    auto second = rocprofiler::context::context{};
    auto source = hsa::code_object{};
    source.contexts.emplace_back(&first);
    source.contexts.emplace_back(&second);

    auto constructed = hsa::code_object{std::move(source)};
    ASSERT_EQ(constructed.contexts.size(), 2);
    EXPECT_EQ(constructed.contexts.at(0), &first);
    EXPECT_EQ(constructed.contexts.at(1), &second);

    auto assigned = hsa::code_object{};
    assigned      = std::move(constructed);
    ASSERT_EQ(assigned.contexts.size(), 2);
    EXPECT_EQ(assigned.contexts.at(0), &first);
    EXPECT_EQ(assigned.contexts.at(1), &second);
}
}  // namespace
