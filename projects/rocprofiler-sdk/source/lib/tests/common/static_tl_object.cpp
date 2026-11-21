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

#include "lib/common/static_tl_object.hpp"

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <atomic>
#include <thread>
#include <vector>

namespace
{
std::atomic<int> thread_atexit_registrations = {0};
std::atomic<int> tracked_destroyed           = {0};

struct tracked
{
    std::vector<int> data = {1, 2, 3};

    ~tracked() { ++tracked_destroyed; }
};

struct first_use_tag
{};
struct rebuild_tag
{};
}  // namespace

// The compiler registers C++ thread_local destructors through this, and glibc takes dl_load_lock
// to record them. Defined here, it binds the calls from code linked into this executable (the
// common library included), so counting them shows whether first use of an object registers one.
extern "C" int
__cxa_thread_atexit(void (*func)(void*), void* obj, void* dso_symbol)
{
    using impl_t     = int (*)(void (*)(void*), void*, void*);
    static auto real = reinterpret_cast<impl_t>(dlsym(RTLD_NEXT, "__cxa_thread_atexit"));
    ++thread_atexit_registrations;
    return real(func, obj, dso_symbol);
}

// Shows the counting above sees registrations, so the tests below are not vacuous.
TEST(static_tl_object, thread_local_with_destructor_registers_on_first_use)
{
    auto before = thread_atexit_registrations.load();
    std::thread{[]() {
        static thread_local auto data = std::vector<int>{};
        data.emplace_back(1);
    }}.join();
    EXPECT_EQ(thread_atexit_registrations.load() - before, 1);
}

// The SDK first uses these objects inside HIP/HSA calls and runtime callbacks, where taking
// dl_load_lock can deadlock against a library constructor running under dlopen.
TEST(static_tl_object, instance_does_not_register_thread_local_destructor)
{
    using object = rocprofiler::common::static_tl_object<tracked, first_use_tag>;

    auto before_registrations = thread_atexit_registrations.load();
    auto before_destroyed     = tracked_destroyed.load();
    auto size_on_thread       = size_t{0};
    std::thread{[&size_on_thread]() {
        object::instance().data.emplace_back(4);
        size_on_thread = object::instance().data.size();
    }}.join();

    EXPECT_EQ(size_on_thread, 4u);
    EXPECT_EQ(thread_atexit_registrations.load() - before_registrations, 0);
    // still destroyed when the thread exits
    EXPECT_EQ(tracked_destroyed.load() - before_destroyed, 1);
}

TEST(static_tl_object, instance_rebuilt_after_destroy_is_destroyed_at_thread_exit)
{
    using object = rocprofiler::common::static_tl_object<tracked, rebuild_tag>;

    auto before_destroyed = tracked_destroyed.load();
    std::thread{[]() {
        object::instance();
        rocprofiler::common::destroy_static_tl_objects();
        object::instance();
    }}.join();

    EXPECT_EQ(tracked_destroyed.load() - before_destroyed, 2);
}
