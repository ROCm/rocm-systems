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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/scope_destructor.hpp"
#include "lib/rocprofiler-sdk/registration/attach.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <dlfcn.h>
#include <stdexcept>

extern "C" rocprofiler_status_t
rocprofiler_attach();
extern "C" rocprofiler_status_t
rocprofiler_detach();

TEST(rocprofiler_lib, configure_attach_uses_configure_symbol_owner)
{
    auto registration_attach_fixture_name =
        fmt::format("$ORIGIN/../lib/{}", ROCPROFILER_TEST_REGISTRATION_ATTACH_FIXTURE);
    auto* handle = dlopen(registration_attach_fixture_name.c_str(), RTLD_LAZY | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << fmt::format(
        "failed to dlopen {} :: {}", registration_attach_fixture_name, dlerror());
    auto close_handle = rocprofiler::common::scope_destructor{[handle]() { dlclose(handle); }};

    auto configure         = rocprofiler_configure_func_t{};
    auto attach            = rocprofiler_configure_attach_func_t{};
    *(void**) (&configure) = dlsym(handle, "rocprofiler_configure");
    *(void**) (&attach)    = dlsym(handle, "rocprofiler_configure_attach");
    ASSERT_NE(configure, nullptr);
    ASSERT_NE(attach, nullptr);

    // Use libc as a stable foreign owner to model a mismatched global symbol lookup.
    auto* foreign_symbol = dlsym(RTLD_DEFAULT, "malloc");
    ASSERT_NE(foreign_symbol, nullptr);

    auto foreign_configure         = rocprofiler_configure_func_t{};
    auto foreign_attach            = rocprofiler_configure_attach_func_t{};
    *(void**) (&foreign_configure) = foreign_symbol;
    *(void**) (&foreign_attach)    = foreign_symbol;

    EXPECT_EQ(rocprofiler::registration::resolve_attach_for_configure(configure, attach), attach);
    EXPECT_EQ(rocprofiler::registration::resolve_attach_for_configure(configure, foreign_attach),
              attach);
    EXPECT_EQ(rocprofiler::registration::resolve_attach_for_configure(foreign_configure, attach),
              nullptr);
    EXPECT_EQ(rocprofiler::registration::resolve_attach_for_configure(nullptr, attach), nullptr);
    EXPECT_EQ(rocprofiler::registration::resolve_attach_for_configure(configure, nullptr), nullptr);
}

TEST(rocprofiler_lib, detach_with_legacy_and_attachable_clients)
{
    auto  path   = fmt::format("$ORIGIN/../lib/{}", ROCPROFILER_TEST_REGISTRATION_ATTACH_FIXTURE);
    auto* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    ASSERT_NE(handle, nullptr) << dlerror();
    auto close_handle = rocprofiler::common::scope_destructor{[handle]() { dlclose(handle); }};

    rocprofiler_configure_attach_func_t configure_attach = nullptr;
    *(void**) (&configure_attach) = dlsym(handle, "rocprofiler_configure_attach");
    ASSERT_NE(configure_attach, nullptr);
    auto* callbacks = configure_attach(0, nullptr, 0, nullptr);
    ASSERT_NE(callbacks, nullptr);
    static int detach_calls = 0;
    callbacks->tool_attach =
        [](rocprofiler_client_detach_t, rocprofiler_context_id_t*, uint64_t, void*) { return 0; };
    callbacks->tool_detach = [](void*) {
        if(++detach_calls == 1) throw std::runtime_error{"test detach failure"};
    };

    // A forced legacy client has no attach table; the fixture joins only on attach.
    auto legacy = [](uint32_t, const char*, uint32_t, rocprofiler_client_id_t*) {
        static auto result = rocprofiler_tool_configure_result_t{
            sizeof(rocprofiler_tool_configure_result_t), nullptr, nullptr, nullptr};
        return &result;
    };
    ASSERT_EQ(rocprofiler_force_configure(legacy), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_detach(), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(detach_calls, 0);

    ASSERT_EQ(rocprofiler_attach(), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_detach(), ROCPROFILER_STATUS_ERROR);
    ASSERT_EQ(rocprofiler_attach(), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_detach(), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(detach_calls, 2);
}
