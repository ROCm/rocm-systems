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

#include <gtest/gtest.h>

#include <dlfcn.h>

TEST(rocprofiler_lib, configure_attach_uses_configure_symbol_owner)
{
    auto* handle = dlopen(ROCPROFILER_TEST_REGISTRATION_ATTACH_FIXTURE, RTLD_LAZY | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << dlerror();
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
