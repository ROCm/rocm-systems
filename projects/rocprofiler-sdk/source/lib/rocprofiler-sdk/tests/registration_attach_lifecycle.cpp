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

#include <gtest/gtest.h>
#include <rocprofiler-sdk/registration.h>

#include <dlfcn.h>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" rocprofiler_status_t
rocprofiler_attach();
extern "C" rocprofiler_status_t
rocprofiler_detach();

namespace
{
rocprofiler_tool_configure_result_t*
configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
{
    return nullptr;
}

struct client
{
    explicit client(const char* filename)
    {
        auto path = std::string{"$ORIGIN/../lib/"} + filename;
        handle.reset(dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL));
        if(!handle) throw std::runtime_error(dlerror());
        // The SDK keeps its own dlopen reference until finalization.
        *(void**) (&active)       = dlsym(handle.get(), "fixture_active");
        *(void**) (&attaches)     = dlsym(handle.get(), "fixture_attaches");
        *(void**) (&detaches)     = dlsym(handle.get(), "fixture_detaches");
        *(void**) (&failure)      = dlsym(handle.get(), "fixture_failure");
        *(void**) (&throw_detach) = dlsym(handle.get(), "fixture_throw_detach");
        if(!active || !attaches || !detaches || !failure || !throw_detach)
            throw std::runtime_error("fixture symbol missing");
    }

    std::unique_ptr<void, decltype(&dlclose)> handle{nullptr, dlclose};

    int (*active)()           = nullptr;
    int (*attaches)()         = nullptr;
    int (*detaches)()         = nullptr;
    void (*failure)(int)      = nullptr;
    void (*throw_detach)(int) = nullptr;
};
}  // namespace

TEST(rocprofiler_attach, multiclient_failure_and_retry)
{
    client first{ROCPROFILER_TEST_ATTACH_CLIENT_1};
    client failing{ROCPROFILER_TEST_ATTACH_CLIENT_2};
    client passive{ROCPROFILER_TEST_ATTACH_CLIENT_3};
    client later{ROCPROFILER_TEST_ATTACH_CLIENT_4};
    // Reuse normal loaded-library discovery; no tool-path environment override is needed.
    ASSERT_EQ(rocprofiler_force_configure(configure), ROCPROFILER_STATUS_SUCCESS);
    ASSERT_EQ(passive.active(), 1);

    // Test both refusal and exception after a callback has partially entered.
    for(int mode : {1, 2})
    {
        SCOPED_TRACE(mode);
        failing.failure(mode);
        ASSERT_NE(rocprofiler_attach(), ROCPROFILER_STATUS_SUCCESS);
        EXPECT_EQ(first.attaches(), mode);
        EXPECT_EQ(failing.attaches(), mode);
        EXPECT_EQ(first.detaches(), mode);
        EXPECT_EQ(failing.detaches(), mode);
        EXPECT_EQ(first.active(), 0);
        EXPECT_EQ(failing.active(), 0);
        EXPECT_EQ(passive.active(), 1);
        EXPECT_EQ(later.attaches(), 0);
        EXPECT_EQ(later.detaches(), 0);
        EXPECT_EQ(later.active(), 0);
    }

    failing.failure(0);
    ASSERT_EQ(rocprofiler_attach(), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(first.active(), 1);
    EXPECT_EQ(failing.active(), 1);
    EXPECT_EQ(later.active(), 1);

    // An already attached client must not be entered twice.
    ASSERT_EQ(rocprofiler_attach(), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(first.attaches(), 3);
    EXPECT_EQ(failing.attaches(), 3);
    EXPECT_EQ(later.attaches(), 1);

    first.throw_detach(1);
    ASSERT_NE(rocprofiler_detach(), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(first.detaches(), 3);
    EXPECT_EQ(failing.detaches(), 3);
    EXPECT_EQ(later.detaches(), 1);
    EXPECT_EQ(first.active(), 0);
    EXPECT_EQ(failing.active(), 0);
    EXPECT_EQ(later.active(), 0);
    EXPECT_EQ(passive.active(), 1);
    int finalized = -1;
    ASSERT_EQ(rocprofiler_is_finalized(&finalized), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(finalized, 0);

    // Cleanup after the exception must leave the SDK usable for a new session.
    first.throw_detach(0);
    ASSERT_EQ(rocprofiler_attach(), ROCPROFILER_STATUS_SUCCESS);
    ASSERT_EQ(rocprofiler_detach(), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(first.detaches(), 4);
    EXPECT_EQ(failing.detaches(), 4);
    EXPECT_EQ(later.detaches(), 2);
    EXPECT_EQ(first.active(), 0);
    EXPECT_EQ(failing.active(), 0);
    EXPECT_EQ(later.active(), 0);
    EXPECT_EQ(passive.active(), 1);
}
