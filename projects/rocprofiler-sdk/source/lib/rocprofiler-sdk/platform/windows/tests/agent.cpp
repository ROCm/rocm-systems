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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Coverage for the native-Windows D3DKMT + wkmi agent enumerator. The negative
// path must pass on any host (including GPU-less CI) by pointing the loader at a
// bogus D3DKMT module via ROCPROFILER_D3DKMT_MODULE. The positive path is gated
// on an actual GPU host and skipped otherwise.

#include "lib/rocprofiler-sdk/platform/windows/agent.hpp"
#include "lib/rocprofiler-sdk/platform/windows/wkmi_bridge.hpp"

#include "lib/common/environment.hpp"

#include <rocprofiler-sdk/agent.h>

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

namespace common   = ::rocprofiler::common;
namespace platform = ::rocprofiler::platform;
namespace windows  = ::rocprofiler::platform::windows;

namespace
{
// Save/restore env vars this test mutates so a failure in one TEST does not
// leak state into the next. Mirrors the EnvGuard in platform_dispatch.cpp.
class EnvGuard
{
public:
    explicit EnvGuard(std::initializer_list<std::string_view> names)
    {
        for(auto n : names)
            saved_.emplace_back(n, common::get_env(n, std::string{}));
    }
    ~EnvGuard()
    {
        for(const auto& [n, v] : saved_)
            common::set_env(n, v, 1);
    }

    EnvGuard(const EnvGuard&) = delete;
    EnvGuard& operator=(const EnvGuard&) = delete;

private:
    std::vector<std::pair<std::string, std::string>> saved_;
};

// True when a real GPU topology is reachable through the default D3DKMT module.
// Used to opt the positive-path test in only on actual GPU hosts.
bool
have_real_gpu()
{
    EnvGuard guard{{"ROCPROFILER_D3DKMT_MODULE"}};
    // Empty override => loader falls back to gdi32.dll.
    common::set_env("ROCPROFILER_D3DKMT_MODULE", std::string{}, 1);
    return platform::windows::is_available();
}
}  // namespace

// With the D3DKMT module forced to a name that cannot resolve the entry points,
// is_available() must be false and enumerate() must degrade to an empty vector
// without crashing. This runs on every host, GPU or not.
TEST(agent, windows_negative_path_bogus_module)
{
    EnvGuard guard{{"ROCPROFILER_D3DKMT_MODULE"}};
    common::set_env("ROCPROFILER_D3DKMT_MODULE", std::string{"rocprofiler-no-such-d3dkmt.dll"}, 1);

    EXPECT_FALSE(platform::windows::is_available());

    auto agents = platform::windows::enumerate();
    EXPECT_TRUE(agents.empty());
}

// On a GPU host, enumerate() must surface at least one GPU agent with the
// minimum sane fields populated. Skipped on GPU-less CI.
// Compute-topology assertions (gfx_target_version, cu_count, gfx name) require
// wkmi; in a stub build those fields are zero/empty and only the D3DKMT basics
// (agent type) are guaranteed.
TEST(agent, windows_positive_path_gpu_host)
{
    if(!have_real_gpu())
    {
        GTEST_SKIP() << "no D3DKMT-capable GPU detected; skipping positive-path enumeration test";
    }

    EnvGuard guard{{"ROCPROFILER_D3DKMT_MODULE"}};
    common::set_env("ROCPROFILER_D3DKMT_MODULE", std::string{}, 1);

    auto agents = platform::windows::enumerate();
    ASSERT_FALSE(agents.empty());

    const bool have_wkmi = windows::wkmi_is_present();

    for(const auto& agent : agents)
    {
        ASSERT_NE(agent.get(), nullptr);
        EXPECT_EQ(agent->type, ROCPROFILER_AGENT_TYPE_GPU);

        if(have_wkmi)
        {
            EXPECT_NE(agent->gfx_target_version, 0u);
            EXPECT_GT(agent->cu_count, 0u);
            ASSERT_NE(agent->name, nullptr);
            EXPECT_EQ(std::string_view{agent->name}.substr(0, 3), std::string_view{"gfx"});
        }
    }
}
