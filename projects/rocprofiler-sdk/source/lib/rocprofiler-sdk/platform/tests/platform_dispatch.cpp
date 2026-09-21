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

// Coverage for the runtime platform-dispatch logic in agent.cpp and the
// per-platform is_available()/enumerate() entry points. Standard CI does not
// run inside WSL, so the WSL-specific assertions here only verify that
// is_available() returns false off-WSL and that enumerate() degrades to an
// empty vector without crashing - i.e. the negative path that any non-WSL
// build of librocprofiler-sdk must satisfy.

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/platform/gnulinux/agent.hpp"
#include "lib/rocprofiler-sdk/platform/wsl/agent.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

namespace fs       = ::rocprofiler::common::filesystem;
namespace common   = ::rocprofiler::common;
namespace agent    = ::rocprofiler::agent;
namespace platform = ::rocprofiler::platform;

namespace
{
// Save/restore the env vars this test mutates so a failure in one TEST does
// not leak state into the next one.
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

fs::path
local_topology_path()
{
    auto cmdline = common::read_command_line(getpid());
    if(cmdline.empty()) return {};
    auto exe = fs::path{cmdline.at(0)};
    if(!fs::exists(exe)) return {};
    auto dir = fs::canonical(exe).parent_path() / "data" / "topology" / "nodes";
    return fs::exists(dir) ? dir : fs::path{};
}
}  // namespace

// wsl::is_available() must be safe to call on any Linux host, including
// machines without /dev/dxg. CI runners are bare-metal Linux, so the
// expected answer there is false.
TEST(platform_dispatch, wsl_is_available_off_wsl_is_false)
{
    if(::access("/dev/dxg", F_OK) == 0)
    {
        GTEST_SKIP() << "/dev/dxg present - running inside WSL, skipping negative-path test";
    }
    EXPECT_FALSE(platform::wsl::is_available());
}

// wsl::enumerate() must degrade to an empty vector (not crash, not throw)
// when libdxcore.so is not loadable. Exercises the early-out branches in
// the enumerator on every CI host.
TEST(platform_dispatch, wsl_enumerate_off_wsl_is_empty)
{
    if(::access("/dev/dxg", F_OK) == 0)
    {
        GTEST_SKIP() << "/dev/dxg present - running inside WSL, skipping negative-path test";
    }
    auto agents = platform::wsl::enumerate();
    EXPECT_TRUE(agents.empty());
}

// gnulinux::is_available() must accept the test fixture's sysfs tree via
// the documented env override. This is the same fixture used by
// agent_local_topology in agent.cpp.
TEST(platform_dispatch, gnulinux_is_available_with_local_topology)
{
    auto topo = local_topology_path();
    if(topo.empty()) GTEST_SKIP() << "test data 'data/topology/nodes' not found next to binary";

    EnvGuard guard{{"ROCPROFILER_KFD_TOPOLOGY", "AMD_KFD_TOPOLOGY", "HSA_MODEL_TOPOLOGY"}};
    common::set_env("ROCPROFILER_KFD_TOPOLOGY", topo.string(), 1);
    common::set_env("AMD_KFD_TOPOLOGY", std::string{}, 1);
    common::set_env("HSA_MODEL_TOPOLOGY", std::string{}, 1);

    EXPECT_TRUE(platform::gnulinux::is_available());
}

// Verifies the dispatcher honors ROCPROFILER_FORCE_PLATFORM=gnulinux: with
// the test fixture sysfs, internal_refresh_topology() must populate agents
// from the gnulinux enumerator regardless of any wsl probing.
TEST(platform_dispatch, force_platform_gnulinux_populates_agents)
{
    auto topo = local_topology_path();
    if(topo.empty()) GTEST_SKIP() << "test data 'data/topology/nodes' not found next to binary";

    EnvGuard guard{{"ROCPROFILER_FORCE_PLATFORM",
                    "ROCPROFILER_KFD_TOPOLOGY",
                    "AMD_KFD_TOPOLOGY",
                    "HSA_MODEL_TOPOLOGY"}};
    common::set_env("ROCPROFILER_FORCE_PLATFORM", std::string{"gnulinux"}, 1);
    common::set_env("ROCPROFILER_KFD_TOPOLOGY", topo.string(), 1);
    common::set_env("AMD_KFD_TOPOLOGY", std::string{}, 1);
    common::set_env("HSA_MODEL_TOPOLOGY", std::string{}, 1);

    rocprofiler::registration::init_logging();
    agent::internal_refresh_topology();
    EXPECT_FALSE(agent::get_agents().empty());
}

// Forcing the wsl enumerator on a non-WSL host must NOT fall back to
// gnulinux - the override is explicit, and the user's request to get an
// empty topology back is a valid outcome we want to be able to observe.
TEST(platform_dispatch, force_platform_wsl_returns_empty_off_wsl)
{
    if(::access("/dev/dxg", F_OK) == 0)
    {
        GTEST_SKIP() << "/dev/dxg present - running inside WSL, skipping negative-path test";
    }

    EnvGuard guard{{"ROCPROFILER_FORCE_PLATFORM"}};
    common::set_env("ROCPROFILER_FORCE_PLATFORM", std::string{"wsl"}, 1);

    rocprofiler::registration::init_logging();
    agent::internal_refresh_topology();
    EXPECT_TRUE(agent::get_agents().empty());
}

// Unknown override values must log a warning and fall through to autodetect
// rather than aborting. With the local-topology fixture in place, autodetect
// resolves to gnulinux and produces a non-empty agent vector.
TEST(platform_dispatch, force_platform_unknown_falls_back_to_autodetect)
{
    auto topo = local_topology_path();
    if(topo.empty()) GTEST_SKIP() << "test data 'data/topology/nodes' not found next to binary";

    EnvGuard guard{{"ROCPROFILER_FORCE_PLATFORM",
                    "ROCPROFILER_KFD_TOPOLOGY",
                    "AMD_KFD_TOPOLOGY",
                    "HSA_MODEL_TOPOLOGY"}};
    common::set_env("ROCPROFILER_FORCE_PLATFORM", std::string{"not-a-platform"}, 1);
    common::set_env("ROCPROFILER_KFD_TOPOLOGY", topo.string(), 1);
    common::set_env("AMD_KFD_TOPOLOGY", std::string{}, 1);
    common::set_env("HSA_MODEL_TOPOLOGY", std::string{}, 1);

    rocprofiler::registration::init_logging();
    agent::internal_refresh_topology();
    EXPECT_FALSE(agent::get_agents().empty());
}
