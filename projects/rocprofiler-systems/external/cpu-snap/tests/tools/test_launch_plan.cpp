#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "launch_plan.hpp"

namespace
{

using cpu_snap::run::launch_plan;

constexpr std::uint32_t k_launcher_pid = 4242;

launch_plan
plan_for(std::initializer_list<std::string_view> args, std::string_view existing_ld_preload = {},
         std::string_view default_preload = "/lib/libsampler.so")
{
    const std::vector<std::string_view> words(args);
    return cpu_snap::run::make_launch_plan(words, k_launcher_pid, default_preload, existing_ld_preload);
}

/** Value of @p key in the plan's environment, or "<unset>". */
std::string
env(const launch_plan& plan, std::string_view key)
{
    const auto it = std::find_if(plan.environment.begin(), plan.environment.end(),
                                 [&](const auto& entry) { return entry.first == key; });
    return it == plan.environment.end() ? "<unset>" : it->second;
}

}  // namespace

TEST(LaunchPlanTest, DefaultsAndCommandAreParsed)
{
    const auto plan = plan_for({ "--", "./app", "--flag", "arg" });

    ASSERT_TRUE(plan.error.empty()) << plan.error;
    EXPECT_EQ(plan.command, (std::vector<std::string>{ "./app", "--flag", "arg" }));
    EXPECT_EQ(plan.preload, "/lib/libsampler.so");
    EXPECT_EQ(env(plan, "LD_PRELOAD"), "/lib/libsampler.so");
    EXPECT_EQ(env(plan, "CALLSTACK_FREQ_HZ"), "1000");
    EXPECT_EQ(env(plan, "CALLSTACK_BACKEND"), "auto");
    EXPECT_EQ(env(plan, "CALLSTACK_UNWIND"), "dwarf");
    EXPECT_EQ(env(plan, "CALLSTACK_STACK_BYTES"), "32768");
    EXPECT_EQ(env(plan, "CALLSTACK_DEPTH"), "256");
    EXPECT_EQ(env(plan, "CALLSTACK_TRACE"), "cpu-snap.%e.%p.pftrace");
    EXPECT_EQ(env(plan, "CALLSTACK_MERGE"), "1");
    EXPECT_EQ(env(plan, "CALLSTACK_FILTER_STDLIB"), "0");
    EXPECT_EQ(env(plan, "CALLSTACK_ONLY_PID"), "<unset>");
}

TEST(LaunchPlanTest, OptionsOverrideDefaultsInBothValueForms)
{
    const auto plan = plan_for({ "--freq=2000", "-b", "signal", "--unwind", "fp", "-o", "/tmp/out.%p",
                                 "--depth=64", "--stack-bytes", "16384", "--no-merge",
                                 "--filter-stdlib", "--", "app" });

    ASSERT_TRUE(plan.error.empty()) << plan.error;
    EXPECT_EQ(env(plan, "CALLSTACK_FREQ_HZ"), "2000");
    EXPECT_EQ(env(plan, "CALLSTACK_BACKEND"), "signal");
    EXPECT_EQ(env(plan, "CALLSTACK_UNWIND"), "fp");
    EXPECT_EQ(env(plan, "CALLSTACK_TRACE"), "/tmp/out.%p");
    EXPECT_EQ(env(plan, "CALLSTACK_DEPTH"), "64");
    EXPECT_EQ(env(plan, "CALLSTACK_STACK_BYTES"), "16384");
    EXPECT_EQ(env(plan, "CALLSTACK_MERGE"), "0");
    EXPECT_EQ(env(plan, "CALLSTACK_FILTER_STDLIB"), "1");
}

TEST(LaunchPlanTest, CommandMayStartWithoutSeparatorAndKeepsItsOwnOptions)
{
    const auto plan = plan_for({ "--freq", "500", "app", "--freq", "7" });

    ASSERT_TRUE(plan.error.empty()) << plan.error;
    EXPECT_EQ(env(plan, "CALLSTACK_FREQ_HZ"), "500");
    EXPECT_EQ(plan.command, (std::vector<std::string>{ "app", "--freq", "7" }));
}

TEST(LaunchPlanTest, ExistingLdPreloadIsKeptAfterTheSampler)
{
    const auto plan = plan_for({ "--", "app" }, "/lib/libother.so");

    EXPECT_EQ(env(plan, "LD_PRELOAD"), "/lib/libsampler.so:/lib/libother.so");
}

TEST(LaunchPlanTest, PreloadOptionReplacesTheDefaultLibrary)
{
    const auto plan = plan_for({ "--preload=/my/lib.so", "--", "app" });

    EXPECT_EQ(plan.preload, "/my/lib.so");
    EXPECT_EQ(env(plan, "LD_PRELOAD"), "/my/lib.so");
}

TEST(LaunchPlanTest, PerfDwarfDerivesRingPagesFromTheSnapshot)
{
    EXPECT_EQ(env(plan_for({ "--", "app" }), "CALLSTACK_PAGES"), "256") << "32 KiB x 32 snapshots / 4 KiB";
    EXPECT_EQ(env(plan_for({ "--stack-bytes=8192", "--", "app" }), "CALLSTACK_PAGES"), "64");
    EXPECT_EQ(env(plan_for({ "--stack-bytes=8", "--", "app" }), "CALLSTACK_PAGES"), "8") << "never below 8 pages";
    EXPECT_EQ(env(plan_for({ "--pages=32", "--", "app" }), "CALLSTACK_PAGES"), "32") << "explicit value wins";
    EXPECT_EQ(env(plan_for({ "-u", "fp", "--", "app" }), "CALLSTACK_PAGES"), "<unset>");
    EXPECT_EQ(env(plan_for({ "-b", "signal", "--", "app" }), "CALLSTACK_PAGES"), "<unset>");
    EXPECT_EQ(env(plan_for({ "-b", "perf", "--", "app" }), "CALLSTACK_PAGES"), "256");
}

TEST(LaunchPlanTest, NoChildrenRestrictsSamplingToTheLauncherPid)
{
    const auto plan = plan_for({ "--no-children", "--", "app" });

    EXPECT_EQ(env(plan, "CALLSTACK_ONLY_PID"), std::to_string(k_launcher_pid));
}

TEST(LaunchPlanTest, HelpIsRecognised)
{
    EXPECT_TRUE(plan_for({ "--help" }).help);
    EXPECT_TRUE(plan_for({ "-h" }).help);
}

TEST(LaunchPlanTest, BackendValuesAreAcceptedAndForwarded)
{
    for(const std::string_view name : { "auto", "perf", "signal" })
    {
        const auto plan = plan_for({ "--backend", name, "--", "app" });
        ASSERT_TRUE(plan.error.empty()) << plan.error;
        EXPECT_EQ(env(plan, "CALLSTACK_BACKEND"), name);
    }
}

TEST(LaunchPlanTest, InvalidArgumentsAreReported)
{
    struct bad_case
    {
        std::initializer_list<std::string_view> args;
        std::string_view                        expected_in_error;
    };
    const std::vector<bad_case> cases{
        { {}, "no command" },
        { { "--freq=100" }, "no command" },
        { { "--bogus", "--", "app" }, "unknown option" },
        { { "--freq=0", "--", "app" }, "--freq" },
        { { "--freq=abc", "--", "app" }, "--freq" },
        { { "--freq" }, "--freq" },
        { { "--backend=gpu", "--", "app" }, "--backend" },
        { { "--backend=bogus", "--", "app" }, "auto, perf or signal" },
        { { "--unwind=magic", "--", "app" }, "--unwind" },
        { { "--pages=3", "--", "app" }, "--pages" },
        { { "--stack-bytes=10", "--", "app" }, "--stack-bytes" },
        { { "--output=", "--", "app" }, "--output" },
    };
    for(const auto& c : cases)
    {
        const auto plan = plan_for(c.args);
        EXPECT_NE(plan.error.find(c.expected_in_error), std::string::npos)
            << "expected '" << c.expected_in_error << "' in: " << plan.error;
    }
}

TEST(LaunchPlanTest, MissingSamplerLibraryIsReported)
{
    const auto plan = plan_for({ "--", "app" }, {}, /*default_preload=*/"");

    EXPECT_NE(plan.error.find("--preload"), std::string::npos) << plan.error;
}
