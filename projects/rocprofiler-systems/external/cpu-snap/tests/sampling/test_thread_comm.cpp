#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <string>

#include <sys/syscall.h>
#include <unistd.h>

#include "cpu_snap/sampling/thread_comm.hpp"

namespace
{

namespace utils = cpu_snap::sampling::utils;

pid_t
current_tid()
{
    return static_cast<pid_t>(::syscall(SYS_gettid));
}

}  // namespace

TEST(ThreadCommTest, TryListThreadTidsFailsOnMissingDirectory)
{
    EXPECT_FALSE(utils::try_list_thread_tids("/nonexistent/cpu-snap-task-dir").has_value());
}

TEST(ThreadCommTest, TryListThreadTidsSkipsNonNumericEntries)
{
    const auto dir = std::filesystem::temp_directory_path() /
                     ("cpu_snap_task_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir / "123");
    std::filesystem::create_directories(dir / "456");
    std::filesystem::create_directories(dir / "self");

    auto tids = utils::try_list_thread_tids(dir);
    std::filesystem::remove_all(dir);

    ASSERT_TRUE(tids.has_value());
    std::ranges::sort(*tids);
    EXPECT_EQ(*tids, (std::vector<pid_t>{ 123, 456 }));
}

TEST(ThreadCommTest, TryListSelfThreadTidsContainsCurrentThread)
{
    const auto tids = utils::try_list_self_thread_tids();
    ASSERT_TRUE(tids.has_value());
    EXPECT_NE(std::ranges::find(*tids, current_tid()), tids->end());
}

TEST(ThreadCommTest, ParseStatStartTicksReadsField22)
{
    const std::string stat =
        "1234 (my (odd) name) S 1 1234 1234 0 -1 4194304 100 0 0 0 5 3 0 0 20 0 4 0 "
        "987654 1000000 200 18446744073709551615 1 1 0 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0";
    EXPECT_EQ(utils::parse_stat_start_ticks(stat), 987654u);
}

TEST(ThreadCommTest, ParseStatStartTicksRejectsMalformedInput)
{
    EXPECT_FALSE(utils::parse_stat_start_ticks("").has_value());
    EXPECT_FALSE(utils::parse_stat_start_ticks("1 (x) S 1 2 3").has_value());
}

TEST(ThreadCommTest, ReadThreadStartTicksIsStableForLiveThreadAndAbsentForDeadTid)
{
    const auto first = utils::read_thread_start_ticks(current_tid());
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(utils::read_thread_start_ticks(current_tid()), first);
    EXPECT_FALSE(utils::read_thread_start_ticks(-1).has_value());
}
