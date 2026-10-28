// Unit tests for cpu_snap::core::self_exe_path.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>

#include <unistd.h>

#include "cpu_snap/core/self_exe.hpp"

TEST(SelfExePathTest, ReturnsNonemptyExistingPath)
{
    const std::string path = cpu_snap::core::self_exe_path();
    ASSERT_FALSE(path.empty());
    EXPECT_TRUE(std::filesystem::exists(path));
}

TEST(SelfExePathTest, MatchesReadlink)
{
    std::array<char, 4096> buf{};
    const auto             len = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1);
    ASSERT_GT(len, 0);
    EXPECT_EQ(cpu_snap::core::self_exe_path(),
              std::string(buf.data(), static_cast<std::size_t>(len)));
}
