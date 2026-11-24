// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "bench_hooks.h"
#include <cstdint>

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

class bench_hooks_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(pipe(m_fds), 0);
        bench_hook_reset();
    }
    void TearDown() override
    {
        unsetenv("BENCH_FD");
        close(m_fds[0]);
        close(m_fds[1]);
        bench_hook_reset();
    }
    void set_env(const std::string& v) { setenv("BENCH_FD", v.c_str(), 1); }
    int  m_fds[2] = { -1, -1 };
};

TEST_F(bench_hooks_test, no_env_is_noop)
{
    unsetenv("BENCH_FD");
    bench_hook(BENCH_HOOK_WORK_BEGIN);
    char c;
    fcntl(m_fds[0], F_SETFL, O_NONBLOCK);
    EXPECT_EQ(read(m_fds[0], &c, 1), -1);
    EXPECT_EQ(errno, EAGAIN);
}

TEST_F(bench_hooks_test, invalid_env_is_noop)
{
    for(const char* bad : { "", "abc", "-3", "12x", "999999" })
    {
        set_env(bad);
        bench_hook_reset();
        bench_hook(BENCH_HOOK_WORK_BEGIN);
    }
    EXPECT_EQ(bench_hook_dropped(), 0u);
}

TEST_F(bench_hooks_test, writes_24_byte_records_with_wall_and_cpu_time)
{
    static_assert(sizeof(bench_hook_record) == 24, "record layout");
    set_env(std::to_string(m_fds[1]));
    bench_hook(BENCH_HOOK_WORK_BEGIN);
    bench_hook(BENCH_HOOK_WORK_END);
    bench_hook_record r[2];
    ASSERT_EQ(read(m_fds[0], r, sizeof(r)), static_cast<ssize_t>(sizeof(r)));
    EXPECT_EQ(r[0].id, 1u);
    EXPECT_EQ(r[1].id, 2u);
    EXPECT_EQ(r[0].reserved, 0u);
    EXPECT_GT(r[0].ns, 0u);
    EXPECT_LE(r[0].ns, r[1].ns);
    EXPECT_GT(r[0].cpu_ns, 0u);
    EXPECT_LE(r[0].cpu_ns, r[1].cpu_ns);
}

TEST_F(bench_hooks_test, full_pipe_drops_records_without_blocking)
{
    set_env(std::to_string(m_fds[1]));
    const int total = 8000;  // > 64 KiB / 24 B
    for(int i = 0; i < total; ++i)
        bench_hook(BENCH_HOOK_WORK_BEGIN);
    EXPECT_GT(bench_hook_dropped(), 0u);
    EXPECT_LT(bench_hook_dropped(), static_cast<std::uint64_t>(total));
}
