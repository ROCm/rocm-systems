// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/progress/bar.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <sys/stat.h>

namespace rocprofsys::progress
{
namespace
{
class bar_silent_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_path = "bar_silent_test_" + std::to_string(s_counter++) + ".log";
        m_fp   = std::fopen(m_path.c_str(), "w");
        ASSERT_NE(m_fp, nullptr);
        // Regular file (not a TTY) keeps the bar silent, so no rendering
        // happens regardless of other options.
        m_opts.stream = m_fp;
        m_opts.tick   = std::chrono::milliseconds{ 0 };
    }

    void TearDown() override
    {
        if(m_fp != nullptr)
        {
            std::fclose(m_fp);
            m_fp = nullptr;
        }
        std::remove(m_path.c_str());
    }

    [[nodiscard]] std::uint64_t file_size_after_close()
    {
        std::fclose(m_fp);
        m_fp = nullptr;
        struct stat st
        {};
        if(::stat(m_path.c_str(), &st) != 0) return 0;
        return static_cast<std::uint64_t>(st.st_size);
    }

    std::string       m_path;
    std::FILE*        m_fp = nullptr;
    bar_options       m_opts;
    static inline int s_counter = 0;
};
}  // namespace

TEST_F(bar_silent_test, non_tty_stream_produces_no_output)
{
    {
        bar b{ "label", 100, m_opts };
        b.on_advance(50);
        b.on_advance(50);
    }
    EXPECT_EQ(file_size_after_close(), 0u);
}

TEST_F(bar_silent_test, negative_verbose_silences_bar)
{
    m_opts.verbose = -1;
    {
        bar b{ "label", 100, m_opts };
        b.on_advance(100);
    }
    EXPECT_EQ(file_size_after_close(), 0u);
}

TEST_F(bar_silent_test, enabled_false_silences_bar)
{
    m_opts.enabled = false;
    {
        bar b{ "label", 100, m_opts };
        b.on_advance(100);
    }
    EXPECT_EQ(file_size_after_close(), 0u);
}

TEST_F(bar_silent_test, on_advance_zero_is_noop)
{
    {
        bar b{ "label", 100, m_opts };
        b.on_advance(0);
        b.on_finish();
    }
    EXPECT_EQ(file_size_after_close(), 0u);
}

TEST_F(bar_silent_test, on_finish_is_idempotent)
{
    bar b{ "label", 10, m_opts };
    b.on_finish();
    b.on_finish();
    b.on_finish();
    SUCCEED();  // Must not assert / double-write / re-acquire mutex twice.
}

TEST_F(bar_silent_test, on_set_total_does_not_write)
{
    {
        bar b{ "label", 0, m_opts };
        b.on_set_total(500);
        b.on_advance(250);
    }
    EXPECT_EQ(file_size_after_close(), 0u);
}

}  // namespace rocprofsys::progress
