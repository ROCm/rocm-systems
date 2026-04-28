// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integration test for R-3 / AC-4: per-TID filtering in get_sampling_signals().
//
// Calls the REAL rocprofsys::get_sampling_signals(int64_t tid) from
// source/lib/core/config.cpp — no stubs.  This TU is compiled into a
// standalone binary (rocprof-sys-sampling-config-integration-tests) that links
// rocprofiler-systems-core-library directly.
//
// Initialization note: get_config() uses a magic-static initializer that calls
// configure_settings(true), which in turn calls tim::timemory_init() — this
// creates a recursive-static-init loop (recursive_init_error) when the first
// test touches a get_*() function.  The fix: call configure_settings(false)
// (skips timemory_init) from a global GTest Environment before any test runs.
// After that, _settings_are_configured() == true so get_config() never enters
// its magic-static branch again.
//
// Tested behaviour (from config.cpp:1401–1431):
//   - When ROCPROFSYS_SAMPLING_REALTIME_TIDS / _CPUTIME_TIDS are EMPTY, all
//     threads receive the corresponding signal.
//   - When the TID set is NON-EMPTY (e.g. "0"), only listed threads receive
//     the signal; other threads do not.

#include <gtest/gtest.h>

#include "core/config.hpp"

#include <csignal>
#include <set>

// ── Global environment: initialize config before any test runs ────────────────
// configure_settings(false) populates the settings map and sets
// _settings_are_configured() = true without calling tim::timemory_init(),
// which would re-enter get_config() and trigger recursive_init_error.
namespace
{
class SamplingConfigEnv : public ::testing::Environment
{
public:
    void SetUp() override { rocprofsys::configure_settings(false); }
};
}  // namespace

namespace
{

// Ensure realtime + cputime sampling are enabled and overflow is off.
// Call this before each per-TID filtering test.
void
enable_realtime_and_cputime_sampling()
{
    rocprofsys::set_setting_value("ROCPROFSYS_SAMPLING_REALTIME", true);
    rocprofsys::set_setting_value("ROCPROFSYS_SAMPLING_CPUTIME", true);
    rocprofsys::set_setting_value("ROCPROFSYS_SAMPLING_OVERFLOW", false);
}

// Clear the TID allow-lists so all threads are eligible.
void
clear_tid_filters()
{
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_REALTIME_TIDS", "");
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_CPUTIME_TIDS", "");
}

}  // namespace

// ─── R-3 / AC-4: empty TID allow-list → all threads get the signal ────────────

TEST(get_sampling_signals_integration, empty_realtime_tids_all_threads_receive_signal)
{
    enable_realtime_and_cputime_sampling();
    clear_tid_filters();

    int const rt_sig = rocprofsys::get_sampling_realtime_signal();

    // Multiple distinct TIDs — all should receive the realtime signal.
    for(int64_t tid : { INT64_C(0), INT64_C(1), INT64_C(7), INT64_C(100) })
    {
        auto sigs = rocprofsys::get_sampling_signals(tid);
        EXPECT_NE(sigs.find(rt_sig), sigs.end())
            << "realtime signal must be in set for tid=" << tid
            << " when ROCPROFSYS_SAMPLING_REALTIME_TIDS is empty (no filter)";
    }
}

TEST(get_sampling_signals_integration, empty_cputime_tids_all_threads_receive_signal)
{
    enable_realtime_and_cputime_sampling();
    clear_tid_filters();

    int const cpu_sig = rocprofsys::get_sampling_cputime_signal();

    for(int64_t tid : { INT64_C(0), INT64_C(1), INT64_C(7), INT64_C(100) })
    {
        auto sigs = rocprofsys::get_sampling_signals(tid);
        EXPECT_NE(sigs.find(cpu_sig), sigs.end())
            << "cputime signal must be in set for tid=" << tid
            << " when ROCPROFSYS_SAMPLING_CPUTIME_TIDS is empty (no filter)";
    }
}

// ─── R-3 / AC-4: non-empty TID allow-list → only listed TIDs get the signal ──

TEST(get_sampling_signals_integration, realtime_tids_filter_includes_listed_tid)
{
    enable_realtime_and_cputime_sampling();
    // Allow only tid 0 for realtime; leave cputime unrestricted.
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_REALTIME_TIDS", "0");
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_CPUTIME_TIDS", "");

    int const rt_sig = rocprofsys::get_sampling_realtime_signal();

    auto sigs = rocprofsys::get_sampling_signals(0);
    EXPECT_NE(sigs.find(rt_sig), sigs.end())
        << "realtime signal must be present for tid=0 which is in the allow-list";
}

TEST(get_sampling_signals_integration, realtime_tids_filter_excludes_unlisted_tid)
{
    enable_realtime_and_cputime_sampling();
    // Allow only tid 0 for realtime.
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_REALTIME_TIDS", "0");
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_CPUTIME_TIDS", "");

    int const rt_sig = rocprofsys::get_sampling_realtime_signal();

    // tid 5 is NOT in the allow-list → realtime signal must be absent.
    auto sigs = rocprofsys::get_sampling_signals(5);
    EXPECT_EQ(sigs.find(rt_sig), sigs.end())
        << "realtime signal must NOT be present for tid=5 which is not in the allow-list "
           "(ROCPROFSYS_SAMPLING_REALTIME_TIDS=0)";
}

TEST(get_sampling_signals_integration, cputime_tids_filter_includes_listed_tid)
{
    enable_realtime_and_cputime_sampling();
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_REALTIME_TIDS", "");
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_CPUTIME_TIDS", "0");

    int const cpu_sig = rocprofsys::get_sampling_cputime_signal();

    auto sigs = rocprofsys::get_sampling_signals(0);
    EXPECT_NE(sigs.find(cpu_sig), sigs.end())
        << "cputime signal must be present for tid=0 which is in the allow-list";
}

TEST(get_sampling_signals_integration, cputime_tids_filter_excludes_unlisted_tid)
{
    enable_realtime_and_cputime_sampling();
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_REALTIME_TIDS", "");
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_CPUTIME_TIDS", "0");

    int const cpu_sig = rocprofsys::get_sampling_cputime_signal();

    auto sigs = rocprofsys::get_sampling_signals(5);
    EXPECT_EQ(sigs.find(cpu_sig), sigs.end())
        << "cputime signal must NOT be present for tid=5 which is not in the allow-list "
           "(ROCPROFSYS_SAMPLING_CPUTIME_TIDS=0)";
}

// ─── R-3: both filters set — tid 0 gets both, tid 5 gets neither ──────────────

TEST(get_sampling_signals_integration, both_filters_set_tid0_gets_both_signals)
{
    enable_realtime_and_cputime_sampling();
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_REALTIME_TIDS", "0");
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_CPUTIME_TIDS", "0");

    int const rt_sig  = rocprofsys::get_sampling_realtime_signal();
    int const cpu_sig = rocprofsys::get_sampling_cputime_signal();

    auto sigs = rocprofsys::get_sampling_signals(0);
    EXPECT_NE(sigs.find(rt_sig), sigs.end())
        << "realtime signal must be present for tid=0 (in allow-list)";
    EXPECT_NE(sigs.find(cpu_sig), sigs.end())
        << "cputime signal must be present for tid=0 (in allow-list)";
}

TEST(get_sampling_signals_integration, both_filters_set_tid5_gets_no_signals)
{
    enable_realtime_and_cputime_sampling();
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_REALTIME_TIDS", "0");
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_CPUTIME_TIDS", "0");

    int const rt_sig  = rocprofsys::get_sampling_realtime_signal();
    int const cpu_sig = rocprofsys::get_sampling_cputime_signal();

    auto sigs = rocprofsys::get_sampling_signals(5);
    EXPECT_EQ(sigs.find(rt_sig), sigs.end())
        << "realtime signal must NOT be present for tid=5 (not in allow-list)";
    EXPECT_EQ(sigs.find(cpu_sig), sigs.end())
        << "cputime signal must NOT be present for tid=5 (not in allow-list)";
    EXPECT_TRUE(sigs.empty())
        << "signal set must be empty for tid=5 when both filters exclude it";
}

// ─── R-3: multi-tid allow-list ─────────────────────────────────────────────────

TEST(get_sampling_signals_integration, realtime_tids_multi_range_includes_listed_tids)
{
    enable_realtime_and_cputime_sampling();
    // Allow tids 0 and 2 for realtime.
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_REALTIME_TIDS",
                                               "0,2");
    rocprofsys::set_setting_value<std::string>("ROCPROFSYS_SAMPLING_CPUTIME_TIDS", "");

    int const rt_sig = rocprofsys::get_sampling_realtime_signal();

    auto sigs0 = rocprofsys::get_sampling_signals(0);
    auto sigs2 = rocprofsys::get_sampling_signals(2);
    auto sigs1 = rocprofsys::get_sampling_signals(1);

    EXPECT_NE(sigs0.find(rt_sig), sigs0.end())
        << "realtime signal must be present for tid=0 (in allow-list '0,2')";
    EXPECT_NE(sigs2.find(rt_sig), sigs2.end())
        << "realtime signal must be present for tid=2 (in allow-list '0,2')";
    EXPECT_EQ(sigs1.find(rt_sig), sigs1.end())
        << "realtime signal must NOT be present for tid=1 (not in allow-list '0,2')";
}

// Custom main: register config environment before RUN_ALL_TESTS so
// configure_settings(false) runs before any test touches get_config().
int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new SamplingConfigEnv);
    return RUN_ALL_TESTS();
}
