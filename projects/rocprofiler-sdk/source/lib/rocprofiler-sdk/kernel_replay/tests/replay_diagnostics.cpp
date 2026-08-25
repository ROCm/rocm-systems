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

// Unit tests for the replay window's admission-control decisions and its per-dispatch reporting.
//
// These need no GPU, no HSA runtime and no rocprofiler configuration, which matters: on a CPU-only
// CI runner every other kernel-replay test skips, so without these the decision logic that decides
// whether a replay is sound would have no automated coverage at all.
//
// The decision functions take an explicit replay_policy_t rather than reading the environment, so
// each policy combination is exercised directly instead of through process environment manipulation.

#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/replay_diagnostics.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

using namespace rocprofiler;
using namespace rocprofiler::kernel_replay;

namespace
{
constexpr size_t kMiB = 1024U * 1024U;

replay_policy_t
permissive_policy()
{
    auto policy                      = replay_policy_t{};
    policy.decline_on_vmem           = false;
    policy.decline_on_untracked_pool = false;
    policy.max_snapshot_bytes        = 0;  // unlimited
    policy.warn_seconds              = 0.0;
    policy.assumed_gbps              = 4.0;
    return policy;
}

memory_tracker::untracked_summary_t
vmem_only(size_t bytes, size_t regions)
{
    auto summary         = memory_tracker::untracked_summary_t{};
    summary.vmem_bytes   = bytes;
    summary.vmem_regions = regions;
    return summary;
}

memory_tracker::untracked_summary_t
pool_only(size_t bytes, size_t regions)
{
    auto summary         = memory_tracker::untracked_summary_t{};
    summary.pool_bytes   = bytes;
    summary.pool_regions = regions;
    return summary;
}

memory_snapshot::snapshot_footprint_t
footprint(size_t bytes, size_t regions)
{
    auto out    = memory_snapshot::snapshot_footprint_t{};
    out.bytes   = bytes;
    out.regions = regions;
    return out;
}
}  // namespace

// ------------------------- untracked-memory policy -------------------------

TEST(kernel_replay_diagnostics, no_untracked_memory_admits_replay)
{
    auto policy = replay_policy_t{};
    EXPECT_EQ(check_untracked(memory_tracker::untracked_summary_t{}, policy),
              decline_reason::none);
}

// A live virtual-memory mapping is unambiguous evidence that application data is outside the
// snapshot: nothing maps virtual memory unless the application asked for it. Declining is the
// default because the alternative is reporting counters derived from inputs that changed between
// passes, with nothing in the output to say so.
TEST(kernel_replay_diagnostics, live_vmem_mapping_declines_by_default)
{
    auto policy = replay_policy_t{};
    ASSERT_TRUE(policy.decline_on_vmem) << "declining on virtual memory must be the default";

    EXPECT_EQ(check_untracked(vmem_only(64 * kMiB, 1), policy), decline_reason::untracked_memory);
}

// The override exists so a user who understands the consequence can still collect numbers.
TEST(kernel_replay_diagnostics, live_vmem_mapping_admits_when_override_set)
{
    auto policy            = replay_policy_t{};
    policy.decline_on_vmem = false;

    EXPECT_EQ(check_untracked(vmem_only(64 * kMiB, 1), policy), decline_reason::none);
}

// Weaker evidence, so the default is the other way round: runtime-internal allocations land in this
// bucket too, and declining on them would disable replay for ordinary applications.
TEST(kernel_replay_diagnostics, untracked_pool_memory_admits_by_default)
{
    auto policy = replay_policy_t{};
    ASSERT_FALSE(policy.decline_on_untracked_pool)
        << "declining on non-coarse GPU-resident pool memory must be opt-in";

    EXPECT_EQ(check_untracked(pool_only(8 * kMiB, 3), policy), decline_reason::none);
}

TEST(kernel_replay_diagnostics, untracked_pool_memory_declines_under_strict_policy)
{
    auto policy                      = replay_policy_t{};
    policy.decline_on_untracked_pool = true;

    EXPECT_EQ(check_untracked(pool_only(8 * kMiB, 3), policy), decline_reason::untracked_memory);
}

// Byte counts of zero with a non-zero region count must still decline: a mapping of unknown size is
// no more snapshottable than a large one, and the region count is the reliable signal.
TEST(kernel_replay_diagnostics, zero_byte_vmem_region_still_declines)
{
    auto policy = replay_policy_t{};
    EXPECT_EQ(check_untracked(vmem_only(0, 1), policy), decline_reason::untracked_memory);
}

// The converse: bytes without regions cannot happen from the tracker, but must not decline if it
// does, or a bookkeeping slip would silently disable replay everywhere.
TEST(kernel_replay_diagnostics, bytes_without_regions_does_not_decline)
{
    auto policy = replay_policy_t{};
    EXPECT_EQ(check_untracked(vmem_only(kMiB, 0), policy), decline_reason::none);
}

TEST(kernel_replay_diagnostics, untracked_summary_any_reflects_both_classes)
{
    EXPECT_FALSE(memory_tracker::untracked_summary_t{}.any());
    EXPECT_TRUE(vmem_only(kMiB, 1).any());
    EXPECT_TRUE(pool_only(kMiB, 1).any());
    EXPECT_FALSE(vmem_only(kMiB, 0).any()) << "bytes alone must not count as evidence";
}

// ------------------------- footprint admission -------------------------

TEST(kernel_replay_diagnostics, footprint_within_budget_is_admitted)
{
    auto policy                = permissive_policy();
    policy.max_snapshot_bytes  = 256 * kMiB;

    EXPECT_EQ(check_admission(footprint(64 * kMiB, 4), 4, policy), decline_reason::none);
}

TEST(kernel_replay_diagnostics, footprint_over_budget_is_declined)
{
    auto policy               = permissive_policy();
    policy.max_snapshot_bytes = 256 * kMiB;

    EXPECT_EQ(check_admission(footprint(512 * kMiB, 4), 4, policy),
              decline_reason::footprint_budget);
}

// Exactly at the budget is admitted: the check is "over", not "at or over", so a budget set to an
// exact known footprint does what the user meant.
TEST(kernel_replay_diagnostics, footprint_exactly_at_budget_is_admitted)
{
    auto policy               = permissive_policy();
    policy.max_snapshot_bytes = 256 * kMiB;

    EXPECT_EQ(check_admission(footprint(256 * kMiB, 1), 2, policy), decline_reason::none);
}

// A zero budget means unlimited, which is what an explicit
// ROCPROFILER_KERNEL_REPLAY_MAX_SNAPSHOT_BYTES=0 must mean -- otherwise setting it to zero would
// disable replay entirely rather than removing the limit.
TEST(kernel_replay_diagnostics, zero_budget_means_unlimited)
{
    auto policy               = permissive_policy();
    policy.max_snapshot_bytes = 0;

    EXPECT_EQ(check_admission(footprint(64ULL * 1024 * kMiB, 1000), 100, policy),
              decline_reason::none);
}

// The pass count must not affect the memory decision: the snapshot is one copy of the footprint held
// for the window, not one per pass. Getting this wrong would decline large pass counts that are
// perfectly affordable.
TEST(kernel_replay_diagnostics, pass_count_does_not_affect_the_memory_budget)
{
    auto policy               = permissive_policy();
    policy.max_snapshot_bytes = 256 * kMiB;

    for(uint64_t passes : {uint64_t{0}, uint64_t{1}, uint64_t{2}, uint64_t{64}, uint64_t{1000}})
        EXPECT_EQ(check_admission(footprint(64 * kMiB, 4), passes, policy), decline_reason::none)
            << "passes=" << passes;
}

// An indefinite loop (pass count 0) has no projected total, so it is treated as the minimum that
// restores at all. It must not be treated as zero work, which would suppress the cost warning for
// precisely the unbounded case that most needs it.
TEST(kernel_replay_diagnostics, indefinite_pass_count_is_admitted_and_projected)
{
    auto policy               = permissive_policy();
    policy.max_snapshot_bytes = 0;
    policy.warn_seconds       = 1.0;

    // 8 GB over an assumed 4 GB/s is well past the warning threshold for any pass count, including
    // the indefinite one; the call must still admit rather than decline, since a warning is not a
    // decline.
    EXPECT_EQ(check_admission(footprint(8ULL * 1024 * kMiB, 1), 0, policy), decline_reason::none);
}

// The projected-time warning must not be able to turn into a decline: cost is the user's call, and
// silently refusing to profile an expensive kernel would be worse than doing what was asked.
TEST(kernel_replay_diagnostics, projected_time_warning_never_declines)
{
    auto policy               = permissive_policy();
    policy.max_snapshot_bytes = 0;
    policy.warn_seconds       = 0.000001;

    EXPECT_EQ(check_admission(footprint(kMiB, 1), 2, policy), decline_reason::none);
}

TEST(kernel_replay_diagnostics, empty_footprint_is_admitted)
{
    auto policy = replay_policy_t{};
    EXPECT_EQ(check_admission(footprint(0, 0), 8, policy), decline_reason::none);
}

// ------------------------- resolved policy -------------------------

// resolve_policy() caches its result, so this asserts only invariants that must hold for any
// environment. The per-combination behaviour is covered by the tests above, which take the policy
// explicitly.
TEST(kernel_replay_diagnostics, resolved_policy_is_usable)
{
    const auto& policy = resolve_policy();

    EXPECT_GT(policy.assumed_gbps, 0.0)
        << "a non-positive bandwidth would divide by zero when projecting copy time";
    EXPECT_GE(policy.warn_seconds, 0.0);

    // The same reference every time: a per-dispatch re-read would put a /proc/meminfo open inside
    // the replay gate.
    EXPECT_EQ(&policy, &resolve_policy());
}

// ------------------------- outcome reporting -------------------------

TEST(kernel_replay_diagnostics, every_decline_reason_has_a_distinct_name)
{
    const auto reasons = {decline_reason::none,
                          decline_reason::untracked_memory,
                          decline_reason::footprint_budget,
                          decline_reason::snapshot_failed,
                          decline_reason::queue_drain_stuck,
                          decline_reason::agent_drain_stuck,
                          decline_reason::pass_drain_stuck,
                          decline_reason::reentrant_dispatch};

    auto seen = std::vector<std::string>{};
    for(auto reason : reasons)
    {
        const auto name = std::string{to_string(reason)};
        EXPECT_FALSE(name.empty());
        EXPECT_NE(name, "unknown") << "a decline reason is missing from to_string()";
        for(const auto& prior : seen)
            EXPECT_NE(name, prior) << "duplicate decline reason name: " << name;
        seen.push_back(name);
    }
}

// The line is what a user or a test greps to tell a clean replay from one that ran on mutated
// inputs, so the fields have to be there and have to carry the right values.
TEST(kernel_replay_diagnostics, outcome_line_reports_a_successful_replay)
{
    auto outcome              = replay_outcome_t{};
    outcome.dispatch_id       = 42;
    outcome.reason            = decline_reason::none;
    outcome.requested_passes  = 4;
    outcome.executed_passes   = 4;
    outcome.footprint_bytes   = 1234;
    outcome.footprint_regions = 7;

    const auto line = format_replay_outcome(outcome);

    EXPECT_NE(line.find("dispatch_id=42"), std::string::npos) << line;
    EXPECT_NE(line.find("outcome=replayed"), std::string::npos) << line;
    EXPECT_NE(line.find("reason=none"), std::string::npos) << line;
    EXPECT_NE(line.find("requested_passes=4"), std::string::npos) << line;
    EXPECT_NE(line.find("executed_passes=4"), std::string::npos) << line;
    EXPECT_NE(line.find("footprint_bytes=1234"), std::string::npos) << line;
    EXPECT_NE(line.find("footprint_regions=7"), std::string::npos) << line;
    EXPECT_NE(line.find("reentrancy=0"), std::string::npos) << line;
}

TEST(kernel_replay_diagnostics, outcome_line_reports_a_decline_with_its_cause)
{
    auto outcome                    = replay_outcome_t{};
    outcome.dispatch_id             = 9;
    outcome.reason                  = decline_reason::untracked_memory;
    outcome.requested_passes        = 8;
    outcome.executed_passes         = 0;
    outcome.untracked.vmem_bytes    = 4096;
    outcome.untracked.vmem_regions  = 2;
    outcome.untracked.pool_bytes    = 512;
    outcome.untracked.pool_regions  = 1;

    const auto line = format_replay_outcome(outcome);

    EXPECT_NE(line.find("outcome=declined"), std::string::npos) << line;
    EXPECT_NE(line.find("reason=untracked_memory"), std::string::npos) << line;
    EXPECT_NE(line.find("executed_passes=0"), std::string::npos) << line;
    EXPECT_NE(line.find("untracked_vmem_bytes=4096"), std::string::npos) << line;
    EXPECT_NE(line.find("untracked_vmem_regions=2"), std::string::npos) << line;
    EXPECT_NE(line.find("untracked_pool_bytes=512"), std::string::npos) << line;
    EXPECT_NE(line.find("untracked_pool_regions=1"), std::string::npos) << line;
}

// A replay that completed all its passes but had a dispatch injected into its window is not a
// decline, and it is also not trustworthy. The line has to distinguish that third state.
TEST(kernel_replay_diagnostics, outcome_line_flags_reentrancy_on_an_otherwise_clean_replay)
{
    auto outcome                 = replay_outcome_t{};
    outcome.reason               = decline_reason::none;
    outcome.executed_passes      = 4;
    outcome.reentrancy_observed  = true;

    const auto line = format_replay_outcome(outcome);

    EXPECT_NE(line.find("outcome=replayed"), std::string::npos) << line;
    EXPECT_NE(line.find("reentrancy=1"), std::string::npos) << line;
}

TEST(kernel_replay_diagnostics, logging_an_outcome_does_not_throw)
{
    auto outcome        = replay_outcome_t{};
    outcome.dispatch_id = 1;
    EXPECT_NO_THROW(log_replay_outcome(outcome));

    outcome.reason = decline_reason::queue_drain_stuck;
    EXPECT_NO_THROW(log_replay_outcome(outcome));
}

// ------------------------- reentrancy marker -------------------------

TEST(kernel_replay_diagnostics, replay_window_marker_tracks_enter_and_exit)
{
    ASSERT_FALSE(in_replay_window()) << "no window should be open before the first enter";

    enter_replay_window();
    EXPECT_TRUE(in_replay_window());
    EXPECT_FALSE(replay_reentrancy_observed()) << "entering must clear the previous window's flag";

    note_replay_reentrancy();
    EXPECT_TRUE(replay_reentrancy_observed());

    exit_replay_window();
    EXPECT_FALSE(in_replay_window());

    // The flag survives the exit so the outcome record, written after the window closes, can still
    // report it.
    EXPECT_TRUE(replay_reentrancy_observed());

    // ...and a new window starts clean, so one dispatch's reentrancy is not attributed to the next.
    enter_replay_window();
    EXPECT_FALSE(replay_reentrancy_observed());
    exit_replay_window();
}

// The marker is thread-scoped: a replay window on one thread must not make an unrelated dispatch on
// another thread skip the reader lock, which would drop the isolation the lock provides.
TEST(kernel_replay_diagnostics, replay_window_marker_is_thread_scoped)
{
    enter_replay_window();
    ASSERT_TRUE(in_replay_window());

    bool other_thread_saw_window = true;
    auto observer                = std::thread{[&]() { other_thread_saw_window = in_replay_window(); }};
    observer.join();

    EXPECT_FALSE(other_thread_saw_window)
        << "a replay window on one thread must not be visible to another; that thread's dispatches "
           "would skip the per-agent replay lock and write into the snapshot window";

    exit_replay_window();
}

// The explanatory message is long, and a tool that launches a kernel from a PASS callback launches
// one per pass, so it is emitted once per process rather than once per occurrence.
TEST(kernel_replay_diagnostics, reentrancy_warning_is_emitted_once)
{
    EXPECT_TRUE(should_warn_replay_reentrancy()) << "the first occurrence must warn";
    EXPECT_FALSE(should_warn_replay_reentrancy());
    EXPECT_FALSE(should_warn_replay_reentrancy());
}
