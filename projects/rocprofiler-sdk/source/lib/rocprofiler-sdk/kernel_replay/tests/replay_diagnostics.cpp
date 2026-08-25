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
// each policy combination is exercised directly instead of through process environment
// manipulation.

#include "lib/rocprofiler-sdk/kernel_replay/replay_diagnostics.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"

#include <gtest/gtest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
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
    EXPECT_EQ(check_untracked(memory_tracker::untracked_summary_t{}, policy), decline_reason::none);
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
    EXPECT_FALSE(memory_tracker::any_untracked(memory_tracker::untracked_summary_t{}));
    EXPECT_TRUE(memory_tracker::any_untracked(vmem_only(kMiB, 1)));
    EXPECT_TRUE(memory_tracker::any_untracked(pool_only(kMiB, 1)));
    EXPECT_FALSE(memory_tracker::any_untracked(vmem_only(kMiB, 0)))
        << "bytes alone must not count as evidence";
}

// ------------------------- footprint admission -------------------------

TEST(kernel_replay_diagnostics, footprint_within_budget_is_admitted)
{
    auto policy               = permissive_policy();
    policy.max_snapshot_bytes = 256 * kMiB;

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

// The pass count must not affect the memory decision: the snapshot is one copy of the footprint
// held for the window, not one per pass. Getting this wrong would decline large pass counts that
// are perfectly affordable.
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

// ------------------------- environment-driven policy -------------------------
//
// resolve_policy() reads the environment once per process. That is the right behavior -- a getenv
// and a /proc/meminfo open per dispatch would sit inside the replay gate for no benefit -- but it
// makes the variable-to-field mapping untestable in-process, because the first caller fixes the
// answer for every later one. So each case runs in a child process with the variable set.
//
// This is worth the machinery. These knobs are the only way a user can override a decline, and a
// typo in a variable name here is invisible: setting it appears to work and simply does nothing.

namespace
{
using env_pair_t = std::pair<const char*, const char*>;

// Re-run this executable with `env` applied, executing only the named DISABLED_ check. Returns the
// child's exit code, so a failed assertion inside the child fails the parent's test.
int
run_policy_check_in_child(const char* test_name, const std::vector<env_pair_t>& env)
{
    const auto filter = std::string{"--gtest_filter="} + test_name;

    auto pid = fork();
    if(pid == 0)
    {
        for(const auto& [name, value] : env)
            setenv(name, value, /*overwrite=*/1);

        const char* argv[] = {
            "/proc/self/exe", filter.c_str(), "--gtest_also_run_disabled_tests", nullptr};
        execv("/proc/self/exe", const_cast<char**>(argv));
        _exit(127);  // execv only returns on failure
    }

    int status = 0;
    if(pid < 0 || waitpid(pid, &status, 0) != pid) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
}  // namespace

// The checks themselves. DISABLED_ so a default run skips them: they assert on an environment only
// their parent test establishes, and would fail if run bare.
TEST(kernel_replay_policy_env, DISABLED_check_vmem_override)
{
    EXPECT_FALSE(resolve_policy().decline_on_vmem);
}

TEST(kernel_replay_policy_env, DISABLED_check_untracked_pool_override)
{
    EXPECT_TRUE(resolve_policy().decline_on_untracked_pool);
}

TEST(kernel_replay_policy_env, DISABLED_check_budget_override)
{
    EXPECT_EQ(resolve_policy().max_snapshot_bytes, 12345678U);
}

TEST(kernel_replay_policy_env, DISABLED_check_numeric_overrides)
{
    EXPECT_DOUBLE_EQ(resolve_policy().warn_seconds, 42.5);
    EXPECT_DOUBLE_EQ(resolve_policy().assumed_gbps, 7.25);
}

// A bandwidth of zero would make the projected-copy-time division produce infinity, and a negative
// one would produce a negative duration. Either would turn a diagnostic into nonsense, so the
// resolver falls back to the default rather than propagating the value.
TEST(kernel_replay_policy_env, DISABLED_check_nonpositive_bandwidth_falls_back)
{
    EXPECT_DOUBLE_EQ(resolve_policy().assumed_gbps, replay_policy_t{}.assumed_gbps);
}

// A default run derives the budget from MemAvailable, which must leave headroom rather than
// promising the whole of it: the snapshot is one full copy held for the entire window, and Linux
// overcommits, so guessing high is answered by the OOM killer instead of by snap()'s error path.
TEST(kernel_replay_policy_env, DISABLED_check_default_budget_is_bounded)
{
    const auto budget = resolve_policy().max_snapshot_bytes;

    // 0 is legitimate here: it means /proc/meminfo could not be read, which degrades to "no
    // budget" rather than to a wrong answer.
    if(budget == 0) GTEST_SKIP() << "MemAvailable unreadable; no default budget to check";

    EXPECT_LT(budget, size_t{1} << 50) << "an implausible budget suggests a unit error";
}

TEST(kernel_replay_policy_env, decline_on_vmem_can_be_turned_off)
{
    EXPECT_EQ(run_policy_check_in_child("kernel_replay_policy_env.DISABLED_check_vmem_override",
                                        {{"ROCPROFILER_KERNEL_REPLAY_DECLINE_ON_VMEM", "0"}}),
              0);
}

TEST(kernel_replay_policy_env, decline_on_untracked_pool_can_be_turned_on)
{
    EXPECT_EQ(
        run_policy_check_in_child("kernel_replay_policy_env.DISABLED_check_untracked_pool_override",
                                  {{"ROCPROFILER_KERNEL_REPLAY_DECLINE_ON_UNTRACKED_POOL", "1"}}),
        0);
}

TEST(kernel_replay_policy_env, snapshot_budget_can_be_set_explicitly)
{
    EXPECT_EQ(
        run_policy_check_in_child("kernel_replay_policy_env.DISABLED_check_budget_override",
                                  {{"ROCPROFILER_KERNEL_REPLAY_MAX_SNAPSHOT_BYTES", "12345678"}}),
        0);
}

TEST(kernel_replay_policy_env, warn_threshold_and_bandwidth_can_be_set)
{
    EXPECT_EQ(run_policy_check_in_child("kernel_replay_policy_env.DISABLED_check_numeric_overrides",
                                        {{"ROCPROFILER_KERNEL_REPLAY_WARN_SECONDS", "42.5"},
                                         {"ROCPROFILER_KERNEL_REPLAY_ASSUMED_GBPS", "7.25"}}),
              0);
}

TEST(kernel_replay_policy_env, a_nonpositive_bandwidth_is_rejected)
{
    EXPECT_EQ(run_policy_check_in_child(
                  "kernel_replay_policy_env.DISABLED_check_nonpositive_bandwidth_falls_back",
                  {{"ROCPROFILER_KERNEL_REPLAY_ASSUMED_GBPS", "0"}}),
              0);

    EXPECT_EQ(run_policy_check_in_child(
                  "kernel_replay_policy_env.DISABLED_check_nonpositive_bandwidth_falls_back",
                  {{"ROCPROFILER_KERNEL_REPLAY_ASSUMED_GBPS", "-1"}}),
              0);
}

TEST(kernel_replay_policy_env, the_default_budget_leaves_host_headroom)
{
    EXPECT_EQ(run_policy_check_in_child(
                  "kernel_replay_policy_env.DISABLED_check_default_budget_is_bounded", {}),
              0);
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
    auto outcome                   = replay_outcome_t{};
    outcome.dispatch_id            = 9;
    outcome.reason                 = decline_reason::untracked_memory;
    outcome.requested_passes       = 8;
    outcome.executed_passes        = 0;
    outcome.untracked.vmem_bytes   = 4096;
    outcome.untracked.vmem_regions = 2;
    outcome.untracked.pool_bytes   = 512;
    outcome.untracked.pool_regions = 1;

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
    auto outcome                = replay_outcome_t{};
    outcome.reason              = decline_reason::none;
    outcome.executed_passes     = 4;
    outcome.reentrancy_observed = true;

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

namespace
{
constexpr auto agent_a = rocprofiler_agent_id_t{.handle = 11};
constexpr auto agent_b = rocprofiler_agent_id_t{.handle = 22};
}  // namespace

TEST(kernel_replay_diagnostics, replay_window_marker_tracks_enter_and_exit)
{
    ASSERT_FALSE(in_replay_window(agent_a)) << "no window should be open before the first enter";

    enter_replay_window(agent_a);
    EXPECT_TRUE(in_replay_window(agent_a));
    EXPECT_FALSE(replay_reentrancy_observed(agent_a)) << "a new window starts clean";

    note_replay_reentrancy(agent_a);
    EXPECT_TRUE(replay_reentrancy_observed(agent_a));

    exit_replay_window(agent_a);
    EXPECT_FALSE(in_replay_window(agent_a));

    // A later window on the same agent starts clean, so one dispatch's reentrancy is not attributed
    // to the next.
    enter_replay_window(agent_a);
    EXPECT_FALSE(replay_reentrancy_observed(agent_a));
    exit_replay_window(agent_a);
}

// The marker is agent-scoped, and this is load-bearing rather than cosmetic. A tool callback that
// launches work on a second GPU while agent A is replaying contends for agent B's mutex, which this
// thread does not hold, so that dispatch must keep its reader lock. If the marker were agent-blind
// it would skip it, and a concurrent replay on agent B would lose the isolation the lock exists to
// provide -- silently, and only on multi-GPU runs.
TEST(kernel_replay_diagnostics, replay_window_marker_is_agent_scoped)
{
    enter_replay_window(agent_a);

    EXPECT_TRUE(in_replay_window(agent_a));
    EXPECT_FALSE(in_replay_window(agent_b))
        << "a window on one agent must not suppress locking on another agent";

    exit_replay_window(agent_a);
    EXPECT_FALSE(in_replay_window(agent_a));
}

// Windows nest across agents, and closing the inner one must not close the outer one. A callback
// running inside agent A's window may launch on agent B; a replay request there is *not* declined
// (it contends for a different mutex), so a second window opens on this same thread. If the marker
// held one value, exiting B's window would erase this thread's record of A's, and the next dispatch
// on A would take the reader lock while this thread still holds A's writer lock -- an untimed,
// unkillable block. Only reachable on multi-GPU runs, so it is asserted here rather than found
// there.
TEST(kernel_replay_diagnostics, an_inner_window_on_another_agent_does_not_close_the_outer_one)
{
    enter_replay_window(agent_a);
    ASSERT_TRUE(in_replay_window(agent_a));

    enter_replay_window(agent_b);
    EXPECT_TRUE(in_replay_window(agent_b));
    EXPECT_TRUE(in_replay_window(agent_a)) << "the enclosing window must remain visible";

    exit_replay_window(agent_b);
    EXPECT_FALSE(in_replay_window(agent_b));
    EXPECT_TRUE(in_replay_window(agent_a))
        << "closing the inner window must leave the enclosing one open; otherwise the next "
           "dispatch "
           "on the outer agent blocks forever on a lock this thread already holds";

    exit_replay_window(agent_a);
    EXPECT_FALSE(in_replay_window(agent_a));
}

// Reentrancy is attributed to the window it happened in. Agent B's window completing cleanly must
// not clear the flag that says agent A's counters are untrustworthy, and must not inherit it either
// -- each dispatch's outcome line has to describe that dispatch.
TEST(kernel_replay_diagnostics, reentrancy_is_recorded_against_the_window_it_happened_in)
{
    enter_replay_window(agent_a);
    note_replay_reentrancy(agent_a);

    enter_replay_window(agent_b);
    EXPECT_FALSE(replay_reentrancy_observed(agent_b))
        << "an inner window must not inherit the enclosing window's reentrancy";
    EXPECT_TRUE(replay_reentrancy_observed(agent_a));

    exit_replay_window(agent_b);
    EXPECT_TRUE(replay_reentrancy_observed(agent_a))
        << "an inner window closing must not clear the enclosing window's reentrancy";

    exit_replay_window(agent_a);
}

// Nesting is bounded by the agent count rather than fixed at two, so the same invariants have to
// hold at depth. Every enclosing window stays visible until it is closed, innermost first.
TEST(kernel_replay_diagnostics, windows_nest_to_the_depth_of_the_agent_count)
{
    constexpr auto agents = std::array<rocprofiler_agent_id_t, 4>{
        rocprofiler_agent_id_t{.handle = 101},
        rocprofiler_agent_id_t{.handle = 102},
        rocprofiler_agent_id_t{.handle = 103},
        rocprofiler_agent_id_t{.handle = 104},
    };

    for(size_t i = 0; i < agents.size(); ++i)
    {
        enter_replay_window(agents.at(i));
        for(size_t j = 0; j <= i; ++j)
            EXPECT_TRUE(in_replay_window(agents.at(j))) << "depth " << i << ", agent index " << j;
    }

    for(size_t i = agents.size(); i-- > 0;)
    {
        exit_replay_window(agents.at(i));
        EXPECT_FALSE(in_replay_window(agents.at(i)));
        for(size_t j = 0; j < i; ++j)
            EXPECT_TRUE(in_replay_window(agents.at(j))) << "unwinding at " << i << ", agent " << j;
    }
}

// Closing a window that was never opened must not corrupt the stack. Reachable if a guard is
// destroyed after finalization has already torn the thread's state down.
TEST(kernel_replay_diagnostics, exiting_without_a_matching_window_is_harmless)
{
    ASSERT_FALSE(in_replay_window(agent_a));

    EXPECT_NO_THROW(exit_replay_window(agent_a));
    EXPECT_FALSE(in_replay_window(agent_a));

    // ...and the marker still works afterwards.
    enter_replay_window(agent_a);
    EXPECT_TRUE(in_replay_window(agent_a));
    exit_replay_window(agent_a);
    EXPECT_FALSE(in_replay_window(agent_a));
}

// A zero agent handle is not reserved, so "not in a window" cannot be encoded as handle 0.
TEST(kernel_replay_diagnostics, replay_window_marker_handles_a_zero_agent_handle)
{
    constexpr auto agent_zero = rocprofiler_agent_id_t{.handle = 0};

    ASSERT_FALSE(in_replay_window(agent_zero));

    enter_replay_window(agent_zero);
    EXPECT_TRUE(in_replay_window(agent_zero));
    EXPECT_FALSE(in_replay_window(agent_a));

    exit_replay_window(agent_zero);
    EXPECT_FALSE(in_replay_window(agent_zero));
}

// The marker is thread-scoped: a replay window on one thread must not make an unrelated dispatch on
// another thread skip the reader lock, which would drop the isolation the lock provides.
TEST(kernel_replay_diagnostics, replay_window_marker_is_thread_scoped)
{
    enter_replay_window(agent_a);
    ASSERT_TRUE(in_replay_window(agent_a));

    bool other_thread_saw_window = true;
    auto observer = std::thread{[&]() { other_thread_saw_window = in_replay_window(agent_a); }};
    observer.join();

    EXPECT_FALSE(other_thread_saw_window)
        << "a replay window on one thread must not be visible to another; that thread's dispatches "
           "would skip the per-agent replay lock and write into the snapshot window";

    exit_replay_window(agent_a);
}

// Two threads replaying two agents concurrently is the normal multi-GPU case. Each thread's stack
// has to be its own, or one thread's exit would make the other's dispatches take a lock it holds.
TEST(kernel_replay_diagnostics, concurrent_windows_on_separate_threads_do_not_interfere)
{
    constexpr int    thread_count = 8;
    std::atomic<int> failures{0};

    auto worker = [&](uint64_t handle) {
        const auto mine  = rocprofiler_agent_id_t{.handle = handle};
        const auto other = rocprofiler_agent_id_t{.handle = handle + thread_count};

        for(int iter = 0; iter < 200; ++iter)
        {
            enter_replay_window(mine);
            if(!in_replay_window(mine)) ++failures;
            if(in_replay_window(other)) ++failures;
            note_replay_reentrancy(mine);
            if(!replay_reentrancy_observed(mine)) ++failures;
            exit_replay_window(mine);
            if(in_replay_window(mine)) ++failures;
        }
    };

    auto threads = std::vector<std::thread>{};
    threads.reserve(thread_count);
    for(int i = 0; i < thread_count; ++i)
        threads.emplace_back(worker, static_cast<uint64_t>(1000 + i));
    for(auto& thr : threads)
        thr.join();

    EXPECT_EQ(failures.load(), 0);
}

// The explanatory message is long, and a tool that launches a kernel from a PASS callback launches
// one per pass, so it is emitted once per process rather than once per occurrence.
TEST(kernel_replay_diagnostics, reentrancy_warning_is_emitted_once)
{
    EXPECT_TRUE(should_warn_replay_reentrancy()) << "the first occurrence must warn";
    EXPECT_FALSE(should_warn_replay_reentrancy());
    EXPECT_FALSE(should_warn_replay_reentrancy());
}

// ------------------------- the per-agent dispatch lock -------------------------
//
// dispatch_lock_for() is the whole of the isolation contract as a dispatch sees it: which mutex to
// take, or that it must be skipped. The failure modes are asymmetric and both silent. Handing back
// the wrong agent's mutex loses isolation on multi-GPU runs -- a replay on one GPU stops excluding
// dispatches on it. Handing back a mutex when it should have answered nullptr blocks the process in
// an untimed lock acquisition, unkillable by signal.
//
// The predicate underneath is tested above; what is asserted here is the mapping from agent to
// mutex, which is the part a dispatch actually acts on.

namespace
{
// Opens a window for the duration of a test. A failing ASSERT returns from the test body, so a bare
// enter/exit pair leaks the frame into every test that follows and turns one real failure into a
// cascade of unrelated ones.
struct scoped_window
{
    explicit scoped_window(rocprofiler_agent_id_t agent)
    : m_agent{agent}
    {
        enter_replay_window(m_agent);
    }
    ~scoped_window() { exit_replay_window(m_agent); }

    scoped_window(const scoped_window&)     = delete;
    scoped_window(scoped_window&&) noexcept = delete;
    scoped_window& operator=(const scoped_window&) = delete;
    scoped_window& operator=(scoped_window&&) noexcept = delete;

private:
    rocprofiler_agent_id_t m_agent;
};
}  // namespace

TEST(kernel_replay_dispatch_lock, an_ordinary_dispatch_gets_its_agents_lock)
{
    ASSERT_FALSE(in_replay_window(agent_a));

    auto* mutex = dispatch_lock_for(agent_a);
    ASSERT_NE(mutex, nullptr) << "a dispatch outside any window must hold the lock";
    EXPECT_EQ(mutex, &agent_replay_mutex(agent_a)) << "and it must be that agent's lock";
}

// The same agent has to name the same mutex on every call, or the window's exclusive hold and a
// dispatch's shared hold would be on different objects and neither would see the other.
TEST(kernel_replay_dispatch_lock, an_agent_always_names_the_same_lock)
{
    EXPECT_EQ(&agent_replay_mutex(agent_a), &agent_replay_mutex(agent_a));
    EXPECT_EQ(dispatch_lock_for(agent_a), &agent_replay_mutex(agent_a));
}

// One mutex per agent, not one shared by all of them. Sharing would serialize every GPU behind the
// slowest replay: correct, but it would turn an 8-GPU run into a sequential one.
TEST(kernel_replay_dispatch_lock, distinct_agents_have_distinct_locks)
{
    EXPECT_NE(&agent_replay_mutex(agent_a), &agent_replay_mutex(agent_b));
}

// The nullptr case. A tool callback launching GPU work on the replaying agent reaches here on the
// thread that already holds that agent's lock exclusively.
TEST(kernel_replay_dispatch_lock, a_dispatch_inside_its_agents_window_skips_the_lock)
{
    {
        const auto window = scoped_window{agent_a};

        EXPECT_EQ(dispatch_lock_for(agent_a), nullptr)
            << "taking the shared lock here would block forever on the exclusive lock this thread "
               "already holds";
    }

    EXPECT_NE(dispatch_lock_for(agent_a), nullptr) << "and the lock is required again afterwards";
}

// The multi-GPU case, and the reason the marker is agent-scoped. A callback running inside agent
// A's window that launches on agent B contends for B's mutex, which this thread does not hold, so
// that dispatch must still take it -- otherwise a concurrent replay on B loses its isolation.
TEST(kernel_replay_dispatch_lock, a_dispatch_on_another_agent_still_takes_its_lock)
{
    const auto window = scoped_window{agent_a};

    auto* mutex = dispatch_lock_for(agent_b);
    ASSERT_NE(mutex, nullptr) << "a window on one agent must not waive the lock on another";
    EXPECT_EQ(mutex, &agent_replay_mutex(agent_b));
}

// Skipping the lock is what makes the enclosing dispatch's counters untrustworthy, so it must be
// recorded against that window -- otherwise the outcome line reports a clean replay of a kernel
// whose inputs were mutated underneath it.
TEST(kernel_replay_dispatch_lock, skipping_the_lock_marks_the_enclosing_window)
{
    const auto window = scoped_window{agent_a};
    ASSERT_FALSE(replay_reentrancy_observed(agent_a));

    EXPECT_EQ(dispatch_lock_for(agent_a), nullptr);
    EXPECT_TRUE(replay_reentrancy_observed(agent_a));
}

// ...and the converse: a dispatch that did take its lock is not reentrancy, on either agent. A
// false positive here would report every multi-GPU replay as untrustworthy.
TEST(kernel_replay_dispatch_lock, taking_the_lock_marks_nothing)
{
    const auto window = scoped_window{agent_a};
    ASSERT_FALSE(replay_reentrancy_observed(agent_a));

    EXPECT_NE(dispatch_lock_for(agent_b), nullptr);

    EXPECT_FALSE(replay_reentrancy_observed(agent_a))
        << "work launched on a second GPU does not touch this agent's snapshot";
    EXPECT_FALSE(replay_reentrancy_observed(agent_b));
}

// The lock a dispatch is told to take is genuinely the one a window would hold, so the two really
// do exclude each other. Asserted by holding it exclusively on another thread and observing that
// the shared acquisition cannot proceed -- if agent_replay_mutex() handed out per-call objects, or
// dispatch_lock_for() named a different one, this would sail through.
TEST(kernel_replay_dispatch_lock, the_dispatch_lock_is_the_lock_a_window_holds)
{
    constexpr auto agent = rocprofiler_agent_id_t{.handle = 0xBEEF};

    auto* dispatch_mutex = dispatch_lock_for(agent);
    ASSERT_NE(dispatch_mutex, nullptr);

    std::atomic<bool> held{false};
    std::atomic<bool> release{false};
    std::atomic<bool> acquired{false};

    auto window = std::thread{[&]() {
        auto guard = std::unique_lock<std::shared_mutex>{agent_replay_mutex(agent)};
        held.store(true);
        while(!release.load())
            std::this_thread::yield();
    }};

    while(!held.load())
        std::this_thread::yield();

    auto dispatch = std::thread{[&]() {
        auto guard = std::shared_lock<std::shared_mutex>{*dispatch_mutex};
        acquired.store(true);
    }};

    // The exclusive holder is still in its critical section, so the shared acquisition must not
    // have completed. Sampled rather than waited on: this asserts the absence of an event, so it
    // can only ever be evidence, and a generous sleep here would slow the suite for no added
    // strength.
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    EXPECT_FALSE(acquired.load()) << "a dispatch acquired the lock while a window held it";

    release.store(true);
    window.join();
    dispatch.join();

    EXPECT_TRUE(acquired.load()) << "and it must proceed once the window releases";
}

// ------------------------- virtual-memory accounting -------------------------
//
// The map/unmap bookkeeping is what turns "this application uses a virtual-memory allocator" into a
// decision, and it is testable without a GPU: record_vmem_map/unmap only touch the tracker's side
// inventory. Which is worth doing, because the interception itself needs a real VMM allocation (a
// physical handle and a reserved address range) and so is only reachable on a GPU runner.
//
// Direction of error matters here. Over-counting declines a replay that would have been fine, which
// a user sees and can override. Under-counting admits a replay whose later passes silently run on
// mutated inputs, which nobody sees. Every case below is written so that a bug in the conservative
// direction fails the test.

namespace
{
// Handles that no real agent will hold, so these tests cannot collide with a live inventory.
constexpr auto vmem_agent    = hsa_agent_t{.handle = 0xF00D1};
constexpr auto other_agent   = hsa_agent_t{.handle = 0xF00D2};
constexpr auto unknown_agent = hsa_agent_t{.handle = 0};

// Distinct fake addresses; nothing dereferences them.
void*
fake_va(uintptr_t offset)
{
    return reinterpret_cast<void*>(uintptr_t{0x100000000} + offset);
}
}  // namespace

TEST(kernel_replay_vmem_accounting, a_mapping_is_counted_against_its_own_agent_only)
{
    auto* va = fake_va(0x1000);
    memory_tracker::record_vmem_map(va, 4 * kMiB, vmem_agent);

    const auto mine = memory_tracker::untracked_device_memory(vmem_agent);
    EXPECT_EQ(mine.vmem_regions, 1U);
    EXPECT_EQ(mine.vmem_bytes, 4 * kMiB);

    const auto theirs = memory_tracker::untracked_device_memory(other_agent);
    EXPECT_EQ(theirs.vmem_regions, 0U)
        << "one GPU's virtual-memory mappings must not decline replay on another GPU";

    memory_tracker::record_vmem_unmap(va, 4 * kMiB);
    EXPECT_EQ(memory_tracker::untracked_device_memory(vmem_agent).vmem_regions, 0U);
}

// hsa_amd_pointer_info does not always name an owning agent for a VMM range. Attributing such a
// mapping to no agent would hide it from every replay window, which is the one outcome that must
// not happen: the mapping is still device-visible application data outside the snapshot.
TEST(kernel_replay_vmem_accounting, a_mapping_with_no_resolved_agent_counts_everywhere)
{
    auto* va = fake_va(0x2000);
    memory_tracker::record_vmem_map(va, kMiB, unknown_agent);

    EXPECT_EQ(memory_tracker::untracked_device_memory(vmem_agent).vmem_regions, 1U);
    EXPECT_EQ(memory_tracker::untracked_device_memory(other_agent).vmem_regions, 1U);

    memory_tracker::record_vmem_unmap(va, kMiB);
    EXPECT_EQ(memory_tracker::untracked_device_memory(vmem_agent).vmem_regions, 0U);
}

// HSA permits unmapping a prefix of a larger mapping, which is how a stream-ordered allocator
// returns part of a pooled range. Dropping the whole record on any unmap would make the rest of a
// still-mapped range invisible.
TEST(kernel_replay_vmem_accounting, a_partial_unmap_shrinks_the_record_instead_of_dropping_it)
{
    auto* va = fake_va(0x3000);
    memory_tracker::record_vmem_map(va, 8 * kMiB, vmem_agent);

    memory_tracker::record_vmem_unmap(va, 2 * kMiB);

    const auto after = memory_tracker::untracked_device_memory(vmem_agent);
    EXPECT_EQ(after.vmem_regions, 1U) << "6 MiB is still mapped; the evidence must survive";
    EXPECT_EQ(after.vmem_bytes, 6 * kMiB);

    memory_tracker::record_vmem_unmap(va, 6 * kMiB);
    EXPECT_EQ(memory_tracker::untracked_device_memory(vmem_agent).vmem_regions, 0U);
}

// An unmap larger than the record (or of an address never mapped) must not underflow the byte count
// or leave a phantom region behind.
TEST(kernel_replay_vmem_accounting, an_oversized_or_unknown_unmap_is_harmless)
{
    auto* va = fake_va(0x4000);
    memory_tracker::record_vmem_map(va, kMiB, vmem_agent);
    memory_tracker::record_vmem_unmap(va, 64 * kMiB);

    const auto after = memory_tracker::untracked_device_memory(vmem_agent);
    EXPECT_EQ(after.vmem_regions, 0U);
    EXPECT_EQ(after.vmem_bytes, 0U);

    EXPECT_NO_THROW(memory_tracker::record_vmem_unmap(fake_va(0x5000), kMiB));
    EXPECT_EQ(memory_tracker::untracked_device_memory(vmem_agent).vmem_regions, 0U);
}

// A remap of the same address replaces the record rather than accumulating a second one; otherwise
// a pooled allocator that reuses one address would inflate the reported footprint without bound.
TEST(kernel_replay_vmem_accounting, remapping_the_same_address_replaces_the_record)
{
    auto* va = fake_va(0x6000);
    memory_tracker::record_vmem_map(va, kMiB, vmem_agent);
    memory_tracker::record_vmem_map(va, 4 * kMiB, vmem_agent);

    const auto after = memory_tracker::untracked_device_memory(vmem_agent);
    EXPECT_EQ(after.vmem_regions, 1U);
    EXPECT_EQ(after.vmem_bytes, 4 * kMiB);

    memory_tracker::record_vmem_unmap(va, 4 * kMiB);
}

// The end-to-end shape of the decision: a live mapping, under the default policy, declines.
TEST(kernel_replay_vmem_accounting, a_live_mapping_declines_under_the_default_policy)
{
    auto* va = fake_va(0x7000);
    memory_tracker::record_vmem_map(va, 512 * kMiB, vmem_agent);

    const auto summary = memory_tracker::untracked_device_memory(vmem_agent);
    ASSERT_TRUE(memory_tracker::any_untracked(summary));
    EXPECT_EQ(check_untracked(summary, replay_policy_t{}), decline_reason::untracked_memory);

    memory_tracker::record_vmem_unmap(va, 512 * kMiB);
    EXPECT_EQ(
        check_untracked(memory_tracker::untracked_device_memory(vmem_agent), replay_policy_t{}),
        decline_reason::none);
}

// ------------------------- restore identity gate -------------------------
//
// region_is_restorable() decides whether restore() may write a captured region back. It is the one
// predicate in kernel replay whose failure mode is corrupting the profiled application rather than
// producing a bad measurement, and it takes the inventory by reference, so every case below is
// reachable without a GPU: build the map the racing thread would have left behind and ask.
//
// The asymmetry to preserve: refusing to restore a region that was in fact still ours costs
// accuracy on later passes, and is reported. Restoring into an allocation that is *not* ours writes
// stale bytes into live application data and is reported nowhere. So every ambiguous case must
// answer false.

namespace
{
constexpr auto snap_agent = hsa_agent_t{.handle = 0xF00D3};

void*
region_addr(uintptr_t offset)
{
    return reinterpret_cast<void*>(uintptr_t{0x200000000} + offset);
}

memory_tracker::tracked_map_t
inventory_with(void* ptr, size_t size, uint64_t generation)
{
    auto map = memory_tracker::tracked_map_t{};
    map.emplace(ptr, memory_tracker::alloc_info_t{size, snap_agent, generation});
    return map;
}
}  // namespace

TEST(kernel_replay_restore_gate, the_captured_allocation_is_restorable)
{
    auto*      ptr = region_addr(0x0);
    const auto map = inventory_with(ptr, 4 * kMiB, 7);

    EXPECT_TRUE(memory_snapshot::region_is_restorable(map, ptr, 4 * kMiB, 7));
}

// The allocation was freed during the window and nothing took its place. Writing to it would touch
// retired device memory.
TEST(kernel_replay_restore_gate, a_freed_allocation_is_not_restorable)
{
    auto*      ptr = region_addr(0x1000);
    const auto map = memory_tracker::tracked_map_t{};

    EXPECT_FALSE(memory_snapshot::region_is_restorable(map, ptr, 4 * kMiB, 7));
}

// The load-bearing case. The allocation was freed and a *different* one was handed back at the same
// base address, which is the normal behaviour of a caching or pooling allocator rather than an
// exotic race. Base and size both still match, so only the generation separates them.
TEST(kernel_replay_restore_gate, an_address_reused_by_a_different_allocation_is_not_restorable)
{
    auto* ptr = region_addr(0x2000);

    // Same base, same size, later generation: a new allocation wearing the old one's address.
    const auto map = inventory_with(ptr, 4 * kMiB, 8);

    EXPECT_FALSE(memory_snapshot::region_is_restorable(map, ptr, 4 * kMiB, 7))
        << "restoring here would write a dead buffer's bytes over live application data";
}

// A generation is monotonic, never reissued, so a *lower* generation at the same address is equally
// not the region that was captured. Asserted separately because an ordering comparison would pass
// the reuse test above and fail this one.
TEST(kernel_replay_restore_gate, a_lower_generation_at_the_same_address_is_not_restorable)
{
    auto*      ptr = region_addr(0x2800);
    const auto map = inventory_with(ptr, 4 * kMiB, 6);

    EXPECT_FALSE(memory_snapshot::region_is_restorable(map, ptr, 4 * kMiB, 7));
}

// Reallocated smaller at the same base. The snapshot holds more bytes than the allocation now has,
// so copying it back would run past the end.
TEST(kernel_replay_restore_gate, a_shrunk_allocation_is_not_restorable)
{
    auto*      ptr = region_addr(0x3000);
    const auto map = inventory_with(ptr, kMiB, 7);

    EXPECT_FALSE(memory_snapshot::region_is_restorable(map, ptr, 4 * kMiB, 7));
}

// Grown at the same base. The size check alone would admit this -- the recorded bytes do fit -- so
// the generation is what rejects it. A grown allocation is a new allocation.
TEST(kernel_replay_restore_gate, an_allocation_grown_at_the_same_base_is_not_restorable)
{
    auto*      ptr = region_addr(0x4000);
    const auto map = inventory_with(ptr, 16 * kMiB, 9);

    EXPECT_FALSE(memory_snapshot::region_is_restorable(map, ptr, 4 * kMiB, 7))
        << "the recorded bytes would fit, so only the generation stamp rejects this";
}

// Generation 0 means "any live allocation of at least this size", which is how module-scope
// variables are handled: they live in the loaded executable and the tracker never stamps them.
TEST(kernel_replay_restore_gate, generation_zero_accepts_any_live_allocation_of_that_size)
{
    auto*      ptr = region_addr(0x5000);
    const auto map = inventory_with(ptr, 4 * kMiB, 123);

    EXPECT_TRUE(memory_snapshot::region_is_restorable(map, ptr, 4 * kMiB, 0));
    EXPECT_TRUE(memory_snapshot::region_is_restorable(map, ptr, kMiB, 0))
        << "a prefix of a larger live allocation is still within it";
    EXPECT_FALSE(memory_snapshot::region_is_restorable(map, ptr, 8 * kMiB, 0))
        << "the size check still applies without a generation";
}

// A tracked allocation is never stamped 0, so a tracked region cannot silently fall through to the
// weaker generation-0 check. This is what keeps the module-variable escape hatch from becoming a
// hole in the ABA gate.
TEST(kernel_replay_restore_gate, a_tracked_allocation_is_never_stamped_zero)
{
    auto*      ptr = region_addr(0x6000);
    const auto map = inventory_with(ptr, 4 * kMiB, 0);

    // If a generation of 0 ever reached the map, this is the check that would misfire: a region
    // captured at generation 7 would match an unstamped record.
    EXPECT_FALSE(memory_snapshot::region_is_restorable(map, ptr, 4 * kMiB, 7));
}

// Interior pointers are not restorable: the inventory is keyed on the allocation base, and an
// address partway into a region names no record. restore() only ever asks about bases it captured,
// so this asserts the lookup is exact rather than nearest.
TEST(kernel_replay_restore_gate, an_interior_pointer_names_no_region)
{
    auto* base = region_addr(0x7000);
    auto* mid  = static_cast<void*>(static_cast<char*>(base) + kMiB);

    const auto map = inventory_with(base, 4 * kMiB, 7);

    EXPECT_TRUE(memory_snapshot::region_is_restorable(map, base, 4 * kMiB, 7));
    EXPECT_FALSE(memory_snapshot::region_is_restorable(map, mid, kMiB, 7));
}

// A zero-length request is admitted for a live matching region (snap() drops empty regions before
// this point, so the answer only has to be harmless, not meaningful) but still rejected once the
// allocation is gone.
TEST(kernel_replay_restore_gate, a_zero_length_region_follows_the_liveness_of_its_base)
{
    auto* ptr = region_addr(0x8000);

    EXPECT_TRUE(memory_snapshot::region_is_restorable(inventory_with(ptr, 4 * kMiB, 7), ptr, 0, 7));
    EXPECT_FALSE(memory_snapshot::region_is_restorable(memory_tracker::tracked_map_t{}, ptr, 0, 7));
}

// Every distinct way an address can stop naming the captured allocation, in one place, so a future
// change to the predicate has to keep answering false for all of them.
TEST(kernel_replay_restore_gate, every_ambiguous_case_answers_false)
{
    auto*          ptr           = region_addr(0x9000);
    constexpr auto captured_size = 4 * kMiB;
    constexpr auto captured_gen  = uint64_t{42};

    struct case_t
    {
        const char*                   what;
        memory_tracker::tracked_map_t map;
    };

    auto cases = std::vector<case_t>{};
    cases.push_back({"freed", memory_tracker::tracked_map_t{}});
    cases.push_back({"reused at the same size", inventory_with(ptr, captured_size, 43)});
    cases.push_back({"reused at a smaller size", inventory_with(ptr, kMiB, 43)});
    cases.push_back({"reused at a larger size", inventory_with(ptr, 16 * kMiB, 43)});
    cases.push_back({"shrunk in place", inventory_with(ptr, kMiB, captured_gen)});
    cases.push_back({"unstamped", inventory_with(ptr, captured_size, 0)});
    cases.push_back({"a different address entirely",
                     inventory_with(region_addr(0xA000), captured_size, captured_gen)});

    for(const auto& c : cases)
        EXPECT_FALSE(memory_snapshot::region_is_restorable(c.map, ptr, captured_size, captured_gen))
            << c.what;

    // ...and the one case that must answer true, so the test cannot pass by always refusing.
    EXPECT_TRUE(memory_snapshot::region_is_restorable(
        inventory_with(ptr, captured_size, captured_gen), ptr, captured_size, captured_gen));
}

// ------------------------- module-variable liveness gate -------------------------
//
// A __device__ / __constant__ global is not in the tracker, so the gate above says nothing about
// it. Its address is valid exactly as long as the executable holding its data segment is loaded,
// and the replay window serializes dispatches on an agent rather than code-object loading -- so a
// host thread is free to dlclose a module (or a framework to unload a JIT'd kernel) between snap
// and restore. module_region_is_restorable() is the check that stops us writing into a segment the
// loader has already released.
//
// The polarity here is the opposite of the ABA gate's. An address that no longer belongs to us is
// still the dangerous case, but an *empty* set of loaded executables means enumeration failed, not
// that nothing is loaded: the code-object module returns nothing once it has shut down. Refusing
// every module variable then would stop reverting globals between passes for no safety gain, so an
// empty set is admitted.

namespace
{
std::unordered_set<uint64_t>
loaded(std::initializer_list<uint64_t> handles)
{
    return std::unordered_set<uint64_t>{handles};
}

hsa_executable_t
exec(uint64_t handle)
{
    return hsa_executable_t{.handle = handle};
}
}  // namespace

TEST(kernel_replay_module_gate, a_variable_in_a_loaded_executable_is_restorable)
{
    EXPECT_TRUE(memory_snapshot::module_region_is_restorable(loaded({11, 22, 33}), exec(22)));
}

// The case the check exists for: the snapshot names an executable that is no longer loaded.
TEST(kernel_replay_module_gate, a_variable_whose_executable_was_unloaded_is_not_restorable)
{
    EXPECT_FALSE(memory_snapshot::module_region_is_restorable(loaded({11, 33}), exec(22)))
        << "restoring here would write into a data segment the loader has released";
}

// Everything unloaded but the enumeration still succeeded is reported as a non-empty set in
// practice; the degenerate "one executable left, not ours" spelling is the same refusal.
TEST(kernel_replay_module_gate, only_the_owning_handle_admits_the_variable)
{
    EXPECT_FALSE(memory_snapshot::module_region_is_restorable(loaded({1}), exec(2)));
    EXPECT_TRUE(memory_snapshot::module_region_is_restorable(loaded({1}), exec(1)));
}

// An empty set means "could not enumerate", so it admits. Asserted on its own because it is the one
// place this predicate deliberately answers true for an unverifiable region, and a future change
// that "hardens" it to false would silently stop reverting globals between passes.
TEST(kernel_replay_module_gate, an_unenumerable_executable_set_admits_rather_than_refuses)
{
    EXPECT_TRUE(
        memory_snapshot::module_region_is_restorable(std::unordered_set<uint64_t>{}, exec(22)))
        << "an empty set is a failed enumeration, not an empty process";
}

// A zero handle is what a mem_block_t carries when it came from the tracker rather than an
// executable. restore() never asks about those -- it branches on from_tracker first -- but the
// predicate must not accidentally treat 0 as a wildcard that matches any loaded set.
TEST(kernel_replay_module_gate, a_zero_handle_is_not_a_wildcard)
{
    EXPECT_FALSE(memory_snapshot::module_region_is_restorable(loaded({11, 22}), exec(0)));
    EXPECT_TRUE(memory_snapshot::module_region_is_restorable(loaded({0, 11}), exec(0)))
        << "0 is matched by handle like any other, not specially";
}

// Handles are not reissued for a different executable within a process lifetime, but the predicate
// makes no such assumption: it compares handles, so a reload that happened to land on the same
// handle is admitted and one that did not is refused. This documents that the module gate is
// coarser than the ABA gate on purpose -- there is no generation stamp for executables.
TEST(kernel_replay_module_gate, the_check_is_handle_identity_and_nothing_more)
{
    EXPECT_TRUE(memory_snapshot::module_region_is_restorable(loaded({22}), exec(22)));
    EXPECT_FALSE(memory_snapshot::module_region_is_restorable(loaded({23}), exec(22)));
}
