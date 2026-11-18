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

// The decision table behind multi-process host replay. GPU-free and MPI-free: the tool's collective
// is modelled as a bitwise AND over the processes still taking part, which is the contract
// peer_sync_cb documents.

#include "lib/rocprofiler-sdk/range_replay/peer_sync.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace peer = ::rocprofiler::range_replay::peer;

namespace
{
using peer::action_t;

constexpr uint32_t E  = peer::flag_eligible;
constexpr uint32_t C  = peer::flag_continue;
constexpr uint32_t EC = E | C;
}  // namespace

TEST(range_replay_peer_sync, flag_bits_match_the_public_header)
{
    EXPECT_EQ(peer::flag_eligible, 1u);
    EXPECT_EQ(peer::flag_continue, 2u);
}

TEST(range_replay_peer_sync, local_flags_offer_nothing_when_declined)
{
    EXPECT_EQ(peer::local_flags(false, false), 0u);
    EXPECT_EQ(peer::local_flags(false, true), 0u) << "a declined range cannot want another pass";
    EXPECT_EQ(peer::local_flags(true, false), E);
    EXPECT_EQ(peer::local_flags(true, true), EC);
}

TEST(range_replay_peer_sync, a_process_that_is_done_keeps_its_own_outcome)
{
    for(const uint32_t global : {0u, E, EC})
    {
        EXPECT_EQ(peer::resolve(0, global, 0), action_t::keep) << "declined, global " << global;
        EXPECT_EQ(peer::resolve(0, global, 3), action_t::keep);
        EXPECT_EQ(peer::resolve(E, global, 3), action_t::keep) << "finished, global " << global;
    }
}

TEST(range_replay_peer_sync, everyone_wanting_another_pass_restores)
{
    EXPECT_EQ(peer::resolve(EC, EC, 0), action_t::restore_and_continue);
    EXPECT_EQ(peer::resolve(EC, EC, 5), action_t::restore_and_continue);
}

TEST(range_replay_peer_sync, a_peer_that_declined_stops_the_others_as_declined)
{
    EXPECT_EQ(peer::resolve(EC, 0, 0), action_t::stop_declined);
    EXPECT_EQ(peer::resolve(EC, 0, 4), action_t::stop_declined);
    // global is E-less whenever any process offered 0, regardless of the CONTINUE bit
    EXPECT_EQ(peer::resolve(EC, C, 4), action_t::stop_declined);
}

TEST(range_replay_peer_sync, a_peer_that_finished_stops_the_others_as_replayed)
{
    EXPECT_EQ(peer::resolve(EC, E, 1), action_t::stop_replayed);
    EXPECT_EQ(peer::resolve(EC, E, 7), action_t::stop_replayed);
}

TEST(range_replay_peer_sync, stopping_before_any_extra_pass_is_not_a_replay)
{
    // Pass 0 is the application's own run. If the job stops there, no process re-executed anything.
    EXPECT_EQ(peer::resolve(EC, E, 0), action_t::stop_declined);
}

TEST(range_replay_peer_sync, the_job_answer_cannot_grant_bits_the_process_did_not_offer)
{
    // A reduction over the wrong communicator, or a tool returning all ones, must not turn a
    // process that offered nothing into one that continues: it keeps its outcome.
    EXPECT_EQ(peer::resolve(0, ~0u, 0), action_t::keep);
    EXPECT_EQ(peer::resolve(E, ~0u, 2), action_t::keep);
    // A process offering only ELIGIBLE|CONTINUE is judged on the bits it offered.
    EXPECT_EQ(peer::resolve(EC, ~0u, 2), action_t::restore_and_continue);
}

TEST(range_replay_peer_sync, a_failed_exchange_declines_rather_than_continuing_alone)
{
    // peer_sync_cb returns 0 when the peers cannot be reached.
    EXPECT_EQ(peer::resolve(EC, 0, 0), action_t::stop_declined);
    EXPECT_EQ(peer::resolve(EC, 0, 3), action_t::stop_declined);
}

namespace
{
struct rank_script_t
{
    // Boundary at which the range declines on this rank, or -1 for never.
    int decline_at = -1;
    // Passes the rank wants, counting the application's own run as pass 0 (pass_count_cb).
    int total_passes = 2;
};

struct rank_outcome_t
{
    int      exit_boundary = -1;
    bool     replayed      = false;
    bool     declined      = false;
    uint32_t calls         = 0;
};

// Runs the boundary protocol for ranks whose pass ends are modelled by `scripts`, exactly as
// finish_pass drives it: every rank that has not finished offers its flags once per boundary, the
// job answer is the AND of those offers, and each rank applies peer::resolve to it.
std::vector<rank_outcome_t>
run_job(const std::vector<rank_script_t>& scripts)
{
    auto outcomes = std::vector<rank_outcome_t>(scripts.size());
    auto active   = std::vector<bool>(scripts.size(), true);

    for(int boundary = 0; boundary < 64; ++boundary)
    {
        if(std::none_of(active.begin(), active.end(), [](bool v) { return v; })) break;

        auto offers = std::vector<uint32_t>(scripts.size(), 0);
        auto global = ~0u;
        for(size_t r = 0; r < scripts.size(); ++r)
        {
            if(!active[r]) continue;
            const auto& script = scripts[r];
            const bool  eligible =
                script.total_passes > 1 && (script.decline_at < 0 || boundary < script.decline_at);
            const bool wants_more = boundary + 1 < script.total_passes;
            offers[r]             = peer::local_flags(eligible, wants_more);
            global &= offers[r];
            ++outcomes[r].calls;
        }

        for(size_t r = 0; r < scripts.size(); ++r)
        {
            if(!active[r]) continue;
            const auto act = peer::resolve(offers[r], global, static_cast<uint64_t>(boundary));
            if(act == action_t::restore_and_continue) continue;

            active[r]                 = false;
            outcomes[r].exit_boundary = boundary;
            outcomes[r].declined      = (act == action_t::stop_declined) || offers[r] == 0;
            outcomes[r].replayed      = !outcomes[r].declined;
        }
    }
    return outcomes;
}
}  // namespace

// The property that matters for MPI: however the ranks' ranges end, they all leave the protocol at
// the same boundary, so none is left in a collective whose partners have gone, and none restores
// while a peer is still running its pass.
TEST(range_replay_peer_sync, every_rank_leaves_at_the_same_boundary)
{
    const std::vector<int> declines = {-1, 0, 1, 2};
    const std::vector<int> totals   = {1, 2, 3, 4};

    for(const int d0 : declines)
        for(const int d1 : declines)
            for(const int d2 : declines)
                for(const int t0 : totals)
                    for(const int t1 : totals)
                        for(const int t2 : totals)
                        {
                            const auto out = run_job({{d0, t0}, {d1, t1}, {d2, t2}});
                            const auto at  = out[0].exit_boundary;
                            ASSERT_GE(at, 0);
                            for(const auto& rank : out)
                            {
                                ASSERT_EQ(rank.exit_boundary, at) << "declines " << d0 << d1 << d2
                                                                  << " totals " << t0 << t1 << t2;
                                ASSERT_EQ(rank.calls, static_cast<uint32_t>(at + 1))
                                    << "every rank makes exactly one call per boundary it reaches";
                            }
                        }
}

TEST(range_replay_peer_sync, with_no_declines_the_job_runs_the_smallest_pass_count)
{
    const auto out = run_job({{-1, 5}, {-1, 3}, {-1, 4}});
    for(const auto& rank : out)
    {
        EXPECT_EQ(rank.exit_boundary, 2) << "stops after the smallest total (3 passes: 0, 1, 2)";
        EXPECT_TRUE(rank.replayed);
        EXPECT_FALSE(rank.declined);
    }
}

TEST(range_replay_peer_sync, one_declining_rank_declines_the_ranks_that_wanted_more)
{
    // Rank 1 declines at the first boundary (for example a snapshot that failed on its GPU).
    const auto out = run_job({{-1, 4}, {0, 4}, {-1, 4}});
    for(const auto& rank : out)
        EXPECT_EQ(rank.exit_boundary, 0) << "nobody restores or re-executes";
    EXPECT_TRUE(out[0].declined);
    EXPECT_TRUE(out[1].declined);
    EXPECT_TRUE(out[2].declined);
}

TEST(range_replay_peer_sync, a_late_decline_stops_the_job_with_the_passes_already_run)
{
    // Rank 2 declines at boundary 2; ranks 0 and 1 had run passes 0..2 by then.
    const auto out = run_job({{-1, 6}, {-1, 6}, {2, 6}});
    for(const auto& rank : out)
        EXPECT_EQ(rank.exit_boundary, 2);
    EXPECT_TRUE(out[2].declined);
}

TEST(range_replay_peer_sync, a_single_process_job_behaves_like_the_local_loop)
{
    const auto out = run_job({{-1, 3}});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].exit_boundary, 2);
    EXPECT_TRUE(out[0].replayed);
}
