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

// Unit tests for the dispatch-log reader poll-scheduling helpers: poll timeout,
// poll-set membership, terminal-revents classification, and spin backstop.
#include "lib/rocprofiler-sdk/kfd/poll_reader.hpp"

#include <gtest/gtest.h>
#include <poll.h>
#include <climits>
#include <string>
#include <vector>

namespace
{
using namespace rocprofiler::kfd;
// A fully-ready, mapped, non-quarantined session with an fd.
poll_session_view
ready_view()
{
    return poll_session_view{true, true, false, true};
}
std::vector<size_t>
poll_slots(const std::vector<poll_session_view>& s)
{
    std::vector<size_t> slots;
    build_poll_slots(s.data(), s.size(), slots);
    return slots;
}
}  // namespace

// Only a plain decimal in [1, INT_MAX] parses; everything else returns 0.
TEST(poll_timeout, parse)
{
    struct tc
    {
        const char* in;
        int         want;
        const char* label;
    };
    const tc rows[] = {{"1", 1, "one"},
                       {"10", 10, "ten"},
                       {"100", 100, "hundred"},
                       {"2147483647", INT_MAX, "INT_MAX digits"},
                       {"", 0, "empty"},
                       {"0", 0, "zero"},
                       {"00", 0, "00"},
                       {"-1", 0, "neg 1"},
                       {"-10", 0, "neg 10"},
                       {"abc", 0, "non-numeric"},
                       {" 10", 0, "leading space"},
                       {"10 ", 0, "trailing space"},
                       {"10ms", 0, "trailing junk"},
                       {"+10", 0, "plus sign"},
                       {"0x10", 0, "hex"},
                       {"2147483648", 0, "INT_MAX+1"},
                       {"4294967296", 0, "UINT32_MAX+1"},
                       {"18446744073709551615", 0, "UINT64_MAX"},
                       {"18446744073709551616", 0, "UINT64_MAX+1"}};
    for(const auto& tc : rows)
        EXPECT_EQ(poll_timeout_ms_from_str(tc.in), tc.want) << tc.label;
    EXPECT_EQ(poll_timeout_ms_from_str(std::to_string(INT_MAX)), INT_MAX) << "INT_MAX to_string";
    EXPECT_EQ(poll_timeout_ms_from_str(std::string(64, '9')), 0) << "64 nines";
}
// One entry per live, mapped, non-quarantined session with an fd; keeps orig index.
TEST(poll_set, membership)
{
    auto not_ready = ready_view(), not_mapped = ready_view(), no_fd = ready_view();
    not_ready.ready = false, not_mapped.mapped = false, no_fd.has_fd = false;
    struct tc
    {
        std::vector<poll_session_view> sessions;
        std::vector<size_t>            want;
        const char*                    label;
    };
    const tc rows[] = {
        {{}, {}, "zero sessions"},
        {{ready_view()}, {0u}, "one ready"},
        {{ready_view(), ready_view(), ready_view()}, {0u, 1u, 2u}, "multiple in order"},
        {{not_ready, ready_view(), not_mapped, no_fd}, {1u}, "unready/unmapped/fdless excluded"}};
    for(const auto& tc : rows)
        EXPECT_EQ(poll_slots(tc.sessions), tc.want) << tc.label;
}
// A quarantined terminal (post-final-drain) drops out; survivor keeps its index.
TEST(poll_set, quarantined_terminal_is_excluded)
{
    std::vector<poll_session_view> sessions = {ready_view(), ready_view()};
    EXPECT_EQ(poll_slots(sessions).size(), 2u);

    sessions[0].quarantined = true;
    auto slots              = poll_slots(sessions);
    ASSERT_EQ(slots.size(), 1u);
    EXPECT_EQ(slots[0], 1u);
}
// HUP/NVAL terminal; ERR alone recoverable; POLLIN/0 neither; HUP wins over ERR.
TEST(terminal_revents, classify)
{
    struct tc
    {
        short           revents;
        terminal_action want;
        const char*     label;
    };
    const tc rows[] = {
        {POLLIN, terminal_action::none, "plain readable"},
        {0, terminal_action::none, "no events"},
        {POLLHUP, terminal_action::quarantine, "hup"},
        {POLLNVAL, terminal_action::quarantine, "nval"},
        {static_cast<short>(POLLIN | POLLHUP), terminal_action::quarantine, "in|hup final drain"},
        {POLLERR, terminal_action::keep_live, "err alone"},
        {static_cast<short>(POLLIN | POLLERR), terminal_action::keep_live, "in|err"},
        {static_cast<short>(POLLERR | POLLHUP), terminal_action::quarantine, "hup wins over err"}};
    for(const auto& tc : rows)
        EXPECT_EQ(classify_terminal_revents(tc.revents), tc.want) << tc.label;
}
// The backstop trips only strictly past kMaxConsecutiveEmptyWakes.
TEST(spin_backstop, threshold)
{
    struct tc
    {
        int         wakes;
        bool        want;
        const char* label;
    };
    const tc rows[] = {{0, false, "zero"},
                       {1, false, "one"},
                       {kMaxConsecutiveEmptyWakes, false, "at threshold"},
                       {kMaxConsecutiveEmptyWakes + 1, true, "one past"},
                       {kMaxConsecutiveEmptyWakes + 1000, true, "far past"}};
    for(const auto& tc : rows)
        EXPECT_EQ(empty_wake_backstop_tripped(tc.wakes), tc.want) << tc.label;
}
// Only a STREAM pollfd (slot >= 1) reporting POLLIN counts; the control fd at slot
// 0 is ignored, so a bare control-eventfd nudge does not feed the spin backstop.
TEST(stream_pollin, control_fd_ignored)
{
    // Control fd alone readable: not a stream readable state.
    pollfd ctrl_only[] = {{.fd = 0, .events = POLLIN, .revents = POLLIN}};
    EXPECT_FALSE(any_stream_pollin(ctrl_only, 1)) << "control fd only";

    // Control readable, one stream not readable.
    pollfd ctrl_and_idle[] = {{.fd = 0, .events = POLLIN, .revents = POLLIN},
                              {.fd = 1, .events = POLLIN, .revents = 0}};
    EXPECT_FALSE(any_stream_pollin(ctrl_and_idle, 2)) << "control readable, stream idle";

    // One stream readable.
    pollfd one_stream[] = {{.fd = 0, .events = POLLIN, .revents = 0},
                           {.fd = 1, .events = POLLIN, .revents = POLLIN}};
    EXPECT_TRUE(any_stream_pollin(one_stream, 2)) << "one stream readable";

    // A later stream readable while an earlier one is idle.
    pollfd second_stream[] = {{.fd = 0, .events = POLLIN, .revents = 0},
                              {.fd = 1, .events = POLLIN, .revents = 0},
                              {.fd = 2, .events = POLLIN, .revents = POLLIN}};
    EXPECT_TRUE(any_stream_pollin(second_stream, 3)) << "second stream readable";

    // Backstop-tripped shape: only the control fd is polled (count == 1), so even a
    // stale stream revents cannot be seen.
    pollfd backstop[] = {{.fd = 0, .events = POLLIN, .revents = POLLIN},
                         {.fd = 1, .events = POLLIN, .revents = POLLIN}};
    EXPECT_FALSE(any_stream_pollin(backstop, 1)) << "backstop polls control only";
}
