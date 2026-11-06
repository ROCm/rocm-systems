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
#include <sys/eventfd.h>
#include <unistd.h>
#include <cerrno>
#include <climits>
#include <cstdint>
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

// A stream stuck at POLLERR (no POLLIN) must feed the empty-wake backstop, so the
// reader falls back to a timeout-only wait instead of spinning: poll() returns
// POLLERR immediately every pass. any_stream_pollin misses it (no POLLIN bit), so
// any_stream_error is the second signal that keeps the backstop reachable.
TEST(stream_error, pollerr_feeds_backstop)
{
    // POLLERR only, no POLLIN: any_stream_pollin misses it, any_stream_error sees it.
    pollfd err_only[] = {{.fd = 0, .events = POLLIN, .revents = 0},
                         {.fd = 1, .events = POLLIN, .revents = POLLERR}};
    EXPECT_FALSE(any_stream_pollin(err_only, 2)) << "POLLERR is not POLLIN";
    EXPECT_TRUE(any_stream_error(err_only, 2)) << "POLLERR on a stream";

    // The control fd is never a stream error, even if it somehow reported POLLERR.
    pollfd ctrl_err[] = {{.fd = 0, .events = POLLIN, .revents = POLLERR}};
    EXPECT_FALSE(any_stream_error(ctrl_err, 1)) << "control slot excluded";

    // No error present.
    pollfd clean[] = {{.fd = 0, .events = POLLIN, .revents = POLLIN},
                      {.fd = 1, .events = POLLIN, .revents = POLLIN}};
    EXPECT_FALSE(any_stream_error(clean, 2)) << "no POLLERR";

    // A later stream in error while an earlier one is idle.
    pollfd second_err[] = {{.fd = 0, .events = POLLIN, .revents = 0},
                           {.fd = 1, .events = POLLIN, .revents = 0},
                           {.fd = 2, .events = POLLIN, .revents = POLLERR}};
    EXPECT_TRUE(any_stream_error(second_err, 3)) << "second stream in error";

    // Backstop-tripped shape: only the control fd is polled (count == 1).
    pollfd backstop[] = {{.fd = 0, .events = POLLIN, .revents = 0},
                         {.fd = 1, .events = POLLIN, .revents = POLLERR}};
    EXPECT_FALSE(any_stream_error(backstop, 1)) << "backstop polls control only";
}

// A stream wake (POLLIN or POLLERR) that copied nothing is unproductive; a bare
// control nudge is not.
TEST(stream_error, unproductive_predicate)
{
    EXPECT_TRUE(wake_is_unproductive(/*pollin=*/true, /*pollerr=*/false, /*copied=*/0));
    EXPECT_TRUE(wake_is_unproductive(false, true, 0));
    EXPECT_FALSE(wake_is_unproductive(true, false, 5)) << "copied something -> productive";
    EXPECT_FALSE(wake_is_unproductive(false, false, 0)) << "control nudge only -> not a spin";
}

// The full sticky-POLLERR state transition: the backstop trips and then STAYS
// tripped through control-only timeouts (it must not re-burst), clearing only on
// real copy progress. Drives the pure transition the reader loop uses.
TEST(spin_backstop, sticky_pollerr_latches_until_progress)
{
    int  wakes = 0;
    auto trip  = [&] { return empty_wake_backstop_tripped(wakes); };

    // Unproductive POLLERR wakes (not timed out, not yet tripped) climb the counter.
    for(int i = 0; i <= kMaxConsecutiveEmptyWakes; ++i)
    {
        ASSERT_FALSE(trip()) << "not tripped yet at " << i;
        wakes = next_empty_wakes(wakes,
                                 /*unproductive=*/true,
                                 /*copied_any=*/false,
                                 /*timed_out=*/false,
                                 /*backstop_tripped=*/trip());
    }
    ASSERT_TRUE(trip()) << "tripped once past the threshold";

    // Now tripped: control-only timeouts (backstop_tripped=true, copied 0) must HOLD
    // the latch. The old code reset here, re-bursting ~100 Hz.
    for(int i = 0; i < 1000; ++i)
        wakes = next_empty_wakes(wakes,
                                 /*unproductive=*/false,
                                 /*copied_any=*/false,
                                 /*timed_out=*/true,
                                 /*backstop_tripped=*/true);
    EXPECT_TRUE(trip()) << "a tripped-backstop timeout must not clear the latch";

    // Real copy progress clears it -> the stream is polled again next pass.
    wakes = next_empty_wakes(wakes,
                             /*unproductive=*/false,
                             /*copied_any=*/true,
                             /*timed_out=*/false,
                             /*backstop_tripped=*/true);
    EXPECT_FALSE(trip()) << "copy progress clears the latch";

    // A GENUINE idle timeout (stream polled, did not fire) also clears it.
    wakes = 5;
    wakes = next_empty_wakes(wakes,
                             /*unproductive=*/false,
                             /*copied_any=*/false,
                             /*timed_out=*/true,
                             /*backstop_tripped=*/false);
    EXPECT_EQ(wakes, 0) << "an untripped idle timeout is real evidence the spin cleared";
}

// The reader must read() the kernel's notify count only on a POLLIN wake. A
// HUP/ERR-only wake carries no count (nothing to consume) and reading it would
// only churn EAGAIN or, on a torn-down fd, add noise.
TEST(notify_read, wants_read_only_on_pollin)
{
    struct tc
    {
        short       revents;
        bool        want;
        const char* label;
    };
    const tc rows[] = {{POLLIN, true, "pollin reads"},
                       {static_cast<short>(POLLIN | POLLRDNORM), true, "pollin|rdnorm reads"},
                       {static_cast<short>(POLLIN | POLLHUP), true, "pollin|hup still reads"},
                       {POLLHUP, false, "hup only does not read"},
                       {POLLERR, false, "err only does not read"},
                       {POLLNVAL, false, "nval only does not read"},
                       {static_cast<short>(POLLERR | POLLHUP), false, "err|hup does not read"},
                       {0, false, "no bits does not read"}};
    for(const auto& tc : rows)
        EXPECT_EQ(stream_wake_wants_notify_read(tc.revents), tc.want) << tc.label;
}

namespace
{
// The reader's per-wake notify-count consume, mirrored against a real fd so the
// read-and-clear semantics are exercised end to end. An EFD_NONBLOCK eventfd has
// exactly the kernel stream fd's contract: read(&u64,8) returns 8 and zeroes the
// counter, or -1/EAGAIN when it is already 0, and never blocks. Returns true iff a
// read() was issued (i.e. the wake carried POLLIN).
bool
consume_notify_if_pollin(int fd, short revents)
{
    if(!stream_wake_wants_notify_read(revents)) return false;
    uint64_t count = 0;
    for(;;)
    {
        const ssize_t n = ::read(fd, &count, sizeof(count));
        if(n == static_cast<ssize_t>(sizeof(count))) return true;
        if(n < 0 && errno == EINTR) continue;
        return true;  // EAGAIN (count already 0) or fault: the read was still issued
    }
}

// poll(fd, POLLIN, 0): the kernel's level check. An eventfd reports POLLIN while
// its counter is nonzero, matching notify_count != 0.
bool
poll_ready(int fd)
{
    pollfd    p  = {.fd = fd, .events = POLLIN, .revents = 0};
    const int rc = ::poll(&p, 1, 0);
    return rc > 0 && (p.revents & POLLIN) != 0;
}
}  // namespace

// (a) A POLLIN wake leads to exactly one read that consumes the count, after which
// the fd is no longer level-ready. (b) Readiness is LEVEL: it persists across a
// poll that does not read, so a reader that only polls without reading would be
// woken forever -- the read is what clears it. This is the busy-spin the reader
// must avoid, and this test proves the read clears the level.
TEST(notify_read, pollin_wake_reads_once_and_clears_level)
{
    const int efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    ASSERT_GE(efd, 0);

    // Kernel notify: notify_count becomes nonzero.
    const uint64_t bump = 1;
    ASSERT_EQ(::write(efd, &bump, sizeof(bump)), static_cast<ssize_t>(sizeof(bump)));

    // Level readiness persists across a poll that does not read (the lost-wakeup /
    // busy-spin case the old edge design had): two polls with no read both fire.
    EXPECT_TRUE(poll_ready(efd)) << "notify pending -> POLLIN";
    EXPECT_TRUE(poll_ready(efd)) << "not consumed by polling -> still POLLIN (level)";

    // The reader's wake consumes it exactly once.
    EXPECT_TRUE(consume_notify_if_pollin(efd, POLLIN)) << "POLLIN wake issues a read";
    EXPECT_FALSE(poll_ready(efd)) << "read cleared the count -> no longer ready";

    // A second consume finds nothing (EAGAIN) and does not re-arm.
    EXPECT_TRUE(consume_notify_if_pollin(efd, POLLIN)) << "second read still issued";
    EXPECT_FALSE(poll_ready(efd)) << "still empty";

    ::close(efd);
}

// (c) A HUP/ERR-only wake (no POLLIN) must NOT read: the count is left intact for
// whatever POLLIN wake follows, and no spurious read is issued.
TEST(notify_read, hup_only_wake_does_not_read)
{
    const int efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    ASSERT_GE(efd, 0);

    const uint64_t bump = 1;
    ASSERT_EQ(::write(efd, &bump, sizeof(bump)), static_cast<ssize_t>(sizeof(bump)));

    // HUP-only: no read issued, count preserved, fd still level-ready.
    EXPECT_FALSE(consume_notify_if_pollin(efd, POLLHUP)) << "HUP-only issues no read";
    EXPECT_TRUE(poll_ready(efd)) << "count untouched -> still ready";

    // The subsequent POLLIN wake is what consumes it.
    EXPECT_TRUE(consume_notify_if_pollin(efd, static_cast<short>(POLLIN | POLLHUP)));
    EXPECT_FALSE(poll_ready(efd)) << "POLLIN wake cleared it";

    ::close(efd);
}
