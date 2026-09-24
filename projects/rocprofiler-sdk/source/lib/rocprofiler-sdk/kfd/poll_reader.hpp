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

#pragma once

// Pure poll-scheduling logic, factored out of kfd_reader.cpp so it can be
// unit-tested without a GPU, an mmap, the reader thread, or its singletons.
// These are the decisions the reader makes around poll(): the timeout to use,
// whether a session belongs in the active poll set, and the consecutive
// zero-copy backstop that turns a sticky-EPOLLIN spin into a timeout-only wait.
//
// Readiness is the kernel's level-triggered .poll predicate alone: a stream fd
// stays readable while wptr != rptr, so poll() re-triggers on its own. The SDK
// keeps no second readiness authority.

#include <climits>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <poll.h>

namespace rocprofiler
{
namespace kfd
{
// poll() timeout when ROCPROFILER_KFD_DISPATCH_LOG_POLL_TIMEOUT_MS is unset. The
// sparse-tail / lost-interrupt watchdog: on timeout the reader does a full scan.
constexpr int kPollTimeoutMsDefault = 10;

// Upper bound on drain passes per wake. Reaching it returns the reader to
// poll(); the kernel's level trigger fires it again at once if wptr != rptr, so
// a continuously advancing producer cannot livelock the reader in one wake.
constexpr int kMaxDrainPassesPerWake = 8;

// Consecutive wakes that reported readiness (poll returned before the timeout)
// yet copied nothing, before the reader falls back to a timeout-only wait. This
// is the spin backstop: if the kernel reports level EPOLLIN while the drain
// copies nothing (an incoherent or reset-mismatched rptr the drain reconcile did
// not resolve), the reader would otherwise free-run at 100% CPU. Past this count
// it warns once and stops trusting the level trigger until a copy makes progress.
constexpr int kMaxConsecutiveEmptyWakes = 64;

// Parse the poll-timeout env value. Domain is [1, INT_MAX] ms; anything else
// (empty, zero, negative, non-numeric, overflow) returns 0 to mean "use the
// default" -- 0 would busy-poll and a negative would block poll() forever, so
// both are rejected rather than passed to poll(). Bounded per digit so it can
// never wrap.
inline int
poll_timeout_ms_from_str(std::string_view v)
{
    if(v.empty()) return 0;
    uint64_t ms = 0;
    for(char c : v)
    {
        if(c < '0' || c > '9') return 0;
        ms = ms * 10 + static_cast<uint64_t>(c - '0');
        if(ms > static_cast<uint64_t>(INT_MAX)) return 0;
    }
    if(ms == 0) return 0;
    return static_cast<int>(ms);
}

// Whether a session belongs in the active poll set: it must be published (ready),
// mapped, and not quarantined. A terminal (POLLHUP) stream is quarantined after
// its final drain so a sticky HUP cannot spin poll().
inline bool
session_is_pollable(bool ready, bool quarantined, bool mapped)
{
    return ready && mapped && !quarantined;
}

// The poll-set attributes of one session, as build_pollfds() sees them. `has_fd`
// is `stream_fd >= 0`; a session with no fd is skipped before the pollable check.
struct poll_session_view
{
    bool has_fd      = false;
    bool ready       = false;
    bool quarantined = false;
    bool mapped      = false;
};

// Fill `slot_of` with the session indices that belong in the active poll set, in
// order. The caller places the control eventfd at pollfd[0] and one entry per
// returned index at pollfd[k+1], so slot_of[k] maps pollfd k+1 back to its
// session. This is the exact membership build_pollfds() uses; factored out so the
// production decision (fd skip, then the pollable predicate) is unit-testable.
template <typename SessionView, typename SlotOut>
void
build_poll_slots(const SessionView* sessions, size_t count, SlotOut& slot_of)
{
    for(size_t i = 0; i < count; ++i)
    {
        const auto& s = sessions[i];
        if(!s.has_fd) continue;
        if(!session_is_pollable(s.ready, s.quarantined, s.mapped)) continue;
        slot_of.push_back(i);
    }
}

// How the terminal handler must treat a stream's poll revents.
enum class terminal_action
{
    none,       // no terminal/error bit set
    keep_live,  // POLLERR only: a recoverable reset; log but keep the stream live
    quarantine  // POLLHUP/POLLNVAL: terminal; drain, then quarantine when level-dry
};

// Classify a stream's revents for the terminal handler. POLLHUP/POLLNVAL are
// terminal; POLLERR alone is a recoverable reset (the stream node survives). The
// reader still defers an actual quarantine until the terminal stream is level-dry.
inline terminal_action
classify_terminal_revents(short revents)
{
    if((revents & (POLLHUP | POLLNVAL)) != 0) return terminal_action::quarantine;
    if((revents & POLLERR) != 0) return terminal_action::keep_live;
    return terminal_action::none;
}

// The control/wake eventfd occupies poll slot 0; the stream fds follow at 1..N.
// build_pollfds() (the producer) and any_stream_pollin() / the terminal-revents
// scan (the consumers) all depend on this layout, so the index lives here as the
// single source of truth -- a caller that builds the array differently must change
// this, not silently diverge from a comment.
constexpr size_t kControlPollSlot     = 0;
constexpr size_t kFirstStreamPollSlot = kControlPollSlot + 1;

// Whether any STREAM pollfd reported POLLIN. Slot kControlPollSlot is the
// control/wake eventfd; only a stream readable state should feed the empty-wake
// backstop, since a control nudge that copies nothing is not a spin symptom.
inline bool
any_stream_pollin(const pollfd* fds, size_t count)
{
    for(size_t k = kFirstStreamPollSlot; k < count; ++k)
        if((fds[k].revents & POLLIN) != 0) return true;
    return false;
}

// Whether any STREAM pollfd reported POLLERR. poll() reports POLLERR without
// waiting, so a stream stuck at POLLERR wakes the reader immediately every pass
// while copying nothing -- exactly the spin the empty-wake backstop bounds. It
// therefore feeds the backstop alongside POLLIN. POLLHUP/POLLNVAL are excluded:
// those are quarantined out of the poll set after one drain, so they self-limit.
inline bool
any_stream_error(const pollfd* fds, size_t count)
{
    for(size_t k = kFirstStreamPollSlot; k < count; ++k)
        if((fds[k].revents & POLLERR) != 0) return true;
    return false;
}

// Spin backstop for a sticky level trigger. `empty_wakes` is how many consecutive
// pre-timeout wakes have copied nothing. Once it passes kMaxConsecutiveEmptyWakes
// the reader must stop trusting the level trigger and wait on the timeout only,
// so an incoherent rptr the drain reconcile could not resolve degrades to a
// diagnosable warning rather than an un-killable 100% CPU hang.
//
// UNVALIDATED-pending-hardware: the drain reconcile (copy_pipes forcing rptr:=wptr
// on a regression) should keep wptr and rptr coherent, but until that is validated
// on hardware this backstop is what bounds a mismatch.
inline bool
empty_wake_backstop_tripped(int empty_wakes)
{
    return empty_wakes > kMaxConsecutiveEmptyWakes;
}

// A wake is unproductive when a stream reported readiness (POLLIN) or an error
// (POLLERR, which poll() re-reports every pass until the reset clears) yet the
// drain copied nothing -- the spin symptom the backstop bounds. A bare control
// nudge (neither bit) is never unproductive.
inline bool
wake_is_unproductive(bool stream_pollin, bool stream_pollerr, uint64_t copied)
{
    return (stream_pollin || stream_pollerr) && copied == 0;
}

// The empty-wake backstop as a pure transition, so the sticky-POLLERR state
// machine is testable without a reader thread. The latch (empty_wakes) only clears
// on real copy progress or a GENUINE idle timeout -- one where the stream was
// actually polled (`!backstop_tripped`) and did not fire. A timeout taken while the
// backstop is tripped polls only the control fd, so it proves nothing about the
// stream and must HOLD the latch; otherwise a sticky POLLERR would reset every
// timeout and re-burst (~100 Hz) forever.
inline int
next_empty_wakes(int  empty_wakes,
                 bool unproductive,
                 bool copied_any,
                 bool timed_out,
                 bool backstop_tripped)
{
    if(unproductive) return empty_wakes + 1;
    if(copied_any) return 0;
    if(timed_out && !backstop_tripped) return 0;
    return empty_wakes;
}
}  // namespace kfd
}  // namespace rocprofiler
