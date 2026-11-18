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

// Cross-process agreement for host replay (an MPI job replaying a range together).
//
// The processes of a job must re-execute a range the same number of times and agree before any of
// them restores memory: a restore rewinds device memory a peer may still be reading, and a rank
// that re-executes alone blocks forever in the first MPI call whose partner is not coming. The tool
// supplies the collective (peer_sync_cb); this header is the SDK's side of it, kept free of any
// runtime dependency so the decision table can be tested without a GPU or MPI.
//
// At the end of every pass, each process calls the collective once with its flags and gets back the
// bitwise AND over the job. That single exchange tells it everything it needs:
//
//   local flags            meaning on this process
//   ---------------------  ----------------------------------------------------------------------
//   0                      declined (or never eligible): its range is over, whatever peers do
//   ELIGIBLE               finished its last pass: done, but peers may still want more
//   ELIGIBLE | CONTINUE    wants to restore and run another pass
//
// A process that already finished or declined has nothing left to decide, so it keeps its own
// outcome. Only a process that wants to continue depends on the answer.

#include <cstdint>

namespace rocprofiler
{
namespace range_replay
{
namespace peer
{
// Mirrors rocprofiler_range_replay_peer_flags_t; repeated so this header needs no public include.
constexpr uint32_t flag_eligible = 1U << 0;
constexpr uint32_t flag_continue = 1U << 1;

enum class action_t
{
    // The process's own outcome stands: it declined, or it already finished its passes.
    keep,
    // Every process is eligible and wants another pass: restore and re-execute.
    restore_and_continue,
    // Every process is eligible but one has run all the passes it wants. Stop where we are; the
    // range was replayed.
    stop_replayed,
    // A peer declined, or the exchange failed. Stop where we are and report PEER_DECLINED.
    stop_declined,
};

// What a process does once the collective returned `global`.
//
// `global` is masked with `local` first: the AND of the job can never grant a process a bit it did
// not offer, so a tool that returns more than that (a bug, or a reduction over the wrong
// communicator) cannot make a declined process continue. A failed exchange returns 0, which falls
// out as stop_declined, the conservative answer.
//
// `pass` is the pass that just ended (0 is the application's own run). Stopping at pass 0 means no
// extra pass ever ran, so the range was not replayed even though this process was ready to.
constexpr action_t
resolve(uint32_t local, uint32_t global, uint64_t pass)
{
    if((local & flag_continue) == 0) return action_t::keep;

    const uint32_t agreed = global & local;
    if((agreed & flag_eligible) == 0) return action_t::stop_declined;
    if((agreed & flag_continue) == 0)
        return (pass == 0) ? action_t::stop_declined : action_t::stop_replayed;
    return action_t::restore_and_continue;
}

// Flags a process offers at the end of a pass.
constexpr uint32_t
local_flags(bool eligible, bool wants_another_pass)
{
    if(!eligible) return 0;
    return flag_eligible | (wants_another_pass ? flag_continue : 0U);
}
}  // namespace peer
}  // namespace range_replay
}  // namespace rocprofiler
