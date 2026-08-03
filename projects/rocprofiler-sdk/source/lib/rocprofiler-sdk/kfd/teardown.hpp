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

// The signal-less teardown order (design requirement 7).
//
// This ordering is the ONLY one that guarantees no EOP-proven completion is
// stranded and no task is enqueued after the task group is joined, so it is
// factored into one place, templated on the steps, and unit-tested for order --
// a future edit that reorders it fails a test instead of producing a rare
// use-after-free at exit.
//
// Why each step must precede the next:
//   1. stop_new_reservations   - eligibility must fail first, or step 2 races new
//                                PENDING entries into existence behind it.
//   2. quiesce_interceptor     - fences in-flight registration/publication, so
//                                after it no NEW PENDING can appear at all.
//   3. stop_and_join_reader    - final status + final drain, THEN join. Only the
//                                reader creates PENDING -> EOP_PROVEN transitions
//                                and retry-owner insertions, so nothing can be
//                                added to the retry owner after this returns.
//   4. flush_retry_owner       - steps 1-3 guarantee no producer remains, so the
//                                flush is final; anything still held is finalized
//                                IN PLACE on this (teardown) thread.
//   5. leak_remaining_pending  - whatever never got an EOP becomes LEAKED and is
//                                ledgered, so correlation_id_finalize skips it.
//   6. join_task_group         - safe only now: steps 1-4 mean no producer can
//                                submit another task (invariant 12).
//
// Step 7 (queue_controller_fini / kfd::finalize / correlation_id_finalize with
// the loss-ledger skip) is the caller's existing finalization, which runs after.

namespace rocprofiler
{
namespace kfd
{
template <typename StepsT>
void
run_signal_less_teardown(StepsT& steps)
{
    steps.stop_new_reservations();
    steps.quiesce_interceptor();
    steps.stop_and_join_reader();
    steps.flush_retry_owner();
    steps.leak_remaining_pending();
    steps.join_task_group();
}
}  // namespace kfd
}  // namespace rocprofiler
