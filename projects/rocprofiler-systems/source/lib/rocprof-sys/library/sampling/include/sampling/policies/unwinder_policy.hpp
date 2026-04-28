// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ============================================================================
// UnwinderPolicy — named requirement
// ============================================================================
// Required member functions:
//   std::span<uintptr_t const> unwind(ThreadContext const& ctx) noexcept;
//       Capture raw instruction pointers from the supplied (signal-handler)
//       context into a thread-local fixed-size array. Returns a span into
//       that array. Must be async-signal-safe: no malloc, no printf, no locks.
//       The returned span is valid until the next call on the same thread.
//   bool valid_pc(uintptr_t pc) noexcept;
//       Return false if pc is a sentinel / non-executable region to cut off
//       the walk. Return true for all other addresses.
//
// Production: rocprofsys::sampling::LibunwindUnwinder
//             - thread-local pre-allocated uintptr_t[64] array (R-A1 mitigation)
//             - calls unw_getcontext + unw_init_local + unw_step in signal context
//             - DWARF resolution deferred to post_process
// Test double: rocprofsys::sampling::test::MockUnwinder
//             - records each unwind() call
//             - returns the next scripted std::vector<uintptr_t> (as span into internal
//             buffer)

// Forward declarations only — implementations live in src/linux/ and tests/doubles/.
namespace rocprofsys::sampling
{
class libunwind_unwinder;
}
namespace rocprofsys::sampling::test
{
struct mock_unwinder;
}
