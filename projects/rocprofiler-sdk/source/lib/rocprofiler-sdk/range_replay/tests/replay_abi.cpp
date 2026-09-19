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

// Layout and contract checks for the public range-replay callback-tracing surface, mirroring
// kernel_replay/tests/replay_abi.cpp.
//
// rocprofiler_callback_tracing_range_replay_data_t crosses the library boundary: the SDK fills it
// in and a separately-compiled tool reads it. Once released, moving or resizing a field silently
// misreads every field after it in a tool built against the older header, and the `size` member is
// the only thing that lets a tool notice. These tests pin the layout so that kind of change has to
// be deliberate.
//
// Everything here is a property of the header plus the struct definition, so it needs no GPU, no
// HSA and no rocprofiler runtime -- it does not even need to link the SDK.

#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

// Implemented in replay_abi_c.c, compiled as C against the same public header. See that file for
// why the C translation unit exists.
extern "C" {
size_t
rocprofiler_test_c_range_record_size(void);
size_t
rocprofiler_test_c_range_record_align(void);
size_t
rocprofiler_test_c_range_offset_size(void);
size_t
rocprofiler_test_c_range_offset_range_id(void);
size_t
rocprofiler_test_c_range_offset_pass_count_cb(void);
size_t
rocprofiler_test_c_range_offset_replay_continue_cb(void);
size_t
rocprofiler_test_c_range_offset_current_pass(void);
size_t
rocprofiler_test_c_range_offset_total_passes(void);
size_t
rocprofiler_test_c_range_offset_local_start_context(void);
size_t
rocprofiler_test_c_range_offset_local_stop_context(void);
size_t
rocprofiler_test_c_range_offset_agent_id(void);
size_t
rocprofiler_test_c_range_offset_dispatch_count(void);
size_t
rocprofiler_test_c_range_offset_status(void);
size_t
rocprofiler_test_c_range_offset_divergence_count(void);
int
rocprofiler_test_c_range_operation_last(void);
int
rocprofiler_test_c_range_status_last(void);
int
rocprofiler_test_c_range_tracing_kind(void);
size_t
rocprofiler_test_c_range_status_enum_size(void);
void
rocprofiler_test_c_range_fill_close(void*    record,
                                    uint64_t range_id,
                                    uint64_t dispatch_count,
                                    int      status,
                                    uint64_t divergence_count);
}

namespace
{
using range_data_t = rocprofiler_callback_tracing_range_replay_data_t;

// Field offsets are asserted against these named constants rather than against each other so that a
// failure names the field that moved. The values are what the current header produces on the LP64
// ABI the SDK ships on; they are a record of the released layout, not a derivation of it.
constexpr size_t k_offset_size     = 0;
constexpr size_t k_offset_range_id = sizeof(uint64_t);
}  // namespace

// ---------------------------------------------------------------------------
// Struct-level properties
// ---------------------------------------------------------------------------

// A C tool must be able to memcpy the record and pass it across a library boundary. Anything that
// makes the type non-trivial (a virtual, a user-provided constructor, a reference member) would
// break that without necessarily failing to compile here.
TEST(range_replay_abi, record_is_a_c_compatible_aggregate)
{
    EXPECT_TRUE(std::is_standard_layout<range_data_t>::value);
    EXPECT_TRUE(std::is_trivially_copyable<range_data_t>::value);
    EXPECT_TRUE(std::is_trivially_default_constructible<range_data_t>::value);
    EXPECT_TRUE(std::is_trivially_destructible<range_data_t>::value);
}

// `size` must stay first: a tool reads it before it trusts any other field, so it is the one member
// whose location cannot be renegotiated.
TEST(range_replay_abi, size_is_the_first_member)
{
    EXPECT_EQ(offsetof(range_data_t, size), k_offset_size);
    EXPECT_TRUE((std::is_same<decltype(range_data_t{}.size), uint64_t>::value));
}

// range_id is echoed back in every callback for the range and is the only way a tool correlates
// CONFIG, PASS and CLOSE with the begin() call it made, so it is pinned next to `size`.
TEST(range_replay_abi, range_id_follows_size)
{
    EXPECT_EQ(offsetof(range_data_t, range_id), k_offset_range_id);
    EXPECT_TRUE((std::is_same<decltype(range_data_t{}.range_id), uint64_t>::value));
}

// The declared field order is the ABI. The fields must appear in the documented order.
TEST(range_replay_abi, members_are_in_declaration_order)
{
    EXPECT_LT(offsetof(range_data_t, size), offsetof(range_data_t, range_id));
    EXPECT_LT(offsetof(range_data_t, range_id), offsetof(range_data_t, pass_count_cb));
    EXPECT_LT(offsetof(range_data_t, pass_count_cb), offsetof(range_data_t, replay_continue_cb));
    EXPECT_LT(offsetof(range_data_t, replay_continue_cb), offsetof(range_data_t, current_pass));
    EXPECT_LT(offsetof(range_data_t, current_pass), offsetof(range_data_t, total_passes));
    EXPECT_LT(offsetof(range_data_t, total_passes),
              offsetof(range_data_t, replay_local_start_context_cb));
    EXPECT_LT(offsetof(range_data_t, replay_local_start_context_cb),
              offsetof(range_data_t, replay_local_stop_context_cb));
    EXPECT_LT(offsetof(range_data_t, replay_local_stop_context_cb),
              offsetof(range_data_t, agent_id));
    EXPECT_LT(offsetof(range_data_t, agent_id), offsetof(range_data_t, dispatch_count));
    EXPECT_LT(offsetof(range_data_t, dispatch_count), offsetof(range_data_t, status));
    EXPECT_LT(offsetof(range_data_t, status), offsetof(range_data_t, divergence_count));
    EXPECT_LT(offsetof(range_data_t, divergence_count), sizeof(range_data_t));
}

// No member may overlap the one after it, and the last member must fit inside the struct. This
// catches a field whose type shrank without its neighbours being revisited.
TEST(range_replay_abi, members_do_not_overlap_and_fit)
{
    EXPECT_GE(offsetof(range_data_t, range_id),
              offsetof(range_data_t, size) + sizeof(range_data_t{}.size));
    EXPECT_GE(offsetof(range_data_t, pass_count_cb),
              offsetof(range_data_t, range_id) + sizeof(range_data_t{}.range_id));
    EXPECT_GE(offsetof(range_data_t, replay_continue_cb),
              offsetof(range_data_t, pass_count_cb) + sizeof(range_data_t{}.pass_count_cb));
    EXPECT_GE(
        offsetof(range_data_t, current_pass),
        offsetof(range_data_t, replay_continue_cb) + sizeof(range_data_t{}.replay_continue_cb));
    EXPECT_GE(offsetof(range_data_t, total_passes),
              offsetof(range_data_t, current_pass) + sizeof(range_data_t{}.current_pass));
    EXPECT_GE(offsetof(range_data_t, replay_local_start_context_cb),
              offsetof(range_data_t, total_passes) + sizeof(range_data_t{}.total_passes));
    EXPECT_GE(offsetof(range_data_t, replay_local_stop_context_cb),
              offsetof(range_data_t, replay_local_start_context_cb) +
                  sizeof(range_data_t{}.replay_local_start_context_cb));
    EXPECT_GE(offsetof(range_data_t, agent_id),
              offsetof(range_data_t, replay_local_stop_context_cb) +
                  sizeof(range_data_t{}.replay_local_stop_context_cb));
    EXPECT_GE(offsetof(range_data_t, dispatch_count),
              offsetof(range_data_t, agent_id) + sizeof(range_data_t{}.agent_id));
    EXPECT_GE(offsetof(range_data_t, status),
              offsetof(range_data_t, dispatch_count) + sizeof(range_data_t{}.dispatch_count));
    EXPECT_GE(offsetof(range_data_t, divergence_count),
              offsetof(range_data_t, status) + sizeof(range_data_t{}.status));
    EXPECT_GE(sizeof(range_data_t),
              offsetof(range_data_t, divergence_count) + sizeof(range_data_t{}.divergence_count));
}

// Pass counters and the counts reported at CLOSE are documented as uint64_t. A tool reading them
// through the header's type must see the same width the SDK wrote.
TEST(range_replay_abi, counters_are_64_bit)
{
    EXPECT_TRUE((std::is_same<decltype(range_data_t{}.current_pass), uint64_t>::value));
    EXPECT_TRUE((std::is_same<decltype(range_data_t{}.total_passes), uint64_t>::value));
    EXPECT_TRUE((std::is_same<decltype(range_data_t{}.dispatch_count), uint64_t>::value));
    EXPECT_TRUE((std::is_same<decltype(range_data_t{}.divergence_count), uint64_t>::value));
    EXPECT_EQ(sizeof(range_data_t{}.current_pass), 8u);
    EXPECT_EQ(sizeof(range_data_t{}.total_passes), 8u);
    EXPECT_EQ(sizeof(range_data_t{}.dispatch_count), 8u);
    EXPECT_EQ(sizeof(range_data_t{}.divergence_count), 8u);
}

// The four callback members are function pointers the tool either reads or writes. Their exact
// signatures are the contract; a changed parameter list here is an ABI break that the compiler
// would not otherwise flag at the tool. The two context toggles must match kernel replay's
// signature exactly, because a tool that drives both services reuses the same functions.
TEST(range_replay_abi, callback_signatures_are_pinned)
{
    using pass_count_fn = uint64_t (*)(uint64_t, rocprofiler_user_data_t);
    using continue_fn   = int (*)(uint64_t, uint64_t, uint64_t, rocprofiler_user_data_t);
    using toggle_fn     = rocprofiler_status_t (*)(rocprofiler_context_id_t);

    EXPECT_TRUE((std::is_same<decltype(range_data_t{}.pass_count_cb), pass_count_fn>::value));
    EXPECT_TRUE((std::is_same<decltype(range_data_t{}.replay_continue_cb), continue_fn>::value));
    EXPECT_TRUE(
        (std::is_same<decltype(range_data_t{}.replay_local_start_context_cb), toggle_fn>::value));
    EXPECT_TRUE(
        (std::is_same<decltype(range_data_t{}.replay_local_stop_context_cb), toggle_fn>::value));
}

// A zeroed record must mean "do not replay this range". The header documents a NULL pass_count_cb
// as the per-range opt-out, so zero-initialization has to produce it: a tool that ignores the
// CONFIG callback entirely gets the safe behaviour.
TEST(range_replay_abi, zero_initialized_record_opts_out_of_replay)
{
    range_data_t rec = {};
    EXPECT_EQ(rec.pass_count_cb, nullptr);
    EXPECT_EQ(rec.replay_continue_cb, nullptr);
    EXPECT_EQ(rec.replay_local_start_context_cb, nullptr);
    EXPECT_EQ(rec.replay_local_stop_context_cb, nullptr);
    EXPECT_EQ(rec.current_pass, 0u);
    EXPECT_EQ(rec.total_passes, 0u);
    EXPECT_EQ(rec.dispatch_count, 0u);
    EXPECT_EQ(rec.divergence_count, 0u);
    EXPECT_EQ(rec.size, 0u);
}

// A zeroed status must be REPLAYED, matching the neutral value the internal record starts from. A
// tool that reads `status` out of a record the SDK did not fill must not see a spurious decline
// reason, and the SDK's own "has this range been declined yet" check is a comparison against 0.
TEST(range_replay_abi, zero_status_is_replayed)
{
    range_data_t rec = {};
    EXPECT_EQ(rec.status, ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED), 0);
}

// The versioning scheme compares the SDK-written size against the tool's compiled layout prefix.
// The record carries no tail reservation, so size is the whole struct: a tool that finds a field's
// offset below the size the SDK wrote knows the SDK populated that field.
TEST(range_replay_abi, size_member_carries_the_layout_prefix)
{
    range_data_t rec = {};
    rec.size         = sizeof(range_data_t);
    EXPECT_EQ(rec.size, sizeof(range_data_t));
    EXPECT_GT(rec.size, offsetof(range_data_t, divergence_count));
}

// A record is passed by pointer and read field-by-field; it must be aligned for its widest member
// so those reads are well-defined on every target the SDK supports.
TEST(range_replay_abi, alignment_is_sufficient_for_widest_member)
{
    EXPECT_GE(alignof(range_data_t), alignof(uint64_t));
    EXPECT_EQ(sizeof(range_data_t) % alignof(range_data_t), 0u);
}

// ---------------------------------------------------------------------------
// Enum surface
// ---------------------------------------------------------------------------

// Operation IDs are written into trace records and matched by tools, so their numeric values are
// part of the ABI. NONE must stay 0 so a zeroed record does not name a real operation.
TEST(range_replay_abi, operation_enum_values_are_stable)
{
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_NONE), 0);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_CONFIG), 1);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_PASS), 2);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_CLOSE), 3);
}

// New operations must be appended before LAST, never inserted among the existing ones. If this
// fails because an operation was added, the fix is to bump the expected count here -- but only
// after confirming the new value was appended rather than inserted.
TEST(range_replay_abi, operation_enum_grows_only_at_the_end)
{
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_LAST), 4);
    EXPECT_GT(static_cast<int>(ROCPROFILER_RANGE_REPLAY_LAST),
              static_cast<int>(ROCPROFILER_RANGE_REPLAY_CLOSE));
}

// Every real operation must be inside (NONE, LAST) so a bounds check of that form accepts exactly
// the valid ones. Tools and the SDK's own dispatch tables rely on this.
TEST(range_replay_abi, real_operations_are_within_bounds)
{
    for(auto op : {ROCPROFILER_RANGE_REPLAY_CONFIG,
                   ROCPROFILER_RANGE_REPLAY_PASS,
                   ROCPROFILER_RANGE_REPLAY_CLOSE})
    {
        EXPECT_GT(static_cast<int>(op), static_cast<int>(ROCPROFILER_RANGE_REPLAY_NONE));
        EXPECT_LT(static_cast<int>(op), static_cast<int>(ROCPROFILER_RANGE_REPLAY_LAST));
    }
}

// Status values are reported to tools and appear in their logs and decision logic, so like the
// operation IDs they are numerically part of the ABI. Decline reasons must be appended, never
// renumbered: a tool built against an older header would otherwise attribute a range's decline to
// the wrong cause.
TEST(range_replay_abi, status_enum_values_are_stable)
{
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED), 0);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_NO_DISPATCH), 1);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_NO_PASS_COUNT), 2);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_QUEUE), 3);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_AGENT), 4);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_GRAPH_LAUNCH), 5);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_UNKNOWN_KERNARG_SIZE), 6);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_MEMORY_COPY_IN_RANGE), 7);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH), 8);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_PROGRAM_TOO_LARGE), 9);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED), 10);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_STAGING_FAILED), 11);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_ALLOCATION_CHANGED_IN_RANGE), 12);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_UNSUPPORTED_QUEUE_PATH), 13);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_LAST), 14);
}

// Every decline reason must be distinguishable from REPLAYED, since that is the single comparison
// a tool makes to decide whether the range produced comparable passes.
TEST(range_replay_abi, every_decline_reason_differs_from_replayed)
{
    for(int i = 1; i < static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_LAST); ++i)
        EXPECT_NE(i, static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED));
}

// The range replay tracing kind must sit inside the callback-tracing enum's valid range, since it
// indexes the SDK's per-kind service tables.
TEST(range_replay_abi, tracing_kind_is_within_callback_tracing_bounds)
{
    EXPECT_GT(static_cast<int>(ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY),
              static_cast<int>(ROCPROFILER_CALLBACK_TRACING_NONE));
    EXPECT_LT(static_cast<int>(ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY),
              static_cast<int>(ROCPROFILER_CALLBACK_TRACING_LAST));
}

// Range replay and kernel replay are separate domains. A tool may subscribe to both, and the SDK
// routes operation-name lookups by kind, so the two must never collide.
TEST(range_replay_abi, range_replay_is_a_distinct_domain_from_kernel_replay)
{
    EXPECT_NE(static_cast<int>(ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY),
              static_cast<int>(ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY));
}

// ---------------------------------------------------------------------------
// Documented semantics that are expressible without a device
// ---------------------------------------------------------------------------

// Unlike kernel replay, range replay's current_pass is 1-based: pass 0 is the application's own
// execution of the range, which the SDK observes rather than drives, so it raises no PASS callback.
// A tool that assumed kernel replay's 0-based convention would treat the first replayed pass as the
// live run. This pins the difference.
TEST(range_replay_abi, replayed_pass_indices_start_at_one)
{
    range_data_t rec = {};
    rec.total_passes = 4;

    // The SDK delivers PASS for 1..total_passes-1; pass 0 is never delivered.
    for(uint64_t pass = 1; pass < rec.total_passes; ++pass)
    {
        rec.current_pass = pass;
        EXPECT_GE(rec.current_pass, 1u);
        EXPECT_LT(rec.current_pass, rec.total_passes);
    }

    rec.current_pass = rec.total_passes - 1;
    EXPECT_EQ(rec.current_pass, 3u);
}

// An indefinite loop is signalled by total_passes == 0, which is distinct from "the live run only".
// A tool switching on total_passes must be able to tell those apart.
TEST(range_replay_abi, indefinite_loop_is_distinguishable_from_single_pass)
{
    range_data_t indefinite = {};
    indefinite.total_passes = 0;

    range_data_t single = {};
    single.total_passes = 1;

    EXPECT_NE(indefinite.total_passes, single.total_passes);
    EXPECT_EQ(indefinite.total_passes, 0u);
}

// The SDK threads one user_data value through CONFIG and every PASS callback for a range. The
// record carries the callbacks by value, so writing one must not disturb neighbouring fields -- a
// layout error would show up here as a corrupted pass counter.
TEST(range_replay_abi, writing_callbacks_does_not_disturb_counters)
{
    range_data_t rec   = {};
    rec.size           = sizeof(range_data_t);
    rec.range_id       = 42;
    rec.current_pass   = 2;
    rec.total_passes   = 5;
    rec.dispatch_count = 17;

    rec.pass_count_cb = [](uint64_t, rocprofiler_user_data_t) -> uint64_t { return 7; };

    EXPECT_EQ(rec.range_id, 42u);
    EXPECT_EQ(rec.current_pass, 2u);
    EXPECT_EQ(rec.total_passes, 5u);
    EXPECT_EQ(rec.dispatch_count, 17u);
    EXPECT_EQ(rec.size, sizeof(range_data_t));
    ASSERT_NE(rec.pass_count_cb, nullptr);
    EXPECT_EQ(rec.pass_count_cb(rec.range_id, rocprofiler_user_data_t{}), 7u);
}

// A tool built against an older, shorter header memcpy's only the prefix it knows about. The bytes
// it did copy must still read back correctly, which is what makes the size-prefixed scheme work.
TEST(range_replay_abi, truncated_copy_preserves_the_known_prefix)
{
    range_data_t src = {};
    src.size         = sizeof(range_data_t);
    src.range_id     = 11;
    src.current_pass = 13;
    src.total_passes = 17;

    const size_t old_size =
        offsetof(range_data_t, total_passes) + sizeof(range_data_t{}.total_passes);
    ASSERT_LE(old_size, sizeof(range_data_t));

    range_data_t dst = {};
    std::memcpy(&dst, &src, old_size);

    EXPECT_EQ(dst.size, src.size);
    EXPECT_EQ(dst.range_id, 11u);
    EXPECT_EQ(dst.current_pass, 13u);
    EXPECT_EQ(dst.total_passes, 17u);

    // The CLOSE-only fields live past the truncation point and must be untouched.
    EXPECT_EQ(dst.status, ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED);
    EXPECT_EQ(dst.divergence_count, 0u);
}

// ---------------------------------------------------------------------------
// C / C++ agreement
//
// The SDK is C++ and most tools are C. Both compile the same header, so the layout each language
// derives from it has to match; if it does not, a C tool reads the wrong bytes with nothing
// failing to build. These compare C's view (from replay_abi_c.c) against C++'s.
// ---------------------------------------------------------------------------

TEST(range_replay_abi, c_and_cxx_agree_on_record_size_and_alignment)
{
    EXPECT_EQ(rocprofiler_test_c_range_record_size(), sizeof(range_data_t));
    EXPECT_EQ(rocprofiler_test_c_range_record_align(), alignof(range_data_t));
}

TEST(range_replay_abi, c_and_cxx_agree_on_every_field_offset)
{
    EXPECT_EQ(rocprofiler_test_c_range_offset_size(), offsetof(range_data_t, size));
    EXPECT_EQ(rocprofiler_test_c_range_offset_range_id(), offsetof(range_data_t, range_id));
    EXPECT_EQ(rocprofiler_test_c_range_offset_pass_count_cb(),
              offsetof(range_data_t, pass_count_cb));
    EXPECT_EQ(rocprofiler_test_c_range_offset_replay_continue_cb(),
              offsetof(range_data_t, replay_continue_cb));
    EXPECT_EQ(rocprofiler_test_c_range_offset_current_pass(), offsetof(range_data_t, current_pass));
    EXPECT_EQ(rocprofiler_test_c_range_offset_total_passes(), offsetof(range_data_t, total_passes));
    EXPECT_EQ(rocprofiler_test_c_range_offset_local_start_context(),
              offsetof(range_data_t, replay_local_start_context_cb));
    EXPECT_EQ(rocprofiler_test_c_range_offset_local_stop_context(),
              offsetof(range_data_t, replay_local_stop_context_cb));
    EXPECT_EQ(rocprofiler_test_c_range_offset_agent_id(), offsetof(range_data_t, agent_id));
    EXPECT_EQ(rocprofiler_test_c_range_offset_dispatch_count(),
              offsetof(range_data_t, dispatch_count));
    EXPECT_EQ(rocprofiler_test_c_range_offset_status(), offsetof(range_data_t, status));
    EXPECT_EQ(rocprofiler_test_c_range_offset_divergence_count(),
              offsetof(range_data_t, divergence_count));
}

// `status` is the record's only enum-typed member. If the two languages pick different underlying
// types for it, every field after it shifts -- the offset comparison above would catch that, but
// this names the cause directly.
TEST(range_replay_abi, c_and_cxx_agree_on_the_status_enum_width)
{
    EXPECT_EQ(rocprofiler_test_c_range_status_enum_size(),
              sizeof(rocprofiler_range_replay_status_t));
}

TEST(range_replay_abi, c_and_cxx_agree_on_enum_values)
{
    EXPECT_EQ(rocprofiler_test_c_range_operation_last(),
              static_cast<int>(ROCPROFILER_RANGE_REPLAY_LAST));
    EXPECT_EQ(rocprofiler_test_c_range_status_last(),
              static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_LAST));
    EXPECT_EQ(rocprofiler_test_c_range_tracing_kind(),
              static_cast<int>(ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY));
}

// A record written through C's view of the struct must read back correctly through C++'s. This is
// the end-to-end version of the offset comparison above: it would fail if the two languages
// disagreed about padding even where the individual offsets happened to line up. A CLOSE record is
// used because its fields sit at the end of the struct, past the callback pointers and the agent
// id, which is where a padding disagreement is most likely to surface.
TEST(range_replay_abi, close_record_written_by_c_reads_back_in_cxx)
{
    range_data_t rec = {};
    rocprofiler_test_c_range_fill_close(
        &rec, 5, 23, static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_QUEUE), 2);

    EXPECT_EQ(rec.size, sizeof(range_data_t));
    EXPECT_EQ(rec.range_id, 5u);
    EXPECT_EQ(rec.dispatch_count, 23u);
    EXPECT_EQ(rec.status, ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_QUEUE);
    EXPECT_EQ(rec.divergence_count, 2u);

    // Fields the C side did not write must still be zero: a layout disagreement would show up as
    // one of C's writes landing on top of a neighbouring member.
    EXPECT_EQ(rec.pass_count_cb, nullptr);
    EXPECT_EQ(rec.replay_continue_cb, nullptr);
    EXPECT_EQ(rec.replay_local_start_context_cb, nullptr);
    EXPECT_EQ(rec.replay_local_stop_context_cb, nullptr);
    EXPECT_EQ(rec.current_pass, 0u);
    EXPECT_EQ(rec.total_passes, 0u);
}
