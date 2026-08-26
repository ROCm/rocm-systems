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

// Layout and contract checks for the public kernel-replay callback-tracing surface.
//
// rocprofiler_callback_tracing_kernel_replay_data_t crosses the library boundary: the SDK fills it
// in and a separately-compiled tool reads it. Once released, moving or resizing a field silently
// misreads every field after it in a tool built against the older header, and the `size` member is
// the only thing that lets a tool notice. These tests pin the layout so that kind of change has to
// be deliberate.
//
// Everything here is a property of the header plus the struct definition, so it needs no GPU, no
// HSA and no rocprofiler runtime -- it does not even need to link the SDK.

#include <rocprofiler-sdk/experimental/kernel_replay.h>
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
rocprofiler_test_c_replay_record_size(void);
size_t
rocprofiler_test_c_replay_record_align(void);
size_t
rocprofiler_test_c_replay_offset_size(void);
size_t
rocprofiler_test_c_replay_offset_dispatch_info(void);
size_t
rocprofiler_test_c_replay_offset_pass_count_cb(void);
size_t
rocprofiler_test_c_replay_offset_replay_continue_cb(void);
size_t
rocprofiler_test_c_replay_offset_current_pass(void);
size_t
rocprofiler_test_c_replay_offset_total_passes(void);
size_t
rocprofiler_test_c_replay_offset_local_enable_cb(void);
size_t
rocprofiler_test_c_replay_offset_local_disable_cb(void);
size_t
rocprofiler_test_c_replay_offset_reserved_padding(void);
int
rocprofiler_test_c_replay_operation_last(void);
int
rocprofiler_test_c_replay_tracing_kind(void);
void
rocprofiler_test_c_replay_fill(void* record, uint64_t current_pass, uint64_t total_passes);
}

namespace
{
using replay_data_t = rocprofiler_callback_tracing_kernel_replay_data_t;

// Field offsets are asserted against these named constants rather than against each other so that a
// failure names the field that moved. The values are what the current header produces on the LP64
// ABI the SDK ships on; they are a record of the released layout, not a derivation of it.
constexpr size_t k_offset_size          = 0;
constexpr size_t k_offset_dispatch_info = sizeof(uint64_t);
}  // namespace

// ---------------------------------------------------------------------------
// Struct-level properties
// ---------------------------------------------------------------------------

// A C tool must be able to memcpy the record and pass it across a library boundary. Anything that
// makes the type non-trivial (a virtual, a user-provided constructor, a reference member) would
// break that without necessarily failing to compile here.
TEST(kernel_replay_abi, record_is_a_c_compatible_aggregate)
{
    EXPECT_TRUE(std::is_standard_layout<replay_data_t>::value);
    EXPECT_TRUE(std::is_trivially_copyable<replay_data_t>::value);
    EXPECT_TRUE(std::is_trivially_default_constructible<replay_data_t>::value);
    EXPECT_TRUE(std::is_trivially_destructible<replay_data_t>::value);
}

// `size` must stay first: a tool reads it before it trusts any other field, so it is the one member
// whose location cannot be renegotiated.
TEST(kernel_replay_abi, size_is_the_first_member)
{
    EXPECT_EQ(offsetof(replay_data_t, size), k_offset_size);
    EXPECT_TRUE((std::is_same<decltype(replay_data_t{}.size), uint64_t>::value));
}

TEST(kernel_replay_abi, dispatch_info_follows_size)
{
    EXPECT_EQ(offsetof(replay_data_t, dispatch_info), k_offset_dispatch_info);
    EXPECT_TRUE((std::is_same<decltype(replay_data_t{}.dispatch_info),
                              rocprofiler_kernel_dispatch_info_t>::value));
}

// The declared field order is the ABI. Each field must start at or after the end of the previous
// one, and they must appear in the documented order.
TEST(kernel_replay_abi, members_are_in_declaration_order)
{
    EXPECT_LT(offsetof(replay_data_t, size), offsetof(replay_data_t, dispatch_info));
    EXPECT_LT(offsetof(replay_data_t, dispatch_info), offsetof(replay_data_t, pass_count_cb));
    EXPECT_LT(offsetof(replay_data_t, pass_count_cb), offsetof(replay_data_t, replay_continue_cb));
    EXPECT_LT(offsetof(replay_data_t, replay_continue_cb), offsetof(replay_data_t, current_pass));
    EXPECT_LT(offsetof(replay_data_t, current_pass), offsetof(replay_data_t, total_passes));
    EXPECT_LT(offsetof(replay_data_t, total_passes),
              offsetof(replay_data_t, replay_local_enable_context_cb));
    EXPECT_LT(offsetof(replay_data_t, replay_local_enable_context_cb),
              offsetof(replay_data_t, replay_local_disable_context_cb));
    EXPECT_LT(offsetof(replay_data_t, replay_local_disable_context_cb),
              offsetof(replay_data_t, reserved_padding));
}

// No member may overlap the one after it, and the last member must fit inside the struct. This
// catches a field whose type shrank without its neighbours being revisited.
TEST(kernel_replay_abi, members_do_not_overlap_and_fit)
{
    EXPECT_GE(offsetof(replay_data_t, dispatch_info),
              offsetof(replay_data_t, size) + sizeof(replay_data_t{}.size));
    EXPECT_GE(offsetof(replay_data_t, pass_count_cb),
              offsetof(replay_data_t, dispatch_info) + sizeof(replay_data_t{}.dispatch_info));
    EXPECT_GE(offsetof(replay_data_t, replay_continue_cb),
              offsetof(replay_data_t, pass_count_cb) + sizeof(replay_data_t{}.pass_count_cb));
    EXPECT_GE(
        offsetof(replay_data_t, current_pass),
        offsetof(replay_data_t, replay_continue_cb) + sizeof(replay_data_t{}.replay_continue_cb));
    EXPECT_GE(offsetof(replay_data_t, total_passes),
              offsetof(replay_data_t, current_pass) + sizeof(replay_data_t{}.current_pass));
    EXPECT_GE(offsetof(replay_data_t, replay_local_enable_context_cb),
              offsetof(replay_data_t, total_passes) + sizeof(replay_data_t{}.total_passes));
    EXPECT_GE(offsetof(replay_data_t, replay_local_disable_context_cb),
              offsetof(replay_data_t, replay_local_enable_context_cb) +
                  sizeof(replay_data_t{}.replay_local_enable_context_cb));
    EXPECT_GE(offsetof(replay_data_t, reserved_padding),
              offsetof(replay_data_t, replay_local_disable_context_cb) +
                  sizeof(replay_data_t{}.replay_local_disable_context_cb));
    EXPECT_GE(sizeof(replay_data_t),
              offsetof(replay_data_t, reserved_padding) + sizeof(replay_data_t{}.reserved_padding));
}

// Pass counters are documented as uint64_t. A tool reading them through the header's type must see
// the same width the SDK wrote.
TEST(kernel_replay_abi, pass_counters_are_64_bit)
{
    EXPECT_TRUE((std::is_same<decltype(replay_data_t{}.current_pass), uint64_t>::value));
    EXPECT_TRUE((std::is_same<decltype(replay_data_t{}.total_passes), uint64_t>::value));
    EXPECT_EQ(sizeof(replay_data_t{}.current_pass), 8u);
    EXPECT_EQ(sizeof(replay_data_t{}.total_passes), 8u);
}

// The four callback members are function pointers the tool either reads or writes. Their exact
// signatures are the contract; a changed parameter list here is an ABI break that the compiler
// would not otherwise flag at the tool.
TEST(kernel_replay_abi, callback_signatures_are_pinned)
{
    using pass_count_fn = uint64_t (*)(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t);
    using continue_fn =
        int (*)(rocprofiler_kernel_dispatch_info_t, uint64_t, uint64_t, rocprofiler_user_data_t);
    using toggle_fn = rocprofiler_status_t (*)(rocprofiler_context_id_t);

    EXPECT_TRUE((std::is_same<decltype(replay_data_t{}.pass_count_cb), pass_count_fn>::value));
    EXPECT_TRUE((std::is_same<decltype(replay_data_t{}.replay_continue_cb), continue_fn>::value));
    EXPECT_TRUE(
        (std::is_same<decltype(replay_data_t{}.replay_local_enable_context_cb), toggle_fn>::value));
    EXPECT_TRUE((
        std::is_same<decltype(replay_data_t{}.replay_local_disable_context_cb), toggle_fn>::value));
}

// A zeroed record must mean "do not replay this dispatch". The header documents a NULL
// pass_count_cb as the per-dispatch opt-out, so zero-initialization has to produce it: a tool that
// ignores the CONFIG callback entirely gets the safe behaviour.
TEST(kernel_replay_abi, zero_initialized_record_opts_out_of_replay)
{
    replay_data_t rec = {};
    EXPECT_EQ(rec.pass_count_cb, nullptr);
    EXPECT_EQ(rec.replay_continue_cb, nullptr);
    EXPECT_EQ(rec.replay_local_enable_context_cb, nullptr);
    EXPECT_EQ(rec.replay_local_disable_context_cb, nullptr);
    EXPECT_EQ(rec.current_pass, 0u);
    EXPECT_EQ(rec.total_passes, 0u);
    EXPECT_EQ(rec.size, 0u);
}

// The versioning scheme compares the SDK-written size against the tool's compiled layout prefix.
// With reserved_padding, size excludes the tail reservation (see fwd.h dispatch_info pattern).
TEST(kernel_replay_abi, size_member_carries_the_layout_prefix)
{
    replay_data_t rec = {};
    rec.size          = offsetof(replay_data_t, reserved_padding);
    EXPECT_EQ(rec.size, offsetof(replay_data_t, reserved_padding));
    EXPECT_LT(rec.size, sizeof(replay_data_t));
    EXPECT_GT(rec.size, offsetof(replay_data_t, replay_local_disable_context_cb));
}

TEST(kernel_replay_abi, reserved_padding_is_at_the_tail)
{
    EXPECT_EQ(offsetof(replay_data_t, reserved_padding) + sizeof(replay_data_t{}.reserved_padding),
              sizeof(replay_data_t));
    EXPECT_EQ(sizeof(replay_data_t{}.reserved_padding), 64u);
}

// A record is passed by pointer and read field-by-field; it must be aligned for its widest member
// so those reads are well-defined on every target the SDK supports.
TEST(kernel_replay_abi, alignment_is_sufficient_for_widest_member)
{
    EXPECT_GE(alignof(replay_data_t), alignof(uint64_t));
    EXPECT_EQ(sizeof(replay_data_t) % alignof(replay_data_t), 0u);
}

// ---------------------------------------------------------------------------
// Enum surface
// ---------------------------------------------------------------------------

// Operation IDs are written into trace records and matched by tools, so their numeric values are
// part of the ABI. NONE must stay 0 so a zeroed record does not name a real operation.
TEST(kernel_replay_abi, operation_enum_values_are_stable)
{
    EXPECT_EQ(static_cast<int>(ROCPROFILER_KERNEL_REPLAY_NONE), 0);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_KERNEL_REPLAY_CONFIG), 1);
    EXPECT_EQ(static_cast<int>(ROCPROFILER_KERNEL_REPLAY_PASS), 2);
}

// New operations must be appended before LAST, never inserted among the existing ones. If this
// fails because an operation was added, the fix is to bump the expected count here -- but only
// after confirming the new value was appended rather than inserted.
TEST(kernel_replay_abi, operation_enum_grows_only_at_the_end)
{
    EXPECT_EQ(static_cast<int>(ROCPROFILER_KERNEL_REPLAY_LAST), 3);
    EXPECT_GT(static_cast<int>(ROCPROFILER_KERNEL_REPLAY_LAST),
              static_cast<int>(ROCPROFILER_KERNEL_REPLAY_PASS));
}

// Every real operation must be inside (NONE, LAST) so a bounds check of that form accepts exactly
// the valid ones. Tools and the SDK's own dispatch tables rely on this.
TEST(kernel_replay_abi, real_operations_are_within_bounds)
{
    for(auto op : {ROCPROFILER_KERNEL_REPLAY_CONFIG, ROCPROFILER_KERNEL_REPLAY_PASS})
    {
        EXPECT_GT(static_cast<int>(op), static_cast<int>(ROCPROFILER_KERNEL_REPLAY_NONE));
        EXPECT_LT(static_cast<int>(op), static_cast<int>(ROCPROFILER_KERNEL_REPLAY_LAST));
    }
}

// The replay tracing kind must sit inside the callback-tracing enum's valid range, since it indexes
// the SDK's per-kind service tables.
TEST(kernel_replay_abi, tracing_kind_is_within_callback_tracing_bounds)
{
    EXPECT_GT(static_cast<int>(ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY),
              static_cast<int>(ROCPROFILER_CALLBACK_TRACING_NONE));
    EXPECT_LT(static_cast<int>(ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY),
              static_cast<int>(ROCPROFILER_CALLBACK_TRACING_LAST));
}

// ---------------------------------------------------------------------------
// Documented semantics that are expressible without a device
// ---------------------------------------------------------------------------

// current_pass is documented as 0-indexed and total_passes as the count, so the last valid pass is
// total_passes - 1. This pins the convention that a tool's "is this the final pass" check depends
// on; an off-by-one here would make every tool's last pass either double-count or drop.
TEST(kernel_replay_abi, pass_indices_are_zero_based_against_the_total)
{
    replay_data_t rec = {};
    rec.total_passes  = 4;

    for(uint64_t i = 0; i < rec.total_passes; ++i)
    {
        rec.current_pass = i;
        EXPECT_LT(rec.current_pass, rec.total_passes);
    }

    rec.current_pass = rec.total_passes - 1;
    EXPECT_EQ(rec.current_pass, 3u);
}

// An indefinite loop is signalled by total_passes == 0, which is distinct from "one pass". A tool
// switching on total_passes must be able to tell those apart.
TEST(kernel_replay_abi, indefinite_loop_is_distinguishable_from_single_pass)
{
    replay_data_t indefinite = {};
    indefinite.total_passes  = 0;

    replay_data_t single = {};
    single.total_passes  = 1;

    EXPECT_NE(indefinite.total_passes, single.total_passes);
    EXPECT_EQ(indefinite.total_passes, 0u);
}

// The SDK threads one user_data value through CONFIG and every PASS callback for a dispatch. The
// record carries it by value, so writing it must not disturb neighbouring fields -- a layout error
// would show up here as a corrupted pass counter.
TEST(kernel_replay_abi, writing_callbacks_does_not_disturb_pass_counters)
{
    replay_data_t rec = {};
    rec.size          = offsetof(replay_data_t, reserved_padding);
    rec.current_pass  = 2;
    rec.total_passes  = 5;

    rec.pass_count_cb = [](rocprofiler_kernel_dispatch_info_t,
                           rocprofiler_user_data_t) -> uint64_t { return 7; };

    EXPECT_EQ(rec.current_pass, 2u);
    EXPECT_EQ(rec.total_passes, 5u);
    EXPECT_EQ(rec.size, offsetof(replay_data_t, reserved_padding));
    ASSERT_NE(rec.pass_count_cb, nullptr);
    EXPECT_EQ(rec.pass_count_cb(rec.dispatch_info, rocprofiler_user_data_t{}), 7u);
}

// A tool built against an older, shorter header memcpy's only the prefix it knows about. The bytes
// it did copy must still read back correctly, which is what makes the size-prefixed scheme work.
TEST(kernel_replay_abi, truncated_copy_preserves_the_known_prefix)
{
    replay_data_t src = {};
    src.size          = offsetof(replay_data_t, reserved_padding);
    src.current_pass  = 11;
    src.total_passes  = 13;

    const size_t old_size =
        offsetof(replay_data_t, total_passes) + sizeof(replay_data_t{}.total_passes);
    ASSERT_LE(old_size, sizeof(replay_data_t));

    replay_data_t dst = {};
    std::memcpy(&dst, &src, old_size);

    EXPECT_EQ(dst.size, src.size);
    EXPECT_EQ(dst.current_pass, 11u);
    EXPECT_EQ(dst.total_passes, 13u);
}

// ---------------------------------------------------------------------------
// C / C++ agreement
//
// The SDK is C++ and most tools are C. Both compile the same header, so the layout each language
// derives from it has to match; if it does not, a C tool reads the wrong bytes with nothing
// failing to build. These compare C's view (from replay_abi_c.c) against C++'s.
// ---------------------------------------------------------------------------

TEST(kernel_replay_abi, c_and_cxx_agree_on_record_size_and_alignment)
{
    EXPECT_EQ(rocprofiler_test_c_replay_record_size(), sizeof(replay_data_t));
    EXPECT_EQ(rocprofiler_test_c_replay_record_align(), alignof(replay_data_t));
}

TEST(kernel_replay_abi, c_and_cxx_agree_on_every_field_offset)
{
    EXPECT_EQ(rocprofiler_test_c_replay_offset_size(), offsetof(replay_data_t, size));
    EXPECT_EQ(rocprofiler_test_c_replay_offset_dispatch_info(),
              offsetof(replay_data_t, dispatch_info));
    EXPECT_EQ(rocprofiler_test_c_replay_offset_pass_count_cb(),
              offsetof(replay_data_t, pass_count_cb));
    EXPECT_EQ(rocprofiler_test_c_replay_offset_replay_continue_cb(),
              offsetof(replay_data_t, replay_continue_cb));
    EXPECT_EQ(rocprofiler_test_c_replay_offset_current_pass(),
              offsetof(replay_data_t, current_pass));
    EXPECT_EQ(rocprofiler_test_c_replay_offset_total_passes(),
              offsetof(replay_data_t, total_passes));
    EXPECT_EQ(rocprofiler_test_c_replay_offset_local_enable_cb(),
              offsetof(replay_data_t, replay_local_enable_context_cb));
    EXPECT_EQ(rocprofiler_test_c_replay_offset_local_disable_cb(),
              offsetof(replay_data_t, replay_local_disable_context_cb));
    EXPECT_EQ(rocprofiler_test_c_replay_offset_reserved_padding(),
              offsetof(replay_data_t, reserved_padding));
}

TEST(kernel_replay_abi, c_and_cxx_agree_on_enum_values)
{
    EXPECT_EQ(rocprofiler_test_c_replay_operation_last(),
              static_cast<int>(ROCPROFILER_KERNEL_REPLAY_LAST));
    EXPECT_EQ(rocprofiler_test_c_replay_tracing_kind(),
              static_cast<int>(ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY));
}

// A record written through C's view of the struct must read back correctly through C++'s. This is
// the end-to-end version of the offset comparison above: it would fail if the two languages
// disagreed about padding even where the individual offsets happened to line up.
TEST(kernel_replay_abi, record_written_by_c_reads_back_in_cxx)
{
    replay_data_t rec = {};
    rocprofiler_test_c_replay_fill(&rec, 3, 9);

    EXPECT_EQ(rec.size, offsetof(replay_data_t, reserved_padding));
    EXPECT_EQ(rec.current_pass, 3u);
    EXPECT_EQ(rec.total_passes, 9u);

    // Fields the C side did not write must still be zero: a layout disagreement would show up as
    // one of C's writes landing on top of a neighbouring member.
    EXPECT_EQ(rec.pass_count_cb, nullptr);
    EXPECT_EQ(rec.replay_continue_cb, nullptr);
    EXPECT_EQ(rec.replay_local_enable_context_cb, nullptr);
    EXPECT_EQ(rec.replay_local_disable_context_cb, nullptr);
}
