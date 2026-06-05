/* Copyright (c) 2026 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

/* OS-agnostic unit tests for src/utils.h utility helpers.  Linked
   against amd-dbgapi-internal (the test seam) so we can reach the
   detail namespace and library-internal types directly.

   Tests deliberately avoid:
     - paths that go through fatal_error (narrow overflow, monotonic
       counter wrap-around, alignment_to_mask on non-power-of-two) -
       those terminate the process and would need death tests, which
       are out of scope for this commit;
     - notifier_t::create() - implementation lives in the per-OS TUs
       (utils_linux.cpp / utils_windows.cpp); covered in the per-OS
       test files alongside this one.  */

#include "utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace amd::dbgapi;
using namespace amd::dbgapi::utils;

/* ------------------------------------------------------------------ */
/* narrow                                                              */
/* ------------------------------------------------------------------ */

TEST (UtilsNarrow, IdentityConversion)
{
  EXPECT_EQ (narrow<uint32_t> (uint32_t{ 42 }), 42u);
  EXPECT_EQ (narrow<int32_t> (int32_t{ -7 }), -7);
}

TEST (UtilsNarrow, WideningSignedToSigned)
{
  EXPECT_EQ (narrow<int64_t> (int32_t{ -1 }), -1);
}

TEST (UtilsNarrow, FitsInDestination)
{
  EXPECT_EQ (narrow<uint8_t> (uint32_t{ 255 }), 255u);
  EXPECT_EQ (narrow<int8_t> (int32_t{ -128 }), -128);
}

TEST (UtilsNarrow, NarrowAssign)
{
  uint8_t out = 0;
  narrow_assign (out, uint32_t{ 200 });
  EXPECT_EQ (out, 200u);
}

/* ------------------------------------------------------------------ */
/* bit_mask / bit_count / bit_extract / zero_extend / sign_extend      */
/* ------------------------------------------------------------------ */

TEST (UtilsBitMask, BasicRanges)
{
  EXPECT_EQ (bit_mask<uint64_t> (0, 0), 0x1ull);
  EXPECT_EQ (bit_mask<uint64_t> (0, 7), 0xFFull);
  EXPECT_EQ (bit_mask<uint64_t> (4, 7), 0xF0ull);
  EXPECT_EQ (bit_mask<uint64_t> (0, 63), ~uint64_t{ 0 });
}

TEST (UtilsBitCount, KnownValues)
{
  EXPECT_EQ (bit_count<uint32_t> (0), 0);
  EXPECT_EQ (bit_count<uint32_t> (1), 1);
  EXPECT_EQ (bit_count<uint32_t> (0xF), 4);
  EXPECT_EQ (bit_count<uint32_t> (0xFFFFFFFFu), 32);
  EXPECT_EQ (bit_count<uint64_t> (~uint64_t{ 0 }), 64);
}

TEST (UtilsTrailingZeroes, KnownValues)
{
  EXPECT_EQ (trailing_zeroes_count<uint32_t> (0u), 32);
  EXPECT_EQ (trailing_zeroes_count<uint32_t> (1u), 0);
  EXPECT_EQ (trailing_zeroes_count<uint32_t> (8u), 3);
  EXPECT_EQ (trailing_zeroes_count<uint32_t> (1u << 31), 31);
}

TEST (UtilsBitExtract, KnownValues)
{
  EXPECT_EQ (bit_extract (uint32_t{ 0xDEADBEEFu }, 0, 7), 0xEFu);
  EXPECT_EQ (bit_extract (uint32_t{ 0xDEADBEEFu }, 16, 31), 0xDEADu);
  EXPECT_EQ (bit_extract (uint32_t{ 0xABCDu }, 4, 7), 0xCu);
}

TEST (UtilsExtend, ZeroAndSign)
{
  EXPECT_EQ (zero_extend (uint8_t{ 0xFF }, 4), uint8_t{ 0x0F });
  EXPECT_EQ (sign_extend (uint8_t{ 0x08 }, 4), int8_t{ -8 });
  EXPECT_EQ (sign_extend (uint8_t{ 0x07 }, 4), int8_t{ 7 });
}

TEST (UtilsSplit, Halves)
{
  auto [lo, hi] = split (0xDEADBEEFCAFEBABEull);
  EXPECT_EQ (lo, 0xCAFEBABEu);
  EXPECT_EQ (hi, 0xDEADBEEFu);
}

/* ------------------------------------------------------------------ */
/* power-of-two and alignment                                          */
/* ------------------------------------------------------------------ */

TEST (UtilsPowerOfTwo, IsAndNext)
{
  EXPECT_FALSE (is_power_of_two<uint32_t> (0));
  EXPECT_TRUE (is_power_of_two<uint32_t> (1));
  EXPECT_TRUE (is_power_of_two<uint32_t> (1024));
  EXPECT_FALSE (is_power_of_two<uint32_t> (1023));

  EXPECT_EQ (next_power_of_two<uint32_t> (0u), 1u);
  EXPECT_EQ (next_power_of_two<uint32_t> (1u), 1u);
  EXPECT_EQ (next_power_of_two<uint32_t> (3u), 4u);
  EXPECT_EQ (next_power_of_two<uint32_t> (1000u), 1024u);
}

TEST (UtilsAlignment, MaskRoundtrip)
{
  EXPECT_EQ (alignment_to_mask<uint64_t> (8), ~uint64_t{ 7 });
  EXPECT_EQ (mask_to_alignment<uint64_t> (~uint64_t{ 7 }), 8u);
}

TEST (UtilsAlignment, AlignUpDownAndIs)
{
  EXPECT_EQ (align_down (uint64_t{ 17 }, 8u), 16u);
  EXPECT_EQ (align_up (uint64_t{ 17 }, 8u), 24u);
  EXPECT_EQ (align_up (uint64_t{ 16 }, 8u), 16u);
  EXPECT_TRUE (is_aligned (uint64_t{ 16 }, 8u));
  EXPECT_FALSE (is_aligned (uint64_t{ 17 }, 8u));
}

/* ------------------------------------------------------------------ */
/* scope_exit / scope_success / scope_fail                             */
/* ------------------------------------------------------------------ */

TEST (UtilsScopeExit, FiresOnScopeEnd)
{
  bool fired = false;
  {
    auto guard = make_scope_exit ([&] { fired = true; });
    (void)guard;
  }
  EXPECT_TRUE (fired);
}

TEST (UtilsScopeExit, DoesNotFireAfterRelease)
{
  bool fired = false;
  {
    auto guard = make_scope_exit ([&] { fired = true; });
    guard.release ();
  }
  EXPECT_FALSE (fired);
}

TEST (UtilsScopeSuccess, FiresWhenNoException)
{
  bool fired = false;
  {
    auto guard = make_scope_success ([&] { fired = true; });
    (void)guard;
  }
  EXPECT_TRUE (fired);
}

TEST (UtilsScopeSuccess, DoesNotFireWhenException)
{
  bool fired = false;
  try
    {
      auto guard = make_scope_success ([&] { fired = true; });
      (void)guard;
      throw std::runtime_error ("boom");
    }
  catch (const std::runtime_error &)
    {
    }
  EXPECT_FALSE (fired);
}

TEST (UtilsScopeFail, FiresOnlyWhenException)
{
  bool fired_on_exit = false;
  {
    auto guard = make_scope_fail ([&] { fired_on_exit = true; });
    (void)guard;
  }
  EXPECT_FALSE (fired_on_exit);

  bool fired_on_throw = false;
  try
    {
      auto guard = make_scope_fail ([&] { fired_on_throw = true; });
      (void)guard;
      throw std::runtime_error ("boom");
    }
  catch (const std::runtime_error &)
    {
    }
  EXPECT_TRUE (fired_on_throw);
}

/* ------------------------------------------------------------------ */
/* unique_resource_t                                                   */
/* ------------------------------------------------------------------ */

TEST (UtilsUniqueResource, DeleterCalledOnce)
{
  int delete_count = 0;
  auto deleter = [&] (int) { ++delete_count; };
  {
    unique_resource_t<int, decltype (deleter)> r{ 42, deleter };
    EXPECT_EQ (r.get (), 42);
  }
  EXPECT_EQ (delete_count, 1);
}

TEST (UtilsUniqueResource, ReleasePreventsDelete)
{
  int delete_count = 0;
  auto deleter = [&] (int) { ++delete_count; };
  {
    unique_resource_t<int, decltype (deleter)> r{ 42, deleter };
    r.release ();
  }
  EXPECT_EQ (delete_count, 0);
}

TEST (UtilsUniqueResource, MoveTransfersOwnership)
{
  int delete_count = 0;
  auto deleter = [&] (int) { ++delete_count; };
  {
    unique_resource_t<int, decltype (deleter)> a{ 1, deleter };
    unique_resource_t<int, decltype (deleter)> b{ std::move (a) };
    EXPECT_EQ (b.get (), 1);
  }
  EXPECT_EQ (delete_count, 1);
}

/* ------------------------------------------------------------------ */
/* doubly_linked_list_t                                                */
/* ------------------------------------------------------------------ */

namespace utils_test_ns
{

struct node_t : public amd::dbgapi::utils::detail::doubly_linked_entry_t<node_t, void>
{
  int value{ 0 };
  explicit node_t (int v) : value (v) {}
};

} /* namespace utils_test_ns */

TEST (UtilsDoublyLinkedList, EmptyAtConstruction)
{
  doubly_linked_list_t<utils_test_ns::node_t> list;
  EXPECT_TRUE (list.empty ());
  EXPECT_EQ (list.size (), 0u);
  EXPECT_TRUE (list.begin () == list.end ());
}

TEST (UtilsDoublyLinkedList, InsertAndIterate)
{
  using utils_test_ns::node_t;
  doubly_linked_list_t<node_t> list;
  node_t a{ 1 }, b{ 2 }, c{ 3 };
  list.insert (a);
  list.insert (b);
  list.insert (c);

  EXPECT_EQ (list.size (), 3u);

  /* insert pushes to the head, so iteration order is c, b, a.  */
  std::vector<int> seen;
  for (auto &n : list)
    seen.push_back (n.value);
  EXPECT_THAT (seen, ::testing::ElementsAre (3, 2, 1));
}

TEST (UtilsDoublyLinkedList, Remove)
{
  using utils_test_ns::node_t;
  doubly_linked_list_t<node_t> list;
  node_t a{ 10 }, b{ 20 };
  list.insert (a);
  list.insert (b);
  EXPECT_EQ (list.size (), 2u);

  list.remove (a);
  EXPECT_EQ (list.size (), 1u);
  EXPECT_EQ (list.begin ()->value, 20);

  list.remove (b);
  EXPECT_TRUE (list.empty ());
}

/* ------------------------------------------------------------------ */
/* monotonic_counter_t                                                 */
/* ------------------------------------------------------------------ */

TEST (UtilsMonotonicCounter, IncrementsUniquely)
{
  monotonic_counter_t<uint32_t> counter;
  EXPECT_EQ (counter (), 0u);
  EXPECT_EQ (counter (), 1u);
  EXPECT_EQ (counter (), 2u);
}

TEST (UtilsMonotonicCounter, ResetRestartsAtInitial)
{
  monotonic_counter_t<uint32_t, /* InitialValue */ 100> counter;
  EXPECT_EQ (counter (), 100u);
  EXPECT_EQ (counter (), 101u);
  counter.reset ();
  EXPECT_EQ (counter (), 100u);
}

/* ------------------------------------------------------------------ */
/* string_printf / human_readable_size                                 */
/* ------------------------------------------------------------------ */

TEST (UtilsStringPrintf, Formats)
{
  EXPECT_EQ (string_printf ("hello %s %d", "world", 7), "hello world 7");
  EXPECT_EQ (string_printf ("%08x", 0xABu), "000000ab");
}

/* Empty output (via a no-op conversion) must round-trip to "" — the
   two-pass vsnprintf path must not over-allocate or trip on size==0.
   Using "%s" with "" instead of a literal empty format to avoid the
   gcc -Wformat-zero-length diagnostic.  */
TEST (UtilsStringPrintf, EmptyOutputYieldsEmptyString)
{
  EXPECT_EQ (string_printf ("%s", ""), "");
  EXPECT_EQ (string_printf ("%s", "").size (), 0u);
}

/* No format conversions: the input must pass through unchanged.  */
TEST (UtilsStringPrintf, LiteralPassthrough)
{
  EXPECT_EQ (string_printf ("plain text with no conversions"),
             "plain text with no conversions");
}

/* '%' must be escapable.  */
TEST (UtilsStringPrintf, PercentEscape)
{
  EXPECT_EQ (string_printf ("100%%"), "100%");
  EXPECT_EQ (string_printf ("%d%%", 42), "42%");
}

/* Embedded NUL in the *output* is the printf contract: %c with '\0'
   yields a single-byte string whose only byte is 0.  Verifies
   string_vprintf sizes from vsnprintf's return value, not strlen().  */
TEST (UtilsStringPrintf, EmbeddedNulFromFormatPreservesLength)
{
  std::string s = string_printf ("a%cb", '\0');
  ASSERT_EQ (s.size (), 3u);
  EXPECT_EQ (s[0], 'a');
  EXPECT_EQ (s[1], '\0');
  EXPECT_EQ (s[2], 'b');
}

/* Force the two-pass vsnprintf into a large allocation to confirm
   the size returned by the first pass is honored on the second.  */
TEST (UtilsStringPrintf, LargeOutputRoundTrips)
{
  std::string big (8192, 'x');
  std::string out = string_printf ("%s", big.c_str ());
  EXPECT_EQ (out, big);
  EXPECT_EQ (out.size (), 8192u);
}

/* Multiple positional conversions to confirm the va_list copy in
   string_vprintf doesn't consume the args twice.  */
TEST (UtilsStringPrintf, MultipleConversions)
{
  EXPECT_EQ (string_printf ("%d-%d-%d-%s", 1, 2, 3, "end"), "1-2-3-end");
}

TEST (UtilsHumanReadableSize, KnownValues)
{
  /* The exact human-readable representation is implementation-defined
     (current impl emits "1.0K", "1.0M" - single-letter unit suffix).
     We assert the magnitude and the unit-letter only.  */
  EXPECT_THAT (human_readable_size (1024), ::testing::HasSubstr ("1.0"));
  EXPECT_THAT (human_readable_size (1024), ::testing::HasSubstr ("K"));
  EXPECT_THAT (human_readable_size (1024 * 1024), ::testing::HasSubstr ("1.0"));
  EXPECT_THAT (human_readable_size (1024 * 1024), ::testing::HasSubstr ("M"));
}

/* 0 bytes: < KiB branch, printed as a raw integer with no unit
   suffix.  Guards against an off-by-one that would otherwise pull
   tiny sizes into the "1.0K" branch.  */
TEST (UtilsHumanReadableSize, ZeroIsRawInteger)
{
  EXPECT_EQ (human_readable_size (0), "0");
}

/* KiB - 1 is still the raw-integer branch.  */
TEST (UtilsHumanReadableSize, JustBelowKibIsRawInteger)
{
  EXPECT_EQ (human_readable_size (1023), "1023");
}

/* MiB - 1 must round into the K-suffix branch, not roll over to M.  */
TEST (UtilsHumanReadableSize, JustBelowMibUsesKSuffix)
{
  std::string s = human_readable_size ((1024 * 1024) - 1);
  EXPECT_THAT (s, ::testing::HasSubstr ("K"));
  EXPECT_THAT (s, ::testing::Not (::testing::HasSubstr ("M")));
}

/* 1 GiB lands in the G-suffix branch.  */
TEST (UtilsHumanReadableSize, OneGibUsesGSuffix)
{
  std::string s = human_readable_size (1024ull * 1024ull * 1024ull);
  EXPECT_THAT (s, ::testing::HasSubstr ("1.0"));
  EXPECT_THAT (s, ::testing::HasSubstr ("G"));
}

/* ------------------------------------------------------------------ */
/* is_flag bitwise operators (use a local enum that is_flag opts in)   */
/* ------------------------------------------------------------------ */

namespace test_flags_ns
{

enum class my_flag_t : uint32_t
{
  none = 0,
  a = 1u << 0,
  b = 1u << 1,
  c = 1u << 2,
};

} /* namespace test_flags_ns */

template <>
struct amd::dbgapi::is_flag<test_flags_ns::my_flag_t> : std::true_type
{
};

TEST (UtilsFlag, BitwiseOps)
{
  using test_flags_ns::my_flag_t;

  my_flag_t f = my_flag_t::a | my_flag_t::b;
  EXPECT_FALSE (!f);
  EXPECT_TRUE (!my_flag_t::none);
  EXPECT_TRUE ((f & my_flag_t::a) == static_cast<uint32_t> (my_flag_t::a));
  EXPECT_TRUE ((f & my_flag_t::c) == 0u);

  f |= my_flag_t::c;
  EXPECT_TRUE ((f & my_flag_t::c) != 0u);

  f &= ~my_flag_t::a;
  EXPECT_TRUE ((f & my_flag_t::a) == 0u);
}
