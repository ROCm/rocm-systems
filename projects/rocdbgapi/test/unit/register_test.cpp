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

/* Unit tests for src/register.{h,cpp}.

   Two surfaces are exercised here:

   1) amdgpu_regnum_t enum + free operators (purely value-level, no
      runtime state):
       - The interval invariants of the enum (first_vgpr < last_vgpr <
         first_sgpr < ... < last_regnum).
       - operator-(regnum, regnum) returning a signed distance.
       - operator+/- with amdgpu_regdiff_t.
       - operator++ pre and post forms.
       - is_pseudo_register() boundaries.

   2) register_class_t.  Constructible from architecture_t&; we use a
      real architecture from architecture_t::s_architecture_map
      (populated at static-init) so no amd_dbgapi_initialize() is
      needed.  Covered: name() round-trip, add_registers(),
      remove_registers() (including the split case where a removal
      cuts an existing range), contains(), register_set(), and
      get_info() error contract.

   Also confirms that the architecture's pre-populated register
   classes ("scalar", "vector", "trap", "system", "general") created
   in architecture.cpp:2608+ are reachable through arch.range<>() /
   arch.count<>() without any process state.

   Out of scope here:
     - read_register / write_register / is_register_available (need
       wave_t).
     - DWARF register mapping (amdgpu_regnum_to_dwarf_register /
       dwarf_register_to_amdgpu_regnum are in an anonymous namespace;
       indirect coverage via the public C API requires
       amd_dbgapi_initialize and is deferred to feature tests).  */

#include "architecture.h"
#include "exception.h"
#include "register.h"

#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <set>
#include <string>

using namespace amd::dbgapi;

namespace
{

/* Only s0 and s127 are individual enumerators in amdgpu_regnum_t;
   intermediate sgprs (s1, s2, ..., s31, s64, ...) are computed as
   s0 + N.  */
constexpr amdgpu_regnum_t
sgpr (amdgpu_regdiff_t n)
{
  return amdgpu_regnum_t::s0 + n;
}

} /* namespace */

/* ------------------------------------------------------------------ */
/* amdgpu_regnum_t enum invariants                                     */
/* ------------------------------------------------------------------ */

TEST (RegnumEnum, IntervalOrdering)
{
  /* The enum carves out non-overlapping ranges and each tier's
     last < next tier's first.  The exact numeric values change as
     new architectures land; only the ordering is contract.  */
  EXPECT_LT (amdgpu_regnum_t::first_vgpr, amdgpu_regnum_t::last_vgpr);
  EXPECT_LT (amdgpu_regnum_t::last_vgpr, amdgpu_regnum_t::first_sgpr);
  EXPECT_LT (amdgpu_regnum_t::first_sgpr, amdgpu_regnum_t::last_sgpr);
  EXPECT_LT (amdgpu_regnum_t::last_sgpr, amdgpu_regnum_t::first_hwreg);
  EXPECT_LT (amdgpu_regnum_t::first_hwreg, amdgpu_regnum_t::last_hwreg);
  EXPECT_LT (amdgpu_regnum_t::last_hwreg, amdgpu_regnum_t::first_ttmp);
  EXPECT_LT (amdgpu_regnum_t::first_ttmp, amdgpu_regnum_t::last_ttmp);
  EXPECT_LT (amdgpu_regnum_t::last_ttmp, amdgpu_regnum_t::first_aliased);
  EXPECT_LT (amdgpu_regnum_t::first_aliased, amdgpu_regnum_t::last_aliased);
  EXPECT_LT (amdgpu_regnum_t::last_aliased, amdgpu_regnum_t::first_pseudo);
  EXPECT_LE (amdgpu_regnum_t::first_pseudo, amdgpu_regnum_t::last_pseudo);
  EXPECT_EQ (amdgpu_regnum_t::last_pseudo, amdgpu_regnum_t::last_regnum);
}

TEST (RegnumEnum, BankSizesMatchExpectedCounts)
{
  /* Documented in the enum: 1024 wave32 vgprs, 256 wave64 vgprs +
     256 wave64 accvgprs, 256 wave32 accvgprs, 128 sgprs.  */
  EXPECT_EQ (amdgpu_regnum_t::v1023_32 - amdgpu_regnum_t::v0_32 + 1, 1024);
  EXPECT_EQ (amdgpu_regnum_t::v255_64 - amdgpu_regnum_t::v0_64 + 1, 256);
  EXPECT_EQ (amdgpu_regnum_t::a255_64 - amdgpu_regnum_t::a0_64 + 1, 256);
  EXPECT_EQ (amdgpu_regnum_t::a255_32 - amdgpu_regnum_t::a0_32 + 1, 256);
  EXPECT_EQ (amdgpu_regnum_t::s127 - amdgpu_regnum_t::s0 + 1, 128);
}

TEST (RegnumEnum, ArithmeticAndIncrement)
{
  /* operator- yields a signed distance.  */
  amdgpu_regnum_t base = amdgpu_regnum_t::s0;
  amdgpu_regnum_t five_in = base + amdgpu_regdiff_t{ 5 };
  EXPECT_EQ (five_in - base, 5);
  EXPECT_EQ (base - five_in, -5);

  /* operator-(amdgpu_regnum_t, amdgpu_regdiff_t).  */
  EXPECT_EQ (five_in - amdgpu_regdiff_t{ 5 }, base);

  /* Pre-increment returns the incremented value.  */
  amdgpu_regnum_t r = amdgpu_regnum_t::s0;
  amdgpu_regnum_t &pre = ++r;
  EXPECT_EQ (pre, amdgpu_regnum_t::s0 + amdgpu_regdiff_t{ 1 });
  EXPECT_EQ (&pre, &r);

  /* Post-increment returns the previous value.  */
  amdgpu_regnum_t r2 = amdgpu_regnum_t::s0;
  amdgpu_regnum_t prev = r2++;
  EXPECT_EQ (prev, amdgpu_regnum_t::s0);
  EXPECT_EQ (r2, amdgpu_regnum_t::s0 + amdgpu_regdiff_t{ 1 });
}

TEST (RegnumEnum, IsPseudoRegister)
{
  EXPECT_FALSE (is_pseudo_register (amdgpu_regnum_t::v0_64));
  EXPECT_FALSE (is_pseudo_register (amdgpu_regnum_t::s0));
  EXPECT_FALSE (is_pseudo_register (amdgpu_regnum_t::pc));
  EXPECT_FALSE (is_pseudo_register (amdgpu_regnum_t::last_aliased));

  EXPECT_TRUE (is_pseudo_register (amdgpu_regnum_t::first_pseudo));
  EXPECT_TRUE (is_pseudo_register (amdgpu_regnum_t::pseudo_status));
  EXPECT_TRUE (is_pseudo_register (amdgpu_regnum_t::null));
  EXPECT_TRUE (is_pseudo_register (amdgpu_regnum_t::last_pseudo));
}

/* ------------------------------------------------------------------ */
/* register_class_t — backed by a real architecture                    */
/* ------------------------------------------------------------------ */

namespace
{

/* Pick gfx942 as the representative architecture.  It's stable, has
   the standard register set, and is what the other unit tests use.  */
const architecture_t *
gfx942 ()
{
  return architecture_t::find (std::string{ "gfx942" });
}

} /* namespace */

TEST (RegisterClassArchPrePopulated, GfxArchHasStandardClasses)
{
  const architecture_t *arch = gfx942 ();
  ASSERT_NE (arch, nullptr);

  /* architecture.cpp populates these five at construction time.  */
  std::set<std::string> names;
  for (auto &&rc : arch->range<register_class_t> ())
    names.insert (rc.name ());

  EXPECT_TRUE (names.count ("scalar")) << "missing 'scalar' class";
  EXPECT_TRUE (names.count ("vector")) << "missing 'vector' class";
  EXPECT_TRUE (names.count ("trap")) << "missing 'trap' class";
  EXPECT_TRUE (names.count ("system")) << "missing 'system' class";
  EXPECT_TRUE (names.count ("general")) << "missing 'general' class";

  EXPECT_GE (arch->count<register_class_t> (), 5u);
}

TEST (RegisterClassArchPrePopulated, ScalarClassContainsSgprs)
{
  const architecture_t *arch = gfx942 ();
  ASSERT_NE (arch, nullptr);

  const register_class_t *scalar = nullptr;
  for (auto &&rc : arch->range<register_class_t> ())
    if (rc.name () == "scalar")
      {
        scalar = &rc;
        break;
      }
  ASSERT_NE (scalar, nullptr);

  /* On gfx9 the "scalar" class covers [s0..s101] (scalar_register_count
     is 102, not the full 128).  s127 is intentionally outside.  pc is
     an aliased reg, also outside.  */
  EXPECT_TRUE (scalar->contains (amdgpu_regnum_t::s0));
  EXPECT_TRUE (scalar->contains (sgpr (101)));
  EXPECT_FALSE (scalar->contains (sgpr (102)));
  EXPECT_FALSE (scalar->contains (amdgpu_regnum_t::s127));
  EXPECT_FALSE (scalar->contains (amdgpu_regnum_t::pc));
}

/* The following tests build their own fresh register_class_t against
   the architecture (without touching the architecture's pre-populated
   classes) so add_registers / remove_registers state is isolated.  */

TEST (RegisterClassBookkeeping, AddRegistersAcceptsAndRejectsOverlap)
{
  const architecture_t *arch = gfx942 ();
  ASSERT_NE (arch, nullptr);

  register_class_t rc (amd_dbgapi_register_class_id_t{ 0x100 }, *arch,
                       "test");

  /* Insert two disjoint ranges.  */
  EXPECT_TRUE (rc.add_registers (amdgpu_regnum_t::s0, sgpr (7)));
  EXPECT_TRUE (
    rc.add_registers (sgpr (16), amdgpu_regnum_t::s127));

  /* A range with the same 'first' as an existing one is rejected by
     std::map::emplace.  */
  EXPECT_FALSE (
    rc.add_registers (amdgpu_regnum_t::s0, amdgpu_regnum_t::s127));
}

TEST (RegisterClassBookkeeping, ContainsRespectsRanges)
{
  const architecture_t *arch = gfx942 ();
  ASSERT_NE (arch, nullptr);

  register_class_t rc (amd_dbgapi_register_class_id_t{ 0x101 }, *arch,
                       "test");
  rc.add_registers (amdgpu_regnum_t::s0, sgpr (7));
  rc.add_registers (sgpr (64), sgpr (64));

  EXPECT_TRUE (rc.contains (amdgpu_regnum_t::s0));
  EXPECT_TRUE (rc.contains (sgpr (7)));
  EXPECT_TRUE (rc.contains (sgpr (64)));

  EXPECT_FALSE (rc.contains (sgpr (8)));
  EXPECT_FALSE (rc.contains (sgpr (63)));
  EXPECT_FALSE (rc.contains (sgpr (65)));
}

TEST (RegisterClassBookkeeping, RegisterSetEnumeratesAllRegisters)
{
  const architecture_t *arch = gfx942 ();
  ASSERT_NE (arch, nullptr);

  register_class_t rc (amd_dbgapi_register_class_id_t{ 0x102 }, *arch,
                       "test");
  rc.add_registers (amdgpu_regnum_t::s0, sgpr (2));
  rc.add_registers (sgpr (10), sgpr (11));

  auto regs = rc.register_set ();
  EXPECT_EQ (regs.size (), 5u);
  EXPECT_TRUE (regs.count (amdgpu_regnum_t::s0));
  EXPECT_TRUE (regs.count (sgpr (1)));
  EXPECT_TRUE (regs.count (sgpr (2)));
  EXPECT_TRUE (regs.count (sgpr (10)));
  EXPECT_TRUE (regs.count (sgpr (11)));
  EXPECT_FALSE (regs.count (sgpr (3)));
}

TEST (RegisterClassBookkeeping, RemoveRegistersSplitsRange)
{
  const architecture_t *arch = gfx942 ();
  ASSERT_NE (arch, nullptr);

  register_class_t rc (amd_dbgapi_register_class_id_t{ 0x103 }, *arch,
                       "test");
  rc.add_registers (amdgpu_regnum_t::s0, sgpr (31));

  /* Carve out the middle [s10, s20].  The implementation splits the
     existing range into [s0, s9] and [s21, s31].  */
  EXPECT_TRUE (
    rc.remove_registers (sgpr (10), sgpr (20)));

  EXPECT_TRUE (rc.contains (amdgpu_regnum_t::s0));
  EXPECT_TRUE (rc.contains (sgpr (9)));
  EXPECT_FALSE (rc.contains (sgpr (10)));
  EXPECT_FALSE (rc.contains (sgpr (15)));
  EXPECT_FALSE (rc.contains (sgpr (20)));
  EXPECT_TRUE (rc.contains (sgpr (21)));
  EXPECT_TRUE (rc.contains (sgpr (31)));

  auto regs = rc.register_set ();
  /* [s0..s9] is 10 regs, [s21..s31] is 11 regs.  */
  EXPECT_EQ (regs.size (), 21u);
}

TEST (RegisterClassBookkeeping, RemoveRegistersOutsideRangeReturnsFalse)
{
  const architecture_t *arch = gfx942 ();
  ASSERT_NE (arch, nullptr);

  register_class_t rc (amd_dbgapi_register_class_id_t{ 0x104 }, *arch,
                       "test");
  rc.add_registers (amdgpu_regnum_t::s0, sgpr (7));

  /* Try to remove a range entirely outside what we added.  */
  EXPECT_FALSE (
    rc.remove_registers (sgpr (10), sgpr (11)));
}

TEST (RegisterClassGetInfo, BogusQueryThrows)
{
  const architecture_t *arch = gfx942 ();
  ASSERT_NE (arch, nullptr);

  register_class_t rc (amd_dbgapi_register_class_id_t{ 0x105 }, *arch,
                       "test");
  auto bogus = static_cast<amd_dbgapi_register_class_info_t> (0xdeadbeefu);
  EXPECT_THROW (rc.get_info (bogus, 0, nullptr), api_error_t);
}

TEST (RegisterClassGetInfo, ArchitectureIdRoundTrips)
{
  const architecture_t *arch = gfx942 ();
  ASSERT_NE (arch, nullptr);

  register_class_t rc (amd_dbgapi_register_class_id_t{ 0x106 }, *arch,
                       "test");
  amd_dbgapi_architecture_id_t out{ 0 };
  rc.get_info (AMD_DBGAPI_REGISTER_CLASS_INFO_ARCHITECTURE, sizeof (out),
               &out);
  EXPECT_EQ (out.handle, arch->id ().handle);
}

/* C25 negative-test sweep: value_size and enum edge cases.              */

TEST (RegisterClassGetInfo, ValueSizeZeroThrowsCompatibility)
{
  const architecture_t *arch = gfx942 ();
  ASSERT_NE (arch, nullptr);
  register_class_t rc (amd_dbgapi_register_class_id_t{ 0x200 }, *arch, "test");
  amd_dbgapi_architecture_id_t out{};
  try
    {
      rc.get_info (AMD_DBGAPI_REGISTER_CLASS_INFO_ARCHITECTURE, 0, &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (RegisterClassGetInfo, ValueSizeMaxThrowsCompatibility)
{
  const architecture_t *arch = gfx942 ();
  ASSERT_NE (arch, nullptr);
  register_class_t rc (amd_dbgapi_register_class_id_t{ 0x201 }, *arch, "test");
  amd_dbgapi_architecture_id_t out{};
  try
    {
      rc.get_info (AMD_DBGAPI_REGISTER_CLASS_INFO_ARCHITECTURE, SIZE_MAX,
                   &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (RegisterClassGetInfo, ValueSizeTooLargeByOneThrowsCompatibility)
{
  const architecture_t *arch = gfx942 ();
  ASSERT_NE (arch, nullptr);
  register_class_t rc (amd_dbgapi_register_class_id_t{ 0x202 }, *arch, "test");
  uint8_t out[sizeof (amd_dbgapi_architecture_id_t) + 1]{};
  try
    {
      rc.get_info (AMD_DBGAPI_REGISTER_CLASS_INFO_ARCHITECTURE,
                   sizeof (amd_dbgapi_architecture_id_t) + 1, out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (RegisterClassGetInfo, NegativeEnumThrowsInvalidArgument)
{
  const architecture_t *arch = gfx942 ();
  ASSERT_NE (arch, nullptr);
  register_class_t rc (amd_dbgapi_register_class_id_t{ 0x203 }, *arch, "test");
  amd_dbgapi_architecture_id_t out{};
  try
    {
      rc.get_info (static_cast<amd_dbgapi_register_class_info_t> (-1),
                   sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (RegisterClassGetInfo, IntMaxEnumThrowsInvalidArgument)
{
  const architecture_t *arch = gfx942 ();
  ASSERT_NE (arch, nullptr);
  register_class_t rc (amd_dbgapi_register_class_id_t{ 0x204 }, *arch, "test");
  amd_dbgapi_architecture_id_t out{};
  try
    {
      rc.get_info (static_cast<amd_dbgapi_register_class_info_t> (INT_MAX),
                   sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}
