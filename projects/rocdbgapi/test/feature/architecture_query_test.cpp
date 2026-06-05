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

/* Architecture queries through the public amd_dbgapi API.

   The architecture registry is populated at static-init time, so these
   tests don't require a kernel driver and run on every host.  Coverage
   here complements the unit-test version (test/unit/architecture_test.cpp,
   which exercises architecture_t::find through the internal seam) by
   going through the *public* entry points the way a real client (rocgdb,
   rocprofiler) would.  That catches symbol-export, ABI alignment, and
   callback-wiring regressions the unit tests can't see.

   Per-architecture coverage is intentionally narrow: one or two
   "shape" properties (instruction size sane, breakpoint round-trips)
   per gfx target.  Deep per-arch semantics belong in unit tests where
   they don't need a live library.  */

#include "amd-dbgapi.h"

#include "support/dbgapi_fixture.h"

#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

using amd::dbgapi::test::dbgapi_fixture_t;

namespace
{

/* (elf_amdgpu_machine, expected canonical name) for a representative
   slice of supported architectures.  Values match
   src/architecture.cpp's gfx*_architecture_t ctors and EF_AMDGPU_MACH_*
   in llvm's BinaryFormat/ELF.h.  Keeping the list short so a future
   header/ABI change doesn't snowball into a 30-line diff here.  */
struct arch_entry
{
  uint32_t elf_machine;
  const char *name;
};

const std::vector<arch_entry> &
supported_archs ()
{
  /* The public NAME query returns the short architecture suffix
     (architecture_t::name() in src/architecture.cpp:8362), not the
     full target triple.  Keep those in sync when adding entries.  */
  static const std::vector<arch_entry> a = {
    { 0x02c, "gfx900" },
    { 0x02f, "gfx906" },
    { 0x030, "gfx908" },
    { 0x03f, "gfx90a" },
    { 0x04c, "gfx942" },
    { 0x036, "gfx1030" },
  };
  return a;
}

} /* namespace */

using ArchitectureQuery = dbgapi_fixture_t;

TEST_F (ArchitectureQuery, GetArchitectureRejectsBogusElfMachine)
{
  amd_dbgapi_architecture_id_t arch{};
  /* 0xFFFFFF is in the EF_AMDGPU_MACH bitfield range but is not a
     defined architecture.  Library must report
     INVALID_ELF_AMDGPU_MACHINE rather than asserting.  */
  EXPECT_EQ (amd_dbgapi_get_architecture (0xFFFFFF, &arch),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ELF_AMDGPU_MACHINE);
}

TEST_F (ArchitectureQuery, GetArchitectureReturnsNonNoneForEachSupportedMachine)
{
  for (const auto &e : supported_archs ())
    {
      amd_dbgapi_architecture_id_t arch{};
      ASSERT_EQ (amd_dbgapi_get_architecture (e.elf_machine, &arch),
                 AMD_DBGAPI_STATUS_SUCCESS)
        << "elf_machine = 0x" << std::hex << e.elf_machine;
      EXPECT_NE (arch.handle, AMD_DBGAPI_ARCHITECTURE_NONE.handle)
        << "elf_machine = 0x" << std::hex << e.elf_machine;
    }
}

TEST_F (ArchitectureQuery, ArchitectureInfoNameMatchesCanonicalTriple)
{
  for (const auto &e : supported_archs ())
    {
      amd_dbgapi_architecture_id_t arch{};
      ASSERT_EQ (amd_dbgapi_get_architecture (e.elf_machine, &arch),
                 AMD_DBGAPI_STATUS_SUCCESS);

      char *name = nullptr;
      ASSERT_EQ (
        amd_dbgapi_architecture_get_info (arch, AMD_DBGAPI_ARCHITECTURE_INFO_NAME,
                                          sizeof (name), &name),
        AMD_DBGAPI_STATUS_SUCCESS);
      ASSERT_NE (name, nullptr);
      EXPECT_STREQ (name, e.name);
      std::free (name);
    }
}

TEST_F (ArchitectureQuery, ArchitectureInfoElfMachineRoundTrips)
{
  for (const auto &e : supported_archs ())
    {
      amd_dbgapi_architecture_id_t arch{};
      ASSERT_EQ (amd_dbgapi_get_architecture (e.elf_machine, &arch),
                 AMD_DBGAPI_STATUS_SUCCESS);

      uint32_t machine = 0;
      ASSERT_EQ (
        amd_dbgapi_architecture_get_info (
          arch, AMD_DBGAPI_ARCHITECTURE_INFO_ELF_AMDGPU_MACHINE,
          sizeof (machine), &machine),
        AMD_DBGAPI_STATUS_SUCCESS);
      EXPECT_EQ (machine, e.elf_machine);
    }
}

TEST_F (ArchitectureQuery, ArchitectureInfoBreakpointShapeIsSane)
{
  /* For each supported arch, breakpoint-instruction-size must fit in
     largest-instruction-size, the pc-adjust must not exceed the
     breakpoint size, and the instruction bytes themselves must be
     retrievable into a buffer of exactly that size.  */
  for (const auto &e : supported_archs ())
    {
      amd_dbgapi_architecture_id_t arch{};
      ASSERT_EQ (amd_dbgapi_get_architecture (e.elf_machine, &arch),
                 AMD_DBGAPI_STATUS_SUCCESS)
        << "arch = " << e.name;

      amd_dbgapi_size_t largest = 0;
      ASSERT_EQ (
        amd_dbgapi_architecture_get_info (
          arch, AMD_DBGAPI_ARCHITECTURE_INFO_LARGEST_INSTRUCTION_SIZE,
          sizeof (largest), &largest),
        AMD_DBGAPI_STATUS_SUCCESS)
        << "arch = " << e.name;
      EXPECT_GT (largest, 0u) << "arch = " << e.name;

      amd_dbgapi_size_t bp_size = 0;
      ASSERT_EQ (
        amd_dbgapi_architecture_get_info (
          arch, AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_SIZE,
          sizeof (bp_size), &bp_size),
        AMD_DBGAPI_STATUS_SUCCESS)
        << "arch = " << e.name;
      EXPECT_GT (bp_size, 0u) << "arch = " << e.name;
      EXPECT_LE (bp_size, largest) << "arch = " << e.name;

      amd_dbgapi_size_t pc_adjust = 0;
      ASSERT_EQ (
        amd_dbgapi_architecture_get_info (
          arch, AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_PC_ADJUST,
          sizeof (pc_adjust), &pc_adjust),
        AMD_DBGAPI_STATUS_SUCCESS)
        << "arch = " << e.name;
      EXPECT_LE (pc_adjust, bp_size) << "arch = " << e.name;

      void *bp_bytes = nullptr;
      ASSERT_EQ (
        amd_dbgapi_architecture_get_info (
          arch, AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION,
          sizeof (bp_bytes), &bp_bytes),
        AMD_DBGAPI_STATUS_SUCCESS)
        << "arch = " << e.name;
      EXPECT_NE (bp_bytes, nullptr) << "arch = " << e.name;
      std::free (bp_bytes);
    }
}

TEST_F (ArchitectureQuery, ArchitectureInfoPcRegisterIsNonNone)
{
  for (const auto &e : supported_archs ())
    {
      amd_dbgapi_architecture_id_t arch{};
      ASSERT_EQ (amd_dbgapi_get_architecture (e.elf_machine, &arch),
                 AMD_DBGAPI_STATUS_SUCCESS);

      amd_dbgapi_register_id_t pc{};
      ASSERT_EQ (
        amd_dbgapi_architecture_get_info (
          arch, AMD_DBGAPI_ARCHITECTURE_INFO_PC_REGISTER, sizeof (pc), &pc),
        AMD_DBGAPI_STATUS_SUCCESS)
        << "arch = " << e.name;
      EXPECT_NE (pc.handle, AMD_DBGAPI_REGISTER_NONE.handle)
        << "arch = " << e.name;
    }
}

TEST_F (ArchitectureQuery, ArchitectureInfoSizeMismatchIsRejected)
{
  amd_dbgapi_architecture_id_t arch{};
  ASSERT_EQ (amd_dbgapi_get_architecture (0x04c /* gfx942 */, &arch),
             AMD_DBGAPI_STATUS_SUCCESS);

  /* Pass a wrong value_size and expect INVALID_ARGUMENT_COMPATIBILITY,
     not silent corruption.  */
  uint8_t too_small = 0;
  EXPECT_EQ (amd_dbgapi_architecture_get_info (
               arch, AMD_DBGAPI_ARCHITECTURE_INFO_ELF_AMDGPU_MACHINE,
               sizeof (too_small), &too_small),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
}

TEST_F (ArchitectureQuery, GetArchitectureRejectsNullOutPointer)
{
  EXPECT_EQ (amd_dbgapi_get_architecture (0x04c, nullptr),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

/* C25 negative-test sweep: value_size and enum edge cases at the public
   C API boundary.  Status codes replace exceptions here.  */

TEST_F (ArchitectureQuery, ArchitectureInfoValueSizeZeroIsRejected)
{
  amd_dbgapi_architecture_id_t arch{};
  ASSERT_EQ (amd_dbgapi_get_architecture (0x04c /* gfx942 */, &arch),
             AMD_DBGAPI_STATUS_SUCCESS);
  uint32_t out = 0;
  EXPECT_EQ (
    amd_dbgapi_architecture_get_info (
      arch, AMD_DBGAPI_ARCHITECTURE_INFO_ELF_AMDGPU_MACHINE, 0, &out),
    AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
}

TEST_F (ArchitectureQuery, ArchitectureInfoValueSizeMaxIsRejected)
{
  amd_dbgapi_architecture_id_t arch{};
  ASSERT_EQ (amd_dbgapi_get_architecture (0x04c, &arch),
             AMD_DBGAPI_STATUS_SUCCESS);
  uint32_t out = 0;
  EXPECT_EQ (
    amd_dbgapi_architecture_get_info (
      arch, AMD_DBGAPI_ARCHITECTURE_INFO_ELF_AMDGPU_MACHINE, SIZE_MAX, &out),
    AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
}

TEST_F (ArchitectureQuery, ArchitectureInfoValueSizeTooLargeByOneIsRejected)
{
  amd_dbgapi_architecture_id_t arch{};
  ASSERT_EQ (amd_dbgapi_get_architecture (0x04c, &arch),
             AMD_DBGAPI_STATUS_SUCCESS);
  uint8_t out[sizeof (uint32_t) + 1]{};
  EXPECT_EQ (
    amd_dbgapi_architecture_get_info (
      arch, AMD_DBGAPI_ARCHITECTURE_INFO_ELF_AMDGPU_MACHINE,
      sizeof (uint32_t) + 1, out),
    AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
}

TEST_F (ArchitectureQuery, ArchitectureInfoNegativeEnumIsRejected)
{
  amd_dbgapi_architecture_id_t arch{};
  ASSERT_EQ (amd_dbgapi_get_architecture (0x04c, &arch),
             AMD_DBGAPI_STATUS_SUCCESS);
  uint32_t out = 0;
  EXPECT_EQ (
    amd_dbgapi_architecture_get_info (
      arch, static_cast<amd_dbgapi_architecture_info_t> (-1), sizeof (out),
      &out),
    AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST_F (ArchitectureQuery, ArchitectureInfoIntMaxEnumIsRejected)
{
  amd_dbgapi_architecture_id_t arch{};
  ASSERT_EQ (amd_dbgapi_get_architecture (0x04c, &arch),
             AMD_DBGAPI_STATUS_SUCCESS);
  uint32_t out = 0;
  EXPECT_EQ (
    amd_dbgapi_architecture_get_info (
      arch, static_cast<amd_dbgapi_architecture_info_t> (INT_MAX),
      sizeof (out), &out),
    AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

/* ------------------------------------------------------------------ */
/* C31 — ELF e_machine edge cases                                      */
/* ------------------------------------------------------------------ */

/* 0 is the common "uninitialized" value for an ELF header field and
   must be rejected, not mapped to the first registered architecture.  */
TEST_F (ArchitectureQuery, GetArchitectureRejectsZeroElfMachine)
{
  amd_dbgapi_architecture_id_t arch{};
  EXPECT_EQ (amd_dbgapi_get_architecture (0u, &arch),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ELF_AMDGPU_MACHINE);
}

/* UINT32_MAX exercises the upper rail of the elf_amdgpu_machine_t
   cast.  Must reach INVALID_ELF_AMDGPU_MACHINE without crashing.  */
TEST_F (ArchitectureQuery, GetArchitectureRejectsUint32MaxElfMachine)
{
  amd_dbgapi_architecture_id_t arch{};
  EXPECT_EQ (amd_dbgapi_get_architecture (UINT32_MAX, &arch),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ELF_AMDGPU_MACHINE);
}

/* EM_X86_64 (0x3E) is a valid ELF e_machine but never an AMDGPU
   EF_AMDGPU_MACH value.  This guards against a naive lookup that
   accepts any non-zero input.  */
TEST_F (ArchitectureQuery, GetArchitectureRejectsX86_64ElfMachine)
{
  amd_dbgapi_architecture_id_t arch{};
  EXPECT_EQ (amd_dbgapi_get_architecture (0x3E, &arch),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ELF_AMDGPU_MACHINE);
}

/* Repeated lookups of the same elf_amdgpu_machine must return the
   same architecture id — the registry is a singleton per arch.  */
TEST_F (ArchitectureQuery, GetArchitectureIsIdempotent)
{
  amd_dbgapi_architecture_id_t a1{};
  amd_dbgapi_architecture_id_t a2{};
  ASSERT_EQ (amd_dbgapi_get_architecture (0x04c, &a1),
             AMD_DBGAPI_STATUS_SUCCESS);
  ASSERT_EQ (amd_dbgapi_get_architecture (0x04c, &a2),
             AMD_DBGAPI_STATUS_SUCCESS);
  EXPECT_EQ (a1.handle, a2.handle);
}

/* Different elf_amdgpu_machine values must produce different
   architecture ids — a regression where two arches collide would
   leak data across processes targeting different GPUs.  */
TEST_F (ArchitectureQuery, GetArchitectureReturnsDistinctIdsAcrossMachines)
{
  amd_dbgapi_architecture_id_t a1{};
  amd_dbgapi_architecture_id_t a2{};
  ASSERT_EQ (amd_dbgapi_get_architecture (0x04c /* gfx942 */, &a1),
             AMD_DBGAPI_STATUS_SUCCESS);
  ASSERT_EQ (amd_dbgapi_get_architecture (0x036 /* gfx1030 */, &a2),
             AMD_DBGAPI_STATUS_SUCCESS);
  EXPECT_NE (a1.handle, a2.handle);
  EXPECT_NE (a1.handle, AMD_DBGAPI_ARCHITECTURE_NONE.handle);
  EXPECT_NE (a2.handle, AMD_DBGAPI_ARCHITECTURE_NONE.handle);
}

/* The architecture handle returned for a given e_machine must accept
   *_get_info on the round-trip query — proves we got a live registry
   entry, not just a placeholder id.  */
TEST_F (ArchitectureQuery, GetArchitectureReturnsLiveHandleForEachSupported)
{
  for (const auto &e : supported_archs ())
    {
      amd_dbgapi_architecture_id_t arch{};
      ASSERT_EQ (amd_dbgapi_get_architecture (e.elf_machine, &arch),
                 AMD_DBGAPI_STATUS_SUCCESS)
        << "arch = " << e.name;
      uint32_t machine = 0;
      EXPECT_EQ (
        amd_dbgapi_architecture_get_info (
          arch, AMD_DBGAPI_ARCHITECTURE_INFO_ELF_AMDGPU_MACHINE,
          sizeof (machine), &machine),
        AMD_DBGAPI_STATUS_SUCCESS)
        << "arch = " << e.name;
    }
}
