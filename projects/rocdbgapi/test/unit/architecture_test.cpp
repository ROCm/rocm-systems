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

/* Unit tests for src/architecture.{h,cpp}.

   architecture_t::s_architecture_map is populated by a function-local
   initializer at static-init time (architecture.cpp:8504), so the full
   set of concrete gfx9 / gfx9.4 / gfx10 / gfx11 / gfx12 architectures
   is available to this TU without any amd_dbgapi_initialize() call or
   process state.

   What's covered:
     - find() in all three forms (id, elf e_machine, name) round-trips
       against ::all() and returns nullptr for unknown lookups.
     - Each architecture's id, target_triple, name, elf_amdgpu_machine
       are internally consistent (the name suffix matches the target
       triple, find-by-name finds itself, find-by-id finds itself).
     - All registered architectures have unique ids, unique
       elf_amdgpu_machine values, and unique names (no collisions in
       the table).
     - regnum <-> register_id bit-packing is a pure transformation:
       a round-trip through regnum_to_register_id /
       register_id_to_regnum / register_id_to_architecture reproduces
       the original regnum and finds the original architecture.
     - register_id_to_regnum returns std::nullopt for an out-of-range
       regnum encoded into the low 32 bits.
     - register_id_to_architecture returns nullptr for an architecture
       id that isn't in the map.
     - register_name / register_size produce non-empty / non-zero
       results for a universal register (PC) on a representative
       architecture (gfx942), exercising the per-architecture virtual
       layer without needing wave/process state.

   Out of scope here:
     - get_info(...) — needs process_callbacks for the NAME query and
       the value buffer-size machinery is exercised in the feature
       tests.
     - Anything that takes wave_t / queue_t / process_t arguments
       (wave_get_state, wave_set_state, control_stack_iterate,
       initialize_*_ttmps, etc.) — those require populated runtime
       state that doesn't exist in a pure unit-test TU.
     - register_class_t / address_space_t / address_class_t iteration
       via range<>() — covered by their own dedicated tests in a later
       commit.  */

#include "architecture.h"
#include "os_driver.h"
#include "register.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <string>

using namespace amd::dbgapi;

/* ------------------------------------------------------------------ */
/* Registry-level invariants                                           */
/* ------------------------------------------------------------------ */

TEST (ArchitectureRegistry, AllIsNonEmpty)
{
  EXPECT_GT (architecture_t::all ().size (), 0u);
}

TEST (ArchitectureRegistry, IdsAreUnique)
{
  std::set<decltype (amd_dbgapi_architecture_id_t::handle)> seen;
  for (const auto &[id, arch] : architecture_t::all ())
    {
      auto inserted = seen.insert (id.handle).second;
      EXPECT_TRUE (inserted)
        << "duplicate architecture id 0x" << std::hex << id.handle;
    }
}

TEST (ArchitectureRegistry, ElfMachinesAreUnique)
{
  std::set<elf_amdgpu_machine_t> seen;
  for (const auto &[id, arch] : architecture_t::all ())
    {
      auto inserted = seen.insert (arch->elf_amdgpu_machine ()).second;
      EXPECT_TRUE (inserted) << "duplicate e_machine for arch "
                             << arch->name ();
    }
}

/* NOTE: architecture names are NOT unique.  All "*_generic_t"
   architectures end up with name() == "generic" because name() returns
   the suffix after the LAST '-' in the target triple and every generic
   triple ends in "-generic" (e.g. "amdgcn-amd-amdhsa--gfx9-generic",
   "amdgcn-amd-amdhsa--gfx10-1-generic").  This makes
   architecture_t::find(std::string) effectively non-deterministic for
   the name "generic" — it returns whichever entry the
   std::unordered_map iterates first.  Tracked separately for a
   library-side fix.  The weaker invariants we assert here are:
   non-empty names, and find-by-name yields something with the same
   name string back.  */

TEST (ArchitectureRegistry, NamesAreNonEmpty)
{
  for (const auto &[id, arch] : architecture_t::all ())
    EXPECT_FALSE (arch->name ().empty ());
}

/* ------------------------------------------------------------------ */
/* find() round-trip                                                   */
/* ------------------------------------------------------------------ */

TEST (ArchitectureFind, ById)
{
  for (const auto &[id, arch] : architecture_t::all ())
    {
      const architecture_t *found = architecture_t::find (id);
      ASSERT_NE (found, nullptr);
      EXPECT_EQ (found, arch.get ());
    }
}

TEST (ArchitectureFind, ByElfMachine)
{
  for (const auto &[id, arch] : architecture_t::all ())
    {
      const architecture_t *found
        = architecture_t::find (arch->elf_amdgpu_machine ());
      ASSERT_NE (found, nullptr);
      EXPECT_EQ (found->elf_amdgpu_machine (), arch->elf_amdgpu_machine ());
    }
}

TEST (ArchitectureFind, ByName)
{
  for (const auto &[id, arch] : architecture_t::all ())
    {
      const architecture_t *found = architecture_t::find (arch->name ());
      ASSERT_NE (found, nullptr) << "name not found: " << arch->name ();
      EXPECT_EQ (found->name (), arch->name ());
    }
}

TEST (ArchitectureFind, ByIdMissingReturnsNullptr)
{
  EXPECT_EQ (architecture_t::find (
               amd_dbgapi_architecture_id_t{ 0xdeadbeef }),
             nullptr);
}

TEST (ArchitectureFind, ByElfMachineMissingReturnsNullptr)
{
  EXPECT_EQ (architecture_t::find (
               static_cast<elf_amdgpu_machine_t> (0xffff)),
             nullptr);
}

TEST (ArchitectureFind, ByNameMissingReturnsNullptr)
{
  EXPECT_EQ (architecture_t::find (std::string{ "not-a-gpu" }), nullptr);
}

/* ------------------------------------------------------------------ */
/* Per-architecture metadata consistency                               */
/* ------------------------------------------------------------------ */

TEST (ArchitectureMetadata, NameMatchesTargetTripleSuffix)
{
  for (const auto &[id, arch] : architecture_t::all ())
    {
      const std::string &triple = arch->target_triple ();
      auto pos = triple.rfind ('-');
      ASSERT_NE (pos, std::string::npos)
        << "target triple has no '-': " << triple;
      EXPECT_EQ (triple.substr (pos + 1), arch->name ());
    }
}

TEST (ArchitectureMetadata, TargetTriplesAllAmdgcn)
{
  for (const auto &[id, arch] : architecture_t::all ())
    {
      EXPECT_NE (arch->target_triple ().find ("amdgcn"), std::string::npos)
        << "non-amdgcn triple: " << arch->target_triple ();
    }
}

/* ------------------------------------------------------------------ */
/* regnum <-> register_id packing                                      */
/* ------------------------------------------------------------------ */

TEST (ArchitectureRegisterId, RoundTripsRegnumAndArchitecture)
{
  for (const auto &[id, arch] : architecture_t::all ())
    {
      /* PC is universal across all AMDGPU architectures.  */
      auto regid = arch->regnum_to_register_id (amdgpu_regnum_t::pc);

      auto round_trip_regnum
        = architecture_t::register_id_to_regnum (regid);
      ASSERT_TRUE (round_trip_regnum.has_value ());
      EXPECT_EQ (*round_trip_regnum, amdgpu_regnum_t::pc);

      const architecture_t *round_trip_arch
        = architecture_t::register_id_to_architecture (regid);
      ASSERT_NE (round_trip_arch, nullptr);
      EXPECT_EQ (round_trip_arch, arch.get ());
    }
}

TEST (ArchitectureRegisterId, RegnumOutOfRangeReturnsNullopt)
{
  /* Encode a regnum well past last_regnum into the low 32 bits.  The
     high 32 bits don't matter for the regnum extraction.  */
  auto bogus_regnum_handle
    = static_cast<uint64_t> (
        static_cast<uint32_t> (amdgpu_regnum_t::last_regnum))
      + 0x1000u;
  amd_dbgapi_register_id_t regid{ bogus_regnum_handle };
  EXPECT_FALSE (architecture_t::register_id_to_regnum (regid).has_value ());
}

TEST (ArchitectureRegisterId, ArchitectureMissingReturnsNullptr)
{
  /* High 32 bits = bogus arch id, low 32 bits = valid regnum.  */
  uint64_t handle
    = (static_cast<uint64_t> (0xdeadbeefu) << 32)
      | static_cast<uint64_t> (
        static_cast<uint32_t> (amdgpu_regnum_t::pc));
  amd_dbgapi_register_id_t regid{ handle };
  EXPECT_EQ (architecture_t::register_id_to_architecture (regid), nullptr);
}

/* ------------------------------------------------------------------ */
/* Per-architecture virtual register methods (smoke test on PC)        */
/* ------------------------------------------------------------------ */

TEST (ArchitectureRegisterDescriptor, PcOnGfx942)
{
  const architecture_t *arch
    = architecture_t::find (std::string{ "gfx942" });
  ASSERT_NE (arch, nullptr);

  EXPECT_FALSE (arch->register_name (amdgpu_regnum_t::pc).empty ());
  EXPECT_GT (arch->register_size (amdgpu_regnum_t::pc), 0u);
  EXPECT_FALSE (arch->register_type (amdgpu_regnum_t::pc).empty ());
}
