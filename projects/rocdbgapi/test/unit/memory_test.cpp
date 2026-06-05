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

/* Unit tests for memory.h surfaces that don't require a wave_t / agent_t.

   Covered:
     * Address type wrappers (base_address_t<T>): default ctor zeroes,
       value ctor, implicit operator uint64_t, +/-/+=/-= arithmetic, and
       the global_address_t -> agent/host conversion operators.
     * std::numeric_limits specializations for the three address types
       (min/max/lowest/is_specialized return the correct wrapped type).
     * std::hash specializations: same uint64 hashes the same across the
       three address types and matches std::hash<uint64_t>.
     * address_space_t::global() and ::host() static accessors: kind,
       name, address_size, null_address, dwarf_value, is_internal/valid
       (host has no dwarf -> internal; global has DW_ASPACE_none).
     * address_space_t::get_info: ADDRESS_SIZE / NULL_ADDRESS / ACCESS /
       DWARF on the global accessor, and unknown query throws
       INVALID_ARGUMENT.  (The NAME query routes the result through the
       client allocate_memory callback, which isn't installed in a unit
       test.  Tested at the C-API level in feature tests instead.)
     * address_class_t value object: name(), dwarf_value(), address_space()
       and get_info(ADDRESS_SPACE / DWARF) + unknown query.
     * memory_cache_t<agent_address_t> through a stub delegate fn:
       - contains_all() false on empty, true after prefetch().
       - prefetch() aligns to cache_line_size (64) and calls the delegate
         once per coalesced span.
       - write_global_memory() into a prefetched range marks lines dirty
         and write_back() commits them via the delegate; clean lines are
         skipped.
       - discard() drops cache lines without invoking the delegate.
       - read_global_memory() with no prior prefetch goes straight through
         the delegate (cache miss path).

   NOT covered here (deferred):
     * Concrete address_space_t::lower / convert paths -- need a real
       wave_t / agent_t.
     * generic_address_space_t aperture lookup -- needs an architecture
       set up with apertures, deferred to feature tests.  */

#include "amd-dbgapi.h"
#include "exception.h"
#include "memory.h"

#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

using amd::dbgapi::address_class_t;
using amd::dbgapi::address_space_t;
using amd::dbgapi::agent_address_t;
using amd::dbgapi::api_error_t;
using amd::dbgapi::global_address_t;
using amd::dbgapi::host_address_t;
using amd::dbgapi::memory_cache_t;

/* ------------------------------------------------------------------ */
/* Address type wrappers                                               */
/* ------------------------------------------------------------------ */

TEST (MemoryAddress, DefaultCtorZeroes)
{
  agent_address_t a{};
  host_address_t h{};
  global_address_t g{};
  EXPECT_EQ (static_cast<uint64_t> (a), 0u);
  EXPECT_EQ (static_cast<uint64_t> (h), 0u);
  EXPECT_EQ (static_cast<uint64_t> (g), 0u);
}

TEST (MemoryAddress, ValueCtorAndConversion)
{
  agent_address_t a{ 0xdeadbeefull };
  EXPECT_EQ (static_cast<uint64_t> (a), 0xdeadbeefull);
}

TEST (MemoryAddress, ArithmeticOperators)
{
  agent_address_t a{ 0x1000 };
  EXPECT_EQ (static_cast<uint64_t> (a + 0x40), 0x1040u);
  EXPECT_EQ (static_cast<uint64_t> (a - 0x10), 0x0FF0u);
  a += 0x100;
  EXPECT_EQ (static_cast<uint64_t> (a), 0x1100u);
  a -= 0x80;
  EXPECT_EQ (static_cast<uint64_t> (a), 0x1080u);
}

TEST (MemoryAddress, GlobalConvertsToAgentAndHost)
{
  global_address_t g{ 0xabcd0000ull };
  agent_address_t a = g;
  host_address_t h = g;
  EXPECT_EQ (static_cast<uint64_t> (a), 0xabcd0000ull);
  EXPECT_EQ (static_cast<uint64_t> (h), 0xabcd0000ull);
}

/* ------------------------------------------------------------------ */
/* std::numeric_limits specializations                                 */
/* ------------------------------------------------------------------ */

TEST (MemoryAddress, NumericLimitsAreSpecialized)
{
  EXPECT_TRUE (std::numeric_limits<agent_address_t>::is_specialized);
  EXPECT_TRUE (std::numeric_limits<host_address_t>::is_specialized);
  EXPECT_TRUE (std::numeric_limits<global_address_t>::is_specialized);

  EXPECT_EQ (static_cast<uint64_t> (std::numeric_limits<agent_address_t>::min ()),
             0u);
  EXPECT_EQ (static_cast<uint64_t> (std::numeric_limits<agent_address_t>::max ()),
             std::numeric_limits<uint64_t>::max ());
  EXPECT_EQ (
    static_cast<uint64_t> (std::numeric_limits<agent_address_t>::lowest ()), 0u);

  EXPECT_EQ (static_cast<uint64_t> (std::numeric_limits<host_address_t>::max ()),
             std::numeric_limits<uint64_t>::max ());
  EXPECT_EQ (
    static_cast<uint64_t> (std::numeric_limits<global_address_t>::max ()),
    std::numeric_limits<uint64_t>::max ());
}

/* ------------------------------------------------------------------ */
/* std::hash specializations                                           */
/* ------------------------------------------------------------------ */

TEST (MemoryAddress, HashMatchesUint64)
{
  uint64_t v = 0x1234567890abcdefull;
  auto h_u64 = std::hash<uint64_t>{}(v);
  EXPECT_EQ (std::hash<agent_address_t>{}(agent_address_t{ v }), h_u64);
  EXPECT_EQ (std::hash<host_address_t>{}(host_address_t{ v }), h_u64);
  EXPECT_EQ (std::hash<global_address_t>{}(global_address_t{ v }), h_u64);
}

/* ------------------------------------------------------------------ */
/* address_space_t::global() / ::host() accessors                      */
/* ------------------------------------------------------------------ */

TEST (AddressSpace, GlobalAccessor)
{
  const auto &g = address_space_t::global ();
  EXPECT_EQ (g.kind (), address_space_t::kind_t::global);
  EXPECT_EQ (g.name (), "global");
  EXPECT_EQ (g.address_size (), 64u);
  EXPECT_EQ (g.null_address (), 0u);
  /* global is constructed with DW_ASPACE_none (0), which is still a value
     -- not std::nullopt -- so it is NOT internal.  */
  ASSERT_TRUE (g.dwarf_value ().has_value ());
  EXPECT_EQ (*g.dwarf_value (), 0u);
  EXPECT_FALSE (g.is_internal ());
  EXPECT_TRUE (g.is_valid ());
  /* last_address() = bit_mask(0, address_size - 1) = all 1s for 64 bits.  */
  EXPECT_EQ (g.last_address (), std::numeric_limits<uint64_t>::max ());
}

TEST (AddressSpace, HostAccessor)
{
  const auto &h = address_space_t::host ();
  EXPECT_EQ (h.kind (), address_space_t::kind_t::host);
  EXPECT_EQ (h.name (), "host");
  EXPECT_EQ (h.address_size (), 64u);
  EXPECT_EQ (h.null_address (), 0u);
  /* host_address_space_t passes std::nullopt for dwarf -> internal.  */
  EXPECT_FALSE (h.dwarf_value ().has_value ());
  EXPECT_TRUE (h.is_internal ());
  EXPECT_FALSE (h.is_valid ());
}

TEST (AddressSpace, GlobalAndHostAreStableSingletons)
{
  EXPECT_EQ (&address_space_t::global (), &address_space_t::global ());
  EXPECT_EQ (&address_space_t::host (), &address_space_t::host ());
}

TEST (AddressSpace, GetInfoBasicQueries)
{
  const auto &g = address_space_t::global ();

  amd_dbgapi_size_t size{};
  g.get_info (AMD_DBGAPI_ADDRESS_SPACE_INFO_ADDRESS_SIZE, sizeof (size), &size);
  EXPECT_EQ (size, 64u);

  amd_dbgapi_segment_address_t null{};
  g.get_info (AMD_DBGAPI_ADDRESS_SPACE_INFO_NULL_ADDRESS, sizeof (null), &null);
  EXPECT_EQ (null, 0u);

  amd_dbgapi_address_space_access_t access{};
  g.get_info (AMD_DBGAPI_ADDRESS_SPACE_INFO_ACCESS, sizeof (access), &access);
  EXPECT_EQ (access, AMD_DBGAPI_ADDRESS_SPACE_ACCESS_ALL);

  uint64_t dwarf{};
  g.get_info (AMD_DBGAPI_ADDRESS_SPACE_INFO_DWARF, sizeof (dwarf), &dwarf);
  EXPECT_EQ (dwarf, 0u);
}

TEST (AddressSpace, GetInfoUnknownQueryThrows)
{
  const auto &g = address_space_t::global ();
  uint64_t dummy{};
  try
    {
      g.get_info (static_cast<amd_dbgapi_address_space_info_t> (0xbeef),
                  sizeof (dummy), &dummy);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

/* C25 negative-test sweep: value_size and enum edge cases.              */

TEST (AddressSpace, GetInfoValueSizeZeroThrowsCompatibility)
{
  const auto &g = address_space_t::global ();
  amd_dbgapi_size_t out{};
  try
    {
      g.get_info (AMD_DBGAPI_ADDRESS_SPACE_INFO_ADDRESS_SIZE, 0, &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (AddressSpace, GetInfoValueSizeMaxThrowsCompatibility)
{
  const auto &g = address_space_t::global ();
  amd_dbgapi_size_t out{};
  try
    {
      g.get_info (AMD_DBGAPI_ADDRESS_SPACE_INFO_ADDRESS_SIZE, SIZE_MAX, &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (AddressSpace, GetInfoValueSizeTooLargeByOneThrowsCompatibility)
{
  const auto &g = address_space_t::global ();
  uint8_t out[sizeof (amd_dbgapi_size_t) + 1]{};
  try
    {
      g.get_info (AMD_DBGAPI_ADDRESS_SPACE_INFO_ADDRESS_SIZE,
                  sizeof (amd_dbgapi_size_t) + 1, out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (AddressSpace, GetInfoNegativeEnumThrowsInvalidArgument)
{
  const auto &g = address_space_t::global ();
  amd_dbgapi_size_t out{};
  try
    {
      g.get_info (static_cast<amd_dbgapi_address_space_info_t> (-1),
                  sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (AddressSpace, GetInfoIntMaxEnumThrowsInvalidArgument)
{
  const auto &g = address_space_t::global ();
  amd_dbgapi_size_t out{};
  try
    {
      g.get_info (static_cast<amd_dbgapi_address_space_info_t> (INT_MAX),
                  sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

/* ------------------------------------------------------------------ */
/* address_class_t value object                                        */
/* ------------------------------------------------------------------ */

TEST (AddressClass, AccessorsAndGetInfo)
{
  const auto &g = address_space_t::global ();
  address_class_t klass (amd_dbgapi_address_class_id_t{ 42 },
                         "global_class", /* dwarf_value */ 0x1, g);

  EXPECT_EQ (klass.name (), "global_class");
  EXPECT_EQ (klass.dwarf_value (), 0x1u);
  EXPECT_EQ (&klass.address_space (), &g);

  amd_dbgapi_address_space_id_t as_id{};
  klass.get_info (AMD_DBGAPI_ADDRESS_CLASS_INFO_ADDRESS_SPACE, sizeof (as_id),
                  &as_id);
  EXPECT_EQ (as_id.handle, g.id ().handle);

  uint64_t dwarf{};
  klass.get_info (AMD_DBGAPI_ADDRESS_CLASS_INFO_DWARF, sizeof (dwarf), &dwarf);
  EXPECT_EQ (dwarf, 0x1u);
}

TEST (AddressClass, GetInfoUnknownQueryThrows)
{
  const auto &g = address_space_t::global ();
  address_class_t klass (amd_dbgapi_address_class_id_t{ 1 }, "x", 0, g);
  uint64_t dummy{};
  try
    {
      klass.get_info (static_cast<amd_dbgapi_address_class_info_t> (0xbeef),
                      sizeof (dummy), &dummy);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

/* C25 negative-test sweep: value_size and enum edge cases.              */

TEST (AddressClass, GetInfoValueSizeZeroThrowsCompatibility)
{
  const auto &g = address_space_t::global ();
  address_class_t klass (amd_dbgapi_address_class_id_t{ 1 }, "x", 0, g);
  uint64_t out{};
  try
    {
      klass.get_info (AMD_DBGAPI_ADDRESS_CLASS_INFO_DWARF, 0, &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (AddressClass, GetInfoValueSizeMaxThrowsCompatibility)
{
  const auto &g = address_space_t::global ();
  address_class_t klass (amd_dbgapi_address_class_id_t{ 1 }, "x", 0, g);
  uint64_t out{};
  try
    {
      klass.get_info (AMD_DBGAPI_ADDRESS_CLASS_INFO_DWARF, SIZE_MAX, &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (AddressClass, GetInfoValueSizeTooLargeByOneThrowsCompatibility)
{
  const auto &g = address_space_t::global ();
  address_class_t klass (amd_dbgapi_address_class_id_t{ 1 }, "x", 0, g);
  uint8_t out[sizeof (uint64_t) + 1]{};
  try
    {
      klass.get_info (AMD_DBGAPI_ADDRESS_CLASS_INFO_DWARF,
                      sizeof (uint64_t) + 1, out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (AddressClass, GetInfoNegativeEnumThrowsInvalidArgument)
{
  const auto &g = address_space_t::global ();
  address_class_t klass (amd_dbgapi_address_class_id_t{ 1 }, "x", 0, g);
  uint64_t out{};
  try
    {
      klass.get_info (static_cast<amd_dbgapi_address_class_info_t> (-1),
                      sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (AddressClass, GetInfoIntMaxEnumThrowsInvalidArgument)
{
  const auto &g = address_space_t::global ();
  address_class_t klass (amd_dbgapi_address_class_id_t{ 1 }, "x", 0, g);
  uint64_t out{};
  try
    {
      klass.get_info (static_cast<amd_dbgapi_address_class_info_t> (INT_MAX),
                      sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

/* ------------------------------------------------------------------ */
/* memory_cache_t<agent_address_t> with a stub delegate                */
/* ------------------------------------------------------------------ */

namespace
{

/* A scriptable backing-store delegate that mirrors a flat std::vector
   indexed by agent_address_t.  Records each (address, size, kind) call
   for assertion.  */
struct flat_backing_store_t
{
  std::vector<std::byte> mem;
  struct call_t
  {
    uint64_t address;
    size_t size;
    bool is_write;
  };
  std::vector<call_t> calls;

  explicit flat_backing_store_t (size_t bytes) : mem (bytes) {}

  size_t
  xfer (agent_address_t address, void *read, const void *write, size_t size)
  {
    uint64_t addr = address;
    calls.push_back ({ addr, size, write != nullptr });
    if (addr + size > mem.size ())
      throw amd::dbgapi::memory_access_error_t (
        amd::dbgapi::address_space_t::global (), addr, "oob");

    if (read != nullptr)
      memcpy (read, mem.data () + addr, size);
    else
      memcpy (mem.data () + addr, write, size);
    return size;
  }
};

} /* namespace */

TEST (MemoryCache, ContainsAllFalseOnEmpty)
{
  flat_backing_store_t store (4096);
  memory_cache_t<agent_address_t> cache (
    [&] (agent_address_t a, void *r, const void *w, size_t s) {
      return store.xfer (a, r, w, s);
    });
  EXPECT_FALSE (cache.contains_all (agent_address_t{ 0 }, 64));
  cache.discard (agent_address_t{ 0 }, amd_dbgapi_size_t (-1));
}

TEST (MemoryCache, PrefetchPopulatesAndAlignsToCacheLine)
{
  flat_backing_store_t store (4096);
  /* Seed memory with a recognizable pattern at offset 0x100.  */
  for (size_t i = 0; i < 128; ++i)
    store.mem[0x100 + i] = std::byte (i & 0xff);

  memory_cache_t<agent_address_t> cache (
    [&] (agent_address_t a, void *r, const void *w, size_t s) {
      return store.xfer (a, r, w, s);
    });

  /* Prefetch a 10-byte span starting at 0x105 -> aligns down to 0x100,
     up to 0x140 (one cache line of 64 bytes).  */
  cache.prefetch (agent_address_t{ 0x105 }, 10);
  EXPECT_TRUE (cache.contains_all (agent_address_t{ 0x105 }, 10));
  EXPECT_TRUE (cache.contains_all (agent_address_t{ 0x100 }, 64));
  /* The next cache line was not requested.  */
  EXPECT_FALSE (cache.contains_all (agent_address_t{ 0x140 }, 1));

  ASSERT_EQ (store.calls.size (), 1u);
  EXPECT_EQ (store.calls[0].address, 0x100u);
  EXPECT_EQ (store.calls[0].size, 64u);
  EXPECT_FALSE (store.calls[0].is_write);

  /* Read back through the cache: no additional delegate call.  */
  uint8_t buf[10]{};
  size_t got
    = cache.read_global_memory (agent_address_t{ 0x105 }, buf, sizeof (buf));
  EXPECT_EQ (got, 10u);
  EXPECT_EQ (buf[0], 5u);
  EXPECT_EQ (buf[9], 14u);
  EXPECT_EQ (store.calls.size (), 1u);

  cache.discard (agent_address_t{ 0 }, amd_dbgapi_size_t (-1));
}

TEST (MemoryCache, WriteBackCommitsDirtyLines)
{
  flat_backing_store_t store (4096);
  memory_cache_t<agent_address_t> cache (
    [&] (agent_address_t a, void *r, const void *w, size_t s) {
      return store.xfer (a, r, w, s);
    });

  /* Two adjacent cache lines [0x80, 0x100).  */
  cache.prefetch (agent_address_t{ 0x80 }, 128);
  ASSERT_EQ (store.calls.size (), 1u);
  store.calls.clear ();

  /* Dirty only the first line.  */
  uint8_t payload[4] = { 0xaa, 0xbb, 0xcc, 0xdd };
  size_t wrote
    = cache.write_global_memory (agent_address_t{ 0x90 }, payload, 4);
  EXPECT_EQ (wrote, 4u);
  /* write-back policy -> no immediate delegate call on write.  */
  EXPECT_TRUE (store.calls.empty ());

  /* Write-back the full span.  Only the dirty line should be flushed.  */
  cache.write_back (agent_address_t{ 0x80 }, 128);
  ASSERT_EQ (store.calls.size (), 1u);
  EXPECT_TRUE (store.calls[0].is_write);
  EXPECT_EQ (store.calls[0].address, 0x80u);
  EXPECT_EQ (store.calls[0].size, 64u);
  EXPECT_EQ (std::to_integer<uint8_t> (store.mem[0x90]), 0xaa);
  EXPECT_EQ (std::to_integer<uint8_t> (store.mem[0x93]), 0xdd);

  /* Calling write_back again is a no-op (no more dirty lines).  */
  store.calls.clear ();
  cache.write_back (agent_address_t{ 0x80 }, 128);
  EXPECT_TRUE (store.calls.empty ());

  cache.discard (agent_address_t{ 0 }, amd_dbgapi_size_t (-1));
}

TEST (MemoryCache, DiscardDropsLinesWithoutDelegateCalls)
{
  flat_backing_store_t store (4096);
  memory_cache_t<agent_address_t> cache (
    [&] (agent_address_t a, void *r, const void *w, size_t s) {
      return store.xfer (a, r, w, s);
    });
  cache.prefetch (agent_address_t{ 0 }, 64);
  store.calls.clear ();

  /* force_discard=true so we don't trip the "dirty drop" assert even if
     a previous step had dirtied something.  */
  cache.discard (agent_address_t{ 0 }, amd_dbgapi_size_t (-1),
                 /* force_discard */ true);
  EXPECT_TRUE (store.calls.empty ());
  EXPECT_FALSE (cache.contains_all (agent_address_t{ 0 }, 64));
}

TEST (MemoryCache, UncachedReadGoesStraightToDelegate)
{
  flat_backing_store_t store (4096);
  for (size_t i = 0; i < 16; ++i)
    store.mem[i] = std::byte (0xa0 | (i & 0xf));

  memory_cache_t<agent_address_t> cache (
    [&] (agent_address_t a, void *r, const void *w, size_t s) {
      return store.xfer (a, r, w, s);
    });

  /* No prior prefetch -> contains_all() false -> xfer_global_memory
     short-circuits to the delegate.  */
  uint8_t buf[8]{};
  size_t got
    = cache.read_global_memory (agent_address_t{ 0 }, buf, sizeof (buf));
  EXPECT_EQ (got, 8u);
  ASSERT_EQ (store.calls.size (), 1u);
  EXPECT_FALSE (store.calls[0].is_write);
  EXPECT_EQ (store.calls[0].address, 0u);
  EXPECT_EQ (store.calls[0].size, 8u);
  EXPECT_EQ (buf[0], 0xa0);
  EXPECT_EQ (buf[7], 0xa7);
}
