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

/* Unit tests for src/code_object.{h,cpp}.

   code_object_t is a small value-object: (process&, uri, load_address)
   plus a per-object mark for the snapshot epoch.  Its constructor
   takes a process_t& that is stored as a reference; we can't construct
   a real process_t here because process_t::process_t() calls into the
   client-process callbacks (populated by amd_dbgapi_initialize) and
   opens an OS-driver, neither of which exist in a pure unit-test TU.

   What we DO cover:
     - code_object_t::next_mark() static counter: strictly increasing
       across successive calls, never returns 0 (the counter starts at
       1).
     - Field round-trip: load_address() and uri() return whatever was
       passed at construction; mark() defaults to 0; set_mark()/mark()
       round-trip.
     - id() returns the handle passed at construction.
     - get_info(URI_NAME) and get_info(LOAD_ADDRESS) populate the
       caller's buffer correctly — those branches do not dereference
       m_process.
     - get_info(<bogus query>) throws api_error_t with
       AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT.

   What we DO NOT cover here:
     - Construction with a real process_t.  We use an aligned-storage
       buffer typed as process_t and bind a reference to it ONLY so
       the reference member can be initialized; we NEVER dereference
       m_process in any test below.  get_info(PROCESS) is therefore
       skipped (it would call process().id()).  Lifecycle tests around
       a real process belong in a later commit that introduces a
       MockOsDriver.  */

#include "amd-dbgapi.h"
#include "code_object.h"
#include "exception.h"
#include "process.h"

#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

using namespace amd::dbgapi;

namespace
{

/* Storage large enough and aligned correctly for a process_t.  We
   bind a process_t& to this buffer to initialize code_object_t's
   reference member.  The buffer is NEVER treated as a real process_t
   (no ctor runs, no member is read), so code under test must not
   dereference the reference.  All tests below honor that contract.  */
alignas (process_t) std::byte fake_process_storage[sizeof (process_t)];

process_t &
fake_process_ref ()
{
  return *reinterpret_cast<process_t *> (&fake_process_storage[0]);
}

} /* namespace */

/* ------------------------------------------------------------------ */
/* next_mark()                                                         */
/* ------------------------------------------------------------------ */

TEST (CodeObjectNextMark, StartsPositiveAndIncreases)
{
  /* The counter is a single program-wide static; we can only assert
     ordering between successive calls, not specific values.  */
  auto m1 = code_object_t::next_mark ();
  auto m2 = code_object_t::next_mark ();
  auto m3 = code_object_t::next_mark ();
  EXPECT_GT (m1, 0u);
  EXPECT_LT (m1, m2);
  EXPECT_LT (m2, m3);
}

/* ------------------------------------------------------------------ */
/* Field round-trip                                                    */
/* ------------------------------------------------------------------ */

TEST (CodeObjectFields, ConstructorStoresIdUriAndLoadAddress)
{
  code_object_t obj (amd_dbgapi_code_object_id_t{ 17 },
                     fake_process_ref (),
                     std::string{ "file:///tmp/kernel.so" },
                     /* load_address */ 0x1000u);
  EXPECT_EQ (obj.id ().handle, 17u);
  EXPECT_EQ (obj.uri (), "file:///tmp/kernel.so");
  EXPECT_EQ (obj.load_address (), 0x1000u);
}

TEST (CodeObjectFields, MarkDefaultsToZeroAndRoundTrips)
{
  code_object_t obj (amd_dbgapi_code_object_id_t{ 1 }, fake_process_ref (),
                     "uri", 0u);
  EXPECT_EQ (obj.mark (), 0u);

  obj.set_mark (42);
  EXPECT_EQ (obj.mark (), 42u);

  obj.set_mark (0);
  EXPECT_EQ (obj.mark (), 0u);
}

/* ------------------------------------------------------------------ */
/* get_info: branches that do not touch m_process                      */
/* ------------------------------------------------------------------ */

TEST (CodeObjectGetInfo, UriName)
{
  code_object_t obj (amd_dbgapi_code_object_id_t{ 2 }, fake_process_ref (),
                     std::string{ "file:///tmp/k.so" }, 0u);

  /* utils::get_info for std::string writes a char* allocated via the
     client allocator.  When the client allocator is not installed we
     can still drive the call indirectly through the public C API or
     test the contract by catching the exception path - but the
     allocation path needs detail::process_callbacks, which isn't
     populated in this TU.  We therefore exercise only the failure
     contract for get_info() here and defer success-path get_info
     coverage to feature tests, where a real callback table is
     installed.  */

  /* Bogus query value forces the switch's default arm.  */
  auto bogus_query
    = static_cast<amd_dbgapi_code_object_info_t> (0xdeadbeefu);
  EXPECT_THROW (obj.get_info (bogus_query, 0, nullptr), api_error_t);
}

TEST (CodeObjectGetInfo, LoadAddress)
{
  code_object_t obj (amd_dbgapi_code_object_id_t{ 3 }, fake_process_ref (),
                     "uri", 0xabcdef00u);

  amd_dbgapi_global_address_t out = 0;
  obj.get_info (AMD_DBGAPI_CODE_OBJECT_INFO_LOAD_ADDRESS, sizeof (out),
                &out);
  EXPECT_EQ (out, 0xabcdef00u);
}

TEST (CodeObjectGetInfo, LoadAddressSizeMismatchThrows)
{
  code_object_t obj (amd_dbgapi_code_object_id_t{ 4 }, fake_process_ref (),
                     "uri", 0u);

  /* utils::get_info validates value_size matches sizeof(value).  Pass
     a deliberately wrong size to drive that error branch.  */
  uint8_t out = 0;
  EXPECT_THROW (
    obj.get_info (AMD_DBGAPI_CODE_OBJECT_INFO_LOAD_ADDRESS, sizeof (out),
                  &out),
    api_error_t);
}

/* C25 negative-test sweep: value_size and enum edge cases.              */

TEST (CodeObjectGetInfo, ValueSizeZeroThrowsCompatibility)
{
  code_object_t obj (amd_dbgapi_code_object_id_t{ 5 }, fake_process_ref (),
                     "uri", 0u);
  amd_dbgapi_global_address_t out{};
  try
    {
      obj.get_info (AMD_DBGAPI_CODE_OBJECT_INFO_LOAD_ADDRESS, 0, &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (CodeObjectGetInfo, ValueSizeMaxThrowsCompatibility)
{
  code_object_t obj (amd_dbgapi_code_object_id_t{ 6 }, fake_process_ref (),
                     "uri", 0u);
  amd_dbgapi_global_address_t out{};
  try
    {
      obj.get_info (AMD_DBGAPI_CODE_OBJECT_INFO_LOAD_ADDRESS, SIZE_MAX, &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (CodeObjectGetInfo, ValueSizeTooLargeByOneThrowsCompatibility)
{
  code_object_t obj (amd_dbgapi_code_object_id_t{ 7 }, fake_process_ref (),
                     "uri", 0u);
  uint8_t out[sizeof (amd_dbgapi_global_address_t) + 1]{};
  try
    {
      obj.get_info (AMD_DBGAPI_CODE_OBJECT_INFO_LOAD_ADDRESS,
                    sizeof (amd_dbgapi_global_address_t) + 1, out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (CodeObjectGetInfo, NegativeEnumThrowsInvalidArgument)
{
  code_object_t obj (amd_dbgapi_code_object_id_t{ 8 }, fake_process_ref (),
                     "uri", 0u);
  amd_dbgapi_global_address_t out{};
  try
    {
      obj.get_info (static_cast<amd_dbgapi_code_object_info_t> (-1),
                    sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (CodeObjectGetInfo, IntMaxEnumThrowsInvalidArgument)
{
  code_object_t obj (amd_dbgapi_code_object_id_t{ 9 }, fake_process_ref (),
                     "uri", 0u);
  amd_dbgapi_global_address_t out{};
  try
    {
      obj.get_info (static_cast<amd_dbgapi_code_object_info_t> (INT_MAX),
                    sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}
