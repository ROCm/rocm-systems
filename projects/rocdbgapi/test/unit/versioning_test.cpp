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

/* Unit tests for versioning.cpp:
     * amd_dbgapi_get_version - nullptr-tolerant per-arg and macro round-trip.
     * amd_dbgapi_get_build_name - non-null, non-empty.
     * amd_dbgapi_get_status_string - every documented enum value maps to a
       non-empty string; nullptr output pointer and unrecognized status both
       return INVALID_ARGUMENT.  The enum table catches the case where a new
       AMD_DBGAPI_STATUS_* value is added to the public header but the switch
       in versioning.cpp forgets a corresponding case (the source intentionally
       omits a default so the compiler warns, but only this test catches the
       runtime-visible miss). */

#include "amd-dbgapi.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

/* AMD_DBGAPI_VERSION_PATCH is supplied to the library compile via -D
   from CMake (PROJECT_VERSION_PATCH) and is intentionally NOT a public
   macro in amd-dbgapi.h, so the test TU only validates major/minor
   against the public macros.  The patch arg is still exercised for
   null-tolerance and for the cross-call self-consistency check below. */

TEST (Versioning, GetVersionMatchesCompiledMacros)
{
  uint32_t major = 0, minor = 0, patch = 0;
  amd_dbgapi_get_version (&major, &minor, &patch);
  EXPECT_EQ (major, static_cast<uint32_t> (AMD_DBGAPI_VERSION_MAJOR));
  EXPECT_EQ (minor, static_cast<uint32_t> (AMD_DBGAPI_VERSION_MINOR));

  /* Self-consistency: calling again must report the same patch value.  */
  uint32_t patch2 = ~patch;
  amd_dbgapi_get_version (nullptr, nullptr, &patch2);
  EXPECT_EQ (patch, patch2);
}

TEST (Versioning, GetVersionTolerantToNullArgs)
{
  /* Any subset of the three out-pointers may be null.  */
  amd_dbgapi_get_version (nullptr, nullptr, nullptr);

  uint32_t v = 0xdeadbeef;
  amd_dbgapi_get_version (&v, nullptr, nullptr);
  EXPECT_EQ (v, static_cast<uint32_t> (AMD_DBGAPI_VERSION_MAJOR));

  v = 0xdeadbeef;
  amd_dbgapi_get_version (nullptr, &v, nullptr);
  EXPECT_EQ (v, static_cast<uint32_t> (AMD_DBGAPI_VERSION_MINOR));

  /* Patch-only write: simply must not crash and must overwrite the
     sentinel.  We don't have the macro here to compare against.  */
  v = 0xdeadbeef;
  amd_dbgapi_get_version (nullptr, nullptr, &v);
  EXPECT_NE (v, 0xdeadbeefu);
}

TEST (Versioning, GetBuildNameIsNonEmpty)
{
  const char *name = amd_dbgapi_get_build_name ();
  ASSERT_NE (name, nullptr);
  EXPECT_GT (std::strlen (name), 0u);
}

TEST (Versioning, GetStatusStringRejectsNullOutPointer)
{
  amd_dbgapi_status_t st
    = amd_dbgapi_get_status_string (AMD_DBGAPI_STATUS_SUCCESS, nullptr);
  EXPECT_EQ (st, AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST (Versioning, GetStatusStringRejectsUnknownStatus)
{
  const char *s = nullptr;
  amd_dbgapi_status_t st = amd_dbgapi_get_status_string (
    static_cast<amd_dbgapi_status_t> (0x7fffffff), &s);
  EXPECT_EQ (st, AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

/* Table of every public AMD_DBGAPI_STATUS_* value.  If a new status is
   added to amd-dbgapi.h, append it here so the round-trip test fails
   loudly when versioning.cpp forgets to map it.  */
static constexpr amd_dbgapi_status_t kAllStatuses[] = {
  AMD_DBGAPI_STATUS_SUCCESS,
  AMD_DBGAPI_STATUS_ERROR,
  AMD_DBGAPI_STATUS_FATAL,
  AMD_DBGAPI_STATUS_ERROR_NOT_IMPLEMENTED,
  AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE,
  AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED,
  AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
  AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
  AMD_DBGAPI_STATUS_ERROR_ALREADY_INITIALIZED,
  AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
  AMD_DBGAPI_STATUS_ERROR_RESTRICTION,
  AMD_DBGAPI_STATUS_ERROR_ALREADY_ATTACHED,
  AMD_DBGAPI_STATUS_ERROR_INVALID_ARCHITECTURE_ID,
  AMD_DBGAPI_STATUS_ERROR_ILLEGAL_INSTRUCTION,
  AMD_DBGAPI_STATUS_ERROR_INVALID_CODE_OBJECT_ID,
  AMD_DBGAPI_STATUS_ERROR_INVALID_ELF_AMDGPU_MACHINE,
  AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID,
  AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED,
  AMD_DBGAPI_STATUS_ERROR_INVALID_AGENT_ID,
  AMD_DBGAPI_STATUS_ERROR_INVALID_QUEUE_ID,
  AMD_DBGAPI_STATUS_ERROR_INVALID_DISPATCH_ID,
  AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID,
  AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED,
  AMD_DBGAPI_STATUS_ERROR_WAVE_STOPPED,
  AMD_DBGAPI_STATUS_ERROR_WAVE_OUTSTANDING_STOP,
  AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_RESUMABLE,
  AMD_DBGAPI_STATUS_ERROR_INVALID_DISPLACED_STEPPING_ID,
  AMD_DBGAPI_STATUS_ERROR_DISPLACED_STEPPING_BUFFER_NOT_AVAILABLE,
  AMD_DBGAPI_STATUS_ERROR_DISPLACED_STEPPING_ACTIVE,
  AMD_DBGAPI_STATUS_ERROR_RESUME_DISPLACED_STEPPING,
  AMD_DBGAPI_STATUS_ERROR_INVALID_WATCHPOINT_ID,
  AMD_DBGAPI_STATUS_ERROR_NO_WATCHPOINT_AVAILABLE,
  AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_CLASS_ID,
  AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID,
  AMD_DBGAPI_STATUS_ERROR_INVALID_LANE_ID,
  AMD_DBGAPI_STATUS_ERROR_INVALID_ADDRESS_CLASS_ID,
  AMD_DBGAPI_STATUS_ERROR_INVALID_ADDRESS_SPACE_ID,
  AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS,
  AMD_DBGAPI_STATUS_ERROR_INVALID_ADDRESS_SPACE_CONVERSION,
  AMD_DBGAPI_STATUS_ERROR_INVALID_EVENT_ID,
  AMD_DBGAPI_STATUS_ERROR_INVALID_BREAKPOINT_ID,
  AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK,
  AMD_DBGAPI_STATUS_ERROR_INVALID_CLIENT_PROCESS_ID,
  AMD_DBGAPI_STATUS_ERROR_SYMBOL_NOT_FOUND,
  AMD_DBGAPI_STATUS_ERROR_REGISTER_NOT_AVAILABLE,
  AMD_DBGAPI_STATUS_ERROR_INVALID_WORKGROUP_ID,
  AMD_DBGAPI_STATUS_ERROR_INCOMPATIBLE_PROCESS_STATE,
  AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN,
  AMD_DBGAPI_STATUS_ERROR_PROCESS_ALREADY_FROZEN,
  AMD_DBGAPI_STATUS_ERROR_PROCESS_NOT_FROZEN,
  AMD_DBGAPI_STATUS_ERROR_MEMORY_UNAVAILABLE,
};

TEST (Versioning, GetStatusStringMapsEveryDocumentedEnum)
{
  for (amd_dbgapi_status_t code : kAllStatuses)
    {
      const char *s = nullptr;
      amd_dbgapi_status_t st = amd_dbgapi_get_status_string (code, &s);
      ASSERT_EQ (st, AMD_DBGAPI_STATUS_SUCCESS)
        << "status code " << static_cast<int> (code)
        << " unexpectedly rejected";
      ASSERT_NE (s, nullptr)
        << "status code " << static_cast<int> (code) << " mapped to null";
      EXPECT_GT (std::strlen (s), 0u)
        << "status code " << static_cast<int> (code)
        << " mapped to empty string";
    }
}
