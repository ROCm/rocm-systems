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

/* Unit tests for src/exception.h.  Covers the publicly-constructible
   exceptions (api_error_t, fatal_error_t, process_exited_exception_t);
   the memory_*_t variants need address_space_t which depends on
   architecture state and is exercised in later commits.  */

#include "exception.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

using namespace amd::dbgapi;

TEST (Exception, ApiErrorCarriesCodeAndMessage)
{
  api_error_t e (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT, "bad arg");
  EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_STREQ (e.what (), "bad arg");
}

TEST (Exception, ApiErrorEmptyMessage)
{
  api_error_t e (AMD_DBGAPI_STATUS_ERROR);
  EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR);
  EXPECT_STREQ (e.what (), "");
}

TEST (Exception, FatalErrorSetsStatusFatal)
{
  fatal_error_t e ("oh no");
  EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_FATAL);
  EXPECT_STREQ (e.what (), "oh no");
}

TEST (Exception, ProcessExitedCarriesProcessId)
{
  amd_dbgapi_process_id_t pid{ 1234 };
  process_exited_exception_t e (pid, "gone");
  EXPECT_EQ (e.process_id ().handle, pid.handle);
  EXPECT_STREQ (e.what (), "gone");
}

TEST (Exception, Hierarchy)
{
  /* api_error_t and process_exited_exception_t both derive from
     exception_t which derives from std::runtime_error.  Catching the
     base must work for both.  */
  try
    {
      throw api_error_t (AMD_DBGAPI_STATUS_ERROR, "x");
    }
  catch (const exception_t &e)
    {
      EXPECT_STREQ (e.what (), "x");
    }

  try
    {
      throw process_exited_exception_t (amd_dbgapi_process_id_t{ 7 }, "y");
    }
  catch (const std::runtime_error &e)
    {
      EXPECT_STREQ (e.what (), "y");
    }

  try
    {
      throw fatal_error_t ("z");
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_FATAL);
    }
}

TEST (Exception, ThrowMacro)
{
  /* THROW expands to `throw api_error_t (code)`.  */
  EXPECT_THROW (
    {
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    },
    api_error_t);
}
