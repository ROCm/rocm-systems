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

/* Windows-only unit tests for the Event-based notifier_t implementation
   that lives in src/utils_windows.cpp, plus the WS-to-string converter
   in utils_windows.h.  The mark/clear semantics differ from the
   Linux pipe-based implementation tested in utils_linux_test.cpp; the
   contract enforced here is the one documented on notifier_t in
   utils.h.  */

#include "utils.h"
#include "utils_windows.h"

#include <gtest/gtest.h>

#include <windows.h>

#include <memory>
#include <string>

using amd::dbgapi::notifier_t;
using amd::dbgapi::utils::convert_to_string;

TEST (EventNotifier, FactoryReturnsValidObject)
{
  auto n = notifier_t::create ();
  ASSERT_NE (n, nullptr);
  EXPECT_FALSE (n->is_valid ());
}

TEST (EventNotifier, OpenCloseLifecycle)
{
  auto n = notifier_t::create ();
  ASSERT_NE (n, nullptr);

  n->open ();
  EXPECT_TRUE (n->is_valid ());
  /* Producer and consumer share the same HANDLE for the Event-based
     implementation - this is part of the documented contract.  */
  EXPECT_EQ (n->producer_end (), n->consumer_end ());
  EXPECT_NE (n->producer_end (), nullptr);

  n->close ();
  EXPECT_FALSE (n->is_valid ());
}

TEST (EventNotifier, MarkSignalsTheEvent)
{
  auto n = notifier_t::create ();
  ASSERT_NE (n, nullptr);
  n->open ();
  ASSERT_TRUE (n->is_valid ());

  /* Initially not signalled - WaitForSingleObject must time out.  */
  EXPECT_EQ (::WaitForSingleObject (n->consumer_end (), 0), WAIT_TIMEOUT);

  EXPECT_TRUE (n->mark ());
  EXPECT_EQ (::WaitForSingleObject (n->consumer_end (), 0), WAIT_OBJECT_0);

  n->close ();
}

TEST (EventNotifier, ClearResetsTheEvent)
{
  auto n = notifier_t::create ();
  ASSERT_NE (n, nullptr);
  n->open ();
  ASSERT_TRUE (n->is_valid ());

  EXPECT_TRUE (n->mark ());
  EXPECT_TRUE (n->clear ());
  EXPECT_EQ (::WaitForSingleObject (n->consumer_end (), 0), WAIT_TIMEOUT);

  n->close ();
}

TEST (UtilsWindowsConvertToString, EmptyInput)
{
  EXPECT_EQ (convert_to_string (L""), "");
}

TEST (UtilsWindowsConvertToString, BasicAscii)
{
  /* ASCII characters survive the WideCharToMultiByte round-trip
     regardless of the current code page.  */
  EXPECT_EQ (convert_to_string (L"hello"), "hello");
}
