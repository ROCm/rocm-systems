/* The University of Illinois/NCSA
   Open Source License (NCSA)

   Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal with the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimers.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimers in
      the documentation and/or other materials provided with the distribution.
    - Neither the names of Advanced Micro Devices, Inc,
      nor the names of its contributors may be used to endorse or promote
      products derived from this Software without specific prior written
      permission.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
   OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS WITH THE SOFTWARE.  */

#include "agent_utils.h"

#include <gtest/gtest.h>

using namespace amd::debug_agent;

TEST (ParseURITest, FileProtocol_AbsolutePath)
{
  parsed_uri_t result = parse_code_object_uri ("file:///path/to/file.so");
  EXPECT_EQ (result.protocol, "file");
  EXPECT_EQ (result.decoded_path, "/path/to/file.so");
  EXPECT_TRUE (result.params.empty ());
}

TEST (ParseURITest, FileProtocol_NoTripleSlash)
{
  parsed_uri_t result = parse_code_object_uri ("file://path/to/file.so");
  EXPECT_EQ (result.protocol, "file");
  EXPECT_EQ (result.decoded_path, "path/to/file.so");
}

TEST (ParseURITest, PercentDecoding_Spaces)
{
  parsed_uri_t result = parse_code_object_uri ("file:///path%20with%20spaces");
  EXPECT_EQ (result.decoded_path, "/path with spaces");
}

TEST (ParseURITest, PercentDecoding_SpecialChars)
{
  parsed_uri_t result = parse_code_object_uri ("file:///path%2Fwith%2Fslashes");
  EXPECT_EQ (result.decoded_path, "/path/with/slashes");
}

TEST (ParseURITest, MemoryProtocol_QueryParams)
{
  parsed_uri_t result = parse_code_object_uri ("memory://0x1000?offset=0&size=1024");
  EXPECT_EQ (result.protocol, "memory");
  EXPECT_EQ (result.decoded_path, "0x1000");
  EXPECT_EQ (result.params.at ("offset"), "0");
  EXPECT_EQ (result.params.at ("size"), "1024");
}

TEST (ParseURITest, QueryParams_MultipleParams)
{
  parsed_uri_t result = parse_code_object_uri ("memory://addr?offset=100&size=2048&flags=rw");
  EXPECT_EQ (result.params.size (), 3);
  EXPECT_EQ (result.params.at ("offset"), "100");
  EXPECT_EQ (result.params.at ("size"), "2048");
  EXPECT_EQ (result.params.at ("flags"), "rw");
}

TEST (ParseURITest, NoQueryParams)
{
  parsed_uri_t result = parse_code_object_uri ("file:///simple/path.so");
  EXPECT_TRUE (result.params.empty ());
}

TEST (ParseURITest, Fragment_Ignored)
{
  parsed_uri_t result = parse_code_object_uri ("file:///path/file.so#fragment");
  EXPECT_EQ (result.decoded_path, "/path/file.so");
}

TEST (SanitizeURITest, ReplaceColons)
{
  EXPECT_EQ (sanitize_uri_for_filename ("file://path:name"), "file___path_name");
}

TEST (SanitizeURITest, ReplaceSlashes)
{
  EXPECT_EQ (sanitize_uri_for_filename ("file://a/b/c"), "file___a_b_c");
}

TEST (SanitizeURITest, ReplaceHashAndQuestion)
{
  EXPECT_EQ (sanitize_uri_for_filename ("file://path#fragment?query"),
             "file___path_fragment_query");
}

TEST (SanitizeURITest, ReplaceMultipleSpecialChars)
{
  EXPECT_EQ (sanitize_uri_for_filename ("file://path:name#fragment/subdir"),
             "file___path_name_fragment_subdir");
}

TEST (SanitizeURITest, NoSpecialChars)
{
  EXPECT_EQ (sanitize_uri_for_filename ("simple_filename"), "simple_filename");
}

TEST (SanitizeURITest, EmptyString)
{
  EXPECT_EQ (sanitize_uri_for_filename (""), "");
}

TEST (ParseURITest, IncompletePercentEncoding_EndWithPercent)
{
  // URI ending with % should not crash.
  parsed_uri_t result = parse_code_object_uri ("file:///path%");
  EXPECT_EQ (result.decoded_path, "/path%");
}

TEST (ParseURITest, IncompletePercentEncoding_EndWithOneHexDigit)
{
  // URI ending with %X should not crash.
  parsed_uri_t result = parse_code_object_uri ("file:///path%2");
  EXPECT_EQ (result.decoded_path, "/path%2");
}

TEST (ParseURITest, MalformedURI_NoProtocol)
{
  /* URI without :// separator - protocol will be entire string. */
  parsed_uri_t result = parse_code_object_uri ("path/to/file");
  EXPECT_EQ (result.protocol, "path/to/file");
}

TEST (ParseURITest, MalformedURI_NoSeparator)
{
  /* URI without :// separator - protocol will be entire string. */
  parsed_uri_t result = parse_code_object_uri ("file/path/to/file");
  EXPECT_EQ (result.protocol, "file/path/to/file");
}

TEST (ParseURITest, InvalidPercentEncoding_NonHexDigits)
{
  // %ZZ should be copied literally (not decoded).
  parsed_uri_t result = parse_code_object_uri ("file:///path%ZZ");
  EXPECT_EQ (result.decoded_path, "/path%ZZ");
}
