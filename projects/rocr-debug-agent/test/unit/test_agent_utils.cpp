// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "agent_utils.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace amd::debug_agent;

/* The bytes are stored least significant first, and hex_string prints the
   most significant byte first.  */
TEST (AgentUtilsTest, HexStringReversesBytes)
{
  EXPECT_EQ (hex_string (std::vector<uint8_t>{ 0x01, 0xab }), "ab01");
}

/* A file:// URI has protocol "file" and the decoded path after "://".  */
TEST (AgentUtilsTest, ParseFileUri)
{
  parsed_uri_t uri = parse_code_object_uri ("file:///tmp/a.out");
  EXPECT_EQ (uri.protocol, "file");
  EXPECT_EQ (uri.decoded_path, "/tmp/a.out");
}

namespace
{

using params_t = std::unordered_map<std::string, std::string>;

struct uri_case
{
  const char *uri;
  const char *protocol;
  const char *decoded_path;
  params_t params;
};

void
check_uri_cases (const std::vector<uri_case> &cases)
{
  for (const uri_case &c : cases)
    {
      SCOPED_TRACE (c.uri);
      parsed_uri_t uri = parse_code_object_uri (c.uri);
      EXPECT_EQ (uri.protocol, c.protocol);
      EXPECT_EQ (uri.decoded_path, c.decoded_path);
      EXPECT_EQ (uri.params, c.params);
    }
}

} /* namespace */

/* The protocol is lowercased and the path ends at the first '#' or '?'.  */
TEST (AgentUtilsTest, ParseCodeObjectUriComponents)
{
  check_uri_cases ({
    { "file:///tmp/a.out", "file", "/tmp/a.out", {} },
    { "file:///tmp/a.out#offset=4096&size=512",
      "file",
      "/tmp/a.out",
      { { "offset", "4096" }, { "size", "512" } } },
    { "file:///tmp/a.out?offset=4096&size=512",
      "file",
      "/tmp/a.out",
      { { "offset", "4096" }, { "size", "512" } } },
    { "FILE://x", "file", "x", {} },
    { "memory://1234#offset=0x7f00&size=4096",
      "memory",
      "1234",
      { { "offset", "0x7f00" }, { "size", "4096" } } },
    { "/tmp/a.out", "", "/tmp/a.out", {} },
    { "a.out?k=v", "", "a.out", { { "k", "v" } } },
    { "", "", "", {} },
    { "file://", "file", "", {} },
  });
}

/* Only complete %XX escapes in the path are decoded.  */
TEST (AgentUtilsTest, ParseCodeObjectUriPercentDecoding)
{
  check_uri_cases ({
    { "file:///a%20b", "file", "/a b", {} },
    { "file:///a%2Fb", "file", "/a/b", {} },
    { "file:///a%2fb", "file", "/a/b", {} },
    { "file:///a%41", "file", "/aA", {} },
    { "file:///a%", "file", "/a%", {} },
    { "file:///a%4", "file", "/a%4", {} },
    { "file:///a%zz", "file", "/a%zz", {} },
    { "file:///a%41#k=v", "file", "/aA", { { "k", "v" } } },
  });
}

/* Parameters split at the first '=', the first duplicate wins, and values
   are not decoded.  */
TEST (AgentUtilsTest, ParseCodeObjectUriParams)
{
  check_uri_cases ({
    { "file:///a#k=", "file", "/a", { { "k", "" } } },
    { "file:///a#k=a=b", "file", "/a", { { "k", "a=b" } } },
    { "file:///a#flag&k=v", "file", "/a", { { "k", "v" } } },
    { "file:///a#k=1&k=2", "file", "/a", { { "k", "1" } } },
    { "file:///a#k=%41", "file", "/a", { { "k", "%41" } } },
  });
}

/* Each of ": / # ? & =" becomes '_' and other characters are kept.  */
TEST (AgentUtilsTest, SanitizeUriForFilename)
{
  const struct
  {
    const char *uri;
    const char *expected;
  } cases[] = {
    { "file:///tmp/a.out#offset=1&size=2",
      "file____tmp_a.out_offset_1_size_2" },
    { ":/#?&=", "______" },
    { "", "" },
    { "a%20b.c -d", "a%20b.c -d" },
  };

  for (const auto &c : cases)
    {
      SCOPED_TRACE (c.uri);
      EXPECT_EQ (sanitize_uri_for_filename (c.uri), c.expected);
    }
}

/* Strings without a known token are returned unchanged, except "%%".  */
TEST (AgentUtilsTest, ExpandFormatTokensLiteral)
{
  const struct
  {
    const char *format;
    const char *expected;
  } cases[] = {
    { "", "" },     { "/tmp/trace.log", "/tmp/trace.log" },
    { "%%", "%" },  { "%%p", "%p" },
    { "%x", "%x" }, { "abc%", "abc%" },
  };

  for (const auto &c : cases)
    {
      SCOPED_TRACE (c.format);
      EXPECT_EQ (expand_format_tokens (c.format), c.expected);
    }
}

/* %p, %u and %g expand to the process, user and group IDs.  */
TEST (AgentUtilsTest, ExpandFormatTokensIds)
{
  EXPECT_EQ (expand_format_tokens ("%p"), std::to_string (::getpid ()));
  EXPECT_EQ (expand_format_tokens ("%u"), std::to_string (::getuid ()));
  EXPECT_EQ (expand_format_tokens ("%g"), std::to_string (::getgid ()));
}

/* %h expands to the host name.  */
TEST (AgentUtilsTest, ExpandFormatTokensHostname)
{
  char hostname[256] = {};
  ASSERT_EQ (::gethostname (hostname, sizeof (hostname) - 1), 0);
  EXPECT_EQ (expand_format_tokens ("%h"), hostname);
}

/* %e expands to the executable name from /proc/self/comm.  */
TEST (AgentUtilsTest, ExpandFormatTokensExecutable)
{
  std::ifstream comm ("/proc/self/comm");
  std::string name;
  if (!comm || !std::getline (comm, name))
    GTEST_SKIP () << "/proc/self/comm is not readable";

  EXPECT_FALSE (name.empty ());
  EXPECT_EQ (expand_format_tokens ("%e"), name);
}

/* %t expands to the current Unix time in seconds.  */
TEST (AgentUtilsTest, ExpandFormatTokensTime)
{
  const std::time_t before = std::time (nullptr);
  const std::string expanded = expand_format_tokens ("%t");
  const std::time_t after = std::time (nullptr);

  ASSERT_FALSE (expanded.empty ());
  ASSERT_TRUE (std::all_of (expanded.begin (), expanded.end (),
                            [] (unsigned char c) { return std::isdigit (c); }))
    << expanded;
  const long long seconds = std::stoll (expanded);
  EXPECT_GE (seconds, static_cast<long long> (before));
  EXPECT_LE (seconds, static_cast<long long> (after));
}

/* Several tokens in one path are all expanded.  */
TEST (AgentUtilsTest, ExpandFormatTokensCombined)
{
  EXPECT_EQ (expand_format_tokens ("/tmp/trace-%p-%u-%g.log"),
             "/tmp/trace-" + std::to_string (::getpid ()) + "-"
               + std::to_string (::getuid ()) + "-"
               + std::to_string (::getgid ()) + ".log");
}
