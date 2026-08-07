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

#include <variant>

using namespace amd::debug_agent;

/* Helper to extract options from variant result. */
static debug_agent_options_t
get_options (const char *env_str)
{
  auto result = parse_debug_agent_options (env_str);
  EXPECT_TRUE (std::holds_alternative<debug_agent_options_t> (result))
      << "Expected success, got error: "
      << (std::holds_alternative<std::string> (result)
              ? std::get<std::string> (result)
              : "unknown");
  return std::get<debug_agent_options_t> (result);
}

TEST (OptionsParsingTest, NullInput_DefaultValues)
{
  debug_agent_options_t opts = get_options (nullptr);
  EXPECT_FALSE (opts.all_wavefronts);
  EXPECT_FALSE (opts.disable_sigquit);
  EXPECT_FALSE (opts.precise_memory);
  EXPECT_FALSE (opts.precise_alu_exceptions);
  EXPECT_TRUE (opts.lazy);
  EXPECT_FALSE (opts.delay_loading);
  EXPECT_FALSE (opts.code_objects_dir.has_value ());
  EXPECT_FALSE (opts.output_file.has_value ());
  EXPECT_EQ (opts.log_level, log_level_t::warning);
}

TEST (OptionsParsingTest, EmptyString_DefaultValues)
{
  debug_agent_options_t opts = get_options ("");
  EXPECT_FALSE (opts.all_wavefronts);
  EXPECT_TRUE (opts.lazy);
  EXPECT_EQ (opts.log_level, log_level_t::warning);
}

TEST (OptionsParsingTest, AllFlag)
{
  debug_agent_options_t opts = get_options ("--all");
  EXPECT_TRUE (opts.all_wavefronts);
}

TEST (OptionsParsingTest, DisableSigquitFlag)
{
  debug_agent_options_t opts = get_options ("--disable-linux-signals");
  EXPECT_TRUE (opts.disable_sigquit);
}

TEST (OptionsParsingTest, PreciseMemoryFlag)
{
  debug_agent_options_t opts = get_options ("--precise-memory");
  EXPECT_TRUE (opts.precise_memory);
}

TEST (OptionsParsingTest, PreciseAluFlag)
{
  debug_agent_options_t opts = get_options ("--precise-alu-exceptions");
  EXPECT_TRUE (opts.precise_alu_exceptions);
}

TEST (OptionsParsingTest, LoadAllCodeObjectsFlag)
{
  debug_agent_options_t opts = get_options ("--load-all-code-objects");
  EXPECT_FALSE (opts.lazy);
}

TEST (OptionsParsingTest, LazyFlag)
{
  debug_agent_options_t opts = get_options ("--lazy");
  EXPECT_TRUE (opts.delay_loading);
}

TEST (OptionsParsingTest, LogLevel_Info)
{
  debug_agent_options_t opts = get_options ("--log-level=info");
  EXPECT_EQ (opts.log_level, log_level_t::info);
}

TEST (OptionsParsingTest, LogLevel_Verbose)
{
  debug_agent_options_t opts = get_options ("--log-level=verbose");
  EXPECT_EQ (opts.log_level, log_level_t::verbose);
}

TEST (OptionsParsingTest, LogLevel_Warning)
{
  debug_agent_options_t opts = get_options ("--log-level=warning");
  EXPECT_EQ (opts.log_level, log_level_t::warning);
}

TEST (OptionsParsingTest, LogLevel_Error)
{
  debug_agent_options_t opts = get_options ("--log-level=error");
  EXPECT_EQ (opts.log_level, log_level_t::error);
}

TEST (OptionsParsingTest, OutputFile)
{
  debug_agent_options_t opts = get_options ("--output=/tmp/out.txt");
  EXPECT_TRUE (opts.output_file.has_value ());
  EXPECT_EQ (opts.output_file.value (), "/tmp/out.txt");
}

TEST (OptionsParsingTest, CodeObjectsDirectory)
{
  debug_agent_options_t opts = get_options ("--save-code-objects=/tmp");
  EXPECT_TRUE (opts.code_objects_dir.has_value ());
  EXPECT_EQ (opts.code_objects_dir.value (), "/tmp");
}

TEST (OptionsParsingTest, MultipleFlags_Combined)
{
  debug_agent_options_t opts = get_options (
      "--all --precise-memory --log-level=verbose");
  EXPECT_TRUE (opts.all_wavefronts);
  EXPECT_TRUE (opts.precise_memory);
  EXPECT_EQ (opts.log_level, log_level_t::verbose);
}

TEST (OptionsParsingTest, AllOptions_Combined)
{
  debug_agent_options_t opts = get_options (
      "--all --disable-linux-signals --precise-memory "
      "--precise-alu-exceptions --load-all-code-objects "
      "--output=/tmp/log.txt --save-code-objects=/tmp "
      "--log-level=info");

  EXPECT_TRUE (opts.all_wavefronts);
  EXPECT_TRUE (opts.disable_sigquit);
  EXPECT_TRUE (opts.precise_memory);
  EXPECT_TRUE (opts.precise_alu_exceptions);
  EXPECT_FALSE (opts.lazy);
  EXPECT_EQ (opts.output_file.value (), "/tmp/log.txt");
  EXPECT_EQ (opts.code_objects_dir.value (), "/tmp");
  EXPECT_EQ (opts.log_level, log_level_t::info);
}

/* Error path tests - verify function returns error strings. */

TEST (OptionsParsingTest, InvalidLogLevel_ReturnsError)
{
  auto result = parse_debug_agent_options ("--log-level=invalid");
  EXPECT_TRUE (std::holds_alternative<std::string> (result));
  EXPECT_EQ (std::get<std::string> (result), "error: Invalid log level 'invalid'");
}

TEST (OptionsParsingTest, LogLevelMissingArgument_ReturnsError)
{
  auto result = parse_debug_agent_options ("--log-level");
  EXPECT_TRUE (std::holds_alternative<std::string> (result));
  EXPECT_EQ (std::get<std::string> (result), "error: --log-level requires an argument");
}

TEST (OptionsParsingTest, OutputMissingArgument_ReturnsError)
{
  auto result = parse_debug_agent_options ("--output");
  EXPECT_TRUE (std::holds_alternative<std::string> (result));
  EXPECT_EQ (std::get<std::string> (result), "error: --output requires an argument");
}

TEST (OptionsParsingTest, InvalidDirectory_ReturnsError)
{
  auto result = parse_debug_agent_options ("--save-code-objects=/nonexistent/path");
  EXPECT_TRUE (std::holds_alternative<std::string> (result));
  std::string error = std::get<std::string> (result);
  EXPECT_TRUE (error.find ("Cannot access code object save directory") != std::string::npos);
}

TEST (OptionsParsingTest, MutuallyExclusiveFlags_ReturnsError)
{
  auto result = parse_debug_agent_options ("--load-all-code-objects --lazy");
  EXPECT_TRUE (std::holds_alternative<std::string> (result));
  std::string error = std::get<std::string> (result);
  EXPECT_TRUE (error.find ("mutually exclusive") != std::string::npos);
}

TEST (OptionsParsingTest, UnrecognizedOption_ReturnsError)
{
  auto result = parse_debug_agent_options ("--unknown-flag");
  EXPECT_TRUE (std::holds_alternative<std::string> (result));
  EXPECT_EQ (std::get<std::string> (result), "error: Unrecognized option");
}

TEST (OptionsParsingTest, HelpFlag_ReturnsHelp)
{
  auto result = parse_debug_agent_options ("--help");
  EXPECT_TRUE (std::holds_alternative<std::string> (result));
  EXPECT_EQ (std::get<std::string> (result), "help");
}
