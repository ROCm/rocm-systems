// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "agent_config.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <variant>

using namespace amd::debug_agent;

/* -h only sets help_requested and stops parsing; printing the usage and
   exiting are left to the caller.  */
TEST (AgentConfigTest, HelpRequested)
{
  auto result = parse_debug_agent_options ("-h");
  ASSERT_TRUE (std::holds_alternative<debug_agent_options_t> (result));
  EXPECT_TRUE (std::get<debug_agent_options_t> (result).help_requested);
}

namespace
{

const std::string mutual_exclusion_error
  = "error: \"--load-all-code-objects\" and \"--lazy\" are "
    "mutually exclusive";

/* Parse STR and return the options, failing the test on an error.  */
debug_agent_options_t
expect_options (const char *str)
{
  auto result = parse_debug_agent_options (str);
  EXPECT_TRUE (std::holds_alternative<debug_agent_options_t> (result))
    << "input: " << (str ? str : "(null)");
  if (!std::holds_alternative<debug_agent_options_t> (result))
    return {};
  return std::get<debug_agent_options_t> (result);
}

/* Parse STR and return the error message, failing the test on success.  */
std::string
expect_error (const char *str)
{
  auto result = parse_debug_agent_options (str);
  EXPECT_TRUE (std::holds_alternative<std::string> (result))
    << "input: " << (str ? str : "(null)");
  if (!std::holds_alternative<std::string> (result))
    return {};
  return std::get<std::string> (result);
}

/* Check that every field of OPTIONS has its default value.  */
void
expect_defaults (const debug_agent_options_t &options)
{
  EXPECT_FALSE (options.all_wavefronts);
  EXPECT_FALSE (options.disable_sigquit);
  EXPECT_FALSE (options.precise_memory);
  EXPECT_FALSE (options.precise_alu_exceptions);
  EXPECT_TRUE (options.lazy);
  EXPECT_FALSE (options.delay_loading);
  EXPECT_FALSE (options.code_objects_dir.has_value ());
  EXPECT_FALSE (options.output_file.has_value ());
  EXPECT_EQ (options.log_level, log_level_t::warning);
  EXPECT_FALSE (options.help_requested);
}

} /* anonymous namespace */

/* A null option string gives the default options.  */
TEST (AgentConfigTest, NullGivesDefaults)
{
  expect_defaults (expect_options (nullptr));
}

/* Empty and whitespace-only strings give the default options.  */
TEST (AgentConfigTest, EmptyAndWhitespaceGiveDefaults)
{
  expect_defaults (expect_options (""));
  expect_defaults (expect_options ("   \t \n "));
}

/* Each boolean flag, in short and long form, sets only its own field.  */
TEST (AgentConfigTest, BooleanFlags)
{
  struct row
  {
    const char *input;
    bool debug_agent_options_t::*field;
  };

  const row rows[] = {
    { "-a", &debug_agent_options_t::all_wavefronts },
    { "--all", &debug_agent_options_t::all_wavefronts },
    { "-d", &debug_agent_options_t::disable_sigquit },
    { "--disable-linux-signals", &debug_agent_options_t::disable_sigquit },
    { "-p", &debug_agent_options_t::precise_memory },
    { "--precise-memory", &debug_agent_options_t::precise_memory },
    { "-e", &debug_agent_options_t::precise_alu_exceptions },
    { "--precise-alu-exceptions",
      &debug_agent_options_t::precise_alu_exceptions },
    { "-z", &debug_agent_options_t::delay_loading },
    { "--lazy", &debug_agent_options_t::delay_loading },
  };

  for (const row &r : rows)
    {
      debug_agent_options_t expected;
      expected.*r.field = true;

      debug_agent_options_t actual = expect_options (r.input);
      EXPECT_EQ (actual.all_wavefronts, expected.all_wavefronts) << r.input;
      EXPECT_EQ (actual.disable_sigquit, expected.disable_sigquit) << r.input;
      EXPECT_EQ (actual.precise_memory, expected.precise_memory) << r.input;
      EXPECT_EQ (actual.precise_alu_exceptions,
                 expected.precise_alu_exceptions)
        << r.input;
      EXPECT_EQ (actual.delay_loading, expected.delay_loading) << r.input;
      EXPECT_TRUE (actual.lazy) << r.input;
    }
}

/* Several flags can be combined in one string.  */
TEST (AgentConfigTest, CombinedFlags)
{
  debug_agent_options_t options = expect_options ("-a -d -p -e");
  EXPECT_TRUE (options.all_wavefronts);
  EXPECT_TRUE (options.disable_sigquit);
  EXPECT_TRUE (options.precise_memory);
  EXPECT_TRUE (options.precise_alu_exceptions);
  EXPECT_TRUE (options.lazy);
  EXPECT_FALSE (options.delay_loading);
}

/* -c clears lazy; -z alone sets delay_loading and leaves lazy set.  */
TEST (AgentConfigTest, LoadAllAndLazy)
{
  EXPECT_FALSE (expect_options ("-c").lazy);
  EXPECT_FALSE (expect_options ("--load-all-code-objects").lazy);

  debug_agent_options_t options = expect_options ("-z");
  EXPECT_TRUE (options.delay_loading);
  EXPECT_TRUE (options.lazy);
}

/* -c together with -z is rejected, in either order.  */
TEST (AgentConfigTest, LoadAllAndLazyAreMutuallyExclusive)
{
  EXPECT_EQ (expect_error ("-c -z"), mutual_exclusion_error);
  EXPECT_EQ (expect_error ("-z -c"), mutual_exclusion_error);
  EXPECT_EQ (expect_error ("--load-all-code-objects --lazy"),
             mutual_exclusion_error);
}

/* Every log level is accepted in the short, long and separate-word forms.  */
TEST (AgentConfigTest, LogLevels)
{
  struct row
  {
    const char *name;
    log_level_t level;
  };

  const row rows[] = {
    { "none", log_level_t::none },       { "error", log_level_t::error },
    { "warning", log_level_t::warning }, { "info", log_level_t::info },
    { "verbose", log_level_t::verbose },
  };

  for (const row &r : rows)
    {
      const std::string name = r.name;
      EXPECT_EQ (expect_options (("-l" + name).c_str ()).log_level, r.level);
      EXPECT_EQ (expect_options (("-l " + name).c_str ()).log_level, r.level);
      EXPECT_EQ (expect_options (("--log-level=" + name).c_str ()).log_level,
                 r.level);
      EXPECT_EQ (expect_options (("--log-level " + name).c_str ()).log_level,
                 r.level);
    }
}

/* An unknown log level is an error that names the level.  */
TEST (AgentConfigTest, InvalidLogLevel)
{
  EXPECT_EQ (expect_error ("-l bogus"), "error: Invalid log level 'bogus'");
  EXPECT_EQ (expect_error ("--log-level=bogus"),
             "error: Invalid log level 'bogus'");
}

/* Options that need an argument report it when it is missing.  */
TEST (AgentConfigTest, MissingArguments)
{
  EXPECT_EQ (expect_error ("-l"), "error: --log-level requires an argument");
  EXPECT_EQ (expect_error ("--log-level"),
             "error: --log-level requires an argument");
  EXPECT_EQ (expect_error ("-o"), "error: --output requires an argument");
  EXPECT_EQ (expect_error ("--output"),
             "error: --output requires an argument");
}

/* Unknown short and long options are rejected.  */
TEST (AgentConfigTest, UnknownOptions)
{
  EXPECT_EQ (expect_error ("-x"), "error: Unrecognized option");
  EXPECT_EQ (expect_error ("--bogus"), "error: Unrecognized option");
}

/* The output file name has its format tokens expanded.  */
TEST (AgentConfigTest, OutputExpandsTokens)
{
  const std::string pid = std::to_string (getpid ());

  debug_agent_options_t options = expect_options ("-o /tmp/out-%p");
  ASSERT_TRUE (options.output_file.has_value ());
  EXPECT_EQ (*options.output_file, "/tmp/out-" + pid);

  options = expect_options ("--output=/tmp/out-%p-%%");
  ASSERT_TRUE (options.output_file.has_value ());
  EXPECT_EQ (*options.output_file, "/tmp/out-" + pid + "-%");
}

/* Parsing -o must not create the output file; the caller opens it.  */
TEST (AgentConfigTest, OutputFileIsNotCreated)
{
  const std::string path = "/tmp/rocm-debug-agent-config-test-"
                           + std::to_string (getpid ()) + "-%p";
  const std::string expanded = "/tmp/rocm-debug-agent-config-test-"
                               + std::to_string (getpid ()) + "-"
                               + std::to_string (getpid ());

  EXPECT_FALSE (
    expect_options (("-o " + path).c_str ()).output_file->empty ());
  EXPECT_NE (access (expanded.c_str (), F_OK), 0);
  EXPECT_NE (access (path.c_str (), F_OK), 0);

  /* A parse failure after -o must not touch the file either.  */
  expect_error (("-o " + path + " -x").c_str ());
  EXPECT_NE (access (expanded.c_str (), F_OK), 0);
}

/* Fixture that provides a scratch directory for -s tests.  */
class AgentConfigSaveDirTest : public testing::Test
{
protected:
  void SetUp () override
  {
    char tmpl[] = "/tmp/rocm-debug-agent-config-XXXXXX";
    ASSERT_NE (mkdtemp (tmpl), nullptr);
    m_dir = tmpl;
  }

  void TearDown () override
  {
    if (!m_file.empty ())
      unlink (m_file.c_str ());
    if (!m_dir.empty ())
      rmdir (m_dir.c_str ());
  }

  /* Create a regular file in the scratch directory and return its path.  */
  std::string make_file ()
  {
    m_file = m_dir + "/file";
    FILE *f = fopen (m_file.c_str (), "w");
    EXPECT_NE (f, nullptr);
    if (f)
      fclose (f);
    return m_file;
  }

  std::string m_dir;
  std::string m_file;
};

/* -s without a directory saves into the current directory.  */
TEST (AgentConfigTest, SaveCodeObjectsDefaultsToCurrentDirectory)
{
  for (const char *input : { "-s", "--save-code-objects" })
    {
      debug_agent_options_t options = expect_options (input);
      ASSERT_TRUE (options.code_objects_dir.has_value ()) << input;
      EXPECT_EQ (*options.code_objects_dir, ".") << input;
    }
}

/* -s followed by another option does not swallow that option.  */
TEST (AgentConfigTest, SaveCodeObjectsBeforeOption)
{
  debug_agent_options_t options = expect_options ("-s -a");
  ASSERT_TRUE (options.code_objects_dir.has_value ());
  EXPECT_EQ (*options.code_objects_dir, ".");
  EXPECT_TRUE (options.all_wavefronts);
}

/* -s accepts an existing directory in every spelling.  */
TEST_F (AgentConfigSaveDirTest, ExistingDirectory)
{
  for (const std::string &input :
       { "-s " + m_dir, "-s" + m_dir, "--save-code-objects=" + m_dir,
         "--save-code-objects " + m_dir })
    {
      debug_agent_options_t options = expect_options (input.c_str ());
      ASSERT_TRUE (options.code_objects_dir.has_value ()) << input;
      EXPECT_EQ (*options.code_objects_dir, m_dir) << input;
    }
}

/* A missing directory and a regular file are both rejected.  */
TEST_F (AgentConfigSaveDirTest, InvalidDirectory)
{
  const std::string prefix = "error: Cannot access code object save directory";

  EXPECT_EQ (expect_error (("-s " + m_dir + "/missing").c_str ())
               .compare (0, prefix.size (), prefix),
             0);
  EXPECT_EQ (expect_error (("--save-code-objects=" + make_file ()).c_str ())
               .compare (0, prefix.size (), prefix),
             0);
}

/* -h sets help_requested and keeps the options parsed before it.  */
TEST (AgentConfigTest, HelpKeepsEarlierOptions)
{
  EXPECT_TRUE (expect_options ("--help").help_requested);

  debug_agent_options_t options = expect_options ("-a -h");
  EXPECT_TRUE (options.help_requested);
  EXPECT_TRUE (options.all_wavefronts);
}

/* Options after -h are never parsed, so an error there is not reported.  */
TEST (AgentConfigTest, HelpStopsParsing)
{
  EXPECT_TRUE (expect_options ("-h -x").help_requested);
  EXPECT_TRUE (expect_options ("-h -c -z").help_requested);
  EXPECT_EQ (expect_error ("-x -h"), "error: Unrecognized option");
}

/* The global optind is left unchanged after success, error and -h.  */
TEST (AgentConfigTest, OptindIsRestored)
{
  const int saved = optind;
  const char *inputs[] = { "-a -d", "-x", "-l bogus", "-h", "-a -h -x" };

  for (const char *input : inputs)
    {
      optind = 7;
      parse_debug_agent_options (input);
      EXPECT_EQ (optind, 7) << input;
    }

  optind = saved;
}

/* Parsing the same string twice gives the same result.  */
TEST (AgentConfigTest, RepeatedParsingIsStable)
{
  const char *input = "-a -l info -o /tmp/x -p";
  debug_agent_options_t first = expect_options (input);
  debug_agent_options_t second = expect_options (input);

  EXPECT_EQ (first.all_wavefronts, second.all_wavefronts);
  EXPECT_EQ (first.precise_memory, second.precise_memory);
  EXPECT_EQ (first.log_level, second.log_level);
  EXPECT_EQ (first.output_file, second.output_file);
  EXPECT_EQ (first.log_level, log_level_t::info);
}

/* Parsing -l changes the returned options only, not the global level.  */
TEST (AgentConfigTest, ParsingDoesNotChangeGlobalLogLevel)
{
  const log_level_t saved = log_level;

  EXPECT_EQ (expect_options ("-l verbose").log_level, log_level_t::verbose);
  EXPECT_EQ (log_level, saved);

  log_level = saved;
}

/* print_usage writes the usage text, with every long option, to stderr.  */
TEST (AgentConfigTest, PrintUsageListsOptions)
{
  testing::internal::CaptureStderr ();
  print_usage ();
  const std::string out = testing::internal::GetCapturedStderr ();

  EXPECT_NE (out.find ("ROCdebug-agent usage:"), std::string::npos);
  for (const char *name :
       { "--all", "--save-code-objects", "--load-all-code-objects", "--lazy",
         "--precise-memory", "--output", "--disable-linux-signals",
         "--log-level", "--help" })
    EXPECT_NE (out.find (name), std::string::npos) << name;
}
