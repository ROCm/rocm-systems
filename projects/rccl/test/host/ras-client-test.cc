/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/client.cc.
//
// client.cc is a standalone executable: every helper is `static` and its whole
// dependency surface is libc plus four macros from ras_internal.h. This TU
// #includes the hipified client.cc directly, so the statics become callable,
// and macro-renames each libc entry point to a micro_* trampoline that
// dispatches through a swappable slot in fakes/ras_client_fakes.h.
//
// getopt_long is deliberately NOT seamed: its parse behaviour is part of what
// parseArgs is being tested for. ResetRasClientGlobals() resets optind instead.

#include <gtest/gtest.h>

// Every libc header client.cc reaches through os.h, pulled in BEFORE the macro
// renames below so the renames never rewrite a declaration.
#include <arpa/inet.h>
#include <getopt.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "fakes/ras_client_fakes.h"

extern "C" {
ssize_t micro_write(int, const void*, size_t);
ssize_t micro_read(int, void*, size_t);
int micro_close(int);
int micro_socket(int, int, int);
int micro_connect(int, const struct sockaddr*, socklen_t);
int micro_setsockopt(int, int, int, const void*, socklen_t);
int micro_getaddrinfo(const char*, const char*, const struct addrinfo*, struct addrinfo**);
void micro_freeaddrinfo(struct addrinfo*);
int micro_getnameinfo(const struct sockaddr*, socklen_t, char*, socklen_t, char*, socklen_t, int);
const char* micro_gai_strerror(int);
size_t micro_fwrite(const void*, size_t, size_t, FILE*);
int micro_fflush(FILE*);
void micro_exit(int) __attribute__((noreturn));
}

#define write micro_write
#define read micro_read
#define close micro_close
#define socket micro_socket
#define connect micro_connect
#define setsockopt micro_setsockopt
#define getaddrinfo micro_getaddrinfo
#define freeaddrinfo micro_freeaddrinfo
#define getnameinfo micro_getnameinfo
#define gai_strerror micro_gai_strerror
#define fwrite micro_fwrite
#define fflush micro_fflush
#define exit micro_exit
// client.cc's main() would collide with the gtest main in main_altrsmi.cpp.
#define main rasClientMain

#include RAS_CLIENT_CC_PATH

#undef main
#undef exit
#undef fflush
#undef fwrite
#undef gai_strerror
#undef getnameinfo
#undef freeaddrinfo
#undef getaddrinfo
#undef setsockopt
#undef connect
#undef socket
#undef close
#undef read
#undef write

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

void ResetRasClientGlobals() {
  hostName = "localhost";
  port = STR(NCCL_RAS_CLIENT_PORT);
  timeout = -1;
  verbose = false;
  monitorMode = false;
  format = nullptr;
  events = nullptr;
  sock = -1;
  // glibc treats optind == 0 (not 1) as "reinitialize everything", which is what
  // repeated getopt_long calls in one process need.
  optind = 0;
}

// ===========================================================================
// Default fixture for every test in this file. Tests that need extra per-test
// state derive from it; report any derived suite name so it can be registered
// in test/test_categories_micro_ras_client.yaml -- gtest's '*' does not match
// across the literal '.', so an unlisted suite never runs under CTest.
// ===========================================================================
class RasClientMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetRasClientFakes();
    ResetRasClientGlobals();
  }
  void TearDown() override {
    ResetRasClientFakes();
    ResetRasClientGlobals();
  }
};

// Scaffolding smoke test: proves the macro seams reach the fakes and that the
// file-scope statics of client.cc are reachable and resettable from this TU.
TEST_F(RasClientMicrotest, Scaffolding_DefaultSeams_AreReachableAndReset) {
  EXPECT_STREQ("localhost", hostName);
  EXPECT_EQ(-1, timeout);
  EXPECT_EQ(-1, sock);

  char buf[8] = {0};
  EXPECT_EQ(4, socketWrite(7, "abcd", 4));
  EXPECT_EQ("abcd", g_writtenData);

  ScriptReadData("hi\n");
  EXPECT_EQ(3, rasRead(7, buf, sizeof(buf)));
  EXPECT_STREQ("hi\n", buf);

  EXPECT_THROW(micro_exit(3), RasClientExit);
}

// ===========================================================================
// parseArgs: the --format/-f and --timeout/-t validation arms.
// ===========================================================================

namespace {

constexpr int kParseArgsNoExit = -999;

struct ParseArgsOutcome {
  int exitStatus = kParseArgsNoExit;
  std::string log;
};

// getopt permutes argv in place, so it needs writable storage; `args` is taken
// by value precisely so the char* below point at strings that outlive the call.
void InvokeParseArgs(std::vector<std::string> args) {
  args.insert(args.begin(), "rccl-ras-client");
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (std::string& a : args) argv.push_back(&a[0]);
  argv.push_back(nullptr);
  parseArgs(static_cast<int>(args.size()), argv.data());
}

// The catch must sit inside CaptureLog: gtest has a single stderr capture slot
// and an exception escaping the body would leave it open for the next test.
ParseArgsOutcome RunParseArgs(const std::vector<std::string>& args) {
  ParseArgsOutcome out;
  out.log = CaptureLog([&]() {
    try {
      InvokeParseArgs(args);
    } catch (const RasClientExit& e) {
      out.exitStatus = e.status;
    }
  });
  return out;
}

}  // namespace

// --- case 'f': accepting arms ----------------------------------------------

TEST_F(RasClientMicrotest, ParseArgsFormat_ShortText_StoresTextAndContinues) {
  const ParseArgsOutcome out = RunParseArgs({"-f", "text"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("text", format);
  EXPECT_EQ("", out.log);
  // A dropped `break` in case 'f' falls through to case 'h', which stores optarg here.
  EXPECT_STREQ("localhost", hostName);
}

TEST_F(RasClientMicrotest, ParseArgsFormat_ShortJson_StoresJsonAndContinues) {
  const ParseArgsOutcome out = RunParseArgs({"-f", "json"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("json", format);
  EXPECT_EQ("", out.log);
  EXPECT_STREQ("localhost", hostName);
}

TEST_F(RasClientMicrotest, ParseArgsFormat_LongEqualsJson_StoresJson) {
  const ParseArgsOutcome out = RunParseArgs({"--format=json"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("json", format);
  EXPECT_EQ("", out.log);
}

TEST_F(RasClientMicrotest, ParseArgsFormat_LongSeparateText_StoresText) {
  const ParseArgsOutcome out = RunParseArgs({"--format", "text"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("text", format);
  EXPECT_EQ("", out.log);
}

TEST_F(RasClientMicrotest, ParseArgsFormat_UppercaseJson_IsAcceptedCaseInsensitively) {
  const ParseArgsOutcome out = RunParseArgs({"-f", "JSON"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("JSON", format);  // stored verbatim; only the comparison folds case
  EXPECT_EQ("", out.log);
}

TEST_F(RasClientMicrotest, ParseArgsFormat_MixedCaseText_IsAcceptedCaseInsensitively) {
  const ParseArgsOutcome out = RunParseArgs({"--format=TeXt"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("TeXt", format);
  EXPECT_EQ("", out.log);
}

// --- case 'f': rejecting arm -----------------------------------------------

TEST_F(RasClientMicrotest, ParseArgsFormat_UnknownValue_ReportsInvalidFormatAndExitsOne) {
  const ParseArgsOutcome out = RunParseArgs({"-f", "xml"});

  EXPECT_EQ(1, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("xml", format);  // the store happens before the validation
  EXPECT_TRUE(LogHas(out.log, "Invalid format: xml (must be text or json)\n")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "Invalid timeout: "));
}

TEST_F(RasClientMicrotest, ParseArgsFormat_TextWithSuffix_ReportsInvalidFormatAndExitsOne) {
  const ParseArgsOutcome out = RunParseArgs({"--format=texts"});

  EXPECT_EQ(1, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("texts", format);
  EXPECT_TRUE(LogHas(out.log, "Invalid format: texts (must be text or json)\n")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "Invalid timeout: "));
}

TEST_F(RasClientMicrotest, ParseArgsFormat_UnknownValue_TerminatesProcessWithStatusOne) {
  ScopedHook exitHook(g_exit, [](int status) { ::_exit(status); });

  EXPECT_EXIT(InvokeParseArgs({"-f", "xml"}), ::testing::ExitedWithCode(1),
              "Invalid format: xml \\(must be text or json\\)");
}

// --- case 't': accepting arms ----------------------------------------------

TEST_F(RasClientMicrotest, ParseArgsTimeout_ShortPositive_StoresValue) {
  const ParseArgsOutcome out = RunParseArgs({"-t", "37"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  EXPECT_EQ(37, timeout);
  EXPECT_EQ("", out.log);
  // A dropped `break` in case 't' falls through to case 'v', which sets this.
  EXPECT_FALSE(verbose);
}

TEST_F(RasClientMicrotest, ParseArgsTimeout_LongEqualsPositive_StoresValue) {
  const ParseArgsOutcome out = RunParseArgs({"--timeout=37"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  EXPECT_EQ(37, timeout);
  EXPECT_EQ("", out.log);
  EXPECT_FALSE(verbose);
}

TEST_F(RasClientMicrotest, ParseArgsTimeout_LongSeparatePositive_StoresValue) {
  const ParseArgsOutcome out = RunParseArgs({"--timeout", "37"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  EXPECT_EQ(37, timeout);
  EXPECT_EQ("", out.log);
}

TEST_F(RasClientMicrotest, ParseArgsTimeout_Zero_IsAcceptedAndDisablesTimeout) {
  const ParseArgsOutcome out = RunParseArgs({"-t", "0"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  EXPECT_EQ(0, timeout);
  EXPECT_EQ("", out.log);
}

TEST_F(RasClientMicrotest, ParseArgsTimeout_LeadingZeroValue_IsParsedAsBaseTen) {
  const ParseArgsOutcome out = RunParseArgs({"-t", "010"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  EXPECT_EQ(10, timeout);  // 8 if the strtol base were 0, 16 if it were 16
  EXPECT_EQ("", out.log);
}

// --- case 't': rejecting arms ----------------------------------------------

TEST_F(RasClientMicrotest, ParseArgsTimeout_Negative_ReportsInvalidAndExitsOne) {
  const ParseArgsOutcome out = RunParseArgs({"-t", "-5"});

  EXPECT_EQ(1, out.exitStatus);
  EXPECT_EQ(-5, timeout);  // -5, not the -1 initializer: the store precedes the check
  EXPECT_TRUE(LogHas(out.log, "Invalid timeout: -5\n")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "Invalid format: "));
}

TEST_F(RasClientMicrotest, ParseArgsTimeout_TrailingGarbage_ReportsInvalidAndExitsOne) {
  const ParseArgsOutcome out = RunParseArgs({"-t", "5x"});

  EXPECT_EQ(1, out.exitStatus);
  EXPECT_EQ(5, timeout);
  EXPECT_TRUE(LogHas(out.log, "Invalid timeout: 5x\n")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "Invalid format: "));
}

TEST_F(RasClientMicrotest, ParseArgsTimeout_NonNumeric_ReportsInvalidAndExitsOne) {
  const ParseArgsOutcome out = RunParseArgs({"--timeout=abc"});

  EXPECT_EQ(1, out.exitStatus);
  EXPECT_EQ(0, timeout);  // strtol consumed nothing and returned 0
  EXPECT_TRUE(LogHas(out.log, "Invalid timeout: abc\n")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "Invalid format: "));
}

TEST_F(RasClientMicrotest, ParseArgsTimeout_HexLiteral_IsRejectedByBaseTenParse) {
  const ParseArgsOutcome out = RunParseArgs({"-t", "0x10"});

  EXPECT_EQ(1, out.exitStatus);
  EXPECT_EQ(0, timeout);  // base 10 stops at 'x'; base 0 or 16 would accept 16
  EXPECT_TRUE(LogHas(out.log, "Invalid timeout: 0x10\n")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "Invalid format: "));
}

// ===========================================================================
// parseArgs: the getopt_long loop plus the -h / -m / -p / -v arms.
// ===========================================================================

namespace {

// getopt permutes argv in place, so the vector must hold writable char* into
// storage that outlives the parse -- events/hostName/port alias into it.
class RasArgv {
 public:
  RasArgv(std::initializer_list<const char*> args) {
    for (const char* a : args) storage_.emplace_back(a);
    for (std::string& s : storage_) argv_.push_back(&s[0]);
    argv_.push_back(nullptr);
  }
  int argc() const { return static_cast<int>(storage_.size()); }
  char** argv() { return argv_.data(); }

 private:
  std::vector<std::string> storage_;
  std::vector<char*> argv_;
};

// Compiled-in defaults of the statics parseArgs writes, so a test can prove an
// arm left an unrelated global alone rather than merely matching its own input.
constexpr const char kDefaultHost[] = "localhost";
constexpr const char kDefaultPort[] = STR(NCCL_RAS_CLIENT_PORT);

}  // namespace

// Loop header, zero-iteration arm: getopt_long returns -1 on the first call.
TEST_F(RasClientMicrotest, ParseArgsLoop_NoOptions_LeavesEveryGlobalAtItsDefault) {
  RasArgv args{"rasclient"};
  parseArgs(args.argc(), args.argv());

  EXPECT_STREQ(kDefaultHost, hostName);
  EXPECT_STREQ(kDefaultPort, port);
  EXPECT_EQ(-1, timeout);
  EXPECT_FALSE(verbose);
  EXPECT_FALSE(monitorMode);
  EXPECT_EQ(nullptr, format);
  EXPECT_EQ(nullptr, events);

  // Positive anchor: the same fixture state does parse an option, so the
  // assertions above are not passing because parseArgs is inert.
  ResetRasClientGlobals();
  RasArgv anchor{"rasclient", "-v"};
  parseArgs(anchor.argc(), anchor.argv());
  EXPECT_TRUE(verbose);
}

// Loop header, multi-iteration arm: four options in one argv, all four applied.
TEST_F(RasClientMicrotest, ParseArgsLoop_FourOptionsInOneArgv_AppliesAllOfThem) {
  RasArgv args{"rasclient", "-h", "rasnode7", "-p", "31337", "-mtrace", "-v"};
  parseArgs(args.argc(), args.argv());

  EXPECT_STREQ("rasnode7", hostName);
  EXPECT_STREQ("31337", port);
  ASSERT_NE(nullptr, events);
  EXPECT_STREQ("trace", events);
  EXPECT_TRUE(monitorMode);
  EXPECT_TRUE(verbose);
  EXPECT_EQ(-1, timeout);
  EXPECT_EQ(nullptr, format);
}

// case 'h', short spelling. A distinctive host proves the store, not the initializer.
TEST_F(RasClientMicrotest, ParseArgsHost_ShortForm_StoresHostAndLeavesPortAlone) {
  RasArgv args{"rasclient", "-h", "rasnode7"};
  parseArgs(args.argc(), args.argv());

  EXPECT_STREQ("rasnode7", hostName);
  EXPECT_STREQ(kDefaultPort, port);
  EXPECT_FALSE(monitorMode);
  EXPECT_EQ(nullptr, events);
  EXPECT_FALSE(verbose);
  EXPECT_EQ(-1, timeout);
}

// case 'h', long spelling with an attached '=' argument.
TEST_F(RasClientMicrotest, ParseArgsHost_LongFormEqualsArg_StoresHostAndLeavesPortAlone) {
  RasArgv args{"rasclient", "--host=10.0.0.42"};
  parseArgs(args.argc(), args.argv());

  EXPECT_STREQ("10.0.0.42", hostName);
  EXPECT_STREQ(kDefaultPort, port);
  EXPECT_FALSE(monitorMode);
  EXPECT_FALSE(verbose);
}

// case 'p', short spelling.
TEST_F(RasClientMicrotest, ParseArgsPort_ShortForm_StoresPortAndLeavesHostAlone) {
  RasArgv args{"rasclient", "-p", "31337"};
  parseArgs(args.argc(), args.argv());

  EXPECT_STREQ("31337", port);
  EXPECT_STREQ(kDefaultHost, hostName);
  // A dropped break here would fall into case 't' and strtol("31337") into timeout.
  EXPECT_EQ(-1, timeout);
  EXPECT_FALSE(monitorMode);
  EXPECT_FALSE(verbose);
}

// case 'p', long spelling with an attached '=' argument.
TEST_F(RasClientMicrotest, ParseArgsPort_LongFormEqualsArg_StoresPortAndLeavesHostAlone) {
  RasArgv args{"rasclient", "--port=31337"};
  parseArgs(args.argc(), args.argv());

  EXPECT_STREQ("31337", port);
  EXPECT_STREQ(kDefaultHost, hostName);
  EXPECT_EQ(-1, timeout);
  EXPECT_FALSE(verbose);
}

// case 'h' then case 'p' in one argv: a swapped pair of stores cannot hide behind
// either single-option test, because both destinations here are non-default.
TEST_F(RasClientMicrotest, ParseArgsHostAndPort_BothGiven_LandInTheirOwnGlobals) {
  RasArgv args{"rasclient", "-h", "rasnode7", "-p", "31337"};
  parseArgs(args.argc(), args.argv());

  EXPECT_STREQ("rasnode7", hostName);
  EXPECT_STREQ("31337", port);
}

// case 'm', bare short form: getopt leaves optarg NULL, so the if(optarg) guard
// must skip the events store while monitorMode is still set.
TEST_F(RasClientMicrotest, ParseArgsMonitor_BareShortForm_SetsModeAndLeavesEventsNull) {
  RasArgv args{"rasclient", "-m"};
  parseArgs(args.argc(), args.argv());

  EXPECT_TRUE(monitorMode);
  EXPECT_EQ(nullptr, events);
  EXPECT_STREQ(kDefaultHost, hostName);
  EXPECT_STREQ(kDefaultPort, port);
}

// case 'm', bare long form: same NULL-optarg path via --monitor.
TEST_F(RasClientMicrotest, ParseArgsMonitor_BareLongForm_SetsModeAndLeavesEventsNull) {
  RasArgv args{"rasclient", "--monitor"};
  parseArgs(args.argc(), args.argv());

  EXPECT_TRUE(monitorMode);
  EXPECT_EQ(nullptr, events);
  EXPECT_FALSE(verbose);
}

// With "m::" a separated argument is NOT attached, so optarg is NULL and
// "lifecycle" stays a non-option operand. This asymmetry is why the guard exists.
TEST_F(RasClientMicrotest, ParseArgsMonitor_ShortFormSeparateArg_DoesNotAttachAndLeavesEventsNull) {
  RasArgv args{"rasclient", "-m", "lifecycle"};
  parseArgs(args.argc(), args.argv());

  EXPECT_TRUE(monitorMode);
  EXPECT_EQ(nullptr, events);
}

// case 'm', attached short form "-mlifecycle": optarg is non-NULL, events stored.
TEST_F(RasClientMicrotest, ParseArgsMonitor_AttachedShortFormArg_StoresEvents) {
  RasArgv args{"rasclient", "-mlifecycle"};
  parseArgs(args.argc(), args.argv());

  EXPECT_TRUE(monitorMode);
  ASSERT_NE(nullptr, events);
  EXPECT_STREQ("lifecycle", events);
  EXPECT_STREQ(kDefaultHost, hostName);
  EXPECT_STREQ(kDefaultPort, port);
}

// case 'm', long form with '=': a multi-group value reaches events verbatim.
TEST_F(RasClientMicrotest, ParseArgsMonitor_LongFormEqualsGroups_StoresEventsVerbatim) {
  RasArgv args{"rasclient", "--monitor=lifecycle,trace"};
  parseArgs(args.argc(), args.argv());

  EXPECT_TRUE(monitorMode);
  ASSERT_NE(nullptr, events);
  EXPECT_STREQ("lifecycle,trace", events);
  EXPECT_FALSE(verbose);
}

// Two -m in one argv. Dropping the if(optarg) guard makes the second, bare -m
// overwrite events with NULL; with the guard the earlier value survives.
TEST_F(RasClientMicrotest, ParseArgsMonitor_AttachedThenBare_KeepsTheEarlierEvents) {
  RasArgv args{"rasclient", "-mlifecycle", "-m"};
  parseArgs(args.argc(), args.argv());

  EXPECT_TRUE(monitorMode);
  ASSERT_NE(nullptr, events);
  EXPECT_STREQ("lifecycle", events);
}

// case 'v', short spelling.
TEST_F(RasClientMicrotest, ParseArgsVerbose_ShortForm_SetsVerboseOnly) {
  RasArgv args{"rasclient", "-v"};
  parseArgs(args.argc(), args.argv());

  EXPECT_TRUE(verbose);
  EXPECT_FALSE(monitorMode);
  EXPECT_STREQ(kDefaultHost, hostName);
  EXPECT_STREQ(kDefaultPort, port);
  EXPECT_EQ(nullptr, events);
  EXPECT_EQ(-1, timeout);
}

// case 'v', long spelling before another option: "verbose" must stay no_argument,
// or --verbose swallows the -p and the port store never runs.
TEST_F(RasClientMicrotest, ParseArgsVerbose_LongFormBeforeAnotherOption_DoesNotConsumeIt) {
  RasArgv args{"rasclient", "--verbose", "-p", "31337"};
  parseArgs(args.argc(), args.argv());

  EXPECT_TRUE(verbose);
  EXPECT_STREQ("31337", port);
  EXPECT_STREQ(kDefaultHost, hostName);
}
