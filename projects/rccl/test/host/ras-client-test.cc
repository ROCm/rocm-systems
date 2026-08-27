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


// ===========================================================================
// rasRead (src/ras/client.cc:139-155)
//
// Every test below drives the g_read seam with a recording hook so the unit's
// own arithmetic -- the destination offset (bufChar + done) and the requested
// size (count - 1 - done) -- is observable, and pre-fills the caller's buffer
// with a non-zero sentinel so a missing or misplaced '\0' store is visible.
// ===========================================================================

namespace {

// Not 0x00 (which cannot see a missing NUL store) and not any payload byte.
constexpr char kSentinel = static_cast<char>(0xAA);

// One (destination offset, requested size) pair as rasRead computed it.
struct ReadRequest {
  long offset;
  size_t count;
};

// One scripted read outcome: ret < 0 fails with `err`, ret == 0 is EOF.
struct ReadStep {
  ssize_t ret;
  int err;
  std::string data;
};

// Records every request and serves `steps` front-to-back; past the end, and for
// a zero-length request, it returns 0 exactly as a real read(2) would. It never
// writes more than the caller asked for, so an overflow seen in a test is the
// unit's, not the fake's.
class RecordingReader {
 public:
  RecordingReader(const char* base, std::vector<ReadStep> steps) : base_(base), steps_(std::move(steps)) {}

  ssize_t operator()(int, void* buf, size_t count) {
    requests.push_back(ReadRequest{static_cast<const char*>(buf) - base_, count});
    if (count == 0 || pos_ >= steps_.size()) return 0;
    const ReadStep& step = steps_[pos_++];
    if (step.ret < 0) {
      errno = step.err;
      return step.ret;
    }
    if (step.ret == 0) return 0;
    const size_t n = step.data.size() < count ? step.data.size() : count;
    memcpy(buf, step.data.data(), n);
    return static_cast<ssize_t>(n);
  }

  std::vector<ReadRequest> requests;

 private:
  const char* base_;
  std::vector<ReadStep> steps_;
  size_t pos_ = 0;
};

void ExpectRequests(const std::vector<ReadRequest>& got, const std::vector<ReadRequest>& want) {
  ASSERT_EQ(want.size(), got.size());
  for (size_t i = 0; i < want.size(); ++i) {
    EXPECT_EQ(want[i].offset, got[i].offset) << "request " << i << " destination offset";
    EXPECT_EQ(want[i].count, got[i].count) << "request " << i << " requested size";
  }
}

}  // namespace

// Arms: ret > 0 three times, `done += ret`, the `bufChar[done-1] != '\n'` retry
// twice then exit, and the shrinking `count - 1 - done` request.
TEST_F(RasClientMicrotest, RasRead_UntilNewline_NewlineInLastChunk_ReassemblesAndTerminates) {
  char backing[40];
  memset(backing, kSentinel, sizeof(backing));
  char* buf = backing + 4;
  RecordingReader reader(buf, {{5, 0, "abcde"}, {4, 0, "fghi"}, {2, 0, "j\n"}});
  ScopedHook readHook(g_read, [&](int fd, void* b, size_t n) { return reader(fd, b, n); });

  EXPECT_EQ(11, rasRead(9, buf, 32));

  EXPECT_EQ(3, readHook.calls);
  ExpectRequests(reader.requests, {{0, 31}, {5, 26}, {9, 22}});
  EXPECT_EQ("abcdefghij\n", std::string(buf, 11));
  EXPECT_EQ('\0', buf[11]);
  EXPECT_EQ(kSentinel, buf[12]);
  EXPECT_EQ(kSentinel, backing[3]);
}

// Arm: ret == -1 with errno != EINTR returns before bufChar[done] = '\0', so the
// caller's buffer keeps its prior contents. Benign: every call site in client.cc
// checks `bytes < 0` before touching msgBuf.
TEST_F(RasClientMicrotest, RasRead_NonEintrError_ReturnsMinusOneAndLeavesBufferUntouched) {
  char buf[16];
  memset(buf, kSentinel, sizeof(buf));
  RecordingReader reader(buf, {{-1, EIO, ""}});
  ScopedHook readHook(g_read, [&](int fd, void* b, size_t n) { return reader(fd, b, n); });

  EXPECT_EQ(-1, rasRead(9, buf, sizeof(buf)));

  EXPECT_EQ(EIO, errno);
  EXPECT_EQ(1, readHook.calls);
  ExpectRequests(reader.requests, {{0, 15}});
  for (size_t i = 0; i < sizeof(buf); ++i) {
    EXPECT_EQ(kSentinel, buf[i]) << "byte " << i << " was written on the error path";
  }
}

// Arms: EINTR retries at the same offset, and the `done == 0` guard keeps the
// retry from reading bufChar[-1]. The byte below the buffer is '\n' on purpose:
// without the guard the loop would exit immediately and return 0.
TEST_F(RasClientMicrotest, RasRead_EintrOnFirstRead_RetriesAtSameOffsetAndSucceeds) {
  char backing[40];
  memset(backing, kSentinel, sizeof(backing));
  backing[7] = '\n';
  char* buf = backing + 8;
  RecordingReader reader(buf, {{-1, EINTR, ""}, {4, 0, "xyz\n"}});
  ScopedHook readHook(g_read, [&](int fd, void* b, size_t n) { return reader(fd, b, n); });

  EXPECT_EQ(4, rasRead(9, buf, 16));

  EXPECT_EQ(2, readHook.calls);
  ExpectRequests(reader.requests, {{0, 15}, {0, 15}});
  EXPECT_EQ("xyz\n", std::string(buf, 4));
  EXPECT_EQ('\0', buf[4]);
  EXPECT_EQ(kSentinel, buf[5]);
  EXPECT_EQ('\n', backing[7]);
}

// Arm: ret == 0 breaks out of the loop and the partial message is still
// terminated at exactly `done`.
TEST_F(RasClientMicrotest, RasRead_EofBeforeNewline_BreaksAndTerminatesAtBytesRead) {
  char backing[40];
  memset(backing, kSentinel, sizeof(backing));
  char* buf = backing + 4;
  RecordingReader reader(buf, {{4, 0, "part"}, {0, 0, ""}});
  ScopedHook readHook(g_read, [&](int fd, void* b, size_t n) { return reader(fd, b, n); });

  EXPECT_EQ(4, rasRead(9, buf, 32));

  EXPECT_EQ(2, readHook.calls);
  ExpectRequests(reader.requests, {{0, 31}, {4, 27}});
  EXPECT_EQ("part", std::string(buf, 4));
  EXPECT_EQ('\0', buf[4]);
  EXPECT_EQ(kSentinel, buf[5]);
}

// Arm: untilNewline == false (how getNCCLStatus calls this) does exactly one
// read and returns whatever arrived, newline or not.
TEST_F(RasClientMicrotest, RasRead_NotUntilNewline_NoNewlineInData_DoesExactlyOneRead) {
  char backing[40];
  memset(backing, kSentinel, sizeof(backing));
  char* buf = backing + 4;
  RecordingReader reader(buf, {{4, 0, "wxyz"}, {5, 0, "never"}});
  ScopedHook readHook(g_read, [&](int fd, void* b, size_t n) { return reader(fd, b, n); });

  EXPECT_EQ(4, rasRead(9, buf, 16, /*untilNewline=*/false));

  EXPECT_EQ(1, readHook.calls);
  ExpectRequests(reader.requests, {{0, 15}});
  EXPECT_EQ("wxyz", std::string(buf, 4));
  EXPECT_EQ('\0', buf[4]);
  EXPECT_EQ(kSentinel, buf[5]);
}

// Arm: `continue` on EINTR with untilNewline == false falls straight out of the
// do-while, so a mere signal is reported to the caller as a clean zero-byte
// read -- which getNCCLStatus treats as EOF and stops the stream on.
TEST_F(RasClientMicrotest, RasRead_NotUntilNewlineEintr_ReturnsZeroAndTerminatesEmpty) {
  char backing[40];
  memset(backing, kSentinel, sizeof(backing));
  char* buf = backing + 4;
  RecordingReader reader(buf, {{-1, EINTR, ""}, {4, 0, "late"}});
  ScopedHook readHook(g_read, [&](int fd, void* b, size_t n) { return reader(fd, b, n); });

  EXPECT_EQ(0, rasRead(9, buf, 16, /*untilNewline=*/false));

  EXPECT_EQ(1, readHook.calls);
  ExpectRequests(reader.requests, {{0, 15}});
  EXPECT_EQ('\0', buf[0]);
  EXPECT_EQ(kSentinel, buf[1]);
}

// Arm: the `count - 1 - done` reservation. A full buffer holds count-1 payload
// bytes; the next request is zero-length, which reads as EOF, so the '\0' lands
// at count-1 and never one past the caller's buffer.
TEST_F(RasClientMicrotest, RasRead_PayloadFillsBuffer_ReservesNulAndNeverWritesPastCount) {
  char backing[40];
  memset(backing, kSentinel, sizeof(backing));
  char* buf = backing + 4;
  RecordingReader reader(buf, {{10, 0, "ABCDEFGHIJ"}});
  ScopedHook readHook(g_read, [&](int fd, void* b, size_t n) { return reader(fd, b, n); });

  EXPECT_EQ(7, rasRead(9, buf, 8));

  EXPECT_EQ(2, readHook.calls);
  ExpectRequests(reader.requests, {{0, 7}, {7, 0}});
  EXPECT_EQ("ABCDEFG", std::string(buf, 7));
  EXPECT_EQ('\0', buf[7]);
  EXPECT_EQ(kSentinel, buf[8]);
}

// Arm: count == 0 underflows `count - 1 - done` to SIZE_MAX. Unreachable from
// client.cc (every caller passes sizeof of a real array); pinned so a future
// caller that can pass 0 shows up as a changed expectation here.
TEST_F(RasClientMicrotest, RasRead_ZeroCount_UnderflowsRequestToSizeMax) {
  char backing[40];
  memset(backing, kSentinel, sizeof(backing));
  char* buf = backing + 4;
  RecordingReader reader(buf, {});
  ScopedHook readHook(g_read, [&](int fd, void* b, size_t n) { return reader(fd, b, n); });

  EXPECT_EQ(0, rasRead(9, buf, 0));

  EXPECT_EQ(1, readHook.calls);
  ExpectRequests(reader.requests, {{0, static_cast<size_t>(-1)}});
  EXPECT_EQ('\0', buf[0]);
  EXPECT_EQ(kSentinel, buf[1]);
}


// ===========================================================================
// socketWrite (src/ras/client.cc:121-134)
//
// do/while over write(2): retries on EINTR, gives up on any other errno, and
// advances the buffer offset by the number of bytes each write accepted.
// ===========================================================================

namespace {

// One (offset, length) pair as socketWrite computed it. The offset is relative
// to the caller's buffer, so recording it makes the `+ done` arithmetic visible.
struct WriteCall {
  long long offset;
  size_t count;
};

// One scripted write outcome; ret < 0 makes the write fail with err in errno.
struct WriteStep {
  ssize_t ret;
  int err;
};

// Records what socketWrite handed each write() and replays a script of results.
// Once the script is exhausted -- or the hard call cap is hit -- every further
// call fails with EIO. That cap is load-bearing: several mutants of this loop
// (dropped `+ done`, `done = ret`, `<=` in the guard) never terminate, and a
// hook that kept returning success would hang the binary instead of failing.
class WriteRecorder {
 public:
  WriteRecorder(const char* base, std::vector<WriteStep> script) : base_(base), script_(std::move(script)) {}

  ssize_t operator()(int, const void* buf, size_t count) {
    const char* p = static_cast<const char*>(buf);
    calls.push_back(WriteCall{static_cast<long long>(p - base_), count});
    if (calls.size() > kHardCap || step_ >= script_.size()) {
      errno = EIO;
      return -1;
    }
    const WriteStep step = script_[step_++];
    if (step.ret < 0) {
      errno = step.err;
      return step.ret;
    }
    const size_t taken = static_cast<size_t>(step.ret) < count ? static_cast<size_t>(step.ret) : count;
    data.append(p, taken);
    return step.ret;
  }

  std::vector<WriteCall> calls;
  std::string data;  // the payload as reassembled from the accepted chunks

 private:
  static const size_t kHardCap = 16;
  const char* base_;
  std::vector<WriteStep> script_;
  size_t step_ = 0;
};

// "0/19,5/14" -- one offset/count pair per write, in call order.
std::string FormatWriteCalls(const std::vector<WriteCall>& calls) {
  std::string out;
  for (const WriteCall& call : calls) {
    if (!out.empty()) out += ",";
    out += std::to_string(call.offset) + "/" + std::to_string(call.count);
  }
  return out;
}

// 19 distinct bytes: not a multiple of any chunk size used below, and no byte
// repeats, so a dropped offset or a restarted retry changes the reassembly.
const char kPayload[] = "ABCDEFGHIJKLMNOPQRS";
const size_t kPayloadLen = sizeof(kPayload) - 1;

}  // namespace

// Arm: write accepts everything on the first call; the loop guard is false at once.
TEST_F(RasClientMicrotest, SocketWrite_FullWriteFirstTry_WritesWholeBufferOnce) {
  WriteRecorder rec(kPayload, {{19, 0}});
  ScopedHook writeHook(g_write, [&rec](int fd, const void* buf, size_t count) { return rec(fd, buf, count); });

  EXPECT_EQ(static_cast<ssize_t>(19), socketWrite(9, kPayload, kPayloadLen));
  EXPECT_EQ(1, writeHook.calls);
  EXPECT_EQ("0/19", FormatWriteCalls(rec.calls));
  EXPECT_EQ("ABCDEFGHIJKLMNOPQRS", rec.data);
}

// Arm: several short writes; each call must start at buf + done and ask for count - done.
TEST_F(RasClientMicrotest, SocketWrite_ShortWrites_AdvancesOffsetAndReturnsTotal) {
  WriteRecorder rec(kPayload, {{5, 0}, {5, 0}, {5, 0}, {4, 0}});
  ScopedHook writeHook(g_write, [&rec](int fd, const void* buf, size_t count) { return rec(fd, buf, count); });

  EXPECT_EQ(static_cast<ssize_t>(19), socketWrite(9, kPayload, kPayloadLen));
  EXPECT_EQ(4, writeHook.calls);
  EXPECT_EQ("0/19,5/14,10/9,15/4", FormatWriteCalls(rec.calls));
  EXPECT_EQ("ABCDEFGHIJKLMNOPQRS", rec.data);
}

// Arm: ret == -1 with errno == EINTR retries; `continue` re-tests done < count,
// which is still true here, so the retry must resume at the same offset.
TEST_F(RasClientMicrotest, SocketWrite_EintrMidTransfer_RetriesFromSameOffsetAndReturnsTotal) {
  WriteRecorder rec(kPayload, {{7, 0}, {-1, EINTR}, {12, 0}});
  ScopedHook writeHook(g_write, [&rec](int fd, const void* buf, size_t count) { return rec(fd, buf, count); });

  EXPECT_EQ(static_cast<ssize_t>(19), socketWrite(9, kPayload, kPayloadLen));
  EXPECT_EQ(3, writeHook.calls);
  EXPECT_EQ("0/19,7/12,7/12", FormatWriteCalls(rec.calls));
  EXPECT_EQ("ABCDEFGHIJKLMNOPQRS", rec.data);
}

// Arm: ret == -1 with EINTR on the very first call, before any byte is written.
TEST_F(RasClientMicrotest, SocketWrite_EintrBeforeAnyProgress_RetriesFromOffsetZero) {
  WriteRecorder rec(kPayload, {{-1, EINTR}, {19, 0}});
  ScopedHook writeHook(g_write, [&rec](int fd, const void* buf, size_t count) { return rec(fd, buf, count); });

  EXPECT_EQ(static_cast<ssize_t>(19), socketWrite(9, kPayload, kPayloadLen));
  EXPECT_EQ(2, writeHook.calls);
  EXPECT_EQ("0/19,0/19", FormatWriteCalls(rec.calls));
  EXPECT_EQ("ABCDEFGHIJKLMNOPQRS", rec.data);
}

// Arm: ret == -1 with any other errno returns -1 and discards the partial progress.
TEST_F(RasClientMicrotest, SocketWrite_NonEintrError_ReturnsMinusOneAndAbandonsPartialWrite) {
  WriteRecorder rec(kPayload, {{6, 0}, {-1, EIO}});
  ScopedHook writeHook(g_write, [&rec](int fd, const void* buf, size_t count) { return rec(fd, buf, count); });

  errno = 0;
  EXPECT_EQ(static_cast<ssize_t>(-1), socketWrite(9, kPayload, kPayloadLen));
  EXPECT_EQ(EIO, errno);
  EXPECT_EQ(2, writeHook.calls);
  EXPECT_EQ("0/19,6/13", FormatWriteCalls(rec.calls));
  EXPECT_EQ("ABCDEF", rec.data);
}

// Arm: count == 0. The do/while body runs unconditionally, so production issues
// one zero-length write before the guard stops the loop.
TEST_F(RasClientMicrotest, SocketWrite_ZeroCount_StillIssuesOneZeroLengthWrite) {
  WriteRecorder rec(kPayload, {{0, 0}});
  ScopedHook writeHook(g_write, [&rec](int fd, const void* buf, size_t count) { return rec(fd, buf, count); });

  EXPECT_EQ(static_cast<ssize_t>(0), socketWrite(9, kPayload, 0));
  EXPECT_EQ(1, writeHook.calls);
  EXPECT_EQ("0/0", FormatWriteCalls(rec.calls));
  EXPECT_EQ("", rec.data);
}

// Arm: EINTR on the mandatory first iteration of a count == 0 write. `continue`
// re-tests done < count, which is 0 < 0 -- so the retry never happens and the
// interrupted zero-length write is reported as a success.
TEST_F(RasClientMicrotest, SocketWrite_ZeroCountEintr_DoesNotRetryAndReturnsZero) {
  WriteRecorder rec(kPayload, {{-1, EINTR}, {0, 0}});
  ScopedHook writeHook(g_write, [&rec](int fd, const void* buf, size_t count) { return rec(fd, buf, count); });

  EXPECT_EQ(static_cast<ssize_t>(0), socketWrite(9, kPayload, 0));
  EXPECT_EQ(1, writeHook.calls);
  EXPECT_EQ("0/0", FormatWriteCalls(rec.calls));
}


// ===========================================================================
// printUsage, plus parseArgs' three exiting arms: case 'e' (--help, exit 0),
// case 'r' (--version, exit 0) and default: (bad option, exit 1).
// ===========================================================================

namespace {

// printUsage echoes argv[0], and getopt prefixes its own diagnostics with it,
// so a distinctive value separates "the unit printed it" from "it was there".
constexpr const char kUsageProg[] = "ras-client-argv0-probe";

// Runs parseArgs over a caller-supplied argv (argv[0] included) and records both
// the status the default exit seam threw and everything written to stderr.
ParseArgsOutcome RunParseArgv(std::initializer_list<const char*> args) {
  ParseArgsOutcome out;
  RasArgv argv(args);
  out.log = CaptureLog([&]() {
    try {
      parseArgs(argv.argc(), argv.argv());
    } catch (const RasClientExit& e) {
      out.exitStatus = e.status;
    }
  });
  return out;
}

// The text between `needle` and the following '\n'; "" when `needle` is absent.
std::string LineAfter(const std::string& log, const char* needle) {
  const size_t at = log.find(needle);
  if (at == std::string::npos) return "";
  const size_t start = at + std::strlen(needle);
  const size_t end = log.find('\n', start);
  return log.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

size_t CountChar(const std::string& s, char c) {
  size_t n = 0;
  for (char ch : s) {
    if (ch == c) ++n;
  }
  return n;
}

// Every global parseArgs can write, still at its compiled-in default. The exiting
// arms must reach exit without touching any of them.
void ExpectAllGlobalsAtDefault() {
  EXPECT_STREQ(kDefaultHost, hostName);
  EXPECT_STREQ(kDefaultPort, port);
  EXPECT_EQ(-1, timeout);
  EXPECT_FALSE(verbose);
  EXPECT_FALSE(monitorMode);
  EXPECT_EQ(nullptr, format);
  EXPECT_EQ(nullptr, events);
}

}  // namespace

// --- printUsage, called directly -------------------------------------------

// Whole-line needles: "  -f, --format=FMT" and "  -p, --port=PORT" share a prefix,
// so anything shorter than a full line cannot see one line replacing another.
TEST_F(RasClientMicrotest, PrintUsage_CalledDirectly_EmitsEveryLineOfTheUsageText) {
  const std::string log = CaptureLog([]() { printUsage(kUsageProg); });

  EXPECT_TRUE(LogHas(log, "Usage: ras-client-argv0-probe [OPTION]...\n")) << log;
  EXPECT_TRUE(LogHas(log, "Query the state of a running NCCL job.\n")) << log;
  EXPECT_TRUE(LogHas(log, "\nOptions:\n")) << log;
  EXPECT_TRUE(LogHas(log, "  -f, --format=FMT    Output format: text or json (text by default)\n")) << log;
  EXPECT_TRUE(LogHas(log, "  -h, --host=HOST     Host name or IP address of the RAS client socket of the\n")) << log;
  EXPECT_TRUE(LogHas(log, "                      NCCL job to connect to (localhost by default)\n")) << log;
  EXPECT_TRUE(LogHas(log, "  -m, --monitor[=GROUPS] Monitor mode: continuously watch for peer changes.\n")) << log;
  EXPECT_TRUE(LogHas(log, "                      Optional GROUPS: lifecycle, trace, all, or\n")) << log;
  EXPECT_TRUE(LogHas(log, "                      combinations like lifecycle,trace (lifecycle by default)\n")) << log;
  EXPECT_TRUE(LogHas(log, "  -p, --port=PORT     TCP port of the RAS client socket of the NCCL job\n")) << log;
  EXPECT_TRUE(LogHas(log, "  -t, --timeout=SECS  Maximum time for the local NCCL process to wait for\n")) << log;
  EXPECT_TRUE(LogHas(log, "                      responses from other NCCL processes\n")) << log;
  EXPECT_TRUE(LogHas(log, "  -v, --verbose       Increase the verbosity level of the RAS output\n")) << log;
  EXPECT_TRUE(LogHas(log, "      --help          Print this help and exit\n")) << log;
  EXPECT_TRUE(LogHas(log, "      --version       Print the version number and exit\n")) << log;
}

// The two macro interpolations, asserted with their surrounding text so a dropped
// substitution (or a wrong macro) dies rather than merely printing a bare paren.
TEST_F(RasClientMicrotest, PrintUsage_MacroDefaults_AreInterpolatedWithSurroundingText) {
  const std::string log = CaptureLog([]() { printUsage(kUsageProg); });

  char portLine[160];
  char timeoutLine[160];
  std::snprintf(portLine, sizeof(portLine), "                      (%d by default)\n", NCCL_RAS_CLIENT_PORT);
  std::snprintf(timeoutLine, sizeof(timeoutLine), "                      (%d secs by default; 0 disables the timeout)\n",
                RAS_COLLECTIVE_LEG_TIMEOUT_SEC);

  EXPECT_TRUE(LogHas(log, portLine)) << portLine << "\n--- log ---\n" << log;
  EXPECT_TRUE(LogHas(log, timeoutLine)) << timeoutLine << "\n--- log ---\n" << log;
}

// Ordering: the whole-line checks above are position-blind, so a reordered block
// would keep them all green.
TEST_F(RasClientMicrotest, PrintUsage_OptionBlock_IsEmittedInDeclarationOrder) {
  const std::string log = CaptureLog([]() { printUsage(kUsageProg); });

  const size_t usage = log.find("Usage: ras-client-argv0-probe [OPTION]...\n");
  const size_t fmt = log.find("  -f, --format=FMT ");
  const size_t host = log.find("  -h, --host=HOST ");
  const size_t mon = log.find("  -m, --monitor[=GROUPS] ");
  const size_t prt = log.find("  -p, --port=PORT ");
  const size_t tmo = log.find("  -t, --timeout=SECS ");
  const size_t verb = log.find("  -v, --verbose ");
  const size_t help = log.find("      --help ");
  const size_t vers = log.find("      --version ");

  ASSERT_NE(std::string::npos, vers) << log;
  EXPECT_LT(usage, fmt);
  EXPECT_LT(fmt, host);
  EXPECT_LT(host, mon);
  EXPECT_LT(mon, prt);
  EXPECT_LT(prt, tmo);
  EXPECT_LT(tmo, verb);
  EXPECT_LT(verb, help);
  EXPECT_LT(help, vers);
}

// printUsage takes argv0 as a parameter, so a different value must show up.
TEST_F(RasClientMicrotest, PrintUsage_DifferentArgv0_IsEchoedVerbatim) {
  const std::string log = CaptureLog([]() { printUsage("/opt/rocm/bin/rccl-ras-other"); });

  EXPECT_TRUE(LogHas(log, "Usage: /opt/rocm/bin/rccl-ras-other [OPTION]...\n")) << log;
  EXPECT_FALSE(LogHas(log, kUsageProg)) << log;
}

// printUsage writes nowhere but stderr and changes no parse state; without this
// the two exiting arms cannot be told apart from the function they call.
TEST_F(RasClientMicrotest, PrintUsage_CalledDirectly_TouchesNoParseStateAndDoesNotExit) {
  const std::string log = CaptureLog([]() { printUsage(kUsageProg); });

  EXPECT_TRUE(LogHas(log, "Usage: ras-client-argv0-probe [OPTION]...\n")) << log;
  EXPECT_FALSE(LogHas(log, "RCCL RAS client version ")) << log;
  ExpectAllGlobalsAtDefault();
}

// --- case 'e': --help ------------------------------------------------------

// 'e' is not in the short optstring "f:h:m::p:t:v", so only the long option
// reaches it; getopt_long returns 'e' from the longOpts val field.
TEST_F(RasClientMicrotest, ParseArgsHelp_LongForm_PrintsUsageWithArgv0AndExitsZero) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "--help"});

  EXPECT_EQ(0, out.exitStatus);
  EXPECT_TRUE(LogHas(out.log, "Usage: ras-client-argv0-probe [OPTION]...\n")) << out.log;
  EXPECT_TRUE(LogHas(out.log, "      --version       Print the version number and exit\n")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "RCCL RAS client version ")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "unrecognized option")) << out.log;
  ExpectAllGlobalsAtDefault();
}

// --help must exit before any option after it is looked at.
TEST_F(RasClientMicrotest, ParseArgsHelp_FollowedByOtherOptions_ExitsBeforeApplyingThem) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "--help", "-v", "-p", "31337"});

  EXPECT_EQ(0, out.exitStatus);
  EXPECT_TRUE(LogHas(out.log, "Usage: ras-client-argv0-probe [OPTION]...\n")) << out.log;
  EXPECT_FALSE(verbose);
  EXPECT_STREQ(kDefaultPort, port);
}

// --help is no_argument: an earlier option's value must not be swallowed by it.
TEST_F(RasClientMicrotest, ParseArgsHelp_AfterAnAppliedOption_StillExitsZeroWithUsage) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "-p", "31337", "--help"});

  EXPECT_EQ(0, out.exitStatus);
  EXPECT_STREQ("31337", port);
  EXPECT_TRUE(LogHas(out.log, "Usage: ras-client-argv0-probe [OPTION]...\n")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "requires an argument")) << out.log;
}

// A unique long-option abbreviation still resolves to 'e'; "--hel" is not a
// prefix of "--host", so it is unambiguous.
TEST_F(RasClientMicrotest, ParseArgsHelp_UniqueAbbreviation_ReachesTheHelpArmNotDefault) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "--hel"});

  EXPECT_EQ(0, out.exitStatus);
  EXPECT_TRUE(LogHas(out.log, "Usage: ras-client-argv0-probe [OPTION]...\n")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "is ambiguous")) << out.log;
}

// --- case 'r': --version ---------------------------------------------------

// The components are checked positionally against the macros rather than against
// a re-spelled STR() concatenation, so swapping NCCL_MINOR and NCCL_PATCH in the
// production string lands 7 where 30 is expected and this test fails.
TEST_F(RasClientMicrotest, ParseArgsVersion_LongForm_PrintsComponentsInOrderAndExitsZero) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "--version"});

  EXPECT_EQ(0, out.exitStatus);
  ASSERT_TRUE(LogHas(out.log, "RCCL RAS client version ")) << out.log;

  const std::string version = LineAfter(out.log, "RCCL RAS client version ");
  int major = -1;
  int minor = -1;
  int patch = -1;
  ASSERT_EQ(3, std::sscanf(version.c_str(), "%d.%d.%d", &major, &minor, &patch)) << version;
  EXPECT_EQ(NCCL_MAJOR, major) << version;
  EXPECT_EQ(NCCL_MINOR, minor) << version;
  EXPECT_EQ(NCCL_PATCH, patch) << version;
  EXPECT_EQ(2u, CountChar(version, '.')) << version;
  ExpectAllGlobalsAtDefault();
}

// The version arm must not print the usage text; that is what separates it from
// case 'e' and from default:.
TEST_F(RasClientMicrotest, ParseArgsVersion_LongForm_DoesNotPrintTheUsageText) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "--version"});

  EXPECT_EQ(0, out.exitStatus);
  EXPECT_TRUE(LogHas(out.log, "RCCL RAS client version ")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "Usage: ")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "      --version       Print the version number and exit\n")) << out.log;
  EXPECT_FALSE(LogHas(out.log, kUsageProg)) << out.log;
}

// --version exits before later options are applied, and 'r' is unreachable via
// any short option, so "-r" is an unknown option instead.
TEST_F(RasClientMicrotest, ParseArgsVersion_FollowedByOtherOptions_ExitsBeforeApplyingThem) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "--version", "-v"});

  EXPECT_EQ(0, out.exitStatus);
  EXPECT_TRUE(LogHas(out.log, "RCCL RAS client version ")) << out.log;
  EXPECT_FALSE(verbose);
}

TEST_F(RasClientMicrotest, ParseArgsVersion_ShortDashR_IsNotAnOptionAndFallsToDefault) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "-r"});

  EXPECT_EQ(1, out.exitStatus);
  EXPECT_TRUE(LogHas(out.log, "ras-client-argv0-probe: invalid option -- 'r'\n")) << out.log;
  EXPECT_TRUE(LogHas(out.log, "Usage: ras-client-argv0-probe [OPTION]...\n")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "RCCL RAS client version ")) << out.log;
}

// --- default: the four distinct getopt routes ------------------------------
// The optstring has no leading ':', so getopt_long reports every one of these as
// '?'; ':' is never produced and there is no separate arm for it.

TEST_F(RasClientMicrotest, ParseArgsDefault_UnknownShortOption_PrintsUsageAndExitsOne) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "-z"});

  EXPECT_EQ(1, out.exitStatus);
  EXPECT_TRUE(LogHas(out.log, "ras-client-argv0-probe: invalid option -- 'z'\n")) << out.log;
  EXPECT_TRUE(LogHas(out.log, "Usage: ras-client-argv0-probe [OPTION]...\n")) << out.log;
  EXPECT_TRUE(LogHas(out.log, "      --help          Print this help and exit\n")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "RCCL RAS client version ")) << out.log;
  ExpectAllGlobalsAtDefault();
}

TEST_F(RasClientMicrotest, ParseArgsDefault_UnknownLongOption_PrintsUsageAndExitsOne) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "--bogus"});

  EXPECT_EQ(1, out.exitStatus);
  EXPECT_TRUE(LogHas(out.log, "ras-client-argv0-probe: unrecognized option '--bogus'\n")) << out.log;
  EXPECT_TRUE(LogHas(out.log, "Usage: ras-client-argv0-probe [OPTION]...\n")) << out.log;
  EXPECT_FALSE(LogHas(out.log, "RCCL RAS client version ")) << out.log;
  ExpectAllGlobalsAtDefault();
}

// "--ver" prefixes both "--verbose" and "--version", so getopt refuses to guess.
TEST_F(RasClientMicrotest, ParseArgsDefault_AmbiguousLongAbbreviation_PrintsUsageAndExitsOne) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "--ver"});

  EXPECT_EQ(1, out.exitStatus);
  EXPECT_TRUE(LogHas(out.log,
                     "ras-client-argv0-probe: option '--ver' is ambiguous; "
                     "possibilities: '--verbose' '--version'\n"))
      << out.log;
  EXPECT_TRUE(LogHas(out.log, "Usage: ras-client-argv0-probe [OPTION]...\n")) << out.log;
  EXPECT_FALSE(verbose);
  EXPECT_FALSE(LogHas(out.log, "RCCL RAS client version ")) << out.log;
}

TEST_F(RasClientMicrotest, ParseArgsDefault_ShortOptionMissingItsArgument_PrintsUsageAndExitsOne) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "-p"});

  EXPECT_EQ(1, out.exitStatus);
  EXPECT_TRUE(LogHas(out.log, "ras-client-argv0-probe: option requires an argument -- 'p'\n")) << out.log;
  EXPECT_TRUE(LogHas(out.log, "Usage: ras-client-argv0-probe [OPTION]...\n")) << out.log;
  EXPECT_STREQ(kDefaultPort, port);
}

TEST_F(RasClientMicrotest, ParseArgsDefault_LongOptionMissingItsArgument_PrintsUsageAndExitsOne) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "--port"});

  EXPECT_EQ(1, out.exitStatus);
  EXPECT_TRUE(LogHas(out.log, "ras-client-argv0-probe: option '--port' requires an argument\n")) << out.log;
  EXPECT_TRUE(LogHas(out.log, "Usage: ras-client-argv0-probe [OPTION]...\n")) << out.log;
  EXPECT_STREQ(kDefaultPort, port);
}

// An applied option before the bad one proves default: aborts the loop rather
// than the loop never having run.
TEST_F(RasClientMicrotest, ParseArgsDefault_AfterAnAppliedOption_ExitsOneAndStopsParsing) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "-v", "-z", "-p", "31337"});

  EXPECT_EQ(1, out.exitStatus);
  EXPECT_TRUE(verbose);
  EXPECT_STREQ(kDefaultPort, port);
  EXPECT_TRUE(LogHas(out.log, "ras-client-argv0-probe: invalid option -- 'z'\n")) << out.log;
  EXPECT_TRUE(LogHas(out.log, "Usage: ras-client-argv0-probe [OPTION]...\n")) << out.log;
}

// --- death tests: production really terminates -----------------------------

TEST_F(RasClientMicrotest, ParseArgsHelp_RealExit_TerminatesProcessWithStatusZero) {
  ScopedHook exitHook(g_exit, [](int status) { ::_exit(status); });
  RasArgv args{"ras-client-death-probe", "--help"};

  EXPECT_EXIT(parseArgs(args.argc(), args.argv()), ::testing::ExitedWithCode(0),
              "Usage: ras-client-death-probe \\[OPTION\\]\\.\\.\\.");
}

TEST_F(RasClientMicrotest, ParseArgsVersion_RealExit_TerminatesProcessWithStatusZero) {
  ScopedHook exitHook(g_exit, [](int status) { ::_exit(status); });
  RasArgv args{"ras-client-death-probe", "--version"};

  EXPECT_EXIT(parseArgs(args.argc(), args.argv()), ::testing::ExitedWithCode(0),
              "RCCL RAS client version [0-9]+\\.[0-9]+\\.[0-9]+");
}

TEST_F(RasClientMicrotest, ParseArgsDefault_RealExit_TerminatesProcessWithStatusOne) {
  ScopedHook exitHook(g_exit, [](int status) { ::_exit(status); });
  RasArgv args{"ras-client-death-probe", "--bogus"};

  EXPECT_EXIT(parseArgs(args.argc(), args.argv()), ::testing::ExitedWithCode(1),
              "unrecognized option '--bogus'");
}


// ===========================================================================
// setOutputFormat: the format guard, the write arm, the read arm, and the
// "OK\n" response check. Driven directly on the file-scope statics `format`
// and `sock` so these tests stay independent of the parseArgs tests above.
// ===========================================================================

namespace {

// Not one of the two values parseArgs accepts: setOutputFormat itself does no
// validation, so a distinctive value proves the bytes came from `format`.
constexpr const char kOddFormat[] = "quokka";

}  // namespace

// Guard arm, false side: `format` still NULL, so nothing is sent or read.
TEST_F(RasClientMicrotest, SetOutputFormat_FormatNeverSpecified_WritesNothingReadsNothingAndReturnsZero) {
  sock = 77;
  ASSERT_EQ(nullptr, format);

  int rc = -1;
  {
    ScopedHook writeHook(g_write, [](int, const void*, size_t count) { return static_cast<ssize_t>(count); });
    ScopedHook readHook(g_read, [](int, void*, size_t) { return static_cast<ssize_t>(0); });

    const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

    EXPECT_EQ(0, rc);
    EXPECT_EQ(0, writeHook.calls);
    EXPECT_EQ(0, readHook.calls);
    EXPECT_EQ("", log);
    EXPECT_TRUE(g_writtenData.empty()) << g_writtenData;
  }

  // Positive anchor: the same fixture state does write once `format` is set, so
  // the assertions above are not passing because the seams are inert.
  format = kOddFormat;
  ScriptReadData("OK\n");
  EXPECT_EQ(0, setOutputFormat());
  EXPECT_EQ("SET FORMAT quokka\n", g_writtenData);
}

// Guard arm, true side + the success return: exact bytes on the wire, on `sock`.
TEST_F(RasClientMicrotest, SetOutputFormat_ServerAnswersOk_SendsExactCommandOnSockAndReturnsZero) {
  sock = 77;
  format = kOddFormat;

  int writeFd = -1;
  int readFd = -1;
  bool served = false;
  int rc = -1;
  ScopedHook writeHook(g_write, [&](int fd, const void* buf, size_t count) {
    writeFd = fd;
    g_writtenData.append(static_cast<const char*>(buf), count);
    return static_cast<ssize_t>(count);
  });
  // Serves the reply exactly once; a hook that kept returning 0 would spin
  // rasRead's until-newline loop forever.
  ScopedHook readHook(g_read, [&](int fd, void* buf, size_t count) -> ssize_t {
    readFd = fd;
    if (served || count < 3) return 0;
    served = true;
    std::memcpy(buf, "OK\n", 3);
    return 3;
  });

  const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

  EXPECT_EQ(0, rc);
  EXPECT_EQ("SET FORMAT quokka\n", g_writtenData);
  EXPECT_EQ(77, writeFd);
  EXPECT_EQ(77, readFd);
  EXPECT_EQ(1, writeHook.calls);
  EXPECT_EQ(1, readHook.calls);
  EXPECT_EQ("", log);
}

// strcasecmp folds case, so a lowercase acknowledgement is still success.
TEST_F(RasClientMicrotest, SetOutputFormat_ServerAnswersLowercaseOk_IsAcceptedCaseInsensitively) {
  sock = 77;
  format = kOddFormat;
  ScriptReadData("ok\n");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

  EXPECT_EQ(0, rc);
  EXPECT_EQ("SET FORMAT quokka\n", g_writtenData);
  EXPECT_EQ("", log);
}

// strcasecmp compares whole strings: "OK" without the newline is a mismatch.
TEST_F(RasClientMicrotest, SetOutputFormat_ServerAnswersOkWithoutNewline_ReportsUnexpectedResponseAndReturnsOne) {
  sock = 77;
  format = kOddFormat;
  ScriptReadData("OK");  // rasRead then hits EOF and returns the 2 bytes

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

  EXPECT_EQ(1, rc);
  EXPECT_EQ("SET FORMAT quokka\n", g_writtenData);
  EXPECT_TRUE(LogHas(log, "Unexpected response from NCCL: OK\n")) << log;
  EXPECT_FALSE(LogHas(log, "NCCL unexpectedly closed the connection"));
}

// A correct prefix plus trailing bytes is also a mismatch.
TEST_F(RasClientMicrotest, SetOutputFormat_ServerAnswersOkWithTrailingText_ReportsUnexpectedResponseAndReturnsOne) {
  sock = 77;
  format = kOddFormat;
  ScriptReadData("OK\nextra");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

  EXPECT_EQ(1, rc);
  EXPECT_TRUE(LogHas(log, "Unexpected response from NCCL: OK\nextra\n")) << log;
  EXPECT_FALSE(LogHas(log, "Connection timed out"));
}

// "OKAY\n" shares the first two characters with "OK\n" and must still be rejected.
TEST_F(RasClientMicrotest, SetOutputFormat_ServerAnswersOkay_ReportsUnexpectedResponseAndReturnsOne) {
  sock = 77;
  format = kOddFormat;
  ScriptReadData("OKAY\n");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

  EXPECT_EQ(1, rc);
  EXPECT_EQ("SET FORMAT quokka\n", g_writtenData);
  EXPECT_TRUE(LogHas(log, "Unexpected response from NCCL: OKAY\n\n")) << log;
  EXPECT_FALSE(LogHas(log, "read socket"));
}

// Write arm, EAGAIN side. socketWrite surfaces a failed write as -1, never as a
// short count, so the hook must return -1 rather than a partial byte count.
TEST_F(RasClientMicrotest, SetOutputFormat_WriteFailsWithEagain_ReportsConnectionTimedOutAndReturnsOne) {
  sock = 77;
  format = kOddFormat;

  int rc = -1;
  ScopedHook writeHook(g_write, [](int, const void*, size_t) -> ssize_t {
    errno = EAGAIN;
    return -1;
  });
  ScopedHook readHook(g_read, [](int, void*, size_t) { return static_cast<ssize_t>(0); });

  const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

  EXPECT_EQ(1, rc);
  EXPECT_TRUE(LogHas(log, "Connection timed out\n")) << log;
  EXPECT_FALSE(LogHas(log, "write to socket"));
  EXPECT_EQ(1, writeHook.calls);
  EXPECT_EQ(0, readHook.calls);
}

// Write arm, non-EAGAIN side: perror, and the response read must not happen.
TEST_F(RasClientMicrotest, SetOutputFormat_WriteFailsWithBrokenPipe_ReportsPerrorAndReturnsOne) {
  sock = 77;
  format = kOddFormat;

  int rc = -1;
  ScopedHook writeHook(g_write, [](int, const void*, size_t) -> ssize_t {
    errno = EPIPE;
    return -1;
  });
  ScopedHook readHook(g_read, [](int, void*, size_t) { return static_cast<ssize_t>(0); });

  const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

  const std::string expected = std::string("write to socket: ") + std::strerror(EPIPE) + "\n";
  EXPECT_EQ(1, rc);
  EXPECT_TRUE(LogHas(log, expected.c_str())) << log;
  EXPECT_FALSE(LogHas(log, "Connection timed out"));
  EXPECT_EQ(0, readHook.calls);
}

// Read arm, EAGAIN side: the command still went out before the read failed.
TEST_F(RasClientMicrotest, SetOutputFormat_ReadFailsWithEagain_ReportsConnectionTimedOutAndReturnsOne) {
  sock = 77;
  format = kOddFormat;
  ScriptRead(-1, EAGAIN, "");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

  EXPECT_EQ(1, rc);
  EXPECT_EQ("SET FORMAT quokka\n", g_writtenData);
  EXPECT_TRUE(LogHas(log, "Connection timed out\n")) << log;
  EXPECT_FALSE(LogHas(log, "read socket"));
}

// Read arm, non-EAGAIN side.
TEST_F(RasClientMicrotest, SetOutputFormat_ReadFailsWithConnectionReset_ReportsPerrorAndReturnsOne) {
  sock = 77;
  format = kOddFormat;
  ScriptRead(-1, ECONNRESET, "");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

  const std::string expected = std::string("read socket: ") + std::strerror(ECONNRESET) + "\n";
  EXPECT_EQ(1, rc);
  EXPECT_EQ("SET FORMAT quokka\n", g_writtenData);
  EXPECT_TRUE(LogHas(log, expected.c_str())) << log;
  EXPECT_FALSE(LogHas(log, "Connection timed out"));
}

// bytes == 0 arm: the empty read script makes the default read report EOF.
TEST_F(RasClientMicrotest, SetOutputFormat_ServerClosesConnection_ReportsUnexpectedCloseAndReturnsOne) {
  sock = 77;
  format = kOddFormat;

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

  EXPECT_EQ(1, rc);
  EXPECT_EQ("SET FORMAT quokka\n", g_writtenData);
  EXPECT_TRUE(LogHas(log, "NCCL unexpectedly closed the connection\n")) << log;
  EXPECT_FALSE(LogHas(log, "Unexpected response from NCCL"));
  EXPECT_FALSE(LogHas(log, "read socket"));
}

// snprintf truncates silently and its return value is discarded, so an
// over-long format ships a 4095-byte command whose terminating '\n' is gone.
TEST_F(RasClientMicrotest, SetOutputFormat_FormatOverflowsMsgBuf_TruncatesCommandAndDropsTheNewline) {
  const std::string longFormat(4200, 'z');
  sock = 77;
  format = longFormat.c_str();
  ScriptReadData("OK\n");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

  EXPECT_EQ(0, rc);  // truncation is not detected or reported
  EXPECT_EQ("", log);
  ASSERT_EQ(4095u, g_writtenData.size());
  EXPECT_EQ("SET FORMAT zzzz", g_writtenData.substr(0, 15));
  EXPECT_EQ('z', g_writtenData.back());
  EXPECT_EQ(std::string::npos, g_writtenData.find('\n'));
}
