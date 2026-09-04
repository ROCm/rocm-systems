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
// with fakes/libc_seam.h renaming each libc entry point to a micro_*
// trampoline that dispatches through a swappable slot in fakes/libc_fakes.h.
//
// getopt_long is deliberately NOT seamed: its parse behaviour is part of what
// parseArgs is being tested for. ResetRasClientGlobals() resets optind instead.

#include <gtest/gtest.h>

// Every header that DECLARES a name libc_seam.h renames, pulled in BEFORE the seam so a rename never rewrites a
// declaration. client.cc reaches many more headers than these through os.h/nccl.h/ras_internal.h, but those are
// included after the seam and are harmless as long as none of them is the first declaration of a seamed name.
#include <arpa/inet.h>
#include <getopt.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "fakes/libc_fakes.h"
#include "fakes/libc_seam.h"

// client.cc's main() would collide with the gtest main in main_altrsmi.cpp.
#define main rasClientMain

#include RAS_CLIENT_CC_PATH

#undef main
#include "fakes/libc_seam_undef.h"

using RcclUnitTesting::CaptureLog;

// The errnos the unit perror()'d, in call order.
//
// Prefer this over matching the sentence on stderr. Every failure arm in client.cc is report-and-return-1: the arms
// return the same value, close nothing, and write nothing, so the diagnostic is the only thing that separates them.
// Matching its text couples the test to prose AND to libc's strerror table; the errno is the condition the code
// actually branched on, so asserting it checks the arm and survives any rewording.
//
// An empty expectation is meaningful: `ExpectPerrors({})` with rc == 1 pins the EAGAIN/EWOULDBLOCK arm, which is
// exactly "reported a timeout instead of calling perror".
void ExpectPerrors(const std::vector<int>& want) {
  std::vector<int> got;
  got.reserve(g_perrorCalls.size());
  for (const MicroPerrorCall& call : g_perrorCalls) {
    got.push_back(call.err);
  }
  EXPECT_EQ(want, got);
}

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
// in test/test_categories_micro.yaml. The trailing '.*' there anchors the end
// of the suite name, so a derived suite is not covered by this one's pattern
// and would build, pass locally, and never run under CTest.
// ===========================================================================
class RasClientMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetLibcFakes();
    ResetRasClientGlobals();
  }
  void TearDown() override {
    ResetLibcFakes();
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

  EXPECT_THROW(micro_exit(3), MicroExit);
}

// ===========================================================================
// parseArgs: the --format/-f and --timeout/-t validation arms.
// ===========================================================================

namespace {

constexpr int kParseArgsNoExit = -999;

struct ParseArgsOutcome {
  int exitStatus = kParseArgsNoExit;
  std::string log;
  // parseArgs stores optarg straight into its globals (client.cc:76 `format = optarg`), so the argv strings must
  // outlive every assertion that reads one, not merely the parseArgs call. Keeping them here ties their lifetime to
  // the outcome the test holds. Returning by value is safe: a vector move steals the buffer, it does not relocate.
  std::vector<std::string> argvStorage;
};

// getopt permutes argv in place, so it needs writable storage; `storage` is the caller's for the lifetime reason above.
void InvokeParseArgs(std::vector<std::string>& storage) {
  std::vector<char*> argv;
  argv.reserve(storage.size() + 1);
  for (std::string& a : storage) {
    argv.push_back(&a[0]);
  }
  argv.push_back(nullptr);
  parseArgs(static_cast<int>(storage.size()), argv.data());
}

// The catch must sit inside CaptureLog: gtest has a single stderr capture slot
// and an exception escaping the body would leave it open for the next test.
ParseArgsOutcome RunParseArgs(const std::vector<std::string>& args) {
  ParseArgsOutcome out;
  out.argvStorage.reserve(args.size() + 1);
  out.argvStorage.emplace_back("rccl-ras-client");
  out.argvStorage.insert(out.argvStorage.end(), args.begin(), args.end());
  out.log = CaptureLog([&]() {
    try {
      InvokeParseArgs(out.argvStorage);
    } catch (const MicroExit& e) {
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
  // A dropped `break` in case 'f' falls through to case 'h', which stores optarg here.
  EXPECT_STREQ("localhost", hostName);
}

TEST_F(RasClientMicrotest, ParseArgsFormat_ShortJson_StoresJsonAndContinues) {
  const ParseArgsOutcome out = RunParseArgs({"-f", "json"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("json", format);
  EXPECT_STREQ("localhost", hostName);
}

TEST_F(RasClientMicrotest, ParseArgsFormat_LongEqualsJson_StoresJson) {
  const ParseArgsOutcome out = RunParseArgs({"--format=json"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("json", format);
}

TEST_F(RasClientMicrotest, ParseArgsFormat_LongSeparateText_StoresText) {
  const ParseArgsOutcome out = RunParseArgs({"--format", "text"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("text", format);
}

TEST_F(RasClientMicrotest, ParseArgsFormat_UppercaseJson_IsAcceptedCaseInsensitively) {
  const ParseArgsOutcome out = RunParseArgs({"-f", "JSON"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("JSON", format);  // stored verbatim; only the comparison folds case
}

TEST_F(RasClientMicrotest, ParseArgsFormat_MixedCaseText_IsAcceptedCaseInsensitively) {
  const ParseArgsOutcome out = RunParseArgs({"--format=TeXt"});

  EXPECT_EQ(kParseArgsNoExit, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("TeXt", format);
}

// --- case 'f': rejecting arm -----------------------------------------------

TEST_F(RasClientMicrotest, ParseArgsFormat_UnknownValue_ReportsInvalidFormatAndExitsOne) {
  const ParseArgsOutcome out = RunParseArgs({"-f", "xml"});

  EXPECT_EQ(1, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("xml", format);  // the store happens before the validation
}

TEST_F(RasClientMicrotest, ParseArgsFormat_TextWithSuffix_ReportsInvalidFormatAndExitsOne) {
  const ParseArgsOutcome out = RunParseArgs({"--format=texts"});

  EXPECT_EQ(1, out.exitStatus);
  ASSERT_NE(nullptr, format);
  EXPECT_STREQ("texts", format);
}

TEST_F(RasClientMicrotest, ParseArgsFormat_UnknownValue_TerminatesProcessWithStatusOne) {
  ScopedHook exitHook(g_exit, [](int status) { ::_exit(status); });
  std::vector<std::string> argv{"rccl-ras-client", "-f", "xml"};

  EXPECT_EXIT(InvokeParseArgs(argv), ::testing::ExitedWithCode(1), ".*");
}

// --- case 't': accepting arms ----------------------------------------------

// --- case 't': rejecting arms ----------------------------------------------

// case 't' is one strtol plus one `< 0 || *endPtr` check, so the three spellings differ only in how getopt hands the
// value over and the values differ only in what strtol makes of them. One test per side, driven by a table.
TEST_F(RasClientMicrotest, ParseArgsTimeout_AcceptedValues_StoreTheParsedSeconds) {
  const struct { std::vector<std::string> argv; int want; } cases[] = {
      {{"-t", "37"}, 37},          // short, separate argument
      {{"--timeout=37"}, 37},      // long, attached
      {{"--timeout", "37"}, 37},   // long, separate
      {{"-t", "0"}, 0},            // 0 is valid and disables the timeout
      {{"-t", "010"}, 10},         // base 10: 8 if the base were 0, 16 if it were 16
  };
  for (const auto& c : cases) {
    ResetLibcFakes();
    ResetRasClientGlobals();
    const ParseArgsOutcome out = RunParseArgs(c.argv);

    EXPECT_EQ(kParseArgsNoExit, out.exitStatus) << c.argv[0];
    EXPECT_EQ(c.want, timeout) << c.argv[0];
    // A dropped `break` in case 't' falls through to case 'v', which sets this.
    EXPECT_FALSE(verbose) << c.argv[0];
  }
}

// The rejecting side. `timeout` still carries what strtol produced: the store precedes the check, so a mutant that
// moved the check first would leave the -1 initializer here instead.
TEST_F(RasClientMicrotest, ParseArgsTimeout_RejectedValues_ExitOneAfterStoringWhatStrtolParsed) {
  const struct { std::vector<std::string> argv; int stored; } cases[] = {
      {{"-t", "-5"}, -5},          // negative
      {{"-t", "5x"}, 5},           // trailing garbage
      {{"--timeout=abc"}, 0},      // strtol consumed nothing
      {{"-t", "0x10"}, 0},         // base 10 stops at 'x'
  };
  for (const auto& c : cases) {
    ResetLibcFakes();
    ResetRasClientGlobals();
    const ParseArgsOutcome out = RunParseArgs(c.argv);

    EXPECT_EQ(1, out.exitStatus) << c.argv[0];
    EXPECT_EQ(c.stored, timeout) << c.argv[0];
  }
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

// Scripted read outcomes reuse MicroReadStep from fakes/libc_fakes.h: ret < 0 fails with `err`, ret == 0 is EOF.

// Records every request and serves `steps` front-to-back; past the end, and for
// a zero-length request, it returns 0 exactly as a real read(2) would. It never
// writes more than the caller asked for, so an overflow seen in a test is the
// unit's, not the fake's.
class RecordingReader {
 public:
  RecordingReader(const char* base, std::vector<MicroReadStep> steps) : base_(base), steps_(std::move(steps)) {}

  ssize_t operator()(int, void* buf, size_t count) {
    requests.push_back(ReadRequest{static_cast<const char*>(buf) - base_, count});
    if (count == 0 || pos_ >= steps_.size()) return 0;
    const MicroReadStep& step = steps_[pos_++];
    if (step.ret < 0) {
      errno = step.err;
      return step.ret;
    }
    if (step.ret == 0) return 0;
    // Same rule the default read seam enforces: a positive ret is the promise, so it must match the bytes on offer,
    // and the delivery is clamped to it so a mismatched script cannot over-deliver where NDEBUG drops the assert.
    assert(static_cast<size_t>(step.ret) == step.data.size() && "RecordingReader: positive ret must equal data.size()");
    const size_t promised = static_cast<size_t>(step.ret) < step.data.size() ? static_cast<size_t>(step.ret)
                                                                             : step.data.size();
    const size_t n = promised < count ? promised : count;
    memcpy(buf, step.data.data(), n);
    return static_cast<ssize_t>(n);
  }

  std::vector<ReadRequest> requests;

 private:
  const char* base_;
  std::vector<MicroReadStep> steps_;
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

// Same arm through the DEFAULT read seam rather than RecordingReader: the zero-length request must read as EOF without
// consuming a script step, or every later step would land on the wrong call. The second step below stays untouched.
TEST_F(RasClientMicrotest, RasRead_PayloadFillsBuffer_ZeroLengthRequestSpendsNoScriptStep) {
  char buf[4];
  ScriptReadData("abc");
  ScriptReadData("XYZ");

  EXPECT_EQ(3, rasRead(9, buf, sizeof(buf)));

  EXPECT_EQ("abc", std::string(buf, 3));
  EXPECT_EQ('\0', buf[3]);
  EXPECT_EQ(1u, g_readScriptPos);  // only the "abc" step was consumed
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

// Arm: ret == 0 on a non-empty write. `done += 0` makes no progress and the guard is unchanged, so production retries
// the identical call forever; against a real write(2) returning 0 this loop never terminates. Only WriteRecorder's
// kHardCap turns that into a finite EIO here, which is what makes the path observable at all.
TEST_F(RasClientMicrotest, SocketWrite_ZeroReturnOnNonEmptyBuffer_MakesNoProgressAndRepeatsTheSameCall) {
  WriteRecorder rec(kPayload, {{0, 0}, {0, 0}, {0, 0}});
  ScopedHook writeHook(g_write, [&rec](int fd, const void* buf, size_t count) { return rec(fd, buf, count); });

  errno = 0;
  // The fourth call exhausts the script and fails with EIO; without that the call above would not return.
  EXPECT_EQ(static_cast<ssize_t>(-1), socketWrite(9, kPayload, kPayloadLen));
  EXPECT_EQ(EIO, errno);
  EXPECT_EQ(4, writeHook.calls);
  EXPECT_EQ("0/19,0/19,0/19,0/19", FormatWriteCalls(rec.calls));
  EXPECT_EQ("", rec.data);
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
  out.argvStorage.assign(args.begin(), args.end());
  out.log = CaptureLog([&]() {
    try {
      InvokeParseArgs(out.argvStorage);
    } catch (const MicroExit& e) {
      out.exitStatus = e.status;
    }
  });
  return out;
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

// printUsage's entire product is the text it writes to stderr, so with output assertions gone there is nothing left
// to check about WHAT it prints: the golden-string comparison, the per-line needles and the argv0 echo all went with
// them. What remains observable is that it writes only to stderr, leaves every parse global alone and returns rather
// than exiting, which is what the two exiting arms of parseArgs rely on.

// printUsage writes nowhere but stderr and changes no parse state; without this
// the two exiting arms cannot be told apart from the function they call.
TEST_F(RasClientMicrotest, PrintUsage_CalledDirectly_TouchesNoParseStateAndDoesNotExit) {
  const std::string log = CaptureLog([]() { printUsage(kUsageProg); });

  ExpectAllGlobalsAtDefault();
}

// --- case 'e': --help ------------------------------------------------------

// --help must exit before any option after it is looked at.
TEST_F(RasClientMicrotest, ParseArgsHelp_FollowedByOtherOptions_ExitsBeforeApplyingThem) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "--help", "-v", "-p", "31337"});

  EXPECT_EQ(0, out.exitStatus);
  EXPECT_FALSE(verbose);
  EXPECT_STREQ(kDefaultPort, port);
}

// --help is no_argument: an earlier option's value must not be swallowed by it.
TEST_F(RasClientMicrotest, ParseArgsHelp_AfterAnAppliedOption_StillExitsZeroWithUsage) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "-p", "31337", "--help"});

  EXPECT_EQ(0, out.exitStatus);
  EXPECT_STREQ("31337", port);
}

// --- case 'r': --version ---------------------------------------------------

// --version exits before later options are applied, and 'r' is unreachable via
// any short option, so "-r" is an unknown option instead.
TEST_F(RasClientMicrotest, ParseArgsVersion_FollowedByOtherOptions_ExitsBeforeApplyingThem) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "--version", "-v"});

  EXPECT_EQ(0, out.exitStatus);
  EXPECT_FALSE(verbose);
}

// --- default: the four distinct getopt routes ------------------------------
// The optstring has no leading ':', so getopt_long reports every one of these as
// '?'; ':' is never produced and there is no separate arm for it.

// An applied option before the bad one proves default: aborts the loop rather
// than the loop never having run.
TEST_F(RasClientMicrotest, ParseArgsDefault_AfterAnAppliedOption_ExitsOneAndStopsParsing) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "-v", "-z", "-p", "31337"});

  EXPECT_EQ(1, out.exitStatus);
  EXPECT_TRUE(verbose);
  EXPECT_STREQ(kDefaultPort, port);
}

// --- the exiting arms: case 'e', case 'r' and default: ----------------------

// All three print and exit without touching a global, so once the printed text is not asserted they are told apart
// only by the status. Grouped by status rather than one test per spelling; the inputs are what carry the coverage.
//
// The exit-0 set reaches case 'e' and case 'r', including a unique long-option abbreviation ("--hel" is not a prefix
// of "--host", so getopt resolves it rather than reporting ambiguity).
TEST_F(RasClientMicrotest, ParseArgsExitingArms_ExitZeroInputs_LeaveEveryGlobalAtItsDefault) {
  for (const char* opt : {"--help", "--hel", "--version"}) {
    ResetLibcFakes();
    ResetRasClientGlobals();
    const ParseArgsOutcome out = RunParseArgv({kUsageProg, opt});

    EXPECT_EQ(0, out.exitStatus) << opt;
    ExpectAllGlobalsAtDefault();
  }
}

// The exit-1 set is every route into default:. The optstring has no leading ':', so getopt_long reports all of them
// as '?' and there is no separate arm for a missing argument; "-r" is here because 'r' is long-only, and "--ver"
// because it prefixes both "--verbose" and "--version". The two trailing-option cases also pin the optstring's "p:"
// arity: with a bare "p", a trailing -p would be a valid no-argument option and would skip this arm entirely.
TEST_F(RasClientMicrotest, ParseArgsExitingArms_ExitOneInputs_LeaveEveryGlobalAtItsDefault) {
  for (const char* opt : {"-z", "--bogus", "--ver", "-r", "-p", "--port"}) {
    ResetLibcFakes();
    ResetRasClientGlobals();
    const ParseArgsOutcome out = RunParseArgv({kUsageProg, opt});

    EXPECT_EQ(1, out.exitStatus) << opt;
    ExpectAllGlobalsAtDefault();
  }
}

// --- death tests: production really terminates -----------------------------

TEST_F(RasClientMicrotest, ParseArgsHelp_RealExit_TerminatesProcessWithStatusZero) {
  ScopedHook exitHook(g_exit, [](int status) { ::_exit(status); });
  RasArgv args{"ras-client-death-probe", "--help"};

  EXPECT_EXIT(parseArgs(args.argc(), args.argv()), ::testing::ExitedWithCode(0), ".*");
}

TEST_F(RasClientMicrotest, ParseArgsDefault_RealExit_TerminatesProcessWithStatusOne) {
  ScopedHook exitHook(g_exit, [](int status) { ::_exit(status); });
  RasArgv args{"ras-client-death-probe", "--bogus"};

  EXPECT_EXIT(parseArgs(args.argc(), args.argv()), ::testing::ExitedWithCode(1), ".*");
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
}

// The response check is one strcasecmp against "OK\n", so every accepting and every rejecting response exercises the
// same comparison and differs only in its input. One test per side, driven by a table, instead of one per response:
// with the diagnostic no longer asserted the per-response tests held identical expectations anyway.
TEST_F(RasClientMicrotest, SetOutputFormat_AcceptedResponses_ReturnZeroAndStillSendTheCommand) {
  for (const char* reply : {"OK\n", "ok\n", "Ok\n"}) {
    ResetLibcFakes();
    ResetRasClientGlobals();
    sock = 77;
    format = kOddFormat;
    ScriptReadData(reply);

    int rc = -1;
    const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

    EXPECT_EQ(0, rc) << reply;
    EXPECT_EQ("SET FORMAT quokka\n", g_writtenData) << reply;
    ExpectPerrors({});
  }
}

// The rejecting side, including the two that are not a mismatch at all but an empty read: "" is what the default read
// yields with no script, i.e. the server closed. All of them reach a report-and-return-1 arm.
TEST_F(RasClientMicrotest, SetOutputFormat_RejectedResponses_ReturnOneAfterSendingTheCommand) {
  for (const char* reply : {"OK",          // whole-string compare: no newline is a mismatch
                            "OK\nextra",   // correct prefix, trailing bytes
                            "OKAY\n",      // shares the first two characters
                            "okay\n",
                            ""}) {         // no script at all: rasRead reports EOF
    ResetLibcFakes();
    ResetRasClientGlobals();
    sock = 77;
    format = kOddFormat;
    if (*reply != '\0') ScriptReadData(reply);

    int rc = -1;
    const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

    EXPECT_EQ(1, rc) << reply;
    EXPECT_EQ("SET FORMAT quokka\n", g_writtenData) << reply;
    ExpectPerrors({});
  }
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
  ExpectPerrors({});
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

  EXPECT_EQ(1, rc);
  ExpectPerrors({EPIPE});
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
  ExpectPerrors({});
}

// Read arm, non-EAGAIN side.
TEST_F(RasClientMicrotest, SetOutputFormat_ReadFailsWithConnectionReset_ReportsPerrorAndReturnsOne) {
  sock = 77;
  format = kOddFormat;
  ScriptRead(-1, ECONNRESET, "");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = setOutputFormat(); });

  EXPECT_EQ(1, rc);
  EXPECT_EQ("SET FORMAT quokka\n", g_writtenData);
  ExpectPerrors({ECONNRESET});
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
  ASSERT_EQ(4095u, g_writtenData.size());
  EXPECT_EQ("SET FORMAT zzzz", g_writtenData.substr(0, 15));
  EXPECT_EQ('z', g_writtenData.back());
  EXPECT_EQ(std::string::npos, g_writtenData.find('\n'));
}


// ===========================================================================
// getNCCLStatus: the STATUS command, the streaming read loop, and every
// failure arm (short write, read error, short fwrite, fflush error).
// ===========================================================================

namespace {

// Chunks with no '\n' and three different lengths. The absence of a newline is
// load-bearing: it is what makes the untilNewline=false argument observable.
constexpr const char kChunk1[] = "alpha";         // 5 bytes
constexpr const char kChunk2[] = "bravocharlie";  // 12 bytes
constexpr const char kChunk3[] = "delta7";        // 6 bytes

struct FwriteCall {
  size_t size;
  size_t nmemb;
  FILE* stream;
};

std::vector<FwriteCall> g_fwriteCalls;

// Records (size, nmemb, stream) then behaves like the default fwrite. Without the (size, nmemb)
// record the fwrite(buf,1,bytes) vs fwrite(buf,bytes,1) swap is invisible to g_stdoutData.
size_t RecordingFwrite(const void* ptr, size_t size, size_t nmemb, FILE* stream) {
  g_fwriteCalls.push_back(FwriteCall{size, nmemb, stream});
  g_stdoutData.append(static_cast<const char*>(ptr), size * nmemb);
  return nmemb;
}

void ScriptThreeChunks() {
  ScriptReadData(kChunk1);
  ScriptReadData(kChunk2);
  ScriptReadData(kChunk3);
}

}  // namespace

// Derived fixture solely so g_fwriteCalls is cleared alongside the shared fakes;
// suite name RasClientMicrotestGetStatus needs its own registration line.
class RasClientMicrotestGetStatus : public RasClientMicrotest {
 protected:
  void SetUp() override {
    RasClientMicrotest::SetUp();
    g_fwriteCalls.clear();
    sock = 17;  // distinctive, so the fd the client forwards is provably its own
  }
  void TearDown() override {
    g_fwriteCalls.clear();
    RasClientMicrotest::TearDown();
  }
};

// Command arm, verbose off: exactly "STATUS\n" on the wire, and the empty read
// script makes the first rasRead return 0, taking the EOF break to return 0.
TEST_F(RasClientMicrotestGetStatus, GetNcclStatus_NonVerbose_SendsPlainStatusAndReturnsZeroOnEof) {
  verbose = false;
  int writeFd = -1;
  ScopedHook writeHook(g_write, [&](int fd, const void* buf, size_t count) -> ssize_t {
    writeFd = fd;
    g_writtenData.append(static_cast<const char*>(buf), count);
    return static_cast<ssize_t>(count);
  });
  ScopedHook fwriteHook(g_fwrite, RecordingFwrite);
  ScopedHook fflushHook(g_fflush, [](FILE*) { return 0; });

  const std::string log = CaptureLog([]() { EXPECT_EQ(0, getNCCLStatus()); });

  EXPECT_EQ("STATUS\n", g_writtenData);
  EXPECT_EQ(17, writeFd);
  EXPECT_EQ(1, writeHook.calls);
  EXPECT_EQ(0, fwriteHook.calls);
  EXPECT_EQ(0, fflushHook.calls);
  EXPECT_EQ("", g_stdoutData);
}

// Command arm, verbose on: the prefix is "VERBOSE " with its trailing space.
TEST_F(RasClientMicrotestGetStatus, GetNcclStatus_Verbose_SendsVerbosePrefixedStatusAndReturnsZero) {
  verbose = true;
  ScopedHook fwriteHook(g_fwrite, RecordingFwrite);

  const std::string log = CaptureLog([]() { EXPECT_EQ(0, getNCCLStatus()); });

  EXPECT_EQ("VERBOSE STATUS\n", g_writtenData);
  EXPECT_EQ(0, fwriteHook.calls);
}

// Loop body: three reads of different lengths each become their own
// fwrite(buf, 1, bytes, stdout) + fflush, in order, then EOF returns 0.
TEST_F(RasClientMicrotestGetStatus, GetNcclStatus_ThreeChunks_StreamsEachToStdoutInOrder) {
  ScriptThreeChunks();
  ScopedHook fwriteHook(g_fwrite, RecordingFwrite);
  ScopedHook fflushHook(g_fflush, [](FILE*) { return 0; });

  const std::string log = CaptureLog([]() { EXPECT_EQ(0, getNCCLStatus()); });

  EXPECT_EQ("STATUS\n", g_writtenData);
  EXPECT_EQ("alphabravocharliedelta7", g_stdoutData);
  EXPECT_EQ(3, fwriteHook.calls);
  EXPECT_EQ(3, fflushHook.calls);
  EXPECT_EQ(3u, g_readScriptPos);  // every scripted read was consumed, then past-the-end EOF
  ASSERT_EQ(3u, g_fwriteCalls.size());
  EXPECT_EQ(1u, g_fwriteCalls[0].size);
  EXPECT_EQ(5u, g_fwriteCalls[0].nmemb);
  EXPECT_EQ(1u, g_fwriteCalls[1].size);
  EXPECT_EQ(12u, g_fwriteCalls[1].nmemb);
  EXPECT_EQ(1u, g_fwriteCalls[2].size);
  EXPECT_EQ(6u, g_fwriteCalls[2].nmemb);
  EXPECT_EQ(stdout, g_fwriteCalls[0].stream);
}

// Write arm, EAGAIN: socketWrite returns -1 with a timeout errno.
TEST_F(RasClientMicrotestGetStatus, GetNcclStatus_WriteFailsWithEagain_ReportsTimeoutAndReturnsOne) {
  ScriptThreeChunks();  // never consumed: the command write fails first
  ScopedHook fwriteHook(g_fwrite, RecordingFwrite);
  ScopedHook writeHook(g_write, [](int, const void*, size_t) -> ssize_t {
    errno = EAGAIN;
    return -1;
  });

  const std::string log = CaptureLog([]() { EXPECT_EQ(1, getNCCLStatus()); });

  EXPECT_EQ(1, writeHook.calls);
  EXPECT_EQ(0, fwriteHook.calls);
  EXPECT_EQ(0u, g_readScriptPos);
  ExpectPerrors({});
}

// Write arm, non-EAGAIN: perror prints the label, ": " and the strerror text.
TEST_F(RasClientMicrotestGetStatus, GetNcclStatus_WriteFailsWithEpipe_ReportsWriteToSocketAndReturnsOne) {
  ScopedHook fwriteHook(g_fwrite, RecordingFwrite);
  ScopedHook writeHook(g_write, [](int, const void*, size_t) -> ssize_t {
    errno = EPIPE;
    return -1;
  });

  const std::string log = CaptureLog([]() { EXPECT_EQ(1, getNCCLStatus()); });

  EXPECT_EQ(1, writeHook.calls);
  EXPECT_EQ(0, fwriteHook.calls);
  ExpectPerrors({EPIPE});
}

// Read arm, EWOULDBLOCK: the first chunk has already been streamed, so the
// "Connection timed out" here is provably the read site, not the write site.
TEST_F(RasClientMicrotestGetStatus, GetNcclStatus_ReadFailsWithEwouldblock_ReportsTimeoutAndReturnsOne) {
  ScriptReadData(kChunk1);
  ScriptRead(-1, EWOULDBLOCK, "");
  ScopedHook fwriteHook(g_fwrite, RecordingFwrite);

  const std::string log = CaptureLog([]() { EXPECT_EQ(1, getNCCLStatus()); });

  EXPECT_EQ("STATUS\n", g_writtenData);
  EXPECT_EQ("alpha", g_stdoutData);
  EXPECT_EQ(1, fwriteHook.calls);
  ExpectPerrors({});
}

// Read arm, non-EAGAIN errno.
TEST_F(RasClientMicrotestGetStatus, GetNcclStatus_ReadFailsWithEio_ReportsReadSocketAndReturnsOne) {
  ScriptReadData(kChunk1);
  ScriptRead(-1, EIO, "");
  ScopedHook fwriteHook(g_fwrite, RecordingFwrite);

  const std::string log = CaptureLog([]() { EXPECT_EQ(1, getNCCLStatus()); });

  EXPECT_EQ("alpha", g_stdoutData);
  EXPECT_EQ(1, fwriteHook.calls);
  ExpectPerrors({EIO});
}

// fwrite arm: the second chunk is short-written, so the loop must abort there
// without fflushing it and without touching the third chunk.
TEST_F(RasClientMicrotestGetStatus, GetNcclStatus_ShortFwrite_ReportsFailureAndReturnsOne) {
  ScriptThreeChunks();
  ScopedHook fflushHook(g_fflush, [](FILE*) { return 0; });
  int fwriteSeq = 0;  // sequencing only; fwriteHook.calls stays the assertion surface
  ScopedHook fwriteHook(g_fwrite, [&](const void* ptr, size_t size, size_t nmemb, FILE* stream) -> size_t {
    if (++fwriteSeq == 2) return nmemb - 1;
    return RecordingFwrite(ptr, size, nmemb, stream);
  });

  const std::string log = CaptureLog([]() { EXPECT_EQ(1, getNCCLStatus()); });

  EXPECT_EQ("alpha", g_stdoutData);
  EXPECT_EQ(2, fwriteHook.calls);
  EXPECT_EQ(1, fflushHook.calls);
  EXPECT_EQ(2u, g_readScriptPos);
}

// fflush arm: the second flush fails, after its chunk already reached stdout.
TEST_F(RasClientMicrotestGetStatus, GetNcclStatus_FflushFails_ReportsPerrorAndReturnsOne) {
  ScriptThreeChunks();
  ScopedHook fwriteHook(g_fwrite, RecordingFwrite);
  int fflushSeq = 0;  // sequencing only; fflushHook.calls stays the assertion surface
  ScopedHook fflushHook(g_fflush, [&](FILE*) {
    if (++fflushSeq == 2) {
      errno = ENOSPC;
      return EOF;
    }
    return 0;
  });

  const std::string log = CaptureLog([]() { EXPECT_EQ(1, getNCCLStatus()); });

  EXPECT_EQ("alphabravocharlie", g_stdoutData);
  EXPECT_EQ(2, fwriteHook.calls);
  EXPECT_EQ(2, fflushHook.calls);
  EXPECT_EQ(2u, g_readScriptPos);
  ExpectPerrors({ENOSPC});
}



// ===========================================================================
// connectToNCCL -- address resolution and the connect walk
// (src/ras/client.cc:165-218, plus the fail: cleanup at 293-296)
//
// Every test here ends the unit at a `goto fail` arm, so nothing below line 218
// runs with anything but its default seam. The tests whose connect succeeds get
// there by leaving the read script empty: the handshake read returns 0 (EOF),
// which is the one handshake exit that cannot loop back to `retry:`.
// ===========================================================================

namespace {

// Not "localhost"/"28028": using the defaults here would make an assertion on
// the stored host/port pass even if connectToNCCL never read them.
constexpr const char kOtherHost[] = "rasnode7";
constexpr const char kOtherPort[] = "31337";

// The default getaddrinfo hands back entry i as 127.0.0.(1+i):(28028+i).
constexpr int kEntryPort0 = 28028;
constexpr int kEntryPort1 = 28029;

// One socket(2) call as connectToNCCL issued it.
struct SocketCall {
  int family;
  int socktype;
  int protocol;
};

// One connect(2) call: which fd, and which addrinfo entry it was aimed at.
struct ConnectCall {
  int fd;
  int port;
  uint32_t addr;
  socklen_t addrlen;
};

// One setsockopt(2) call, with the timeval it carried.
struct SockoptCall {
  int fd;
  int level;
  int optname;
  long sec;
  long usec;
};

// Hands out a distinct fd per socket() call and records the sequence. A hook
// cannot read its own ScopedHook's .calls (CTAD forbids it), and per-entry fds
// are what make "which addrinfo entry got closed" assertable.
class FdSequence {
 public:
  explicit FdSequence(int base) : base_(base) {}
  int operator()() {
    issued.push_back(base_ + static_cast<int>(issued.size()));
    return issued.back();
  }
  std::vector<int> issued;

 private:
  int base_;
};

int PortOf(const struct sockaddr* sa) {
  return ntohs(reinterpret_cast<const struct sockaddr_in*>(sa)->sin_port);
}

uint32_t AddrOf(const struct sockaddr* sa) {
  return ntohl(reinterpret_cast<const struct sockaddr_in*>(sa)->sin_addr.s_addr);
}


constexpr const char kFailedHeader[] = "Failed to connect to the NCCL RAS service!\n";
constexpr const char kArgsAdvice[] = "the host/port arguments are correct and match NCCL_RAS_ADDR.\n";
constexpr const char kLocalAdvice[] = "the RAS client was started on a node where the NCCL job is running.\n";

// The one handshake outcome reachable from a successful connect that cannot
// reach the `timeout:` label, whose `goto retry` would re-enter the walk.
constexpr const char kHandshakeEof[] = "NCCL unexpectedly closed the connection\n";

}  // namespace

// Arm: getaddrinfo returns non-zero. hostName/port/gai_strerror(ret) reach the
// message, and fail: has nothing to clean up because addrInfo is still null.
TEST_F(RasClientMicrotest, ConnectToNccl_GetaddrinfoFails_ReportsResolveErrorAndLeavesNothingToClean) {
  hostName = kOtherHost;
  port = kOtherPort;
  g_getaddrinfoResult = EAI_NONAME;

  int gaiCode = 0;
  ScopedHook gai(g_gaiStrerror, [&gaiCode](int code) {
    gaiCode = code;
    return "scripted resolver failure";
  });
  ScopedHook sockHook(g_socket, [](int, int, int) { return 77; });

  int ret = -1;
  const std::string log = CaptureLog([&ret]() { ret = connectToNCCL(); });

  EXPECT_EQ(1, ret);
  EXPECT_EQ(1, gai.calls);
  EXPECT_EQ(EAI_NONAME, gaiCode);
  EXPECT_EQ(0, sockHook.calls);
  EXPECT_EQ(0, g_freeaddrinfoCalls);
  EXPECT_TRUE(g_closedFds.empty());
  EXPECT_EQ(-1, sock);
}

// Arm: the two `hints` stores plus the (hostName, port) pair handed to
// getaddrinfo -- a resolver fake that ignores them makes those stores invisible.
TEST_F(RasClientMicrotest, ConnectToNccl_ResolvesConfiguredEndpoint_PassesUnspecStreamHints) {
  hostName = kOtherHost;
  port = kOtherPort;

  std::string node, service;
  int family = -1, socktype = -1;
  // Returns non-zero without touching *res, so addrInfo stays null and the walk never starts.
  ScopedHook resolve(g_getaddrinfo,
                     [&](const char* n, const char* s, const struct addrinfo* hints, struct addrinfo**) {
                       node = n ? n : "<null>";
                       service = s ? s : "<null>";
                       family = hints->ai_family;
                       socktype = hints->ai_socktype;
                       return EAI_AGAIN;
                     });

  int ret = -1;
  const std::string log = CaptureLog([&ret]() { ret = connectToNCCL(); });

  EXPECT_EQ(1, ret);
  EXPECT_EQ(1, resolve.calls);
  EXPECT_EQ(kOtherHost, node);
  EXPECT_EQ(kOtherPort, service);
  EXPECT_EQ(AF_UNSPEC, family);
  EXPECT_EQ(SOCK_STREAM, socktype);
}

// Arm: socket() == -1 -> perror("socket") + `continue`. The skipped entry must
// never be connected to and never closed -- both are what `continue` buys.
TEST_F(RasClientMicrotest, ConnectToNccl_SocketFailsOnFirstEntry_SkipsToNextAddrinfo) {
  g_addrinfoCount = 3;

  std::vector<int> connectPorts;
  int socketAttempts = 0;
  ScopedHook sockHook(g_socket, [&socketAttempts](int, int, int) -> int {
    if (socketAttempts++ == 0) {
      errno = EAFNOSUPPORT;
      return -1;
    }
    return 101;
  });
  ScopedHook connectHook(g_connect, [&](int, const struct sockaddr* sa, socklen_t) -> int {
    connectPorts.push_back(PortOf(sa));
    if (PortOf(sa) == kEntryPort1) return 0;
    errno = ECONNREFUSED;
    return -1;
  });

  int ret = -1;
  const std::string log = CaptureLog([&ret]() { ret = connectToNCCL(); });

  EXPECT_EQ(1, ret);
  EXPECT_EQ(2, sockHook.calls);
  EXPECT_EQ(101, sock);
  ASSERT_EQ(1u, connectPorts.size());
  EXPECT_EQ(kEntryPort1, connectPorts[0]);
  EXPECT_EQ(std::vector<int>({101}), g_closedFds);
  EXPECT_EQ(1, g_freeaddrinfoCalls);
  ExpectPerrors({EAFNOSUPPORT});
}

// Arm: the first entry's connect fails and the *middle* entry's succeeds ->
// `break` with sock set. Pins the per-entry socket()/connect() arguments so a
// walk that always takes the first or the last entry cannot pass.
TEST_F(RasClientMicrotest, ConnectToNccl_MiddleEntryAccepts_BreaksWithThatEntrysSocket) {
  g_addrinfoCount = 3;

  std::vector<SocketCall> socketCalls;
  std::vector<ConnectCall> connectCalls;
  ScopedHook sockHook(g_socket, [&](int fam, int type, int proto) {
    socketCalls.push_back(SocketCall{fam, type, proto});
    return 200 + static_cast<int>(socketCalls.size()) - 1;
  });
  ScopedHook connectHook(g_connect, [&](int fd, const struct sockaddr* sa, socklen_t len) -> int {
    connectCalls.push_back(ConnectCall{fd, PortOf(sa), AddrOf(sa), len});
    if (PortOf(sa) == kEntryPort1) return 0;
    errno = ECONNREFUSED;
    return -1;
  });

  int ret = -1;
  const std::string log = CaptureLog([&ret]() { ret = connectToNCCL(); });

  EXPECT_EQ(1, ret);  // the handshake EOF, not the walk, is what fails here
  EXPECT_EQ(201, sock);
  ASSERT_EQ(2u, socketCalls.size());
  for (const SocketCall& c : socketCalls) {
    EXPECT_EQ(AF_INET, c.family);
    EXPECT_EQ(SOCK_STREAM, c.socktype);
    EXPECT_EQ(IPPROTO_TCP, c.protocol);
  }
  ASSERT_EQ(2u, connectCalls.size());
  EXPECT_EQ(200, connectCalls[0].fd);
  EXPECT_EQ(kEntryPort0, connectCalls[0].port);
  EXPECT_EQ(0x7f000001u, connectCalls[0].addr);
  EXPECT_EQ(sizeof(struct sockaddr_in), connectCalls[0].addrlen);
  EXPECT_EQ(201, connectCalls[1].fd);
  EXPECT_EQ(kEntryPort1, connectCalls[1].port);
  EXPECT_EQ(0x7f000002u, connectCalls[1].addr);
  EXPECT_EQ(std::vector<int>({200, 201}), g_closedFds);
  EXPECT_EQ(1, g_freeaddrinfoCalls);
}

// Arm: every entry's connect fails -> sock == -1 after the walk, and the
// ternary's false branch (both host and port at their defaults).
TEST_F(RasClientMicrotest, ConnectToNccl_AllEntriesRefusedOnDefaultEndpoint_ReportsLocalNodeAdvice) {
  g_addrinfoCount = 3;
  g_connectResult = -1;
  g_connectErrno = ECONNREFUSED;

  FdSequence fds(300);
  ScopedHook sockHook(g_socket, [&fds](int, int, int) { return fds(); });

  int ret = -1;
  const std::string log = CaptureLog([&ret]() { ret = connectToNCCL(); });

  EXPECT_EQ(1, ret);
  EXPECT_EQ(-1, sock);
  EXPECT_EQ(3, sockHook.calls);
  EXPECT_EQ(std::vector<int>({300, 301, 302}), fds.issued);
  EXPECT_EQ(std::vector<int>({300, 301, 302}), g_closedFds);
  EXPECT_EQ(1, g_freeaddrinfoCalls);  // 2 would mean addrInfo was not nulled and fail: freed it again
}

// Arm: the ternary's true branch reached via a non-default host alone. With
// `&&` in place of `||` this falls to the localhost advice instead.
TEST_F(RasClientMicrotest, ConnectToNccl_AllEntriesRefusedOnCustomHost_ReportsHostPortAdvice) {
  hostName = kOtherHost;
  g_addrinfoCount = 3;
  g_connectResult = -1;

  FdSequence fds(310);
  ScopedHook sockHook(g_socket, [&fds](int, int, int) { return fds(); });

  int ret = -1;
  const std::string log = CaptureLog([&ret]() { ret = connectToNCCL(); });

  EXPECT_EQ(1, ret);
  EXPECT_EQ(-1, sock);
  EXPECT_EQ(std::vector<int>({310, 311, 312}), fds.issued);
  EXPECT_EQ(std::vector<int>({310, 311, 312}), g_closedFds);
}

// Arm: the ternary's true branch reached via a non-default port alone, with
// hostName left at "localhost" -- the other half of the `||`.
TEST_F(RasClientMicrotest, ConnectToNccl_AllEntriesRefusedOnCustomPort_ReportsHostPortAdvice) {
  port = kOtherPort;
  g_addrinfoCount = 3;
  g_connectResult = -1;

  FdSequence fds(320);
  ScopedHook sockHook(g_socket, [&fds](int, int, int) { return fds(); });

  int ret = -1;
  const std::string log = CaptureLog([&ret]() { ret = connectToNCCL(); });

  EXPECT_EQ(1, ret);
  EXPECT_EQ(-1, sock);
  EXPECT_STREQ("localhost", hostName);
  EXPECT_EQ(std::vector<int>({320, 321, 322}), fds.issued);
  EXPECT_EQ(std::vector<int>({320, 321, 322}), g_closedFds);
}

// Arm: `if (timeout)` false. Zero disables the timeout, so neither socket
// option is set -- but the walk still runs to completion.
TEST_F(RasClientMicrotest, ConnectToNccl_TimeoutZero_SkipsBothSocketTimeoutOptions) {
  timeout = 0;
  g_connectResult = -1;

  ScopedHook sockHook(g_socket, [](int, int, int) { return 330; });
  ScopedHook optHook(g_setsockopt, [](int, int, int, const void*, socklen_t) { return 0; });

  int ret = -1;
  const std::string log = CaptureLog([&ret]() { ret = connectToNCCL(); });

  EXPECT_EQ(1, ret);
  EXPECT_EQ(0, optHook.calls);
  EXPECT_EQ(1, sockHook.calls);
  EXPECT_EQ(std::vector<int>({330}), g_closedFds);
}

// Arm: `if (timeout)` true -> SO_SNDTIMEO then SO_RCVTIMEO, both carrying the
// 1-second TIMEOUT_INCREMENT tv, on the fd socket() just returned.
TEST_F(RasClientMicrotest, ConnectToNccl_TimeoutEnabled_SetsSendThenReceiveTimeoutToOneSecond) {
  timeout = 5;
  g_connectResult = -1;

  std::vector<SockoptCall> opts;
  ScopedHook sockHook(g_socket, [](int, int, int) { return 340; });
  ScopedHook optHook(g_setsockopt, [&](int fd, int level, int optname, const void* val, socklen_t len) {
    struct timeval tv = {-1, -1};
    if (val && len >= static_cast<socklen_t>(sizeof(tv))) memcpy(&tv, val, sizeof(tv));
    opts.push_back(SockoptCall{fd, level, optname, static_cast<long>(tv.tv_sec), static_cast<long>(tv.tv_usec)});
    return 0;
  });

  int ret = -1;
  const std::string log = CaptureLog([&ret]() { ret = connectToNCCL(); });

  EXPECT_EQ(1, ret);
  ASSERT_EQ(2u, opts.size());
  EXPECT_EQ(SO_SNDTIMEO, opts[0].optname);
  EXPECT_EQ(SO_RCVTIMEO, opts[1].optname);
  for (const SockoptCall& o : opts) {
    EXPECT_EQ(340, o.fd);
    EXPECT_EQ(SOL_SOCKET, o.level);
    EXPECT_EQ(1L, o.sec);
    EXPECT_EQ(0L, o.usec);
  }
}

// Arm: the first setsockopt fails -> `||` short-circuits, so SO_RCVTIMEO is
// never attempted, perror reports it, and the walk continues to connect.
TEST_F(RasClientMicrotest, ConnectToNccl_SendTimeoutOptionFails_SkipsReceiveOptionAndStillConnects) {
  timeout = 3;

  ScopedHook sockHook(g_socket, [](int, int, int) { return 350; });
  ScopedHook optHook(g_setsockopt, [](int, int, int, const void*, socklen_t) {
    errno = ENOPROTOOPT;
    return -1;
  });
  ScopedHook connectHook(g_connect, [](int, const struct sockaddr*, socklen_t) { return 0; });

  int ret = -1;
  const std::string log = CaptureLog([&ret]() { ret = connectToNCCL(); });

  EXPECT_EQ(1, ret);
  EXPECT_EQ(1, optHook.calls);
  EXPECT_EQ(1, connectHook.calls);
  EXPECT_EQ(350, sock);
  EXPECT_EQ(std::vector<int>({350}), g_closedFds);
  ExpectPerrors({ENOPROTOOPT});
}

// Arm: only the second setsockopt fails -> both calls happen and the single
// shared perror still fires. Non-fatal: connect is still attempted.
TEST_F(RasClientMicrotest, ConnectToNccl_ReceiveTimeoutOptionFails_ReportsAfterBothCallsAndContinues) {
  timeout = 3;

  ScopedHook sockHook(g_socket, [](int, int, int) { return 360; });
  ScopedHook optHook(g_setsockopt, [](int, int, int optname, const void*, socklen_t) -> int {
    if (optname == SO_RCVTIMEO) {
      errno = ENOPROTOOPT;
      return -1;
    }
    return 0;
  });
  ScopedHook connectHook(g_connect, [](int, const struct sockaddr*, socklen_t) {
    errno = ECONNREFUSED;
    return -1;
  });

  int ret = -1;
  const std::string log = CaptureLog([&ret]() { ret = connectToNCCL(); });

  EXPECT_EQ(1, ret);
  EXPECT_EQ(2, optHook.calls);
  EXPECT_EQ(1, connectHook.calls);
  EXPECT_EQ(std::vector<int>({360}), g_closedFds);
  ExpectPerrors({ENOPROTOOPT});
}

// Arm: the getnameinfo success path -- NI_NUMERICHOST|NI_NUMERICSERV over the
// failed entry's own sockaddr, and both output buffers reach the message.
TEST_F(RasClientMicrotest, ConnectToNccl_ConnectFails_FormatsTheNumericNameOfTheFailedAddress) {
  g_connectResult = -1;

  int flags = 0;
  socklen_t salen = 0;
  int queriedPort = 0;
  ScopedHook nameHook(g_getnameinfo, [&](const struct sockaddr* sa, socklen_t len, char* host, socklen_t hostlen,
                                         char* serv, socklen_t servlen, int f) {
    flags = f;
    salen = len;
    queriedPort = PortOf(sa);
    snprintf(host, hostlen, "%s", "resolved.host");
    snprintf(serv, servlen, "%s", "5150");
    return 0;
  });
  ScopedHook sockHook(g_socket, [](int, int, int) { return 370; });

  int ret = -1;
  const std::string log = CaptureLog([&ret]() { ret = connectToNCCL(); });

  EXPECT_EQ(1, ret);
  EXPECT_EQ(1, nameHook.calls);
  EXPECT_EQ(NI_NUMERICHOST | NI_NUMERICSERV, flags);
  EXPECT_EQ(sizeof(struct sockaddr_in), salen);
  EXPECT_EQ(kEntryPort0, queriedPort);
}

// Arm: getnameinfo fails -> the strcpy fallback puts the *configured* host and
// port in the message, discarding whatever getnameinfo left in the buffers.
TEST_F(RasClientMicrotest, ConnectToNccl_GetnameinfoFails_FallsBackToConfiguredHostAndPort) {
  hostName = kOtherHost;
  port = kOtherPort;
  g_addrinfoCount = 3;
  g_connectResult = -1;

  FdSequence fds(380);
  ScopedHook sockHook(g_socket, [&fds](int, int, int) { return fds(); });
  // Writes a marker first: if the fallback is dropped, the marker is what prints.
  ScopedHook nameHook(g_getnameinfo, [](const struct sockaddr*, socklen_t, char* host, socklen_t hostlen, char* serv,
                                        socklen_t servlen, int) {
    snprintf(host, hostlen, "%s", "STALE_HOST");
    snprintf(serv, servlen, "%s", "9999");
    return EAI_FAMILY;
  });

  int ret = -1;
  const std::string log = CaptureLog([&ret]() { ret = connectToNCCL(); });

  EXPECT_EQ(1, ret);
  EXPECT_EQ(3, nameHook.calls);
}

// Arm: freeaddrinfo receives the list *head*, not the exhausted walk cursor,
// exactly once. Overriding the free seam means this hook owns the teardown of
// what the default getaddrinfo calloc'd.
TEST_F(RasClientMicrotest, ConnectToNccl_WalkExhausted_FreesTheHeadOfTheResolvedListOnce) {
  g_addrinfoCount = 3;
  g_connectResult = -1;

  int freedHeadPort = -1;
  bool freedNull = false;
  FdSequence fds(400);
  ScopedHook sockHook(g_socket, [&fds](int, int, int) { return fds(); });
  ScopedHook freeHook(g_freeaddrinfo, [&](struct addrinfo* ai) {
    if (!ai) {
      freedNull = true;
      return;
    }
    if (freedHeadPort == -1) freedHeadPort = PortOf(ai->ai_addr);
    while (ai) {
      struct addrinfo* next = ai->ai_next;
      free(ai->ai_addr);
      free(ai);
      ai = next;
    }
  });

  int ret = -1;
  const std::string log = CaptureLog([&ret]() { ret = connectToNCCL(); });

  EXPECT_EQ(1, ret);
  EXPECT_EQ(1, freeHook.calls);
  EXPECT_FALSE(freedNull);
  EXPECT_EQ(kEntryPort0, freedHeadPort);
  EXPECT_EQ(std::vector<int>({400, 401, 402}), fds.issued);
  EXPECT_EQ(std::vector<int>({400, 401, 402}), g_closedFds);
}



// ===========================================================================
// connectToNCCL: the RAS client protocol handshake (src/ras/client.cc:220-250)
//
// Every test here drives connectToNCCL from the top with timeout == -1, so the
// resolve/connect walk succeeds on the default seams and the TIMEOUT
// negotiation at :252 is skipped. Note `if (timeout)` at :278 is still TRUE for
// -1, so one further setsockopt runs before `return 0`; that belongs to the
// TIMEOUT block and is deliberately not asserted on here.
// ===========================================================================

namespace {

struct HandshakeOutcome {
  int ret;
  std::string log;
};

// Runs the handshake on the fixture defaults: timeout is -1, so the TIMEOUT
// negotiation is skipped and connectToNCCL ends right after this block.
HandshakeOutcome RunConnectToNCCL() {
  HandshakeOutcome out{-999, {}};
  out.log = CaptureLog([&]() { out.ret = connectToNCCL(); });
  return out;
}

// The exact bytes the handshake must put on the wire. Spelled as a literal on
// purpose: rebuilding it from STR(NCCL_RAS_CLIENT_PROTOCOL) would launder any
// mutation of the string client.cc actually sends.
constexpr const char kClientHello[] = "CLIENT PROTOCOL 2\n";

// The fd DefaultSocket hands out; `fail:` and `timeout:` both close exactly this.
constexpr int kSocketFd = 42;

void ExpectClosedExactlyOnce(int fd) {
  ASSERT_EQ(1u, g_closedFds.size());
  EXPECT_EQ(fd, g_closedFds[0]);
}

}  // namespace

// The banner check is one strncasecmp over 16 bytes plus a strtol; the version compare only decides whether a warning
// is printed, never whether the handshake succeeds. So every accepted banner leaves the same state, and with the
// warning no longer asserted the five separate tests held identical expectations. One table per side.
TEST_F(RasClientMicrotest, ConnectHandshake_AcceptedBanners_ReturnZeroAndLeaveTheSocketOpen) {
  for (const char* banner : {"SERVER PROTOCOL 2\n",    // exact match, no warning
                             "server protocol 2\n",    // strncasecmp folds case
                             "SERVER PROTOCOL 7\n",    // version mismatch: warns, does not abort
                             "SERVER PROTOCOL abc\n",  // strtol converts nothing -> 0, same warning
                             "SERVER PROTOCOL 0x2\n"}) {  // base 10 stops at 'x', so also 0
    ResetLibcFakes();
    ResetRasClientGlobals();
    ScriptReadData(banner);

    const HandshakeOutcome out = RunConnectToNCCL();

    EXPECT_EQ(0, out.ret) << banner;
    EXPECT_EQ(kClientHello, g_writtenData) << banner;
    EXPECT_EQ(kSocketFd, sock) << banner;
    EXPECT_TRUE(g_closedFds.empty()) << banner;  // not the `fail:` arm
    ExpectPerrors({});
  }
}

// The rejecting side. "" is no script at all, i.e. rasRead reports EOF; the rest fail the 16-byte compare, including
// the one that differs only in byte 15, which a comparison one byte short would accept.
TEST_F(RasClientMicrotest, ConnectHandshake_RejectedBanners_ReturnOneAndCloseTheSocketOnce) {
  for (const char* banner : {"SERVER PROTOCOL\n",       // the trailing space is part of the 16
                             "SERVER PROTOCOLX2\n",     // byte 15 must be the space specifically
                             "ERROR unknown command\n",  // differs in the first byte
                             ""}) {                      // server closed before replying
    ResetLibcFakes();
    ResetRasClientGlobals();
    if (*banner != '\0') ScriptReadData(banner);

    const HandshakeOutcome out = RunConnectToNCCL();

    EXPECT_EQ(1, out.ret) << banner;
    EXPECT_EQ(kClientHello, g_writtenData) << banner;  // the write arm ran; only the reply is bad
    ExpectClosedExactlyOnce(kSocketFd);
    ExpectPerrors({});
  }
}

// Arm: the CLIENT PROTOCOL write fails with an errno that is not EAGAIN, so the
// perror/`goto fail` pair runs instead of the retry.
TEST_F(RasClientMicrotest, ConnectHandshake_WriteFailsWithEio_PerrorsAndFails) {
  ScopedHook writeHook(g_write, [](int, const void*, size_t) -> ssize_t {
    errno = EIO;
    return -1;
  });

  const HandshakeOutcome out = RunConnectToNCCL();

  EXPECT_EQ(1, out.ret);
  EXPECT_EQ(1, writeHook.calls);
  EXPECT_EQ("", g_writtenData);
  ExpectPerrors({EIO});
  ExpectClosedExactlyOnce(kSocketFd);
}

// Arm: rasRead fails with an errno that is not EAGAIN. The write arm must have
// already completed, which the recorded wire bytes pin.
TEST_F(RasClientMicrotest, ConnectHandshake_ReadFailsWithEio_PerrorsAndFails) {
  ScriptRead(-1, EIO, "");

  const HandshakeOutcome out = RunConnectToNCCL();

  EXPECT_EQ(1, out.ret);
  EXPECT_EQ(kClientHello, g_writtenData);
  ExpectPerrors({EIO});
  ExpectClosedExactlyOnce(kSocketFd);
}

// Arm: the write fails with EAGAIN, so `goto timeout` closes the socket and
// re-runs resolve/connect/handshake. The hook must succeed on the second pass --
// a seam that returns EAGAIN forever makes the retry loop non-terminating.
TEST_F(RasClientMicrotest, ConnectHandshake_WriteFailsWithEagainOnce_RetriesOnceThenSucceeds) {
  ScriptReadData("SERVER PROTOCOL 2\n");
  int writes = 0;
  ScopedHook connectHook(g_connect, [](int, const struct sockaddr*, socklen_t) { return 0; });
  ScopedHook writeHook(g_write, [&writes](int, const void* buf, size_t count) -> ssize_t {
    if (++writes == 1) {
      errno = EAGAIN;
      return -1;
    }
    if (writes > 4) {  // cap: a mutant that keeps retrying must fail out, not hang the binary
      errno = EIO;
      return -1;
    }
    g_writtenData.append(static_cast<const char*>(buf), count);
    return static_cast<ssize_t>(count);
  });

  const HandshakeOutcome out = RunConnectToNCCL();

  EXPECT_EQ(0, out.ret);
  EXPECT_EQ(2, writeHook.calls);
  EXPECT_EQ(2, connectHook.calls);         // the retry re-ran the whole resolve/connect walk
  EXPECT_EQ(kClientHello, g_writtenData);  // only the second, successful write reached the wire
  EXPECT_EQ(kSocketFd, sock);
  ExpectClosedExactlyOnce(kSocketFd);
}

// Arm: rasRead fails with EAGAIN, so `goto timeout` retries; the second pass
// reads the next scripted step. Two client hellos on the wire prove the whole
// handshake re-ran rather than resuming mid-way.
TEST_F(RasClientMicrotest, ConnectHandshake_ReadFailsWithEagainOnce_RetriesOnceThenSucceeds) {
  ScriptRead(-1, EAGAIN, "");
  ScriptReadData("SERVER PROTOCOL 2\n");
  ScopedHook connectHook(g_connect, [](int, const struct sockaddr*, socklen_t) { return 0; });

  const HandshakeOutcome out = RunConnectToNCCL();

  EXPECT_EQ(0, out.ret);
  EXPECT_EQ(2, connectHook.calls);
  EXPECT_EQ(std::string(kClientHello) + kClientHello, g_writtenData);
  EXPECT_EQ(kSocketFd, sock);
  ExpectClosedExactlyOnce(kSocketFd);
}


// ===========================================================================
// connectToNCCL: the "TIMEOUT" negotiation (timeout >= 0), the socket-timeout
// bump (if (timeout)), and the fail:/timeout: labels. Driven on the file-scope
// `timeout` static; the resolve/connect walk and the protocol handshake ahead
// of the block run on the fakes' defaults.
// ===========================================================================

namespace {

// The handshake ahead of the block consumes one scripted read, so every pass
// through connectToNCCL needs a server hello before the reply it waits on.
constexpr const char kCtServerHello[] = "SERVER PROTOCOL 2\n";
constexpr const char kCtClientHello[] = "CLIENT PROTOCOL 2\n";

// tv starts at {TIMEOUT_INCREMENT, 0} and the bump adds timeout + EXTRA(5), so
// 37 yields 43 -- a value no operand-dropping mutant of that line also reaches.
constexpr int kCtDistinctTimeout = 37;

struct CtSetsockoptCall {
  int optname;
  long tvSec;
  long tvUsec;
};

// g_lastSetsockopt* keeps only the final call, which cannot see a dropped or
// swapped call earlier in the walk; render the whole sequence as one string.
std::string CtTrace(const std::vector<CtSetsockoptCall>& calls) {
  std::string out;
  for (const CtSetsockoptCall& c : calls) {
    if (c.optname == SO_SNDTIMEO) {
      out += "SNDTIMEO";
    } else if (c.optname == SO_RCVTIMEO) {
      out += "RCVTIMEO";
    } else {
      out += std::to_string(c.optname);
    }
    out += "={" + std::to_string(c.tvSec) + "," + std::to_string(c.tvUsec) + "} ";
  }
  return out;
}

void CtRecord(std::vector<CtSetsockoptCall>* into, int optname, const void* optval, socklen_t optlen) {
  struct timeval tv = {-1, -1};
  if (optval && optlen >= static_cast<socklen_t>(sizeof tv)) {
    std::memcpy(&tv, optval, sizeof tv);
  }
  into->push_back(CtSetsockoptCall{optname, static_cast<long>(tv.tv_sec), static_cast<long>(tv.tv_usec)});
}

}  // namespace

// timeout < 0 skips the negotiation but is still truthy, so the bump runs with
// the RAS_COLLECTIVE_LEG_TIMEOUT_SEC arm: 1 + 5 + 5 == 11.
TEST_F(RasClientMicrotest, ConnectTimeout_NegativeTimeout_SkipsNegotiationAndBumpsRcvtimeoToLegPlusExtra) {
  timeout = -1;
  ScriptReadData(kCtServerHello);

  std::vector<CtSetsockoptCall> opts;
  int rc = -1;
  std::string log;
  {
    ScopedHook sso(g_setsockopt, [&](int, int, int optname, const void* optval, socklen_t optlen) {
      CtRecord(&opts, optname, optval, optlen);
      return 0;
    });
    log = CaptureLog([&]() { rc = connectToNCCL(); });
    EXPECT_EQ(3, sso.calls);
  }

  EXPECT_EQ(0, rc);
  EXPECT_EQ(std::string(kCtClientHello), g_writtenData);  // no TIMEOUT line was sent
  EXPECT_EQ("SNDTIMEO={1,0} RCVTIMEO={1,0} RCVTIMEO={11,0} ", CtTrace(opts));
  EXPECT_EQ(42, sock);
  EXPECT_TRUE(g_closedFds.empty());
  EXPECT_EQ(1, g_freeaddrinfoCalls);
}

// timeout == 0 is >= 0 so the negotiation runs, yet it is falsy, so the bump --
// and with it every setsockopt in the whole function -- is skipped.
TEST_F(RasClientMicrotest, ConnectTimeout_ZeroTimeout_NegotiatesZeroAndIssuesNoSetsockopt) {
  timeout = 0;
  ScriptReadData(kCtServerHello);
  ScriptReadData("OK\n");

  std::vector<CtSetsockoptCall> opts;
  int rc = -1;
  std::string log;
  {
    ScopedHook sso(g_setsockopt, [&](int, int, int optname, const void* optval, socklen_t optlen) {
      CtRecord(&opts, optname, optval, optlen);
      return 0;
    });
    log = CaptureLog([&]() { rc = connectToNCCL(); });
    EXPECT_EQ(0, sso.calls);
  }

  EXPECT_EQ(0, rc);
  EXPECT_EQ("CLIENT PROTOCOL 2\nTIMEOUT 0\n", g_writtenData);
  EXPECT_EQ("", CtTrace(opts));
  EXPECT_EQ(42, sock);
  EXPECT_TRUE(g_closedFds.empty());
}

// timeout > 0: the exact bytes on the wire, and 1 + 37 + 5 == 43 on SO_RCVTIMEO.
TEST_F(RasClientMicrotest, ConnectTimeout_PositiveTimeout_SendsExactTimeoutLineAndBumpsRcvtimeoBySumOfTimeoutAndExtra) {
  timeout = kCtDistinctTimeout;
  ScriptReadData(kCtServerHello);
  ScriptReadData("OK\n");

  std::vector<CtSetsockoptCall> opts;
  int rc = -1;
  std::string log;
  {
    ScopedHook sso(g_setsockopt, [&](int, int, int optname, const void* optval, socklen_t optlen) {
      CtRecord(&opts, optname, optval, optlen);
      return 0;
    });
    log = CaptureLog([&]() { rc = connectToNCCL(); });
    EXPECT_EQ(3, sso.calls);
  }

  EXPECT_EQ(0, rc);
  EXPECT_EQ("CLIENT PROTOCOL 2\nTIMEOUT 37\n", g_writtenData);
  EXPECT_EQ("SNDTIMEO={1,0} RCVTIMEO={1,0} RCVTIMEO={43,0} ", CtTrace(opts));
  EXPECT_EQ(42, sock);
  EXPECT_TRUE(g_closedFds.empty());
  EXPECT_EQ(1, g_freeaddrinfoCalls);
}

// strcasecmp, not strcmp: a lowercase reply is still accepted.
TEST_F(RasClientMicrotest, ConnectTimeout_ServerAnswersLowercaseOk_IsAcceptedCaseInsensitively) {
  timeout = kCtDistinctTimeout;
  ScriptReadData(kCtServerHello);
  ScriptReadData("ok\n");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = connectToNCCL(); });

  EXPECT_EQ(0, rc);
  EXPECT_EQ("CLIENT PROTOCOL 2\nTIMEOUT 37\n", g_writtenData);
  EXPECT_EQ(SO_RCVTIMEO, g_lastSetsockoptOptname);
  EXPECT_EQ(43, g_lastSetsockoptTimeval.tv_sec);
  EXPECT_EQ(0, g_lastSetsockoptTimeval.tv_usec);
}

// bytes < 0 with a non-EAGAIN errno: perror and fail, never the retry label.
TEST_F(RasClientMicrotest, ConnectTimeout_ReplyReadFailsWithConnectionReset_ReportsPerrorAndReturnsOne) {
  timeout = kCtDistinctTimeout;
  ScriptReadData(kCtServerHello);
  ScriptRead(-1, ECONNRESET, "");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = connectToNCCL(); });

  EXPECT_EQ(1, rc);
  EXPECT_EQ("CLIENT PROTOCOL 2\nTIMEOUT 37\n", g_writtenData);
  ExpectPerrors({ECONNRESET});
  ASSERT_EQ(1u, g_closedFds.size());
  EXPECT_EQ(42, g_closedFds[0]);
}

// The TIMEOUT write failing with a non-EAGAIN errno: perror and fail.
TEST_F(RasClientMicrotest, ConnectTimeout_RequestWriteFailsWithBrokenPipe_ReportsPerrorAndReturnsOne) {
  timeout = kCtDistinctTimeout;
  ScriptReadData(kCtServerHello);

  int rc = -1;
  std::string log;
  {
    // Content-keyed so only the block's own write fails; the handshake's must not.
    ScopedHook wr(g_write, [](int, const void* buf, size_t count) -> ssize_t {
      const std::string chunk(static_cast<const char*>(buf), count);
      if (chunk.compare(0, 8, "TIMEOUT ") == 0) {
        errno = EPIPE;
        return -1;
      }
      g_writtenData.append(chunk);
      return static_cast<ssize_t>(count);
    });
    log = CaptureLog([&]() { rc = connectToNCCL(); });
    EXPECT_EQ(2, wr.calls);
  }

  EXPECT_EQ(1, rc);
  EXPECT_EQ(std::string(kCtClientHello), g_writtenData);
  ExpectPerrors({EPIPE});
  ASSERT_EQ(1u, g_closedFds.size());
  EXPECT_EQ(42, g_closedFds[0]);
}

// goto timeout from the reply read: close, retry the whole walk once, succeed.
// The script's tail is EOF, so a mutant that keeps retrying still terminates.
TEST_F(RasClientMicrotest, ConnectTimeout_ReplyReadFailsWithEagain_RetriesOnceThenSucceeds) {
  timeout = kCtDistinctTimeout;
  ScriptReadData(kCtServerHello);
  ScriptRead(-1, EAGAIN, "");
  ScriptReadData(kCtServerHello);
  ScriptReadData("OK\n");

  std::vector<CtSetsockoptCall> opts;
  int rc = -1;
  std::string log;
  {
    ScopedHook sso(g_setsockopt, [&](int, int, int optname, const void* optval, socklen_t optlen) {
      CtRecord(&opts, optname, optval, optlen);
      return 0;
    });
    log = CaptureLog([&]() { rc = connectToNCCL(); });
  }

  EXPECT_EQ(0, rc);
  EXPECT_EQ("CLIENT PROTOCOL 2\nTIMEOUT 37\nCLIENT PROTOCOL 2\nTIMEOUT 37\n", g_writtenData);
  EXPECT_EQ("SNDTIMEO={1,0} RCVTIMEO={1,0} SNDTIMEO={1,0} RCVTIMEO={1,0} RCVTIMEO={43,0} ", CtTrace(opts));
  ASSERT_EQ(1u, g_closedFds.size());
  EXPECT_EQ(42, g_closedFds[0]);
  EXPECT_EQ(2, g_freeaddrinfoCalls);
  EXPECT_EQ(42, sock);
}

// goto timeout from the TIMEOUT write. The hook fails exactly one write ever,
// which is what keeps the retry loop from spinning forever.
TEST_F(RasClientMicrotest, ConnectTimeout_RequestWriteFailsWithEagain_RetriesOnceThenSucceeds) {
  timeout = kCtDistinctTimeout;
  ScriptReadData(kCtServerHello);
  ScriptReadData(kCtServerHello);
  ScriptReadData("OK\n");

  bool stalledOnce = false;
  int rc = -1;
  std::string log;
  {
    ScopedHook wr(g_write, [&](int, const void* buf, size_t count) -> ssize_t {
      const std::string chunk(static_cast<const char*>(buf), count);
      if (!stalledOnce && chunk.compare(0, 8, "TIMEOUT ") == 0) {
        stalledOnce = true;
        errno = EAGAIN;
        return -1;
      }
      g_writtenData.append(chunk);
      return static_cast<ssize_t>(count);
    });
    log = CaptureLog([&]() { rc = connectToNCCL(); });
    EXPECT_EQ(4, wr.calls);
  }

  EXPECT_EQ(0, rc);
  EXPECT_EQ("CLIENT PROTOCOL 2\nCLIENT PROTOCOL 2\nTIMEOUT 37\n", g_writtenData);
  ASSERT_EQ(1u, g_closedFds.size());
  EXPECT_EQ(42, g_closedFds[0]);
  EXPECT_EQ(SO_RCVTIMEO, g_lastSetsockoptOptname);
  EXPECT_EQ(43, g_lastSetsockoptTimeval.tv_sec);
  EXPECT_EQ(0, g_lastSetsockoptTimeval.tv_usec);
}

// The bump's setsockopt failure is non-fatal: it is reported and 0 is returned.
TEST_F(RasClientMicrotest, ConnectTimeout_BumpSetsockoptFails_ReportsPerrorAndStillReturnsZero) {
  timeout = kCtDistinctTimeout;
  ScriptReadData(kCtServerHello);
  ScriptReadData("OK\n");

  std::vector<CtSetsockoptCall> opts;
  int rc = -1;
  std::string log;
  {
    // Keyed on the bumped tv so the connect walk's two 1-second calls still pass.
    ScopedHook sso(g_setsockopt, [&](int, int, int optname, const void* optval, socklen_t optlen) -> int {
      CtRecord(&opts, optname, optval, optlen);
      if (opts.back().tvSec != TIMEOUT_INCREMENT) {
        errno = ENOPROTOOPT;
        return -1;
      }
      return 0;
    });
    log = CaptureLog([&]() { rc = connectToNCCL(); });
    EXPECT_EQ(3, sso.calls);
  }

  EXPECT_EQ(0, rc);
  ExpectPerrors({ENOPROTOOPT});  // exactly one, so the retry did not report the same failure twice
  EXPECT_EQ("SNDTIMEO={1,0} RCVTIMEO={1,0} RCVTIMEO={43,0} ", CtTrace(opts));
  EXPECT_EQ(42, sock);
  EXPECT_TRUE(g_closedFds.empty());
}

// fail: with sock still -1 -- the guard is what keeps close(-1) from happening.
TEST_F(RasClientMicrotest, ConnectFailLabel_SockNeverOpened_ReturnsOneAndClosesNothing) {
  g_addrinfoCount = 0;  // no candidate address, so the connect walk never runs

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = connectToNCCL(); });

  EXPECT_EQ(1, rc);
  EXPECT_TRUE(g_closedFds.empty());
  EXPECT_EQ(-1, sock);
  EXPECT_EQ(1, g_freeaddrinfoCalls);  // the walk's own free; fail: adds no second
}


// ===========================================================================
// monitorNCCLEvents: command + activation, the leftover-data flush, and the
// continuous monitor loop.
//
// Two structural findings pinned by the tests below:
//   * the loop's `errno == EINTR` arm is unreachable -- rasRead retries EINTR
//     itself and only ever returns -1 with errno != EINTR;
//   * `okEnd` can never be null on the accepted path -- activation requires
//     msgBuf[2] == '\n', so strchr always finds it at index 2.
// ===========================================================================

namespace {

constexpr int kMonitorSock = 91;
constexpr char kMonitorGroups[] = "lifecycle,trace";
constexpr char kMonitorBanner[] = "RAS Monitor Mode - watching for peer changes (Ctrl+C to exit)...\n";
constexpr char kMonitorRule[] = "================================================================\n";

// One (size, nmemb) pair as client.cc handed it to fwrite. Recording both is
// what makes the fwrite(buf,1,n) vs fwrite(buf,n,1) swap visible.
struct MonFwriteCall {
  size_t size;
  size_t nmemb;
};

}  // namespace

// --- stage 1: the command on the wire ---------------------------------------

TEST_F(RasClientMicrotest, MonitorEvents_NoEventGroups_SendsBareMonitorCommandAndReturnsZero) {
  sock = kMonitorSock;
  ScriptReadData("OK\n");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(0, rc);
  EXPECT_EQ("MONITOR\n", g_writtenData);
  // The descriptor, not just the bytes: monitorNCCLEvents must use `sock`, and the payload alone cannot tell.
  EXPECT_EQ(std::vector<int>({kMonitorSock}), g_writtenFds);
  EXPECT_EQ(std::vector<int>({kMonitorSock, kMonitorSock}), g_readFds);
  EXPECT_EQ("", g_stdoutData);
}

TEST_F(RasClientMicrotest, MonitorEvents_EventGroupsRequested_SendsThemInTheMonitorCommand) {
  sock = kMonitorSock;
  events = kMonitorGroups;
  ScriptReadData("OK\n");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(0, rc);
  EXPECT_EQ("MONITOR lifecycle,trace\n", g_writtenData);
}

TEST_F(RasClientMicrotest, MonitorEvents_WriteFailsWithEagain_ReportsConnectionTimedOutAndReturnsOne) {
  sock = kMonitorSock;

  int rc = -1;
  ScopedHook writeHook(g_write, [](int, const void*, size_t) -> ssize_t {
    errno = EAGAIN;
    return -1;
  });
  ScopedHook readHook(g_read, [](int, void*, size_t) { return static_cast<ssize_t>(0); });

  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(1, rc);
  ExpectPerrors({});
  EXPECT_EQ(1, writeHook.calls);
  EXPECT_EQ(0, readHook.calls);
}

TEST_F(RasClientMicrotest, MonitorEvents_WriteFailsWithBrokenPipe_ReportsSendFailureAndReturnsOne) {
  sock = kMonitorSock;

  int rc = -1;
  ScopedHook writeHook(g_write, [](int, const void*, size_t) -> ssize_t {
    errno = EPIPE;
    return -1;
  });
  ScopedHook readHook(g_read, [](int, void*, size_t) { return static_cast<ssize_t>(0); });

  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(1, rc);
  ExpectPerrors({EPIPE});
  EXPECT_EQ(0, readHook.calls);
}

// --- stage 1: the activation response ---------------------------------------

TEST_F(RasClientMicrotest, MonitorEvents_ActivationReadFailsWithEagain_ReportsConnectionTimedOutAndReturnsOne) {
  sock = kMonitorSock;
  ScriptRead(-1, EAGAIN, "");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(1, rc);
  EXPECT_EQ("MONITOR\n", g_writtenData);
  ExpectPerrors({});
}

TEST_F(RasClientMicrotest, MonitorEvents_ActivationReadFailsWithConnectionReset_ReportsPerrorAndReturnsOne) {
  sock = kMonitorSock;
  ScriptRead(-1, ECONNRESET, "");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(1, rc);
  EXPECT_EQ("MONITOR\n", g_writtenData);
  ExpectPerrors({ECONNRESET});
}

// strncasecmp(...,3) only inspects the OK line, unlike setOutputFormat's
// whole-string strcasecmp, which rejects the very same response.
TEST_F(RasClientMicrotest, MonitorEvents_ActivationResponseHasTrailingText_IsAcceptedUnlikeSetOutputFormat) {
  sock = kMonitorSock;
  ScriptReadData("OK\nmore");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(0, rc);
  EXPECT_EQ("more", g_stdoutData);
  // g_stdoutData alone cannot tell stdout from stderr, so pin the stream the unit actually chose.
  EXPECT_EQ(stdout, g_lastFwriteStream);
}

// The TIMEOUT reply is checked with one whole-string strcasecmp against "OK\n", so every rejecting reply lands in the
// same `goto fail`. The handshake copy of the command went out either way, which is what separates a rejected reply
// from a failed write.
TEST_F(RasClientMicrotest, ConnectTimeout_RejectedReplies_CloseTheSocketOnceAndReturnOne) {
  for (const char* reply : {"OKAY\n",  // shares its first two bytes with "OK\n"
                            "OK",      // the expected reply carries a trailing newline
                            ""}) {     // script exhausted after the handshake: EOF
    ResetLibcFakes();
    ResetRasClientGlobals();
    timeout = kCtDistinctTimeout;
    ScriptReadData(kCtServerHello);
    if (*reply != '\0') ScriptReadData(reply);

    int rc = -1;
    const std::string log = CaptureLog([&]() { rc = connectToNCCL(); });

    EXPECT_EQ(1, rc) << reply;
    EXPECT_EQ("CLIENT PROTOCOL 2\nTIMEOUT 37\n", g_writtenData) << reply;
    ASSERT_EQ(1u, g_closedFds.size()) << reply;
    EXPECT_EQ(42, g_closedFds[0]) << reply;
    EXPECT_EQ(1, g_freeaddrinfoCalls) << reply;  // fail: sees addrInfo already nulled
    ExpectPerrors({});
  }
}

// Activation is a length guard plus strncasecmp(msgBuf, "OK\n", 3), so every rejecting reply returns 1 before the
// receive timeout is touched -- which is what g_lastSetsockoptOptname still being -1 shows.
TEST_F(RasClientMicrotest, MonitorEvents_RejectedActivationReplies_ReturnOneBeforeTouchingTheTimeout) {
  for (const char* reply : {"",        // empty script: the default read reports EOF at once
                            "OK",      // bytes < 3, so the length guard fires before the compare
                            "OKAY\n"}) {  // three bytes that do not match
    ResetLibcFakes();
    ResetRasClientGlobals();
    sock = kMonitorSock;
    if (*reply != '\0') ScriptReadData(reply);

    int rc = -1;
    const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

    EXPECT_EQ(1, rc) << reply;
    EXPECT_EQ(-1, g_lastSetsockoptOptname) << reply;
    EXPECT_EQ("", g_stdoutData) << reply;
    ExpectPerrors({});
  }
}

// The accepting side, and the one place monitorNCCLEvents deliberately differs from setOutputFormat: strncasecmp
// looks at three bytes only, so trailing text after the OK line is accepted here and flushed as the first payload,
// where setOutputFormat's whole-string compare rejects the very same reply.
TEST_F(RasClientMicrotest, MonitorEvents_AcceptedActivationReplies_ReturnZeroAndFlushAnyLeftover) {
  const struct { const char* reply; const char* leftover; } cases[] = {
      {"OK\n", ""},
      {"ok\n", ""},       // strncasecmp folds case
      {"OK\nmore", "more"},  // trailing text is payload, not a mismatch
  };
  for (const auto& c : cases) {
    ResetLibcFakes();
    ResetRasClientGlobals();
    sock = kMonitorSock;
    ScriptReadData(c.reply);

    int rc = -1;
    const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

    EXPECT_EQ(0, rc) << c.reply;
    EXPECT_EQ(c.leftover, g_stdoutData) << c.reply;
    if (*c.leftover != '\0') {
      // g_stdoutData alone cannot tell stdout from stderr, so pin the stream the unit actually chose.
      EXPECT_EQ(stdout, g_lastFwriteStream) << c.reply;
    }
  }
}

// --- stage 1: disabling the receive timeout ---------------------------------

TEST_F(RasClientMicrotest, MonitorEvents_ActivationSucceeds_DisablesTheReceiveTimeoutAndPrintsBothBanners) {
  sock = kMonitorSock;
  ScriptReadData("OK\n");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(0, rc);
  EXPECT_EQ(SO_RCVTIMEO, g_lastSetsockoptOptname);
  EXPECT_EQ(0, g_lastSetsockoptTimeval.tv_sec);
  EXPECT_EQ(0, g_lastSetsockoptTimeval.tv_usec);
}

// Unlike connectToNCCL's setsockopt, this one is fatal: no banners, no loop.
TEST_F(RasClientMicrotest, MonitorEvents_SetsockoptFails_ReportsPerrorAndReturnsOneBeforeTheBanners) {
  sock = kMonitorSock;
  ScriptReadData("OK\n");

  int rc = -1;
  auto baseRead = g_read;
  ScopedHook readHook(g_read, [baseRead](int fd, void* buf, size_t n) { return baseRead(fd, buf, n); });
  ScopedHook sockoptHook(g_setsockopt, [](int, int, int, const void*, socklen_t) {
    errno = ENOPROTOOPT;
    return -1;
  });

  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(1, rc);
  ExpectPerrors({ENOPROTOOPT});
  EXPECT_EQ(1, sockoptHook.calls);
  EXPECT_EQ(1, readHook.calls);  // the loop was never entered
  EXPECT_EQ("", g_stdoutData);
}

// --- stage 2: the leftover-data flush ---------------------------------------

// Boundary: for exactly "OK\n" (bytes == 3) okEnd is the last byte, so the
// flush must not run at all -- a zero-length fwrite would still be observable.
TEST_F(RasClientMicrotest, MonitorEvents_ActivationResponseIsExactlyTheOkLine_SkipsTheLeftoverFlushEntirely) {
  sock = kMonitorSock;
  ScriptReadData("OK\n");

  int rc = -1;
  ScopedHook fwriteHook(g_fwrite, [](const void*, size_t, size_t nmemb, FILE*) { return nmemb; });
  ScopedHook fflushHook(g_fflush, [](FILE*) { return 0; });

  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(0, rc);
  EXPECT_EQ(0, fwriteHook.calls);
  EXPECT_EQ(0, fflushHook.calls);
}

// okLen (3) differs from remainingBytes (7), so an off-by-one in either is visible.
TEST_F(RasClientMicrotest, MonitorEvents_LeftoverAfterOkLine_FlushesOnlyTheRemainder) {
  sock = kMonitorSock;
  ScriptReadData("OK\nabcdef\n");

  int rc = -1;
  std::vector<MonFwriteCall> seen;
  ScopedHook fwriteHook(g_fwrite, [&seen](const void* p, size_t size, size_t nmemb, FILE*) {
    seen.push_back(MonFwriteCall{size, nmemb});
    g_stdoutData.append(static_cast<const char*>(p), size * nmemb);
    return nmemb;
  });

  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(0, rc);
  EXPECT_EQ("abcdef\n", g_stdoutData);
  ASSERT_EQ(1u, seen.size());
  EXPECT_EQ(1u, seen[0].size);
  EXPECT_EQ(7u, seen[0].nmemb);
}

TEST_F(RasClientMicrotest, MonitorEvents_LeftoverIsASingleByte_StillFlushesThatByte) {
  sock = kMonitorSock;
  ScriptReadData("OK\nX");

  int rc = -1;
  std::vector<MonFwriteCall> seen;
  ScopedHook fwriteHook(g_fwrite, [&seen](const void* p, size_t size, size_t nmemb, FILE*) {
    seen.push_back(MonFwriteCall{size, nmemb});
    g_stdoutData.append(static_cast<const char*>(p), size * nmemb);
    return nmemb;
  });

  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(0, rc);
  EXPECT_EQ("X", g_stdoutData);
  ASSERT_EQ(1u, seen.size());
  EXPECT_EQ(1u, seen[0].size);
  EXPECT_EQ(1u, seen[0].nmemb);
}

// The flush is byte-counted, not string-based: an embedded NUL is forwarded.
TEST_F(RasClientMicrotest, MonitorEvents_LeftoverContainsEmbeddedNul_FlushesEveryRawByte) {
  sock = kMonitorSock;
  ScriptReadData(std::string("OK\nA\0B\n", 7));

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(0, rc);
  EXPECT_EQ(std::string("A\0B\n", 4), g_stdoutData);
}

// Leftover-site fwrite failure. readHook.calls == 1 is what separates this from
// the identically-worded failure inside the monitor loop.
TEST_F(RasClientMicrotest, MonitorEvents_LeftoverFwriteShort_ReportsFwriteFailureBeforeEnteringTheLoop) {
  sock = kMonitorSock;
  ScriptReadData("OK\nabcdef\n");

  int rc = -1;
  auto baseRead = g_read;
  ScopedHook readHook(g_read, [baseRead](int fd, void* buf, size_t n) { return baseRead(fd, buf, n); });
  ScopedHook fwriteHook(g_fwrite, [](const void*, size_t, size_t nmemb, FILE*) { return nmemb - 1; });
  ScopedHook fflushHook(g_fflush, [](FILE*) { return 0; });

  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(1, rc);
  EXPECT_EQ(1, readHook.calls);
  EXPECT_EQ(1, fwriteHook.calls);
  EXPECT_EQ(0, fflushHook.calls);
  EXPECT_EQ("", g_stdoutData);
}

TEST_F(RasClientMicrotest, MonitorEvents_LeftoverFflushFails_ReportsPerrorBeforeEnteringTheLoop) {
  sock = kMonitorSock;
  ScriptReadData("OK\nabcdef\n");

  int rc = -1;
  auto baseRead = g_read;
  ScopedHook readHook(g_read, [baseRead](int fd, void* buf, size_t n) { return baseRead(fd, buf, n); });
  ScopedHook fflushHook(g_fflush, [](FILE*) {
    errno = EIO;
    return -1;
  });

  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(1, rc);
  ExpectPerrors({EIO});
  EXPECT_EQ("abcdef\n", g_stdoutData);
  EXPECT_EQ(1, readHook.calls);
  EXPECT_EQ(1, fflushHook.calls);
}

// --- stage 3: the monitor loop ----------------------------------------------

// Three chunks of distinct, unequal length: a mutant that drops, reorders or
// double-counts one cannot produce this exact concatenation.
TEST_F(RasClientMicrotest, MonitorEvents_ServerSendsSeveralChunks_ForwardsThemInOrderThenReportsClose) {
  sock = kMonitorSock;
  ScriptReadData("OK\n");
  ScriptReadData("alpha\n");
  ScriptReadData("be\n");
  ScriptReadData("gamma-delta\n");

  int rc = -1;
  std::vector<MonFwriteCall> seen;
  ScopedHook fwriteHook(g_fwrite, [&seen](const void* p, size_t size, size_t nmemb, FILE*) {
    seen.push_back(MonFwriteCall{size, nmemb});
    g_stdoutData.append(static_cast<const char*>(p), size * nmemb);
    return nmemb;
  });
  ScopedHook fflushHook(g_fflush, [](FILE*) { return 0; });

  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(0, rc);
  EXPECT_EQ("alpha\nbe\ngamma-delta\n", g_stdoutData);
  ASSERT_EQ(3u, seen.size());
  EXPECT_EQ(1u, seen[0].size);
  EXPECT_EQ(6u, seen[0].nmemb);
  EXPECT_EQ(1u, seen[1].size);
  EXPECT_EQ(3u, seen[1].nmemb);
  EXPECT_EQ(1u, seen[2].size);
  EXPECT_EQ(12u, seen[2].nmemb);
  EXPECT_EQ(3, fflushHook.calls);
}

// rasRead swallows and retries EINTR itself, so the loop's `errno == EINTR`
// arm is unreachable: an interrupted read surfaces as ordinary data.
TEST_F(RasClientMicrotest, MonitorEvents_LoopReadInterrupted_IsRetriedInsideRasReadNotReportedAsStopped) {
  sock = kMonitorSock;
  ScriptReadData("OK\n");
  ScriptRead(-1, EINTR, "");
  ScriptReadData("beta\n");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(0, rc);
  EXPECT_EQ("beta\n", g_stdoutData);
}

TEST_F(RasClientMicrotest, MonitorEvents_LoopReadFailsWithConnectionReset_ReportsPerrorAndReturnsOne) {
  sock = kMonitorSock;
  ScriptReadData("OK\n");
  ScriptReadData("alpha\n");
  ScriptRead(-1, ECONNRESET, "");

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(1, rc);
  EXPECT_EQ("alpha\n", g_stdoutData);
  ExpectPerrors({ECONNRESET});
}

// Loop-site fwrite failure: readHook.calls == 3 and the already-forwarded first
// chunk are what tell this apart from the leftover-flush site.
TEST_F(RasClientMicrotest, MonitorEvents_LoopFwriteShort_ReportsFwriteFailureAfterTheEarlierChunks) {
  sock = kMonitorSock;
  ScriptReadData("OK\n");
  ScriptReadData("alpha\n");
  ScriptReadData("bb\n");

  int rc = -1;
  auto baseRead = g_read;
  auto baseFwrite = g_fwrite;
  std::vector<MonFwriteCall> seen;
  ScopedHook readHook(g_read, [baseRead](int fd, void* buf, size_t n) { return baseRead(fd, buf, n); });
  ScopedHook fwriteHook(g_fwrite, [&seen, baseFwrite](const void* p, size_t size, size_t nmemb, FILE* f) -> size_t {
    seen.push_back(MonFwriteCall{size, nmemb});
    if (seen.size() == 2) return nmemb - 1;  // the second chunk is the one that fails
    return baseFwrite(p, size, nmemb, f);
  });

  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(1, rc);
  EXPECT_EQ("alpha\n", g_stdoutData);
  EXPECT_EQ(2, fwriteHook.calls);
  EXPECT_EQ(3, readHook.calls);
}

TEST_F(RasClientMicrotest, MonitorEvents_LoopFflushFails_ReportsPerrorAndReturnsOne) {
  sock = kMonitorSock;
  ScriptReadData("OK\n");
  ScriptReadData("data1\n");

  int rc = -1;
  auto baseRead = g_read;
  ScopedHook readHook(g_read, [baseRead](int fd, void* buf, size_t n) { return baseRead(fd, buf, n); });
  ScopedHook fflushHook(g_fflush, [](FILE*) {
    errno = ENOSPC;
    return -1;
  });

  const std::string log = CaptureLog([&]() { rc = monitorNCCLEvents(); });

  EXPECT_EQ(1, rc);
  ExpectPerrors({ENOSPC});
  EXPECT_EQ("data1\n", g_stdoutData);
  EXPECT_EQ(stdout, g_lastFwriteStream);
  EXPECT_EQ(2, readHook.calls);
  EXPECT_EQ(1, fflushHook.calls);
}


// ===========================================================================
// main (renamed rasClientMain here): the only place the file's five helpers are
// composed. Each helper is tested on its own above, so these tests assert only
// main's own bookkeeping -- which worker ran, what it returned, and the exact
// close(sock) sequence per arm.
// ===========================================================================

namespace {

// Not the fakes' default 42 and not a std stream fd, so a g_closedFds match
// cannot be a coincidence.
constexpr int kMainSockFd = 57;

// The handshake connectToNCCL always sends before any worker command.
const std::string kMainClientHello = "CLIENT PROTOCOL " STR(NCCL_RAS_CLIENT_PROTOCOL) "\n";

// Drives connectToNCCL to success on the first addrinfo entry, handing back
// kMainSockFd. `timeout` stays -1 so the optional TIMEOUT exchange is skipped.
void MainArmSuccessfulConnect() {
  g_nextSocketFd = kMainSockFd;
  ScriptReadData("SERVER PROTOCOL " STR(NCCL_RAS_CLIENT_PROTOCOL) "\n");
}

// main reaches parseArgs, whose exiting arms throw MicroExit. The catch has to sit inside the CaptureLog body at every
// call site: gtest has a single stderr capture slot and an exception crossing it leaves that slot armed for the whole
// binary. No test below expects main to exit, so record the escape as a failure here rather than letting it propagate.
int CallRasClientMain(RasArgv& args) {
  try {
    return rasClientMain(args.argc(), args.argv());
  } catch (const MicroExit& e) {
    ADD_FAILURE() << "rasClientMain exited with status " << e.status;
    return e.status;
  }
}

}  // namespace

// connectToNCCL fails before any socket exists: nothing to close, no worker.
TEST_F(RasClientMicrotest, RasClientMain_ConnectFailsBeforeAnySocket_ReturnsOneAndClosesNothing) {
  g_getaddrinfoResult = EAI_NONAME;
  RasArgv args{"rccl-ras-client"};

  int rc = -1;
  std::string log;
  {
    ScopedHook socketHook(g_socket, [](int, int, int) { return kMainSockFd; });
    log = CaptureLog([&]() { rc = CallRasClientMain(args); });
    EXPECT_EQ(0, socketHook.calls);
  }

  EXPECT_EQ(1, rc);
  EXPECT_TRUE(g_closedFds.empty()) << g_closedFds.size();
  EXPECT_TRUE(g_writtenData.empty()) << g_writtenData;
  EXPECT_EQ(-1, sock);
}

// connectToNCCL fails with the socket already open: its own fail: label closes
// it, main adds no second close, and `sock` is left holding the closed fd.
TEST_F(RasClientMicrotest, RasClientMain_ConnectFailsAfterSocketOpened_ClosesOnceAndLeavesSockStale) {
  g_nextSocketFd = kMainSockFd;
  ScriptReadData("HELLO SAILOR\n");
  RasArgv args{"rccl-ras-client"};

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = CallRasClientMain(args); });

  EXPECT_EQ(1, rc);
  EXPECT_EQ(std::vector<int>{kMainSockFd}, g_closedFds);
  EXPECT_EQ(kMainClientHello, g_writtenData);
  EXPECT_TRUE(g_stdoutData.empty()) << g_stdoutData;
  EXPECT_EQ(kMainSockFd, sock);
}

// setOutputFormat failing: main owns the close, and no worker command follows.
TEST_F(RasClientMicrotest, RasClientMain_SetOutputFormatFails_ClosesSockOnceAndReturnsOne) {
  MainArmSuccessfulConnect();
  format = "json";
  ScriptReadData("NOPE\n");
  RasArgv args{"rccl-ras-client"};

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = CallRasClientMain(args); });

  EXPECT_EQ(1, rc);
  EXPECT_EQ(kMainClientHello + "SET FORMAT json\n", g_writtenData);
  EXPECT_EQ(std::vector<int>{kMainSockFd}, g_closedFds);
  EXPECT_TRUE(g_stdoutData.empty()) << g_stdoutData;
}

// monitorMode clear selects getNCCLStatus: only it sends "STATUS\n" and only it
// streams a non-"OK\n" first reply straight to stdout.
TEST_F(RasClientMicrotest, RasClientMain_MonitorModeClear_RunsGetNcclStatusAndReturnsZero) {
  MainArmSuccessfulConnect();
  ScriptReadData("peer 0 ok\n");
  RasArgv args{"rccl-ras-client"};

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = CallRasClientMain(args); });

  EXPECT_EQ(0, rc);
  EXPECT_EQ(kMainClientHello + "STATUS\n", g_writtenData);
  EXPECT_EQ("peer 0 ok\n", g_stdoutData);
  EXPECT_EQ(std::vector<int>{kMainSockFd}, g_closedFds);
}

// monitorMode set selects monitorNCCLEvents: only it sends "MONITOR\n", only it
// prints the monitor banner, and it consumes the "OK\n" ack rather than echoing
// it. Driven through argv, so deleting the parseArgs call also fails this test.
TEST_F(RasClientMicrotest, RasClientMain_MonitorFlagInArgv_RunsMonitorNcclEventsAndReturnsZero) {
  MainArmSuccessfulConnect();
  ScriptReadData("OK\n");
  ScriptReadData("peer 3 left\n");
  RasArgv args{"rccl-ras-client", "--monitor"};

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = CallRasClientMain(args); });

  EXPECT_EQ(0, rc);
  EXPECT_TRUE(monitorMode);
  EXPECT_EQ(kMainClientHello + "MONITOR\n", g_writtenData);
  EXPECT_EQ("peer 3 left\n", g_stdoutData);
  EXPECT_EQ(std::vector<int>{kMainSockFd}, g_closedFds);
}

// Worker returning non-zero: main closes exactly once and reports 1.
TEST_F(RasClientMicrotest, RasClientMain_WorkerReturnsNonZero_ClosesSockOnceAndReturnsOne) {
  MainArmSuccessfulConnect();
  ScriptRead(-1, ECONNRESET, "");
  RasArgv args{"rccl-ras-client"};

  int rc = -1;
  const std::string log = CaptureLog([&]() { rc = CallRasClientMain(args); });

  EXPECT_EQ(1, rc);
  EXPECT_EQ(kMainClientHello + "STATUS\n", g_writtenData);
  ExpectPerrors({ECONNRESET});
  EXPECT_EQ(std::vector<int>{kMainSockFd}, g_closedFds);
  EXPECT_TRUE(g_stdoutData.empty()) << g_stdoutData;
}

// The final close failing turns an otherwise fully successful run into a 1.
TEST_F(RasClientMicrotest, RasClientMain_FinalCloseFails_ReportsPerrorAndReturnsOne) {
  MainArmSuccessfulConnect();
  ScriptReadData("peer 0 ok\n");
  RasArgv args{"rccl-ras-client"};

  std::vector<int> closed;
  int rc = -1;
  std::string log;
  {
    ScopedHook closeHook(g_close, [&](int fd) {
      closed.push_back(fd);
      errno = EIO;
      return -1;
    });
    log = CaptureLog([&]() { rc = CallRasClientMain(args); });
    EXPECT_EQ(1, closeHook.calls);
  }

  EXPECT_EQ(1, rc);
  EXPECT_EQ(std::vector<int>{kMainSockFd}, closed);
  ExpectPerrors({EIO});
  EXPECT_EQ("peer 0 ok\n", g_stdoutData);
}



// ===========================================================================
// parseArgs: the -p / --port arm's own missing-argument oracle.
//
// The port arm's arity lives in two places -- longOpts' required_argument and
// the optstring's "p:" -- and relaxing either one turns a missing argument into
// a case 'p' with a NULL optarg instead of the default: exit. The pair below
// pins that inside the port block, so the arm keeps a guard of its own rather
// than depending on a test in the default: block continuing to exist.
// ===========================================================================

// The optstring's "p:" carries the short spelling's arity; "p" alone would make
// a trailing -p a valid no-argument option and skip the default: exit entirely.
TEST_F(RasClientMicrotest, ParseArgsPort_ShortFormMissingArgument_ExitsOneWithUsage) {
  const ParseArgsOutcome out = RunParseArgv({kUsageProg, "-v", "-p"});

  EXPECT_EQ(1, out.exitStatus);
  ASSERT_NE(nullptr, port);
  EXPECT_STREQ(kDefaultPort, port);
  // The preceding -v was applied, so the exit above aborted a running loop.
  EXPECT_TRUE(verbose);
}
