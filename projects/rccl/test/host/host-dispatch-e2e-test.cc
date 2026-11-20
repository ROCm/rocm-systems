/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// End-to-end tests for rccl-UnitTestsHostAll: the real binary, run as a caller runs it, over a
// tree of fixture binaries built for the purpose (host_test_dispatch_fixture.cc).
//
// Why these exist at all. host-test-dispatch-test.cc reaches the dispatcher's decisions one
// function at a time, and the scheduler through a fake, which covers the loop and nothing on
// either side of it: the listing phase, the filter resolution gtest does inside the process, the
// environment each child is handed, the discovery of the binaries and the exit code are all
// decided in main(), where no unit test reaches. Every one of them can be broken without a
// single microtest going red -- forcing also_run_disabled_tests to false, dropping the GTEST_*
// strip or the per-child LLVM_PROFILE_FILE all leave that file's tests green. These tests are
// what notices.
//
// Each one builds its own tree: a copy of the dispatcher, a micro/ directory beside it holding
// copies of the fixture binary under the names the manifest gives, and the manifest. The
// dispatcher finds its binaries relative to its own path, so the copy is what makes the tree
// the test's rather than the build's.

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include "ScopedDir.h"

namespace {

namespace fs = std::filesystem;

// The qualifiers the dispatcher derives from the fixture binary names below. Spelled out so a
// change to qualifierFor() that these tests disagree with shows up here as a failure rather than
// as a silently renamed suite in everyone's report.
constexpr const char *kFirstBinary = "rccl-UnitTestsMicroFixtureOne";
constexpr const char *kSecondBinary = "rccl-UnitTestsMicroFixtureTwo";
constexpr const char *kFirstQualifier = "MicroFixtureOne";
constexpr const char *kSecondQualifier = "MicroFixtureTwo";

// What one run of the dispatcher produced.
struct DispatchRun {
    int status = -1;             // exit status, or -1 when it did not exit normally
    std::string console;         // its stdout and stderr, interleaved as a caller sees them
    std::string report;          // the merged report, when the run was asked for one
    // What it cost. The CPU time is the dispatcher's own, charged to it and to the children it
    // reaped, as wait4(2) reports it; the wall time is what the caller waited. A dispatcher
    // that is doing its job spends almost none of the first while a child runs, so the two
    // together are what separates a wait from a spin.
    std::chrono::duration<double> cpu{0};
    std::chrono::duration<double> wall{0};
};

// The dispatcher, its fixture binaries and its manifest, in a directory of the test's own.
//
// Under the build directory, not TMPDIR: TMPDIR may be mounted noexec, and the fixture binaries
// these tests copy in already live in the build tree.
class FixtureTree {
public:
    // Copy the dispatcher in, and the fixture binary in once per name in *binaries*. The
    // directory is made here rather than on construction, so RCCL_REQUIRE_TREE can skip first.
    bool build(const std::vector<std::string> &binaries)
    {
        if (!dir_) {
            dir_.emplace("rccl-host-dispatch-e2e", RCCL_HOST_TEST_BUILD_DIR);
        }
        std::error_code ec;
        fs::create_directories(micro(), ec);
        fs::copy_file(HOST_DISPATCH_FRONT_DOOR_PATH, frontDoor(),
                      fs::copy_options::overwrite_existing, ec);
        if (ec) {
            return false;
        }
        std::string manifest;
        for (const std::string &binary : binaries) {
            fs::copy_file(HOST_DISPATCH_FIXTURE_PATH, fs::path(micro()) / binary,
                          fs::copy_options::overwrite_existing, ec);
            if (ec) {
                return false;
            }
            manifest += binary + "\n";
        }
        std::ofstream(fs::path(micro()) / "host_test_binaries.txt") << manifest;
        fs::create_directories(records(), ec);
        return true;
    }

    std::string path() const { return dir_->path(); }
    std::string frontDoor() const { return (fs::path(dir_->path()) / "front-door").string(); }
    std::string micro() const { return (fs::path(dir_->path()) / "micro").string(); }
    std::string records() const { return (fs::path(dir_->path()) / "records").string(); }
    std::string reportPath() const { return (fs::path(dir_->path()) / "report.xml").string(); }

    // What the fixture binary copied in as *binary* recorded of the environment it was given.
    std::string recordedEnvironment(const std::string &binary) const
    {
        return recorded(binary + ".env");
    }

    // The command line that binary was handed, one argument per line.
    std::string recordedArguments(const std::string &binary) const
    {
        return recorded(binary + ".args");
    }

private:
    std::string recorded(const std::string &name) const
    {
        std::ifstream record(fs::path(records()) / name);
        std::ostringstream text;
        text << record.rdbuf();
        return text.str();
    }

    std::optional<ScopedDir> dir_;
};

// One name=value the dispatcher is to be run with. An empty value unsets the variable, so a test
// can run free of whatever the run that is executing IT was given.
struct EnvVar {
    std::string name;
    std::string value;
};

// Run the dispatcher in *tree* with *args* and *environment*, and collect what it produced.
//
// *stopAfter*, when set, SIGTERMs the dispatcher that long into the run -- a caller's Ctrl-C or
// an outer watchdog, arriving mid-run rather than between runs.
DispatchRun runFrontDoor(const FixtureTree &tree, const std::vector<std::string> &args,
                 const std::vector<EnvVar> &environment,
                 std::optional<std::chrono::milliseconds> stopAfter = std::nullopt)
{
    const std::string consolePath = (fs::path(tree.path()) / "console.txt").string();

    std::vector<std::string> owned = {tree.frontDoor()};
    owned.insert(owned.end(), args.begin(), args.end());
    std::vector<char *> argv;
    for (std::string &argument : owned) {
        argv.push_back(argument.data());
    }
    argv.push_back(nullptr);

    const pid_t pid = ::fork();
    if (pid == 0) {
        for (const EnvVar &variable : environment) {
            if (variable.value.empty()) {
                ::unsetenv(variable.name.c_str());
            } else {
                ::setenv(variable.name.c_str(), variable.value.c_str(), 1);
            }
        }
        ::freopen(consolePath.c_str(), "w", stdout);
        ::dup2(fileno(stdout), fileno(stderr));
        ::execv(argv[0], argv.data());
        ::_exit(127);
    }

    DispatchRun run;
    const auto started = std::chrono::steady_clock::now();
    if (pid > 0 && stopAfter) {
        std::this_thread::sleep_for(*stopAfter);
        ::kill(pid, SIGTERM);
    }
    int status = 0;
    struct rusage usage {};
    if (pid < 0 || ::wait4(pid, &status, 0, &usage) != pid) {
        return run;
    }
    run.wall = std::chrono::steady_clock::now() - started;
    const auto seconds = [](const struct timeval &time) {
        return std::chrono::duration<double>(static_cast<double>(time.tv_sec) +
                                             static_cast<double>(time.tv_usec) / 1e6);
    };
    run.cpu = seconds(usage.ru_utime) + seconds(usage.ru_stime);
    run.status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    {
        std::ifstream console(consolePath);
        std::ostringstream text;
        text << console.rdbuf();
        run.console = text.str();
    }
    {
        std::ifstream report(tree.reportPath());
        std::ostringstream text;
        text << report.rdbuf();
        run.report = text.str();
    }
    return run;
}

// The environment every one of these runs starts from.
//
// The dispatcher under test is itself usually running as a child of a dispatcher, which hands
// it a GTEST_* environment and an LLVM_PROFILE_FILE of its own. Left in place, those would
// decide the inner run's selection and scatter its profiles, so each test states the environment
// it means to test with and clears the rest.
std::vector<EnvVar> baseEnvironment(const FixtureTree &tree)
{
    return {
        {"RCCL_FIXTURE_RECORD_DIR", tree.records()},
        // Nothing here reads coverage, and the export shells out to llvm-profdata per binary.
        {"RCCL_HOST_COVERAGE_EXPORT", "0"},
        {"LLVM_PROFILE_FILE", ""},
        {"GTEST_FILTER", ""},
        {"GTEST_OUTPUT", ""},
        {"GTEST_ALSO_RUN_DISABLED_TESTS", ""},
        {"GTEST_TOTAL_SHARDS", ""},
        {"GTEST_SHARD_INDEX", ""},
        {"GTEST_SHARD_STATUS_FILE", ""},
        // One child at a time: the console transcript these tests read back is then in a fixed
        // order, and two fixture binaries are not a parallelism test.
        {"NUM_PARALLEL_HOST_TEST_BINARIES", "1"},
        {"RCCL_HOST_DISPATCH_LOG_DIR", "-"},
    };
}

// The whole tree, one fixture binary per name.
#define RCCL_REQUIRE_TREE(tree, ...)                                                      \
    do {                                                                                  \
        RCCL_SKIP_WITHOUT_DIRECTORY(RCCL_HOST_TEST_BUILD_DIR);                            \
        if (!fs::exists(HOST_DISPATCH_FRONT_DOOR_PATH) ||                                 \
            !fs::exists(HOST_DISPATCH_FIXTURE_PATH)) {                                    \
            GTEST_SKIP() << "no build tree to copy the front door and fixture from";      \
        }                                                                                 \
        ASSERT_TRUE((tree).build({__VA_ARGS__})) << "could not build the fixture tree";   \
    } while (false)

// Whether *text* contains *needle*, as a gtest-friendly predicate.
::testing::AssertionResult contains(const std::string &text, const std::string &needle)
{
    if (text.find(needle) != std::string::npos) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << "\"" << needle << "\" is not in:\n" << text;
}

// Whether every attribute value in *xml* is escaped, as a parser requires.
//
// Not a parser: a parser is a dependency this binary does not have, and the one rule worth
// checking is narrow -- an attribute value may contain no raw '<', and no '&' that does not
// begin an entity. Both are exactly what an unescaped path smuggles in.
::testing::AssertionResult attributesAreEscaped(const std::string &xml)
{
    const std::vector<std::string> entities = {"&amp;", "&lt;", "&gt;", "&quot;", "&apos;",
                                               "&#"};
    bool inTag = false;
    bool inValue = false;
    for (size_t at = 0; at < xml.size(); ++at) {
        const char c = xml[at];
        if (!inTag) {
            inTag = c == '<';
            continue;
        }
        if (c == '"') {
            inValue = !inValue;
            continue;
        }
        if (!inValue) {
            inTag = c != '>';
            continue;
        }
        if (c == '<') {
            return ::testing::AssertionFailure()
                   << "a raw '<' at offset " << at << " of:\n"
                   << xml;
        }
        if (c == '&') {
            const std::string rest = xml.substr(at, 6);
            const bool escaped =
                std::any_of(entities.begin(), entities.end(),
                            [&rest](const std::string &entity) {
                                return rest.compare(0, entity.size(), entity) == 0;
                            });
            if (!escaped) {
                return ::testing::AssertionFailure()
                       << "a raw '&' at offset " << at << " of:\n"
                       << xml;
            }
        }
    }
    return ::testing::AssertionSuccess();
}

// Every process still alive whose executable came out of *directory*, named as "pid: cmdline".
//
// Read from /proc rather than from the process tree, because the processes this looks for are
// exactly the ones that are no longer in it: a child the dispatcher failed to kill outlives it,
// is reparented to init, and so cannot be found by walking down from this test. They are
// recognisable because each fixture binary is a copy the test itself put in its own directory.
//
// *settle* is allowed for ones that are on their way out of their own accord: a process the
// dispatcher killed may not have been reaped by init by the time the dispatcher's own exit is
// seen here. A genuinely abandoned fixture sleeps for minutes, so no settling time hides one.
std::vector<std::string> survivorsFrom(const std::string &directory,
                                       std::chrono::milliseconds settle)
{
    const auto deadline = std::chrono::steady_clock::now() + settle;
    std::vector<std::string> survivors;
    for (;;) {
        survivors.clear();
        std::error_code ec;
        for (const fs::directory_entry &entry : fs::directory_iterator("/proc", ec)) {
            const std::string pid = entry.path().filename().string();
            if (pid.empty() || !std::all_of(pid.begin(), pid.end(), ::isdigit)) {
                continue;
            }
            const fs::path executable = fs::read_symlink(entry.path() / "exe", ec);
            if (ec) {
                continue;   // gone, or not ours to look at
            }
            if (executable.string().rfind(directory, 0) == 0) {
                survivors.push_back(pid + ": " + executable.string());
            }
        }
        if (survivors.empty() || std::chrono::steady_clock::now() >= deadline) {
            return survivors;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
    }
}

// What *survivors* says, as one line for a failure message.
std::string joined(const std::vector<std::string> &lines)
{
    std::string text;
    for (const std::string &line : lines) {
        text += line + "\n";
    }
    return text;
}

}  // namespace

TEST(HostDispatchEndToEnd, EveryFrontedBinarysCasesReachTheMergedReport)
{
    // The whole point of the front door, end to end: the cases of every binary the manifest
    // names are published under that binary's qualifier, in one report, from one invocation.
    FixtureTree tree;
    RCCL_REQUIRE_TREE(tree, kFirstBinary, kSecondBinary);

    const DispatchRun run = runFrontDoor(
        tree, {"--gtest_output=xml:" + tree.reportPath()}, baseEnvironment(tree));

    EXPECT_TRUE(contains(run.report,
                         std::string("<testsuite name=\"") + kFirstQualifier + "/FixtureSuite\""));
    EXPECT_TRUE(contains(run.report, std::string("<testsuite name=\"") + kSecondQualifier +
                                         "/FixtureSuite\""));
    EXPECT_TRUE(contains(run.report, "name=\"Passes\""));
}

TEST(HostDispatchEndToEnd, ARunWhoseCasesAllPassExitsZero)
{
    // The exit code is the only thing most callers read. A filter that selects nothing but
    // passing cases has to come back green, or a green tree is indistinguishable from a broken
    // dispatcher.
    FixtureTree tree;
    RCCL_REQUIRE_TREE(tree, kFirstBinary);

    const DispatchRun run = runFrontDoor(tree,
                                 {"--gtest_output=xml:" + tree.reportPath(),
                                  "--gtest_filter=*/FixtureSuite.Passes"},
                                 baseEnvironment(tree));

    EXPECT_EQ(run.status, 0) << run.console;
    EXPECT_TRUE(contains(run.report, "name=\"Passes\""));
    EXPECT_FALSE(contains(run.report, "name=\"Fails\""));
}

TEST(HostDispatchEndToEnd, AFailingCaseInOneChildFailsTheRun)
{
    // The other half of the same contract: a failure anywhere under the front door has to reach
    // the caller's exit code, through the child's report, the merge and main().
    FixtureTree tree;
    RCCL_REQUIRE_TREE(tree, kFirstBinary);

    const DispatchRun run = runFrontDoor(tree,
                                 {"--gtest_output=xml:" + tree.reportPath(),
                                  "--gtest_filter=*/FixtureSuite.Fails"},
                                 baseEnvironment(tree));

    EXPECT_EQ(run.status, 1) << run.console;
    EXPECT_TRUE(contains(run.report, "<failure message="));
}

TEST(HostDispatchEndToEnd, ADisabledCaseTheFilterNamesIsStillReported)
{
    // gtest's third state, decided by gtest itself inside this process: a DISABLED_ case a
    // filter matched is not run, but it IS reported, as notrun/suppressed. The dispatcher has to
    // carry that through to the child, by naming the case in the child's filter -- drop it and
    // the case vanishes from the report of a run that explicitly asked for it, with nothing
    // failing to say so.
    FixtureTree tree;
    RCCL_REQUIRE_TREE(tree, kFirstBinary);

    const DispatchRun run = runFrontDoor(
        tree,
        {"--gtest_output=xml:" + tree.reportPath(),
         "--gtest_filter=*/FixtureSuite.Passes:*/FixtureSuite.DISABLED_Skipped"},
        baseEnvironment(tree));

    EXPECT_EQ(run.status, 0) << run.console;
    EXPECT_TRUE(contains(run.report, "name=\"Passes\" status=\"run\""));
    EXPECT_TRUE(contains(run.report,
                         "name=\"DISABLED_Skipped\" status=\"notrun\" result=\"suppressed\""));
}

TEST(HostDispatchEndToEnd, AskingForDisabledCasesHandsTheChildTheWholeBinary)
{
    // --gtest_also_run_disabled_tests has to reach BOTH the selection here and each child. If
    // only the child is told, the dispatcher no longer sees the DISABLED_ cases as runnable, so
    // a run that selected everything is handed to the child as an explicit list of names
    // instead of as "*" -- which still works, and stops working the moment the list is long
    // enough to be split, or a case is added between the listing and the run.
    FixtureTree tree;
    RCCL_REQUIRE_TREE(tree, kFirstBinary);

    const DispatchRun run = runFrontDoor(tree,
                                         {"--gtest_output=xml:" + tree.reportPath(),
                                          "--gtest_also_run_disabled_tests"},
                                         baseEnvironment(tree));

    EXPECT_TRUE(contains(tree.recordedArguments(kFirstBinary), "--gtest_filter=*\n"));
    EXPECT_TRUE(contains(run.report, "name=\"DISABLED_Skipped\" status=\"run\""));
}

TEST(HostDispatchEndToEnd, TheListingPublishesEveryBinarysCasesQualified)
{
    // --gtest_list_tests is how a caller builds a work list, and how the shared category YAML is
    // checked against the tree. It runs through the enumeration and gtest's own printer, neither
    // of which a unit test can reach.
    FixtureTree tree;
    RCCL_REQUIRE_TREE(tree, kFirstBinary, kSecondBinary);

    const DispatchRun run = runFrontDoor(tree, {"--gtest_list_tests"}, baseEnvironment(tree));

    EXPECT_EQ(run.status, 0) << run.console;
    EXPECT_TRUE(contains(run.console, std::string(kFirstQualifier) + "/FixtureSuite."));
    EXPECT_TRUE(contains(run.console, std::string(kSecondQualifier) + "/FixtureSuite."));
}

TEST(HostDispatchEndToEnd, AChildNeverSeesTheGtestVariablesTheDispatcherActedOn)
{
    // The failure this guards is silent and total: a child that inherits GTEST_FILTER applies it
    // a second time, to a list that has already been narrowed to exact names, runs none of them
    // and exits 0. The run is green and has tested nothing.
    FixtureTree tree;
    RCCL_REQUIRE_TREE(tree, kFirstBinary);

    std::vector<EnvVar> environment = baseEnvironment(tree);
    environment.push_back({"GTEST_FILTER", "NothingMatchesThis"});
    environment.push_back({"GTEST_TOTAL_SHARDS", "4"});
    environment.push_back({"GTEST_SHARD_INDEX", "0"});
    // Forwarded, by the table in host_test_dispatch.cc: it changes nothing about which cases run.
    environment.push_back({"GTEST_COLOR", "no"});

    const DispatchRun run = runFrontDoor(
        tree, {"--gtest_output=xml:" + tree.reportPath(), "--gtest_filter=*/FixtureSuite.Passes"},
        environment);

    const std::string recorded = tree.recordedEnvironment(kFirstBinary);
    EXPECT_FALSE(contains(recorded, "GTEST_FILTER="));
    EXPECT_FALSE(contains(recorded, "GTEST_TOTAL_SHARDS="));
    EXPECT_FALSE(contains(recorded, "GTEST_SHARD_INDEX="));
    EXPECT_TRUE(contains(recorded, "GTEST_COLOR=no"));
    EXPECT_EQ(run.status, 0) << run.console;
}

TEST(HostDispatchEndToEnd, EachChildWritesItsProfileUnderItsOwnName)
{
    // The children share one LLVM_PROFILE_FILE pattern, so without the per-binary directory the
    // dispatcher inserts, their profiles land on one another and the merged coverage is one
    // binary's. Nothing fails when that breaks: the report is still green and merely wrong.
    FixtureTree tree;
    RCCL_REQUIRE_TREE(tree, kFirstBinary, kSecondBinary);

    const std::string pattern = (fs::path(tree.path()) / "profiles" / "rccl_%p.profraw").string();
    std::vector<EnvVar> environment = baseEnvironment(tree);
    environment.push_back({"LLVM_PROFILE_FILE", pattern});

    runFrontDoor(tree, {"--gtest_output=xml:" + tree.reportPath()}, environment);

    EXPECT_TRUE(contains(tree.recordedEnvironment(kFirstBinary),
                         (fs::path(tree.path()) / "profiles" / kFirstBinary / "rccl_%p.profraw")
                             .string()));
    EXPECT_TRUE(contains(tree.recordedEnvironment(kSecondBinary),
                         (fs::path(tree.path()) / "profiles" / kSecondBinary / "rccl_%p.profraw")
                             .string()));
}

TEST(HostDispatchEndToEnd, EachChildsProfileDirectoryIsMadeForIt)
{
    // The profile runtime makes the directory itself only for patterns with %m; a child handed a
    // plain %p one, as an outside collector may use, would otherwise write no profile at all.
    FixtureTree tree;
    RCCL_REQUIRE_TREE(tree, kFirstBinary, kSecondBinary);

    const fs::path profiles = fs::path(tree.path()) / "profiles";
    std::vector<EnvVar> environment = baseEnvironment(tree);
    environment.push_back({"LLVM_PROFILE_FILE", (profiles / "rccl_%p.profraw").string()});

    runFrontDoor(tree, {"--gtest_output=xml:" + tree.reportPath()}, environment);

    EXPECT_TRUE(fs::is_directory(profiles / kFirstBinary));
    EXPECT_TRUE(fs::is_directory(profiles / kSecondBinary));
}

TEST(HostDispatchEndToEnd, AChildThatHangsIsKilledAtItsTimeoutAndReportedAsOne)
{
    // A hung binary used to cost the whole run: the merged report is written once, after the
    // last child is reaped. The timeout bounds what one binary can cost, and the binary it
    // killed still has to appear in the report as a failure rather than simply be missing.
    FixtureTree tree;
    RCCL_REQUIRE_TREE(tree, kFirstBinary);

    std::vector<EnvVar> environment = baseEnvironment(tree);
    environment.push_back({"RCCL_FIXTURE_HANG", "1"});
    environment.push_back({"RCCL_HOST_DISPATCH_CHILD_TIMEOUT", "1"});

    const auto started = std::chrono::steady_clock::now();
    const DispatchRun run = runFrontDoor(
        tree, {"--gtest_output=xml:" + tree.reportPath()}, environment);
    const auto elapsed = std::chrono::steady_clock::now() - started;

    EXPECT_EQ(run.status, 1) << run.console;
    EXPECT_TRUE(contains(run.report, kFirstQualifier));
    // The fixture sleeps for minutes; the timeout plus its grace period is seconds.
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(), 60);
}

TEST(HostDispatchEndToEnd, AStopReportsWhatItKilledLeavesNothingBehindAndSleepsThroughTheGrace)
{
    // What a Ctrl-C, or the outer watchdog's SIGTERM, has to buy: the run ends promptly, the
    // report is still written and still names the binary that was cut short, the exit code says
    // so, no test binary outlives the dispatcher, and the wind-down costs no CPU. Each of those
    // is decided in a different place -- the handler, the scheduler's wind-down, classifyChild,
    // main() -- and none of them is reachable from a unit test, because every scheduler test
    // goes through FakeScheduler.
    //
    // The fixture is told to ignore SIGTERM, so the wind-down has to go the whole way through
    // its grace period and the SIGKILL that follows. That is the expensive path, and the reason
    // this case is the costliest in the file: it spends the whole of kTerminateGrace. It is one
    // test rather than two because a second run of the identical scenario, differing only in
    // what it asserted, cost another full grace period -- ~6 s on every host test run, since
    // this binary is on the front door's critical path -- to buy one more EXPECT.
    //
    // The last assertion is the mutation-sensitive one. It is easy to make the wind-down cost a
    // whole core and notice nothing, because every observable thing above still holds: a reap
    // that reports a requested stop on every call turns the scheduler's loop into a spin for the
    // whole of the grace -- measured at 5.03 s of CPU in 6.01 s of wall -- and the report, the
    // exit code and the kills all come out the same. On a loaded CI machine that core is taken
    // from the tests still running. So the assertion is on the RATIO, not on a wall-clock bound:
    // the dispatcher may sit in the grace period as long as it likes, as long as it is asleep
    // for it. wait4(2) hands back the dispatcher's own CPU, which is what makes it cheap to ask.
    FixtureTree tree;
    RCCL_REQUIRE_TREE(tree, kFirstBinary);

    std::vector<EnvVar> environment = baseEnvironment(tree);
    environment.push_back({"RCCL_FIXTURE_HANG", "1"});
    environment.push_back({"RCCL_FIXTURE_IGNORE_TERM", "1"});

    const DispatchRun run =
        runFrontDoor(tree, {"--gtest_output=xml:" + tree.reportPath()}, environment,
                     std::chrono::milliseconds{1000});

    EXPECT_EQ(run.status, 1) << run.console;
    EXPECT_TRUE(contains(run.report, kFirstQualifier));
    EXPECT_TRUE(contains(run.report, "run was stopped"));
    const std::vector<std::string> survivors =
        survivorsFrom(tree.path(), std::chrono::milliseconds{500});
    EXPECT_TRUE(survivors.empty()) << "left running after the stop:\n" << joined(survivors);

    // The run has to have reached the wind-down for the measurement below to mean anything: a
    // stop the dispatcher took no time at all over has no grace period to have spun through.
    ASSERT_GT(run.wall.count(), 2.0) << "the stop did not reach the teardown grace\n"
                                     << run.console;
    // Generous by a long way against a spin, which charges most of the wall time to the CPU.
    EXPECT_LT(run.cpu.count(), 0.25 * run.wall.count())
        << "the dispatcher spent " << run.cpu.count() << " s of CPU over " << run.wall.count()
        << " s of wall after the stop";
}

TEST(HostDispatchEndToEnd, AManifestNamingABinaryThatIsNotThereFailsTheRun)
{
    // A binary that cannot be listed holds an unknown number of cases, so the run cannot be
    // green whatever the rest of the tree does -- and it has to be named, because a binary
    // missing from a report is indistinguishable from one the tree never had.
    FixtureTree tree;
    RCCL_REQUIRE_TREE(tree, kFirstBinary);
    std::ofstream(fs::path(tree.micro()) / "host_test_binaries.txt")
        << kFirstBinary << "\n" << kSecondBinary << "\n";

    const DispatchRun run = runFrontDoor(
        tree, {"--gtest_output=xml:" + tree.reportPath()}, baseEnvironment(tree));

    EXPECT_EQ(run.status, 1) << run.console;
    EXPECT_TRUE(contains(run.report, std::string(kSecondQualifier) + "/HostTestDispatch"));
    EXPECT_TRUE(contains(run.report, "could not list its tests"));
}

TEST(HostDispatchEndToEnd, AReportAboutALogPathFullOfXmlStaysReadable)
{
    // The log directory is the caller's to name, and the dispatcher writes that path into the
    // crash fragment's message. A directory called "logs & <today>" then puts a bare '&' and a
    // bare '<' inside an attribute value, and no XML parser will read the result -- on exactly
    // the run that failed, which is the only run whose report anybody opens.
    FixtureTree tree;
    RCCL_REQUIRE_TREE(tree, kFirstBinary);

    const std::string logDir = (fs::path(tree.path()) / "logs & <today>").string();
    std::vector<EnvVar> environment = baseEnvironment(tree);
    environment.push_back({"RCCL_HOST_DISPATCH_LOG_DIR", logDir});
    environment.push_back({"RCCL_FIXTURE_HANG", "1"});
    environment.push_back({"RCCL_HOST_DISPATCH_CHILD_TIMEOUT", "1"});

    const DispatchRun run = runFrontDoor(
        tree, {"--gtest_output=xml:" + tree.reportPath()}, environment);

    EXPECT_EQ(run.status, 1) << run.console;
    EXPECT_TRUE(attributesAreEscaped(run.report));
    EXPECT_TRUE(contains(run.report, "logs &amp; &lt;today&gt;"));
}
