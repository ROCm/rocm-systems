/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for host_test_dispatch.cc: the --gtest_list_tests parsing it dispatches
// from, the name qualification that keeps the multi-arm binaries' identical case names apart,
// the JUnit XML merge, and the argument handling.
//
// Two parts are deliberately not covered here.
//
// resolveFilter() cannot be: it registers the published names with gtest and calls
// RUN_ALL_TESTS() to make gtest resolve the filter against them, which cannot be nested inside a
// running test. There is also nothing of ours left to assert -- the matching IS gtest's, which
// is the point of delegating it rather than carrying a filter matcher of our own to get subtly
// wrong (the hand-rolled one selected DISABLED_ cases that gtest would not have run).
//
// The process plumbing (fork/exec, discovery, the coverage manifest) is not covered either:
// those are thin wrappers over syscalls with no branching logic of their own, and exercising
// them would test the kernel. Everything else that decides WHAT runs and WHAT is reported is
// below.

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#define RCCL_HOST_DISPATCH_NO_MAIN
#include "host_test_dispatch.cc"

using namespace rccl_host_dispatch;

namespace {

// ── parseListTests ──────────────────────────────────────────────────

TEST(HostDispatchListingMicrotest, SuiteLinesQualifyTheCasesBeneathThem)
{
    const std::vector<TestCaseName> cases = parseListTests(
        "FirstSuite.\n"
        "  CaseOne\n"
        "  CaseTwo\n"
        "SecondSuite.\n"
        "  CaseThree\n");
    EXPECT_EQ(cases, (std::vector<TestCaseName>{{"FirstSuite", "CaseOne"},
                                                {"FirstSuite", "CaseTwo"},
                                                {"SecondSuite", "CaseThree"}}));
}

TEST(HostDispatchListingMicrotest, SuiteAndCaseStaySplit)
{
    // Kept apart rather than joined into "Suite.Case": a parameterized case name is printed from
    // its parameter and can contain '.' itself, so splitting a joined name later would cut in
    // the wrong place and hand the child a filter that matches nothing.
    const std::vector<TestCaseName> cases = parseListTests("Suite.\n  Case/1.5\n");
    ASSERT_EQ(cases.size(), 1u);
    EXPECT_EQ(cases[0].suite, "Suite");
    EXPECT_EQ(cases[0].test, "Case/1.5");
}

TEST(HostDispatchListingMicrotest, ParamAnnotationIsNotPartOfTheCaseName)
{
    const std::vector<TestCaseName> cases =
        parseListTests("Prefix/Suite.\n  Case/0  # GetParam() = 42\n");
    EXPECT_EQ(cases, (std::vector<TestCaseName>{{"Prefix/Suite", "Case/0"}}));
}

TEST(HostDispatchListingMicrotest, OutputBeforeTheFirstSuiteLineIsIgnored)
{
    // A binary that prints a banner (an env-var dump, a "Running main() from ..." line) must not
    // have that banner read as a suite, or its first suite's cases would be attributed to it.
    const std::vector<TestCaseName> cases =
        parseListTests("Running main() from gtest_main.cc\n  NotACase\nSuite.\n  Case\n");
    EXPECT_EQ(cases, (std::vector<TestCaseName>{{"Suite", "Case"}}));
}

TEST(HostDispatchListingMicrotest, EmptyOutputListsNothing)
{
    EXPECT_TRUE(parseListTests("").empty());
}

// ── qualification ───────────────────────────────────────────────────

TEST(HostDispatchQualifierMicrotest, BinarySuffixBecomesTheQualifier)
{
    EXPECT_EQ(qualifierFor("rccl-UnitTestsMicro"), "Micro");
    EXPECT_EQ(qualifierFor("rccl-UnitTestsMicroInit"), "MicroInit");
    EXPECT_EQ(qualifierFor("rccl-UnitTestsMicroInit-uncached"), "MicroInitUncached");
}

TEST(HostDispatchQualifierMicrotest, VariantDashesAreFoldedAway)
{
    // A dash would make the qualifier unnameable in a positive --gtest_filter, which is the only
    // way a lane config can ask for one arm of a multi-arm binary.
    EXPECT_EQ(qualifierFor("rccl-UnitTestsMicroRoctxNvml-roctx"), "MicroRoctxNvmlRoctx");
    EXPECT_EQ(qualifierFor("rccl-UnitTestsMicroEnqueue-devlinker"), "MicroEnqueueDevlinker");
    EXPECT_EQ(qualifierFor("rccl-UnitTestsMicroInit-faultinj").find('-'), std::string::npos);
}

TEST(HostDispatchQualifierMicrotest, MultiArmBinariesGetDistinctQualifiers)
{
    // The whole point: these three compile the same init.cc cases, so the qualifier is the only
    // thing that keeps their results from collapsing into one identity.
    EXPECT_NE(qualifierFor("rccl-UnitTestsMicroInit"), qualifierFor("rccl-UnitTestsMicroInit-uncached"));
    EXPECT_NE(qualifierFor("rccl-UnitTestsMicroInit"), qualifierFor("rccl-UnitTestsMicroInit-faultinj"));
}

TEST(HostDispatchQualifierMicrotest, QualifiedSuiteIsGtestPrefixShaped)
{
    // gtest's own Prefix/Suite shape, so a published name stays a legal gtest identity that
    // gtest itself resolves filters against and "*InitMicrotest.*" still selects.
    EXPECT_EQ(qualifiedSuite("MicroInit", "InitMicrotest"), "MicroInit/InitMicrotest");
}

// ── childFilters ────────────────────────────────────────────────────

// A selection of *selected* cases out of a binary holding *runnable* of them.
Selection selectionOf(size_t selected, size_t runnable, const std::string &stem = "Case")
{
    Selection selection;
    selection.path = "/build/test/host/rccl-UnitTestsMicro";
    selection.qualifier = "Micro";
    selection.runnable = runnable;
    for (size_t i = 0; i < selected; ++i) {
        selection.cases.push_back(TestCaseName{"Suite", stem + std::to_string(i)});
    }
    return selection;
}

TEST(HostDispatchChildFilterMicrotest, WholeBinaryIsAskedForWithAStar)
{
    EXPECT_EQ(childFilters(selectionOf(3, 3)), (std::vector<std::string>{"*"}));
}

TEST(HostDispatchChildFilterMicrotest, PartialSelectionIsNamedCaseByCase)
{
    // Never the caller's own pattern: it is written against the qualified names, so in a
    // multi-arm binary it would also match the cases of the arm it was meant to exclude.
    const std::vector<std::string> filters = childFilters(selectionOf(2, 5));
    ASSERT_EQ(filters.size(), 1u);
    EXPECT_EQ(filters[0], "Suite.Case0:Suite.Case1");
}

// ── group ───────────────────────────────────────────────────────────

// Two cases in one binary, the second of them disabled, with only the first selected.
std::vector<CandidateCase> oneEnabledOneDisabled()
{
    return {CandidateCase{0, "Micro/Suite", TestCaseName{"Suite", "Enabled"}},
            CandidateCase{0, "Micro/Suite", TestCaseName{"Suite", "DISABLED_Skipped"}}};
}

TEST(HostDispatchGroupMicrotest, DisabledCasesDoNotCountAsHeldByTheBinary)
{
    // gtest lists a DISABLED_ case but never selects it, so a binary holding one could never
    // have "all its cases" selected. Counting it would make every such selection look partial,
    // and the biggest binary would then be asked for its 2000-odd cases by name -- past the exec
    // argument limit -- to run exactly what "*" runs.
    const std::vector<Selection> selections =
        group("/dir", {"rccl-UnitTestsMicro"}, oneEnabledOneDisabled(), {0}, false);
    ASSERT_EQ(selections.size(), 1u);
    EXPECT_EQ(selections[0].runnable, 1u);
    EXPECT_EQ(childFilters(selections[0]), (std::vector<std::string>{"*"}));
}

TEST(HostDispatchGroupMicrotest, DisabledCasesCountWhenTheCallerAsksForThem)
{
    const std::vector<Selection> selections =
        group("/dir", {"rccl-UnitTestsMicro"}, oneEnabledOneDisabled(), {0}, true);
    ASSERT_EQ(selections.size(), 1u);
    EXPECT_EQ(selections[0].runnable, 2u);
    // One of two selected is a real partial selection now, so the case is named.
    EXPECT_EQ(childFilters(selections[0]), (std::vector<std::string>{"Suite.Enabled"}));
}

TEST(HostDispatchGroupMicrotest, CasesAreGroupedUnderTheBinaryThatOwnsThem)
{
    const std::vector<CandidateCase> candidates = {
        CandidateCase{0, "Micro/Suite", TestCaseName{"Suite", "A"}},
        CandidateCase{1, "MicroInit/Suite", TestCaseName{"Suite", "B"}},
        CandidateCase{1, "MicroInit/Suite", TestCaseName{"Suite", "C"}}};
    const std::vector<Selection> selections =
        group("/dir", {"rccl-UnitTestsMicro", "rccl-UnitTestsMicroInit"}, candidates, {0, 1, 2},
              false);
    ASSERT_EQ(selections.size(), 2u);
    EXPECT_EQ(selections[0].path, "/dir/rccl-UnitTestsMicro");
    EXPECT_EQ(selections[1].path, "/dir/rccl-UnitTestsMicroInit");
    EXPECT_EQ(selections[1].cases.size(), 2u);
}

TEST(HostDispatchGroupMicrotest, UnselectedBinariesAreNotRun)
{
    // A binary with nothing selected must not be forked at all: it would run its whole suite,
    // since an empty filter means "everything" to gtest.
    const std::vector<Selection> selections =
        group("/dir", {"rccl-UnitTestsMicro"}, oneEnabledOneDisabled(), {}, false);
    EXPECT_TRUE(selections.empty());
}

TEST(HostDispatchGroupMicrotest, DisabledIsRecognisedOnEitherHalfOfTheName)
{
    EXPECT_TRUE(isDisabled(TestCaseName{"Suite", "DISABLED_Case"}));
    EXPECT_TRUE(isDisabled(TestCaseName{"DISABLED_Suite", "Case"}));
    EXPECT_FALSE(isDisabled(TestCaseName{"Suite", "NotDISABLED_Case"}));
}

TEST(HostDispatchChildFilterMicrotest, AnOverLongFilterIsSplitAcrossInvocations)
{
    // execve rejects a single argument over MAX_ARG_STRLEN (128KiB) with E2BIG, which the
    // biggest host binary's case list exceeds. The child is run once per part instead, and the
    // reports are merged, so splitting costs nothing but a second fork.
    const std::vector<std::string> filters =
        childFilters(selectionOf(4000, 9000, std::string(60, 'C')));
    ASSERT_GT(filters.size(), 1u);
    for (const std::string &filter : filters) {
        EXPECT_LE(filter.size(), kMaxFilterChars);
    }
}

TEST(HostDispatchChildFilterMicrotest, SplittingKeepsEveryCaseExactlyOnce)
{
    const Selection selection = selectionOf(4000, 9000, std::string(60, 'C'));
    size_t named = 0;
    for (const std::string &filter : childFilters(selection)) {
        named += std::count(filter.begin(), filter.end(), ':') + 1;
    }
    EXPECT_EQ(named, selection.cases.size());
}

// ── applyQualifier ──────────────────────────────────────────────────

TEST(HostDispatchXmlMicrotest, QualifierMovesSuiteNameAndClassnameTogether)
{
    const std::string qualified = applyQualifier(
        "<testsuite name=\"InitMicrotest\" tests=\"1\">"
        "<testcase name=\"Case\" classname=\"InitMicrotest\"/>"
        "</testsuite>",
        "MicroInit");
    EXPECT_NE(qualified.find("<testsuite name=\"MicroInit/InitMicrotest\""), std::string::npos);
    EXPECT_NE(qualified.find("classname=\"MicroInit/InitMicrotest\""), std::string::npos);
}

TEST(HostDispatchXmlMicrotest, CaseNameIsNotQualified)
{
    const std::string qualified =
        applyQualifier("<testcase name=\"Case\" classname=\"Suite\"/>", "Micro");
    EXPECT_NE(qualified.find("name=\"Case\""), std::string::npos);
}

// ── parseChildReport ────────────────────────────────────────────────

// A minimal but structurally faithful child report.
std::string childReport(int tests, int failures)
{
    return fmt::format(
        "<?xml version=\"1.0\"?>\n"
        "<testsuites tests=\"{}\" failures=\"{}\" disabled=\"0\" errors=\"0\" time=\"1.5\""
        " name=\"AllTests\">\n"
        "  <testsuite name=\"Suite\" tests=\"1\">\n"
        "    <testcase name=\"Case\" classname=\"Suite\"/>\n"
        "  </testsuite>\n"
        "</testsuites>\n",
        tests, failures);
}

// *report* parsed, failing the calling test when it does not parse at all.
SuiteFragment parsedOrFail(const std::string &report, const std::string &qualifier)
{
    const std::optional<SuiteFragment> fragment = parseChildReport(report, qualifier);
    EXPECT_TRUE(fragment.has_value()) << "report did not parse under qualifier " << qualifier;
    return fragment.value_or(SuiteFragment{});
}

TEST(HostDispatchReportMicrotest, RootTotalsAreTakenFromTheChild)
{
    const SuiteFragment fragment = parsedOrFail(childReport(7, 2), "Micro");
    EXPECT_EQ(fragment.tests, 7);
    EXPECT_EQ(fragment.failures, 2);
    EXPECT_DOUBLE_EQ(fragment.time, 1.5);
}

TEST(HostDispatchReportMicrotest, FragmentCarriesTheQualifiedSuitesOnly)
{
    const SuiteFragment fragment = parsedOrFail(childReport(1, 0), "Micro");
    EXPECT_NE(fragment.xml.find("<testsuite name=\"Micro/Suite\""), std::string::npos);
    EXPECT_EQ(fragment.xml.find("<testsuites"), std::string::npos);
}

TEST(HostDispatchReportMicrotest, OutputThatIsNotAGtestReportIsRejected)
{
    // How a child that died before writing its report is recognised: the file is absent or
    // truncated, and the dispatcher must not read that as a clean run of zero tests.
    EXPECT_FALSE(parseChildReport("", "Micro").has_value());
    EXPECT_FALSE(parseChildReport("Segmentation fault\n", "Micro").has_value());
    EXPECT_FALSE(parseChildReport("<testsuites tests=\"1\">\n", "Micro").has_value());
}

TEST(HostDispatchReportMicrotest, CrashFragmentReportsOneFailure)
{
    const SuiteFragment fragment = crashFragment("MicroInit", "exit status 139");
    EXPECT_EQ(fragment.tests, 1);
    EXPECT_EQ(fragment.failures, 1);
    EXPECT_NE(fragment.xml.find("MicroInit/HostTestDispatch"), std::string::npos);
    EXPECT_NE(fragment.xml.find("exit status 139"), std::string::npos);
}

// ── mergeReports ────────────────────────────────────────────────────

TEST(HostDispatchMergeMicrotest, TotalsAreSummedAcrossChildren)
{
    const SuiteFragment first = parsedOrFail(childReport(3, 1), "Micro");
    const SuiteFragment second = parsedOrFail(childReport(4, 2), "MicroInit");
    const std::string merged = mergeReports({first, second});
    EXPECT_NE(merged.find("tests=\"7\""), std::string::npos);
    EXPECT_NE(merged.find("failures=\"3\""), std::string::npos);
}

TEST(HostDispatchMergeMicrotest, EverySuiteSurvivesUnderItsOwnQualifier)
{
    const SuiteFragment first = parsedOrFail(childReport(1, 0), "MicroInit");
    const SuiteFragment second = parsedOrFail(childReport(1, 0), "MicroInitUncached");
    const std::string merged = mergeReports({first, second});
    EXPECT_NE(merged.find("<testsuite name=\"MicroInit/Suite\""), std::string::npos);
    EXPECT_NE(merged.find("<testsuite name=\"MicroInitUncached/Suite\""), std::string::npos);
}

TEST(HostDispatchMergeMicrotest, NoChildrenStillProducesAValidEmptyReport)
{
    // The report is the caller's second signal that the run happened at all (coco fails a test
    // whose XML is missing or has no <testsuites>), so a filter that selects nothing must still
    // produce a well-formed document rather than no file.
    const std::string merged = mergeReports({});
    EXPECT_NE(merged.find("<testsuites tests=\"0\""), std::string::npos);
    EXPECT_NE(merged.find("</testsuites>"), std::string::npos);
}

// ── parseArgs ───────────────────────────────────────────────────────

TEST(HostDispatchArgsMicrotest, FilterAndOutputAreTakenFromTheCaller)
{
    const Options options =
        parseArgs({"--gtest_filter=Suite.*", "--gtest_output=xml:/tmp/report.xml"});
    EXPECT_EQ(options.filter, "Suite.*");
    EXPECT_EQ(options.outputPath, "/tmp/report.xml");
    EXPECT_FALSE(options.listTests);
}

TEST(HostDispatchArgsMicrotest, FilterDefaultsToEverything)
{
    EXPECT_EQ(parseArgs({}).filter, "*");
}

TEST(HostDispatchArgsMicrotest, OutputDirectoryGetsGtestsDefaultFilename)
{
    EXPECT_EQ(parseArgs({"--gtest_output=xml:/tmp/reports/"}).outputPath,
              "/tmp/reports/test_detail.xml");
}

TEST(HostDispatchArgsMicrotest, UnrecognisedFlagsAreForwardedToTheChildren)
{
    // Forwarding rather than rejecting is what keeps the dispatcher from having to track gtest's
    // CLI: --gtest_repeat, --gtest_shuffle and a binary's own switches all still work.
    const Options options = parseArgs({"--gtest_shuffle", "--gtest_repeat=3", "--verbose"});
    EXPECT_EQ(options.passthrough,
              (std::vector<std::string>{"--gtest_shuffle", "--gtest_repeat=3", "--verbose"}));
}

TEST(HostDispatchArgsMicrotest, AlsoRunDisabledIsBothActedOnAndForwarded)
{
    // The only flag that is both: the children need it to run their DISABLED_ cases, and the
    // dispatcher needs it or gtest would not select those cases for dispatch at all, leaving the
    // caller's flag silently ineffective.
    const Options options = parseArgs({"--gtest_also_run_disabled_tests"});
    EXPECT_TRUE(options.alsoRunDisabled);
    EXPECT_EQ(options.passthrough, (std::vector<std::string>{"--gtest_also_run_disabled_tests"}));
}

TEST(HostDispatchArgsMicrotest, ListTestsIsActedOnNotForwarded)
{
    const Options options = parseArgs({"--gtest_list_tests"});
    EXPECT_TRUE(options.listTests);
    EXPECT_TRUE(options.passthrough.empty());
}

TEST(HostDispatchArgsMicrotest, HelpIsActedOnNotForwarded)
{
    // Forwarding help is what the passthrough rule would do by default, and it is wrong: every
    // child would print the same gtest help screen and exit without writing a report, which the
    // dispatcher scores as that binary having crashed. Asking for help must not fail the run.
    for (const std::string flag : {"--help", "-h", "-?", "/?"}) {
        const Options options = parseArgs({flag});
        EXPECT_TRUE(options.help) << flag;
        EXPECT_TRUE(options.passthrough.empty()) << flag;
    }
}

TEST(HostDispatchArgsMicrotest, HelpIsNotAssumedOfEveryFlagThatMerelyLooksLikeIt)
{
    // The help flags are an exact set, not a prefix match: a child's own flag could begin with
    // any of them, and swallowing it here would make that flag silently ineffective.
    const Options options = parseArgs({"--helper-mode", "-household"});
    EXPECT_FALSE(options.help);
    EXPECT_EQ(options.passthrough, (std::vector<std::string>{"--helper-mode", "-household"}));
}

// ── coverageDirectory ───────────────────────────────────────────────

TEST(HostDispatchCoverageMicrotest, ObjectsAreRecordedBesideTheProfiles)
{
    // The harness points LLVM_PROFILE_FILE at a pattern, not a file; only its directory is a real
    // path, and that is where the object manifest has to land for the harness to find it.
    setenv("LLVM_PROFILE_FILE", "/tmp/coverage_raw/rccl_%h_%p_%m.profraw", 1);
    EXPECT_EQ(coverageDirectory(), "/tmp/coverage_raw");
    unsetenv("LLVM_PROFILE_FILE");
}

TEST(HostDispatchCoverageMicrotest, NoProfileVariableMeansNoManifest)
{
    unsetenv("LLVM_PROFILE_FILE");
    EXPECT_EQ(coverageDirectory(), "");
}

}  // namespace
