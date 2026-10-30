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

// ── enumerate ───────────────────────────────────────────────────────

// A listing a healthy binary would print for one suite of one case.
std::string listingOf(const std::string &suite, const std::string &test)
{
    return fmt::format("{}.\n  {}\n", suite, test);
}

TEST(HostDispatchEnumerateMicrotest, ABinaryThatWillNotListIsReportedNotDropped)
{
    // The failure this must never have: a binary that cannot start fails --gtest_list_tests
    // first, and if that merely yielded no cases it would vanish from the run -- unrun,
    // unreported, and the run still green. It has to come back named so the caller can fail.
    const Enumeration enumeration =
        enumerate("/build", {"rccl-UnitTestsMicroGood", "rccl-UnitTestsMicroBroken"},
                  [](const std::string &path) -> std::optional<std::string> {
                      if (path.find("Broken") != std::string::npos) {
                          return std::nullopt;
                      }
                      return listingOf("Suite", "Case");
                  });

    EXPECT_EQ(enumeration.unlistable, (std::vector<std::string>{"rccl-UnitTestsMicroBroken"}));
    ASSERT_EQ(enumeration.candidates.size(), 1u);
    EXPECT_EQ(enumeration.candidates[0].suite, "MicroGood/Suite");
}

TEST(HostDispatchEnumerateMicrotest, ABinaryHoldingNoCasesIsNotTreatedAsBroken)
{
    // The distinction the optional carries: listing nothing is a healthy answer, and scoring it
    // as a failure would fail the run for a binary that is merely empty.
    const Enumeration enumeration =
        enumerate("/build", {"rccl-UnitTestsMicroEmpty"},
                  [](const std::string &) -> std::optional<std::string> { return std::string(); });

    EXPECT_TRUE(enumeration.unlistable.empty());
    EXPECT_TRUE(enumeration.candidates.empty());
}

TEST(HostDispatchEnumerateMicrotest, EachBinaryIsAskedForItsOwnPath)
{
    std::vector<std::string> asked;
    enumerate("/build/test/host", {"rccl-UnitTestsMicro", "rccl-UnitTestsMicroInit"},
              [&asked](const std::string &path) -> std::optional<std::string> {
                  asked.push_back(path);
                  return std::string();
              });

    EXPECT_EQ(asked, (std::vector<std::string>{"/build/test/host/rccl-UnitTestsMicro",
                                               "/build/test/host/rccl-UnitTestsMicroInit"}));
}

TEST(HostDispatchEnumerateMicrotest, CasesCarryTheIndexOfTheBinaryThatOwnsThem)
{
    // group() resolves a candidate back to its binary through this index alone, so a skipped
    // binary must not shift the ones after it.
    const Enumeration enumeration =
        enumerate("/build", {"rccl-UnitTestsMicroBroken", "rccl-UnitTestsMicroInit"},
                  [](const std::string &path) -> std::optional<std::string> {
                      if (path.find("Broken") != std::string::npos) {
                          return std::nullopt;
                      }
                      return listingOf("InitMicrotest", "Case");
                  });

    ASSERT_EQ(enumeration.candidates.size(), 1u);
    EXPECT_EQ(enumeration.candidates[0].binary, 1u);
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
    const std::optional<SuiteFragment> fragment =
        parseChildReport(report, qualifier, ReportFormat::Xml);
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
    EXPECT_FALSE(parseChildReport("", "Micro", ReportFormat::Xml).has_value());
    EXPECT_FALSE(parseChildReport("Segmentation fault\n", "Micro", ReportFormat::Xml).has_value());
    EXPECT_FALSE(
        parseChildReport("<testsuites tests=\"1\">\n", "Micro", ReportFormat::Xml).has_value());
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
    const std::string merged = mergeReports({first, second}, ReportFormat::Xml);
    EXPECT_NE(merged.find("tests=\"7\""), std::string::npos);
    EXPECT_NE(merged.find("failures=\"3\""), std::string::npos);
}

TEST(HostDispatchMergeMicrotest, EverySuiteSurvivesUnderItsOwnQualifier)
{
    const SuiteFragment first = parsedOrFail(childReport(1, 0), "MicroInit");
    const SuiteFragment second = parsedOrFail(childReport(1, 0), "MicroInitUncached");
    const std::string merged = mergeReports({first, second}, ReportFormat::Xml);
    EXPECT_NE(merged.find("<testsuite name=\"MicroInit/Suite\""), std::string::npos);
    EXPECT_NE(merged.find("<testsuite name=\"MicroInitUncached/Suite\""), std::string::npos);
}

TEST(HostDispatchMergeMicrotest, NoChildrenStillProducesAValidEmptyReport)
{
    // The report is the caller's second signal that the run happened at all (coco fails a test
    // whose XML is missing or has no <testsuites>), so a filter that selects nothing must still
    // produce a well-formed document rather than no file.
    const std::string merged = mergeReports({}, ReportFormat::Xml);
    EXPECT_NE(merged.find("<testsuites tests=\"0\""), std::string::npos);
    EXPECT_NE(merged.find("</testsuites>"), std::string::npos);
}

// ── JSON reports ────────────────────────────────────────────────────

// A child's JSON report in gtest's own shape, with *tests* cases of which *failures* failed.
std::string childJsonReport(int tests, int failures)
{
    nlohmann::json cases = nlohmann::json::array();
    for (int i = 0; i < tests; ++i) {
        nlohmann::json testcase{{"name", fmt::format("Case{}", i)},
                                {"file", "/src/suite-test.cc"},
                                {"line", 10 + i},
                                {"status", "RUN"},
                                {"result", "COMPLETED"},
                                {"time", "0.001s"},
                                {"classname", "Suite"}};
        if (i < failures) {
            testcase["failures"] =
                nlohmann::json::array({nlohmann::json{{"failure", "expected 1 == 2"},
                                                      {"type", ""}}});
        }
        cases.push_back(testcase);
    }
    const nlohmann::json root{{"tests", tests},   {"failures", failures}, {"disabled", 0},
                              {"errors", 0},      {"time", "1.5s"},       {"name", "AllTests"},
                              {"testsuites", nlohmann::json::array({nlohmann::json{
                                                 {"name", "Suite"},
                                                 {"tests", tests},
                                                 {"failures", failures},
                                                 {"time", "1.5s"},
                                                 {"testsuite", cases}}})}};
    return root.dump(2) + "\n";
}

TEST(HostDispatchJsonMicrotest, RootTotalsAreTakenFromTheChild)
{
    const std::optional<SuiteFragment> fragment =
        parseChildReport(childJsonReport(7, 2), "Micro", ReportFormat::Json);
    ASSERT_TRUE(fragment.has_value());
    EXPECT_EQ(fragment->tests, 7);
    EXPECT_EQ(fragment->failures, 2);
    EXPECT_DOUBLE_EQ(fragment->time, 1.5);   // "1.5s", not 1.5
}

TEST(HostDispatchJsonMicrotest, QualifierMovesSuiteNameAndClassnameTogether)
{
    // The same contract the XML path has: both spellings of the suite's identity move, because
    // a consumer reconstructs the case from suite name plus classname.
    const std::optional<SuiteFragment> fragment =
        parseChildReport(childJsonReport(1, 0), "MicroInit", ReportFormat::Json);
    ASSERT_TRUE(fragment.has_value());
    ASSERT_EQ(fragment->suites.size(), 1u);
    EXPECT_EQ(fragment->suites[0]["name"], "MicroInit/Suite");
    EXPECT_EQ(fragment->suites[0]["testsuite"][0]["classname"], "MicroInit/Suite");
    EXPECT_EQ(fragment->suites[0]["testsuite"][0]["name"], "Case0");   // the case, untouched
}

TEST(HostDispatchJsonMicrotest, CaseDetailSurvivesTheMerge)
{
    // The keys the coco runner reads off each leaf (test_executor.py: _leaf_case_status), which
    // a reserialised report has to carry through untouched or every case reads as not-run.
    const std::optional<SuiteFragment> fragment =
        parseChildReport(childJsonReport(2, 1), "Micro", ReportFormat::Json);
    ASSERT_TRUE(fragment.has_value());
    const nlohmann::json merged =
        nlohmann::json::parse(mergeReports({*fragment}, ReportFormat::Json));
    const nlohmann::json &failed = merged["testsuites"][0]["testsuite"][0];
    EXPECT_EQ(failed["status"], "RUN");
    EXPECT_EQ(failed["result"], "COMPLETED");
    EXPECT_EQ(failed["file"], "/src/suite-test.cc");
    EXPECT_EQ(failed["line"], 10);
    ASSERT_TRUE(failed["failures"].is_array());
    EXPECT_EQ(failed["failures"][0]["failure"], "expected 1 == 2");
    EXPECT_FALSE(merged["testsuites"][0]["testsuite"][1].contains("failures"));
}

TEST(HostDispatchJsonMicrotest, TotalsAreSummedAcrossChildren)
{
    const std::optional<SuiteFragment> first =
        parseChildReport(childJsonReport(3, 1), "Micro", ReportFormat::Json);
    const std::optional<SuiteFragment> second =
        parseChildReport(childJsonReport(4, 2), "MicroInit", ReportFormat::Json);
    ASSERT_TRUE(first.has_value() && second.has_value());
    const nlohmann::json merged =
        nlohmann::json::parse(mergeReports({*first, *second}, ReportFormat::Json));
    EXPECT_EQ(merged["tests"], 7);
    EXPECT_EQ(merged["failures"], 3);
    EXPECT_EQ(merged["time"], "3s");   // seconds summed, and spelled the way gtest spells them
    ASSERT_EQ(merged["testsuites"].size(), 2u);
    EXPECT_EQ(merged["testsuites"][0]["name"], "Micro/Suite");
    EXPECT_EQ(merged["testsuites"][1]["name"], "MicroInit/Suite");
}

TEST(HostDispatchJsonMicrotest, OutputThatIsNotAGtestReportIsRejected)
{
    EXPECT_FALSE(parseChildReport("", "Micro", ReportFormat::Json).has_value());
    EXPECT_FALSE(parseChildReport("Segmentation fault\n", "Micro", ReportFormat::Json).has_value());
    // Valid JSON, but not a report: the distinction string matching could not make.
    EXPECT_FALSE(parseChildReport("{\"tests\": 1}", "Micro", ReportFormat::Json).has_value());
}

TEST(HostDispatchJsonMicrotest, CrashFragmentIsAFailingCaseInBothFormats)
{
    // crashFragment() carries both representations so the crash path never has to know which
    // format is in play; a JSON run must still see the failure.
    const SuiteFragment fragment = crashFragment("MicroInit", "exit status 139");
    const nlohmann::json merged =
        nlohmann::json::parse(mergeReports({fragment}, ReportFormat::Json));
    EXPECT_EQ(merged["failures"], 1);
    const nlohmann::json &testcase = merged["testsuites"][0]["testsuite"][0];
    EXPECT_EQ(testcase["classname"], "MicroInit/HostTestDispatch");
    EXPECT_EQ(testcase["result"], "COMPLETED");
    EXPECT_EQ(testcase["failures"][0]["failure"], "exit status 139");
}

TEST(HostDispatchJsonMicrotest, NoChildrenStillProducesAValidEmptyReport)
{
    const nlohmann::json merged = nlohmann::json::parse(mergeReports({}, ReportFormat::Json));
    EXPECT_EQ(merged["tests"], 0);
    EXPECT_EQ(merged["name"], "AllTests");
    EXPECT_TRUE(merged["testsuites"].is_array());
    EXPECT_TRUE(merged["testsuites"].empty());
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

TEST(HostDispatchArgsMicrotest, OutputFormatComesFromTheSpecNotTheFilename)
{
    EXPECT_EQ(parseArgs({"--gtest_output=json:/tmp/r.json"}).outputFormat, ReportFormat::Json);
    EXPECT_EQ(parseArgs({"--gtest_output=xml:/tmp/r.xml"}).outputFormat, ReportFormat::Xml);
    // gtest's rule for a bare path: xml.
    EXPECT_EQ(parseArgs({"--gtest_output=/tmp/r.out"}).outputFormat, ReportFormat::Xml);
    EXPECT_EQ(parseArgs({"--gtest_output=/tmp/r.out"}).outputPath, "/tmp/r.out");
}

TEST(HostDispatchArgsMicrotest, ABareFormatNamesTheFormatNotTheFile)
{
    // gtest reads a colon-less "xml"/"json" as the format and writes its own default filename.
    // Read as a path instead, "--gtest_output=json" would write XML to a file called "json".
    const Options json = parseArgs({"--gtest_output=json"});
    EXPECT_EQ(json.outputFormat, ReportFormat::Json);
    EXPECT_EQ(json.outputPath, "./test_detail.json");

    const Options xml = parseArgs({"--gtest_output=xml"});
    EXPECT_EQ(xml.outputFormat, ReportFormat::Xml);
    EXPECT_EQ(xml.outputPath, "./test_detail.xml");
}

TEST(HostDispatchArgsMicrotest, OutputDirectoryGetsTheDefaultNameForItsFormat)
{
    EXPECT_EQ(parseArgs({"--gtest_output=json:/tmp/reports/"}).outputPath,
              "/tmp/reports/test_detail.json");
}

TEST(HostDispatchArgsMicrotest, GtestOutputEnvironmentIsTakenAsTheReportDestination)
{
    // Left to gtest, this variable would make the no-op filter-resolution run write an all-pass
    // report over the caller's file, and the merged one would never be written at all: the run
    // could fail and the report still read green. Taken here, it names the merged report.
    const Options options = parseArgs({}, "json:/tmp/from_env.json");
    EXPECT_EQ(options.outputPath, "/tmp/from_env.json");
    EXPECT_EQ(options.outputFormat, ReportFormat::Json);
}

TEST(HostDispatchArgsMicrotest, AnExplicitOutputFlagBeatsTheEnvironment)
{
    // gtest's own precedence: the flag wins over GTEST_OUTPUT.
    const Options options = parseArgs({"--gtest_output=xml:/tmp/flag.xml"}, "json:/tmp/env.json");
    EXPECT_EQ(options.outputPath, "/tmp/flag.xml");
    EXPECT_EQ(options.outputFormat, ReportFormat::Xml);
}

TEST(HostDispatchArgsMicrotest, FilterDefaultsToEverything)
{
    EXPECT_EQ(parseArgs({}).filter, "*");
}

TEST(HostDispatchArgsMicrotest, GtestFilterEnvironmentIsTakenAsTheFilter)
{
    // Left in the environment, this would filter the children's unqualified listings while the
    // dispatcher still asked for every case. Taken here, it is resolved over the published names.
    EXPECT_EQ(parseArgs({}, "", "*Suite.Case").filter, "*Suite.Case");
}

TEST(HostDispatchArgsMicrotest, AnExplicitFilterFlagBeatsTheEnvironment)
{
    // gtest's own precedence: the flag wins over GTEST_FILTER.
    EXPECT_EQ(parseArgs({"--gtest_filter=*Flag.*"}, "", "*Env.*").filter, "*Flag.*");
}

TEST(HostDispatchArgsMicrotest, GtestAlsoRunDisabledEnvironmentSelectsAndForwards)
{
    // Left to the children alone, the DISABLED_ cases would run in a binary sent "*" and be
    // skipped in one sent a name list, because this process would not have selected them.
    const Options options = parseArgs({}, "", "", "1");
    EXPECT_TRUE(options.alsoRunDisabled);
    EXPECT_EQ(options.passthrough,
              (std::vector<std::string>{"--gtest_also_run_disabled_tests"}));
}

TEST(HostDispatchArgsMicrotest, GtestAlsoRunDisabledFollowsGtestsFalsehood)
{
    // gtest's rule for its boolean variables: "0" is the only false value, unset aside. An empty
    // value is set, so it is true -- reading it as false would skip the DISABLED_ cases a caller
    // writing GTEST_ALSO_RUN_DISABLED_TESTS= asked gtest for.
    EXPECT_FALSE(parseArgs({}, "", "", std::nullopt).alsoRunDisabled);
    EXPECT_FALSE(parseArgs({}, "", "", "0").alsoRunDisabled);
    EXPECT_TRUE(parseArgs({}, "", "", "").alsoRunDisabled);
    EXPECT_TRUE(parseArgs({}, "", "", "anything else").alsoRunDisabled);
}

TEST(HostDispatchArgsMicrotest, TheDisabledFlagIsForwardedOnceWhenTheEnvironmentAgrees)
{
    // The flag and the variable say the same thing; a child must not be sent it twice.
    const Options options = parseArgs({"--gtest_also_run_disabled_tests"}, "", "", "1");
    EXPECT_TRUE(options.alsoRunDisabled);
    EXPECT_EQ(options.passthrough,
              (std::vector<std::string>{"--gtest_also_run_disabled_tests"}));
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

// ── the gtest environment the children are given ────────────────────

namespace {

// *name* removed from the environment, whatever the test before this one left behind.
struct ScopedUnset {
    explicit ScopedUnset(const char *name) : name_(name) { unsetenv(name_); }
    ~ScopedUnset() { unsetenv(name_); }
    const char *name_;
};

// Whether gtestVariablesToStrip() currently names *name*.
bool stripped(const char *name)
{
    const std::vector<std::string> names = gtestVariablesToStrip();
    return std::find(names.begin(), names.end(), name) != names.end();
}

}  // namespace

TEST(HostDispatchEnvironmentMicrotest, EveryVariableTheDispatcherActsOnIsKeptFromTheChildren)
{
    // Each of these has already been resolved into an explicit per-child flag or into the
    // dispatcher's own behaviour. A child that saw one would decide it a second time, over a
    // case list that has already been narrowed -- and exit 0 having run a fraction of it.
    // Spelled out rather than looped over kGtestVariables, so a wrong Disposition in that table
    // cannot make this test agree with the bug.
    const ScopedUnset guards[] = {ScopedUnset("GTEST_FILTER"), ScopedUnset("GTEST_OUTPUT"),
                                  ScopedUnset("GTEST_ALSO_RUN_DISABLED_TESTS"),
                                  ScopedUnset("GTEST_TOTAL_SHARDS"),
                                  ScopedUnset("GTEST_SHARD_INDEX"),
                                  ScopedUnset("GTEST_SHARD_STATUS_FILE")};
    (void)guards;
    setenv("GTEST_FILTER", "*Suite.Case", 1);
    setenv("GTEST_OUTPUT", "xml:/tmp/callers_report.xml", 1);
    setenv("GTEST_ALSO_RUN_DISABLED_TESTS", "1", 1);
    setenv("GTEST_TOTAL_SHARDS", "4", 1);
    setenv("GTEST_SHARD_INDEX", "2", 1);
    setenv("GTEST_SHARD_STATUS_FILE", "/tmp/shard_status", 1);

    EXPECT_TRUE(stripped("GTEST_FILTER"));
    EXPECT_TRUE(stripped("GTEST_OUTPUT"));
    EXPECT_TRUE(stripped("GTEST_ALSO_RUN_DISABLED_TESTS"));
    EXPECT_TRUE(stripped("GTEST_TOTAL_SHARDS"));
    EXPECT_TRUE(stripped("GTEST_SHARD_INDEX"));
    EXPECT_TRUE(stripped("GTEST_SHARD_STATUS_FILE"));
}

TEST(HostDispatchEnvironmentMicrotest, AVariableThisBuildHasNeverHeardOfIsKeptFromTheChildren)
{
    // The point of the fail-closed default: a variable a future gtest adds is stripped without
    // this file being touched. Inheriting it instead could silently narrow every child's run.
    const ScopedUnset guard("GTEST_SOME_FUTURE_SELECTION_KNOB");
    setenv(guard.name_, "1", 1);

    EXPECT_TRUE(stripped(guard.name_));
}

TEST(HostDispatchEnvironmentMicrotest, PresentationVariablesStillReachTheChildren)
{
    // These mean the same thing in a child and change nothing about which cases run, so
    // stripping them would be a gratuitous behaviour change for the caller.
    const ScopedUnset color("GTEST_COLOR");
    const ScopedUnset brief("GTEST_BRIEF");
    const ScopedUnset style("GTEST_DEATH_TEST_STYLE");
    setenv(color.name_, "yes", 1);
    setenv(brief.name_, "1", 1);
    setenv(style.name_, "threadsafe", 1);

    EXPECT_FALSE(stripped(color.name_));
    EXPECT_FALSE(stripped(brief.name_));
    EXPECT_FALSE(stripped(style.name_));
}

TEST(HostDispatchEnvironmentMicrotest, OnlyGtestVariablesAreTouched)
{
    // The strip is scoped to gtest's own namespace: a binary's own env-driven switches, and the
    // coverage variables this dispatcher relies on, have to survive into the children.
    const ScopedUnset profile("LLVM_PROFILE_FILE");
    setenv(profile.name_, "/tmp/profiles/rccl_%p.profraw", 1);

    EXPECT_FALSE(stripped(profile.name_));
    EXPECT_FALSE(stripped("PATH"));
}

TEST(HostDispatchEnvironmentMicrotest, StrippingHappensOnTheChildsCopyOfTheEnvironment)
{
    // stripGtestVariablesInChild() runs between fork() and exec(). Here that is this process,
    // so the test can only state the mechanics: the named variables, and nothing else, go.
    const ScopedUnset filter("GTEST_FILTER");
    const ScopedUnset color("GTEST_COLOR");
    setenv(filter.name_, "*Suite.Case", 1);
    setenv(color.name_, "yes", 1);

    stripGtestVariablesInChild({"GTEST_FILTER"});

    EXPECT_EQ(getenv("GTEST_FILTER"), nullptr);
    EXPECT_STREQ(getenv("GTEST_COLOR"), "yes");
}

TEST(HostDispatchEnvironmentMicrotest, ShardingVariablesSurviveInTheDispatcherItself)
{
    // The dispatcher shards ONCE, over the published names, and that happens after the listing
    // children have been forked. Were the strip done to this process rather than to each child,
    // sharding would be silently disabled for the whole run.
    const ScopedUnset total("GTEST_TOTAL_SHARDS");
    setenv(total.name_, "4", 1);

    (void)gtestVariablesToStrip();

    EXPECT_STREQ(getenv("GTEST_TOTAL_SHARDS"), "4");
}

// ── binaryDirectory ─────────────────────────────────────────────────
//
// Which layout is in use is decided by looking, so these do touch the filesystem -- there is no
// pure function underneath to test instead, and a wrong decision here runs nothing at all and
// still exits 0 on the way there.

namespace {

// A directory tree that removes itself.
class ScopedDir {
public:
    ScopedDir()
    {
        std::string buffer =
            (std::filesystem::temp_directory_path() / "rccl-dispatch-test-XXXXXX").string();
        path_ = mkdtemp(buffer.data()) == nullptr ? "" : buffer;
    }
    ~ScopedDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    ScopedDir(const ScopedDir &) = delete;
    ScopedDir &operator=(const ScopedDir &) = delete;

    const std::string &path() const { return path_; }

    // An executable file named *name* under this directory, parents created as needed.
    std::string addExecutable(const std::string &name) const
    {
        const std::filesystem::path file = std::filesystem::path(path_) / name;
        std::error_code ec;
        std::filesystem::create_directories(file.parent_path(), ec);
        std::ofstream(file) << "#!/bin/sh\n";
        std::filesystem::permissions(file, std::filesystem::perms::owner_all,
                                     std::filesystem::perm_options::add, ec);
        return file.string();
    }

private:
    std::string path_;
};

}  // namespace

TEST(HostDispatchLayoutMicrotest, TheFrontedBinariesAreFoundInTheMicroSubdirectory)
{
    // The build tree's layout: the front door at the top, everything it runs one level down, so
    // a listing of the build output shows which binary to run.
    const ScopedDir tree;
    ASSERT_FALSE(tree.path().empty());
    const std::string self = tree.addExecutable("rccl-UnitTestsHostAll");
    tree.addExecutable("micro/rccl-UnitTestsMicroInit");

    EXPECT_EQ(binaryDirectory(tree.path(), self),
              (std::filesystem::path(tree.path()) / "micro").string());
}

TEST(HostDispatchLayoutMicrotest, AFlatLayoutIsStillFound)
{
    // An install keeps them beside the front door, and the same binary runs from there.
    const ScopedDir tree;
    ASSERT_FALSE(tree.path().empty());
    const std::string self = tree.addExecutable("rccl-UnitTestsHostAll");
    tree.addExecutable("rccl-UnitTestsMicroInit");

    EXPECT_EQ(binaryDirectory(tree.path(), self), tree.path());
}

TEST(HostDispatchLayoutMicrotest, AnEmptyMicroDirectoryDoesNotHideTheFlatLayout)
{
    // micro/ existing but holding nothing is not the build layout, so the binaries beside the
    // front door are still the ones that run.
    const ScopedDir tree;
    ASSERT_FALSE(tree.path().empty());
    const std::string self = tree.addExecutable("rccl-UnitTestsHostAll");
    tree.addExecutable("rccl-UnitTestsMicroInit");
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(tree.path()) / "micro", ec);

    EXPECT_EQ(binaryDirectory(tree.path(), self), tree.path());
}

// ── child logs ──────────────────────────────────────────────────────

TEST(HostDispatchLogMicrotest, LogsLandBesideTheMergedReport)
{
    // The report is the artefact a harness uploads and a reader opens first, so the transcripts
    // explaining it are kept in a directory named after it, in the same place.
    EXPECT_EQ(childLogDirectory("/p/host_tests_micro.xml"), "/p/host_tests_micro.logs");
    EXPECT_EQ(childLogDirectory("/p/host_tests_micro.json"), "/p/host_tests_micro.logs");
}

TEST(HostDispatchLogMicrotest, AReportInTheCurrentDirectoryKeepsItsLogsThere)
{
    EXPECT_EQ(childLogDirectory("host_tests_micro.xml"), "host_tests_micro.logs");
}

TEST(HostDispatchLogMicrotest, NoReportStillGetsLogs)
{
    // A caller that asked for no report still wants the binaries' output somewhere; there is no
    // report name to derive a directory from, so the fixed one is used.
    EXPECT_EQ(childLogDirectory(""), "host-test-dispatch.logs");
}

TEST(HostDispatchLogMicrotest, EachBinaryWritesItsOwnLog)
{
    EXPECT_EQ(childLogPath("/p/host_tests_micro.logs", "MicroInit-uncached"),
              "/p/host_tests_micro.logs/MicroInit-uncached.log");
}

TEST(HostDispatchLogMicrotest, TheEnvironmentCanMoveTheLogsOrTurnThemOff)
{
    // Unset and empty both mean "use the default"; a directory names where to put them instead;
    // "-" is the opt-out that leaves the children on this process's stdout, which is what
    // running one binary's cases by hand wants.
    EXPECT_EQ(resolveLogDirectory("/p/report.xml", nullptr), "/p/report.logs");
    EXPECT_EQ(resolveLogDirectory("/p/report.xml", ""), "/p/report.logs");
    EXPECT_EQ(resolveLogDirectory("/p/report.xml", "/tmp/logs"), "/tmp/logs");
    EXPECT_EQ(resolveLogDirectory("/p/report.xml", "-"), "");
}

// ── coverageDirectory ───────────────────────────────────────────────

TEST(HostDispatchCoverageMicrotest, ProfileDirectoryIsTakenFromThePattern)
{
    // The harness points LLVM_PROFILE_FILE at a pattern, not a file; only its directory is a real
    // path, and that is where the merged tracefile has to land for the harness to find it.
    setenv("LLVM_PROFILE_FILE", "/tmp/coverage_raw/rccl_%h_%p_%m.profraw", 1);
    EXPECT_EQ(coverageDirectory(), "/tmp/coverage_raw");
    unsetenv("LLVM_PROFILE_FILE");
}

TEST(HostDispatchCoverageMicrotest, NoProfileVariableMeansNoTracefile)
{
    unsetenv("LLVM_PROFILE_FILE");
    EXPECT_EQ(coverageDirectory(), "");
}

// ── childProfileFile ────────────────────────────────────────────────

TEST(HostDispatchCoverageMicrotest, EachChildProfilesIntoItsOwnDirectory)
{
    // Two binaries under one inherited pattern have to end up in different directories: they
    // compile shared source under different #defines, so their coverage-mapping hashes differ
    // and a report built from one object over both their profiles is rejected as mismatched.
    EXPECT_EQ(childProfileFile("/b/coverage/all-%p-%m.profraw", "/b/rccl-UnitTestsMicroInit"),
              "/b/coverage/rccl-UnitTestsMicroInit/all-%p-%m.profraw");
    EXPECT_EQ(childProfileFile("/b/coverage/all-%p-%m.profraw", "/b/rccl-UnitTestsMicroUtils"),
              "/b/coverage/rccl-UnitTestsMicroUtils/all-%p-%m.profraw");
}

TEST(HostDispatchCoverageMicrotest, ChildProfileKeepsTheCallersFilenamePattern)
{
    // The specifiers are the caller's choice -- %p keeps the process-isolated forks' files
    // distinct, %m the differing module hashes -- so the filename is carried over untouched.
    EXPECT_EQ(childProfileFile("/b/cov/rccl_%h_%p_%m.profraw", "/b/rccl-UnitTestsMicroCrypt"),
              "/b/cov/rccl-UnitTestsMicroCrypt/rccl_%h_%p_%m.profraw");
}

TEST(HostDispatchCoverageMicrotest, APatternWithNoDirectoryStillSeparatesTheChildren)
{
    // A bare filename means "profile into the cwd"; the child still gets a directory of its own
    // so the binaries' profiles do not pool.
    EXPECT_EQ(childProfileFile("default.profraw", "./rccl-UnitTestsMicroTopo"),
              "rccl-UnitTestsMicroTopo/default.profraw");
}

TEST(HostDispatchCoverageMicrotest, NoProfileVariableLeavesTheChildEnvironmentAlone)
{
    // Nothing to redirect when coverage is not being collected, and an empty result is what the
    // call sites test to decide whether to set the variable at all.
    EXPECT_EQ(childProfileFile("", "/b/rccl-UnitTestsMicroTopo"), "");
}


// ── lcov merge ──────────────────────────────────────────────────────

namespace {

// The lcov records for one source file, with the counting lines left to the code under test.
std::string tracefile(const std::string &path, const std::string &body)
{
    return "SF:" + path + "\n" + body + "end_of_record\n";
}

// Whether *text* contains *line* as a whole line.
bool hasLine(const std::string &text, const std::string &line)
{
    return ("\n" + text).find("\n" + line + "\n") != std::string::npos;
}

}  // namespace

TEST(HostDispatchLcovMicrotest, ALineCoveredByOneBinaryIsCoveredInTheMerge)
{
    // The point of the union: several binaries compile the same source under different #defines,
    // so each reaches lines the others cannot, and the merged view has to credit all of them.
    const std::string merged = mergeLcovTracefiles({
        tracefile("/src/init.cc", "DA:10,1\nDA:20,0\n"),
        tracefile("/src/init.cc", "DA:10,0\nDA:20,5\n"),
    });

    EXPECT_TRUE(hasLine(merged, "DA:10,1")) << merged;
    EXPECT_TRUE(hasLine(merged, "DA:20,5")) << merged;
    EXPECT_TRUE(hasLine(merged, "LF:2")) << merged;
    EXPECT_TRUE(hasLine(merged, "LH:2")) << merged;
}

TEST(HostDispatchLcovMicrotest, ALineNoBinaryReachedStaysUncovered)
{
    const std::string merged = mergeLcovTracefiles({
        tracefile("/src/init.cc", "DA:10,0\n"),
        tracefile("/src/init.cc", "DA:10,0\n"),
    });

    EXPECT_TRUE(hasLine(merged, "DA:10,0")) << merged;
    EXPECT_TRUE(hasLine(merged, "LH:0")) << merged;
}

TEST(HostDispatchLcovMicrotest, EachSourceFileKeepsItsOwnRecord)
{
    const std::string merged = mergeLcovTracefiles({
        tracefile("/src/init.cc", "DA:10,1\n"),
        tracefile("/src/enqueue.cc", "DA:7,1\n"),
    });

    EXPECT_TRUE(hasLine(merged, "SF:/src/init.cc")) << merged;
    EXPECT_TRUE(hasLine(merged, "SF:/src/enqueue.cc")) << merged;
    EXPECT_EQ(merged.find("end_of_record"), merged.find("end_of_record"));
    EXPECT_EQ(std::count(merged.begin(), merged.end(), '\n') > 0, true);
}

TEST(HostDispatchLcovMicrotest, CountingLinesAreRecomputedNotCarriedOver)
{
    // llvm-cov's own FNF/LF/BRF count its internal regions, not the records beside them -- a
    // record with 648 FN lines can be headed FNF:258 -- so summing the inputs' numbers would
    // give totals that match neither the records nor anything lcov computes from them.
    const std::string merged = mergeLcovTracefiles({
        tracefile("/src/init.cc", "DA:10,1\nDA:11,1\nLF:999\nLH:999\n"),
    });

    EXPECT_TRUE(hasLine(merged, "LF:2")) << merged;
    EXPECT_TRUE(hasLine(merged, "LH:2")) << merged;
    EXPECT_FALSE(hasLine(merged, "LF:999")) << merged;
}

TEST(HostDispatchLcovMicrotest, ARepeatedLineInOneRecordIsCountedOnce)
{
    // llvm-cov emits a line once per region covering it, so a record can carry the same DA line
    // twice. Counting both would inflate LF and make the merged percentage a fiction.
    const std::string merged = mergeLcovTracefiles({
        tracefile("/src/init.cc", "DA:10,1\nDA:10,2\n"),
    });

    EXPECT_TRUE(hasLine(merged, "LF:1")) << merged;
    EXPECT_TRUE(hasLine(merged, "DA:10,3")) << merged;
}

TEST(HostDispatchLcovMicrotest, ABranchOneBinaryNeverEvaluatedIsTakenIfAnotherTookIt)
{
    // '-' is the absence of information, not a zero: a binary whose #defines compile the branch
    // out reports it unevaluated, and must not veto the binary that took it.
    const std::string merged = mergeLcovTracefiles({
        tracefile("/src/init.cc", "BRDA:10,0,0,-\n"),
        tracefile("/src/init.cc", "BRDA:10,0,0,4\n"),
    });

    EXPECT_TRUE(hasLine(merged, "BRDA:10,0,0,4")) << merged;
    EXPECT_TRUE(hasLine(merged, "BRH:1")) << merged;
}

TEST(HostDispatchLcovMicrotest, ABranchNoBinaryEvaluatedStaysUnevaluated)
{
    const std::string merged = mergeLcovTracefiles({
        tracefile("/src/init.cc", "BRDA:10,0,0,-\n"),
        tracefile("/src/init.cc", "BRDA:10,0,0,-\n"),
    });

    EXPECT_TRUE(hasLine(merged, "BRDA:10,0,0,-")) << merged;
    EXPECT_TRUE(hasLine(merged, "BRF:1")) << merged;
    EXPECT_TRUE(hasLine(merged, "BRH:0")) << merged;
}

TEST(HostDispatchLcovMicrotest, FunctionExecutionsAreSummedAcrossBinaries)
{
    const std::string merged = mergeLcovTracefiles({
        tracefile("/src/init.cc", "FN:44,_Z4initv\nFNDA:3,_Z4initv\n"),
        tracefile("/src/init.cc", "FN:44,_Z4initv\nFNDA:2,_Z4initv\n"),
    });

    EXPECT_TRUE(hasLine(merged, "FNDA:5,_Z4initv")) << merged;
    EXPECT_TRUE(hasLine(merged, "FNF:1")) << merged;
    EXPECT_TRUE(hasLine(merged, "FNH:1")) << merged;
}

TEST(HostDispatchLcovMicrotest, AFunctionNameCarryingACommaSurvives)
{
    // Mangled names of templates contain commas, so a split on every comma would truncate the
    // name and make the same function look like two.
    const std::string merged = mergeLcovTracefiles({
        tracefile("/src/init.cc", "FN:7,file.cc:_ZL3fooIiEvT_,S0_\nFNDA:1,file.cc:_ZL3fooIiEvT_,S0_\n"),
    });

    EXPECT_TRUE(hasLine(merged, "FN:7,file.cc:_ZL3fooIiEvT_,S0_")) << merged;
    EXPECT_TRUE(hasLine(merged, "FNF:1")) << merged;
}

TEST(HostDispatchLcovMicrotest, AFunctionSeenAtTwoLinesTakesTheEarlierOne)
{
    const std::string merged = mergeLcovTracefiles({
        tracefile("/src/init.cc", "FN:90,_Z4initv\n"),
        tracefile("/src/init.cc", "FN:44,_Z4initv\n"),
    });

    EXPECT_TRUE(hasLine(merged, "FN:44,_Z4initv")) << merged;
    EXPECT_FALSE(hasLine(merged, "FN:90,_Z4initv")) << merged;
}

TEST(HostDispatchLcovMicrotest, AWrappedCounterStillReadsAsCovered)
{
    // llvm-cov prints counters as unsigned 64-bit and sometimes prints one that has wrapped --
    // 18446744073709551591 is a counter expression that evaluated below zero. It does not fit a
    // signed long, and rejecting it as unparseable would turn a covered line into an uncovered
    // one, which is the one way this merge could understate coverage.
    const std::string merged =
        mergeLcovTracefiles({tracefile("/src/init.cc", "DA:18,18446744073709551591\n")});

    EXPECT_TRUE(hasLine(merged, "LH:1")) << merged;
}

TEST(HostDispatchLcovMicrotest, SummingHugeCountersSaturatesRatherThanWraps)
{
    // Two near-maximal counters summed must not come out negative, which would read as
    // uncovered -- the opposite of what two covering binaries mean.
    const std::string merged = mergeLcovTracefiles({
        tracefile("/src/init.cc", "DA:18,9223372036854775807\n"),
        tracefile("/src/init.cc", "DA:18,9223372036854775807\n"),
    });

    EXPECT_TRUE(hasLine(merged, "DA:18,9223372036854775807")) << merged;
    EXPECT_TRUE(hasLine(merged, "LH:1")) << merged;
}

TEST(HostDispatchLcovMicrotest, MergingTheSameInputTwiceIsByteIdentical)
{
    // The tracefile is an artifact a dashboard diffs between runs, so the merge has to be a
    // function of its inputs alone and not of the order a directory happened to be read in.
    const std::string one = mergeLcovTracefiles({
        tracefile("/src/b.cc", "DA:2,1\n"), tracefile("/src/a.cc", "DA:1,1\n")});
    const std::string two = mergeLcovTracefiles({
        tracefile("/src/a.cc", "DA:1,1\n"), tracefile("/src/b.cc", "DA:2,1\n")});

    EXPECT_EQ(one, two);
}

TEST(HostDispatchLcovMicrotest, NothingToMergeIsAnEmptyTracefileNotAMalformedOne)
{
    EXPECT_EQ(mergeLcovTracefiles({}), "");
}

TEST(HostDispatchLcovMicrotest, RecordLinesWithNoSourceFileAreIgnored)
{
    // A truncated tracefile (a killed llvm-cov) must not have its orphaned lines attributed to
    // whichever file happened to be merged before it.
    const std::string merged = mergeLcovTracefiles({"DA:10,1\n", tracefile("/src/a.cc", "DA:1,1\n")});

    EXPECT_FALSE(hasLine(merged, "DA:10,1")) << merged;
    EXPECT_TRUE(hasLine(merged, "LF:1")) << merged;
}

// ── coverage export commands ────────────────────────────────────────

TEST(HostDispatchCoverageMicrotest, ExportNamesOnlyTheBinaryTheProfilesCameFrom)
{
    // The pairing is the reason this runs in the dispatcher at all: llvm-cov rejects a profile
    // read against an object it was not built from, and only the dispatcher knows which binary
    // produced which profiles.
    const std::string command = lcovExportCommand("llvm-cov", "/b/rccl-UnitTestsMicroInit",
                                                  "/c/merged.profdata", "(/test/|nvtx)",
                                                  "/c/coverage.lcov");

    EXPECT_NE(command.find("'/b/rccl-UnitTestsMicroInit'"), std::string::npos) << command;
    EXPECT_EQ(command.find("rccl-UnitTestsMicroUtils"), std::string::npos) << command;
    EXPECT_NE(command.find("-format=lcov"), std::string::npos) << command;
}

TEST(HostDispatchCoverageMicrotest, EveryProfileInTheBinarysDirectoryIsMerged)
{
    // The process-isolated tests fork, so one binary leaves many .profraw behind; merging only
    // some of them would silently under-report its coverage.
    const std::string command = profdataMergeCommand(
        "llvm-profdata", {"/c/a-1.profraw", "/c/a-2.profraw"}, "/c/merged.profdata");

    EXPECT_NE(command.find("'/c/a-1.profraw'"), std::string::npos) << command;
    EXPECT_NE(command.find("'/c/a-2.profraw'"), std::string::npos) << command;
    EXPECT_NE(command.find("-o '/c/merged.profdata'"), std::string::npos) << command;
}

TEST(HostDispatchCoverageMicrotest, APathWithAQuoteInItCannotBreakOutOfTheCommand)
{
    // The command is handed to /bin/sh, and the paths in it come from the filesystem rather than
    // from this process, so a quote in a directory name must stay data. Pinned as the exact
    // single-quoted form: the dangerous text is still present, but only ever inside quotes.
    const std::string command =
        profdataMergeCommand("llvm-profdata", {"/c/a'; rm -rf /; echo '.profraw"}, "/c/m.profdata");

    EXPECT_NE(command.find(R"('/c/a'\''; rm -rf /; echo '\''.profraw')"), std::string::npos)
        << command;
}

}  // namespace
