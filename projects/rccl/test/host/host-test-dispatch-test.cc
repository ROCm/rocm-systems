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
#include <chrono>
#include <csignal>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "LogCapture.hpp"
#include "ScopedDir.h"

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
        group("/dir", {"rccl-UnitTestsMicro"}, oneEnabledOneDisabled(), {{0}, {}}, false);
    ASSERT_EQ(selections.size(), 1u);
    EXPECT_EQ(selections[0].runnable, 1u);
    EXPECT_EQ(childFilters(selections[0]), (std::vector<std::string>{"*"}));
}

TEST(HostDispatchGroupMicrotest, DisabledCasesCountWhenTheCallerAsksForThem)
{
    const std::vector<Selection> selections =
        group("/dir", {"rccl-UnitTestsMicro"}, oneEnabledOneDisabled(), {{0}, {}}, true);
    ASSERT_EQ(selections.size(), 1u);
    EXPECT_EQ(selections[0].runnable, 2u);
    // One of two selected is a real partial selection now, so the case is named.
    EXPECT_EQ(childFilters(selections[0]), (std::vector<std::string>{"Suite.Enabled"}));
}

TEST(HostDispatchGroupMicrotest, APartialSelectionStillNamesTheDisabledCasesTheFilterMatched)
{
    // A whole-binary selection sends "*" and gtest reports the binary's DISABLED_ cases as
    // suppressed. A partial one sends an explicit list, and without the suppressed names in it
    // the merged report has no entry for them at all -- while --gtest_list_tests goes on
    // listing them, so a consumer comparing the two sees cases that vanished. Named here, gtest
    // reports them from the child exactly as the "*" path does.
    const std::vector<CandidateCase> candidates = {
        CandidateCase{0, "Micro/Suite", TestCaseName{"Suite", "Enabled"}},
        CandidateCase{0, "Micro/Suite", TestCaseName{"Suite", "Other"}},
        CandidateCase{0, "Micro/Suite", TestCaseName{"Suite", "DISABLED_Skipped"}}};
    const std::vector<Selection> selections =
        group("/dir", {"rccl-UnitTestsMicro"}, candidates, {{0}, {2}}, false);

    ASSERT_EQ(selections.size(), 1u);
    EXPECT_EQ(childFilters(selections[0]),
              (std::vector<std::string>{"Suite.Enabled:Suite.DISABLED_Skipped"}));
}

TEST(HostDispatchGroupMicrotest, SuppressedCasesDoNotMakeAWholeBinaryLookPartial)
{
    // The "*" shortcut turns on every runnable case being selected. Counting the suppressed
    // ones into that would tip the biggest binary into being asked for its 2000-odd cases by
    // name, past the exec argument limit, to run exactly what "*" runs.
    const std::vector<Selection> selections =
        group("/dir", {"rccl-UnitTestsMicro"}, oneEnabledOneDisabled(), {{0}, {1}}, false);

    ASSERT_EQ(selections.size(), 1u);
    EXPECT_EQ(selections[0].suppressed.size(), 1u);
    EXPECT_EQ(childFilters(selections[0]), (std::vector<std::string>{"*"}));
}

TEST(HostDispatchGroupMicrotest, ABinaryMatchedOnlyByADisabledCaseIsStillRun)
{
    // Nothing of it will run, but it has a report to write: gtest asked for a DISABLED_ case
    // alone still emits the suppressed entry for it, and dropping the binary would lose that.
    const std::vector<Selection> selections =
        group("/dir", {"rccl-UnitTestsMicro"}, oneEnabledOneDisabled(), {{}, {1}}, false);

    ASSERT_EQ(selections.size(), 1u);
    EXPECT_TRUE(selections[0].cases.empty());
    EXPECT_EQ(childFilters(selections[0]),
              (std::vector<std::string>{"Suite.DISABLED_Skipped"}));
}

TEST(HostDispatchGroupMicrotest, CasesAreGroupedUnderTheBinaryThatOwnsThem)
{
    const std::vector<CandidateCase> candidates = {
        CandidateCase{0, "Micro/Suite", TestCaseName{"Suite", "A"}},
        CandidateCase{1, "MicroInit/Suite", TestCaseName{"Suite", "B"}},
        CandidateCase{1, "MicroInit/Suite", TestCaseName{"Suite", "C"}}};
    const std::vector<Selection> selections =
        group("/dir", {"rccl-UnitTestsMicro", "rccl-UnitTestsMicroInit"}, candidates,
              {{0, 1, 2}, {}}, false);
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
// There is one layout, so where the binaries come from is a path join; what still needs proving
// is that nothing outside micro/ is picked up, since a wrong directory here runs nothing at all
// and can still exit 0 on the way there.

namespace {

// An executable file named *name* under *dir*, parents created as needed. Executability is the
// whole point: discoverBinaries() filters on isExecutableFile(), so a plain file would be
// skipped for that reason rather than the one a test here means to demonstrate. Nothing runs
// these, so the shebang body is only there to make them look like what they stand in for.
std::string addExecutable(const ScopedDir &dir, const std::string &name)
{
    const std::filesystem::path file = std::filesystem::path(dir.path()) / name;
    std::error_code ec;
    std::filesystem::create_directories(file.parent_path(), ec);
    std::ofstream(file) << "#!/bin/sh\n";
    std::filesystem::permissions(file, std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::add, ec);
    return file.string();
}

}  // namespace

TEST(HostDispatchLayoutMicrotest, TheFrontedBinariesAreTakenFromTheMicroSubdirectory)
{
    // One layout everywhere: the front door at the top, everything it runs one level down in
    // micro/ -- in the build tree and in an install alike.
    EXPECT_EQ(binaryDirectory("/opt/rocm/bin"),
              (std::filesystem::path("/opt/rocm/bin") / "micro").string());
}

TEST(HostDispatchLayoutMicrotest, ABinaryBesideTheFrontDoorIsNotFronted)
{
    // A stale copy left beside the front door by an earlier build is not in micro/, so it is not
    // what runs -- the directory is where the binaries are put, never where they happen to be.
    const ScopedDir tree("rccl-dispatch-test");
    const std::string self = addExecutable(tree, "rccl-UnitTestsHostAll");
    addExecutable(tree, "rccl-UnitTestsMicroInit");
    addExecutable(tree, "micro/rccl-UnitTestsMicroCrypt");

    EXPECT_EQ(discoverBinaries(binaryDirectory(tree.path()), self),
              std::vector<std::string>{"rccl-UnitTestsMicroCrypt"});
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

// ── prepareLogDirectory ─────────────────────────────────────────────

TEST(HostDispatchLogMicrotest, ANewLogDirectoryIsCreatedPrivateToThisUser)
{
    // The default directory's name is derived from the report's, which the test runner makes
    // with mkstemp() in /tmp. Another local user who sees that name appear can create the
    // matching .logs directory first, so what this process makes must be its own to write.
    const ScopedDir dir("rccl-host-dispatch-logdir");
    const std::string logs = (std::filesystem::path(dir.path()) / "run.logs").string();

    EXPECT_FALSE(prepareLogDirectory(logs).has_value());

    ASSERT_TRUE(std::filesystem::is_directory(logs));
    EXPECT_EQ(std::filesystem::status(logs).permissions(),
              std::filesystem::perms::owner_all);
}

TEST(HostDispatchLogMicrotest, MissingParentsOfTheLogDirectoryAreMade)
{
    // The log directory follows the report, and a caller may name a report in a directory that
    // does not exist yet. Only the leaf is ours to hold alone; the parents are ordinary.
    const ScopedDir dir("rccl-host-dispatch-logdir");
    const std::string logs =
        (std::filesystem::path(dir.path()) / "a" / "b" / "run.logs").string();

    EXPECT_FALSE(prepareLogDirectory(logs).has_value());
    EXPECT_TRUE(std::filesystem::is_directory(logs));
}

TEST(HostDispatchLogMicrotest, ADirectoryThisUserAlreadyOwnsIsReused)
{
    // Re-running into the same report path is the ordinary case, and the second run finds its
    // own directory from the first. Refusing that would turn every repeat run console-only.
    const ScopedDir dir("rccl-host-dispatch-logdir");
    const std::string logs = (std::filesystem::path(dir.path()) / "run.logs").string();
    ASSERT_FALSE(prepareLogDirectory(logs).has_value());

    EXPECT_FALSE(prepareLogDirectory(logs).has_value());
}

TEST(HostDispatchLogMicrotest, ALogDirectoryThatIsASymlinkIsRefused)
{
    // This is the planted directory: a link whose target is a directory this user does own, so
    // every ownership check passes while the logs land wherever it points. Only lstat sees it.
    const ScopedDir dir("rccl-host-dispatch-logdir");
    const std::filesystem::path elsewhere = std::filesystem::path(dir.path()) / "elsewhere";
    std::filesystem::create_directory(elsewhere);
    const std::filesystem::path logs = std::filesystem::path(dir.path()) / "run.logs";
    std::filesystem::create_directory_symlink(elsewhere, logs);

    const std::optional<std::string> problem = prepareLogDirectory(logs.string());
    ASSERT_TRUE(problem.has_value());
    EXPECT_NE(problem->find("symbolic link"), std::string::npos) << *problem;
}

TEST(HostDispatchLogMicrotest, ALogDirectoryThatIsAFileIsRefused)
{
    const ScopedDir dir("rccl-host-dispatch-logdir");
    const std::filesystem::path logs = std::filesystem::path(dir.path()) / "run.logs";
    std::ofstream(logs.string(), std::ios::binary) << "not a directory";

    const std::optional<std::string> problem = prepareLogDirectory(logs.string());
    ASSERT_TRUE(problem.has_value());
    EXPECT_NE(problem->find("not a directory"), std::string::npos) << *problem;
}

TEST(HostDispatchLogMicrotest, ALogDirectoryThatCannotBeMadeIsRefusedWithItsReason)
{
    // The caller turns the redirection off on any refusal, so the reason is all the user gets
    // to work out why their logs went to the console.
    const ScopedDir dir("rccl-host-dispatch-logdir");
    const std::filesystem::path blocked = std::filesystem::path(dir.path()) / "blocked";
    std::ofstream(blocked.string(), std::ios::binary) << "not a directory";

    const std::optional<std::string> problem = prepareLogDirectory((blocked / "run.logs").string());

    ASSERT_TRUE(problem.has_value());
    EXPECT_FALSE(problem->empty());
}

// ── resolveParallelism ──────────────────────────────────────────────

TEST(HostDispatchParallelismMicrotest, AnUnconfiguredRunTakesTheDefault)
{
    EXPECT_EQ(resolveParallelism(nullptr, 16), 16u);
    EXPECT_EQ(resolveParallelism("", 16), 16u);
}

TEST(HostDispatchParallelismMicrotest, TheDefaultIsHalfTheMachine)
{
    // Half and not all of it: the test tree is rarely the only thing on the machine -- a CI
    // agent runs other jobs beside it, a developer has a build waiting -- and the run flattens
    // out well before one job per core anyway.
    EXPECT_EQ(defaultParallelism(32), 16u);
    EXPECT_EQ(defaultParallelism(8), 4u);
}

TEST(HostDispatchParallelismMicrotest, ASmallMachineStillRunsOneBinaryAtATime)
{
    // Halving rounds down, and 0 would be a loop that starts nothing and never finishes.
    EXPECT_EQ(defaultParallelism(3), 1u);
    EXPECT_EQ(defaultParallelism(2), 1u);
    EXPECT_EQ(defaultParallelism(1), 1u);
    EXPECT_EQ(defaultParallelism(0), 1u);
}

TEST(HostDispatchParallelismMicrotest, TheEnvironmentSetsTheNumberOfBinariesInFlight)
{
    EXPECT_EQ(resolveParallelism("1", 32), 1u);
    EXPECT_EQ(resolveParallelism("4", 32), 4u);
}

TEST(HostDispatchParallelismMicrotest, OversubscriptionIsHonouredNotClamped)
{
    // The children spend part of their time on exec and on writing their logs, so asking for
    // more jobs than processors is a reasonable thing to want; nothing here knows better.
    EXPECT_EQ(resolveParallelism("64", 8), 64u);
}

TEST(HostDispatchParallelismMicrotest, ARequestThatIsNotAPositiveCountMeansTheDefault)
{
    // Zero, negative and unparseable all say the caller did not mean to constrain the run; none
    // of them may yield 0, which would be a loop that starts nothing and never finishes.
    EXPECT_EQ(resolveParallelism("0", 8), 8u);
    EXPECT_EQ(resolveParallelism("-4", 8), 8u);
    EXPECT_EQ(resolveParallelism("all", 8), 8u);
}

TEST(HostDispatchParallelismMicrotest, ADefaultOfNoneStillRunsOneBinary)
{
    EXPECT_EQ(resolveParallelism(nullptr, 0), 1u);
}

TEST(HostDispatchParallelismMicrotest, ThisMachineIsReportedAsAtLeastOneProcessor)
{
    EXPECT_GE(detectedParallelism(), 1u);
}

// ── resolveChildTimeout ─────────────────────────────────────────────

TEST(HostDispatchChildTimeoutMicrotest, AnUnconfiguredRunHasNoChildTimeout)
{
    // The dispatcher has always waited for its children indefinitely. A caller that has not
    // asked for a limit must keep that, or a slow machine would start killing passing binaries.
    EXPECT_EQ(resolveChildTimeout(nullptr), std::chrono::seconds{0});
    EXPECT_EQ(resolveChildTimeout(""), std::chrono::seconds{0});
}

TEST(HostDispatchChildTimeoutMicrotest, TheEnvironmentSetsHowLongAChildMayRun)
{
    EXPECT_EQ(resolveChildTimeout("120"), std::chrono::seconds{120});
    EXPECT_EQ(resolveChildTimeout("1"), std::chrono::seconds{1});
}

TEST(HostDispatchChildTimeoutMicrotest, ZeroIsAnExplicitOptOut)
{
    EXPECT_EQ(resolveChildTimeout("0"), std::chrono::seconds{0});
}

TEST(HostDispatchChildTimeoutMicrotest, ARequestThatIsNotSecondsLeavesTheRunUnbounded)
{
    // Unbounded is the only safe reading of a value that cannot be honoured -- a guessed limit
    // would kill binaries the caller never meant to limit. main() says so on stderr, because
    // silently unbounded is the outcome the caller set the variable to avoid.
    EXPECT_EQ(resolveChildTimeout("soon"), std::chrono::seconds{0});
    EXPECT_EQ(resolveChildTimeout("-30"), std::chrono::seconds{0});
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

    // formatLcov walks a std::map, so the whole output is deterministic and path-sorted: pinning
    // it is what proves each DA: line sits inside its own file's record rather than the other's.
    EXPECT_EQ(merged,
              "SF:/src/enqueue.cc\nFNF:0\nFNH:0\nDA:7,1\nLF:1\nLH:1\nend_of_record\n"
              "SF:/src/init.cc\nFNF:0\nFNH:0\nDA:10,1\nLF:1\nLH:1\nend_of_record\n");
}

TEST(HostDispatchLcovMicrotest, CountingLinesAreRecomputedNotCarriedOver)
{
    // Why the inputs' counting lines are ignored rather than summed: see parseLcov().
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

// ── writeMergedCoverage ─────────────────────────────────────────────

// The llvm tools, replaced by a function that records every command and writes the file the
// real tool would have written. Either step can be made to fail for a named binary, which is
// how the degraded paths -- the ones that decide whether a binary's coverage survives into the
// merged tracefile -- are reached without llvm-profdata or llvm-cov on the machine.
class FakeLlvm {
public:
    void failMergeFor(std::string binary) { failMerge_ = std::move(binary); }
    void failExportFor(std::string binary) { failExport_ = std::move(binary); }

    CommandFn command()
    {
        return [this](const std::string &command) {
            commands_.push_back(command);
            const bool merge = command.find(" merge ") != std::string::npos;
            // The output is the last quoted path: '-o <profdata>' for a merge, '> <lcov>' for
            // an export. Both are written before the status is checked, exactly as the shell
            // would: the redirect in particular creates the file even when llvm-cov fails.
            const std::string output = lastQuoted(command);
            const std::string &fails = merge ? failMerge_ : failExport_;
            std::ofstream(output, std::ios::binary | std::ios::trunc)
                << (merge ? "" : tracefileFor(output));
            if (!fails.empty() && command.find(fails) != std::string::npos) {
                return 1;
            }
            return 0;
        };
    }

    const std::vector<std::string> &commands() const { return commands_; }

private:
    // The lcov record a binary's export stands for, named after the directory it came from so a
    // merged tracefile says which binaries contributed to it.
    static std::string tracefileFor(const std::string &output)
    {
        const std::string owner = std::filesystem::path(output).parent_path().filename().string();
        return fmt::format("SF:/src/{}.cc\nDA:1,1\nend_of_record\n", owner);
    }

    static std::string lastQuoted(const std::string &command)
    {
        const size_t close = command.rfind('\'');
        const size_t open = command.rfind('\'', close - 1);
        return command.substr(open + 1, close - open - 1);
    }

    std::vector<std::string> commands_;
    std::string failMerge_;
    std::string failExport_;
};

// Seed *directory* with the profiles *binary* would have left behind, in the per-binary
// subdirectory childProfileFile() writes them to.
void seedProfiles(const std::string &directory, const std::string &binary, size_t count)
{
    const std::filesystem::path binaryDir =
        std::filesystem::path(directory) / std::filesystem::path(binary).filename();
    std::filesystem::create_directories(binaryDir);
    for (size_t i = 0; i < count; ++i) {
        std::ofstream(binaryDir / fmt::format("{}.profraw", i), std::ios::binary) << "profile";
    }
}

std::string mergedTracefile(const std::string &directory)
{
    return readFile((std::filesystem::path(directory) / kCoverageTracefile).string());
}

TEST(HostDispatchCoverageMicrotest, ProfilesAreReadFromTheDirectoryNamedAfterTheBinary)
{
    // The pairing childProfileFile() set up: profiles live under <coverage dir>/<binary
    // filename>/, and reading them from anywhere else would hand llvm-cov another binary's data.
    // The export is on unless the caller turned it off, and a consumer that reads the .profraw
    // files itself does exactly that; these tests pin what writeMergedCoverage does when asked.
    const ScopedUnset exportOn("RCCL_HOST_COVERAGE_EXPORT");
    const ScopedDir dir("rccl-host-dispatch-coverage");
    seedProfiles(dir.path(), "/build/rccl-UnitTestsMicroInit", 2);
    FakeLlvm llvm;

    writeMergedCoverage(dir.path(), {"/build/rccl-UnitTestsMicroInit"}, llvm.command());

    ASSERT_EQ(llvm.commands().size(), 2u);
    EXPECT_NE(llvm.commands()[0].find("rccl-UnitTestsMicroInit/0.profraw"), std::string::npos)
        << llvm.commands()[0];
    EXPECT_NE(llvm.commands()[0].find("rccl-UnitTestsMicroInit/1.profraw"), std::string::npos)
        << llvm.commands()[0];
    EXPECT_NE(mergedTracefile(dir.path()).find("SF:/src/rccl-UnitTestsMicroInit.cc"),
              std::string::npos);
}

TEST(HostDispatchCoverageMicrotest, ABinaryThatLeftNoProfilesIsPassedOverSilently)
{
    // A binary excluded by the run's filter never ran, so it has nothing to export; invoking
    // llvm-profdata on an empty profile list would turn that into a warning about every binary
    // the caller deliberately did not select.
    const ScopedUnset exportOn("RCCL_HOST_COVERAGE_EXPORT");
    const ScopedDir dir("rccl-host-dispatch-coverage");
    seedProfiles(dir.path(), "/build/rccl-UnitTestsMicroInit", 1);
    FakeLlvm llvm;

    writeMergedCoverage(dir.path(), {"/build/rccl-UnitTestsMicroNeverRan",
                                     "/build/rccl-UnitTestsMicroInit"},
                        llvm.command());

    EXPECT_EQ(llvm.commands().size(), 2u);   // the one binary that ran, merged and exported
    EXPECT_EQ(mergedTracefile(dir.path()).find("NeverRan"), std::string::npos);
}

TEST(HostDispatchCoverageMicrotest, AFailedMergeDropsOnlyThatBinary)
{
    // Best effort is the whole contract here: one binary llvm-profdata cannot read must not
    // cost the run the coverage of every other binary beside it.
    const ScopedUnset exportOn("RCCL_HOST_COVERAGE_EXPORT");
    const ScopedDir dir("rccl-host-dispatch-coverage");
    seedProfiles(dir.path(), "/build/rccl-UnitTestsMicroBroken", 1);
    seedProfiles(dir.path(), "/build/rccl-UnitTestsMicroInit", 1);
    FakeLlvm llvm;
    llvm.failMergeFor("rccl-UnitTestsMicroBroken");

    writeMergedCoverage(dir.path(),
                        {"/build/rccl-UnitTestsMicroBroken", "/build/rccl-UnitTestsMicroInit"},
                        llvm.command());

    const std::string merged = mergedTracefile(dir.path());
    EXPECT_EQ(merged.find("Broken"), std::string::npos) << merged;
    EXPECT_NE(merged.find("SF:/src/rccl-UnitTestsMicroInit.cc"), std::string::npos) << merged;
}

TEST(HostDispatchCoverageMicrotest, AFailedExportLeavesNoPartialTracefileBehind)
{
    // The redirect creates the tracefile before llvm-cov runs. Leaving the empty file beside
    // merged.profdata would make run_host_tests.sh take the export as done and skip the fallback
    // that would otherwise have recovered this binary's coverage.
    const ScopedUnset exportOn("RCCL_HOST_COVERAGE_EXPORT");
    const ScopedDir dir("rccl-host-dispatch-coverage");
    seedProfiles(dir.path(), "/build/rccl-UnitTestsMicroBroken", 1);
    seedProfiles(dir.path(), "/build/rccl-UnitTestsMicroInit", 1);
    FakeLlvm llvm;
    llvm.failExportFor("rccl-UnitTestsMicroBroken");

    writeMergedCoverage(dir.path(),
                        {"/build/rccl-UnitTestsMicroBroken", "/build/rccl-UnitTestsMicroInit"},
                        llvm.command());

    EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(dir.path()) /
                                         "rccl-UnitTestsMicroBroken" / "coverage.lcov"));
    const std::string merged = mergedTracefile(dir.path());
    EXPECT_EQ(merged.find("Broken"), std::string::npos) << merged;
    EXPECT_NE(merged.find("SF:/src/rccl-UnitTestsMicroInit.cc"), std::string::npos) << merged;
}

TEST(HostDispatchCoverageMicrotest, EveryExportedBinaryReachesTheMergedTracefile)
{
    const ScopedUnset exportOn("RCCL_HOST_COVERAGE_EXPORT");
    const ScopedDir dir("rccl-host-dispatch-coverage");
    seedProfiles(dir.path(), "/build/rccl-UnitTestsMicroInit", 1);
    seedProfiles(dir.path(), "/build/rccl-UnitTestsMicroUtils", 1);
    FakeLlvm llvm;

    writeMergedCoverage(dir.path(),
                        {"/build/rccl-UnitTestsMicroInit", "/build/rccl-UnitTestsMicroUtils"},
                        llvm.command());

    const std::string merged = mergedTracefile(dir.path());
    EXPECT_NE(merged.find("SF:/src/rccl-UnitTestsMicroInit.cc"), std::string::npos) << merged;
    EXPECT_NE(merged.find("SF:/src/rccl-UnitTestsMicroUtils.cc"), std::string::npos) << merged;
}

TEST(HostDispatchCoverageMicrotest, NoBinaryExportedMeansNoTracefileAtAll)
{
    // An empty tracefile reads downstream as a tree with no coverage at all; writing nothing
    // leaves the previous phase's file, and the caller's own fallback, intact.
    const ScopedUnset exportOn("RCCL_HOST_COVERAGE_EXPORT");
    const ScopedDir dir("rccl-host-dispatch-coverage");
    FakeLlvm llvm;

    writeMergedCoverage(dir.path(), {"/build/rccl-UnitTestsMicroInit"}, llvm.command());

    EXPECT_TRUE(llvm.commands().empty());
    EXPECT_FALSE(
        std::filesystem::exists(std::filesystem::path(dir.path()) / kCoverageTracefile));
}

TEST(HostDispatchCoverageMicrotest, NothingIsSaidAboutAnExportThatHasNothingToExport)
{
    // LLVM_PROFILE_FILE being set does not mean the tree was built instrumented -- nothing ties
    // the two together, and run_host_tests.sh exports the variable either way. Announcing the
    // binaries and then writing no tracefile is indistinguishable from an export that failed
    // for every one of them, so an uninstrumented run says nothing at all.
    const ScopedUnset exportOn("RCCL_HOST_COVERAGE_EXPORT");
    const ScopedDir dir("rccl-host-dispatch-coverage");
    FakeLlvm llvm;

    const std::string printed = RcclUnitTesting::CaptureStdout([&] {
        writeMergedCoverage(dir.path(), {"/build/rccl-UnitTestsMicroInit"}, llvm.command());
    });

    EXPECT_EQ(printed, "");
}

TEST(HostDispatchCoverageMicrotest, OnlyTheBinariesWithProfilesAreCounted)
{
    // The progress count is the reader's only estimate of how long this phase will take, so it
    // counts what will actually be exported rather than every binary the run knows of.
    const ScopedUnset exportOn("RCCL_HOST_COVERAGE_EXPORT");
    const ScopedDir dir("rccl-host-dispatch-coverage");
    seedProfiles(dir.path(), "/build/rccl-UnitTestsMicroInit", 1);
    FakeLlvm llvm;

    const std::string printed = RcclUnitTesting::CaptureStdout([&] {
        writeMergedCoverage(dir.path(),
                            {"/build/rccl-UnitTestsMicroNeverRan",
                             "/build/rccl-UnitTestsMicroInit"},
                            llvm.command());
    });

    EXPECT_NE(printed.find("exporting 1 binaries"), std::string::npos) << printed;
    EXPECT_NE(printed.find("[1/1] rccl-UnitTestsMicroInit"), std::string::npos) << printed;
    EXPECT_EQ(printed.find("NeverRan"), std::string::npos) << printed;
}

// ── coverageExportRequested ──────────────────────────────────────────

TEST(HostDispatchCoverageMicrotest, TheExportIsOnUnlessItIsTurnedOff)
{
    // Opt-out, never opt-in: run_host_tests.sh builds its overall report from the tracefile this
    // writes, so a caller who has not heard of the variable must keep getting it.
    EXPECT_TRUE(coverageExportRequested(nullptr));
    EXPECT_TRUE(coverageExportRequested(""));
    EXPECT_TRUE(coverageExportRequested("1"));
    EXPECT_FALSE(coverageExportRequested("0"));
}

TEST(HostDispatchCoverageMicrotest, TurningTheExportOffRunsNoLlvmAndWritesNoTracefile)
{
    // What it buys the caller who asked: an llvm-profdata merge and an llvm-cov export per
    // binary, skipped. The profiles are untouched -- the children were handed their
    // LLVM_PROFILE_FILE long before this -- so a consumer collecting them itself still can.
    const ScopedDir dir("rccl-host-dispatch-coverage");
    seedProfiles(dir.path(), "/build/rccl-UnitTestsMicroInit", 2);
    FakeLlvm llvm;
    setenv(std::string(kCoverageExportEnv).c_str(), "0", 1);

    writeMergedCoverage(dir.path(), {"/build/rccl-UnitTestsMicroInit"}, llvm.command());
    unsetenv(std::string(kCoverageExportEnv).c_str());

    EXPECT_TRUE(llvm.commands().empty());
    EXPECT_FALSE(
        std::filesystem::exists(std::filesystem::path(dir.path()) / kCoverageTracefile));
    // The profiles the caller does want are still where the children left them.
    EXPECT_EQ(profilesIn((std::filesystem::path(dir.path()) / "rccl-UnitTestsMicroInit").string())
                  .size(),
              2u);
}

// ── runSelections ───────────────────────────────────────────────────

// A scheduler driven through the operations it performs, with no process ever created.
//
// Each "child" writes a one-suite report at the path it was given, named after its own filter,
// so a fragment can be traced back to the exact part that produced it. Children are reaped in
// the order the test asks for, which is how completion order is made to differ from selection
// order.
class FakeScheduler {
public:
    struct Started {
        std::string path;
        std::string filter;
        std::string reportPath;
    };

    explicit FakeScheduler(std::string_view directory) : directory_(directory) {}

    SpawnFn spawn()
    {
        return [this](const std::string &path, const std::vector<std::string> &args,
                      const std::string &) {
            const std::string filter = args.at(0).substr(std::string("--gtest_filter=").size());
            const std::string reportPath =
                args.at(1).substr(args.at(1).find(':') + 1);
            started_.push_back({path, filter, reportPath});
            if (silent_.count(nextPid_) == 0) {
                std::ofstream(reportPath, std::ios::binary | std::ios::trunc)
                    << "<testsuites tests=\"1\" failures=\"0\"><testsuite name=\"" << filter
                    << "\"/></testsuites>";
            }
            live_.push_back(nextPid_);
            return nextPid_++;
        };
    }

    // Reaps the most recently started child first, so completions come back in an order the
    // scheduler did not choose.
    //
    // A child the test has marked as hanging is not reapable until it has been killed, so a
    // bounded reap expires against it exactly as a real one would; the clock then advances by
    // the interval the scheduler asked to wait, which is what makes the deadline deterministic.
    ReapFn reapNewestFirst()
    {
        return [this](int &status, std::optional<std::chrono::milliseconds> within) {
            status = 0;
            boundedReaps_ += within.has_value() ? 1 : 0;
            const auto reapable = std::find_if(live_.rbegin(), live_.rend(), [this](pid_t pid) {
                return hanging_.count(pid) == 0 || killed_.count(pid) != 0;
            });
            if (reapable == live_.rend()) {
                if (live_.empty()) {
                    return pid_t{-1};
                }
                // Only hanging children are left. A bounded wait expires; an unbounded one would
                // block forever against a real child, so a test that gets here without a timeout
                // has gone wrong and says so rather than looping.
                EXPECT_TRUE(within.has_value()) << "an unbounded reap would never return here";
                now_ += within.value_or(std::chrono::milliseconds{0});
                // An escalation that never reaches SIGKILL would signal this child forever.
                // Bounded here so that shows up as a failed assertion rather than a suite that
                // hangs and has to be read from a stack trace.
                if (++expiries_ > kMaxExpiries) {
                    ADD_FAILURE() << "the scheduler signalled a hanging child " << expiries_
                                  << " times without ever killing it";
                    return pid_t{-1};
                }
                return pid_t{0};
            }
            const pid_t pid = *reapable;
            live_.erase(std::next(reapable).base());
            inFlightHighWater_ = std::max(inFlightHighWater_, live_.size() + 1);
            const auto exit = exits_.find(pid);
            status = killed_.count(pid) != 0 ? 128 + SIGKILL
                                             : (exit == exits_.end() ? 0 : exit->second);
            return pid;
        };
    }

    // The child started *index*-th never finishes on its own, so only a SIGKILL ends it.
    void hangChild(size_t index) { hanging_.insert(static_cast<pid_t>(100 + index)); }

    // The child started *index*-th is reaped with *status* rather than 0.
    void exitChild(size_t index, int status) { exits_[static_cast<pid_t>(100 + index)] = status; }

    // The child started *index*-th leaves no report behind, as a binary dying partway does.
    void silenceChild(size_t index) { silent_.insert(static_cast<pid_t>(100 + index)); }

    // The run is asked to stop once *spawns* children have been started, standing in for the
    // SIGTERM an outer watchdog sends partway through.
    void stopAfter(size_t spawns) { stopAfter_ = spawns; }

    TerminatedFn terminated()
    {
        return [this] { return stopAfter_.has_value() && started_.size() >= *stopAfter_; };
    }

    KillFn kill()
    {
        return [this](pid_t pid, int signal) {
            signals_.push_back({pid, signal});
            if (signal == SIGKILL) {
                killed_.insert(pid);
            }
        };
    }

    NowFn now()
    {
        return [this] { return now_; };
    }

    SchedulerOperations ops()
    {
        return SchedulerOperations{spawn(),  reapNewestFirst(), kill(),
                                   now(),    reportPaths(),     terminated()};
    }

    const std::vector<Started> &started() const { return started_; }
    // The most children ever started but not yet reaped.
    size_t inFlightHighWater() const { return inFlightHighWater_; }
    // Every signal the scheduler sent, in order, as (pid, signal).
    const std::vector<std::pair<pid_t, int>> &signals() const { return signals_; }
    // How many reaps were given a deadline to wait until rather than left to block.
    size_t boundedReaps() const { return boundedReaps_; }

    ReportPathFn reportPaths()
    {
        return [this](const std::string &qualifier, size_t part) {
            return (std::filesystem::path(directory_) /
                    fmt::format("{}-{}.xml", qualifier, part))
                .string();
        };
    }

private:
    std::string directory_;
    std::vector<Started> started_;
    std::vector<pid_t> live_;
    std::set<pid_t> hanging_;
    std::set<pid_t> killed_;
    std::set<pid_t> silent_;
    std::map<pid_t, int> exits_;
    std::vector<std::pair<pid_t, int>> signals_;
    std::chrono::steady_clock::time_point now_{};
    static constexpr size_t kMaxExpiries = 16;
    size_t expiries_ = 0;
    size_t boundedReaps_ = 0;
    size_t inFlightHighWater_ = 0;
    std::optional<size_t> stopAfter_;
    pid_t nextPid_ = 100;
};

// A selection of *cases* cases out of a binary holding one more, so childFilters() names them
// case by case rather than asking for the whole binary with a star.
Selection schedulerSelection(const std::string &qualifier, size_t cases)
{
    Selection selection;
    selection.path = "/build/" + qualifier;
    selection.qualifier = qualifier;
    selection.runnable = cases + 1;   // a partial selection, so the filter names cases
    for (size_t i = 0; i < cases; ++i) {
        selection.cases.push_back(TestCaseName{"Suite", fmt::format("{}Case{}", qualifier, i)});
    }
    return selection;
}

// A selection whose case names are long enough that childFilters() cannot fit them into one
// --gtest_filter and has to hand the binary back in parts.
Selection splitSelection()
{
    Selection selection = schedulerSelection("Split", 0);
    for (size_t i = 0; i < 1500; ++i) {
        selection.cases.push_back(TestCaseName{std::string(80, 'S'), fmt::format("Case{}", i)});
    }
    selection.runnable = selection.cases.size() + 1;
    return selection;
}

TEST(HostDispatchSchedulerMicrotest, EveryPartOfASplitBinaryIsRun)
{
    // A binary whose filter does not fit in one command line is run in parts. Dropping the
    // re-queue of the next part would silently run only the first part of such a binary, and
    // every case beyond it would vanish from the report without failing the run.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    // Case names long enough that kMaxFilterChars forces more than one part.
    const Selection selection = splitSelection();
    const std::vector<std::vector<std::string>> filters = {childFilters(selection)};
    ASSERT_GT(filters[0].size(), 1u) << "the selection must split for this test to mean anything";

    int failures = 0;
    const std::vector<SuiteFragment> fragments =
        runSelections({selection}, filters, ScheduleOptions{4, ReportFormat::Xml, {}, ""},
                      fake.ops(), failures);

    ASSERT_EQ(fake.started().size(), filters[0].size());
    for (size_t part = 0; part < filters[0].size(); ++part) {
        EXPECT_EQ(fake.started()[part].filter, filters[0][part]);
    }
    EXPECT_EQ(fragments.size(), filters[0].size());
    EXPECT_EQ(failures, 0);
}

TEST(HostDispatchSchedulerMicrotest, PartsOfOneBinaryNeverRunAtTheSameTime)
{
    // Each part writes the same per-binary log, and part 0 truncates it. Two parts in flight
    // together would interleave their output and lose whatever the first had already written.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    const Selection selection = splitSelection();
    const std::vector<std::vector<std::string>> filters = {childFilters(selection)};
    ASSERT_GT(filters[0].size(), 1u);

    int failures = 0;
    runSelections({selection}, filters, ScheduleOptions{8, ReportFormat::Xml, {}, ""},
                  fake.ops(), failures);

    // Eight jobs were allowed, but one binary's parts are serial, so only ever one was live.
    EXPECT_EQ(fake.inFlightHighWater(), 1u);
}

TEST(HostDispatchSchedulerMicrotest, NoMoreChildrenThanJobsRunAtOnce)
{
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    std::vector<Selection> selections;
    std::vector<std::vector<std::string>> filters;
    for (const std::string &qualifier : {"Alpha", "Beta", "Gamma", "Delta", "Epsilon"}) {
        selections.push_back(schedulerSelection(qualifier, 2));
        filters.push_back(childFilters(selections.back()));
    }

    int failures = 0;
    runSelections(selections, filters, ScheduleOptions{2, ReportFormat::Xml, {}, ""}, fake.ops(),
                  failures);

    EXPECT_EQ(fake.inFlightHighWater(), 2u);
    EXPECT_EQ(fake.started().size(), selections.size());
}

TEST(HostDispatchSchedulerMicrotest, FragmentsComeOutInSelectionOrderNotCompletionOrder)
{
    // The merged report is what a human reads and what a diff of two runs compares, so it must
    // not reorder itself according to which child happened to finish first.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    std::vector<Selection> selections;
    std::vector<std::vector<std::string>> filters;
    for (const std::string &qualifier : {"Alpha", "Beta", "Gamma"}) {
        selections.push_back(schedulerSelection(qualifier, 1));
        filters.push_back(childFilters(selections.back()));
    }

    int failures = 0;
    // All three in flight at once and reaped newest first, so completion order is the reverse of
    // selection order.
    const std::vector<SuiteFragment> fragments =
        runSelections(selections, filters, ScheduleOptions{3, ReportFormat::Xml, {}, ""},
                      fake.ops(), failures);

    ASSERT_EQ(fragments.size(), 3u);
    EXPECT_NE(fragments[0].xml.find("AlphaCase0"), std::string::npos) << fragments[0].xml;
    EXPECT_NE(fragments[1].xml.find("BetaCase0"), std::string::npos) << fragments[1].xml;
    EXPECT_NE(fragments[2].xml.find("GammaCase0"), std::string::npos) << fragments[2].xml;
    EXPECT_EQ(failures, 0);
}

TEST(HostDispatchSchedulerMicrotest, AChildThatCannotBeStartedIsReportedAndItsRestSkipped)
{
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    const Selection selection = schedulerSelection("Alpha", 1);

    int failures = 0;
    SchedulerOperations ops = fake.ops();
    ops.spawn = [](const std::string &, const std::vector<std::string> &, const std::string &) {
        return pid_t{-1};
    };
    const std::vector<SuiteFragment> fragments =
        runSelections({selection}, {childFilters(selection)},
                      ScheduleOptions{1, ReportFormat::Xml, {}, ""}, ops, failures);

    ASSERT_EQ(fragments.size(), 1u);
    EXPECT_NE(fragments[0].xml.find("fork failed"), std::string::npos) << fragments[0].xml;
    EXPECT_EQ(failures, 1);
}

TEST(HostDispatchSchedulerMicrotest, AChildThatWroteNoReportFailsTheRunEvenHavingExitedCleanly)
{
    // A binary with no report contributes no cases, and gtest counts only what it was given --
    // so without a synthesised failure here, a binary that produced nothing is indistinguishable
    // from a shorter green run. A clean exit deliberately, so this rests on the missing report
    // alone and not on the status branch below.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    const Selection selection = schedulerSelection("Alpha", 1);
    fake.silenceChild(0);

    int failures = 0;
    const std::vector<SuiteFragment> fragments =
        runSelections({selection}, {childFilters(selection)},
                      ScheduleOptions{1, ReportFormat::Xml, {}, ""}, fake.ops(), failures);

    ASSERT_EQ(fragments.size(), 1u);
    EXPECT_NE(fragments[0].xml.find("wrote no gtest report"), std::string::npos)
        << fragments[0].xml;
    EXPECT_EQ(failures, 1);
}

TEST(HostDispatchSchedulerMicrotest, AChildThatDiedWithoutAReportNamesItsStatusAndItsLog)
{
    // The two are all the evidence left of why: the status says how it died, and with no report
    // to read, its log holds the only record of where it had got to.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    const Selection selection = schedulerSelection("Alpha", 1);
    fake.silenceChild(0);
    fake.exitChild(0, 134);   // SIGABRT, as an assertion failure in the child would give

    int failures = 0;
    const std::vector<SuiteFragment> fragments = runSelections(
        {selection}, {childFilters(selection)},
        ScheduleOptions{1, ReportFormat::Xml, {}, dir.path()}, fake.ops(), failures);

    ASSERT_EQ(fragments.size(), 1u);
    EXPECT_NE(fragments[0].xml.find("exit status 134"), std::string::npos) << fragments[0].xml;
    EXPECT_NE(fragments[0].xml.find("Alpha.log"), std::string::npos) << fragments[0].xml;
    EXPECT_EQ(failures, 2);   // no report, and a status saying so too
}

TEST(HostDispatchSchedulerMicrotest, AChildThatExitsNonzeroAfterAPassingReportFailsTheRun)
{
    // The report is green but the binary did not come back clean -- a crash in a global
    // destructor, a leak check failing at exit. Taking the report at face value would publish
    // those cases as passed and lose the only sign that anything was wrong.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    const Selection selection = schedulerSelection("Alpha", 1);
    fake.exitChild(0, 1);

    int failures = 0;
    const std::vector<SuiteFragment> fragments =
        runSelections({selection}, {childFilters(selection)},
                      ScheduleOptions{1, ReportFormat::Xml, {}, ""}, fake.ops(), failures);

    ASSERT_EQ(fragments.size(), 2u);   // the child's own report, and the exit reported beside it
    EXPECT_NE(fragments[1].xml.find("exited 1 after writing a passing report"), std::string::npos)
        << fragments[1].xml;
    EXPECT_EQ(failures, 1);
}

TEST(HostDispatchSchedulerMicrotest, AChildThatExitsCleanlyWithAPassingReportFailsNothing)
{
    // The two branches above must not fire on the ordinary case, which is every binary in a
    // green run: a zero exit and a report with no failures in it.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    const Selection selection = schedulerSelection("Alpha", 1);

    int failures = 0;
    const std::vector<SuiteFragment> fragments =
        runSelections({selection}, {childFilters(selection)},
                      ScheduleOptions{1, ReportFormat::Xml, {}, ""}, fake.ops(), failures);

    EXPECT_EQ(fragments.size(), 1u);
    EXPECT_EQ(failures, 0);
}

TEST(HostDispatchSchedulerMicrotest, AChildStillRunningAtItsDeadlineIsTermedThenKilled)
{
    // SIGTERM first so a child with a handler can flush its log and say where it was; SIGKILL
    // after the grace period, because a child wedged in the kernel will never act on SIGTERM and
    // the run cannot wait for it.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    const Selection selection = schedulerSelection("Alpha", 1);
    fake.hangChild(0);

    int failures = 0;
    runSelections({selection}, {childFilters(selection)},
                  ScheduleOptions{1, ReportFormat::Xml, {}, "", std::chrono::seconds{30}},
                  fake.ops(), failures);

    ASSERT_EQ(fake.signals().size(), 2u);
    EXPECT_EQ(fake.signals()[0], std::make_pair(pid_t{100}, SIGTERM));
    EXPECT_EQ(fake.signals()[1], std::make_pair(pid_t{100}, SIGKILL));
}

TEST(HostDispatchSchedulerMicrotest, ATimedOutChildIsReportedAsATimeoutNotAMissingReport)
{
    // A killed child leaves no report, so without this it would be reported as a binary that
    // wrote none -- which reads as a crash and sends the reader looking for one. The message
    // has to name the limit it exceeded and the log holding the case it was stuck in.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    const Selection selection = schedulerSelection("Alpha", 1);
    fake.hangChild(0);

    int failures = 0;
    const std::vector<SuiteFragment> fragments = runSelections(
        {selection}, {childFilters(selection)},
        ScheduleOptions{1, ReportFormat::Xml, {}, dir.path(), std::chrono::seconds{30}},
        fake.ops(), failures);

    ASSERT_EQ(fragments.size(), 1u);
    EXPECT_NE(fragments[0].xml.find("timed out after 30 s"), std::string::npos) << fragments[0].xml;
    EXPECT_NE(fragments[0].xml.find("Alpha.log"), std::string::npos) << fragments[0].xml;
    EXPECT_EQ(fragments[0].xml.find("wrote no gtest report"), std::string::npos)
        << fragments[0].xml;
    EXPECT_EQ(failures, 1);
}

TEST(HostDispatchSchedulerMicrotest, ATimedOutBinaryDoesNotGetToHangItsRemainingPartsToo)
{
    // Each part of a split binary would wait out the whole timeout again to learn what the first
    // one already established, so a hang in a binary split five ways would cost five timeouts.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    const Selection selection = splitSelection();
    const std::vector<std::vector<std::string>> filters = {childFilters(selection)};
    ASSERT_GT(filters[0].size(), 1u);
    fake.hangChild(0);

    int failures = 0;
    const std::vector<SuiteFragment> fragments =
        runSelections({selection}, filters,
                      ScheduleOptions{1, ReportFormat::Xml, {}, "", std::chrono::seconds{30}},
                      fake.ops(), failures);

    EXPECT_EQ(fake.started().size(), 1u);
    ASSERT_EQ(fragments.size(), 1u);
    EXPECT_NE(fragments[0].xml.find("remaining part"), std::string::npos) << fragments[0].xml;
    EXPECT_EQ(failures, 1);
}

TEST(HostDispatchSchedulerMicrotest, TheBinariesBesideAHangingOneStillReportTheirResults)
{
    // This is the whole point of the per-binary timeout: an outer CI timeout kills the
    // dispatcher before it writes the merged report, so a single hang costs every binary's
    // results rather than its own.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    std::vector<Selection> selections;
    std::vector<std::vector<std::string>> filters;
    for (const std::string &qualifier : {"Alpha", "Beta", "Gamma"}) {
        selections.push_back(schedulerSelection(qualifier, 1));
        filters.push_back(childFilters(selections.back()));
    }
    fake.hangChild(1);   // Beta, the one started second

    int failures = 0;
    const std::vector<SuiteFragment> fragments =
        runSelections(selections, filters,
                      ScheduleOptions{3, ReportFormat::Xml, {}, "", std::chrono::seconds{30}},
                      fake.ops(), failures);

    ASSERT_EQ(fragments.size(), 3u);
    EXPECT_NE(fragments[0].xml.find("AlphaCase0"), std::string::npos) << fragments[0].xml;
    EXPECT_NE(fragments[1].xml.find("timed out"), std::string::npos) << fragments[1].xml;
    EXPECT_NE(fragments[2].xml.find("GammaCase0"), std::string::npos) << fragments[2].xml;
    EXPECT_EQ(failures, 1);
}

TEST(HostDispatchSchedulerMicrotest, AChildThatFinishesWithinItsDeadlineIsNeverSignalled)
{
    // The timeout must be inert for a run that does not hang, or a binary that merely ran long
    // on a loaded machine would come back killed and the suite would be flaky by configuration.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    std::vector<Selection> selections;
    std::vector<std::vector<std::string>> filters;
    for (const std::string &qualifier : {"Alpha", "Beta"}) {
        selections.push_back(schedulerSelection(qualifier, 1));
        filters.push_back(childFilters(selections.back()));
    }

    int failures = 0;
    const std::vector<SuiteFragment> fragments =
        runSelections(selections, filters,
                      ScheduleOptions{2, ReportFormat::Xml, {}, "", std::chrono::seconds{30}},
                      fake.ops(), failures);

    EXPECT_TRUE(fake.signals().empty());
    EXPECT_EQ(fragments.size(), 2u);
    EXPECT_EQ(failures, 0);
}

TEST(HostDispatchSchedulerMicrotest, APlantedLogSymlinkIsNotTruncatedThroughToItsTarget)
{
    // The first part of a binary truncates its log so a re-run does not append to the last
    // one's. Through a symlink someone else planted, that truncation lands on their target --
    // the destructive half of the hazard prepareLogDirectory() covers, and the one that still
    // applies when the caller named a log directory this process did not create.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    const std::filesystem::path target = std::filesystem::path(dir.path()) / "precious";
    std::ofstream(target.string(), std::ios::binary) << "do not truncate me";
    std::filesystem::create_symlink(target, std::filesystem::path(dir.path()) / "Alpha.log");

    FakeScheduler fake(dir.path());
    const Selection selection = schedulerSelection("Alpha", 1);
    int failures = 0;
    runSelections({selection}, {childFilters(selection)},
                  ScheduleOptions{1, ReportFormat::Xml, {}, dir.path()}, fake.ops(), failures);

    EXPECT_EQ(std::filesystem::file_size(target), std::string("do not truncate me").size());
}

TEST(HostDispatchSchedulerMicrotest, AnUnconfiguredRunNeverBoundsItsReap)
{
    // No timeout means no deadline to wake for, and a bounded reap has to poll to honour one.
    // The default run blocks in waitpid() as it always has, and costs nothing for a feature it
    // is not using.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    const Selection selection = schedulerSelection("Alpha", 1);

    int failures = 0;
    runSelections({selection}, {childFilters(selection)},
                  ScheduleOptions{1, ReportFormat::Xml, {}, ""}, fake.ops(), failures);

    EXPECT_EQ(fake.boundedReaps(), 0u);
    EXPECT_EQ(failures, 0);
}

TEST(HostDispatchSchedulerMicrotest, AStoppedRunKillsItsChildrenAndKeepsWhatFinished)
{
    // An outer watchdog's SIGTERM, or a user's Ctrl-C. The binaries already finished are the
    // whole point of handling it: killing the dispatcher outright loses all of them, which is
    // what turns one hung binary into a report of ~11,000 timed-out cases.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    std::vector<Selection> selections;
    std::vector<std::vector<std::string>> filters;
    for (const std::string &qualifier : {"Alpha", "Beta", "Gamma"}) {
        selections.push_back(schedulerSelection(qualifier, 1));
        filters.push_back(childFilters(selections.back()));
    }
    fake.stopAfter(2);   // observed once all three are in flight and the first has been reaped
                         // -- the fake reaps newest first, so Gamma is the one that finishes

    int failures = 0;
    const std::vector<SuiteFragment> fragments =
        runSelections(selections, filters, ScheduleOptions{3, ReportFormat::Xml, {}, ""},
                      fake.ops(), failures);

    ASSERT_EQ(fragments.size(), 3u);
    EXPECT_NE(fragments[0].xml.find("run was stopped"), std::string::npos) << fragments[0].xml;
    EXPECT_NE(fragments[1].xml.find("run was stopped"), std::string::npos) << fragments[1].xml;
    EXPECT_NE(fragments[2].xml.find("GammaCase0"), std::string::npos) << fragments[2].xml;
    EXPECT_EQ(fake.signals().size(), 2u) << "both surviving children must be signalled";
    for (const std::pair<pid_t, int> &signal : fake.signals()) {
        EXPECT_EQ(signal.second, SIGTERM);
    }
    EXPECT_EQ(failures, 2);
}

TEST(HostDispatchSchedulerMicrotest, AStoppedRunNamesTheBinariesItNeverStarted)
{
    // Silence would read as a selection that was never asked for. The report says they did not
    // run, so a consumer comparing it with --gtest_list_tests sees why.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    std::vector<Selection> selections;
    std::vector<std::vector<std::string>> filters;
    for (const std::string &qualifier : {"Alpha", "Beta", "Gamma"}) {
        selections.push_back(schedulerSelection(qualifier, 1));
        filters.push_back(childFilters(selections.back()));
    }
    fake.stopAfter(1);   // one at a time, so the stop lands with two still unstarted

    int failures = 0;
    const std::vector<SuiteFragment> fragments =
        runSelections(selections, filters, ScheduleOptions{1, ReportFormat::Xml, {}, ""},
                      fake.ops(), failures);

    ASSERT_EQ(fragments.size(), 3u);
    EXPECT_NE(fragments[0].xml.find("AlphaCase0"), std::string::npos) << fragments[0].xml;
    EXPECT_NE(fragments[1].xml.find("stopped before this binary started"), std::string::npos)
        << fragments[1].xml;
    EXPECT_NE(fragments[2].xml.find("stopped before this binary started"), std::string::npos)
        << fragments[2].xml;
    EXPECT_TRUE(fake.started().size() == 1u) << "nothing may start after the run is stopped";
    EXPECT_EQ(failures, 2);
}

TEST(HostDispatchSchedulerMicrotest, ARunThatIsNeverStoppedSignalsNothing)
{
    // The stop path must be inert otherwise: an ordinary run has no watchdog behind it and must
    // not kill a child for taking its time.
    const ScopedDir dir("rccl-host-dispatch-scheduler");
    FakeScheduler fake(dir.path());
    std::vector<Selection> selections;
    std::vector<std::vector<std::string>> filters;
    for (const std::string &qualifier : {"Alpha", "Beta"}) {
        selections.push_back(schedulerSelection(qualifier, 1));
        filters.push_back(childFilters(selections.back()));
    }

    int failures = 0;
    const std::vector<SuiteFragment> fragments =
        runSelections(selections, filters, ScheduleOptions{2, ReportFormat::Xml, {}, ""},
                      fake.ops(), failures);

    EXPECT_TRUE(fake.signals().empty());
    EXPECT_EQ(fragments.size(), 2u);
    EXPECT_EQ(failures, 0);
}

// ── signalChildGroup ────────────────────────────────────────────────
//
// The two tests below run real processes: the behaviour under test is the kernel's grouping of
// them, which no fake can stand in for.

// Whether *pid* is a zombie: dead, but not yet reaped by whoever inherited it.
//
// /proc/<pid>/stat is "<pid> (<comm>) <state> ..." and comm can itself hold spaces and
// parentheses, so the state is read from the LAST ')' rather than by splitting on spaces.
bool isZombie(pid_t pid)
{
    const std::string stat = readFile(fmt::format("/proc/{}/stat", pid));
    const size_t paren = stat.rfind(')');
    return paren != std::string::npos && stat.compare(paren, 3, ") Z") == 0;
}

// Wait up to a second for *pid* to be gone, and say whether it is.
//
// A zombie counts as gone, and is in fact the clearest proof the signal landed. The grandchild
// below is orphaned by the same kill that ends it, so it is reparented to PID 1 and stays a
// zombie until PID 1 reaps it -- which a container started on `sleep infinity` or
// `tail -f /dev/null`, with no `--init`, never does. kill(pid, 0) keeps succeeding on it, so
// without this the test would fail for the process having died exactly as it was asked to.
bool waitUntilGone(pid_t pid)
{
    for (int attempt = 0; attempt < 100; ++attempt) {
        if ((::kill(pid, 0) != 0 && errno == ESRCH) || isZombie(pid)) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }
    return false;
}

TEST(HostDispatchSignalMicrotest, AChildIsStartedInAProcessGroupOfItsOwn)
{
    // The group is what makes a timeout reach the child's own children, and the child must be
    // in it before the dispatcher could send the first signal.
    const pid_t pid = spawnChild("/bin/sh", {"-c", "while :; do sleep 1; done"});
    ASSERT_GT(pid, 0);

    EXPECT_EQ(getpgid(pid), pid);

    signalChildGroup(pid, SIGKILL);
    int status = 0;
    waitpid(pid, &status, 0);
}

TEST(HostDispatchSignalMicrotest, ASignalReachesWhatTheChildItselfForked)
{
    // A test binary forks -- death tests, ProcessIsolatedTestRunner -- and signalling the child
    // pid alone would leave those grandchildren running on the CI node after the timeout.
    const ScopedDir dir("rccl-host-dispatch-signal");
    const std::string pidFile = (std::filesystem::path(dir.path()) / "grandchild.pid").string();
    const pid_t pid = spawnChild(
        "/bin/sh", {"-c", "sh -c 'while :; do sleep 1; done' & echo $! > " + pidFile +
                           "; while :; do sleep 1; done"});
    ASSERT_GT(pid, 0);

    pid_t grandchild = 0;
    for (int attempt = 0; attempt < 500 && grandchild <= 0; ++attempt) {
        std::ifstream(pidFile) >> grandchild;
        if (grandchild <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
    }
    ASSERT_GT(grandchild, 0) << "the child never reported the process it forked";

    signalChildGroup(pid, SIGKILL);
    int status = 0;
    waitpid(pid, &status, 0);

    EXPECT_TRUE(waitUntilGone(grandchild)) << "grandchild " << grandchild << " outlived the kill";
}

TEST(HostDispatchSignalMicrotest, AChildWithNoGroupOfItsOwnIsStillSignalled)
{
    // A child that died between the fork and its own setpgid() leaves no group behind. Taking
    // the ESRCH for "already gone" would skip the pid that is still there in the race the other
    // way round, where only the parent's setpgid() has yet to run.
    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        for (;;) {
            pause();   // no setpgid(): this process stays in the test's own group
        }
    }

    signalChildGroup(pid, SIGKILL);

    // Polled rather than a blocking wait: a signal that never arrives must fail this test, not
    // hang the run it is meant to protect.
    int status = 0;
    pid_t reaped = 0;
    for (int attempt = 0; attempt < 100 && reaped == 0; ++attempt) {
        reaped = waitpid(pid, &status, WNOHANG);
        if (reaped == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
    }
    if (reaped != pid) {
        ::kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
    }
    EXPECT_EQ(reaped, pid) << "the child outlived a signal aimed at a group it was not in";
    EXPECT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL);
}

// ── reapAnyChild ────────────────────────────────────────────────────
//
// Real processes again: what the bounded wait has to get right is that it does NOT block in
// waitpid(), which only a child that outlives the bound can show. A fake scheduler stands in
// for this everywhere else, so dropping the WNOHANG would otherwise cost nothing here and a
// timed run would wait for the hung binary it exists to time out.

TEST(HostDispatchReapMicrotest, ABoundedWaitComesBackWhileTheChildIsStillRunning)
{
    const pid_t pid = spawnChild("/bin/sh", {"-c", "sleep 5"});
    ASSERT_GT(pid, 0);

    int status = -1;
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    const pid_t reaped = reapAnyChild(status, std::chrono::milliseconds{50});
    const std::chrono::milliseconds elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                              start);

    signalChildGroup(pid, SIGKILL);
    int discarded = 0;
    waitpid(pid, &discarded, 0);

    EXPECT_EQ(reaped, 0) << "a wait that elapsed reported a child as finished";
    EXPECT_EQ(status, 0);
    // Generous, so a loaded machine cannot fail it, and still far below the five seconds the
    // child runs for: anything near that means the wait blocked on the child after all.
    EXPECT_LT(elapsed, std::chrono::seconds{2}) << "the bounded wait blocked for "
                                                << elapsed.count() << " ms";
}

TEST(HostDispatchReapMicrotest, ABoundedWaitStillReapsAChildThatFinishesWithinIt)
{
    // The other half of the same path: the bound is a ceiling on the wait, not a replacement
    // for it, so a child that exits inside the window is still reaped and still carries its
    // exit status back.
    const pid_t pid = spawnChild("/bin/sh", {"-c", "exit 7"});
    ASSERT_GT(pid, 0);

    int status = 0;
    const pid_t reaped = reapAnyChild(status, std::chrono::seconds{5});

    EXPECT_EQ(reaped, pid);
    EXPECT_EQ(status, 7);
}

}  // namespace
