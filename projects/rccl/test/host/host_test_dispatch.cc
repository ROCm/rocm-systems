/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// rccl-UnitTestsHostAll -- one gtest-shaped front door for every host-only test binary.
//
// The host tree deliberately builds many small binaries rather than one: a binary exists
// wherever a production TU gates code at the preprocessor (the three init.cc arms) or needs a
// macro set binary-wide (WarpSpeed), and every new unit under test tends to bring another. That
// is good for the tests and bad for everything downstream, because each consumer then has to
// keep its own copy of the binary list and they all drift (test/host/run_host_tests.sh, the coco
// lane config, the coco llvm-cov --object list).
//
// This binary removes those lists. It discovers its sibling binaries at RUN time, speaks the
// subset of the gtest CLI its callers use (--gtest_list_tests, --gtest_filter, --gtest_output),
// forks each binary that owns a selected case, and merges their JUnit XML into the single report
// the caller asked for. Adding a host test binary therefore needs no change anywhere else.
//
// Case names are qualified with the owning binary's distinguishing suffix, e.g.
//
//     rccl-UnitTestsMicroInit-uncached   ->  MicroInitUncached/InitMicrotest.CommAllocHonoursUncached
//
// in gtest's own Prefix/Suite.Case shape, so `*InitMicrotest.*` still selects it. The
// qualification is not cosmetic: the three init arms and the two enqueue arms publish identical
// Suite.Case names, and without it their results would collide into one identity when every
// binary reports through one front door.
//
// Coverage: each child inherits LLVM_PROFILE_FILE, whose %p expands per process, so the children
// drop their own .profraw with no help from here. What the harness cannot know is which binaries
// those profiles came from -- llvm-cov needs an --object for each, and a profile whose object is
// missing is silently discarded. So every binary this dispatcher runs is appended to
// coverage-objects.txt next to the profiles, making the test tree, not the harness, the source of
// truth for the object set.

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace rccl_host_dispatch {

namespace fs = std::filesystem;

// Binaries this dispatcher fronts: every sibling executable whose name starts with this. The
// host tree's own naming convention is the registry; nothing enumerates the binaries by hand.
constexpr std::string_view kBinaryPrefix = "rccl-UnitTestsMicro";

// Dropped from a binary's name to form the case-name qualifier, leaving e.g. "Micro",
// "MicroInit-uncached". Kept distinct from kBinaryPrefix so the qualifier stays short.
constexpr std::string_view kQualifierStrip = "rccl-UnitTests";

// Written beside the .profraw files, naming every binary that contributed profile data.
constexpr std::string_view kCoverageManifest = "coverage-objects.txt";

// Longest --gtest_filter handed to one child. The kernel caps a single argv entry at
// MAX_ARG_STRLEN (32 pages, 128KiB on Linux) and rejects the exec outright with E2BIG, which the
// biggest host binary's 2199 case names comfortably exceed, so a long selection is split across
// several invocations of that binary rather than truncated.
constexpr size_t kMaxFilterChars = 60000;

// ── small helpers ───────────────────────────────────────────────────

// Whether *text* begins with *prefix*. C++20's std::string::starts_with in all but name; this
// tree is C++17, and the std::string::rfind(x, 0) == 0 idiom it replaces reads as a backwards
// search, which is not what any of its call sites mean.
constexpr bool startsWith(std::string_view text, std::string_view prefix)
{
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

// *text* as an integer, or *fallback* when it does not parse.
long toLong(std::string_view text, long fallback)
{
    long value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    return error == std::errc() ? value : fallback;
}

// *text* as a double, or *fallback* when it does not parse.
//
// std::from_chars for floating point is deliberately not used: it only reached libstdc++ in
// GCC 11, and the host tests also build under older toolchains where it is still missing.
double toDouble(std::string_view text, double fallback)
{
    const std::string copy(text);
    char *end = nullptr;
    const double value = std::strtod(copy.c_str(), &end);
    return end == copy.c_str() ? fallback : value;
}

// gtest's JSON time fields, e.g. "0.015s", as seconds. 0 when the string is not one.
double jsonSeconds(std::string_view text)
{
    if (!text.empty() && text.back() == 's') {
        text.remove_suffix(1);
    }
    return toDouble(text, 0.0);
}

// ── --gtest_list_tests parsing ──────────────────────────────────────

// One case as its binary reports it, kept split because only the listing knows where the suite
// ends: a parameterized case name can itself contain '.' (it is printed from the parameter), so
// rejoining and re-splitting a full name is not lossless.
struct TestCaseName {
    std::string suite;
    std::string test;

    bool operator==(const TestCaseName &other) const
    {
        return suite == other.suite && test == other.test;
    }
};

// The cases in one binary's --gtest_list_tests *output*.
//
// The format is a suite line ending in '.' at column 0, then its cases indented beneath it.
// Trailing "  # GetParam() = ..." annotations and any banner printed before the first suite line
// are discarded.
std::vector<TestCaseName> parseListTests(const std::string &output)
{
    std::vector<TestCaseName> cases;
    std::string suite;
    std::istringstream stream(output);
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const size_t comment = line.find("  # ");
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        while (!line.empty() && line.back() == ' ') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        if (line[0] != ' ') {
            // A suite line is "Name." -- anything else at column 0 is a banner we ignore, and
            // ignoring it must also drop the cases of the suite before it from the listing.
            suite = line.back() == '.' ? line.substr(0, line.size() - 1) : "";
            continue;
        }
        if (suite.empty()) {
            continue;
        }
        const size_t start = line.find_first_not_of(' ');
        cases.push_back(TestCaseName{suite, line.substr(start)});
    }
    return cases;
}

// ── name qualification ──────────────────────────────────────────────

// The case-name qualifier for *binary*, e.g. "rccl-UnitTestsMicroInit-uncached" ->
// "MicroInitUncached".
//
// The variant dashes are folded away rather than kept: gtest splits a --gtest_filter at its FIRST
// '-', so a qualifier containing one could never be named by a positive pattern -- "MicroInit-
// uncached/*" would read as "everything in MicroInit, except uncached/*".
//
// The plain "rccl-UnitTestsMicro" has nothing left after the strip, so it keeps "Micro": a
// qualifier must never be empty, or its cases would be indistinguishable from an unqualified name.
std::string qualifierFor(const std::string &binary)
{
    const std::string stripped = startsWith(binary, kQualifierStrip)
                                     ? binary.substr(kQualifierStrip.size())
                                     : binary;
    std::string qualifier;
    bool capitalise = false;
    for (const char c : stripped) {
        if (c == '-' || c == '_') {
            capitalise = true;
            continue;
        }
        qualifier += capitalise ? static_cast<char>(std::toupper(c)) : c;
        capitalise = false;
    }
    return qualifier.empty() ? std::string("Micro") : qualifier;
}

// The suite name the dispatcher publishes *suite* as: "<qualifier>/<Suite>".
std::string qualifiedSuite(const std::string &qualifier, const std::string &suite)
{
    return qualifier + "/" + suite;
}

// ── filter resolution ─────────────────────────────────────────────

// One case this dispatcher could run: the binary that owns it and the name it publishes.
struct CandidateCase {
    size_t binary = 0;        // index into the caller's binary list
    std::string suite;        // qualified
    TestCaseName raw;         // as the owning binary knows it
};

namespace {

// A registered stand-in with nothing to run. See :func:`resolveFilter`.
class NoopTest : public ::testing::Test {
    void TestBody() override {}
};

}  // namespace

// The indices of *candidates* that *filter* selects, decided by gtest itself.
//
// The dispatcher has to know which binary owns a selected case BEFORE it forks anything, which
// means answering "does this filter select this name?" in this process. Rather than reimplement
// gtest's filter grammar -- which is not just ':'-joined globs with one '-' half, but also the
// DISABLED_ rule, sharding and whatever the next release adds -- this registers the published
// names as empty tests and lets gtest resolve the filter against them, then reads back
// should_run(). gtest does not expose a matcher any other way: UnitTestOptions::FilterMatchesTest
// is declared only in src/gtest-internal-inl.h, which is not installed.
//
// The empty tests are "run" (the default printer is released first, so nothing is printed): the
// filter is applied inside UnitTest::Run(), and should_run() is meaningless before it. For the
// ~7000 cases of the whole host tree that costs about 15ms.
std::vector<size_t> resolveFilter(const std::vector<CandidateCase> &candidates,
                                  const std::string &filter, bool also_run_disabled)
{
    ::testing::GTEST_FLAG(filter) = filter;
    ::testing::GTEST_FLAG(also_run_disabled_tests) = also_run_disabled;

    for (const CandidateCase &candidate : candidates) {
        ::testing::RegisterTest(candidate.suite.c_str(), candidate.raw.test.c_str(), nullptr,
                                nullptr, __FILE__, __LINE__,
                                []() -> ::testing::Test * { return new NoopTest; });
    }

    ::testing::UnitTest *unit_test = ::testing::UnitTest::GetInstance();
    ::testing::TestEventListeners &listeners = unit_test->listeners();
    delete listeners.Release(listeners.default_result_printer());
    (void)RUN_ALL_TESTS();

    // Matched back by name rather than by registration order: gtest groups its reporting by
    // suite, so the two orders agree only while every suite's cases are contiguous here.
    std::unordered_set<std::string> runnable;
    for (int s = 0; s < unit_test->total_test_suite_count(); ++s) {
        const ::testing::TestSuite *suite = unit_test->GetTestSuite(s);
        for (int t = 0; t < suite->total_test_count(); ++t) {
            if (suite->GetTestInfo(t)->should_run()) {
                runnable.insert(std::string(suite->name()) + "." + suite->GetTestInfo(t)->name());
            }
        }
    }

    std::vector<size_t> selected;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (runnable.count(fmt::format("{}.{}", candidates[i].suite, candidates[i].raw.test)) > 0) {
            selected.push_back(i);
        }
    }
    return selected;
}

// Whether gtest would refuse to run *name* for being disabled.
//
// gtest's own rule: a DISABLED_ prefix on either half of the name, no wildcards involved.
bool isDisabled(const TestCaseName &name)
{
    constexpr std::string_view prefix = "DISABLED_";
    return startsWith(name.suite, prefix) || startsWith(name.test, prefix);
}

// One discovered binary and the cases the caller's filter selects from it.
struct Selection {
    std::string path;
    std::string qualifier;
    std::vector<TestCaseName> cases;   // as the binary itself knows them
    size_t runnable = 0;               // cases the binary holds that this run could run at all
};

// *candidates* at *indices*, grouped into one Selection per owning binary.
//
// *also_run_disabled* decides whether a binary's DISABLED_ cases count towards what it holds,
// and so whether selecting everything else still counts as selecting the whole binary.
std::vector<Selection> group(const std::string &directory,
                             const std::vector<std::string> &binaries,
                             const std::vector<CandidateCase> &candidates,
                             const std::vector<size_t> &indices, bool also_run_disabled)
{
    std::vector<size_t> held(binaries.size(), 0);
    for (const CandidateCase &candidate : candidates) {
        if (also_run_disabled || !isDisabled(candidate.raw)) {
            ++held[candidate.binary];
        }
    }
    std::vector<Selection> selections;
    for (const size_t index : indices) {
        const CandidateCase &candidate = candidates[index];
        const std::string path = directory + "/" + binaries[candidate.binary];
        if (selections.empty() || selections.back().path != path) {
            selections.push_back(Selection{path, qualifierFor(binaries[candidate.binary]), {},
                                           held[candidate.binary]});
        }
        selections.back().cases.push_back(candidate.raw);
    }
    return selections;
}

// The --gtest_filter(s) to hand *selection*'s binary, one per invocation of it.
//
// Normally an exact ':'-joined list of case names, never the caller's own pattern: that pattern
// is written against the qualified names and would, in a multi-arm binary, also match the cases
// of the arm it was meant to exclude. Two exceptions:
//
//   * every case the binary could run selected -> "*", much shorter in the logs. The DISABLED_
//     cases it holds are no reason to fall back to a list: naming them explicitly would not run
//     them either, but it would push the biggest binary's filter past the exec limit and split
//     its run into several invocations for nothing;
//   * a list past kMaxFilterChars -> split, because exec rejects an over-long argument.
std::vector<std::string> childFilters(const Selection &selection)
{
    if (selection.cases.size() == selection.runnable) {
        return {"*"};
    }
    std::vector<std::string> filters;
    std::string current;
    for (const TestCaseName &name : selection.cases) {
        const std::string entry = name.suite + "." + name.test;
        if (!current.empty() && current.size() + 1 + entry.size() > kMaxFilterChars) {
            filters.push_back(current);
            current.clear();
        }
        if (!current.empty()) {
            current += ":";
        }
        current += entry;
    }
    if (!current.empty()) {
        filters.push_back(current);
    }
    return filters;
}

// ── report merging ──────────────────────────────────────────────────

// The report formats gtest can write, and this dispatcher therefore has to merge.
//
// Both are real consumers: run_host_tests.sh asks for xml, and the coco test runner asks for
// json through GTEST_OUTPUT (tools/scripts/test_runner/lib/test_executor.py).
enum class ReportFormat { Xml, Json };

// *format* as gtest spells it in --gtest_output=<format>:<path>.
std::string_view formatName(ReportFormat format)
{
    return format == ReportFormat::Json ? "json" : "xml";
}

// One child's report, as merged into the dispatcher's own.
//
// Only the member matching the run's format is populated, except in crashFragment(), which
// fills both so the crash path never has to know which format is in play.
struct SuiteFragment {
    std::string xml;   // the child's <testsuite> elements, qualifier applied
    nlohmann::json suites = nlohmann::json::array();   // the same suites, as JSON objects
    long tests = 0;
    long failures = 0;
    long disabled = 0;
    long errors = 0;
    long skipped = 0;
    double time = 0.0;
};

// The value of *attribute* on the first element in *xml* that carries it, or *fallback*.
std::string_view attributeValue(std::string_view xml, std::string_view attribute,
                                std::string_view fallback)
{
    const std::string needle = fmt::format("{}=\"", attribute);
    const size_t start = xml.find(needle);
    if (start == std::string_view::npos) {
        return fallback;
    }
    const size_t from = start + needle.size();
    const size_t end = xml.find('"', from);
    if (end == std::string_view::npos) {
        return fallback;
    }
    return xml.substr(from, end - from);
}

// Rewrite every suite and case name in *xml* to carry *qualifier*.
//
// gtest writes the suite name twice -- as <testsuite name=...> and as each <testcase
// classname=...> -- and both are part of the identity a results consumer reconstructs, so both
// have to move together. <testcase name=...> is the case, not the suite, and is left alone.
std::string applyQualifier(std::string_view xml, std::string_view qualifier)
{
    constexpr std::string_view kSuiteAttribute = "<testsuite name=\"";
    constexpr std::string_view kClassAttribute = "classname=\"";

    const std::string insert = fmt::format("{}/", qualifier);
    std::string out;
    out.reserve(xml.size() + xml.size() / 8);
    size_t pos = 0;
    while (pos < xml.size()) {
        const size_t suite = xml.find(kSuiteAttribute, pos);
        const size_t klass = xml.find(kClassAttribute, pos);
        const size_t hit = std::min(suite, klass);
        if (hit == std::string_view::npos) {
            out.append(xml.substr(pos));
            break;
        }
        const size_t after =
            hit + (hit == suite ? kSuiteAttribute.size() : kClassAttribute.size());
        out.append(xml.substr(pos, after - pos));
        out.append(insert);
        pos = after;
    }
    return out;
}

// *text* (one child's whole JSON report) reduced to the fragment the dispatcher splices in.
//
// Parsed rather than pattern-matched: gtest's JSON escapes strings (its own report writes
// \/ for every path separator) and nests failure messages that contain braces and quotes, none
// of which survives being cut out of the text by hand. nlohmann also gives the only honest
// answer to "is this a gtest report at all?" -- a parse error, or a missing testsuites array.
std::optional<SuiteFragment> parseChildJsonReport(const std::string &text,
                                                  std::string_view qualifier)
{
    const nlohmann::json root = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
    if (root.is_discarded() || !root.is_object() || !root.contains("testsuites") ||
        !root["testsuites"].is_array()) {
        return std::nullopt;
    }

    SuiteFragment fragment;
    fragment.tests = root.value("tests", 0);
    fragment.failures = root.value("failures", 0);
    fragment.disabled = root.value("disabled", 0);
    fragment.errors = root.value("errors", 0);
    fragment.skipped = root.value("skipped", 0);
    fragment.time = jsonSeconds(root.value("time", std::string("0s")));

    // Suite and case keep their own identity; only the names that carry the suite's identity
    // move, exactly as in the XML path: the suite's name and each case's classname.
    for (nlohmann::json suite : root["testsuites"]) {
        if (!suite.is_object()) {
            continue;
        }
        suite["name"] = fmt::format("{}/{}", qualifier, suite.value("name", std::string()));
        if (suite.contains("testsuite") && suite["testsuite"].is_array()) {
            for (nlohmann::json &testcase : suite["testsuite"]) {
                if (testcase.is_object() && testcase.contains("classname")) {
                    testcase["classname"] =
                        fmt::format("{}/{}", qualifier,
                                    testcase.value("classname", std::string()));
                }
            }
        }
        fragment.suites.push_back(std::move(suite));
    }
    return fragment;
}

// *xml* (one child's whole report) reduced to the fragment the dispatcher splices in.
//
// Returns nullopt when the document has no <testsuites> root, which is how a child that died
// before writing its report, or wrote something that is not a gtest report, is recognised.
std::optional<SuiteFragment> parseChildXmlReport(std::string_view xml, std::string_view qualifier)
{
    const size_t root = xml.find("<testsuites");
    if (root == std::string_view::npos) {
        return std::nullopt;
    }
    const size_t rootEnd = xml.find('>', root);
    const size_t close = xml.rfind("</testsuites>");
    if (rootEnd == std::string_view::npos || close == std::string_view::npos || close < rootEnd) {
        return std::nullopt;
    }
    const std::string_view header = xml.substr(root, rootEnd - root);
    SuiteFragment fragment;
    fragment.tests = toLong(attributeValue(header, "tests", "0"), 0);
    fragment.failures = toLong(attributeValue(header, "failures", "0"), 0);
    fragment.disabled = toLong(attributeValue(header, "disabled", "0"), 0);
    fragment.errors = toLong(attributeValue(header, "errors", "0"), 0);
    fragment.skipped = toLong(attributeValue(header, "skipped", "0"), 0);
    fragment.time = toDouble(attributeValue(header, "time", "0"), 0.0);
    fragment.xml = applyQualifier(xml.substr(rootEnd + 1, close - rootEnd - 1), qualifier);
    return fragment;
}

// *text* (one child's whole report, in *format*) reduced to the fragment to splice in.
std::optional<SuiteFragment> parseChildReport(const std::string &text, std::string_view qualifier,
                                              ReportFormat format)
{
    return format == ReportFormat::Json ? parseChildJsonReport(text, qualifier)
                                        : parseChildXmlReport(text, qualifier);
}

// A one-case <testsuite> standing in for a child that produced no usable report.
//
// Synthesised rather than ignored: a binary that crashes, or that rejects an argument and exits
// without running anything, is a failure of the run and has to reach the results consumer as one.
SuiteFragment crashFragment(std::string_view qualifier, std::string_view detail)
{
    SuiteFragment fragment;
    fragment.tests = 1;
    fragment.failures = 1;
    fragment.xml = fmt::format(
        "\n  <testsuite name=\"{qualifier}/HostTestDispatch\" tests=\"1\" failures=\"1\""
        " disabled=\"0\" errors=\"0\" time=\"0\">\n"
        "    <testcase name=\"BinaryProducedAReport\" status=\"run\" result=\"completed\""
        " time=\"0\" classname=\"{qualifier}/HostTestDispatch\">\n"
        "      <failure message=\"{detail}\" type=\"\"/>\n"
        "    </testcase>\n"
        "  </testsuite>\n",
        fmt::arg("qualifier", qualifier), fmt::arg("detail", detail));

    // The same case as JSON. Spelled with the keys the results consumer actually reads -- a
    // non-empty failures array, status RUN and result COMPLETED -- because a case missing them
    // is classified as not-run and would drop the failure rather than report it (see
    // _leaf_case_status in tools/scripts/test_runner/lib/test_executor.py).
    const std::string suiteName = fmt::format("{}/HostTestDispatch", qualifier);
    fragment.suites.push_back(nlohmann::json{
        {"name", suiteName},
        {"tests", 1},
        {"failures", 1},
        {"disabled", 0},
        {"errors", 0},
        {"time", "0s"},
        {"testsuite",
         nlohmann::json::array(
             {nlohmann::json{{"name", "BinaryProducedAReport"},
                             {"status", "RUN"},
                             {"result", "COMPLETED"},
                             {"time", "0s"},
                             {"classname", suiteName},
                             {"failures", nlohmann::json::array({nlohmann::json{
                                              {"failure", detail}, {"type", ""}}})}}})}});
    return fragment;
}

// The merged report for *fragments*, as a complete gtest JSON document.
std::string mergeJsonReports(const std::vector<SuiteFragment> &fragments)
{
    long tests = 0, failures = 0, disabled = 0, errors = 0, skipped = 0;
    double time = 0.0;
    nlohmann::json suites = nlohmann::json::array();
    for (const SuiteFragment &fragment : fragments) {
        tests += fragment.tests;
        failures += fragment.failures;
        disabled += fragment.disabled;
        errors += fragment.errors;
        skipped += fragment.skipped;
        time += fragment.time;
        for (const nlohmann::json &suite : fragment.suites) {
            suites.push_back(suite);
        }
    }
    const nlohmann::json root{{"tests", tests},
                              {"failures", failures},
                              {"disabled", disabled},
                              {"errors", errors},
                              {"skipped", skipped},
                              {"time", fmt::format("{:g}s", time)},
                              {"name", "AllTests"},
                              {"testsuites", suites}};
    // Two-space indent and a trailing newline, as gtest's own JSON writer produces, so a
    // report from one binary and a merged one still diff cleanly against each other.
    return root.dump(2) + "\n";
}

// The merged report for *fragments*, as a complete JUnit XML document.
std::string mergeXmlReports(const std::vector<SuiteFragment> &fragments)
{
    long tests = 0, failures = 0, disabled = 0, errors = 0, skipped = 0;
    double time = 0.0;
    std::string body;
    for (const SuiteFragment &fragment : fragments) {
        tests += fragment.tests;
        failures += fragment.failures;
        disabled += fragment.disabled;
        errors += fragment.errors;
        skipped += fragment.skipped;
        time += fragment.time;
        body += fragment.xml;
    }
    // {:g} and not fmt's default {}: gtest's own reports carry the six-significant-digit form
    // ostream <<double produces, and a results consumer diffing the two should see no difference
    // between a child's time attribute and the total the dispatcher writes from it.
    return fmt::format(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<testsuites tests=\"{}\" failures=\"{}\" disabled=\"{}\" errors=\"{}\" skipped=\"{}\""
        " time=\"{:g}\" name=\"AllTests\">{}</testsuites>\n",
        tests, failures, disabled, errors, skipped, time, body);
}

// The merged report for *fragments*, in *format*.
std::string mergeReports(const std::vector<SuiteFragment> &fragments, ReportFormat format)
{
    return format == ReportFormat::Json ? mergeJsonReports(fragments) : mergeXmlReports(fragments);
}

// ── command line ────────────────────────────────────────────────────

// The help flags gtest itself recognises. Spelled out here because gtest offers no predicate
// for them: the decision has to be made before gtest ever sees the caller's argv.
constexpr std::string_view kHelpFlags[] = {"--help", "-h", "-?", "/?"};

// Whether *arg* asks for help rather than for a run.
//
// gtest also treats an UNRECOGNISED --gtest_* flag as a help request. That one cannot be
// mirrored here: knowing which --gtest_* flags exist is exactly the knowledge this dispatcher
// refuses to carry, so a misspelled flag still reaches the children, each of which prints help.
bool isHelpFlag(std::string_view arg)
{
    return std::find(std::begin(kHelpFlags), std::end(kHelpFlags), arg) != std::end(kHelpFlags);
}

// The caller's gtest arguments, split into what the dispatcher acts on and what it forwards.
struct Options {
    bool help = false;
    bool listTests = false;
    bool alsoRunDisabled = false;
    std::string filter = "*";
    std::string outputPath;       // empty: the caller asked for no report
    ReportFormat outputFormat = ReportFormat::Xml;
    std::vector<std::string> passthrough;
};

// A --gtest_output= value ("<format>:<path>", or a bare path) applied to *options*.
//
// gtest's own rules, mirrored: no prefix means xml, and a path ending in '/' is a directory to
// write the default filename into.
void applyOutputSpec(std::string_view value, Options *options)
{
    const size_t colon = value.find(':');
    std::string_view format = colon == std::string_view::npos ? "xml" : value.substr(0, colon);
    std::string path(colon == std::string_view::npos ? value : value.substr(colon + 1));
    options->outputFormat = format == "json" ? ReportFormat::Json : ReportFormat::Xml;
    if (!path.empty() && path.back() == '/') {
        path += fmt::format("test_detail.{}", formatName(options->outputFormat));
    }
    options->outputPath = path;
}

// *args* (argv[1:]) parsed into :class:`Options`, with *outputEnv* standing for GTEST_OUTPUT.
//
// Everything the dispatcher does not act on is forwarded verbatim, so flags it has never heard of
// (--gtest_repeat, --gtest_shuffle, a binary's own env-driven switches) keep working.
//
// GTEST_OUTPUT is read here rather than left to gtest because gtest would act on it in THIS
// process: it seeds the output flag at static init, so the no-op run that resolves the filter
// would write its own all-pass report over the caller's file and the real results would never
// reach it. Taking the variable ourselves turns it into the destination of the merged report.
Options parseArgs(const std::vector<std::string> &args, std::string_view outputEnv = "")
{
    constexpr std::string_view kFilterFlag = "--gtest_filter=";
    constexpr std::string_view kOutputFlag = "--gtest_output=";

    Options options;
    if (!outputEnv.empty()) {
        applyOutputSpec(outputEnv, &options);   // an explicit flag below overrides this
    }
    for (const std::string &arg : args) {
        if (isHelpFlag(arg)) {
            // Acted on and NOT forwarded: help is a question about this binary, and forwarding
            // it would fork every child to print its own copy of the same gtest help screen,
            // none of which writes a report -- which the run would then score as a failure.
            options.help = true;
        } else if (arg == "--gtest_list_tests") {
            options.listTests = true;
        } else if (arg == "--gtest_also_run_disabled_tests") {
            // Acted on AND forwarded: the children need it to run their DISABLED_ cases, and
            // the filter resolution here needs it to select them in the first place.
            options.alsoRunDisabled = true;
            options.passthrough.push_back(arg);
        } else if (startsWith(arg, kFilterFlag)) {
            options.filter = arg.substr(kFilterFlag.size());
        } else if (startsWith(arg, kOutputFlag)) {
            applyOutputSpec(std::string_view(arg).substr(kOutputFlag.size()), &options);
        } else {
            options.passthrough.push_back(arg);
        }
    }
    return options;
}

// ── process and filesystem plumbing ─────────────────────────────────

// Whether *path* is a regular file with an execute bit.
//
// The execute bit is asked of access() and not of the filesystem permissions: only access()
// answers it for THIS process, which is what deciding whether we can exec the file needs.
bool isExecutableFile(const fs::path &path)
{
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec) {
        return false;
    }
    return access(path.c_str(), X_OK) == 0;
}

// This executable's own path, resolved through /proc, or empty when that is unavailable.
std::string selfPath()
{
    std::error_code ec;
    const fs::path resolved = fs::read_symlink("/proc/self/exe", ec);
    return ec ? std::string() : resolved.string();
}

// The directory holding this executable, resolved through /proc so a PATH-less argv[0] works.
std::string selfDirectory(const char *argv0)
{
    std::string path = selfPath();
    if (path.empty()) {
        path = argv0 == nullptr ? "" : argv0;
    }
    const fs::path parent = fs::path(path).parent_path();
    return parent.empty() ? std::string(".") : parent.string();
}

// Every binary this dispatcher fronts, by name, in a stable order.
//
// Self-exclusion is by resolved path, not by name, so renaming this binary cannot make it
// recurse into itself.
std::vector<std::string> discoverBinaries(const std::string &directory, const std::string &self)
{
    std::vector<std::string> binaries;
    std::error_code ec;
    for (const fs::directory_entry &entry : fs::directory_iterator(directory, ec)) {
        const std::string name = entry.path().filename().string();
        if (!startsWith(name, kBinaryPrefix)) {
            continue;
        }
        if (entry.path().string() == self || !isExecutableFile(entry.path())) {
            continue;
        }
        binaries.push_back(name);
    }
    std::sort(binaries.begin(), binaries.end());
    return binaries;
}

// *argument* quoted for a /bin/sh command line.
std::string shellQuote(const std::string &argument)
{
    std::string quoted = "'";
    for (const char c : argument) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    return quoted + "'";
}

// *path*'s --gtest_list_tests output, or nullopt when the binary failed to produce one.
//
// The two are deliberately distinguishable: a healthy binary that holds no cases lists nothing
// and is not a problem, whereas one that cannot start (missing .so, an abort in static init)
// fails here FIRST, before anything else the dispatcher does to it, and must not be mistaken
// for the former. See enumerate().
std::optional<std::string> listTestsOutput(const std::string &path)
{
    const std::string command = fmt::format("{} --gtest_list_tests 2>/dev/null", shellQuote(path));
    std::unique_ptr<FILE, int (*)(FILE *)> pipe(popen(command.c_str(), "r"), pclose);
    if (pipe == nullptr) {
        return std::nullopt;
    }
    std::string output;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
        output += buffer;
    }
    // pclose() is the child's exit status, so it has to be read before the handle is released;
    // a non-zero status means the listing is partial, whatever arrived on stdout.
    if (pclose(pipe.release()) != 0) {
        return std::nullopt;
    }
    return output;
}

// ── enumeration ───────────────────────────────────────────────────

// What the discovered binaries hold, and which of them would not say.
struct Enumeration {
    std::vector<CandidateCase> candidates;
    std::vector<std::string> unlistable;   // binaries whose --gtest_list_tests failed
};

// How a binary is asked for its cases. A seam so the enumeration can be tested without one.
using ListingFn = std::function<std::optional<std::string>(const std::string &)>;

// Every case every discovered binary holds, published name and all, in discovery order.
//
// A binary that will not list its cases is RECORDED, never skipped. Dropping it would be the
// one failure this dispatcher must not have: discovery is what replaced the hand-kept binary
// lists, so a binary that cannot start would otherwise disappear from the run entirely -- unrun,
// unreported, absent from the coverage manifest -- and the run would still exit 0. The
// hand-kept list it replaced caught exactly this, by checking each binary wrote a report.
Enumeration enumerate(const std::string &directory, const std::vector<std::string> &binaries,
                      const ListingFn &listing)
{
    Enumeration enumeration;
    for (size_t i = 0; i < binaries.size(); ++i) {
        const std::optional<std::string> output =
            listing((fs::path(directory) / binaries[i]).string());
        if (!output) {
            enumeration.unlistable.push_back(binaries[i]);
            continue;
        }
        const std::string qualifier = qualifierFor(binaries[i]);
        for (const TestCaseName &name : parseListTests(*output)) {
            enumeration.candidates.push_back(
                CandidateCase{i, qualifiedSuite(qualifier, name.suite), name});
        }
    }
    return enumeration;
}

// The variables gtest sharding is driven by. Cleared from the environment the children
// inherit; see clearShardingEnvironment().
constexpr std::string_view kShardingVariables[] = {"GTEST_TOTAL_SHARDS", "GTEST_SHARD_INDEX",
                                                   "GTEST_SHARD_STATUS_FILE"};

// Drop the sharding variables, so the children run what they are told and nothing less.
//
// Sharding has to happen once, and it already has: this process resolved the caller's shard
// over the published names, and each child is then handed its cases BY NAME. A child that still
// saw these variables would shard that exact list a second time and run a fraction of it -- for
// N shards the run as a whole covers 1/N of the tree instead of all of it, and every shard
// still exits 0. test/common/ProcessIsolatedTestRunner.cpp clears them before its own forks for
// the same reason.
//
// Called after the filter resolution above, which is the legitimate use of these variables.
void clearShardingEnvironment()
{
    for (const std::string_view variable : kShardingVariables) {
        unsetenv(std::string(variable).c_str());
    }
}

// Run *path* with *args*, inheriting stdio, and return its exit status (128+signal if killed).
int runChild(const std::string &path, const std::vector<std::string> &args)
{
    std::vector<char *> argv;
    argv.push_back(const_cast<char *>(path.c_str()));
    for (const std::string &arg : args) {
        argv.push_back(const_cast<char *>(arg.c_str()));
    }
    argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        fmt::print(stderr, "host-test-dispatch: fork failed: {}\n", std::strerror(errno));
        return 1;
    }
    if (pid == 0) {
        execv(path.c_str(), argv.data());
        fmt::print(stderr, "host-test-dispatch: exec {} failed: {}\n", path,
                   std::strerror(errno));
        _exit(127);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 1;
}

// Read *path* whole, or return an empty string.
std::string readFile(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return "";
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

// The directory LLVM_PROFILE_FILE points into, or empty when coverage is not being collected.
//
// The variable holds a filename pattern (…/rccl_%h_%p_%m.profraw), so only its directory part is
// meaningful here, and a pattern with no directory part means the profiles land in the cwd.
std::string coverageDirectory()
{
    const char *profile = std::getenv("LLVM_PROFILE_FILE");
    if (profile == nullptr || *profile == '\0') {
        return "";
    }
    const fs::path parent = fs::path(profile).parent_path();
    return parent.empty() ? std::string(".") : parent.string();
}

// Record *binaries* as llvm-cov --object candidates next to the profile data.
//
// Appended, never rewritten, and deliberately not deduplicated: several dispatcher invocations
// (one per lane) write here concurrently, and an append of one short line is the only write that
// is atomic between them. The reader dedupes.
void recordCoverageObjects(const std::string &directory, const std::vector<std::string> &binaries)
{
    if (directory.empty() || binaries.empty()) {
        return;
    }
    const std::string path = (fs::path(directory) / kCoverageManifest).string();
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) {
        fmt::print(stderr, "host-test-dispatch: cannot record coverage objects in {}: {}\n", path,
                   std::strerror(errno));
        return;
    }
    for (const std::string &binary : binaries) {
        const std::string line = fmt::format("{}\n", binary);
        if (write(fd, line.c_str(), line.size()) < 0) {
            fmt::print(stderr, "host-test-dispatch: short write to {}: {}\n", path,
                       std::strerror(errno));
            break;
        }
    }
    close(fd);
}

}  // namespace rccl_host_dispatch

#ifndef RCCL_HOST_DISPATCH_NO_MAIN

namespace {

using namespace rccl_host_dispatch;

// A temporary directory for the children's reports, removed when the guard goes out of scope.
//
// mkdtemp and not std::filesystem: the standard has no atomic create-unique-directory, and
// inventing a name to hand fs::create_directory races every other dispatcher running beside us.
class ReportDirectory {
public:
    ReportDirectory()
    {
        std::string buffer = (fs::temp_directory_path() / "rccl-host-dispatch-XXXXXX").string();
        if (mkdtemp(buffer.data()) != nullptr) {
            path_ = buffer;
        }
    }

    ~ReportDirectory()
    {
        if (!path_.empty()) {
            std::error_code ec;
            fs::remove_all(path_, ec);
        }
    }

    ReportDirectory(const ReportDirectory &) = delete;
    ReportDirectory &operator=(const ReportDirectory &) = delete;

    bool valid() const { return !path_.empty(); }

    // Where part *part* of *qualifier*'s report should be written, in *format*.
    std::string reportPath(const std::string &qualifier, size_t part, ReportFormat format) const
    {
        return (fs::path(path_) /
                fmt::format("{}{}.{}", qualifier, part, formatName(format)))
            .string();
    }

private:
    std::string path_;
};

}  // namespace

int main(int argc, char **argv)
{
    const std::vector<std::string> args(argv + 1, argv + argc);
    const char *outputEnv = std::getenv("GTEST_OUTPUT");
    const Options options = parseArgs(args, outputEnv == nullptr ? "" : outputEnv);

    const std::string directory = selfDirectory(argv[0]);

    if (options.help) {
        // Answered before anything is enumerated: listing the cases costs a fork per binary,
        // and a caller asking what the flags are has not asked for any of that. The binary
        // names come from the directory scan alone, which is one readdir.
        fmt::print("{} runs the host-only test binaries beside it, as one gtest suite.\n",
                   fs::path(argv[0]).filename().string());
        for (const std::string &binary : discoverBinaries(directory, selfPath())) {
            fmt::print("  {}\n", binary);
        }
        fmt::print("\nCase names are qualified with the owning binary, e.g. MicroInit/Suite.Case,"
                   "\nso --gtest_filter patterns can name one binary's cases.\n\n");
        std::fflush(stdout);
        // gtest prints its own help from InitGoogleTest, so the caller sees the real flag list
        // for the gtest this was built against rather than a copy of it that can go stale.
        int help_argc = 2;
        char help_flag[] = "--help";
        char *help_argv[] = {argv[0], help_flag, nullptr};
        ::testing::InitGoogleTest(&help_argc, help_argv);
        return 0;
    }

    // gtest is initialised from an argv of our own, never the caller's: the flags the caller
    // passed are meant for the children, and letting gtest read --gtest_output here would have
    // the filter resolution in resolveFilter() overwrite the caller's report with an empty one.
    int gtest_argc = 1;
    char *gtest_argv[] = {argv[0], nullptr};
    ::testing::InitGoogleTest(&gtest_argc, gtest_argv);

    // The private argv is not enough on its own: gtest seeds this flag from GTEST_OUTPUT at
    // static init, so without clearing it the no-op run in resolveFilter() would write an
    // all-pass report to the caller's file -- a green report for a run that had not happened
    // yet, and that the merged report would never replace. parseArgs() has already taken the
    // variable as the destination for the real one.
    ::testing::GTEST_FLAG(output) = "";

    const std::vector<std::string> binaries = discoverBinaries(directory, selfPath());
    if (binaries.empty()) {
        fmt::print(stderr, "host-test-dispatch: no {}* binaries next to {} -- nothing to run\n",
                   kBinaryPrefix, directory);
        return 1;
    }

    const Enumeration enumeration = enumerate(directory, binaries, listTestsOutput);
    for (const std::string &binary : enumeration.unlistable) {
        fmt::print(stderr, "host-test-dispatch: {} could not list its tests\n", binary);
    }
    const std::vector<CandidateCase> &candidates = enumeration.candidates;
    const std::vector<size_t> chosen =
        resolveFilter(candidates, options.filter, options.alsoRunDisabled);

    if (options.listTests) {
        // gtest resolved the filter over the published names, so it can print the listing too,
        // in exactly the format a caller parsing it expects.
        ::testing::GTEST_FLAG(list_tests) = true;
        const int status = RUN_ALL_TESTS();
        // A listing missing a binary's cases is wrong, not merely short, and a caller building
        // a work list from it cannot tell. The warnings above name the binaries; this makes the
        // incompleteness something a script notices too.
        return enumeration.unlistable.empty() ? status : 1;
    }

    if (chosen.empty() && !candidates.empty()) {
        // Not an error -- gtest exits 0 for a filter that matches nothing, and the shared
        // categories yaml hands every binary every category's patterns, most of which it does
        // not own. But the usual cause here is a pattern written against a single binary, whose
        // names carry no qualifier, so say so rather than leave an empty report to explain it.
        fmt::print(stderr,
                   "host-test-dispatch: filter \"{}\" selected no case. Published names are"
                   " qualified, e.g. \"{}\"; lead the pattern with '*' to match any binary.\n",
                   options.filter, candidates.front().suite);
    }

    const std::vector<Selection> selections = group(directory, binaries, candidates, chosen, options.alsoRunDisabled);

    const ReportDirectory reportDir;
    if (!reportDir.valid()) {
        fmt::print(stderr, "host-test-dispatch: cannot create a report directory: {}\n",
                   std::strerror(errno));
        return 1;
    }

    clearShardingEnvironment();

    std::vector<SuiteFragment> fragments;
    std::vector<std::string> ran;
    int failures = 0;

    // A binary that would not list its cases is reported before any are run: it holds an
    // unknown number of them, so the run cannot be green whatever the rest of the tree does.
    for (const std::string &binary : enumeration.unlistable) {
        fragments.push_back(crashFragment(qualifierFor(binary),
                                          "binary could not list its tests -- it likely failed "
                                          "to start"));
        ++failures;
    }
    for (const Selection &selection : selections) {
        const std::vector<std::string> filters = childFilters(selection);
        ran.push_back(selection.path);
        for (size_t part = 0; part < filters.size(); ++part) {
            const std::string reportPath =
                reportDir.reportPath(selection.qualifier, part, options.outputFormat);
            // The child is asked for the format the caller wants, so the merge is always of
            // like with like and no report is ever transcoded between the two.
            std::vector<std::string> childArgs = {
                fmt::format("--gtest_filter={}", filters[part]),
                fmt::format("--gtest_output={}:{}", formatName(options.outputFormat), reportPath),
            };
            childArgs.insert(childArgs.end(), options.passthrough.begin(),
                             options.passthrough.end());

            const std::string part_note =
                filters.size() > 1 ? fmt::format(", part {}/{}", part + 1, filters.size())
                                   : std::string();
            fmt::print("----- {} ({} cases{}) -----\n", selection.qualifier,
                       selection.cases.size(), part_note);
            std::fflush(stdout);
            const int status = runChild(selection.path, childArgs);

            if (std::optional<SuiteFragment> fragment =
                    parseChildReport(readFile(reportPath), selection.qualifier,
                                     options.outputFormat)) {
                fragments.push_back(std::move(*fragment));
            } else {
                // No report: the binary died or never started. Its cases are unaccounted for, so
                // the run has to fail even if the exit status somehow did not say so.
                fragments.push_back(crashFragment(
                    selection.qualifier,
                    fmt::format("binary wrote no gtest report (exit status {})", status)));
                ++failures;
            }
            if (status != 0) {
                ++failures;
            }
            std::error_code ec;
            fs::remove(reportPath, ec);
        }
    }

    recordCoverageObjects(coverageDirectory(), ran);

    if (!options.outputPath.empty()) {
        std::ofstream report(options.outputPath, std::ios::binary | std::ios::trunc);
        if (!report) {
            fmt::print(stderr, "host-test-dispatch: cannot write {}: {}\n", options.outputPath,
                       std::strerror(errno));
            return 1;
        }
        report << mergeReports(fragments, options.outputFormat);
    }

    return failures == 0 ? 0 : 1;
}

#endif  // RCCL_HOST_DISPATCH_NO_MAIN
