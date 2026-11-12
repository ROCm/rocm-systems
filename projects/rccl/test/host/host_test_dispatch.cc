/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// rccl-UnitTestsHostAll -- one gtest-shaped front door for every host-only test binary.
//
// Discovers its rccl-UnitTestsMicro* siblings at run time, forks the ones owning a selected
// case, and merges their reports into the one the caller asked for. What it is for, how the
// case names are qualified, and how the coverage manifest is used: see MICROTEST_README.md,
// "rccl-UnitTestsHostAll".

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace rccl_host_dispatch {

namespace fs = std::filesystem;

// Binaries this dispatcher fronts: every sibling executable whose name starts with this. The
// host tree's own naming convention is the registry; nothing enumerates the binaries by hand.
constexpr std::string_view kBinaryPrefix = "rccl-UnitTestsMicro";

// The subdirectory holding every binary this dispatcher fronts, in the build tree and in an
// install alike. See binaryDirectory(), and RCCL_MICRO_SUBDIR in test/host/CMakeLists.txt.
constexpr std::string_view kMicroSubdirectory = "micro";

// Dropped from a binary's name to form the case-name qualifier, leaving e.g. "Micro",
// "MicroInit-uncached". Kept distinct from kBinaryPrefix so the qualifier stays short.
constexpr std::string_view kQualifierStrip = "rccl-UnitTests";

// The merged lcov tracefile written beside the .profraw files: the whole host tree's coverage
// in one object-free, hash-agnostic file, so no consumer has to pair profiles with binaries.
constexpr std::string_view kCoverageTracefile = "host-coverage.lcov";

// Appended to the merged report's own name to form the directory the children's console output
// is written into, one <Qualifier>.log per fronted binary.
constexpr std::string_view kLogDirectorySuffix = ".logs";

// Used when the caller asked for no report at all, so there is no name to derive one from.
constexpr std::string_view kDefaultLogDirectory = "host-test-dispatch.logs";

// Overrides that directory; "-" turns the redirection off and lets the children inherit this
// process's stdout and stderr.
constexpr std::string_view kLogDirectoryEnv = "RCCL_HOST_DISPATCH_LOG_DIR";

// How many fronted binaries may run at once. Defaults to half the machine's processors.
constexpr std::string_view kParallelismEnv = "NUM_PARALLEL_HOST_TEST_BINARIES";

// How long one child may run before it is terminated, in seconds. Unset or 0 means no limit,
// which is this dispatcher's own behaviour; a caller under an outer CI timeout sets it so that
// one hung binary costs its own cases rather than every binary's, since the merged report is
// only written once the last child is reaped.
constexpr std::string_view kChildTimeoutEnv = "RCCL_HOST_DISPATCH_CHILD_TIMEOUT";

// How long a child gets to act on the SIGTERM it is sent at its deadline before it is sent
// SIGKILL. Long enough for a stuck-in-a-test child to run its handlers and flush its log, short
// enough that a child ignoring the signal does not extend the run by much.
constexpr std::chrono::seconds kTerminateGrace{5};

// How often a timed reap looks for a finished child. Only a run with a child timeout polls at
// all -- without one the reap blocks as it always has -- and the interval only bounds how late
// a deadline is noticed, which against a timeout measured in seconds need not be tight.
constexpr std::chrono::milliseconds kReapPollInterval{5};

// Source files kept out of the exported coverage. The tests' own code is not the thing under
// test, and nvtx is vendored. Overridable for a consumer that wants a different view.
constexpr std::string_view kCoverageIgnoreRegex = "(/test/|nvtx)";

// Set to 0 to leave the children's profiles where they fall and skip the lcov export over them.
// For a consumer that reads the .profraw files itself and has no use for host-coverage.lcov.
constexpr std::string_view kCoverageExportEnv = "RCCL_HOST_COVERAGE_EXPORT";

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

// What a filter selected, split by whether the case will run.
//
// *suppressed* is gtest's third state, not a second kind of no: a DISABLED_ case the filter
// matched is not run, but it IS reported, as status="notrun" result="suppressed". Both lists
// are in candidate order.
struct FilterMatches {
    std::vector<size_t> runnable;
    std::vector<size_t> suppressed;
};

// The indices of *candidates* that *filter* selects, decided by gtest itself.
//
// The empty tests have to be "run" to be selected: should_run() is meaningless until the filter
// is applied inside UnitTest::Run(). Why gtest resolves this rather than a matcher of our own:
// MICROTEST_README.md, "rccl-UnitTestsHostAll".
FilterMatches resolveFilter(const std::vector<CandidateCase> &candidates,
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
    std::unordered_set<std::string> suppressed;
    for (int s = 0; s < unit_test->total_test_suite_count(); ++s) {
        const ::testing::TestSuite *suite = unit_test->GetTestSuite(s);
        for (int t = 0; t < suite->total_test_count(); ++t) {
            const ::testing::TestInfo *info = suite->GetTestInfo(t);
            const std::string name = std::string(suite->name()) + "." + info->name();
            if (info->should_run()) {
                runnable.insert(name);
            } else if (info->is_reportable()) {
                // gtest's own "matched the filter, report it but do not run it". Taking it from
                // gtest rather than re-deriving it from the DISABLED_ prefix keeps the two from
                // disagreeing about what belongs in a report.
                suppressed.insert(name);
            }
        }
    }

    FilterMatches matches;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const std::string name =
            fmt::format("{}.{}", candidates[i].suite, candidates[i].raw.test);
        if (runnable.count(name) > 0) {
            matches.runnable.push_back(i);
        } else if (suppressed.count(name) > 0) {
            matches.suppressed.push_back(i);
        }
    }
    return matches;
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
    // Matched but not run: named in the child's filter so it reports them as suppressed, and
    // kept apart from *cases* so they do not count towards the binary being wholly selected.
    std::vector<TestCaseName> suppressed;
    size_t runnable = 0;               // cases the binary holds that this run could run at all
};

// *candidates* at *indices*, grouped into one Selection per owning binary.
//
// *also_run_disabled* decides whether a binary's DISABLED_ cases count towards what it holds,
// and so whether selecting everything else still counts as selecting the whole binary.
std::vector<Selection> group(const std::string &directory,
                             const std::vector<std::string> &binaries,
                             const std::vector<CandidateCase> &candidates,
                             const FilterMatches &matches, bool also_run_disabled)
{
    std::vector<size_t> held(binaries.size(), 0);
    for (const CandidateCase &candidate : candidates) {
        if (also_run_disabled || !isDisabled(candidate.raw)) {
            ++held[candidate.binary];
        }
    }
    // The two lists are merged back into candidate order, so a binary gets one Selection
    // whichever list its cases came from and the selections stay in binary order -- which is
    // the order the merged report is assembled in.
    std::vector<Selection> selections;
    std::vector<size_t> slot(binaries.size(), std::numeric_limits<size_t>::max());
    size_t run = 0;
    size_t sup = 0;
    while (run < matches.runnable.size() || sup < matches.suppressed.size()) {
        const bool takeRunnable =
            sup == matches.suppressed.size() ||
            (run < matches.runnable.size() && matches.runnable[run] < matches.suppressed[sup]);
        const size_t index = takeRunnable ? matches.runnable[run++] : matches.suppressed[sup++];
        const CandidateCase &candidate = candidates[index];
        if (slot[candidate.binary] == std::numeric_limits<size_t>::max()) {
            slot[candidate.binary] = selections.size();
            selections.push_back(Selection{directory + "/" + binaries[candidate.binary],
                                           qualifierFor(binaries[candidate.binary]),
                                           {},
                                           {},
                                           held[candidate.binary]});
        }
        Selection &selection = selections[slot[candidate.binary]];
        (takeRunnable ? selection.cases : selection.suppressed).push_back(candidate.raw);
    }
    return selections;
}

// The --gtest_filter(s) to hand *selection*'s binary, one per invocation of it.
//
// An exact list of case names, never the caller's pattern, which would also select the
// identically named cases of a multi-arm binary's other arm. Why "*" and why the split:
// MICROTEST_README.md, "rccl-UnitTestsHostAll".
std::vector<std::string> childFilters(const Selection &selection)
{
    if (selection.cases.size() == selection.runnable) {
        return {"*"};
    }
    // The suppressed cases go in the filter too. gtest reports a DISABLED_ case it was asked
    // for as status="notrun" result="suppressed" without running it, which is exactly what the
    // whole-binary "*" above already gets -- so naming them is what makes a partial selection
    // report them the same way a total one does, with gtest still the one deciding.
    std::vector<TestCaseName> named = selection.cases;
    named.insert(named.end(), selection.suppressed.begin(), selection.suppressed.end());

    std::vector<std::string> filters;
    std::string current;
    for (const TestCaseName &name : named) {
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

// A --gtest_output= value ("<format>:<path>", a bare format, or a bare path) applied to *options*.
//
// gtest's own rules, mirrored: a colon-less value that is "xml" or "json" names the FORMAT and
// leaves the path at gtest's default, any other colon-less value is a path in xml, and a path
// ending in '/' is a directory to write the default filename into.
void applyOutputSpec(std::string_view value, Options *options)
{
    const size_t colon = value.find(':');
    const bool bareFormat = colon == std::string_view::npos && (value == "xml" || value == "json");
    std::string_view format =
        colon == std::string_view::npos
            ? (bareFormat ? value : std::string_view("xml"))
            : value.substr(0, colon);
    std::string path(colon == std::string_view::npos
                         ? (bareFormat ? std::string_view("./") : value)
                         : value.substr(colon + 1));
    options->outputFormat = format == "json" ? ReportFormat::Json : ReportFormat::Xml;
    if (!path.empty() && path.back() == '/') {
        path += fmt::format("test_detail.{}", formatName(options->outputFormat));
    }
    options->outputPath = path;
}

// *args* (argv[1:]) parsed into :class:`Options`, with *outputEnv* standing for GTEST_OUTPUT and
// *filterEnv* for GTEST_FILTER.
//
// Everything the dispatcher does not act on is forwarded verbatim, so flags it has never heard of
// (--gtest_repeat, --gtest_shuffle, a binary's own env-driven switches) keep working.
//
// GTEST_OUTPUT is read here rather than left to gtest because gtest would act on it in THIS
// process: it seeds the output flag at static init, so the no-op run that resolves the filter
// would write its own all-pass report over the caller's file and the real results would never
// reach it. Taking the variable ourselves turns it into the destination of the merged report.
//
// GTEST_FILTER is taken for the same reason in reverse: left in the environment it would reach
// the children, which hold UNQUALIFIED names, so a pattern written against the published
// qualified names would select nothing in any of them while this process still asked for every
// case. Taken here it is a default for the dispatcher's own filter, resolved over the published
// names like a --gtest_filter flag, and clearConsumedEnvironment() keeps it from the children.
// gtest's rule for a boolean GTEST_ variable: "0" is the only false value, unset aside. An empty
// value is true, which is why the unset case is a nullopt rather than an empty string.
bool environmentFlagIsSet(std::optional<std::string_view> value)
{
    return value && *value != "0";
}

Options parseArgs(const std::vector<std::string> &args, std::string_view outputEnv = "",
                  std::string_view filterEnv = "",
                  std::optional<std::string_view> disabledEnv = std::nullopt)
{
    constexpr std::string_view kFilterFlag = "--gtest_filter=";
    constexpr std::string_view kOutputFlag = "--gtest_output=";

    Options options;
    if (!outputEnv.empty()) {
        applyOutputSpec(outputEnv, &options);   // an explicit flag below overrides this
    }
    if (!filterEnv.empty()) {
        options.filter = std::string(filterEnv);   // an explicit flag below overrides this
    }
    if (environmentFlagIsSet(disabledEnv)) {
        // Acted on AND forwarded, exactly as the flag is: the selection here has to include the
        // DISABLED_ cases, and each child has to be told to run the ones it is handed. Without
        // this, a binary sent a name list would skip them while one sent "*" ran them.
        options.alsoRunDisabled = true;
        options.passthrough.emplace_back("--gtest_also_run_disabled_tests");
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
            if (!options.alsoRunDisabled) {   // the environment may have added it already
                options.passthrough.push_back(arg);
            }
            options.alsoRunDisabled = true;
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

// Where the fronted binaries live, given the directory holding this executable.
std::string binaryDirectory(const std::string &selfDir)
{
    return (fs::path(selfDir) / kMicroSubdirectory).string();
}

// ── the gtest environment the children are given ────────────────────
//
// gtest configures itself from GTEST_* variables as well as from flags, and a dispatcher sits in
// the one place where that is dangerous: this process reads them too. Each variable is therefore
// one of two things, and the table below says which.
//
//   Consume  this process acts on it, and re-expresses the decision as an explicit per-child
//            flag. A child that saw the variable as well would act on it a SECOND time, over a
//            case list that has already been narrowed -- running a fraction of what it was told
//            to run, and exiting 0 for it.
//   Forward  the variable means the same thing in a child as it does here and changes nothing
//            about which cases run, so the children keep it.
//
// An unknown GTEST_* variable is consumed, which is the safe default of the two: a variable a
// future gtest adds then costs a redundant strip and a line on stderr, where inheriting it could
// silently cost coverage. Nothing here needs updating for gtest to stay correct -- the table only
// buys back the forwarding of variables we have decided are harmless.
enum class Disposition { Consume, Forward };

constexpr std::string_view kGtestVariablePrefix = "GTEST_";

constexpr struct {
    std::string_view name;
    Disposition how;
} kGtestVariables[] = {
    // Resolved here over the published, qualified names; each child is handed its cases by name.
    {"GTEST_FILTER", Disposition::Consume},
    // Becomes the destination of the merged report; each child writes the report directory.
    {"GTEST_OUTPUT", Disposition::Consume},
    // Decides the selection here, so a child must not re-decide it for the list it is handed.
    {"GTEST_ALSO_RUN_DISABLED_TESTS", Disposition::Consume},
    // Sharding happens once, over the published names. A child that sharded its own list again
    // would run 1/N of it, so the run as a whole would cover 1/N of the tree and still be green.
    // test/common/ProcessIsolatedTestRunner.cpp keeps these from its own forks for the same
    // reason.
    {"GTEST_TOTAL_SHARDS", Disposition::Consume},
    {"GTEST_SHARD_INDEX", Disposition::Consume},
    {"GTEST_SHARD_STATUS_FILE", Disposition::Consume},
    // Presentation and repetition: identical meaning in a child, and no effect on the selection.
    {"GTEST_COLOR", Disposition::Forward},
    {"GTEST_BRIEF", Disposition::Forward},
    {"GTEST_PRINT_TIME", Disposition::Forward},
    {"GTEST_PRINT_UTF8", Disposition::Forward},
    {"GTEST_REPEAT", Disposition::Forward},
    {"GTEST_SHUFFLE", Disposition::Forward},
    {"GTEST_RANDOM_SEED", Disposition::Forward},
    {"GTEST_FAIL_FAST", Disposition::Forward},
    {"GTEST_BREAK_ON_FAILURE", Disposition::Forward},
    {"GTEST_THROW_ON_FAILURE", Disposition::Forward},
    {"GTEST_CATCH_EXCEPTIONS", Disposition::Forward},
    {"GTEST_DEATH_TEST_STYLE", Disposition::Forward},
    {"GTEST_STACK_TRACE_DEPTH", Disposition::Forward},
};

// Whether *name* is a GTEST_ variable the table names as harmless to inherit.
bool isForwardedVariable(std::string_view name)
{
    for (const auto &entry : kGtestVariables) {
        if (entry.name == name) {
            return entry.how == Disposition::Forward;
        }
    }
    return false;   // unknown: consumed, per the fail-closed default above
}

// Whether *name* is a GTEST_ variable the table does not name at all.
bool isUnknownGtestVariable(std::string_view name)
{
    if (!startsWith(name, kGtestVariablePrefix)) {
        return false;
    }
    return std::none_of(std::begin(kGtestVariables), std::end(kGtestVariables),
                        [name](const auto &entry) { return entry.name == name; });
}

// The GTEST_ variables currently set that a child must not see, in the environment's order.
//
// Read from the live environment rather than from the table, so a variable this build has never
// heard of is still found and still stripped.
std::vector<std::string> gtestVariablesToStrip()
{
    std::vector<std::string> names;
    for (char **entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
        const std::string_view assignment(*entry);
        const size_t equals = assignment.find('=');
        if (equals == std::string_view::npos) {
            continue;
        }
        const std::string_view name = assignment.substr(0, equals);
        if (startsWith(name, kGtestVariablePrefix) && !isForwardedVariable(name)) {
            names.emplace_back(name);
        }
    }
    return names;
}

// Drop those variables from THIS process's environment.
//
// Called between fork() and exec() in the children, never in the dispatcher: the dispatcher goes
// on using the originals after the first fork -- resolveFilter() shards over them -- and a child
// owns its own copy of the environment, so stripping there cannot reach back. Putting it at the
// fork sites is what makes the guarantee unconditional: there is no ordering in main() to get
// right, and a fork site added later gets it by construction.
void stripGtestVariablesInChild(const std::vector<std::string> &names)
{
    for (const std::string &name : names) {
        unsetenv(name.c_str());
    }
}

// Name on stderr any GTEST_ variable being stripped only because the table has not heard of it.
//
// The strip is the safe choice, but it is a choice made without knowing what the variable does,
// so it is said out loud rather than left to be discovered as missing coverage.
void warnAboutUnknownGtestVariables(const std::vector<std::string> &names)
{
    for (const std::string &name : names) {
        if (isUnknownGtestVariable(name)) {
            fmt::print(stderr,
                       "host-test-dispatch: {} is not a variable this build knows; keeping it"
                       " from the children rather than risk it narrowing their runs\n",
                       name);
        }
    }
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

// LLVM_PROFILE_FILE for the child at *path*, given the *pattern* this process inherited.
std::string childProfileFile(std::string_view pattern, std::string_view path)
{
    if (pattern.empty()) {
        return "";
    }
    const fs::path profile{std::string(pattern)};
    const std::string binary = fs::path(std::string(path)).filename().string();
    return (profile.parent_path() / binary / profile.filename()).string();
}

// The LLVM_PROFILE_FILE the child at *path* should run under, read from this process's
// environment. Empty when coverage is not being collected.
std::string childProfileFile(const std::string &path)
{
    const char *pattern = std::getenv("LLVM_PROFILE_FILE");
    return childProfileFile(pattern == nullptr ? std::string_view() : std::string_view(pattern),
                            path);
}

// *path*'s --gtest_list_tests output, or nullopt when the binary failed to produce one.
//
// The two are deliberately distinguishable: a healthy binary that holds no cases lists nothing
// and is not a problem, whereas one that cannot start (missing .so, an abort in static init)
// fails here FIRST, before anything else the dispatcher does to it, and must not be mistaken
// for the former. See enumerate().
std::optional<std::string> listTestsOutput(const std::string &path)
{
    // The listing children are fork sites like any other, so they get the same stripped
    // environment. popen() hands them this process's one, which must keep its originals, so the
    // strip is expressed to the shell instead -- same list, same guarantee.
    std::string environment;
    for (const std::string &name : gtestVariablesToStrip()) {
        environment += fmt::format("-u {} ", shellQuote(name));
    }

    const std::string profile = childProfileFile(path);
    if (!profile.empty()) {
        environment += fmt::format("LLVM_PROFILE_FILE={} ", shellQuote(profile));
    }
    const std::string command =
        fmt::format("env {}{} --gtest_list_tests 2>/dev/null", environment, shellQuote(path));
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

// The directory the children's logs belong in for a run whose merged report is *outputPath*.
std::string childLogDirectory(std::string_view outputPath)
{
    if (outputPath.empty()) {
        return std::string(kDefaultLogDirectory);
    }
    const fs::path report{std::string(outputPath)};
    const fs::path parent = report.parent_path();
    const std::string name = report.stem().string() + std::string(kLogDirectorySuffix);
    return (parent.empty() ? fs::path(name) : parent / name).string();
}

// Where *qualifier*'s console output goes within *directory*.
std::string childLogPath(std::string_view directory, std::string_view qualifier)
{
    return (fs::path(std::string(directory)) / fmt::format("{}.log", qualifier)).string();
}

// The log directory this run should use, given the merged report at *outputPath* and the
// environment. Empty when the caller asked for the children's output on the console instead.
std::string resolveLogDirectory(std::string_view outputPath, const char *override_)
{
    if (override_ == nullptr || *override_ == '\0') {
        return childLogDirectory(outputPath);
    }
    return std::string_view(override_) == "-" ? std::string() : std::string(override_);
}

// Make *directory* ready to hold the children's logs, or say why it cannot be trusted with them.
std::optional<std::string> prepareLogDirectory(const std::string &directory)
{
    const fs::path path{directory};
    if (path.has_parent_path() && !path.parent_path().empty()) {
        std::error_code ec;
        fs::create_directories(path.parent_path(), ec);
    }
    if (mkdir(directory.c_str(), 0700) == 0) {
        return std::nullopt;
    }
    if (errno != EEXIST) {
        return std::string(std::strerror(errno));
    }
    struct stat info {};
    if (lstat(directory.c_str(), &info) != 0) {
        return std::string(std::strerror(errno));
    }
    if (S_ISLNK(info.st_mode)) {
        return "it is a symbolic link";
    }
    if (!S_ISDIR(info.st_mode)) {
        return "it is not a directory";
    }
    if (info.st_uid != geteuid()) {
        return fmt::format("it belongs to uid {}, not to this process", info.st_uid);
    }
    return std::nullopt;
}

// How many binaries run at once.
size_t resolveParallelism(const char *request, size_t fallbackJobs)
{
    const size_t fallback = fallbackJobs < 1 ? 1 : fallbackJobs;
    if (request == nullptr || *request == '\0') {
        return fallback;
    }
    const long requested = toLong(std::string_view(request), 0);
    return requested >= 1 ? static_cast<size_t>(requested) : fallback;
}

// How long one child may run, given *request* from the environment. Zero means no limit.
//
// A request that is not a positive number is said rather than silently ignored: a caller that
// set the variable wanted a limit, and a typo that quietly leaves the run unbounded is the one
// outcome they were trying to avoid. "0" is an explicit opt-out and says nothing.
std::chrono::seconds resolveChildTimeout(const char *request)
{
    if (request == nullptr || *request == '\0') {
        return std::chrono::seconds{0};
    }
    const long seconds = toLong(std::string_view(request), -1);
    if (seconds < 0) {
        fmt::print(stderr,
                   "host-test-dispatch: {}=\"{}\" is not a number of seconds -- no child timeout"
                   " will be applied\n",
                   kChildTimeoutEnv, request);
        return std::chrono::seconds{0};
    }
    return std::chrono::seconds{seconds};
}

// The number of processors available to this process, or 1 when that cannot be determined.
size_t detectedParallelism()
{
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    return online > 0 ? static_cast<size_t>(online) : 1;
}

// How many binaries an unconfigured run puts in flight on a machine with *processors* of them.
//
// Half, so the run leaves the machine usable for whatever else is on it.
size_t defaultParallelism(size_t processors)
{
    return processors / 2 < 1 ? 1 : processors / 2;
}

// Start *path* with *args* and return its pid, or -1 when it could not be forked.
//
// *logPath*, when set, is the file the child's stdout and stderr are redirected to.
// Empty means inherit this process's stdio.
pid_t spawnChild(const std::string &path, const std::vector<std::string> &args,
                 const std::string &logPath = "")
{
    std::vector<char *> argv;
    argv.push_back(const_cast<char *>(path.c_str()));
    for (const std::string &arg : args) {
        argv.push_back(const_cast<char *>(arg.c_str()));
    }
    argv.push_back(nullptr);

    // Collected before the fork: it reads the environment, which is cheaper and simpler to do
    // here than in the narrow async-signal-safe window after it.
    const std::vector<std::string> strip = gtestVariablesToStrip();
    const std::string profile = childProfileFile(path);

    const pid_t pid = fork();
    if (pid < 0) {
        fmt::print(stderr, "host-test-dispatch: fork failed: {}\n", std::strerror(errno));
        return -1;
    }
    if (pid == 0) {
        // Its own process group, so a timeout can signal the whole tree rather than the one pid:
        // a test binary forks too -- death tests, ProcessIsolatedTestRunner -- and those
        // grandchildren would otherwise survive the kill and keep the CI node. The cost is that
        // the children leave the terminal's foreground group, so an interactive Ctrl-C reaches
        // the dispatcher alone; it signals them on the way out.
        setpgid(0, 0);
        // In the child, on its own copy of the environment: what the dispatcher has already
        // decided must not be decided again over the case list this child was handed.
        stripGtestVariablesInChild(strip);
        if (!profile.empty()) {
            setenv("LLVM_PROFILE_FILE", profile.c_str(), 1);
        }
        if (!logPath.empty()) {
            const int fd =
                open(logPath.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW, 0644);
            if (fd < 0) {
                fmt::print(stderr, "host-test-dispatch: cannot write {}: {} -- {} will print to"
                                   " the console instead\n",
                           logPath, std::strerror(errno), path);
            } else {
                dup2(fd, STDOUT_FILENO);
                dup2(fd, STDERR_FILENO);
                if (fd > STDERR_FILENO) {
                    close(fd);
                }
            }
        }
        execv(path.c_str(), argv.data());
        fmt::print(stderr, "host-test-dispatch: exec {} failed: {}\n", path,
                   std::strerror(errno));
        _exit(127);
    }
    // Again in the parent: whichever of the two runs first, the group is in place before the
    // first signal could be sent. EACCES means the child has already exec'd, which it can only
    // have done after setting it itself.
    setpgid(pid, pid);
    return pid;
}

// Send *signal* to everything *pid* has spawned, and to *pid* itself.
//
// Falls back to the single pid when there is no such group, which is what a child that died
// between the fork and its setpgid() leaves behind.
// Set from the SIGTERM/SIGINT handler; read by the scheduler between reaps.
volatile std::sig_atomic_t g_stopRequested = 0;

extern "C" void requestStop(int)
{
    g_stopRequested = 1;
}

// Ask for SIGTERM and SIGINT to stop the run rather than end it outright, so the children are
// killed with it and a report of what did finish is still written.
//
// Deliberately without SA_RESTART: the scheduler spends an untimed run blocked in waitpid(), and
// the EINTR is what gets it back to the flag this sets.
void installStopHandler()
{
    struct sigaction action {};
    action.sa_handler = requestStop;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGINT, &action, nullptr);
}

void signalChildGroup(pid_t pid, int signal)
{
    if (::kill(-pid, signal) == 0 || errno != ESRCH) {
        return;
    }
    ::kill(pid, signal);
}

// Wait for whichever spawned child finishes first, setting *exitStatus* to its status (128+signal
// if killed) and returning its pid; -1 when there is nothing left to wait for.
//
// *within*, when set, bounds the wait: 0 comes back if it elapses with no child finished, and the
// caller is then the one to decide what that means. Only a bounded wait polls; an unbounded one
// blocks in waitpid() and costs nothing, which is what a run with no child timeout does.
pid_t reapAnyChild(int &exitStatus, std::optional<std::chrono::milliseconds> within)
{
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + within.value_or(std::chrono::milliseconds{0});
    for (;;) {
        int status = 0;
        const pid_t pid = waitpid(-1, &status, within ? WNOHANG : 0);
        if (pid < 0 && errno == EINTR) {
            // A signal arrived. The caller is the one that knows whether it means the run is
            // being stopped, so hand control back rather than re-entering the wait.
            exitStatus = 0;
            return 0;
        }
        if (pid < 0) {
            exitStatus = 1;
            return -1;
        }
        if (pid > 0) {
            exitStatus = WIFEXITED(status) ? WEXITSTATUS(status)
                                           : (WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 1);
            return pid;
        }
        const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            exitStatus = 0;
            return 0;
        }
        std::this_thread::sleep_for(std::min<std::chrono::nanoseconds>(kReapPollInterval,
                                                                      deadline - now));
    }
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

// ── coverage merge ──────────────────────────────────────────────────

// One source file's coverage, as an lcov tracefile records it.
//
// Every container is keyed rather than appended to, because the merge is a union and because
// llvm-cov emits the same entity more than once within a single record (a line covered by two
// regions, a function instantiated twice). Ordered maps, so the output of a merge depends only
// on its inputs and two runs produce byte-identical tracefiles.
struct LcovSource {
    std::map<std::string, long> functionLines;        // FN: mangled name -> declaration line
    std::map<std::string, long long> functionHits;    // FNDA: mangled name -> execution count
    std::map<long, long long> lines;                  // DA: line -> execution count
    // BRDA: (line, block, branch) -> taken count; nullopt for llvm-cov's '-', "not evaluated".
    std::map<std::tuple<long, long, long>, std::optional<long long>> branches;
};

// A coverage counter from an lcov record, saturated rather than wrapped or rejected.
//
// llvm-cov prints its counters as unsigned 64-bit and does sometimes print one that has wrapped
// -- a counter expression evaluating below zero comes out as 18446744073709551591 -- which does
// not fit a signed long and would parse as a failure. Reading that as 0 turns a covered line
// into an uncovered one in the merged view, which is the one failure this file must not have,
// so it saturates instead. Only the non-zeroness of a count is ever read downstream.
long long toCount(std::string_view text)
{
    unsigned long long value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error == std::errc::result_out_of_range) {
        return std::numeric_limits<long long>::max();
    }
    if (error != std::errc()) {
        return 0;
    }
    return static_cast<long long>(
        std::min<unsigned long long>(value, std::numeric_limits<long long>::max()));
}

// *left* + *right*, saturating. Two counters that each nearly fill the type are summed when the
// same line is reached by several binaries, and a wrapped total would read as uncovered.
long long addCounts(long long left, long long right)
{
    constexpr long long kMax = std::numeric_limits<long long>::max();
    return left > kMax - right ? kMax : left + right;
}

// A whole tracefile: its source files, keyed by path.
using LcovData = std::map<std::string, LcovSource>;

// Split *text* on *separator*, keeping empty fields. Used for the comma-separated bodies of the
// lcov record lines, where an empty field is a parse failure rather than something to skip.
std::vector<std::string_view> splitOn(std::string_view text, char separator)
{
    std::vector<std::string_view> fields;
    size_t start = 0;
    for (size_t i = 0; i <= text.size(); ++i) {
        if (i == text.size() || text[i] == separator) {
            fields.push_back(text.substr(start, i - start));
            start = i + 1;
        }
    }
    return fields;
}

// Accumulate the lcov tracefile in *text* into *into*.
//
// The counting lines an input carries -- FNF/FNH/LF/LH/BRF/BRH -- are deliberately ignored and
// recomputed by formatLcov(). They are not summaries OF the records beside them: llvm-cov counts
// its own internal regions there, so a record with 648 FN lines can be headed FNF:258, and
// summing those numbers across binaries would give totals matching neither the records nor
// anything lcov would compute from them.
void parseLcov(std::string_view text, LcovData &into)
{
    LcovSource *current = nullptr;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = std::min(text.find('\n', start), text.size());
        const std::string_view line = text.substr(start, end - start);
        start = end + 1;

        if (startsWith(line, "SF:")) {
            current = &into[std::string(line.substr(3))];
        } else if (line == "end_of_record") {
            current = nullptr;
        } else if (current == nullptr) {
            continue;   // a record line outside any SF: block has no file to belong to
        } else if (startsWith(line, "FN:")) {
            // FN:<line>,<name>. The name may itself contain commas (a template parameter list),
            // so the split is on the FIRST comma only.
            const std::string_view body = line.substr(3);
            const size_t comma = body.find(',');
            if (comma != std::string_view::npos) {
                const std::string name(body.substr(comma + 1));
                const long declaration = toLong(body.substr(0, comma), -1);
                // Lowest line wins, so a name seen at several lines lands somewhere real.
                const auto [entry, inserted] = current->functionLines.emplace(name, declaration);
                if (!inserted && declaration >= 0 && declaration < entry->second) {
                    entry->second = declaration;
                }
            }
        } else if (startsWith(line, "FNDA:")) {
            const std::string_view body = line.substr(5);
            const size_t comma = body.find(',');
            if (comma != std::string_view::npos) {
                long long &hits = current->functionHits[std::string(body.substr(comma + 1))];
                hits = addCounts(hits, toCount(body.substr(0, comma)));
            }
        } else if (startsWith(line, "DA:")) {
            const std::vector<std::string_view> fields = splitOn(line.substr(3), ',');
            if (fields.size() >= 2) {
                long long &count = current->lines[toLong(fields[0], 0)];
                count = addCounts(count, toCount(fields[1]));
            }
        } else if (startsWith(line, "BRDA:")) {
            const std::vector<std::string_view> fields = splitOn(line.substr(5), ',');
            if (fields.size() >= 4) {
                const auto key = std::make_tuple(toLong(fields[0], 0), toLong(fields[1], 0),
                                                 toLong(fields[2], 0));
                std::optional<long long> &taken = current->branches[key];
                if (fields[3] != "-") {
                    // A branch one binary never evaluated and another took is taken: '-' is the
                    // absence of information, so any count at all displaces it.
                    taken = addCounts(taken.value_or(0), toCount(fields[3]));
                }
            }
        }
    }
}

// *data* as an lcov tracefile.
//
// The FNF/FNH/LF/LH/BRF/BRH lines are computed here from the records themselves, which is also
// how lcov and genhtml treat the ones they read, so the totals a consumer reports are the ones
// implied by the merged records.
std::string formatLcov(const LcovData &data)
{
    std::string out;
    for (const auto &[path, source] : data) {
        out += fmt::format("SF:{}\n", path);
        long functionsHit = 0;
        for (const auto &[name, line] : source.functionLines) {
            out += fmt::format("FN:{},{}\n", line, name);
        }
        for (const auto &[name, hits] : source.functionHits) {
            out += fmt::format("FNDA:{},{}\n", hits, name);
            if (hits > 0) {
                ++functionsHit;
            }
        }
        out += fmt::format("FNF:{}\n", source.functionLines.size());
        out += fmt::format("FNH:{}\n", functionsHit);

        long branchesHit = 0;
        for (const auto &[key, taken] : source.branches) {
            const auto [line, block, branch] = key;
            out += fmt::format("BRDA:{},{},{},{}\n", line, block, branch,
                               taken ? std::to_string(*taken) : std::string("-"));
            if (taken && *taken > 0) {
                ++branchesHit;
            }
        }
        if (!source.branches.empty()) {
            out += fmt::format("BRF:{}\n", source.branches.size());
            out += fmt::format("BRH:{}\n", branchesHit);
        }

        long linesHit = 0;
        for (const auto &[line, count] : source.lines) {
            out += fmt::format("DA:{},{}\n", line, count);
            if (count > 0) {
                ++linesHit;
            }
        }
        out += fmt::format("LF:{}\n", source.lines.size());
        out += fmt::format("LH:{}\n", linesHit);
        out += "end_of_record\n";
    }
    return out;
}

// *contents* (whole lcov tracefiles) merged into one.
//
// The union is per line and per branch: a line is covered if ANY input covered it. That is what
// makes this the right merge for the host tree, where several binaries compile the same source
// under different #defines -- their coverage-mapping hashes differ, so the profiles themselves
// cannot be pooled, but the lines they each reached can be.
std::string mergeLcovTracefiles(const std::vector<std::string> &contents)
{
    LcovData merged;
    for (const std::string &tracefile : contents) {
        parseLcov(tracefile, merged);
    }
    return formatLcov(merged);
}

// How a shell command is run, returning its exit status. A seam, so the export can be driven in
// a test without llvm-profdata and llvm-cov on the machine.
using CommandFn = std::function<int(const std::string &)>;

// Run *command* through /bin/sh and return its exit status.
int runShell(const std::string &command)
{
    const int status = std::system(command.c_str());
    if (status < 0) {
        return 1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

// The llvm tool named *name*, preferring the ROCm toolchain's own.
//
// The tools have to match the clang that built the binaries -- a newer llvm-profdata rejects an
// older profile format outright -- and the one beside the compiler matches by construction.
// Falls back to the bare name, i.e. to PATH, which is also what a non-ROCm toolchain wants.
std::string llvmTool(std::string_view name)
{
    const char *rocm = std::getenv("ROCM_PATH");
    const fs::path candidate =
        fs::path(rocm == nullptr || *rocm == '\0' ? "/opt/rocm" : rocm) / "llvm" / "bin" / name;
    std::error_code ec;
    return fs::is_regular_file(candidate, ec) ? candidate.string() : std::string(name);
}

// The command that merges *profiles* into the single .profdata at *output*.
std::string profdataMergeCommand(const std::string &tool, const std::vector<std::string> &profiles,
                                 const std::string &output)
{
    std::string command = fmt::format("{} merge -sparse", shellQuote(tool));
    for (const std::string &profile : profiles) {
        command += fmt::format(" {}", shellQuote(profile));
    }
    return command + fmt::format(" -o {}", shellQuote(output));
}

// The command that exports *binary*'s coverage, under *profdata*, as the lcov tracefile *output*.
//
// *binary* is the only --object: it is the only one its profiles were produced by, and llvm-cov
// rejects a profile read against an object it was not built from as mismatched data. Pairing the
// two is the whole reason this runs here rather than in a harness downstream, which would have to
// reconstruct the pairing from the directory layout.
std::string lcovExportCommand(const std::string &tool, const std::string &binary,
                              const std::string &profdata, std::string_view ignoreRegex,
                              const std::string &output)
{
    return fmt::format("{} export {} -instr-profile={} --ignore-filename-regex={} -format=lcov"
                       " > {}",
                       shellQuote(tool), shellQuote(binary), shellQuote(profdata),
                       shellQuote(std::string(ignoreRegex)), shellQuote(output));
}

// Every .profraw directly under *directory*, sorted.
std::vector<std::string> profilesIn(const std::string &directory)
{
    std::vector<std::string> profiles;
    std::error_code ec;
    for (const fs::directory_entry &entry : fs::directory_iterator(directory, ec)) {
        if (entry.path().extension() == ".profraw") {
            profiles.push_back(entry.path().string());
        }
    }
    std::sort(profiles.begin(), profiles.end());
    return profiles;
}

// Turn the children's profiles into one merged lcov tracefile beside them.
//
// Best effort throughout: a run whose tests all passed must not fail because llvm-cov is not
// installed, so every problem is a warning and the tracefile is simply whatever could be
// exported. Silence would be worse than either -- a missing tracefile reads downstream as "this
// code is not covered" -- so each skipped binary says so by name.
// Whether the lcov export should run at all, given *request* from the environment.
//
// Only "0" turns it off, so a caller who has not heard of this gets the export as before. The
// profiles themselves are unaffected either way: they are written by the children, which were
// handed their LLVM_PROFILE_FILE long before this.
bool coverageExportRequested(const char *request)
{
    return request == nullptr || std::string_view(request) != "0";
}

void writeMergedCoverage(const std::string &directory, const std::vector<std::string> &binaries,
                         const CommandFn &run)
{
    if (directory.empty() || binaries.empty()) {
        return;
    }
    if (!coverageExportRequested(std::getenv(std::string(kCoverageExportEnv).c_str()))) {
        return;
    }

    std::vector<std::pair<std::string, std::vector<std::string>>> pending;
    for (const std::string &binary : binaries) {
        const fs::path binaryDir = fs::path(directory) / fs::path(binary).filename();
        std::vector<std::string> profiles = profilesIn(binaryDir.string());
        if (!profiles.empty()) {   // never ran, or ran uninstrumented; nothing to export either way
            pending.emplace_back(binary, std::move(profiles));
        }
    }
    if (pending.empty()) {
        return;
    }

    const char *ignore = std::getenv("RCCL_HOST_COVERAGE_IGNORE_REGEX");
    const std::string ignoreRegex =
        ignore == nullptr ? std::string(kCoverageIgnoreRegex) : std::string(ignore);
    const std::string profdataTool = llvmTool("llvm-profdata");
    const std::string covTool = llvmTool("llvm-cov");

    // Each binary costs an llvm-profdata merge and an llvm-cov export over the whole hipified
    // tree, so this phase runs for a while after the last test has printed its result. It says
    // what it is doing, and how far through it is, rather than leave the run looking hung.
    fmt::print("----- coverage: exporting {} binaries into {} -----\n", pending.size(),
               (fs::path(directory) / kCoverageTracefile).string());
    std::fflush(stdout);

    std::vector<std::string> tracefiles;
    size_t exported = 0;
    for (const auto &[binary, profiles] : pending) {
        fmt::print("[{}/{}] {}\n", ++exported, pending.size(),
                   fs::path(binary).filename().string());
        std::fflush(stdout);
        const fs::path binaryDir = fs::path(directory) / fs::path(binary).filename();
        const std::string profdata = (binaryDir / "merged.profdata").string();
        const std::string tracefile = (binaryDir / "coverage.lcov").string();
        if (run(profdataMergeCommand(profdataTool, profiles, profdata)) != 0) {
            fmt::print(stderr, "host-test-dispatch: llvm-profdata failed for {} -- its coverage"
                               " is missing from {}\n",
                       binary, kCoverageTracefile);
            continue;
        }
        if (run(lcovExportCommand(covTool, binary, profdata, ignoreRegex, tracefile)) != 0) {
            fmt::print(stderr, "host-test-dispatch: llvm-cov export failed for {} -- its coverage"
                               " is missing from {}\n",
                       binary, kCoverageTracefile);
            // The redirect created the tracefile before llvm-cov ran; leaving the empty file
            // behind would make run_host_tests.sh treat the export as done and skip its fallback.
            std::error_code ec;
            fs::remove(tracefile, ec);
            continue;
        }
        tracefiles.push_back(readFile(tracefile));
    }

    if (tracefiles.empty()) {
        return;
    }
    const std::string path = (fs::path(directory) / kCoverageTracefile).string();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        fmt::print(stderr, "host-test-dispatch: cannot write {}: {}\n", path,
                   std::strerror(errno));
        return;
    }
    out << mergeLcovTracefiles(tracefiles);
    out.close();
    fmt::print("----- coverage: merged {} tracefiles into {} -----\n", tracefiles.size(), path);
    std::fflush(stdout);
}

// The operations runSelections() performs on the world outside its loop: starting a child,
// reaping whichever finishes first, signalling one that has outstayed its deadline, reading the
// clock those deadlines are measured on, and naming the file a part writes its report to.
using SpawnFn = std::function<pid_t(const std::string &path, const std::vector<std::string> &args,
                                    const std::string &logPath)>;
// Returns the pid reaped, 0 if *within* elapsed first, or -1 if there was nothing to wait for.
using ReapFn =
    std::function<pid_t(int &exitStatus, std::optional<std::chrono::milliseconds> within)>;
using KillFn = std::function<void(pid_t pid, int signal)>;
using NowFn = std::function<std::chrono::steady_clock::time_point()>;
using ReportPathFn = std::function<std::string(const std::string &qualifier, size_t part)>;
// Whether the run has been asked to stop since it was last consulted.
using TerminatedFn = std::function<bool()>;

struct SchedulerOperations {
    SpawnFn spawn;
    ReapFn reap;
    KillFn kill;
    NowFn now;
    ReportPathFn reportPath;
    TerminatedFn terminated;   // unset means a run that cannot be asked to stop
};

struct ScheduleOptions {
    size_t jobs = 1;                                 // children allowed to run at once
    ReportFormat outputFormat = ReportFormat::Xml;   // asked of the children, so no report is
                                                     // ever transcoded between the two formats
    std::vector<std::string> passthrough;            // the caller's own flags, handed to each child
    std::string logDirectory;                        // empty means the children inherit our stdio
    std::chrono::seconds childTimeout{0};            // 0 means a child may run for as long as it
                                                     // likes, which is the unconfigured behaviour
};

// Run every part of every selection, at most *schedule.jobs* at a time, and return their
// fragments in selection order. *failures* is incremented once per failing or missing report.
//
// Parts of one selection are never in flight together: each writes the same per-binary log, and
// the next part is only made ready again once the previous one has been reaped.
std::vector<SuiteFragment> runSelections(const std::vector<Selection> &selections,
                                         const std::vector<std::vector<std::string>> &filters,
                                         const ScheduleOptions &schedule,
                                         const SchedulerOperations &ops, int &failures)
{
    std::vector<size_t> nextPart(selections.size(), 0);
    std::vector<std::vector<SuiteFragment>> perSelection(selections.size());
    const bool timed = schedule.childTimeout > std::chrono::seconds{0};

    struct RunningChild {
        pid_t pid;
        size_t selection;
        std::string reportPath;
        std::string logPath;
        // When this child is next due a signal: its deadline while it is still running freely,
        // then one grace period later each time it has been signalled and is still here.
        std::chrono::steady_clock::time_point due;
        bool termed = false;      // SIGTERM has been sent, so the next signal is SIGKILL
        bool timedOut = false;    // whatever it writes, it is a timeout and is reported as one
        bool abandoned = false;   // killed because the run itself was asked to stop
    };
    bool stopping = false;   // the run has been asked to stop and is winding the children down
    std::deque<size_t> ready;   // selections with a part still to start
    for (size_t index = 0; index < selections.size(); ++index) {
        ready.push_back(index);
    }
    std::vector<RunningChild> running;

    while (!ready.empty() || !running.empty()) {
        // Asked to stop -- an outer watchdog's SIGTERM, or a user's Ctrl-C. Nothing further is
        // started, what is running is wound down, and the loop then exits through its ordinary
        // path so the caller still writes a report of everything that did finish.
        if (!stopping && ops.terminated && ops.terminated()) {
            stopping = true;
            const std::chrono::steady_clock::time_point now = ops.now();
            for (const size_t index : ready) {
                perSelection[index].push_back(
                    crashFragment(selections[index].qualifier,
                                  "run was stopped before this binary started"));
                ++failures;
                nextPart[index] = filters[index].size();
            }
            ready.clear();
            for (RunningChild &child : running) {
                child.abandoned = true;
                ops.kill(child.pid, SIGTERM);
                child.termed = true;
                child.due = now + kTerminateGrace;
            }
        }

        while (!ready.empty() && running.size() < schedule.jobs) {
            const size_t index = ready.front();
            ready.pop_front();
            const Selection &selection = selections[index];
            const size_t part = nextPart[index]++;
            const std::string reportPath = ops.reportPath(selection.qualifier, part);
            std::vector<std::string> childArgs = {
                fmt::format("--gtest_filter={}", filters[index][part]),
                fmt::format("--gtest_output={}:{}", formatName(schedule.outputFormat), reportPath),
            };
            childArgs.insert(childArgs.end(), schedule.passthrough.begin(),
                             schedule.passthrough.end());

            const std::string part_note =
                filters[index].size() > 1
                    ? fmt::format(", part {}/{}", part + 1, filters[index].size())
                    : std::string();
            const std::string logPath =
                schedule.logDirectory.empty()
                    ? std::string()
                    : childLogPath(schedule.logDirectory, selection.qualifier);
            if (part == 0 && !logPath.empty()) {
                const int fd = open(logPath.c_str(),
                                    O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0644);
                if (fd >= 0) {
                    close(fd);
                }
            }
            fmt::print("----- {} ({} cases{}){} -----\n", selection.qualifier,
                       selection.cases.size(), part_note,
                       logPath.empty() ? std::string() : fmt::format(" -> {}", logPath));
            std::fflush(stdout);

            const pid_t pid = ops.spawn(selection.path, childArgs, logPath);
            if (pid < 0) {
                perSelection[index].push_back(crashFragment(
                    selection.qualifier, "binary could not be started -- fork failed"));
                ++failures;
                nextPart[index] = filters[index].size();
                continue;
            }
            running.push_back({pid, index, reportPath, logPath,
                               ops.now() + schedule.childTimeout});
        }

        // Bounded only when there are deadlines to honour, and then only until the nearest of
        // them: an unbounded reap blocks in waitpid() and costs nothing, which is what a run
        // without a child timeout gets.
        std::optional<std::chrono::milliseconds> within;
        if ((timed || stopping) && !running.empty()) {
            const auto nearest = std::min_element(running.begin(), running.end(),
                                                  [](const RunningChild &a, const RunningChild &b) {
                                                      return a.due < b.due;
                                                  });
            within = std::max(std::chrono::milliseconds{0},
                              std::chrono::duration_cast<std::chrono::milliseconds>(
                                  nearest->due - ops.now()));
        }

        int status = 0;
        const pid_t finished = ops.reap(status, within);
        if (finished == 0) {
            if (!timed && !stopping) {
                continue;   // an unbounded reap, interrupted by a signal: no deadline to act on
            }
            // Nothing finished before the nearest deadline. Every child now past its own is
            // signalled and given another grace period, so one that ignores SIGKILL costs a
            // signal every kTerminateGrace rather than a spin.
            const std::chrono::steady_clock::time_point now = ops.now();
            for (RunningChild &child : running) {
                if (now < child.due) {
                    continue;
                }
                child.timedOut = !child.abandoned;
                ops.kill(child.pid, child.termed ? SIGKILL : SIGTERM);
                child.termed = true;
                child.due = now + kTerminateGrace;
            }
            continue;
        }
        const auto slot = std::find_if(running.begin(), running.end(),
                                       [finished](const RunningChild &child) {
                                           return child.pid == finished;
                                       });
        if (slot == running.end()) {
            for (const RunningChild &child : running) {
                perSelection[child.selection].push_back(
                    crashFragment(selections[child.selection].qualifier,
                                  "binary disappeared before its result could be collected"));
                ++failures;
            }
            running.clear();
            break;
        }

        const size_t index = slot->selection;
        const Selection &selection = selections[index];
        if (slot->abandoned) {
            // Killed partway through, so its report is missing whatever it had not reached; the
            // cases it did run are lost with it, but every binary already finished is kept.
            perSelection[index].push_back(crashFragment(
                selection.qualifier,
                fmt::format("run was stopped and this binary was killed before it finished{}",
                            slot->logPath.empty() ? std::string()
                                                  : fmt::format("; output in {}", slot->logPath))));
            ++failures;
            nextPart[index] = filters[index].size();
            std::error_code ec;
            fs::remove(slot->reportPath, ec);
            running.erase(slot);
            continue;
        }
        if (slot->timedOut) {
            // Whatever it exited with, and whatever report it left behind, it was killed partway
            // through: the report would be missing exactly the cases that hung. Its remaining
            // parts are skipped too, since they are more of the same binary and would each cost
            // another timeout to learn the same thing.
            const size_t remaining = filters[index].size() - nextPart[index];
            perSelection[index].push_back(crashFragment(
                selection.qualifier,
                fmt::format("binary timed out after {} s and was killed{}{}",
                            schedule.childTimeout.count(),
                            slot->logPath.empty() ? std::string()
                                                  : fmt::format("; output in {}", slot->logPath),
                            remaining == 0
                                ? std::string()
                                : fmt::format("; its {} remaining part(s) were skipped",
                                              remaining))));
            ++failures;
            nextPart[index] = filters[index].size();
            std::error_code ec;
            fs::remove(slot->reportPath, ec);
            running.erase(slot);
            continue;
        }
        bool cleanReport = false;
        if (std::optional<SuiteFragment> fragment =
                parseChildReport(readFile(slot->reportPath), selection.qualifier,
                                 schedule.outputFormat)) {
            cleanReport = fragment->failures == 0 && fragment->errors == 0;
            perSelection[index].push_back(std::move(*fragment));
        } else {
            perSelection[index].push_back(crashFragment(
                selection.qualifier,
                fmt::format("binary wrote no gtest report (exit status {}){}", status,
                            slot->logPath.empty() ? std::string()
                                                  : fmt::format("; output in {}", slot->logPath))));
            ++failures;
        }
        if (status != 0) {
            if (cleanReport) {
                perSelection[index].push_back(crashFragment(
                    selection.qualifier,
                    fmt::format("binary exited {} after writing a passing report", status)));
            }
            ++failures;
        }
        std::error_code ec;
        fs::remove(slot->reportPath, ec);
        running.erase(slot);
        if (nextPart[index] < filters[index].size()) {
            ready.push_back(index);   // this binary's next part, now that its log is free again
        }
    }

    // Selection order, not completion order: the merged report must not depend on how the
    // children happened to be scheduled.
    std::vector<SuiteFragment> fragments;
    for (std::vector<SuiteFragment> &selectionFragments : perSelection) {
        for (SuiteFragment &fragment : selectionFragments) {
            fragments.push_back(std::move(fragment));
        }
    }
    return fragments;
}

}  // namespace rccl_host_dispatch

#ifndef RCCL_HOST_DISPATCH_NO_MAIN

namespace {

using namespace rccl_host_dispatch;

// A temporary directory for the children's reports, removed when the guard goes out of scope.
//
// mkdtemp and not std::filesystem: the standard has no atomic create-unique-directory, and an
// invented name races the other dispatchers running beside us.
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
    const char *filterEnv = std::getenv("GTEST_FILTER");
    const char *disabledEnv = std::getenv("GTEST_ALSO_RUN_DISABLED_TESTS");
    const Options options =
        parseArgs(args, outputEnv == nullptr ? "" : outputEnv,
                  filterEnv == nullptr ? "" : filterEnv,
                  disabledEnv == nullptr ? std::nullopt
                                         : std::optional<std::string_view>(disabledEnv));
    warnAboutUnknownGtestVariables(gtestVariablesToStrip());

    const std::string directory = binaryDirectory(selfDirectory(argv[0]));

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
    // Cleared before the init, which is where gtest builds the printer: the flag is seeded from
    // GTEST_OUTPUT at static init, and parseArgs() has already taken that as the destination for
    // the merged report.
    ::testing::GTEST_FLAG(output) = "";
    int gtest_argc = 1;
    char *gtest_argv[] = {argv[0], nullptr};
    ::testing::InitGoogleTest(&gtest_argc, gtest_argv);

    const std::vector<std::string> binaries = discoverBinaries(directory, selfPath());
    if (binaries.empty()) {
        fmt::print(stderr,
                   "host-test-dispatch: no {}* binaries in {} -- nothing to run\n",
                   kBinaryPrefix, directory);
        return 1;
    }

    const Enumeration enumeration = enumerate(directory, binaries, listTestsOutput);
    for (const std::string &binary : enumeration.unlistable) {
        fmt::print(stderr, "host-test-dispatch: {} could not list its tests\n", binary);
    }
    const std::vector<CandidateCase> &candidates = enumeration.candidates;
    const FilterMatches chosen =
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

    // Suppressed counts as selected here: a filter that matched only DISABLED_ cases runs
    // nothing, but it is not the mistake this warning is about and it still has a report to
    // write.
    if (chosen.runnable.empty() && chosen.suppressed.empty() && !candidates.empty()) {
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

    // The children's console output, one file per binary. Created up front so a directory that
    // cannot be used is said once, here, rather than once per child; failing to make it turns
    // the redirection off, because a missing transcript must not cost a passing run.
    std::string logDir =
        resolveLogDirectory(options.outputPath, std::getenv(std::string(kLogDirectoryEnv).c_str()));
    if (!logDir.empty()) {
        if (const std::optional<std::string> problem = prepareLogDirectory(logDir)) {
            fmt::print(stderr, "host-test-dispatch: cannot use {}: {} -- the binaries will"
                               " print to the console instead\n",
                       logDir, *problem);
            logDir.clear();
        }
    }

    installStopHandler();

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

    const size_t jobs = resolveParallelism(std::getenv(std::string(kParallelismEnv).c_str()),
                                           defaultParallelism(detectedParallelism()));

    std::vector<std::vector<std::string>> filters;
    filters.reserve(selections.size());
    for (const Selection &selection : selections) {
        filters.push_back(childFilters(selection));
        ran.push_back(selection.path);
    }
    const std::chrono::seconds childTimeout =
        resolveChildTimeout(std::getenv(std::string(kChildTimeoutEnv).c_str()));
    const ScheduleOptions schedule{jobs, options.outputFormat, options.passthrough, logDir,
                                   childTimeout};
    const SchedulerOperations ops{
        [](const std::string &path, const std::vector<std::string> &args,
           const std::string &logPath) { return spawnChild(path, args, logPath); },
        reapAnyChild,
        signalChildGroup,
        [] { return std::chrono::steady_clock::now(); },
        [&reportDir, &options](const std::string &qualifier, size_t part) {
            return reportDir.reportPath(qualifier, part, options.outputFormat);
        },
        [] { return g_stopRequested != 0; },
    };
    std::vector<SuiteFragment> ranFragments =
        runSelections(selections, filters, schedule, ops, failures);
    for (SuiteFragment &fragment : ranFragments) {
        fragments.push_back(std::move(fragment));
    }

    if (!options.outputPath.empty()) {
        // Merging the children's reports walks every case in the tree, which is tens of
        // thousands of them, so this too is a pause the caller would otherwise see unexplained.
        fmt::print("----- report: merging {} suites into {} -----\n", fragments.size(),
                   options.outputPath);
        std::fflush(stdout);
        std::ofstream report(options.outputPath, std::ios::binary | std::ios::trunc);
        if (!report) {
            fmt::print(stderr, "host-test-dispatch: cannot write {}: {}\n", options.outputPath,
                       std::strerror(errno));
            return 1;
        }
        report << mergeReports(fragments, options.outputFormat);
        report.close();
        fmt::print("----- report: wrote {} -----\n", options.outputPath);
        std::fflush(stdout);
    }

    // After the report: every test has already run by this point, and the per-binary llvm
    // invocations below are long enough that a category timeout can land in the middle of them.
    writeMergedCoverage(coverageDirectory(), ran, runShell);

    return failures == 0 ? 0 : 1;
}

#endif  // RCCL_HOST_DISPATCH_NO_MAIN
