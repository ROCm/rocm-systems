/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// rccl-UnitTestsHostAll -- one gtest-shaped front door for every host-only test binary.
//
// Reads the list of binaries the build put beside it, forks the ones owning a selected case,
// and merges their reports into the one the caller asked for. What it is for, how the case
// names are qualified, and how the coverage manifest is used: see MICROTEST_README.md,
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
#include <poll.h>
#include <sched.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace rccl_host_dispatch {

namespace fs = std::filesystem;

// The binaries this dispatcher fronts, one name per line, written into their own directory by
// the build that produced them (RCCL_MICRO_MANIFEST in test/host/CMakeLists.txt). The build is
// the only thing that knows what it made, so it says so rather than leaving this process to
// infer it from what happens to be lying in the directory. See manifestBinaries().
constexpr std::string_view kBinaryManifest = "host_test_binaries.txt";

// The subdirectory holding every binary this dispatcher fronts, in the build tree and in an
// install alike. See binaryDirectory(), and RCCL_MICRO_SUBDIR in test/host/CMakeLists.txt.
constexpr std::string_view kMicroSubdirectory = "micro";

// Dropped from a binary's name to form the case-name qualifier, leaving e.g. "Micro",
// "MicroInit-uncached". A name the strip does not apply to keeps its own name as the
// qualifier, so the manifest is free to carry a binary named to any convention or none.
constexpr std::string_view kQualifierStrip = "rccl-UnitTests";

// The merged lcov tracefile written beside the .profraw files: the whole host tree's coverage
// in one object-free, hash-agnostic file, so no consumer has to pair profiles with binaries.
constexpr std::string_view kCoverageTracefile = "host-coverage.lcov";

// The directory the children's console output is written into, one <Qualifier>.log per fronted
// binary. Relative, so it lands under the working directory the caller chose.
//
// This used to be derived from the merged report's path ("<report-stem>.logs" beside it), which
// made the logs' location a function of a caller's private choice of report path. Under the
// python test runner that path is an mkstemp in /tmp, so the logs went to a directory whose name
// was published the moment the report file appeared, and which another local user could
// pre-create. The derivation is gone: a caller who wants the logs somewhere in particular says
// so below, and that is the only thing that moves them.
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

// How long a wait sleeps before looking again when it has no wake pipe to be woken through.
// Only a run whose pipe could not be created polls at all; the interval bounds how late a
// deadline or a stop is noticed, which against a timeout measured in seconds need not be tight.
constexpr std::chrono::milliseconds kReapPollInterval{5};

// The longest a wait will sleep without looking again, even when it does have a pipe.
//
// Nothing is supposed to depend on this: every event a wait must not sleep through writes to
// the pipe. It is here because the cost of being wrong about that is unbounded -- a wake that
// is never written leaves an unbounded wait asleep for the whole run -- and the cost of
// insuring against it is one syscall a second. It also bounds the damage when a wait is
// entered with the handlers not installed, which the pipe cannot distinguish from a quiet run.
constexpr std::chrono::milliseconds kWakeRecheckInterval{1000};

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

// *text* with the five characters XML gives meaning to replaced by their entities.
//
// Only ever applied to text this dispatcher composes itself -- a qualifier, and a message
// naming a path. A child's own report is spliced in verbatim, because gtest escaped it when it
// wrote it and escaping it twice would show the entities to the reader.
std::string escapeXml(std::string_view text)
{
    std::string escaped;
    escaped.reserve(text.size());
    for (const char c : text) {
        switch (c) {
            case '&': escaped += "&amp;"; break;
            case '<': escaped += "&lt;"; break;
            case '>': escaped += "&gt;"; break;
            case '"': escaped += "&quot;"; break;
            case '\'': escaped += "&apos;"; break;
            default: escaped += c; break;
        }
    }
    return escaped;
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
        // Escaped, both of them: the detail carries the child's log path, which is a directory
        // the caller named, and one '&' or '<' in it would make the merged report unparsable --
        // on exactly the run that already failed and is being read to find out why.
        fmt::arg("qualifier", escapeXml(qualifier)), fmt::arg("detail", escapeXml(detail)));

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

// The binaries the build that produced *directory* says it put there, in the order it names
// them; empty when there is no manifest.
std::vector<std::string> manifestBinaries(const std::string &directory)
{
    std::ifstream manifest((fs::path(directory) / kBinaryManifest).string());
    if (!manifest) {
        return {};
    }
    // Resolved once, and compared against every entry: a match is this executable under another
    // name, and running it would fork the run into itself. The error main() prints invites a
    // hand-written manifest, so the name it may be given is not the build's to control.
    std::error_code selfEc;
    const fs::path self = fs::weakly_canonical(selfPath(), selfEc);

    std::vector<std::string> names;
    std::string line;
    while (std::getline(manifest, line)) {
        // CRLF- and whitespace-tolerant: the file is read wherever the tree was copied to.
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        if (fs::path(line).filename() != fs::path(line)) {
            fmt::print(stderr, "host-test-dispatch: ignoring \"{}\" in {}: a binary name, not a"
                               " path, is expected on each line\n",
                       line, kBinaryManifest);
            continue;
        }
        std::error_code ec;
        const fs::path resolved = fs::weakly_canonical(fs::path(directory) / line, ec);
        if (!selfEc && !ec && resolved == self) {
            fmt::print(stderr, "host-test-dispatch: ignoring \"{}\" in {}: it resolves to this"
                               " executable, which would run the front door inside itself\n",
                       line, kBinaryManifest);
            continue;
        }
        if (std::find(names.begin(), names.end(), line) != names.end()) {
            fmt::print(stderr, "host-test-dispatch: ignoring \"{}\" in {}: already named, and a"
                               " second child of that name would overwrite the first's report\n",
                       line, kBinaryManifest);
            continue;
        }
        names.push_back(line);
    }
    return names;
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

// ── enumeration ───────────────────────────────────────────────────

// What the discovered binaries hold, and which of them would not say.
struct Enumeration {
    std::vector<CandidateCase> candidates;
    std::vector<std::string> unlistable;   // binaries whose --gtest_list_tests failed
    std::vector<std::string> unlisted;     // binaries a stop reached before they were asked
};

// How a binary is asked for its cases. A seam so the enumeration can be tested without one.
using ListingFn = std::function<std::optional<std::string>(const std::string &)>;

// Whether the run has been asked to stop since it was last consulted. Shared by the enumeration
// and the scheduler, which is the point: there is one answer to that question, and every phase
// of the run asks it.
using TerminatedFn = std::function<bool()>;

// Every case every discovered binary holds, published name and all, in discovery order.
//
// A binary that will not list its cases is RECORDED, never skipped. Dropping it would be the
// one failure this dispatcher must not have: a binary that cannot start -- or that the build
// says it made but that is not there -- would otherwise disappear from the run entirely,
// unrun, unreported, absent from the coverage manifest, and the run would still exit 0. The
// hand-kept list this replaced caught exactly this, by checking each binary wrote a report.
Enumeration enumerate(const std::string &directory, const std::vector<std::string> &binaries,
                      const ListingFn &listing, const TerminatedFn &terminated = {})
{
    Enumeration enumeration;
    for (size_t i = 0; i < binaries.size(); ++i) {
        // A stop during enumeration ends it, and the binaries it did not reach are recorded as
        // such. Carrying on would cost one timeout per remaining binary; dropping them silently
        // would leave a report that looks like a tree that never had them.
        if (terminated && terminated()) {
            enumeration.unlisted.assign(binaries.begin() + static_cast<long>(i), binaries.end());
            break;
        }
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

// Where *qualifier*'s console output goes within *directory*.
std::string childLogPath(std::string_view directory, std::string_view qualifier)
{
    return (fs::path(std::string(directory)) / fmt::format("{}.log", qualifier)).string();
}

// The log directory this run should use, given *override_* from the environment. Empty when the
// caller asked for the children's output on the console instead.
std::string resolveLogDirectory(const char *override_)
{
    if (override_ == nullptr || *override_ == '\0') {
        return std::string(kDefaultLogDirectory);
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
//
// The affinity mask, not the machine's processor count: a run under taskset, cpuset or a
// container CPU limit may use only a few of the node's processors, and sizing the run to the
// node would oversubscribe those few by the ratio between them. It is also what nproc -- the
// count run_host_tests.sh reports -- uses, so the two agree.
size_t detectedParallelism()
{
    cpu_set_t affinity;
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
        const int usable = CPU_COUNT(&affinity);
        if (usable > 0) {
            return static_cast<size_t>(usable);
        }
    }
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
                 const std::string &logPath = "", const std::string &errPath = "")
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
    // Read before the fork, so the child compares against the parent it was actually forked
    // from rather than against whoever has adopted it by the time it looks.
    const pid_t parent = getpid();

    const pid_t pid = fork();
    if (pid < 0) {
        fmt::print(stderr, "host-test-dispatch: fork failed: {}\n", std::strerror(errno));
        return -1;
    }
    if (pid == 0) {
        // Die with the dispatcher. This covers the one case it cannot handle itself: a SIGKILL,
        // from the runner's escalation or the OOM killer, after which there is no code of ours
        // left to signal anybody. The getppid() check closes the race in which the parent died
        // before the prctl took effect -- the signal would then already have been sent, to a
        // process that had not asked for it yet.
        //
        // It reaches this child and no further: the attribute is per-process and is cleared in
        // the children of a fork, so a binary's own forks are not covered by it. They are
        // covered by the group kill below whenever the dispatcher is alive to send it, which
        // leaves one residual -- a grandchild of a binary that was still running when the
        // dispatcher was killed outright is reparented and survives. Nothing a process can set
        // for itself fixes that; it needs a pid namespace or a cgroup, which is the harness's
        // to give, not ours.
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        if (getppid() != parent) {
            _exit(128 + SIGKILL);
        }
        // Its own process group, so a timeout can signal the whole tree rather than the one pid:
        // a test binary forks too -- death tests, ProcessIsolatedTestRunner -- and those
        // grandchildren would otherwise survive the kill and keep the CI node. The cost is that
        // the children leave the terminal's foreground group, so an interactive Ctrl-C reaches
        // the dispatcher alone; it signals them on the way out, and PR_SET_PDEATHSIG above
        // covers the case where it never gets the chance.
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
                if (errPath.empty()) {
                    dup2(fd, STDERR_FILENO);
                }
                if (fd > STDERR_FILENO) {
                    close(fd);
                }
            }
        }
        // Only the listing sets this, to keep a binary's stderr out of output that is about to
        // be parsed: a line of noise in the middle of a listing is indistinguishable from a
        // suite name.
        if (!errPath.empty()) {
            const int fd = open(errPath.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW, 0644);
            if (fd >= 0) {
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

// The one thing in this process a wait can be woken by, other than the thing it is waiting for.
//
// Every event the scheduler must not sleep through -- a child exiting, a stop being asked for --
// arrives as a signal, and a signal cannot be waited for alongside a deadline. A byte down a
// pipe can be: the handlers write one, and the wait is a poll() on the read end. That removes
// both of the ways a stop used to be missed. There is no window between testing a flag and
// blocking, because a signal landing in that window leaves the byte behind and poll() returns
// at once; and there is no sleep-poll, because the wait is woken rather than retried.
//
// Writes are non-blocking and their result ignored: a full pipe already carries the only
// information a reader takes from it, which is that something happened.
class WakePipe {
public:
    // Idempotent, and safe to leave unopened: a wait with no pipe falls back to a short sleep,
    // which is correct, merely less prompt. Nothing here is worth failing a test run over.
    void open()
    {
        if (fds_[0] >= 0) {
            return;
        }
        if (pipe2(fds_, O_CLOEXEC | O_NONBLOCK) != 0) {
            fds_[0] = fds_[1] = -1;
        }
    }

    int readFd() const { return fds_[0]; }

    // Async-signal-safe: one write(2) on an already-open descriptor, and nothing else.
    void wake() const
    {
        if (fds_[1] < 0) {
            return;
        }
        const char byte = 1;
        const ssize_t written = ::write(fds_[1], &byte, 1);
        (void)written;
    }

    // Take back whatever the handlers left, so one wake does not satisfy every later wait.
    void drain() const
    {
        if (fds_[0] < 0) {
            return;
        }
        char buffer[64];
        while (::read(fds_[0], buffer, sizeof(buffer)) > 0) {
        }
    }

private:
    int fds_[2] = {-1, -1};
};

WakePipe g_wake;

// Set from the SIGTERM/SIGINT handler; read by the scheduler between reaps.
volatile std::sig_atomic_t g_stopRequested = 0;

// Whether a wait has already come back to say the run was asked to stop. The flag above never
// clears, so without this the same news would be delivered again by every later wait. See
// reapAnyChild.
bool g_stopReported = false;

extern "C" void requestStop(int)
{
    g_stopRequested = 1;
    g_wake.wake();
}

// SIGCHLD exists only to wake the wait. The child is reaped by the scheduler, in its own time,
// where the exit status has somewhere to go.
extern "C" void noteChildExit(int)
{
    g_wake.wake();
}

// Ask for SIGTERM and SIGINT to stop the run rather than end it outright, so the children are
// killed with it and a report of what did finish is still written, and ask for SIGCHLD so a
// bounded wait learns of a child that finished early rather than sitting out the whole bound.
//
// With SA_RESTART, unlike the first version of this: the pipe is what wakes the wait now, so no
// EINTR is load-bearing anywhere, and the syscalls this process makes around the scheduler --
// not least the listing's popen/pclose -- are no longer exposed to spurious interruption.
void installStopHandler()
{
    g_wake.open();

    struct sigaction action {};
    action.sa_handler = requestStop;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGINT, &action, nullptr);

    struct sigaction child {};
    child.sa_handler = noteChildExit;
    sigemptyset(&child.sa_mask);
    // SA_NOCLDSTOP: a child merely stopped is not news, and this process would otherwise be
    // woken by every SIGSTOP a test binary's own machinery uses.
    child.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    sigaction(SIGCHLD, &child, nullptr);
}

// Send *signal* to everything *pid* has spawned, and to *pid* itself.
//
// Falls back to the single pid when there is no such group, which is what a child that died
// between the fork and its setpgid() leaves behind.
void signalChildGroup(pid_t pid, int signal)
{
    if (::kill(-pid, signal) == 0 || errno != ESRCH) {
        return;
    }
    ::kill(pid, signal);
}

// Sleep until something happens or *until* passes, whichever is first. "Something" is whatever
// a handler wrote to the wake pipe: a child exited, or the run was asked to stop.
void waitForWake(std::optional<std::chrono::steady_clock::time_point> until)
{
    const auto remaining = [&until]() -> int {
        if (!until) {
            return static_cast<int>(kWakeRecheckInterval.count());
        }
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            *until - std::chrono::steady_clock::now());
        return static_cast<int>(
            std::min(kWakeRecheckInterval.count(),
                     std::max<std::chrono::milliseconds::rep>(0, left.count())));
    };

    if (g_wake.readFd() < 0) {
        // No pipe to be woken through, so look again shortly. Correct, just less prompt.
        std::this_thread::sleep_for(
            std::min<std::chrono::milliseconds>(kReapPollInterval,
                                                std::chrono::milliseconds{remaining()}));
        return;
    }

    struct pollfd woken {};
    woken.fd = g_wake.readFd();
    woken.events = POLLIN;
    if (::poll(&woken, 1, remaining()) > 0) {
        g_wake.drain();
    }
}

// How a child ended: the status it is reported as having exited with, and whether it was killed
// rather than having exited of its own accord.
//
// The two are kept apart because 128+signal is an encoding, not a status a child can be trusted
// to have chosen: a child killed while it was already on its way out still has whatever it had
// written, and only *signalled* says which of the two happened.
struct ChildExit {
    int status = 0;          // 128 + signal when it was killed
    bool signalled = false;  // it did not exit of its own accord
};

// Wait for whichever spawned child finishes first, describing it in *exit* and returning its
// pid; -1 when there is nothing left to wait for.
//
// 0 comes back for the two reasons a wait ends without a child: *within* elapsed, or the run was
// asked to stop. Both are the caller's to interpret, and it can tell them apart by asking for
// the stop itself -- the scheduler reads terminated() at the top of its loop either way.
//
// *within*, when set, bounds the wait; unset means wait until something happens. Neither case
// blocks in waitpid(): the wait is a poll() on the wake pipe, so a child exiting, a stop, and
// the deadline all end it, and the check-then-block race that a bare waitpid() has is gone --
// a signal arriving after the status check leaves a byte in the pipe and the poll returns at
// once.
pid_t reapAnyChild(ChildExit &exit, std::optional<std::chrono::milliseconds> within)
{
    const std::optional<std::chrono::steady_clock::time_point> deadline =
        within ? std::optional<std::chrono::steady_clock::time_point>(
                     std::chrono::steady_clock::now() + *within)
               : std::nullopt;
    for (;;) {
        int status = 0;
        const pid_t pid = waitpid(-1, &status, WNOHANG);
        if (pid > 0) {
            exit.signalled = !WIFEXITED(status);
            exit.status = WIFEXITED(status) ? WEXITSTATUS(status)
                                            : (WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 1);
            return pid;
        }
        if (pid < 0 && errno != EINTR) {
            // ECHILD: nothing left to wait for.
            exit = ChildExit{1, false};
            return -1;
        }
        // Tested after the reap, not before it: a child that exited is reported as having
        // exited even on the pass that notices the stop, so its result is never thrown away
        // for the sake of the news that the run is ending.
        //
        // Reported on the EDGE, once. A level test would make every wait after a stop return at
        // once, and the caller's loop -- which by then has children to wind down and nothing to
        // do until they go -- spins on it for the whole of the teardown grace: measured at
        // 3.64 s of CPU in 4 s of wall, during exactly the window the runner allows before its
        // own SIGKILL. Once the news has been delivered, a wait goes back to being a wait, and
        // the caller reads the stop from terminated() as it did before.
        if (g_stopRequested != 0 && !g_stopReported) {
            g_stopReported = true;
            exit = ChildExit{};
            return 0;
        }
        if (deadline && std::chrono::steady_clock::now() >= *deadline) {
            exit = ChildExit{};
            return 0;
        }
        waitForWake(deadline);
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

// ── the listing each binary is asked for ───────────────────────────

// A file this process creates, owns and removes. Used for a listing's output, which has to go
// somewhere the child can write freely: a pipe would have to be drained while the parent waits,
// and the biggest binary's listing is larger than a pipe will hold, so the two would deadlock
// at exactly the size where it matters.
class ScopedTempFile {
public:
    explicit ScopedTempFile(const std::string &pattern)
    {
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        const int fd = mkstemp(name.data());
        if (fd >= 0) {
            close(fd);
            path_ = name.data();
        }
    }

    ~ScopedTempFile()
    {
        if (!path_.empty()) {
            std::error_code ec;
            fs::remove(path_, ec);
        }
    }

    ScopedTempFile(const ScopedTempFile &) = delete;
    ScopedTempFile &operator=(const ScopedTempFile &) = delete;

    const std::string &path() const { return path_; }
    bool valid() const { return !path_.empty(); }

private:
    std::string path_;
};

// *path*'s --gtest_list_tests output, or nullopt when the binary failed to produce one.
//
// The two are deliberately distinguishable: a healthy binary that holds no cases lists nothing
// and is not a problem, whereas one that cannot start (missing .so, an abort in static init)
// fails here FIRST, before anything else the dispatcher does to it, and must not be mistaken
// for the former. See enumerate().
//
// A listing is a child like any other, and gets what the test children get: its own process
// group, the stripped environment, its own profile, *timeout* to produce an answer in, and an
// end if the run is stopped. It used to be a popen(), which gives no pid -- so a binary that
// hung in a static constructor, or at exit after printing its list, hung the whole run with
// nothing able to interrupt it, before a single test had been chosen.
std::optional<std::string> listTestsOutput(const std::string &path, std::chrono::seconds timeout)
{
    const ScopedTempFile listing(
        (fs::temp_directory_path() / "rccl-host-dispatch-listing-XXXXXX").string());
    if (!listing.valid()) {
        return std::nullopt;
    }

    const pid_t pid = spawnChild(path, {"--gtest_list_tests"}, listing.path(), "/dev/null");
    if (pid < 0) {
        return std::nullopt;
    }

    const std::optional<std::chrono::steady_clock::time_point> due =
        timeout.count() > 0
            ? std::optional<std::chrono::steady_clock::time_point>(
                  std::chrono::steady_clock::now() + timeout)
            : std::nullopt;

    bool killed = false;
    for (;;) {
        std::optional<std::chrono::milliseconds> within;
        if (due) {
            within = std::max(std::chrono::milliseconds{0},
                              std::chrono::duration_cast<std::chrono::milliseconds>(
                                  *due - std::chrono::steady_clock::now()));
        }
        ChildExit exit;
        const pid_t finished = reapAnyChild(exit, within);
        if (finished == pid) {
            // The exit status has to be read before the output is trusted: a non-zero one means
            // the listing is partial, whatever arrived on the way there.
            return (exit.status == 0 && !killed)
                       ? std::optional<std::string>(readFile(listing.path()))
                       : std::nullopt;
        }
        if (finished < 0) {
            return std::nullopt;   // nothing left to wait for; the child is gone unaccounted for
        }
        if (finished > 0) {
            continue;   // some other child of ours; not possible today, and harmless if it were
        }
        // Either the run is ending or this binary has had its time. Both end the same way, and
        // the same way a test child's deadline does: SIGTERM first, SIGKILL if it is ignored.
        signalChildGroup(pid, SIGTERM);
        killed = true;
        break;
    }

    // Collected through the same wait as everything else, and on the same terms a test child
    // gets: the grace period to act on the SIGTERM, then SIGKILL. This used to be a bespoke
    // 200 x 10 ms waitpid loop, because a stop made a general reap return at once and the child
    // would have been left behind -- which reapAnyChild no longer does.
    std::chrono::steady_clock::time_point grace =
        std::chrono::steady_clock::now() + kTerminateGrace;
    bool killedOutright = false;
    for (;;) {
        ChildExit exit;
        const pid_t collected = reapAnyChild(
            exit, std::max(std::chrono::milliseconds{0},
                           std::chrono::duration_cast<std::chrono::milliseconds>(
                               grace - std::chrono::steady_clock::now())));
        if (collected == pid || collected < 0) {
            break;   // collected, or there is nothing left to wait for
        }
        if (collected > 0) {
            continue;   // some other child of ours; not possible today, and harmless if it were
        }
        if (killedOutright) {
            break;   // it has had a SIGKILL and a grace period; it is the kernel's problem now
        }
        signalChildGroup(pid, SIGKILL);
        killedOutright = true;
        grace = std::chrono::steady_clock::now() + kTerminateGrace;
    }
    return std::nullopt;
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

// How a shell command is run, returning its exit status. A seam, so the export can be driven in
// a test without llvm-profdata and llvm-cov on the machine.
using CommandFn = std::function<int(const std::string &)>;

// Wait for the shell child *pid* and return the exit status to report for it, ending it early
// if the run is asked to stop while it is still going.
//
// The stop is acted on here rather than left to the terminal. The child is in its own process
// group -- see runShell -- so an interactive Ctrl-C does not reach it, and a watchdog's SIGTERM
// is addressed to this pid alone and never did. Either way the handler sets g_stopRequested and
// wakes this wait, which then ends the child the same way a test child's deadline does: SIGTERM
// to the group first, SIGKILL after the grace, so an llvm-cov with minutes of work left does
// not outlive the run that asked for it.
int waitForShellChild(pid_t pid)
{
    bool termed = false;
    std::optional<std::chrono::steady_clock::time_point> killAt;
    for (;;) {
        int status = 0;
        const pid_t done = waitpid(pid, &status, WNOHANG);
        if (done == pid) {
            return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
        }
        if (done < 0 && errno != EINTR) {
            return 1;
        }
        const auto now = std::chrono::steady_clock::now();
        if (g_stopRequested != 0 && !termed) {
            signalChildGroup(pid, SIGTERM);
            termed = true;
            killAt = now + kTerminateGrace;
        } else if (termed && killAt && now >= *killAt) {
            signalChildGroup(pid, SIGKILL);
            killAt.reset();
        }
        // Woken by the SIGCHLD this child's exit raises, or by a stop; bounded either way, so a
        // missed wake costs latency rather than the wait.
        waitForWake(killAt);
    }
}

// Run *command* through /bin/sh and return its exit status.
//
// A hand-rolled fork/exec rather than std::system() because of what system() does to signals:
// POSIX has it set SIGINT and SIGQUIT to SIG_IGN *in the calling process* for as long as the
// shell runs. An ignored signal is discarded rather than left pending, so an interactive Ctrl-C
// arriving while an llvm-cov is in flight -- which is where nearly all of a coverage export's
// wall time goes -- was never seen by this process at all: the stop handler did not run, and the
// export carried on into the next binary. Only a Ctrl-C landing in the microseconds between two
// calls was caught. Forking ourselves leaves the handlers installed throughout.
int runShell(const std::string &command)
{
    // Nothing buffered here should be written twice, once by each side of the fork.
    std::fflush(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        fmt::print(stderr, "host-test-dispatch: fork failed: {}\n", std::strerror(errno));
        return 1;
    }
    if (pid == 0) {
        // The other half of system()'s behaviour, and the half that was wanted: the command
        // itself dies on a signal rather than inheriting this process's handling of one.
        struct sigaction dfl {};
        dfl.sa_handler = SIG_DFL;
        sigemptyset(&dfl.sa_mask);
        sigaction(SIGINT, &dfl, nullptr);
        sigaction(SIGTERM, &dfl, nullptr);
        sigaction(SIGCHLD, &dfl, nullptr);
        // Die with the dispatcher even if it is killed outright, and have a group of our own so
        // the wait above can signal the whole tree -- the same reasoning as spawnChild.
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        setpgid(0, 0);
        execl("/bin/sh", "sh", "-c", command.c_str(), static_cast<char *>(nullptr));
        _exit(127);
    }
    setpgid(pid, pid);
    return waitForShellChild(pid);
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

// The lcov flags needed to read llvm-cov's tracefiles, given the text of `lcov --version`.
//
// lcov 2.x enforces consistency checks that reject what llvm-cov emits ("line is hit but no
// branches evaluated"); 1.x has neither those checks nor the error categories that silence them,
// and rejects the unknown names outright, so the flags differ by major version.
//
// This sniff is also in run_host_tests.sh (coverage phase) and in the python test runner's
// _merge_host_coverage(); three languages, three copies, and no way to share one. They are kept
// character-identical so a change to one is greppable from the others.
std::string lcovMergeFlags(std::string_view versionText)
{
    const size_t digit = versionText.find_first_of("0123456789");
    const long major = digit == std::string_view::npos ? 0 : toLong(versionText.substr(digit), 0);
    if (major >= 2) {
        return "--rc branch_coverage=1 --rc check_data_consistency=0"
               " --ignore-errors inconsistent,unsupported,corrupt,format";
    }
    return "--rc lcov_branch_coverage=1";
}

// The command that unions *tracefiles* into the single tracefile *output*.
//
// lcov's own union, rather than one hand-rolled here: a line or branch is covered if ANY input
// covered it, which is the merge the host tree needs -- several binaries compile the same source
// under different #defines, so their coverage-mapping hashes differ and the profiles cannot be
// pooled, but the lines they each reached can be. The tree already depends on lcov for exactly
// this union in run_host_tests.sh and in the python runner; this is the third caller, not a
// third implementation.
std::string lcovMergeCommand(const std::string &tool, const std::string &flags,
                             const std::vector<std::string> &tracefiles,
                             const std::string &output)
{
    std::string command = fmt::format("{} {}", shellQuote(tool), flags);
    for (const std::string &tracefile : tracefiles) {
        command += fmt::format(" --add-tracefile {}", shellQuote(tracefile));
    }
    return command + fmt::format(" -o {}", shellQuote(output));
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

// Whether the lcov export should run at all, given *request* from the environment.
//
// Only "0" turns it off, so a caller who has not heard of this gets the export as before. The
// profiles themselves are unaffected either way: they are written by the children, which were
// handed their LLVM_PROFILE_FILE long before this.
bool coverageExportRequested(const char *request)
{
    return request == nullptr || std::string_view(request) != "0";
}

// Turn the children's profiles into one merged lcov tracefile beside them.
//
// Best effort throughout: a run whose tests all passed must not fail because llvm-cov or lcov is
// not installed, so every problem is a warning and the tracefile is simply whatever could be
// exported. Silence would be worse than either -- a missing tracefile reads downstream as "this
// code is not covered" -- so each skipped binary, and a missing merge, says so by name.
void writeMergedCoverage(const std::string &directory, const std::vector<std::string> &binaries,
                         const CommandFn &run, const TerminatedFn &terminated = {})
{
    if (directory.empty() || binaries.empty()) {
        return;
    }
    if (!coverageExportRequested(std::getenv(std::string(kCoverageExportEnv).c_str()))) {
        return;
    }
    // A stopped run has nothing worth exporting and no time to do it in. The caller that asked
    // for the stop is usually a watchdog counting the seconds to a SIGKILL, and this phase is
    // llvm work measured in minutes over a run that did not finish. Said out loud, because a
    // silently missing tracefile reads downstream as "this code is not covered".
    if (terminated && terminated()) {
        fmt::print(stderr, "host-test-dispatch: run was stopped -- skipping the coverage export;"
                           " the children's profiles are untouched in {}\n",
                   directory);
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

    std::vector<std::string> tracefiles;   // paths, unioned by lcov below
    size_t exported = 0;
    for (const auto &[binary, profiles] : pending) {
        // Checked every binary, not only at the start: this phase is long enough that a stop
        // usually lands in the middle of it. Nothing already exported is written -- a tracefile
        // holding some of the binaries is indistinguishable downstream from a tree whose other
        // binaries covered nothing.
        if (terminated && terminated()) {
            fmt::print(stderr, "host-test-dispatch: run was stopped after {} of {} binaries --"
                               " abandoning the coverage export rather than writing a partial"
                               " {}\n",
                       exported, pending.size(), kCoverageTracefile);
            return;
        }
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
        tracefiles.push_back(tracefile);
    }

    if (tracefiles.empty()) {
        return;
    }
    const std::string path = (fs::path(directory) / kCoverageTracefile).string();
    // lcov is perl and is not needed to run the tests, so a machine without it still gets every
    // per-binary coverage.lcov -- only the union is lost, and it says so rather than leaving a
    // missing file to read as uncovered code.
    const std::string versionPath = (fs::path(directory) / "lcov-version.txt").string();
    if (run(fmt::format("lcov --version > {} 2>/dev/null", shellQuote(versionPath))) != 0) {
        fmt::print(stderr, "host-test-dispatch: lcov is not installed -- {} was not written; the"
                           " {} per-binary tracefiles are in place under {}\n",
                   kCoverageTracefile, tracefiles.size(), directory);
        return;
    }
    const std::string flags = lcovMergeFlags(readFile(versionPath));
    std::error_code ec;
    fs::remove(versionPath, ec);
    if (run(lcovMergeCommand("lcov", flags, tracefiles, path)) != 0) {
        fmt::print(stderr, "host-test-dispatch: lcov could not union the {} per-binary tracefiles"
                           " -- {} was not written\n",
                   tracefiles.size(), path);
        return;
    }
    fmt::print("----- coverage: merged {} tracefiles into {} -----\n", tracefiles.size(), path);
    std::fflush(stdout);
}

// The operations runSelections() performs on the world outside its loop: starting a child,
// reaping whichever finishes first, signalling one that has outstayed its deadline, reading the
// clock those deadlines are measured on, and naming the file a part writes its report to.
using SpawnFn = std::function<pid_t(const std::string &path, const std::vector<std::string> &args,
                                    const std::string &logPath)>;
// Returns the pid reaped, 0 if *within* elapsed first, or -1 if there was nothing to wait for.
using ReapFn = std::function<pid_t(ChildExit &exit, std::optional<std::chrono::milliseconds> within)>;
using KillFn = std::function<void(pid_t pid, int signal)>;
using NowFn = std::function<std::chrono::steady_clock::time_point()>;
using ReportPathFn = std::function<std::string(const std::string &qualifier, size_t part)>;

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

// Everything the scheduler knows about a child once it has been reaped, other than its report.
struct ChildOutcome {
    ChildExit exit;           // how it ended
    bool abandoned = false;   // it was signalled because the run itself was asked to stop
    bool timedOut = false;    // it was signalled because it outlived its deadline
};

// The facts a fragment's prose needs, none of which the outcome itself carries.
struct ChildContext {
    std::string qualifier;
    std::string logPath;                      // empty when the child inherited our stdio
    std::chrono::seconds timeout{0};          // the deadline it was given
    size_t remainingParts = 0;                // parts of this binary not yet run
};

// What to record for one child, and what it means for the run.
struct ChildVerdict {
    std::vector<SuiteFragment> fragments;
    bool failed = false;           // counts once against the run, however many fragments there are
    bool skipRemainingParts = false;
};

// Decide a child's fate from how it ended and what it left behind.
//
// One function rather than a chain of conditionals in the loop, because the interesting cases
// are combinations -- stopped but already exited, timed out but exited of its own accord, exited
// zero having written a report full of failures -- and each of those is a row here rather than a
// branch ordering somewhere in the scheduler. It is also the whole of the policy, so it can be
// read, and tested, without a scheduler at all.
//
// A killed child's report is thrown away: it stopped partway, so the cases it never reached are
// simply absent, and a report missing them is worse than none. A child that was signalled but
// exited by itself anyway -- the reap that races the kill -- kept everything it wrote, and is
// judged on it like any other.
ChildVerdict classifyChild(const ChildOutcome &outcome, const ChildContext &context,
                           std::optional<SuiteFragment> report)
{
    const std::string where =
        context.logPath.empty() ? std::string() : fmt::format("; output in {}", context.logPath);
    const std::string skipped =
        context.remainingParts == 0
            ? std::string()
            : fmt::format("; its {} remaining part(s) were skipped", context.remainingParts);

    ChildVerdict verdict;
    if (outcome.abandoned && outcome.exit.signalled) {
        verdict.fragments.push_back(crashFragment(
            context.qualifier,
            fmt::format("run was stopped and this binary was killed before it finished{}", where)));
        verdict.failed = true;
        verdict.skipRemainingParts = true;
        return verdict;
    }
    if (outcome.timedOut && outcome.exit.signalled) {
        // Its remaining parts are skipped too, since they are more of the same binary and would
        // each cost another timeout to learn the same thing.
        verdict.fragments.push_back(crashFragment(
            context.qualifier,
            fmt::format("binary timed out after {} s and was killed{}{}", context.timeout.count(),
                        where, skipped)));
        verdict.failed = true;
        verdict.skipRemainingParts = true;
        return verdict;
    }

    bool cleanReport = false;
    if (report) {
        // A report of failures is a failed run even when the binary exited zero: the merged
        // report would otherwise say failures="1" while the front door said everything passed.
        cleanReport = report->failures == 0 && report->errors == 0;
        verdict.failed = !cleanReport;
        verdict.fragments.push_back(std::move(*report));
    } else {
        verdict.fragments.push_back(crashFragment(
            context.qualifier, fmt::format("binary wrote no gtest report (exit status {}){}",
                                           outcome.exit.status, where)));
        verdict.failed = true;
    }
    if (outcome.exit.status != 0) {
        if (cleanReport) {
            verdict.fragments.push_back(crashFragment(
                context.qualifier,
                fmt::format("binary exited {} after writing a passing report",
                            outcome.exit.status)));
        }
        verdict.failed = true;
    }
    if (outcome.abandoned && context.remainingParts != 0) {
        // It finished on its own, but the run is ending, so the rest of this binary never runs.
        verdict.fragments.push_back(crashFragment(
            context.qualifier,
            fmt::format("run was stopped{}", skipped)));
        verdict.failed = true;
        verdict.skipRemainingParts = true;
    }
    return verdict;
}

// Run every part of every selection, at most *schedule.jobs* at a time, and return their
// fragments in selection order. *failures* is incremented once per child that failed.
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

        ChildExit exit;
        const pid_t finished = ops.reap(exit, within);
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
        ChildVerdict verdict = classifyChild(
            ChildOutcome{exit, slot->abandoned, slot->timedOut},
            ChildContext{selection.qualifier, slot->logPath, schedule.childTimeout,
                         filters[index].size() - nextPart[index]},
            parseChildReport(readFile(slot->reportPath), selection.qualifier,
                             schedule.outputFormat));
        for (SuiteFragment &fragment : verdict.fragments) {
            perSelection[index].push_back(std::move(fragment));
        }
        failures += verdict.failed ? 1 : 0;
        if (verdict.skipRemainingParts) {
            nextPart[index] = filters[index].size();
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
        // names come from the manifest alone, which is one open.
        fmt::print("{} runs the host-only test binaries beside it, as one gtest suite.\n",
                   fs::path(argv[0]).filename().string());
        for (const std::string &binary : manifestBinaries(directory)) {
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

    const std::vector<std::string> binaries = manifestBinaries(directory);
    if (binaries.empty()) {
        fmt::print(stderr,
                   "host-test-dispatch: no {} in {} -- the build writes it beside the binaries"
                   " it produced, so re-run cmake, or write the file yourself with one binary"
                   " name per line.\n",
                   kBinaryManifest, directory);
        return 1;
    }

    // Before the enumeration, not after it: listing the binaries is the first thing that runs
    // one, and a binary that hangs in a static constructor hangs it. Installed here, the stop
    // ends the listing the same way it ends a test child.
    installStopHandler();
    const std::chrono::seconds childTimeout =
        resolveChildTimeout(std::getenv(std::string(kChildTimeoutEnv).c_str()));
    const TerminatedFn terminated = [] { return g_stopRequested != 0; };

    const Enumeration enumeration = enumerate(
        directory, binaries,
        [childTimeout](const std::string &binary) {
            return listTestsOutput(binary, childTimeout);
        },
        terminated);
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
        return enumeration.unlistable.empty() && enumeration.unlisted.empty() ? status : 1;
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
        resolveLogDirectory(std::getenv(std::string(kLogDirectoryEnv).c_str()));
    if (!logDir.empty()) {
        if (const std::optional<std::string> problem = prepareLogDirectory(logDir)) {
            fmt::print(stderr, "host-test-dispatch: cannot use {}: {} -- the binaries will"
                               " print to the console instead\n",
                       logDir, *problem);
            logDir.clear();
        }
    }

    std::vector<SuiteFragment> fragments;
    std::vector<std::string> ran;
    int failures = 0;

    // A binary that would not list its cases is reported before any are run: it holds an
    // unknown number of them, so the run cannot be green whatever the rest of the tree does.
    for (const std::string &binary : enumeration.unlistable) {
        fragments.push_back(crashFragment(qualifierFor(binary),
                                          "binary could not list its tests -- it is missing, or "
                                          "failed to start"));
        ++failures;
    }
    // Stopped before they were even asked what they hold. Named for the same reason: a binary
    // that is absent from the report is indistinguishable from one the tree never had.
    for (const std::string &binary : enumeration.unlisted) {
        fragments.push_back(crashFragment(qualifierFor(binary),
                                          "run was stopped before this binary was listed"));
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
        terminated,
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
    writeMergedCoverage(coverageDirectory(), ran, runShell, terminated);

    return failures == 0 ? 0 : 1;
}

#endif  // RCCL_HOST_DISPATCH_NO_MAIN
