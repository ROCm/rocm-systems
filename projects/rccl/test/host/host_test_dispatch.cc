/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// rccl-UnitTestsHostAll -- one gtest-shaped front door for every host-only test binary.
//
// Reads the list of binaries the build put beside it, forks the ones owning a selected case,
// and merges their reports into the one the caller asked for.

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

// Written by the build beside the binaries it lists (RCCL_MICRO_MANIFEST in CMakeLists.txt).
constexpr std::string_view kBinaryManifest = "host_test_binaries.txt";

constexpr std::string_view kMicroSubdirectory = "micro";

// Dropped from a binary's name to form the case-name qualifier.
constexpr std::string_view kQualifierStrip = "rccl-UnitTests";

// Merged lcov tracefile, written beside the .profraw files.
constexpr std::string_view kCoverageTracefile = "host-coverage.lcov";

// Holds one <Qualifier>.log per binary, relative to the working directory.
constexpr std::string_view kDefaultLogDirectory = "host-test-dispatch.logs";

// Overrides kDefaultLogDirectory; "-" lets the children inherit stdout and stderr.
constexpr std::string_view kLogDirectoryEnv = "RCCL_HOST_DISPATCH_LOG_DIR";

constexpr std::string_view kParallelismEnv = "NUM_PARALLEL_HOST_TEST_BINARIES";

// Per-child time limit in seconds; unset or 0 means none.
constexpr std::string_view kChildTimeoutEnv = "RCCL_HOST_DISPATCH_CHILD_TIMEOUT";

// Time between SIGTERM and SIGKILL.
constexpr std::chrono::seconds kTerminateGrace{5};

// Source files kept out of the exported coverage.
constexpr std::string_view kCoverageIgnoreRegex = "(/test/|nvtx)";

// Set to 0 to skip the lcov export.
constexpr std::string_view kCoverageExportEnv = "RCCL_HOST_COVERAGE_EXPORT";

// Longest --gtest_filter per child; must stay under MAX_ARG_STRLEN (128KiB).
constexpr size_t kMaxFilterChars = 60000;

// ── small helpers ───────────────────────────────────────────────────

constexpr bool startsWith(std::string_view text, std::string_view prefix)
{
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

long toLong(std::string_view text, long fallback)
{
    long value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    return error == std::errc() ? value : fallback;
}

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

// Kept split: a parameterized case name can itself contain '.'.
struct TestCaseName {
    std::string suite;
    std::string test;

    bool operator==(const TestCaseName &other) const
    {
        return suite == other.suite && test == other.test;
    }
};

// The cases in one binary's --gtest_list_tests *output*.
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
            // Anything at column 0 other than "Name." is a banner; drop what follows it.
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
// "MicroInitUncached". Dashes are dropped because gtest treats '-' in a filter as negation.
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

std::string qualifiedSuite(const std::string &qualifier, const std::string &suite)
{
    return qualifier + "/" + suite;
}

// ── filter resolution ─────────────────────────────────────────────

struct CandidateCase {
    size_t binary = 0;        // index into the caller's binary list
    std::string suite;        // qualified
    TestCaseName raw;         // as the owning binary knows it
};

namespace {

class NoopTest : public ::testing::Test {
    void TestBody() override {}
};

}  // namespace

// What a filter selected, in candidate order. *suppressed* are matched DISABLED_ cases, which
// gtest reports but does not run.
struct FilterMatches {
    std::vector<size_t> runnable;
    std::vector<size_t> suppressed;
};

// The indices of *candidates* that *filter* selects, decided by gtest by running no-op tests.
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

    // By name: gtest's order groups by suite and need not match registration order.
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

bool isDisabled(const TestCaseName &name)
{
    constexpr std::string_view prefix = "DISABLED_";
    return startsWith(name.suite, prefix) || startsWith(name.test, prefix);
}

// One binary and the cases the caller's filter selects from it.
struct Selection {
    std::string path;
    std::string qualifier;
    std::vector<TestCaseName> cases;
    std::vector<TestCaseName> suppressed;
    size_t runnable = 0;  // cases the binary holds that this run could run at all
};

// The matched *candidates*, grouped into one Selection per owning binary, in binary order.
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
std::vector<std::string> childFilters(const Selection &selection)
{
    if (selection.cases.size() == selection.runnable) {
        return {"*"};
    }
    // Name the suppressed cases too, so gtest reports them as it would under "*".
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

enum class ReportFormat { Xml, Json };

std::string_view formatName(ReportFormat format)
{
    return format == ReportFormat::Json ? "json" : "xml";
}

// One child's report. Only the member matching the run's format is populated, except by
// crashFragment(), which fills both.
struct SuiteFragment {
    std::string xml;
    nlohmann::json suites = nlohmann::json::array();
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

// Prefix every <testsuite name=...> and <testcase classname=...> in *xml* with *qualifier*.
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

// One child's whole JSON report reduced to a fragment, or nullopt if it is not a gtest report.
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

// One child's whole XML report reduced to a fragment, or nullopt if it is not a gtest report.
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

std::optional<SuiteFragment> parseChildReport(const std::string &text, std::string_view qualifier,
                                              ReportFormat format)
{
    return format == ReportFormat::Json ? parseChildJsonReport(text, qualifier)
                                        : parseChildXmlReport(text, qualifier);
}

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

// A failing one-case suite standing in for a child that produced no usable report.
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
        fmt::arg("qualifier", escapeXml(qualifier)), fmt::arg("detail", escapeXml(detail)));

    // The consumer needs status RUN, result COMPLETED and a non-empty failures array.
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
    // Matches gtest's own JSON layout.
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

std::string mergeReports(const std::vector<SuiteFragment> &fragments, ReportFormat format)
{
    return format == ReportFormat::Json ? mergeJsonReports(fragments) : mergeXmlReports(fragments);
}

// ── command line ────────────────────────────────────────────────────

// The help flags gtest itself recognises.
constexpr std::string_view kHelpFlags[] = {"--help", "-h", "-?", "/?"};

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
    std::string outputPath;  // empty: no report
    ReportFormat outputFormat = ReportFormat::Xml;
    std::vector<std::string> passthrough;
};

// A --gtest_output= value ("<format>:<path>", a bare format, or a bare path) applied to
// *options*, following gtest's own rules.
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

// gtest's rule for a boolean GTEST_ variable: set and not "0".
bool environmentFlagIsSet(std::optional<std::string_view> value)
{
    return value && *value != "0";
}

// *args* (argv[1:]) parsed into Options, with *outputEnv*, *filterEnv* and *disabledEnv* standing
// for GTEST_OUTPUT, GTEST_FILTER and GTEST_ALSO_RUN_DISABLED_TESTS. Anything not acted on is
// forwarded to the children verbatim.
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
        options.alsoRunDisabled = true;
        options.passthrough.emplace_back("--gtest_also_run_disabled_tests");
    }
    for (const std::string &arg : args) {
        if (isHelpFlag(arg)) {
            options.help = true;
        } else if (arg == "--gtest_list_tests") {
            options.listTests = true;
        } else if (arg == "--gtest_also_run_disabled_tests") {
            if (!options.alsoRunDisabled) {
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

std::string selfPath()
{
    std::error_code ec;
    const fs::path resolved = fs::read_symlink("/proc/self/exe", ec);
    return ec ? std::string() : resolved.string();
}

std::string selfDirectory(const char *argv0)
{
    std::string path = selfPath();
    if (path.empty()) {
        path = argv0 == nullptr ? "" : argv0;
    }
    const fs::path parent = fs::path(path).parent_path();
    return parent.empty() ? std::string(".") : parent.string();
}

// The binaries named by *directory*'s manifest, in order; empty when there is no manifest.
std::vector<std::string> manifestBinaries(const std::string &directory)
{
    std::ifstream manifest((fs::path(directory) / kBinaryManifest).string());
    if (!manifest) {
        return {};
    }
    std::error_code selfEc;
    const fs::path self = fs::weakly_canonical(selfPath(), selfEc);

    std::vector<std::string> names;
    std::string line;
    while (std::getline(manifest, line)) {
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

std::string binaryDirectory(const std::string &selfDir)
{
    return (fs::path(selfDir) / kMicroSubdirectory).string();
}

// ── the gtest environment the children are given ────────────────────
//
// Consume: acted on here and kept from the children, which would otherwise apply it a second
// time to an already-narrowed case list. Forward: harmless to inherit. Unknown GTEST_* variables
// are consumed.
enum class Disposition { Consume, Forward };

constexpr std::string_view kGtestVariablePrefix = "GTEST_";

constexpr struct {
    std::string_view name;
    Disposition how;
} kGtestVariables[] = {
    {"GTEST_FILTER", Disposition::Consume},
    {"GTEST_OUTPUT", Disposition::Consume},
    {"GTEST_ALSO_RUN_DISABLED_TESTS", Disposition::Consume},
    {"GTEST_TOTAL_SHARDS", Disposition::Consume},
    {"GTEST_SHARD_INDEX", Disposition::Consume},
    {"GTEST_SHARD_STATUS_FILE", Disposition::Consume},
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

bool isForwardedVariable(std::string_view name)
{
    for (const auto &entry : kGtestVariables) {
        if (entry.name == name) {
            return entry.how == Disposition::Forward;
        }
    }
    return false;
}

bool isUnknownGtestVariable(std::string_view name)
{
    if (!startsWith(name, kGtestVariablePrefix)) {
        return false;
    }
    return std::none_of(std::begin(kGtestVariables), std::end(kGtestVariables),
                        [name](const auto &entry) { return entry.name == name; });
}

// The GTEST_ variables currently set that a child must not see.
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

// Called between fork() and exec(); the dispatcher itself keeps the originals.
void stripGtestVariablesInChild(const std::vector<std::string> &names)
{
    for (const std::string &name : names) {
        unsetenv(name.c_str());
    }
}

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

std::string childProfileFile(const std::string &path)
{
    const char *pattern = std::getenv("LLVM_PROFILE_FILE");
    return childProfileFile(pattern == nullptr ? std::string_view() : std::string_view(pattern),
                            path);
}

// ── enumeration ───────────────────────────────────────────────────

struct Enumeration {
    std::vector<CandidateCase> candidates;
    std::vector<std::string> unlistable;  // --gtest_list_tests failed
    std::vector<std::string> unlisted;    // not reached before a stop
};

using ListingFn = std::function<std::optional<std::string>(const std::string &)>;

// Whether the run has been asked to stop.
using TerminatedFn = std::function<bool()>;

// Every case every binary holds, in order. A binary that will not list its cases is recorded,
// never skipped, so it is reported as a failure.
Enumeration enumerate(const std::string &directory, const std::vector<std::string> &binaries,
                      const ListingFn &listing, const TerminatedFn &terminated = {})
{
    Enumeration enumeration;
    for (size_t i = 0; i < binaries.size(); ++i) {
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

std::string childLogPath(std::string_view directory, std::string_view qualifier)
{
    return (fs::path(std::string(directory)) / fmt::format("{}.log", qualifier)).string();
}

// Empty means the children write to the console.
std::string resolveLogDirectory(const char *override_)
{
    if (override_ == nullptr || *override_ == '\0') {
        return std::string(kDefaultLogDirectory);
    }
    return std::string_view(override_) == "-" ? std::string() : std::string(override_);
}

// Why an existing *directory* cannot be trusted with this process's output, or nullopt if it can:
// in a shared parent such as /tmp, another user can create it first and plant symlinks in it.
std::optional<std::string> untrustedDirectoryReason(const std::string &directory)
{
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

// Make *directory* ready to hold this process's output (the children's logs or profiles), or say
// why it cannot be trusted with it.
std::optional<std::string> preparePrivateDirectory(const std::string &directory)
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
    return untrustedDirectoryReason(directory);
}

size_t resolveParallelism(const char *request, size_t fallbackJobs)
{
    const size_t fallback = fallbackJobs < 1 ? 1 : fallbackJobs;
    if (request == nullptr || *request == '\0') {
        return fallback;
    }
    const long requested = toLong(std::string_view(request), 0);
    return requested >= 1 ? static_cast<size_t>(requested) : fallback;
}

// nullopt means no limit; "0" asks for that explicitly.
std::optional<std::chrono::seconds> resolveChildTimeout(const char *request)
{
    if (request == nullptr || *request == '\0') {
        return std::nullopt;
    }
    const long seconds = toLong(std::string_view(request), -1);
    if (seconds == 0) {
        return std::nullopt;
    }
    if (seconds < 0) {
        fmt::print(stderr,
                   "host-test-dispatch: {}=\"{}\" is not a number of seconds -- no child timeout"
                   " will be applied\n",
                   kChildTimeoutEnv, request);
        return std::nullopt;
    }
    return std::chrono::seconds{seconds};
}

// The processors in this process's affinity mask, or 1 when that cannot be determined.
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

size_t defaultParallelism(size_t processors)
{
    return processors / 2 < 1 ? 1 : processors / 2;
}

// Start *path* with *args* in its own process group and return its pid, or -1. A non-empty
// *logPath* receives stdout (and stderr, unless *errPath* is given).
pid_t spawnChild(const std::string &path, const std::vector<std::string> &args,
                 const std::string &logPath = "", const std::string &errPath = "")
{
    std::vector<char *> argv;
    argv.push_back(const_cast<char *>(path.c_str()));
    for (const std::string &arg : args) {
        argv.push_back(const_cast<char *>(arg.c_str()));
    }
    argv.push_back(nullptr);

    const std::vector<std::string> strip = gtestVariablesToStrip();
    std::string profile = childProfileFile(path);
    if (!profile.empty()) {
        // The profile runtime creates this directory only for some patterns (those with %m),
        // so a child given a plain %p one would otherwise fail to write its profile.
        const std::string profileDir = fs::path(profile).parent_path().string();
        if (const std::optional<std::string> problem = preparePrivateDirectory(profileDir)) {
            fmt::print(stderr, "host-test-dispatch: cannot use {}: {} -- {} will not write a"
                               " coverage profile\n",
                       profileDir, *problem, path);
            profile = "/dev/null";
        }
    }
    const pid_t parent = getpid();

    const pid_t pid = fork();
    if (pid < 0) {
        fmt::print(stderr, "host-test-dispatch: fork failed: {}\n", std::strerror(errno));
        return -1;
    }
    if (pid == 0) {
        // Die with the dispatcher; the getppid() check covers it dying before the prctl.
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        if (getppid() != parent) {
            _exit(128 + SIGKILL);
        }
        setpgid(0, 0);
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
    // Also in the parent, so the group exists before any signal is sent.
    setpgid(pid, pid);
    return pid;
}

// Self-pipe the signal handlers write to, so a wait can poll() for a signal and a deadline
// together.
class WakePipe {
public:
    void open()
    {
        if (fds_[0] >= 0) {
            return;
        }
        if (pipe2(fds_, O_CLOEXEC | O_NONBLOCK) != 0) {
            fmt::print(stderr, "host-test-dispatch: cannot create wake pipe: {}\n",
                       std::strerror(errno));
            std::exit(1);
        }
    }

    int readFd() const { return fds_[0]; }

    // Async-signal-safe.
    void wake() const
    {
        const char byte = 1;
        const ssize_t written = ::write(fds_[1], &byte, 1);
        (void)written;
    }

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

volatile std::sig_atomic_t g_stopRequested = 0;

bool stopRequested() { return g_stopRequested != 0; }

extern "C" void requestStop(int)
{
    g_stopRequested = 1;
    g_wake.wake();
}

extern "C" void noteChildExit(int)
{
    g_wake.wake();
}

// Make SIGTERM and SIGINT stop the run rather than end the process, and SIGCHLD wake the wait.
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
    child.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    sigaction(SIGCHLD, &child, nullptr);
}

// Send *signal* to *pid*'s process group, or to *pid* alone when there is no group.
void signalChildGroup(pid_t pid, int signal)
{
    if (::kill(-pid, signal) == 0 || errno != ESRCH) {
        return;
    }
    ::kill(pid, signal);
}

using Deadline = std::optional<std::chrono::steady_clock::time_point>;

// *now* + *limit*, or no deadline when there is no limit.
Deadline deadlineAfter(std::chrono::steady_clock::time_point now,
                       std::optional<std::chrono::seconds> limit)
{
    return limit ? Deadline(now + *limit) : std::nullopt;
}

// How long until *deadline*, never negative; nullopt when there is none.
std::optional<std::chrono::milliseconds> timeUntil(Deadline deadline,
                                                   std::chrono::steady_clock::time_point now)
{
    if (!deadline) {
        return std::nullopt;
    }
    return std::max(std::chrono::milliseconds{0},
                    std::chrono::duration_cast<std::chrono::milliseconds>(*deadline - now));
}

// The TERM, grace, KILL sequence for one child: each advance() sends the next signal and sets
// *due* to when the one after it should be sent. With no due, nothing is signalled until
// advance() is called (e.g. on a stop).
class Escalation {
public:
    Escalation() = default;
    explicit Escalation(Deadline due) : due_(due) {}

    Deadline due() const { return due_; }
    bool termed() const { return signalsSent_ > 0; }
    bool killed() const { return signalsSent_ > 1; }

    void advance(pid_t pid, std::chrono::steady_clock::time_point now,
                 const std::function<void(pid_t, int)> &kill)
    {
        kill(pid, termed() ? SIGKILL : SIGTERM);
        ++signalsSent_;
        due_ = now + kTerminateGrace;
    }

private:
    Deadline due_;
    int signalsSent_ = 0;
};

// Sleep until a signal handler wakes the pipe or *until* passes. Every event a wait cares about
// (a stop, a child exit) writes the pipe, so with no *until* this blocks until one does.
void waitForWake(std::optional<std::chrono::steady_clock::time_point> until)
{
    int timeout = -1;
    if (until) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            *until - std::chrono::steady_clock::now());
        timeout = static_cast<int>(std::max<std::chrono::milliseconds::rep>(0, left.count()));
    }

    struct pollfd woken {};
    woken.fd = g_wake.readFd();
    woken.events = POLLIN;
    if (::poll(&woken, 1, timeout) > 0) {
        g_wake.drain();
    }
}

struct ChildExit {
    int status = 0;          // 128 + signal when it was killed
    bool signalled = false;
};

// Wait for whichever child finishes first and return its pid; 0 when *within* elapsed, or when a
// stop is requested and *returnOnStop*; -1 when there is nothing left to wait for.
pid_t reapAnyChild(ChildExit &exit, std::optional<std::chrono::milliseconds> within,
                   bool returnOnStop)
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
            exit = ChildExit{1, false};
            return -1;
        }
        if (returnOnStop && stopRequested()) {
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

// A mkstemp file removed on destruction.
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

// *path*'s --gtest_list_tests output, or nullopt when the binary failed, timed out or was
// stopped.
std::optional<std::string> listTestsOutput(const std::string &path,
                                           std::optional<std::chrono::seconds> timeout)
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

    Escalation escalation(deadlineAfter(std::chrono::steady_clock::now(), timeout));
    for (;;) {
        ChildExit exit;
        const pid_t finished =
            reapAnyChild(exit, timeUntil(escalation.due(), std::chrono::steady_clock::now()),
                         !escalation.termed());
        if (finished == pid) {
            return (exit.status == 0 && !escalation.termed())
                       ? std::optional<std::string>(readFile(listing.path()))
                       : std::nullopt;
        }
        if (finished < 0) {
            return std::nullopt;
        }
        if (finished > 0) {
            continue;
        }
        // Timed out, stopped, or a grace period ran out.
        if (escalation.killed()) {
            return std::nullopt;
        }
        escalation.advance(pid, std::chrono::steady_clock::now(), signalChildGroup);
    }
}

// The directory LLVM_PROFILE_FILE points into, or empty when coverage is not being collected.
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

// Runs a shell command and returns its exit status.
using CommandFn = std::function<int(const std::string &)>;

// Wait for the shell child *pid* and return its exit status, terminating it if the run is
// stopped.
int waitForShellChild(pid_t pid)
{
    Escalation escalation;
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
        if ((stopRequested() && !escalation.termed()) ||
            (!escalation.killed() && escalation.due() && now >= *escalation.due())) {
            escalation.advance(pid, now, signalChildGroup);
        }
        // After the KILL there is nothing left to send, so wait for the exit alone.
        waitForWake(escalation.killed() ? std::nullopt : escalation.due());
    }
}

// Run *command* through /bin/sh and return its exit status. Not std::system(), which ignores
// SIGINT in the caller while the command runs.
int runShell(const std::string &command)
{
    std::fflush(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        fmt::print(stderr, "host-test-dispatch: fork failed: {}\n", std::strerror(errno));
        return 1;
    }
    if (pid == 0) {
        struct sigaction dfl {};
        dfl.sa_handler = SIG_DFL;
        sigemptyset(&dfl.sa_mask);
        sigaction(SIGINT, &dfl, nullptr);
        sigaction(SIGTERM, &dfl, nullptr);
        sigaction(SIGCHLD, &dfl, nullptr);
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        setpgid(0, 0);
        execl("/bin/sh", "sh", "-c", command.c_str(), static_cast<char *>(nullptr));
        _exit(127);
    }
    setpgid(pid, pid);
    return waitForShellChild(pid);
}

// The llvm tool named *name* from $ROCM_PATH/llvm/bin, falling back to PATH.
std::string llvmTool(std::string_view name)
{
    const char *rocm = std::getenv("ROCM_PATH");
    const fs::path candidate =
        fs::path(rocm == nullptr || *rocm == '\0' ? "/opt/rocm" : rocm) / "llvm" / "bin" / name;
    std::error_code ec;
    return fs::is_regular_file(candidate, ec) ? candidate.string() : std::string(name);
}

std::string profdataMergeCommand(const std::string &tool, const std::vector<std::string> &profiles,
                                 const std::string &output)
{
    std::string command = fmt::format("{} merge -sparse", shellQuote(tool));
    for (const std::string &profile : profiles) {
        command += fmt::format(" {}", shellQuote(profile));
    }
    return command + fmt::format(" -o {}", shellQuote(output));
}

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
// Kept identical to run_host_tests.sh and the python runner's _merge_host_coverage().
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

bool coverageExportRequested(const char *request)
{
    return request == nullptr || std::string_view(request) != "0";
}

// Turn the children's profiles into one merged lcov tracefile beside them. Best effort: every
// problem is a warning.
void writeMergedCoverage(const std::string &directory, const std::vector<std::string> &binaries,
                         const CommandFn &run, const TerminatedFn &terminated = {})
{
    if (directory.empty() || binaries.empty()) {
        return;
    }
    if (!coverageExportRequested(std::getenv(std::string(kCoverageExportEnv).c_str()))) {
        return;
    }
    if (terminated && terminated()) {
        fmt::print(stderr, "host-test-dispatch: run was stopped -- skipping the coverage export;"
                           " the children's profiles are untouched in {}\n",
                   directory);
        return;
    }

    std::vector<std::pair<std::string, std::vector<std::string>>> pending;
    for (const std::string &binary : binaries) {
        const fs::path binaryDir = fs::path(directory) / fs::path(binary).filename();
        // The export writes merged.profdata and coverage.lcov here; never through a directory
        // someone else could have filled with symlinks.
        if (const std::optional<std::string> problem =
                untrustedDirectoryReason(binaryDir.string());
            problem && fs::exists(fs::symlink_status(binaryDir))) {
            fmt::print(stderr, "host-test-dispatch: not exporting coverage from {}: {}\n",
                       binaryDir.string(), *problem);
            continue;
        }
        std::vector<std::string> profiles = profilesIn(binaryDir.string());
        if (!profiles.empty()) {
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

    fmt::print("----- coverage: exporting {} binaries into {} -----\n", pending.size(),
               (fs::path(directory) / kCoverageTracefile).string());
    std::fflush(stdout);

    std::vector<std::string> tracefiles;
    size_t exported = 0;
    for (const auto &[binary, profiles] : pending) {
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
            // The redirect already created it; an empty one would look like a finished export.
            std::error_code ec;
            fs::remove(tracefile, ec);
            continue;
        }
        // A binary that only tests test/ code exports no records, and lcov rejects an empty
        // tracefile, which would sink the whole union.
        const std::string records = readFile(tracefile);
        if (records.rfind("SF:", 0) != 0 && records.find("\nSF:") == std::string::npos) {
            fmt::print("{}: no src/ lines covered, left out of {}\n",
                       fs::path(binary).filename().string(), kCoverageTracefile);
            continue;
        }
        tracefiles.push_back(tracefile);
    }

    if (tracefiles.empty()) {
        return;
    }
    const std::string path = (fs::path(directory) / kCoverageTracefile).string();
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

// The operations runSelections() performs on the world outside its loop.
using SpawnFn = std::function<pid_t(const std::string &path, const std::vector<std::string> &args,
                                    const std::string &logPath)>;
// Returns the pid reaped, 0 if *within* elapsed first (or a stop was requested and
// *returnOnStop*), or -1 if there was nothing to wait for.
using ReapFn = std::function<pid_t(ChildExit &exit, std::optional<std::chrono::milliseconds> within,
                                   bool returnOnStop)>;
using KillFn = std::function<void(pid_t pid, int signal)>;
using NowFn = std::function<std::chrono::steady_clock::time_point()>;
using ReportPathFn = std::function<std::string(const std::string &qualifier, size_t part)>;

struct SchedulerOperations {
    SpawnFn spawn;
    ReapFn reap;
    KillFn kill;
    NowFn now;
    ReportPathFn reportPath;
    TerminatedFn terminated;  // unset: cannot be stopped
};

struct ScheduleOptions {
    size_t jobs = 1;
    ReportFormat outputFormat = ReportFormat::Xml;
    std::vector<std::string> passthrough;
    std::string logDirectory;              // empty: children inherit our stdio
    std::optional<std::chrono::seconds> childTimeout;  // nullopt: no limit
};

struct ChildOutcome {
    ChildExit exit;
    bool abandoned = false;  // signalled because the run was stopped
    bool timedOut = false;
};

struct ChildContext {
    std::string qualifier;
    std::string logPath;  // empty when the child inherited our stdio
    std::optional<std::chrono::seconds> timeout;
    size_t remainingParts = 0;  // parts of this binary not yet run
};

struct ChildVerdict {
    std::vector<SuiteFragment> fragments;
    bool failed = false;
    bool skipRemainingParts = false;
};

// Decide a child's result from how it ended and the report it left. A killed child's report is
// discarded.
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
        verdict.fragments.push_back(crashFragment(
            context.qualifier,
            fmt::format("binary timed out after {} s and was killed{}{}", context.timeout->count(),
                        where, skipped)));
        verdict.failed = true;
        verdict.skipRemainingParts = true;
        return verdict;
    }

    bool cleanReport = false;
    if (report) {
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
        verdict.fragments.push_back(crashFragment(
            context.qualifier,
            fmt::format("run was stopped{}", skipped)));
        verdict.failed = true;
        verdict.skipRemainingParts = true;
    }
    return verdict;
}

// Run every part of every selection, at most *schedule.jobs* at a time, and return their
// fragments in selection order. *failures* is incremented once per child that failed. Parts of
// one selection run one at a time, since they share a log.
std::vector<SuiteFragment> runSelections(const std::vector<Selection> &selections,
                                         const std::vector<std::vector<std::string>> &filters,
                                         const ScheduleOptions &schedule,
                                         const SchedulerOperations &ops, int &failures)
{
    std::vector<size_t> nextPart(selections.size(), 0);
    std::vector<std::vector<SuiteFragment>> perSelection(selections.size());

    struct RunningChild {
        pid_t pid;
        size_t selection;
        std::string reportPath;
        std::string logPath;
        Escalation escalation;  // first due at the child's timeout, if it has one
        bool timedOut = false;
        bool abandoned = false;
    };
    bool stopping = false;
    std::deque<size_t> ready;  // selections with a part still to start
    for (size_t index = 0; index < selections.size(); ++index) {
        ready.push_back(index);
    }
    std::vector<RunningChild> running;

    while (!ready.empty() || !running.empty()) {
        // Stop: start nothing more and terminate what is running.
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
                if (!child.escalation.termed()) {
                    child.escalation.advance(child.pid, now, ops.kill);
                }
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
                               Escalation{deadlineAfter(ops.now(), schedule.childTimeout)}});
        }

        // Bounded by the nearest deadline, if any.
        Deadline nearest;
        for (const RunningChild &child : running) {
            const Deadline due = child.escalation.due();
            if (due && (!nearest || *due < *nearest)) {
                nearest = due;
            }
        }

        ChildExit exit;
        const pid_t finished = ops.reap(exit, timeUntil(nearest, ops.now()), !stopping);
        if (finished == 0) {
            // Signal every child past its deadline and give it another grace period.
            const std::chrono::steady_clock::time_point now = ops.now();
            for (RunningChild &child : running) {
                if (!child.escalation.due() || now < *child.escalation.due()) {
                    continue;
                }
                child.timedOut = !child.abandoned;
                child.escalation.advance(child.pid, now, ops.kill);
            }
            continue;
        }
        const auto slot = std::find_if(running.begin(), running.end(),
                                       [finished](const RunningChild &child) {
                                           return child.pid == finished;
                                       });
        if (slot == running.end() && finished > 0) {
            fmt::print(stderr, "host-test-dispatch: ignoring exit of unknown child {}\n",
                       finished);
            continue;
        }
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
            ready.push_back(index);
        }
    }

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

// A temporary directory for the children's reports, removed on destruction.
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
        fmt::print("{} runs the host-only test binaries beside it, as one gtest suite.\n",
                   fs::path(argv[0]).filename().string());
        for (const std::string &binary : manifestBinaries(directory)) {
            fmt::print("  {}\n", binary);
        }
        fmt::print("\nCase names are qualified with the owning binary, e.g. MicroInit/Suite.Case,"
                   "\nso --gtest_filter patterns can name one binary's cases.\n\n");
        std::fflush(stdout);
        // Then gtest's own help.
        int help_argc = 2;
        char help_flag[] = "--help";
        char *help_argv[] = {argv[0], help_flag, nullptr};
        ::testing::InitGoogleTest(&help_argc, help_argv);
        return 0;
    }

    // Init gtest without the caller's flags or GTEST_OUTPUT, which belong to the children and the
    // merged report; resolveFilter() would otherwise write a report of its own.
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

    // Before the enumeration, so a stop can end a hung listing.
    installStopHandler();
    const std::optional<std::chrono::seconds> childTimeout =
        resolveChildTimeout(std::getenv(std::string(kChildTimeoutEnv).c_str()));
    const TerminatedFn terminated = stopRequested;

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
        ::testing::GTEST_FLAG(list_tests) = true;
        const int status = RUN_ALL_TESTS();
        // An incomplete listing fails.
        return enumeration.unlistable.empty() && enumeration.unlisted.empty() ? status : 1;
    }

    // Not an error, as in gtest; usually an unqualified pattern.
    if (chosen.runnable.empty() && chosen.suppressed.empty() && !candidates.empty()) {
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

    // An unusable log directory falls back to the console.
    std::string logDir =
        resolveLogDirectory(std::getenv(std::string(kLogDirectoryEnv).c_str()));
    if (!logDir.empty()) {
        if (const std::optional<std::string> problem = preparePrivateDirectory(logDir)) {
            fmt::print(stderr, "host-test-dispatch: cannot use {}: {} -- the binaries will"
                               " print to the console instead\n",
                       logDir, *problem);
            logDir.clear();
        }
    }

    std::vector<SuiteFragment> fragments;
    std::vector<std::string> ran;
    int failures = 0;

    for (const std::string &binary : enumeration.unlistable) {
        fragments.push_back(crashFragment(qualifierFor(binary),
                                          "binary could not list its tests -- it is missing, or "
                                          "failed to start"));
        ++failures;
    }
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

    // After the report, so a timeout during the slow export cannot lose it.
    writeMergedCoverage(coverageDirectory(), ran, runShell, terminated);

    return failures == 0 ? 0 : 1;
}

#endif  // RCCL_HOST_DISPATCH_NO_MAIN
