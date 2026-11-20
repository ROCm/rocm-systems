/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// A stand-in for one of the host-only test binaries rccl-UnitTestsHostAll fronts.
//
// It is not a gtest binary: it is the smallest thing that behaves like one from the outside --
// it lists three cases, honours --gtest_filter and --gtest_output, and writes a report in the
// format it was asked for. That is deliberate. The end-to-end test (host-dispatch-e2e-test.cc)
// drives the REAL dispatcher against this, so everything on the dispatcher's side has to be
// real; everything on the child's side only has to be observable.
//
// Two things it does that a real child does not:
//
//   * it records the environment it was given (every GTEST_* variable, and LLVM_PROFILE_FILE)
//     into RCCL_FIXTURE_RECORD_DIR, which is how the test sees what the dispatcher stripped and
//     what it rewrote; and
//   * it can be told to hang (RCCL_FIXTURE_HANG), so the child timeout has something to kill,
//     and to ignore SIGTERM while it hangs (RCCL_FIXTURE_IGNORE_TERM), so a stop has to go the
//     whole way through its grace period and its SIGKILL rather than ending at the first
//     signal.
//
// The case named Fails always fails. A test that wants a green run selects the others.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <csignal>
#include <string_view>
#include <unistd.h>
#include <vector>

extern char **environ;

namespace {

constexpr std::string_view kSuite = "FixtureSuite";

// The cases this binary holds, in listing order, and whether running one fails.
constexpr struct {
    std::string_view name;
    bool fails;
} kCases[] = {
    {"Passes", false},
    {"Fails", true},
    {"DISABLED_Skipped", false},
};

bool startsWith(std::string_view text, std::string_view prefix)
{
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

// Whether *pattern*, a gtest filter pattern with '*' and '?' wildcards, matches *name*.
bool matchesPattern(std::string_view pattern, std::string_view name)
{
    if (pattern.empty()) {
        return name.empty();
    }
    if (pattern.front() == '*') {
        const std::string_view rest = pattern.substr(1);
        for (size_t at = 0; at <= name.size(); ++at) {
            if (matchesPattern(rest, name.substr(at))) {
                return true;
            }
        }
        return false;
    }
    if (name.empty()) {
        return false;
    }
    if (pattern.front() != '?' && pattern.front() != name.front()) {
        return false;
    }
    return matchesPattern(pattern.substr(1), name.substr(1));
}

// Whether *filter* (gtest's positive[-negative] colon-separated form) selects *name*.
bool selects(std::string_view filter, const std::string &name)
{
    std::string_view positive = filter;
    std::string_view negative;
    const size_t dash = filter.find('-');
    if (dash != std::string_view::npos) {
        positive = filter.substr(0, dash);
        negative = filter.substr(dash + 1);
    }

    const auto anyMatch = [&name](std::string_view patterns) {
        size_t at = 0;
        while (at <= patterns.size()) {
            const size_t end = patterns.find(':', at);
            const std::string_view one =
                patterns.substr(at, end == std::string_view::npos ? end : end - at);
            if (matchesPattern(one, name)) {
                return true;
            }
            if (end == std::string_view::npos) {
                break;
            }
            at = end + 1;
        }
        return false;
    };

    if (!positive.empty() && !anyMatch(positive)) {
        return false;
    }
    return negative.empty() || !anyMatch(negative);
}

// The value of *flag* in *argument*, or nullopt when *argument* is some other flag.
const char *flagValue(const char *argument, std::string_view flag)
{
    const std::string prefix = std::string(flag) + "=";
    return startsWith(argument, prefix) ? argument + prefix.size() : nullptr;
}

// Record the environment the dispatcher handed this child, for the test to read back.
void recordEnvironment(const char *argv0)
{
    const char *directory = std::getenv("RCCL_FIXTURE_RECORD_DIR");
    if (directory == nullptr || *directory == '\0') {
        return;
    }
    std::string name(argv0 == nullptr ? "fixture" : argv0);
    const size_t slash = name.rfind('/');
    if (slash != std::string::npos) {
        name = name.substr(slash + 1);
    }
    std::ofstream record(std::string(directory) + "/" + name + ".env",
                         std::ios::binary | std::ios::trunc);
    if (!record) {
        return;
    }
    for (char **entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
        const std::string_view assignment(*entry);
        if (startsWith(assignment, "GTEST_") || startsWith(assignment, "LLVM_PROFILE_FILE=")) {
            record << assignment << "\n";
        }
    }
}

// Record the command line too: what the dispatcher decided to hand this child, rather than what
// the child made of it. The two differ in exactly the cases worth testing -- a whole binary is
// asked for as "*", a partial selection as an explicit list of names.
void recordArguments(int argc, char **argv)
{
    const char *directory = std::getenv("RCCL_FIXTURE_RECORD_DIR");
    if (directory == nullptr || *directory == '\0') {
        return;
    }
    std::string name(argv[0] == nullptr ? "fixture" : argv[0]);
    const size_t slash = name.rfind('/');
    if (slash != std::string::npos) {
        name = name.substr(slash + 1);
    }
    std::ofstream record(std::string(directory) + "/" + name + ".args",
                         std::ios::binary | std::ios::trunc);
    for (int i = 1; record && i < argc; ++i) {
        record << argv[i] << "\n";
    }
}

void writeXmlReport(const std::string &path, const std::vector<size_t> &run,
                    const std::vector<size_t> &suppressed)
{
    std::ofstream report(path, std::ios::binary | std::ios::trunc);
    if (!report) {
        return;
    }
    size_t failures = 0;
    for (const size_t index : run) {
        failures += kCases[index].fails ? 1 : 0;
    }
    const size_t tests = run.size() + suppressed.size();
    report << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           << "<testsuites tests=\"" << tests << "\" failures=\"" << failures << "\" disabled=\""
           << suppressed.size() << "\" errors=\"0\" skipped=\"0\" time=\"0\" name=\"AllTests\">\n"
           << "  <testsuite name=\"" << kSuite << "\" tests=\"" << tests << "\" failures=\""
           << failures << "\" disabled=\"" << suppressed.size()
           << "\" errors=\"0\" time=\"0\">\n";
    for (const size_t index : run) {
        report << "    <testcase name=\"" << kCases[index].name
               << "\" status=\"run\" result=\"completed\" time=\"0\" classname=\"" << kSuite
               << "\"";
        if (kCases[index].fails) {
            report << ">\n      <failure message=\"fixture case asked to fail\" type=\"\"/>\n"
                   << "    </testcase>\n";
        } else {
            report << "/>\n";
        }
    }
    for (const size_t index : suppressed) {
        report << "    <testcase name=\"" << kCases[index].name
               << "\" status=\"notrun\" result=\"suppressed\" time=\"0\" classname=\"" << kSuite
               << "\"/>\n";
    }
    report << "  </testsuite>\n</testsuites>\n";
}

void writeJsonReport(const std::string &path, const std::vector<size_t> &run,
                     const std::vector<size_t> &suppressed)
{
    std::ofstream report(path, std::ios::binary | std::ios::trunc);
    if (!report) {
        return;
    }
    size_t failures = 0;
    for (const size_t index : run) {
        failures += kCases[index].fails ? 1 : 0;
    }
    const size_t tests = run.size() + suppressed.size();
    report << "{\n  \"tests\": " << tests << ",\n  \"failures\": " << failures
           << ",\n  \"disabled\": " << suppressed.size()
           << ",\n  \"errors\": 0,\n  \"skipped\": 0,\n  \"time\": \"0s\",\n"
           << "  \"name\": \"AllTests\",\n  \"testsuites\": [\n    {\n      \"name\": \""
           << kSuite << "\",\n      \"tests\": " << tests << ",\n      \"failures\": " << failures
           << ",\n      \"disabled\": " << suppressed.size()
           << ",\n      \"errors\": 0,\n      \"time\": \"0s\",\n      \"testsuite\": [\n";
    bool first = true;
    const auto one = [&](size_t index, bool ran) {
        if (!first) {
            report << ",\n";
        }
        first = false;
        report << "        {\"name\": \"" << kCases[index].name << "\", \"status\": \""
               << (ran ? "RUN" : "NOTRUN") << "\", \"result\": \""
               << (ran ? "COMPLETED" : "SUPPRESSED") << "\", \"time\": \"0s\", \"classname\": \""
               << kSuite << "\"";
        if (ran && kCases[index].fails) {
            report << ", \"failures\": [{\"failure\": \"fixture case asked to fail\","
                      " \"type\": \"\"}]";
        }
        report << "}";
    };
    for (const size_t index : run) {
        one(index, true);
    }
    for (const size_t index : suppressed) {
        one(index, false);
    }
    report << "\n      ]\n    }\n  ]\n}\n";
}

}  // namespace

int main(int argc, char **argv)
{
    std::string filter = "*";
    std::string outputFormat;
    std::string outputPath;
    bool listTests = false;
    bool alsoRunDisabled = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gtest_list_tests") == 0) {
            listTests = true;
        } else if (std::strcmp(argv[i], "--gtest_also_run_disabled_tests") == 0) {
            alsoRunDisabled = true;
        } else if (const char *value = flagValue(argv[i], "--gtest_filter")) {
            filter = value;
        } else if (const char *spec = flagValue(argv[i], "--gtest_output")) {
            const std::string_view text(spec);
            const size_t colon = text.find(':');
            outputFormat = std::string(text.substr(0, colon));
            outputPath = colon == std::string_view::npos ? "" : std::string(text.substr(colon + 1));
        }
        // Anything else is a passthrough flag a real gtest child would act on; ignored here.
    }

    if (listTests) {
        std::printf("%s.\n", std::string(kSuite).c_str());
        for (const auto &entry : kCases) {
            std::printf("  %s\n", std::string(entry.name).c_str());
        }
        return 0;
    }

    recordEnvironment(argv[0]);
    recordArguments(argc, argv);

    if (const char *hang = std::getenv("RCCL_FIXTURE_HANG"); hang != nullptr && *hang != '\0') {
        // A child that dies on the first SIGTERM never makes the dispatcher wait, so the stop
        // path's grace period -- where its cost is paid -- would go untested. Ignoring the
        // signal here is what makes the dispatcher sit out the whole of it.
        if (const char *deaf = std::getenv("RCCL_FIXTURE_IGNORE_TERM");
            deaf != nullptr && *deaf != '\0') {
            std::signal(SIGTERM, SIG_IGN);
        }
        // Long enough that only a timeout or a stop ends it, short enough that a test which
        // somehow fails to kill this child does not wedge a CI run for ever.
        ::sleep(300);
        return 0;
    }

    // gtest's own rule, which the dispatcher relies on: a DISABLED_ case the filter names is
    // reported, as suppressed, but not run.
    std::vector<size_t> run;
    std::vector<size_t> suppressed;
    for (size_t index = 0; index < sizeof(kCases) / sizeof(kCases[0]); ++index) {
        const std::string name = std::string(kSuite) + "." + std::string(kCases[index].name);
        if (!selects(filter, name)) {
            continue;
        }
        const bool disabled = startsWith(kCases[index].name, "DISABLED_");
        (disabled && !alsoRunDisabled ? suppressed : run).push_back(index);
    }

    if (!outputPath.empty()) {
        if (outputFormat == "json") {
            writeJsonReport(outputPath, run, suppressed);
        } else {
            writeXmlReport(outputPath, run, suppressed);
        }
    }

    for (const size_t index : run) {
        if (kCases[index].fails) {
            return 1;
        }
    }
    return 0;
}
