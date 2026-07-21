/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <hip_test_params.hh>
#include <hip_test_context.hh>
#include <regex>
#include <set>
#include <string>
#include <cstdlib>
#include <cstdio>

/**
 * @brief Event listener for HIP test parameter initialization
 *
 * This listener hooks into Catch2 v3 events to select a single test level for
 * the whole run and load its parameters (memory sizes, block sizes,
 * iterations, ...) into the TestParameterStore.
 *
 * The active level is resolved once, at the start of the run, using a strict
 * priority order:
 *   1. Catch2 command-line tag filter, e.g. ./test "[level_2]"
 *   2. HIP_TEST_LEVEL environment variable, e.g. HIP_TEST_LEVEL=level_2
 *   3. Hardcoded default (kDefaultLevel)
 *
 * A higher-priority source is used whenever it yields a level; the lower ones
 * are then ignored (strict precedence, no merging across sources). Note that
 * under ctest each test is launched by name (not by a "[level_X]" filter), so
 * the command-line source is empty there and HIP_TEST_LEVEL is the effective
 * control knob.
 *
 * The resolved level is validated against the supported set (kSupportedLevels).
 * An unsupported level (e.g. HIP_TEST_LEVEL=level_9) is treated as a fatal
 * misconfiguration and aborts the run rather than silently testing with the
 * wrong parameters.
 *
 * Both the command-line filter and HIP_TEST_LEVEL accept multiple levels using
 * the Catch2 tag format (e.g. "[level_1],[level_2]"); when more than one is
 * given the highest level wins. HIP_TEST_LEVEL additionally accepts the bare
 * form without brackets (e.g. "level_2" or "level_1,level_2").
 */
class HipTestParameterListener : public Catch::EventListenerBase {
public:
    using Catch::EventListenerBase::EventListenerBase;

private:
    /// Levels understood by the test suite (must match definitions.yaml).
    static constexpr const char* kSupportedLevels[] = {"level_0", "level_1", "level_2",
                                                       "level_3", "level_4"};

    /// Level used when neither the command line nor HIP_TEST_LEVEL specify one.
    static constexpr const char* kDefaultLevel = "level_2";

    std::string filterLevel;

    /// @brief Whether @p level is one of kSupportedLevels.
    static bool isSupportedLevel(const std::string& level) {
        for (const char* supported : kSupportedLevels) {
            if (level == supported) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Collect all level numbers appearing in a string.
     * @return Distinct level numbers found, sorted ascending (empty if none).
     */
    std::set<int> collectLevels(const std::string& text, bool requireBrackets) {
        const std::regex levelRegex(requireBrackets ? "\\[level_(\\d+)\\]"
                                                    : "\\[?level_(\\d+)\\]?");
        std::set<int> levels;
        for (auto it = std::sregex_iterator(text.begin(), text.end(), levelRegex);
             it != std::sregex_iterator(); ++it) {
            levels.insert(std::stoi((*it)[1].str()));
        }
        return levels;
    }

    /**
     * @brief Reduce a set of levels to a single level string ("level_N").
     * Picks the highest level and warns when more than one was requested.
     * @return "level_N" for the highest level, or "" if the set is empty.
     */
    std::string highestLevel(const std::set<int>& levels, const char* source) {
        if (levels.empty()) {
            return "";
        }
        if (levels.size() > 1) {
            LogPrintf("[Level Filter] Multiple levels requested via %s; using highest (level_%d)\n",
                      source, *levels.rbegin());
        }
        return "level_" + std::to_string(*levels.rbegin());
    }

    /**
     * @brief Resolve the active level using the strict priority order.
     */
    std::string detectLevelFilter() {
        // Priority 1: Catch2 command-line tag filter (e.g. ./test "[level_2]").
        if (m_config != nullptr) {
            std::set<int> cliLevels;
            for (const auto& arg : m_config->getTestsOrTags()) {
                const auto found = collectLevels(arg, true);
                cliLevels.insert(found.begin(), found.end());
            }
            std::string level = highestLevel(cliLevels, "command line");
            if (!level.empty()) {
                LogPrintf("[Level Filter] Detected from command line: %s\n", level.c_str());
                return level;
            }
        }

        // Priority 2: HIP_TEST_LEVEL environment variable.
        if (const char* envLevel = std::getenv("HIP_TEST_LEVEL")) {
            std::string level = highestLevel(collectLevels(envLevel, false),
                                             "HIP_TEST_LEVEL");
            if (!level.empty()) {
                LogPrintf("[Level Filter] Detected from HIP_TEST_LEVEL: %s\n", level.c_str());
                return level;
            }
            LogPrintf("[Level Filter] HIP_TEST_LEVEL='%s' has no valid level, ignoring\n", envLevel);
        }

        // Priority 3: Hardcoded default.
        LogPrintf("[Level Filter] Using default level: %s\n", kDefaultLevel);
        return kDefaultLevel;
    }

public:

    /**
     * @brief Called once when the test run begins
     * Initializes TestParameterStore and loads the resolved level's parameters.
     */
    void testRunStarting(Catch::TestRunInfo const& testRunInfo) override {
        auto& params = TestParameterStore::instance();
        params.initialize();

        filterLevel = detectLevelFilter();

        if (!isSupportedLevel(filterLevel)) {
            LogPrintf("[Level Filter] ERROR: '%s' is not a supported level. Aborting.\n",
                      filterLevel.c_str());
            std::exit(EXIT_FAILURE);
        }

        LogPrintf("[Level Filter] Applying global level: %s\n", filterLevel.c_str());
        params.loadLevelConfig(filterLevel);
    }

    /**
     * @brief Called when test run ends
     * Cleanup resources
     */
    void testRunEnded(Catch::TestRunStats const& testRunStats) override {
        TestParameterStore::instance().clear();
    }
};

// Register the listener - it will be automatically activated
CATCH_REGISTER_LISTENER(HipTestParameterListener)
