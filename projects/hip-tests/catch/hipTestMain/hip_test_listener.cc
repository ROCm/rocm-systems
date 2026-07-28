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
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <hip_test_params.hh>
#include <hip_test_context.hh>
#include <regex>
#include <set>
#include <sstream>
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
 * Both sources are parsed with the same rules, following the Catch2 test spec
 * grammar: a comma separated list of filters that are OR'd, each filter a
 * sequence of AND'd patterns that may be negated with '~'. A pattern counts
 * only if it reads exactly "level_N", so a malformed tag ("[level_2x]") is
 * ignored, while an exclusion ("~[level_2]") removes that level from the set
 * that can still run. Of the levels that remain, the highest wins:
 *   "[level_1],[level_2]"    -> level_2
 *   "~[level_3]~[level_4]"   -> level_2 (highest functional level left)
 *   "[level_0]~[level_0]"    -> no level (nothing left to run)
 * The single exception between the two sources is that HIP_TEST_LEVEL also
 * accepts the bare form without brackets ("level_2", "level_1,level_2",
 * "~level_4").
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
     * @brief Collect every level a tag expression can select.
     *
     * The expression is read the way Catch2 reads a test spec: a comma
     * separated list of filters OR'd together, each filter a sequence of
     * patterns AND'd together, each pattern optionally negated with '~'.
     * Patterns that do not read exactly "level_N" (test names, other tags, and
     * malformed tags such as "[level_2x]") place no constraint on the level and
     * are ignored.
     *
     * A filter naming levels contributes those levels, minus the ones it
     * excludes; a filter that only excludes levels contributes every supported
     * level except the excluded ones; a filter mentioning no level at all
     * contributes nothing. Note that filters are OR'd, so "[level_1],~[level_2]"
     * also runs level_3 and level_4 tests, exactly as Catch2 would select them.
     *
     * @param allowBareTags when true, a pattern may also be written without the
     *        surrounding brackets ("level_2", "~level_2"), which HIP_TEST_LEVEL
     *        accepts. Brackets, when present, must still be balanced.
     * @return Distinct level numbers selectable, sorted ascending (empty if the
     *         expression places no constraint on the level).
     */
    static std::set<int> collectLevels(const std::string& text, bool allowBareTags) {
        static const std::regex bracketedPattern("(~?)\\[([^\\[\\]]*)\\]");
        static const std::regex barePattern("\\s*(~?)\\s*([^\\s]+)\\s*");

        // Level named by a pattern, or -1 when it does not name one. Anything
        // but an exact "level_N" is rejected; the digit count is bounded so
        // that the conversion cannot overflow.
        const auto parseLevel = [](const std::string& tag) {
            static const std::regex levelRegex("level_([0-9]{1,9})");
            std::smatch match;
            return std::regex_match(tag, match, levelRegex) ? std::stoi(match[1].str()) : -1;
        };

        std::set<int> levels;
        std::stringstream stream(text);
        std::string filter;
        while (std::getline(stream, filter, ',')) {
            std::set<int> included, excluded;

            // Sort this filter's level patterns into included and excluded.
            const auto sortPattern = [&](const std::string& negation, const std::string& tag) {
                const int level = parseLevel(tag);
                if (level >= 0) {
                    (negation.empty() ? included : excluded).insert(level);
                }
            };

            bool bracketed = false;
            for (auto it = std::sregex_iterator(filter.begin(), filter.end(), bracketedPattern);
                 it != std::sregex_iterator(); ++it) {
                bracketed = true;
                sortPattern((*it)[1].str(), (*it)[2].str());
            }
            if (!bracketed && allowBareTags) {
                std::smatch match;
                if (std::regex_match(filter, match, barePattern)) {
                    sortPattern(match[1].str(), match[2].str());
                }
            }

            if (included.empty()) {
                if (excluded.empty()) {
                    continue;  // filter says nothing about the level
                }
                // Exclusion only: everything the suite supports is still in play.
                for (const char* supported : kSupportedLevels) {
                    const int level = parseLevel(supported);
                    if (level >= 0) {
                        included.insert(level);
                    }
                }
            }
            for (const int level : included) {
                if (excluded.count(level) == 0) {
                    levels.insert(level);
                }
            }
        }
        return levels;
    }

    /**
     * @brief Reduce a set of levels to a single level string ("level_N").
     * Picks the highest level and warns when more than one can be selected.
     * @return "level_N" for the highest level, or "" if the set is empty.
     */
    std::string highestLevel(const std::set<int>& levels, const char* source) {
        if (levels.empty()) {
            return "";
        }
        if (levels.size() > 1) {
            LogPrintf("[Level Filter] Multiple levels selected via %s; using highest (level_%d)\n",
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
                const auto found = collectLevels(arg, false);
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
            std::string level = highestLevel(collectLevels(envLevel, true),
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
