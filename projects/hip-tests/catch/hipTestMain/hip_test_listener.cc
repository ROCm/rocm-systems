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
#include <hip_test_params.hh>
#include <hip_test_context.hh>
#include <regex>
#include <cstdlib>

/**
 * @brief Event listener for HIP test parameter initialization
 * 
 * This listener hooks into Catch2 v3 events to:
 * - Initialize test parameters before test execution
 * - Detect level filter from command-line args
 * - Load level-specific configs based on filter
 * - Clean up resources after testing
 */
class HipTestParameterListener : public Catch::EventListenerBase {
public:
    using Catch::EventListenerBase::EventListenerBase;
    
private:
    std::string filterLevel; // Detected level from command-line filter
    
    /**
     * @brief Parse command-line arguments to detect level filter
     * Looks for patterns like: ./test "[level_0]" or ./test [level_1]
     * Also checks HIP_TEST_LEVEL environment variable as fallback
     */
    std::string detectLevelFromCommandLine() {
        // Check environment variable first
        if (const char* envLevel = std::getenv("HIP_TEST_LEVEL")) {
            std::string level = envLevel;
            LogPrintf("[Level Filter] Detected from HIP_TEST_LEVEL: %s\n", level.c_str());
            return level;
        }
        
        // Parse command-line arguments (stored in Catch2 config)
        // Unfortunately, Catch2 doesn't expose original argc/argv in listeners
        // So we'll need to use environment variable or detect from running tests
        
        // For now, return empty - we'll detect from first test that runs
        return "";
    }

public:

    /**
     * @brief Called once when the test run begins
     * Initializes TestParameterStore and detects level filter
     */
    void testRunStarting(Catch::TestRunInfo const& testRunInfo) override {
        // Detect level filter from environment or command-line
        filterLevel = detectLevelFromCommandLine();
        
        TestParameterStore::instance().initialize();
        
        // If level was specified, load it immediately
        if (!filterLevel.empty()) {
            LogPrintf("[Level Filter] Applying global level: %s\n", filterLevel.c_str());
            TestParameterStore::instance().loadLevelConfig(filterLevel);
        }
    }

    /**
     * @brief Called before each test case starts
     * Uses filter level if specified, otherwise detects from test tags
     */
    void testCaseStarting(Catch::TestCaseInfo const& testInfo) override {
        auto& params = TestParameterStore::instance();
        
        // Priority 1: Use filter level if explicitly set (from env or detected)
        if (!filterLevel.empty()) {
            // Filter level takes precedence - all tests use same parameters
            if (params.currentTestLevel != filterLevel) {
                LogPrintf("[Level Filter] Test: %s -> Using filter level: %s\n", 
                          testInfo.name.c_str(), filterLevel.c_str());
                params.loadLevelConfig(filterLevel);
            }
            return;
        }
        
        // Priority 2: Auto-detect from first test's level tag (filter inference)
        // This handles: ./test "[level_1]" where we detect level_1 from first test
        std::string detectedLevel = "";
        for (const auto& tag : testInfo.tags) {
            std::string tagStr = std::string(tag.original);
            
            // Remove brackets: "[level_0]" -> "level_0"
            if (tagStr.size() > 2 && tagStr.front() == '[' && tagStr.back() == ']') {
                tagStr = tagStr.substr(1, tagStr.size() - 2);
            }
            
            // Check if it's a level tag
            if (tagStr.find("level_") == 0) {
                detectedLevel = tagStr;
                break;
            }
        }
        
        // If this is the first test with a level tag, set it as filter level
        if (!detectedLevel.empty() && filterLevel.empty()) {
            filterLevel = detectedLevel;
            LogPrintf("[Level Auto-Detection] Inferred filter level: %s from test: %s\n", 
                      filterLevel.c_str(), testInfo.name.c_str());
        }
        
        // Load level-specific config if detected
        if (!detectedLevel.empty()) {
            if (params.currentTestLevel != detectedLevel) {
                LogPrintf("[Level Detection] Test: %s -> Level: %s\n", 
                          testInfo.name.c_str(), detectedLevel.c_str());
                params.loadLevelConfig(detectedLevel);
            }
        } else {
            // Reset to defaults if no level tag
            if (!params.currentTestLevel.empty()) {
                params.currentTestLevel = "";
            }
        }
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

