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
#include <iostream>

/**
 * @brief Event listener for HIP test parameter initialization
 * 
 * This listener hooks into Catch2 v3 events to:
 * - Initialize test parameters before test execution
 * - Detect level tags and load level-specific configs
 * - Clean up resources after testing
 */
class HipTestParameterListener : public Catch::EventListenerBase {
public:
    using Catch::EventListenerBase::EventListenerBase;

    /**
     * @brief Called once when the test run begins
     * Initializes TestParameterStore with runtime-detected device info
     */
    void testRunStarting(Catch::TestRunInfo const& testRunInfo) override {
        std::cout << "\n================================================" << std::endl;
        std::cout << "HIP Test Parameter Initialization" << std::endl;
        std::cout << "================================================" << std::endl;
        
        TestParameterStore::instance().initialize();
        
        int currentDevice = 0;
        if (hipGetDevice(&currentDevice) == hipSuccess) {
            DeviceCapabilities::get().initialize(currentDevice);
            std::cout << "\n";
            DeviceCapabilities::get().print();
        }
        
        std::cout << "================================================\n" << std::endl;
    }

    /**
     * @brief Called before each test case starts
     * Detects level tags (e.g., [level_0], [level_1]) and loads level-specific config
     */
    void testCaseStarting(Catch::TestCaseInfo const& testInfo) override {
        auto& params = TestParameterStore::instance();
        
        // Detect level tag from test case
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
        
        // Load level-specific config if detected
        if (!detectedLevel.empty()) {
            if (params.currentTestLevel != detectedLevel) {
                std::cout << "\n[Level Detection] Test: " << testInfo.name << " -> Level: " << detectedLevel << std::endl;
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
        std::cout << "\n[TestParameterStore] Cleaning up..." << std::endl;
        TestParameterStore::instance().clear();
    }
};

// Register the listener - it will be automatically activated
CATCH_REGISTER_LISTENER(HipTestParameterListener)

