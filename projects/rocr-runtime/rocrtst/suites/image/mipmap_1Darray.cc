/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 */

#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <cstring>

#include "suites/image/mipmap_1Darray.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"

static const uint32_t kMipmapWidth = 1024;
static const uint32_t kMipmapHeight = 1;  // For 1D array

#define RET_IF_HSA_ERR(err) { \
    if ((err) != HSA_STATUS_SUCCESS) { \
        const char* msg = 0; \
        hsa_status_string(err, &msg); \
        std::cout << "hsa api call failure at line " << __LINE__ << ", file: " << \
                                  __FILE__ << ". Call returned " << err << std::endl; \
        std::cout << msg << std::endl; \
        return (err); \
    } \
}

static void PrintStatus(hsa_status_t s) {
    const char* msg = nullptr;
    hsa_status_string(s, &msg);
    std::cout << (msg? std::string("(" ) + msg + ")" : std::string()) << std::endl;
}

Mipmap1DArrayTest::Mipmap1DArrayTest(void) :
    TestBase(), test_image_{0}, num_mipmap_levels_(0),
    image_ext_supported(false) {
    set_num_iteration(10);  // Number of iterations to execute of the main test;
                            // This is a default value which can be overridden
                            // on the command line.
    set_title("RocR Mipmap 1D Array Tests");
    set_description("This series of tests check basic mipmap array functionality, including "
                    "mipmap array creation and destruction, and checking mipmap array levels.");

    // Initialize image descriptor
    memset(&mipmap_desc_, 0, sizeof(mipmap_desc_));
    memset(&image_format_, 0, sizeof(image_format_));
}

Mipmap1DArrayTest::~Mipmap1DArrayTest(void) {
}

// Any one-time setup involving member variables used in the rest of the test
// should be done here.
void Mipmap1DArrayTest::SetUp(void) {
    hsa_status_t err;

    TestBase::SetUp();

    err = rocrtst::SetDefaultAgents(this);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    err = rocrtst::SetPoolsTypical(this);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // Check if image extension is supported
    err = hsa_system_extension_supported(HSA_EXTENSION_IMAGES, 1, 0, &image_ext_supported);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    // Set up basic image format (RGBA, 8-bit per channel)
    image_format_.channel_order = HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA;
    image_format_.channel_type = HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8;

    // Set up image descriptor for 1D image
    mipmap_desc_.geometry = HSA_EXT_IMAGE_GEOMETRY_1D;
    mipmap_desc_.width = kMipmapWidth;
    mipmap_desc_.height = kMipmapHeight;
    mipmap_desc_.depth = 1;
    mipmap_desc_.array_size = 1;
    mipmap_desc_.format = image_format_;

    return;
}

void Mipmap1DArrayTest::Run(void) {
    // Compare required profile for this test case with what we're actually
    // running on
    if (!rocrtst::CheckProfile(this)) {
        return;
    }

    TestBase::Run();
}

void Mipmap1DArrayTest::DisplayTestInfo(void) {
    TestBase::DisplayTestInfo();
}

void Mipmap1DArrayTest::DisplayResults(void) const {
    // Compare required profile for this test case with what we're actually
    // running on
    if (!rocrtst::CheckProfile(this)) {
        return;
    }

    return;
}

void Mipmap1DArrayTest::Close(void) {
    // This will close handles opened within rocrtst utility calls and call
    // hsa_shut_down(), so it should be done after other hsa cleanup
    TestBase::Close();
}

void Mipmap1DArrayTest::MipmapCreateDestroy1DArrayTest(void) {
    if (!image_ext_supported) {
        std::cout <<
        "Image extension not supported, skipping MipmapCreateDestroy1DArrayTest" << std::endl;
        return;
    }

    // Print the test information
    std::cout << "Subtest: MipmapCreateDestroy1DArrayTest" << std::endl;
    
    hsa_status_t err;

    // Compute desired full chain levels
    uint32_t max_dim = mipmap_desc_.width > mipmap_desc_.height ?
                       mipmap_desc_.width : mipmap_desc_.height;
    uint32_t max_levels = 1 + (uint32_t)std::floor(std::log2((double)max_dim));
    num_mipmap_levels_ = max_levels; // full chain

    // Get the GPU agents into a vector
    std::vector<hsa_agent_t> agent_list;
    err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &agent_list);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    // Iterate through the GPU agents and run tests for each agent
    for (const auto& gpu_agent : agent_list) {

        // Query the mipmapped array info
        hsa_amd_mipmap_array_info_t info{};
        err = hsa_amd_mipmap_array_get_info(gpu_agent, &mipmap_desc_, num_mipmap_levels_,
                                            HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, &info);
        ASSERT_EQ(HSA_STATUS_SUCCESS, err) << "Information query: FAILED";
        std::cout << "Reported size=" << info.size << " alignment=" << info.alignment
            << " max_levels=" << info.max_levels << " levels_used=" << info.levels_used << std::endl;

        if (info.max_levels < num_mipmap_levels_) {
            std::cout << "Runtime-reported max_levels < num_mipmap_levels_ (" << 
                        info.max_levels << " < " << num_mipmap_levels_ << ")\n";
            err = HSA_STATUS_ERROR_INVALID_ARGUMENT; // treat as failure condition for this test
        } else {        
            // Create mipmap array
            err = hsa_amd_mipmap_array_create(gpu_agent, &mipmap_desc_, nullptr,
                HSA_ACCESS_PERMISSION_RW, num_mipmap_levels_, &test_image_);
            if (err == HSA_STATUS_SUCCESS) {
                std::cout << "    MipmapCreate1DArrayTest: PASSED" << std::endl;

                // Destroy the mipmap array
                err = hsa_amd_mipmap_array_destroy(&test_image_);
                if (err == HSA_STATUS_SUCCESS) {
                    std::cout << "    MipmapDestroy1DArrayTest: PASSED" << std::endl;
                } else {
                    std::cout << "    MipmapDestroy1DArrayTest: FAILED" << std::endl;
                    PrintStatus(err);
                }
            } else {
                std::cout << "    MipmapCreate1DArrayTest: FAILED" << std::endl;
                PrintStatus(err);
            }
        }
        ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    }
}

void Mipmap1DArrayTest::MipmapGetLevel1DArrayTest(void) {
    if (!image_ext_supported) {
        std::cout <<
        "Image extension not supported, skipping MipmapGetLevel1DArrayTest" << std::endl;
        return;
    }

    // Print the test information
    std::cout << "Subtest: MipmapGetLevel1DArrayTest" << std::endl;

    hsa_status_t err;
    hsa_ext_image_t level_image;

    // Compute desired full chain levels
    uint32_t max_dim = mipmap_desc_.width > mipmap_desc_.height ?
                       mipmap_desc_.width : mipmap_desc_.height;
    uint32_t max_levels = 1 + (uint32_t)std::floor(std::log2((double)max_dim));
    num_mipmap_levels_ = max_levels; // full chain

    // Get the GPU agents into a vector
    std::vector<hsa_agent_t> agent_list;
    err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &agent_list);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    // Iterate through the GPU agents and run tests for each agent
    for (const auto& gpu_agent : agent_list) {

        // Query the mipmapped array info
        hsa_amd_mipmap_array_info_t info{};
        err = hsa_amd_mipmap_array_get_info(gpu_agent, &mipmap_desc_, num_mipmap_levels_,
                                            HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, &info);
        ASSERT_EQ(HSA_STATUS_SUCCESS, err) << "Information query: FAILED";
        std::cout << "Reported size=" << info.size << " alignment=" << info.alignment
            << " max_levels=" << info.max_levels << " levels_used=" << info.levels_used << std::endl;

        if (info.max_levels < num_mipmap_levels_) {
            std::cout << "Runtime-reported max_levels < num_mipmap_levels_ (" << 
                        info.max_levels << " < " << num_mipmap_levels_ << ")\n";
            err = HSA_STATUS_ERROR_INVALID_ARGUMENT; // treat as failure condition for this test
        } else {                
            // Create mipmap array
            err = hsa_amd_mipmap_array_create(gpu_agent, &mipmap_desc_, nullptr,
                HSA_ACCESS_PERMISSION_RW, num_mipmap_levels_, &test_image_);
            if (err == HSA_STATUS_SUCCESS) {

                // Get the mipmap level
                for (int i = 0; i < num_mipmap_levels_; i++) {
                    err = hsa_amd_mipmap_array_get_level(gpu_agent, &test_image_, i, &level_image);
                    if (err == HSA_STATUS_SUCCESS) {
                        std::cout << "    MipmapGetLevel1DArrayTest - level " << i << ": PASSED" << std::endl;
                    } else {
                        std::cout << "    MipmapGetLevel1DArrayTest - level " << i << ": FAILED" << std::endl;
                        PrintStatus(err);
                        break;
                    }
                }
                ASSERT_EQ(HSA_STATUS_SUCCESS, err);

                // Destroy the mipmap array
                err = hsa_amd_mipmap_array_destroy(&test_image_);
                if (err != HSA_STATUS_SUCCESS) {
                    std::cout << "    Could not destroy mipmapped array successfully" << std::endl;
                }
            } else {
                std::cout << "    Could not create mipmapped array successfully" << std::endl;
            }
        }
    }
}

#undef RET_IF_HSA_ERR
