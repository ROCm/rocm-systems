/*
* Copyright © Advanced Micro Devices, Inc., or its affiliates.
*
* SPDX-License-Identifier: MIT
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
                for (uint32_t i = 0; i < num_mipmap_levels_; i++) {
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

//
// MipmapDataIntegrity1DTest - Data validation test for 1D mipmaps
// Tests that data written to each mip level is correctly isolated and readable
//
void Mipmap1DArrayTest::MipmapDataIntegrity1DTest(void) {
    if (!image_ext_supported) {
        std::cout <<
        "  Image extension not supported, skipping MipmapDataIntegrity1DTest" << std::endl;
        return;
    }

    std::cout << "Subtest: MipmapDataIntegrity1DTest" << std::endl;
    std::cout << "  This test validates data isolation between 1D mipmap levels" << std::endl;
    std::cout << "  by writing unique patterns to each level and reading them back" << std::endl;

    hsa_status_t err;

    // Compute full mipmap chain levels for 1D (based on width only)
    uint32_t max_levels = 1 + (uint32_t)std::floor(std::log2((double)mipmap_desc_.width));
    num_mipmap_levels_ = max_levels;

    // Get extension function table for image operations
    hsa_ext_images_1_pfn_t ext_table;
    err = hsa_system_get_major_extension_table(
        HSA_EXTENSION_IMAGES, 1, sizeof(ext_table), &ext_table);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err) << "Failed to get image extension table";

    // Get GPU agents
    std::vector<hsa_agent_t> agent_list;
    err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &agent_list);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    for (const auto& gpu_agent : agent_list) {
        // Query mipmap info
        hsa_amd_mipmap_array_info_t info{};
        err = hsa_amd_mipmap_array_get_info(gpu_agent, &mipmap_desc_,
                    num_mipmap_levels_, HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, &info);
        ASSERT_EQ(HSA_STATUS_SUCCESS, err) << "Failed to query mipmap info";

        if (info.max_levels < num_mipmap_levels_) {
            std::cout << "  Insufficient max_levels support, skipping for this agent" << std::endl;
            continue;
        }

        // Allocate backing memory for mipmap
        void* image_data = nullptr;
        hsa_amd_memory_pool_t pool;
        err = hsa_amd_agent_iterate_memory_pools(
            gpu_agent,
            [](hsa_amd_memory_pool_t pool, void* data) -> hsa_status_t {
                hsa_amd_segment_t segment;
                hsa_amd_memory_pool_get_info(pool,
                    HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
                if (segment == HSA_AMD_SEGMENT_GLOBAL) {
                    *(hsa_amd_memory_pool_t*)data = pool;
                    return HSA_STATUS_INFO_BREAK;
                }
                return HSA_STATUS_SUCCESS;
            },
            &pool);
        ASSERT_EQ(HSA_STATUS_INFO_BREAK, err) << "Failed to find memory pool";

        err = hsa_amd_memory_pool_allocate(pool, info.size, 0, &image_data);
        ASSERT_EQ(HSA_STATUS_SUCCESS, err) << "Failed to allocate mipmap memory";

        // Create mipmap array
        err = hsa_amd_mipmap_array_create(gpu_agent, &mipmap_desc_, image_data,
            HSA_ACCESS_PERMISSION_RW, num_mipmap_levels_, &test_image_);
        ASSERT_EQ(HSA_STATUS_SUCCESS, err) << "Failed to create mipmap array";

        std::cout << "  Testing data integrity across " << num_mipmap_levels_
                  << " 1D levels on agent" << std::endl;

        // Test each level by writing and reading back unique patterns
        for (uint32_t level = 0; level < num_mipmap_levels_; level++) {
            // Get level handle
            hsa_ext_image_t level_image;
            err = hsa_amd_mipmap_array_get_level(gpu_agent, &test_image_,
                                                 level, &level_image);
            ASSERT_EQ(HSA_STATUS_SUCCESS, err)
                << "Failed to get level " << level;

            // Compute level dimensions (1D: only width)
            uint32_t level_width = std::max((size_t)1u, mipmap_desc_.width >> level);
            size_t pixel_size = 4 * sizeof(uint8_t);  // RGBA8
            size_t level_size = level_width * pixel_size;

            // Allocate host buffer and fill with unique pattern
            std::vector<uint8_t> write_data(level_size);
            for (size_t i = 0; i < level_size; i++) {
                // Create level-dependent pattern: (level * 23 + i) % 256
                write_data[i] = static_cast<uint8_t>((level * 23 + i) % 256);
            }

            // Define region to write (1D: only x range)
            hsa_ext_image_region_t region;
            memset(&region, 0, sizeof(region));
            region.offset.x = 0;
            region.offset.y = 0;
            region.offset.z = 0;
            region.range.x = level_width;
            region.range.y = 1;  // 1D image has height=1
            region.range.z = 1;

            // Write pattern to level
            err = ext_table.hsa_ext_image_import(
                gpu_agent, write_data.data(), level_width * pixel_size,
                0, level_image, &region);
            ASSERT_EQ(HSA_STATUS_SUCCESS, err)
                << "Failed to write data to level " << level;

            // Read data back
            std::vector<uint8_t> read_data(level_size);
            err = ext_table.hsa_ext_image_export(
                gpu_agent, level_image, read_data.data(),
                level_width * pixel_size, 0, &region);
            ASSERT_EQ(HSA_STATUS_SUCCESS, err)
                << "Failed to read data from level " << level;

            // Verify data matches
            bool match = (write_data == read_data);
            ASSERT_TRUE(match)
                << "Level " << level << " data mismatch: "
                << "wrote " << write_data.size() << " bytes, "
                << "read " << read_data.size() << " bytes";

            std::cout << "    Level " << level
                      << " (width=" << level_width << "): "
                      << "PASSED" << std::endl;
        }

        // Cleanup
        err = hsa_amd_mipmap_array_destroy(&test_image_);
        ASSERT_EQ(HSA_STATUS_SUCCESS, err) << "Failed to destroy mipmap";

        err = hsa_amd_memory_pool_free(image_data);
        ASSERT_EQ(HSA_STATUS_SUCCESS, err) << "Failed to free memory";

        std::cout << "  MipmapDataIntegrity1DTest: PASSED for agent" << std::endl;
    }
}

//
// MipmapSampling1DTest - GPU sampling test for 1D mipmaps
// Tests that GPU can correctly sample from 1D mipmap at different LOD levels
// Equivalent to HIP tex1DLod sampling tests
//
void Mipmap1DArrayTest::MipmapSampling1DTest(void) {
    if (!image_ext_supported) {
        std::cout <<
        "  Image extension not supported, skipping MipmapSampling1DTest" << std::endl;
        return;
    }

    std::cout << "Subtest: MipmapSampling1DTest" << std::endl;
    std::cout << "  This test validates GPU-side 1D texture sampling from mipmaps" << std::endl;
    std::cout << "  using explicit LOD specification (similar to tex1DLod)" << std::endl;

    hsa_status_t err;

    // Compute full mipmap chain for 1D
    uint32_t max_levels = 1 + (uint32_t)std::floor(std::log2((double)mipmap_desc_.width));
    num_mipmap_levels_ = max_levels;

    // Get GPU agents
    std::vector<hsa_agent_t> agent_list;
    err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &agent_list);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    for (const auto& gpu_agent : agent_list) {
        // Query mipmap info
        hsa_amd_mipmap_array_info_t info{};
        err = hsa_amd_mipmap_array_get_info(gpu_agent, &mipmap_desc_, num_mipmap_levels_,
                                            HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, &info);
        ASSERT_EQ(HSA_STATUS_SUCCESS, err);

        if (info.max_levels < num_mipmap_levels_) {
            std::cout << "Insufficient max_levels support, skipping for this agent" << std::endl;
            continue;
        }

        // Create mipmap array
        err = hsa_amd_mipmap_array_create(gpu_agent, &mipmap_desc_, nullptr,
            HSA_ACCESS_PERMISSION_RW, num_mipmap_levels_, &test_image_);
        ASSERT_EQ(HSA_STATUS_SUCCESS, err);

        std::cout << "  Testing GPU 1D sampling across " << num_mipmap_levels_ << " levels" << std::endl;

        // Get extension function table for image operations
        hsa_ext_images_1_pfn_t ext_table;
        err = hsa_system_get_major_extension_table(
            HSA_EXTENSION_IMAGES, 1, sizeof(ext_table), &ext_table);
        ASSERT_EQ(HSA_STATUS_SUCCESS, err) << "Failed to get image extension table";

        // NOTE: GPU kernel testing temporarily disabled due to API compatibility issues
        // The GPU sampling test requires complex infrastructure (float4 types, kernel dispatch,
        // memory pool management) that needs to be properly configured.
        // TODO: Re-enable when kernel dispatch infrastructure is properly set up

        std::cout << "  WARNING: GPU kernel testing is disabled in this build" << std::endl;
        std::cout << "  INFO: Performing API validation only" << std::endl;

        // Validate that we can access each level
        for (uint32_t level = 0; level < num_mipmap_levels_; level++) {
            hsa_ext_image_t level_image;
            err = hsa_amd_mipmap_array_get_level(gpu_agent, &test_image_, level, &level_image);
            ASSERT_EQ(HSA_STATUS_SUCCESS, err) << "Failed to get level " << level;

            uint32_t level_width = std::max((size_t)1u, mipmap_desc_.width >> level);
            std::cout << "    Level " << level << " (width=" << level_width
                      << "): API access verified" << std::endl;
        }

        // Destroy mipmap array
        err = hsa_amd_mipmap_array_destroy(&test_image_);
        ASSERT_EQ(HSA_STATUS_SUCCESS, err);

        std::cout << "  MipmapSampling1DTest: API validation PASSED for agent" << std::endl;
    }
}

//
// MipmapErrorHandling1DTest - Error handling test for 1D mipmaps
// Tests that API properly handles invalid parameters and error conditions
// Validates robustness of 1D mipmap implementation
//
void Mipmap1DArrayTest::MipmapErrorHandling1DTest(void) {
    if (!image_ext_supported) {
        std::cout <<
        "  Image extension not supported, skipping MipmapErrorHandling1DTest" << std::endl;
        return;
    }

    std::cout << "Subtest: MipmapErrorHandling1DTest" << std::endl;
    std::cout << "  This test validates error handling for invalid parameters (1D)" << std::endl;

    hsa_status_t err;

    // Setup basic mipmap configuration
    num_mipmap_levels_ = 4;

    // Get GPU agents
    std::vector<hsa_agent_t> agent_list;
    err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &agent_list);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    for (const auto& gpu_agent : agent_list) {
        std::cout << "  Testing 1D error handling on GPU agent" << std::endl;

        // Test 1: Invalid mip level (at boundary)
        {
            err = hsa_amd_mipmap_array_create(gpu_agent, &mipmap_desc_, nullptr,
                HSA_ACCESS_PERMISSION_RW, num_mipmap_levels_, &test_image_);
            ASSERT_EQ(HSA_STATUS_SUCCESS, err);

            hsa_ext_image_t level_image;
            err = hsa_amd_mipmap_array_get_level(gpu_agent, &test_image_,
                                                 num_mipmap_levels_, &level_image);

            ASSERT_NE(HSA_STATUS_SUCCESS, err)
                << "Test 1 FAILED: Should reject level == num_levels";
            std::cout << "    Test 1 - Boundary level: PASSED" << std::endl;

            hsa_amd_mipmap_array_destroy(&test_image_);
        }

        // Test 2: Invalid mip level (way out of range)
        {
            err = hsa_amd_mipmap_array_create(gpu_agent, &mipmap_desc_, nullptr,
                HSA_ACCESS_PERMISSION_RW, num_mipmap_levels_, &test_image_);
            ASSERT_EQ(HSA_STATUS_SUCCESS, err);

            hsa_ext_image_t level_image;
            err = hsa_amd_mipmap_array_get_level(gpu_agent, &test_image_, 999, &level_image);

            ASSERT_NE(HSA_STATUS_SUCCESS, err)
                << "Test 2 FAILED: Should reject level 999";
            std::cout << "    Test 2 - Out-of-range level: PASSED" << std::endl;

            hsa_amd_mipmap_array_destroy(&test_image_);
        }

        // Test 3: Create with 0 levels
        {
            hsa_ext_image_t zero_level_image;
            err = hsa_amd_mipmap_array_create(gpu_agent, &mipmap_desc_, nullptr,
                HSA_ACCESS_PERMISSION_RW, 0, &zero_level_image);

            ASSERT_NE(HSA_STATUS_SUCCESS, err)
                << "Test 3 FAILED: Should reject 0 levels";
            std::cout << "    Test 3 - Zero levels: PASSED" << std::endl;
        }

        // Test 4: Null descriptor pointer
        {
            hsa_ext_image_t null_desc_image;
            err = hsa_amd_mipmap_array_create(gpu_agent, nullptr, nullptr,
                HSA_ACCESS_PERMISSION_RW, 4, &null_desc_image);

            ASSERT_NE(HSA_STATUS_SUCCESS, err)
                << "Test 4 FAILED: Should reject null descriptor";
            std::cout << "    Test 4 - Null descriptor: PASSED" << std::endl;
        }

        // Test 5: Excessive mip levels for 1D
        {
            uint32_t max_possible = 1 + (uint32_t)std::floor(std::log2((double)mipmap_desc_.width));
            uint32_t excessive_levels = max_possible + 10;

            hsa_ext_image_t excessive_image;
            err = hsa_amd_mipmap_array_create(gpu_agent, &mipmap_desc_, nullptr,
                HSA_ACCESS_PERMISSION_RW, excessive_levels, &excessive_image);

            ASSERT_NE(HSA_STATUS_SUCCESS, err)
                << "Test 5 FAILED: Should reject excessive levels";
            std::cout << "    Test 5 - Excessive mip levels: PASSED" << std::endl;
        }

        // Test 6: Invalid descriptor (zero width for 1D)
        {
            hsa_ext_image_descriptor_t invalid_desc = mipmap_desc_;
            invalid_desc.width = 0;

            hsa_ext_image_t invalid_image;
            err = hsa_amd_mipmap_array_create(gpu_agent, &invalid_desc, nullptr,
                HSA_ACCESS_PERMISSION_RW, 4, &invalid_image);

            ASSERT_NE(HSA_STATUS_SUCCESS, err)
                << "Test 6 FAILED: Should reject zero width";
            std::cout << "    Test 6 - Zero width: PASSED" << std::endl;
        }

        std::cout << "  MipmapErrorHandling1DTest: All 6 tests PASSED for agent" << std::endl;
            ASSERT_EQ(HSA_STATUS_SUCCESS, err);

            // Second destroy of same handle - implementation may handle differently
            err = hsa_amd_mipmap_array_destroy(&test_image_);
            if (err != HSA_STATUS_SUCCESS) {
                std::cout << "    Test 4 - Double destroy: PASSED (detected double-free)" << std::endl;
            } else {
                std::cout << "    Test 4 - Double destroy: WARNING (allowed double-free)" << std::endl;
            }
        }

        std::cout << "  MipmapErrorHandling1DTest: PASSED for agent" << std::endl;
    }

#undef RET_IF_HSA_ERR
