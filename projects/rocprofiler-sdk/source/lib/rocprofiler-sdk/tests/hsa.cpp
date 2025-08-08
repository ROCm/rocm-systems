// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <gtest/gtest.h>

TEST(hsa, tables)
{
    namespace hsa = ::rocprofiler::hsa;

    // version of HsaApiTable
    auto version = hsa::get_table_version();

    // HsaApiTable components
    auto* core     = hsa::get_core_table();
    auto* amd_ext  = hsa::get_amd_ext_table();
    auto* fini_ext = hsa::get_fini_ext_table();
    auto* img_ext  = hsa::get_img_ext_table();
    auto* amd_tool = hsa::get_amd_tool_table();

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    auto* pcs_ext = hsa::get_pc_sampling_ext_table();
#endif

    // HsaApiTable instance
    auto table = hsa::get_table();

    //------------------------------------------------------------------------//
    //  checks against HSA headers
    //------------------------------------------------------------------------//

    // make sure the version matches values from HSA header
    EXPECT_EQ(version.major_id, HSA_API_TABLE_MAJOR_VERSION);
    EXPECT_EQ(version.minor_id, sizeof(hsa::hsa_api_table_t));
    EXPECT_EQ(version.step_id, HSA_API_TABLE_STEP_VERSION);

    // make sure the version matches values from HSA header
    EXPECT_EQ(core->version.major_id, HSA_CORE_API_TABLE_MAJOR_VERSION);
    EXPECT_EQ(core->version.minor_id, sizeof(hsa::hsa_core_table_t));
    EXPECT_EQ(core->version.step_id, HSA_CORE_API_TABLE_STEP_VERSION);

    // make sure the version matches values from HSA header
    EXPECT_EQ(amd_ext->version.major_id, HSA_AMD_EXT_API_TABLE_MAJOR_VERSION);
    EXPECT_EQ(amd_ext->version.minor_id, sizeof(hsa::hsa_amd_ext_table_t));
    EXPECT_EQ(amd_ext->version.step_id, HSA_AMD_EXT_API_TABLE_STEP_VERSION);

    // make sure the version matches values from HSA header
    EXPECT_EQ(fini_ext->version.major_id, HSA_FINALIZER_API_TABLE_MAJOR_VERSION);
    EXPECT_EQ(fini_ext->version.minor_id, sizeof(hsa::hsa_fini_ext_table_t));
    EXPECT_EQ(fini_ext->version.step_id, HSA_FINALIZER_API_TABLE_STEP_VERSION);

    // make sure the version matches values from HSA header
    EXPECT_EQ(img_ext->version.major_id, HSA_IMAGE_API_TABLE_MAJOR_VERSION);
    EXPECT_EQ(img_ext->version.minor_id, sizeof(hsa::hsa_img_ext_table_t));
    EXPECT_EQ(img_ext->version.step_id, HSA_IMAGE_API_TABLE_STEP_VERSION);

    // make sure the version matches values from HSA header
    EXPECT_EQ(amd_tool->version.major_id, HSA_TOOLS_API_TABLE_MAJOR_VERSION);
    EXPECT_EQ(amd_tool->version.minor_id, sizeof(hsa::hsa_amd_tool_table_t));
    EXPECT_EQ(amd_tool->version.step_id, HSA_TOOLS_API_TABLE_STEP_VERSION);

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    // make sure the version matches values from HSA header
    EXPECT_EQ(pcs_ext->version.major_id, HSA_PC_SAMPLING_API_TABLE_MAJOR_VERSION);
    EXPECT_EQ(pcs_ext->version.minor_id, sizeof(hsa::hsa_pc_sampling_ext_table_t));
    EXPECT_EQ(pcs_ext->version.step_id, HSA_PC_SAMPLING_API_TABLE_STEP_VERSION);
#endif

    //------------------------------------------------------------------------//
    //  checks between instances
    //------------------------------------------------------------------------//

    // make sure the get_table_version is same as what is in HsaApiTable
    EXPECT_EQ(table.version.major_id, version.major_id);
    EXPECT_EQ(table.version.minor_id, version.minor_id);
    EXPECT_EQ(table.version.step_id, version.step_id);

    // make sure HsaApiTable has same pointers
    EXPECT_EQ(table.core_, core);
    EXPECT_EQ(table.amd_ext_, amd_ext);
    EXPECT_EQ(table.finalizer_ext_, fini_ext);
    EXPECT_EQ(table.image_ext_, img_ext);
}
