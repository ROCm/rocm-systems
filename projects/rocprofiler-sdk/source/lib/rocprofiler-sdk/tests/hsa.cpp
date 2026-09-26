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

#include "lib/common/scope_destructor.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/signal_pool.hpp"

#include <gtest/gtest.h>

namespace
{
uint64_t           signal_allocations = 0;
uint64_t           signal_resets      = 0;
hsa_signal_value_t signal_value       = 0;
}  // namespace

TEST(hsa, pooled_signal_reuses_handle_and_resets_value)
{
    namespace hsa = ::rocprofiler::hsa;

    signal_allocations = 0;
    signal_resets      = 0;
    signal_value       = 0;

    auto* core = hsa::get_core_table();
    auto* ext  = hsa::get_amd_ext_table();
    ASSERT_NE(core, nullptr);
    ASSERT_NE(ext, nullptr);

    auto old_create = ext->hsa_amd_signal_create_fn;
    auto old_store  = core->hsa_signal_store_screlease_fn;
    auto restore    = ::rocprofiler::common::scope_destructor{[&]() {
        ext->hsa_amd_signal_create_fn       = old_create;
        core->hsa_signal_store_screlease_fn = old_store;
    }};

    ext->hsa_amd_signal_create_fn = +[](hsa_signal_value_t initial,
                                        uint32_t,
                                        const hsa_agent_t*,
                                        uint64_t,
                                        hsa_signal_t* signal) {
        signal->handle = ++signal_allocations;
        signal_value   = initial;
        return HSA_STATUS_SUCCESS;
    };
    core->hsa_signal_store_screlease_fn = +[](hsa_signal_t, hsa_signal_value_t initial) {
        ++signal_resets;
        signal_value = initial;
    };

    ::rocprofiler::common::container::pool<hsa::signal_t> pool{
        std::piecewise_construct, 1, [](auto& signal) { hsa::construct_hsa_signal(signal); }};

    constexpr auto iterations = size_t{1000};
    auto           last       = hsa_signal_t{};
    for(size_t i = 0; i < iterations; ++i)
    {
        auto& slot = pool.acquire(hsa::construct_hsa_signal, 7, 0, nullptr, 0);
        last       = slot.get().value;
        EXPECT_EQ(signal_value, 7);
        signal_value = -1;
        EXPECT_TRUE(slot.release());
    }

    EXPECT_EQ(signal_allocations, 1);
    EXPECT_EQ(last.handle, 1);
    EXPECT_EQ(signal_resets, iterations);
}

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
