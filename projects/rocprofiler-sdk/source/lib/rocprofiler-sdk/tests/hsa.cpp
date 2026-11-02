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

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>

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

TEST(hsa, high_precision_timestamp_contexts)
{
    namespace ctx = ::rocprofiler::context;
    using ::rocprofiler::hsa::needs_high_precision_timestamps;

    EXPECT_FALSE(needs_high_precision_timestamps(nullptr));
    auto config = ctx::context{};
    EXPECT_FALSE(needs_high_precision_timestamps(&config));

    config.callback_tracer = std::make_unique<ctx::callback_tracing_service>();
    ASSERT_EQ(ctx::add_domain(config.callback_tracer->domains,
                              ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API),
              ROCPROFILER_STATUS_SUCCESS);
    EXPECT_FALSE(needs_high_precision_timestamps(&config));
    ASSERT_EQ(ctx::add_domain(config.callback_tracer->domains,
                              ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API),
              ROCPROFILER_STATUS_SUCCESS);
    EXPECT_FALSE(needs_high_precision_timestamps(&config));

    for(auto kind : {ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                     ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                     ROCPROFILER_CALLBACK_TRACING_HIP_EVENT,
                     ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY})
    {
        auto gpu_config            = ctx::context{};
        gpu_config.callback_tracer = std::make_unique<ctx::callback_tracing_service>();
        ASSERT_EQ(ctx::add_domain(gpu_config.callback_tracer->domains, kind),
                  ROCPROFILER_STATUS_SUCCESS);
        EXPECT_TRUE(needs_high_precision_timestamps(&gpu_config)) << kind;
    }
    for(auto kind : {ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                     ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                     ROCPROFILER_BUFFER_TRACING_HIP_EVENT,
                     ROCPROFILER_BUFFER_TRACING_HIP_GRAPH})
    {
        auto gpu_config            = ctx::context{};
        gpu_config.buffered_tracer = std::make_unique<ctx::buffer_tracing_service>();
        ASSERT_EQ(ctx::add_domain(gpu_config.buffered_tracer->domains, kind),
                  ROCPROFILER_STATUS_SUCCESS);
        EXPECT_TRUE(needs_high_precision_timestamps(&gpu_config)) << kind;
    }

    config.dispatch_counter_collection =
        std::make_unique<ctx::dispatch_counter_collection_service>();
    EXPECT_TRUE(needs_high_precision_timestamps(&config));
}

TEST(hsa, high_precision_timestamp_runtime_compatibility)
{
    using ::rocprofiler::hsa::enable_high_precision_timestamps;
    EXPECT_FALSE(enable_high_precision_timestamps(nullptr));

    auto table    = AmdExtTable{};
    table.version = {
        HSA_AMD_EXT_API_TABLE_MAJOR_VERSION, sizeof(table), HSA_AMD_EXT_API_TABLE_STEP_VERSION, 0};
    EXPECT_FALSE(enable_high_precision_timestamps(&table));

#if HSA_AMD_EXT_API_TABLE_MAJOR_VERSION >= 0x02 && HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x16
    static size_t calls                               = 0;
    calls                                             = 0;
    table.hsa_amd_enable_high_precision_timestamps_fn = []() {
        ++calls;
        return HSA_STATUS_SUCCESS;
    };

    // An old or truncated runtime table must never expose the new slot to the SDK.
    table.version.minor_id = offsetof(AmdExtTable, hsa_amd_enable_high_precision_timestamps_fn);
    EXPECT_FALSE(enable_high_precision_timestamps(&table));
    table.version.minor_id += sizeof(table.hsa_amd_enable_high_precision_timestamps_fn) - 1;
    EXPECT_FALSE(enable_high_precision_timestamps(&table));
    EXPECT_EQ(calls, 0);

    table.version.minor_id = sizeof(table);
    ++table.version.major_id;
    EXPECT_FALSE(enable_high_precision_timestamps(&table));
    EXPECT_EQ(calls, 0);
    table.version.major_id = HSA_AMD_EXT_API_TABLE_MAJOR_VERSION;
    EXPECT_TRUE(enable_high_precision_timestamps(&table));
    EXPECT_EQ(calls, 1);

    table.hsa_amd_enable_high_precision_timestamps_fn = []() {
        ++calls;
        return HSA_STATUS_ERROR;
    };
    EXPECT_FALSE(enable_high_precision_timestamps(&table));
    EXPECT_EQ(calls, 2);
#endif
}

#if HSA_AMD_EXT_API_TABLE_MAJOR_VERSION >= 0x02 && HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x16
TEST(hsa, high_precision_timestamp_api_name)
{
    constexpr auto id   = ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_enable_high_precision_timestamps;
    constexpr auto name = "hsa_amd_enable_high_precision_timestamps";
    EXPECT_STREQ(::rocprofiler::hsa::name_by_id<ROCPROFILER_HSA_TABLE_ID_AmdExt>(id), name);
    EXPECT_EQ(::rocprofiler::hsa::id_by_name<ROCPROFILER_HSA_TABLE_ID_AmdExt>(name), id);
}
#endif
