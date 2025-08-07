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
#include <rocprofiler-sdk/hip/table_id.h>
#include <rocprofiler-sdk/hsa/table_id.h>
#include <rocprofiler-sdk/marker/table_id.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/hip/hip.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/marker/marker.hpp"
#include "lib/rocprofiler-sdk/tests/common.hpp"

#include <gtest/gtest.h>

TEST(rocprofiler_lib, api_id_names)
{
    auto callback_names = get_callback_tracing_names();
    auto buffered_names = get_buffer_tracing_names();

    EXPECT_EQ(callback_names.kind_names.size(), ROCPROFILER_CALLBACK_TRACING_LAST);
    EXPECT_EQ(buffered_names.kind_names.size(), ROCPROFILER_BUFFER_TRACING_LAST);

    // HSA callback
    EXPECT_EQ(callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API).size(),
              ROCPROFILER_HSA_CORE_API_ID_LAST);
    EXPECT_EQ(
        callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API).size(),
        ROCPROFILER_HSA_AMD_EXT_API_ID_LAST);
    EXPECT_EQ(
        callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API).size(),
        ROCPROFILER_HSA_IMAGE_EXT_API_ID_LAST);
    EXPECT_EQ(
        callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API).size(),
        ROCPROFILER_HSA_FINALIZE_EXT_API_ID_LAST);

    // HSA buffer
    EXPECT_EQ(buffered_names.operation_names.at(ROCPROFILER_BUFFER_TRACING_HSA_CORE_API).size(),
              ROCPROFILER_HSA_CORE_API_ID_LAST);
    EXPECT_EQ(buffered_names.operation_names.at(ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API).size(),
              ROCPROFILER_HSA_AMD_EXT_API_ID_LAST);
    EXPECT_EQ(
        buffered_names.operation_names.at(ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API).size(),
        ROCPROFILER_HSA_IMAGE_EXT_API_ID_LAST);
    EXPECT_EQ(
        buffered_names.operation_names.at(ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API).size(),
        ROCPROFILER_HSA_FINALIZE_EXT_API_ID_LAST);

    // HIP callback
    EXPECT_EQ(
        callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API).size(),
        ROCPROFILER_HIP_RUNTIME_API_ID_LAST);
    EXPECT_EQ(
        callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API).size(),
        ROCPROFILER_HIP_COMPILER_API_ID_LAST);

    // HIP buffer
    EXPECT_EQ(buffered_names.operation_names.at(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API).size(),
              ROCPROFILER_HIP_RUNTIME_API_ID_LAST);
    EXPECT_EQ(buffered_names.operation_names.at(ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API).size(),
              ROCPROFILER_HIP_COMPILER_API_ID_LAST);

    // Marker callback
    EXPECT_EQ(
        callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API).size(),
        ROCPROFILER_MARKER_CORE_API_ID_LAST);
    EXPECT_EQ(
        callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API).size(),
        ROCPROFILER_MARKER_CONTROL_API_ID_LAST);
    EXPECT_EQ(
        callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API).size(),
        ROCPROFILER_MARKER_NAME_API_ID_LAST);

    // Marker buffer
    EXPECT_EQ(buffered_names.operation_names.at(ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API).size(),
              ROCPROFILER_MARKER_CORE_API_ID_LAST);
    EXPECT_EQ(
        buffered_names.operation_names.at(ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API).size(),
        ROCPROFILER_MARKER_CONTROL_API_ID_LAST);
    EXPECT_EQ(buffered_names.operation_names.at(ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API).size(),
              ROCPROFILER_MARKER_NAME_API_ID_LAST);
    EXPECT_EQ(
        buffered_names.operation_names.at(ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API).size(),
        ROCPROFILER_MARKER_CORE_RANGE_API_ID_LAST);

    // Code object callback
    EXPECT_EQ(callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT).size(),
              ROCPROFILER_CODE_OBJECT_LAST);

    {
        auto hsa_core_ids     = ::rocprofiler::hsa::get_ids<ROCPROFILER_HSA_TABLE_ID_Core>();
        auto hsa_amd_ext_ids  = ::rocprofiler::hsa::get_ids<ROCPROFILER_HSA_TABLE_ID_AmdExt>();
        auto hsa_img_ext_ids  = ::rocprofiler::hsa::get_ids<ROCPROFILER_HSA_TABLE_ID_ImageExt>();
        auto hsa_fini_ext_ids = ::rocprofiler::hsa::get_ids<ROCPROFILER_HSA_TABLE_ID_FinalizeExt>();

        auto hsa_core_names    = ::rocprofiler::hsa::get_names<ROCPROFILER_HSA_TABLE_ID_Core>();
        auto hsa_amd_ext_names = ::rocprofiler::hsa::get_names<ROCPROFILER_HSA_TABLE_ID_AmdExt>();
        auto hsa_img_ext_names = ::rocprofiler::hsa::get_names<ROCPROFILER_HSA_TABLE_ID_ImageExt>();
        auto hsa_fini_ext_names =
            ::rocprofiler::hsa::get_names<ROCPROFILER_HSA_TABLE_ID_FinalizeExt>();

        ASSERT_EQ(hsa_core_ids.size(), hsa_core_names.size());
        ASSERT_EQ(hsa_amd_ext_ids.size(), hsa_amd_ext_names.size());
        ASSERT_EQ(hsa_img_ext_ids.size(), hsa_img_ext_names.size());
        ASSERT_EQ(hsa_fini_ext_ids.size(), hsa_fini_ext_names.size());

        for(auto itr : hsa_core_ids)
        {
            EXPECT_EQ(std::string_view{hsa_core_names.at(itr)},
                      callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API)
                          .at(itr));

            EXPECT_EQ(itr,
                      ::rocprofiler::hsa::id_by_name<ROCPROFILER_HSA_TABLE_ID_Core>(
                          hsa_core_names.at(itr)));

            EXPECT_EQ(
                std::string_view{
                    ::rocprofiler::hsa::name_by_id<ROCPROFILER_HSA_TABLE_ID_Core>(itr)},
                std::string_view{hsa_core_names.at(itr)});
        }

        for(auto itr : hsa_amd_ext_ids)
        {
            EXPECT_EQ(
                std::string_view{hsa_amd_ext_names.at(itr)},
                callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API)
                    .at(itr));

            EXPECT_EQ(itr,
                      ::rocprofiler::hsa::id_by_name<ROCPROFILER_HSA_TABLE_ID_AmdExt>(
                          hsa_amd_ext_names.at(itr)));

            EXPECT_EQ(
                std::string_view{
                    ::rocprofiler::hsa::name_by_id<ROCPROFILER_HSA_TABLE_ID_AmdExt>(itr)},
                std::string_view{hsa_amd_ext_names.at(itr)});
        }

        for(auto itr : hsa_img_ext_ids)
        {
            EXPECT_EQ(
                std::string_view{hsa_img_ext_names.at(itr)},
                callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API)
                    .at(itr));

            EXPECT_EQ(itr,
                      ::rocprofiler::hsa::id_by_name<ROCPROFILER_HSA_TABLE_ID_ImageExt>(
                          hsa_img_ext_names.at(itr)));

            EXPECT_EQ(
                std::string_view{
                    ::rocprofiler::hsa::name_by_id<ROCPROFILER_HSA_TABLE_ID_ImageExt>(itr)},
                std::string_view{hsa_img_ext_names.at(itr)});
        }

        for(auto itr : hsa_fini_ext_ids)
        {
            EXPECT_EQ(
                std::string_view{hsa_fini_ext_names.at(itr)},
                callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API)
                    .at(itr));

            EXPECT_EQ(itr,
                      ::rocprofiler::hsa::id_by_name<ROCPROFILER_HSA_TABLE_ID_FinalizeExt>(
                          hsa_fini_ext_names.at(itr)));

            EXPECT_EQ(
                std::string_view{
                    ::rocprofiler::hsa::name_by_id<ROCPROFILER_HSA_TABLE_ID_FinalizeExt>(itr)},
                std::string_view{hsa_fini_ext_names.at(itr)});
        }
    }

    {
        auto hip_comp_ids = ::rocprofiler::hip::get_ids<ROCPROFILER_HIP_TABLE_ID_Compiler>();
        auto hip_run_ids  = ::rocprofiler::hip::get_ids<ROCPROFILER_HIP_TABLE_ID_Runtime>();

        auto hip_comp_names = ::rocprofiler::hip::get_names<ROCPROFILER_HIP_TABLE_ID_Compiler>();
        auto hip_run_names  = ::rocprofiler::hip::get_names<ROCPROFILER_HIP_TABLE_ID_Runtime>();

        ASSERT_EQ(hip_comp_ids.size(), hip_comp_names.size());
        ASSERT_EQ(hip_run_ids.size(), hip_run_names.size());

        for(auto itr : hip_comp_ids)
        {
            EXPECT_EQ(
                std::string_view{hip_comp_names.at(itr)},
                callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API)
                    .at(itr));

            EXPECT_EQ(itr,
                      ::rocprofiler::hip::id_by_name<ROCPROFILER_HIP_TABLE_ID_Compiler>(
                          hip_comp_names.at(itr)));

            EXPECT_EQ(
                std::string_view{
                    ::rocprofiler::hip::name_by_id<ROCPROFILER_HIP_TABLE_ID_Compiler>(itr)},
                std::string_view{hip_comp_names.at(itr)});
        }

        for(auto itr : hip_run_ids)
        {
            EXPECT_EQ(
                std::string_view{hip_run_names.at(itr)},
                callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API)
                    .at(itr));

            EXPECT_EQ(itr,
                      ::rocprofiler::hip::id_by_name<ROCPROFILER_HIP_TABLE_ID_Runtime>(
                          hip_run_names.at(itr)));

            EXPECT_EQ(
                std::string_view{
                    ::rocprofiler::hip::name_by_id<ROCPROFILER_HIP_TABLE_ID_Runtime>(itr)},
                std::string_view{hip_run_names.at(itr)});
        }
    }

    {
        auto marker_core_ids =
            ::rocprofiler::marker::get_ids<ROCPROFILER_MARKER_TABLE_ID_RoctxCore>();
        auto marker_ctrl_ids =
            ::rocprofiler::marker::get_ids<ROCPROFILER_MARKER_TABLE_ID_RoctxControl>();
        auto marker_name_ids =
            ::rocprofiler::marker::get_ids<ROCPROFILER_MARKER_TABLE_ID_RoctxName>();

        auto marker_core_names =
            ::rocprofiler::marker::get_names<ROCPROFILER_MARKER_TABLE_ID_RoctxCore>();
        auto marker_ctrl_names =
            ::rocprofiler::marker::get_names<ROCPROFILER_MARKER_TABLE_ID_RoctxControl>();
        auto marker_name_names =
            ::rocprofiler::marker::get_names<ROCPROFILER_MARKER_TABLE_ID_RoctxName>();

        ASSERT_EQ(marker_core_ids.size(), marker_core_names.size());
        ASSERT_EQ(marker_ctrl_ids.size(), marker_ctrl_names.size());
        ASSERT_EQ(marker_name_ids.size(), marker_name_names.size());

        for(auto itr : marker_core_ids)
        {
            EXPECT_EQ(
                std::string_view{marker_core_names.at(itr)},
                callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API)
                    .at(itr));

            EXPECT_EQ(itr,
                      ::rocprofiler::marker::id_by_name<ROCPROFILER_MARKER_TABLE_ID_RoctxCore>(
                          marker_core_names.at(itr)));

            EXPECT_EQ(
                std::string_view{
                    ::rocprofiler::marker::name_by_id<ROCPROFILER_MARKER_TABLE_ID_RoctxCore>(itr)},
                std::string_view{marker_core_names.at(itr)});
        }

        for(auto itr : marker_ctrl_ids)
        {
            EXPECT_EQ(
                std::string_view{marker_ctrl_names.at(itr)},
                callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API)
                    .at(itr));

            EXPECT_EQ(itr,
                      ::rocprofiler::marker::id_by_name<ROCPROFILER_MARKER_TABLE_ID_RoctxControl>(
                          marker_ctrl_names.at(itr)));

            EXPECT_EQ(
                std::string_view{
                    ::rocprofiler::marker::name_by_id<ROCPROFILER_MARKER_TABLE_ID_RoctxControl>(
                        itr)},
                std::string_view{marker_ctrl_names.at(itr)});
        }

        for(auto itr : marker_name_ids)
        {
            EXPECT_EQ(
                std::string_view{marker_name_names.at(itr)},
                callback_names.operation_names.at(ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API)
                    .at(itr));

            EXPECT_EQ(itr,
                      ::rocprofiler::marker::id_by_name<ROCPROFILER_MARKER_TABLE_ID_RoctxName>(
                          marker_name_names.at(itr)));

            EXPECT_EQ(
                std::string_view{
                    ::rocprofiler::marker::name_by_id<ROCPROFILER_MARKER_TABLE_ID_RoctxName>(itr)},
                std::string_view{marker_name_names.at(itr)});
        }
    }
}
