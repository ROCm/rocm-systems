// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>
#include <algorithm>
#include <random>

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/enum_string.hpp>

#define ROCPROFILER_STRNG_IMPL(A)     #A
#define ROCPROFILER_STRNG(A)          ROCPROFILER_STRNG_IMPL(A)
#define ROCPROFILER_CONCAT_2(A, B)    A##B
#define ROCPROFILER_CONCAT_3(A, B, C) A##B##C
#define ROCPROFILER_CONCAT_NAME(A, B) ROCPROFILER_CONCAT_3(A, _, B)

using namespace rocprofiler;

#define TEST_STR(A) EXPECT_EQ(std::string_view{#A}, rocprofiler::sdk::get_enum_label(A))
#define TEST_API_ID_STR(PREFIX, API_NAME)                                                          \
    EXPECT_EQ(std::string_view{ROCPROFILER_STRNG(ROCPROFILER_CONCAT_NAME(PREFIX, API_NAME))},      \
              rocprofiler::sdk::get_enum_label(ROCPROFILER_CONCAT_NAME(PREFIX, API_NAME)))

namespace
{
auto&
get_generator()
{
    static std::mt19937 mt{};
    return mt;
}

template <typename EnumT>
std::vector<EnumT>
get_random_values(int count = 5)
{
    namespace details = rocprofiler::sdk::details;
    std::vector<EnumT> ret(count, EnumT{});

    // Define the distribution
    std::uniform_int_distribution<size_t> gen(details::rocprofiler_enum_info<EnumT>::begin,
                                              details::rocprofiler_enum_info<EnumT>::end - 1);
    std::generate(ret.begin(), ret.end(), [&] { return static_cast<EnumT>(gen(get_generator())); });
    return ret;
}

}  // namespace

TEST(enum_string, table_id)
{
    using namespace rocprofiler::sdk;
    using namespace std::string_view_literals;

    TEST_STR(ROCPROFILER_HSA_TABLE_ID_Core);
    TEST_STR(ROCPROFILER_HSA_TABLE_ID_AmdExt);
    TEST_STR(ROCPROFILER_HSA_TABLE_ID_ImageExt);
    TEST_STR(ROCPROFILER_HSA_TABLE_ID_FinalizeExt);
    TEST_STR(ROCPROFILER_MARKER_TABLE_ID_RoctxCore);
    TEST_STR(ROCPROFILER_MARKER_TABLE_ID_RoctxControl);
    TEST_STR(ROCPROFILER_MARKER_TABLE_ID_RoctxName);
    TEST_STR(ROCPROFILER_RCCL_TABLE_ID);
    TEST_STR(ROCPROFILER_ROCDECODE_TABLE_ID_CORE);
    TEST_STR(ROCPROFILER_ROCJPEG_TABLE_ID_CORE);
}

TEST(enum_string, fwd_h)
{
    using namespace rocprofiler::sdk;

    // rocprofiler_status_t
    TEST_STR(ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT);
    TEST_STR(ROCPROFILER_STATUS_ERROR_BUFFER_BUSY);
    TEST_STR(ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED);

    TEST_STR(ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT_START);
    TEST_STR(ROCPROFILER_PAGE_MIGRATION_QUEUE_EVICTION);

    // rocprofiler_buffer_category_t
    TEST_STR(ROCPROFILER_BUFFER_CATEGORY_TRACING);
    TEST_STR(ROCPROFILER_BUFFER_CATEGORY_COUNTERS);

    // rocprofiler_agent_type_t
    TEST_STR(ROCPROFILER_AGENT_TYPE_CPU);
    TEST_STR(ROCPROFILER_AGENT_TYPE_GPU);

    // rocprofiler_callback_phase_t
    TEST_STR(ROCPROFILER_CALLBACK_PHASE_ENTER);

    // rocprofiler_callback_tracing_kind_t
    TEST_STR(ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API);
    TEST_STR(ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API);
    TEST_STR(ROCPROFILER_CALLBACK_TRACING_RCCL_API);
    TEST_STR(ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API);

    // rocprofiler_buffer_tracing_kind_t
    TEST_STR(ROCPROFILER_BUFFER_TRACING_HSA_CORE_API);
    TEST_STR(ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API);
    TEST_STR(ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY);
    TEST_STR(ROCPROFILER_BUFFER_TRACING_ROCDECODE_API);

    // rocprofiler_code_object_operation_t
    TEST_STR(ROCPROFILER_CODE_OBJECT_NONE);
    TEST_STR(ROCPROFILER_CODE_OBJECT_LOAD);
    TEST_STR(ROCPROFILER_CODE_OBJECT_HOST_KERNEL_SYMBOL_REGISTER);

    // rocprofiler_memory_copy_operation_t
    TEST_STR(ROCPROFILER_MEMORY_COPY_NONE);
    TEST_STR(ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE);
    TEST_STR(ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE);

    // rocprofiler_memory_allocation_operation_t
    TEST_STR(ROCPROFILER_MEMORY_ALLOCATION_NONE);
    TEST_STR(ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE);
    TEST_STR(ROCPROFILER_MEMORY_ALLOCATION_VMEM_FREE);

    // rocprofiler_kernel_dispatch_operation_t
    TEST_STR(ROCPROFILER_KERNEL_DISPATCH_NONE);
    TEST_STR(ROCPROFILER_KERNEL_DISPATCH_ENQUEUE);
    TEST_STR(ROCPROFILER_KERNEL_DISPATCH_COMPLETE);

    // rocprofiler_pc_sampling_method_t
    TEST_STR(ROCPROFILER_PC_SAMPLING_METHOD_NONE);
    TEST_STR(ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC);
    TEST_STR(ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP);

    // rocprofiler_pc_sampling_unit_t
    TEST_STR(ROCPROFILER_PC_SAMPLING_UNIT_NONE);
    TEST_STR(ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS);
    TEST_STR(ROCPROFILER_PC_SAMPLING_UNIT_TIME);

    // rocprofiler_buffer_policy_t
    TEST_STR(ROCPROFILER_BUFFER_POLICY_NONE);
    TEST_STR(ROCPROFILER_BUFFER_POLICY_DISCARD);
    TEST_STR(ROCPROFILER_BUFFER_POLICY_LOSSLESS);

    // rocprofiler_page_migration_operation_t
    TEST_STR(ROCPROFILER_PAGE_MIGRATION_NONE);
    TEST_STR(ROCPROFILER_PAGE_MIGRATION_DROPPED_EVENT);

    // rocprofiler_scratch_memory_operation_t
    TEST_STR(ROCPROFILER_SCRATCH_MEMORY_NONE);
    TEST_STR(ROCPROFILER_SCRATCH_MEMORY_ALLOC);
    TEST_STR(ROCPROFILER_SCRATCH_MEMORY_ASYNC_RECLAIM);

    // rocprofiler_runtime_initialization_operation_t
    TEST_STR(ROCPROFILER_RUNTIME_INITIALIZATION_HSA);
    TEST_STR(ROCPROFILER_RUNTIME_INITIALIZATION_MARKER);
    TEST_STR(ROCPROFILER_RUNTIME_INITIALIZATION_ROCDECODE);

    // rocprofiler_counter_info_version_id_t
    TEST_STR(ROCPROFILER_COUNTER_INFO_VERSION_0);

    // rocprofiler_counter_record_kind_t
    TEST_STR(ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER);

    // rocprofiler_counter_flag_t
    TEST_STR(ROCPROFILER_COUNTER_FLAG_APPEND_DEFINITION);

    // rocprofiler_pc_sampling_record_kind_t
    TEST_STR(ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE);

    // rocprofiler_external_correlation_id_request_kind_t
    TEST_STR(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HSA_CORE_API);
    TEST_STR(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_ROCDECODE_API);

    // rocprofiler_att_parameter_type_t
    TEST_STR(ROCPROFILER_ATT_PARAMETER_TARGET_CU);
    TEST_STR(ROCPROFILER_ATT_PARAMETER_PERFCOUNTERS_CTRL);
}

TEST(enum_string, hip_api_id)
{
    TEST_API_ID_STR(ROCPROFILER_HIP_COMPILER_API_ID, __hipPopCallConfiguration);
    TEST_API_ID_STR(ROCPROFILER_HIP_COMPILER_API_ID, __hipUnregisterFatBinary);

    TEST_API_ID_STR(ROCPROFILER_HIP_RUNTIME_API_ID, hipApiName);
    TEST_API_ID_STR(ROCPROFILER_HIP_RUNTIME_API_ID, hipLaunchCooperativeKernel_spt);

    TEST_API_ID_STR(ROCPROFILER_HIP_RUNTIME_API_ID, hipMemcpyHtoAAsync);
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 1
    TEST_API_ID_STR(ROCPROFILER_HIP_RUNTIME_API_ID, hipGetProcAddress);
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 2
    TEST_API_ID_STR(ROCPROFILER_HIP_RUNTIME_API_ID, hipStreamBeginCaptureToGraph);
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 3
    TEST_API_ID_STR(ROCPROFILER_HIP_RUNTIME_API_ID, hipMemcpyAtoA);
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 4
    TEST_API_ID_STR(ROCPROFILER_HIP_RUNTIME_API_ID, hipDrvGraphExecMemsetNodeSetParams);
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 5
    TEST_API_ID_STR(ROCPROFILER_HIP_RUNTIME_API_ID, hipExtHostAlloc);
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 6
    TEST_API_ID_STR(ROCPROFILER_HIP_RUNTIME_API_ID, hipDeviceGetTexture1DLinearMaxWidth);
#endif
    // TEST_API_ID_STR(ROCPROFILER_HIP_RUNTIME_API_ID, LAST);
}

TEST(enum_string, hsa_api_id)
{
    TEST_API_ID_STR(ROCPROFILER_HSA_CORE_API_ID, hsa_init);
    TEST_API_ID_STR(ROCPROFILER_HSA_CORE_API_ID, hsa_executable_freeze);
    TEST_API_ID_STR(ROCPROFILER_HSA_CORE_API_ID, hsa_executable_iterate_program_symbols);

    TEST_API_ID_STR(ROCPROFILER_HSA_AMD_EXT_API_ID, hsa_amd_coherency_get_type);
    TEST_API_ID_STR(ROCPROFILER_HSA_AMD_EXT_API_ID, hsa_amd_spm_release);
#if HSA_AMD_EXT_API_TABLE_MAJOR_VERSION >= 0x02
    TEST_API_ID_STR(ROCPROFILER_HSA_AMD_EXT_API_ID, hsa_amd_vmem_handle_create);
    TEST_API_ID_STR(ROCPROFILER_HSA_AMD_EXT_API_ID, hsa_amd_vmem_retain_alloc_handle);
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x01
    TEST_API_ID_STR(ROCPROFILER_HSA_AMD_EXT_API_ID, hsa_amd_agent_set_async_scratch_limit);
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x02
    TEST_API_ID_STR(ROCPROFILER_HSA_AMD_EXT_API_ID, hsa_amd_queue_get_info);
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x03
    TEST_API_ID_STR(ROCPROFILER_HSA_AMD_EXT_API_ID, hsa_amd_vmem_address_reserve_align);
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x04
    TEST_API_ID_STR(ROCPROFILER_HSA_AMD_EXT_API_ID, hsa_amd_enable_logging);
#    endif
#endif

    TEST_API_ID_STR(ROCPROFILER_HSA_FINALIZE_EXT_API_ID, hsa_ext_program_create);
    TEST_API_ID_STR(ROCPROFILER_HSA_FINALIZE_EXT_API_ID, hsa_ext_program_finalize);

    TEST_API_ID_STR(ROCPROFILER_HSA_IMAGE_EXT_API_ID, hsa_ext_image_get_capability);
    TEST_API_ID_STR(ROCPROFILER_HSA_IMAGE_EXT_API_ID, hsa_ext_image_create_with_layout);

    TEST_STR(ROCPROFILER_SCRATCH_ALLOC_FLAG_USE_ONCE);
    TEST_STR(ROCPROFILER_SCRATCH_ALLOC_FLAG_ALT);
}

TEST(enum_string, roctx_api_id)
{
    TEST_API_ID_STR(ROCPROFILER_MARKER_CORE_API_ID, roctxMarkA);
    TEST_API_ID_STR(ROCPROFILER_MARKER_CORE_API_ID, roctxGetThreadId);

    TEST_API_ID_STR(ROCPROFILER_MARKER_CONTROL_API_ID, roctxProfilerPause);
    TEST_API_ID_STR(ROCPROFILER_MARKER_CONTROL_API_ID, roctxProfilerResume);

    TEST_API_ID_STR(ROCPROFILER_MARKER_NAME_API_ID, roctxNameOsThread);
    TEST_API_ID_STR(ROCPROFILER_MARKER_NAME_API_ID, roctxNameHipStream);
}

TEST(enum_string, page_migration_triggers)
{
    TEST_STR(ROCPROFILER_PAGE_MIGRATION_TRIGGER_PREFETCH);
    TEST_STR(ROCPROFILER_PAGE_MIGRATION_TRIGGER_TTM_EVICTION);

    TEST_STR(ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_SVM);
    TEST_STR(ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_CRIU_RESTORE);

    TEST_STR(ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU_TRIGGER_MMU_NOTIFY);
    TEST_STR(ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU_TRIGGER_UNMAP_FROM_CPU);
}

TEST(enum_string, rccl_api_id)
{
    TEST_API_ID_STR(ROCPROFILER_RCCL_API_ID, ncclAllGather);
    TEST_API_ID_STR(ROCPROFILER_RCCL_API_ID, ncclSend);
    TEST_API_ID_STR(ROCPROFILER_RCCL_API_ID, ncclCommCount);
    TEST_API_ID_STR(ROCPROFILER_RCCL_API_ID, ncclCommDeregister);
}

TEST(enum_string, openmp_api_id)
{
    using namespace std::string_view_literals;
    using namespace rocprofiler::sdk;

    EXPECT_EQ(get_enum_label(ROCPROFILER_OMPT_ID_thread_begin),
              "ROCPROFILER_OMPT_ID_thread_begin"sv);
    EXPECT_EQ(get_enum_label(ROCPROFILER_OMPT_ID_implicit_task),
              "ROCPROFILER_OMPT_ID_implicit_task"sv);
    EXPECT_EQ(get_enum_label(ROCPROFILER_OMPT_ID_dependences), "ROCPROFILER_OMPT_ID_dependences"sv);
    EXPECT_EQ(get_enum_label(ROCPROFILER_OMPT_ID_mutex_acquire),
              "ROCPROFILER_OMPT_ID_mutex_acquire"sv);
    EXPECT_EQ(get_enum_label(ROCPROFILER_OMPT_ID_dispatch), "ROCPROFILER_OMPT_ID_dispatch"sv);
    EXPECT_EQ(get_enum_label(ROCPROFILER_OMPT_ID_callback_functions),
              "ROCPROFILER_OMPT_ID_callback_functions"sv);
}

TEST(enum_string, rocdecode_api_id)
{
    TEST_API_ID_STR(ROCPROFILER_ROCDECODE_API_ID, rocDecCreateVideoParser);
    TEST_API_ID_STR(ROCPROFILER_ROCDECODE_API_ID, rocDecGetDecoderCaps);
    TEST_API_ID_STR(ROCPROFILER_ROCDECODE_API_ID, rocDecGetErrorName);
}

TEST(enum_string, rocjpeg_api_id)
{
    TEST_API_ID_STR(ROCPROFILER_ROCJPEG_API_ID, rocJpegStreamCreate);
    TEST_API_ID_STR(ROCPROFILER_ROCJPEG_API_ID, rocJpegCreate);
    TEST_API_ID_STR(ROCPROFILER_ROCJPEG_API_ID, rocJpegGetErrorName);
}

TEST(enum_string, runtime_evaluation)
{
    // String representation of all enum values should have a prefix ROCPROFILER
    static constexpr std::string_view prefix{"ROCPROFILER"};

    for(const auto& i : get_random_values<rocprofiler_status_t>())
    {
        EXPECT_EQ(sdk::get_enum_label(i).substr(0, prefix.length()), prefix)
            << "rocprofiler_status_t : " << static_cast<int>(i);
    }

    for(const auto& i : get_random_values<rocprofiler_buffer_tracing_kind_t>())
    {
        EXPECT_EQ(sdk::get_enum_label(i).substr(0, prefix.length()), prefix)
            << "rocprofiler_buffer_tracing_kind_t : " << static_cast<int>(i);
    }

    for(const auto& i : get_random_values<rocprofiler_hsa_core_api_id_t>())
    {
        EXPECT_EQ(sdk::get_enum_label(i).substr(0, prefix.length()), prefix)
            << "rocprofiler_hsa_core_api_id_t : " << static_cast<int>(i);
    }

    for(const auto& i : get_random_values<rocprofiler_hip_runtime_api_id_t>())
    {
        EXPECT_EQ(sdk::get_enum_label(i).substr(0, prefix.length()), prefix)
            << "rocprofiler_hip_runtime_api_id_t : " << static_cast<int>(i);
    }
}

namespace enum_string_test
{
enum test_unsupported_enum
{
    TEST_ENUM_NONE,
    TEST_ENUM_LAST,
};

enum test_unsupported_enum_val
{
    TEST_ENUM_VALUE_NONE,
    TEST_ENUM_VALUE_V1,
    TEST_ENUM_VALUE_V2,
    TEST_ENUM_VALUE_V3,
    TEST_ENUM_VALUE_V4,
    TEST_ENUM_VALUE_LAST,
};
}  // namespace enum_string_test

namespace rocprofiler
{
namespace sdk
{
namespace details
{
using namespace enum_string_test;
ROCPROFILER_ENUM_INFO(test_unsupported_enum_val, TEST_ENUM_VALUE_NONE, TEST_ENUM_VALUE_LAST, false);
ROCPROFILER_ENUM_LABEL(TEST_ENUM_VALUE_V1);
ROCPROFILER_ENUM_LABEL(TEST_ENUM_VALUE_V3);
}  // namespace details
}  // namespace sdk
}  // namespace rocprofiler

TEST(enum_string, unsuported)
{
    using namespace rocprofiler::sdk;
    using namespace enum_string_test;
    using namespace std::string_view_literals;

    static_assert(!details::rocprofiler_enum_info<test_unsupported_enum>::supported);
    static_assert(!details::rocprofiler_enum_info<rocprofiler_att_control_flags_t>::supported);

    TEST_API_ID_STR(TEST_ENUM_VALUE, V1);
    TEST_API_ID_STR(TEST_ENUM_VALUE, V3);

    EXPECT_EQ(get_enum_label(TEST_ENUM_VALUE_V2), "unsupported_test_unsupported_enum_val"sv);
    EXPECT_EQ(get_enum_label(TEST_ENUM_VALUE_V4), "unsupported_test_unsupported_enum_val"sv);
}
