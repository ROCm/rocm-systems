/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// Tests for the API dispatch-table layer in src/amd_detail/api_trace.
//
// These tests deliberately do NOT pull in api-trace-internal.h. That header
// declares the runtime implementations inside `namespace hipFile`, which would
// collide with the extern "C" declarations from <hipfile.h> when both become
// visible through `using namespace hipFile;`. The dispatch table contents are
// reachable via the public header alone.

#include "hipfile.h"
#include "hipfile-api-trace.h"
#include "hipfile-test.h"
#include "hipfile-warnings.h"

#include <cstddef>
#include <gtest/gtest.h>

// The accessor lives in `namespace hipFile` and is normally consumed via
// api-trace-internal.h. Redeclare it locally so this test file stays clear
// of that header.
namespace hipFile {
const hipFileDispatchTable *GetHipFileDispatchTable(void);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

// The accessor must hand back a stable, non-null singleton.
TEST(ApiTrace, GetHipFileDispatchTable_returns_stable_singleton)
{
    const hipFileDispatchTable *first  = hipFile::GetHipFileDispatchTable();
    const hipFileDispatchTable *second = hipFile::GetHipFileDispatchTable();

    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first, second);
}

// The size field must reflect the actual struct size. rocprofiler-register
// consumers use this to detect ABI drift at runtime.
TEST(ApiTrace, DispatchTable_size_field_matches_struct_size)
{
    const hipFileDispatchTable *table = hipFile::GetHipFileDispatchTable();
    ASSERT_NE(table, nullptr);
    EXPECT_EQ(table->size, sizeof(hipFileDispatchTable));
}

// Every function pointer in the table must be populated. A missing assignment
// in UpdateDispatchTable would crash at call time; this catches it at
// dispatch-table construction time.
TEST(ApiTrace, DispatchTable_all_function_pointers_populated)
{
    const hipFileDispatchTable *t = hipFile::GetHipFileDispatchTable();
    ASSERT_NE(t, nullptr);

    EXPECT_NE(t->pfn_hipfile_get_op_error_string, nullptr);
    EXPECT_NE(t->pfn_hipfile_handle_register, nullptr);
    EXPECT_NE(t->pfn_hipfile_handle_deregister, nullptr);
    EXPECT_NE(t->pfn_hipfile_buf_register, nullptr);
    EXPECT_NE(t->pfn_hipfile_buf_deregister, nullptr);
    EXPECT_NE(t->pfn_hipfile_read, nullptr);
    EXPECT_NE(t->pfn_hipfile_write, nullptr);
    EXPECT_NE(t->pfn_hipfile_driver_open, nullptr);
    EXPECT_NE(t->pfn_hipfile_driver_close, nullptr);
    EXPECT_NE(t->pfn_hipfile_use_count, nullptr);
    EXPECT_NE(t->pfn_hipfile_driver_get_properties, nullptr);
    EXPECT_NE(t->pfn_hipfile_driver_set_poll_mode, nullptr);
    EXPECT_NE(t->pfn_hipfile_driver_set_max_direct_io_size, nullptr);
    EXPECT_NE(t->pfn_hipfile_driver_set_max_cache_size, nullptr);
    EXPECT_NE(t->pfn_hipfile_driver_set_max_pinned_mem_size, nullptr);
    EXPECT_NE(t->pfn_hipfile_batch_io_set_up, nullptr);
    EXPECT_NE(t->pfn_hipfile_batch_io_submit, nullptr);
    EXPECT_NE(t->pfn_hipfile_batch_io_get_status, nullptr);
    EXPECT_NE(t->pfn_hipfile_batch_io_cancel, nullptr);
    EXPECT_NE(t->pfn_hipfile_batch_io_destroy, nullptr);
    EXPECT_NE(t->pfn_hipfile_read_async, nullptr);
    EXPECT_NE(t->pfn_hipfile_write_async, nullptr);
    EXPECT_NE(t->pfn_hipfile_stream_register, nullptr);
    EXPECT_NE(t->pfn_hipfile_stream_deregister, nullptr);
    EXPECT_NE(t->pfn_hipfile_get_version, nullptr);
    EXPECT_NE(t->pfn_hipfile_get_parameter_size_t, nullptr);
    EXPECT_NE(t->pfn_hipfile_get_parameter_bool, nullptr);
    EXPECT_NE(t->pfn_hipfile_get_parameter_string, nullptr);
    EXPECT_NE(t->pfn_hipfile_set_parameter_size_t, nullptr);
    EXPECT_NE(t->pfn_hipfile_set_parameter_bool, nullptr);
    EXPECT_NE(t->pfn_hipfile_set_parameter_string, nullptr);
}

// Smoke test the call-through. With no profiler attached, going through the
// dispatch table must produce the same observable result as calling the
// extern "C" API directly. Using hipFileGetOpErrorString because it's a pure
// function with no global state.
TEST(ApiTrace, DispatchTable_call_through_matches_public_api)
{
    const hipFileDispatchTable *t = hipFile::GetHipFileDispatchTable();
    ASSERT_NE(t, nullptr);
    ASSERT_NE(t->pfn_hipfile_get_op_error_string, nullptr);

    const char *via_table = t->pfn_hipfile_get_op_error_string(hipFileSuccess);
    const char *via_api   = hipFileGetOpErrorString(hipFileSuccess);

    ASSERT_NE(via_table, nullptr);
    ASSERT_NE(via_api, nullptr);
    EXPECT_STREQ(via_table, via_api);
}

// Version expected to be a stable, non-failing call regardless of driver
// state. Exercises the dispatch path for a hipFileError_t-returning function.
TEST(ApiTrace, DispatchTable_call_through_get_version)
{
    const hipFileDispatchTable *t = hipFile::GetHipFileDispatchTable();
    ASSERT_NE(t, nullptr);
    ASSERT_NE(t->pfn_hipfile_get_version, nullptr);

    unsigned       maj_t = 0, min_t = 0, pat_t = 0;
    unsigned       maj_a = 0, min_a = 0, pat_a = 0;
    hipFileError_t err_t = t->pfn_hipfile_get_version(&maj_t, &min_t, &pat_t);
    hipFileError_t err_a = hipFileGetVersion(&maj_a, &min_a, &pat_a);

    EXPECT_EQ(err_t.err, err_a.err);
    EXPECT_EQ(err_t.hip_drv_err, err_a.hip_drv_err);
    EXPECT_EQ(maj_t, maj_a);
    EXPECT_EQ(min_t, min_a);
    EXPECT_EQ(pat_t, pat_a);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
