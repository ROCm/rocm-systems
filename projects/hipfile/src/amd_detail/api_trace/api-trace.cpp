/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "api_trace/api-trace-internal.h"

#include <array>
#include <cstddef>

#if defined(HIPFILE_ROCPROFILER_REGISTER) && HIPFILE_ROCPROFILER_REGISTER > 0
#include <rocprofiler-register/rocprofiler-register.h>

#define HIPFILE_ROCP_REG_VERSION                                                                             \
    ROCPROFILER_REGISTER_COMPUTE_VERSION_3(HIPFILE_ROCP_REG_VERSION_MAJOR, HIPFILE_ROCP_REG_VERSION_MINOR,   \
                                           HIPFILE_ROCP_REG_VERSION_PATCH)

ROCPROFILER_REGISTER_DEFINE_IMPORT(hipFile, HIPFILE_ROCP_REG_VERSION)
#elif !defined(HIPFILE_ROCPROFILER_REGISTER)
#define HIPFILE_ROCPROFILER_REGISTER 0
#endif

namespace hipFile {
namespace {
    void UpdateDispatchTable(hipFileDispatchTable *ptr_dispatch_table)
    {
        ptr_dispatch_table->size                              = sizeof(hipFileDispatchTable);
        ptr_dispatch_table->pfn_hipfile_get_op_error_string   = hipFile::hipFileGetOpErrorString;
        ptr_dispatch_table->pfn_hipfile_handle_register       = hipFile::hipFileHandleRegister;
        ptr_dispatch_table->pfn_hipfile_handle_deregister     = hipFile::hipFileHandleDeregister;
        ptr_dispatch_table->pfn_hipfile_buf_register          = hipFile::hipFileBufRegister;
        ptr_dispatch_table->pfn_hipfile_buf_deregister        = hipFile::hipFileBufDeregister;
        ptr_dispatch_table->pfn_hipfile_read                  = hipFile::hipFileRead;
        ptr_dispatch_table->pfn_hipfile_write                 = hipFile::hipFileWrite;
        ptr_dispatch_table->pfn_hipfile_driver_open           = hipFile::hipFileDriverOpen;
        ptr_dispatch_table->pfn_hipfile_driver_close          = hipFile::hipFileDriverClose;
        ptr_dispatch_table->pfn_hipfile_use_count             = hipFile::hipFileUseCount;
        ptr_dispatch_table->pfn_hipfile_driver_get_properties = hipFile::hipFileDriverGetProperties;
        ptr_dispatch_table->pfn_hipfile_driver_set_poll_mode  = hipFile::hipFileDriverSetPollMode;
        ptr_dispatch_table->pfn_hipfile_driver_set_max_direct_io_size =
            hipFile::hipFileDriverSetMaxDirectIOSize;
        ptr_dispatch_table->pfn_hipfile_driver_set_max_cache_size = hipFile::hipFileDriverSetMaxCacheSize;
        ptr_dispatch_table->pfn_hipfile_driver_set_max_pinned_mem_size =
            hipFile::hipFileDriverSetMaxPinnedMemSize;
        ptr_dispatch_table->pfn_hipfile_batch_io_set_up      = hipFile::hipFileBatchIOSetUp;
        ptr_dispatch_table->pfn_hipfile_batch_io_submit      = hipFile::hipFileBatchIOSubmit;
        ptr_dispatch_table->pfn_hipfile_batch_io_get_status  = hipFile::hipFileBatchIOGetStatus;
        ptr_dispatch_table->pfn_hipfile_batch_io_cancel      = hipFile::hipFileBatchIOCancel;
        ptr_dispatch_table->pfn_hipfile_batch_io_destroy     = hipFile::hipFileBatchIODestroy;
        ptr_dispatch_table->pfn_hipfile_read_async           = hipFile::hipFileReadAsync;
        ptr_dispatch_table->pfn_hipfile_write_async          = hipFile::hipFileWriteAsync;
        ptr_dispatch_table->pfn_hipfile_stream_register      = hipFile::hipFileStreamRegister;
        ptr_dispatch_table->pfn_hipfile_stream_deregister    = hipFile::hipFileStreamDeregister;
        ptr_dispatch_table->pfn_hipfile_get_version          = hipFile::hipFileGetVersion;
        ptr_dispatch_table->pfn_hipfile_get_parameter_size_t = hipFile::hipFileGetParameterSizeT;
        ptr_dispatch_table->pfn_hipfile_get_parameter_bool   = hipFile::hipFileGetParameterBool;
        ptr_dispatch_table->pfn_hipfile_get_parameter_string = hipFile::hipFileGetParameterString;
        ptr_dispatch_table->pfn_hipfile_set_parameter_size_t = hipFile::hipFileSetParameterSizeT;
        ptr_dispatch_table->pfn_hipfile_set_parameter_bool   = hipFile::hipFileSetParameterBool;
        ptr_dispatch_table->pfn_hipfile_set_parameter_string = hipFile::hipFileSetParameterString;
    }

#if HIPFILE_ROCPROFILER_REGISTER > 0
    template <typename Tp> struct dispatch_table_info;

#define HIPFILE_DEFINE_DISPATCH_TABLE_INFO(TYPE, NAME)                                                       \
    template <> struct dispatch_table_info<TYPE> {                                                           \
        static constexpr auto name        = #NAME;                                                           \
        static constexpr auto version     = HIPFILE_ROCP_REG_VERSION;                                        \
        static constexpr auto import_func = &ROCPROFILER_REGISTER_IMPORT_FUNC(NAME);                         \
    };

    HIPFILE_DEFINE_DISPATCH_TABLE_INFO(hipFileDispatchTable, hipFile)
#endif

    template <typename Tp> void ToolInit(Tp *table)
    {
#if HIPFILE_ROCPROFILER_REGISTER > 0
        auto table_array = std::array<void *, 1>{static_cast<void *>(table)};
        auto lib_id      = rocprofiler_register_library_indentifier_t{};
        rocprofiler_register_library_api_table(
            dispatch_table_info<Tp>::name, dispatch_table_info<Tp>::import_func,
            dispatch_table_info<Tp>::version, table_array.data(), table_array.size(), &lib_id);
#else
        (void)table;
#endif
    }

    template <typename Tp> Tp &GetDispatchTableImpl()
    {
        static auto dispatch_table = Tp{};
        // Update all function pointers to reference the runtime implementation functions of hipFile
        UpdateDispatchTable(&dispatch_table);
        // The profiler registration process may encapsulate the function pointers
        ToolInit(&dispatch_table);
        return dispatch_table;
    }
} // namespace

const hipFileDispatchTable *
GetHipFileDispatchTable(void)
{
    static auto *hipfile_dispatch_table = &GetDispatchTableImpl<hipFileDispatchTable>();
    return hipfile_dispatch_table;
}
} // namespace hipFile

#if !defined(_WIN32)
constexpr auto
ComputeTableOffset(size_t num_funcs)
{
    return (num_funcs * sizeof(void *)) + sizeof(size_t);
}

// The `HIPFILE_ENFORCE_ABI_VERSIONING` macro will trigger a compiler error if the size of the hipFile
// dispatch API table changes, which is most likely due to the addition of a new dispatch table entry. This
// serves as a reminder for developers to update the table versioning value before changing the value in
// `HIPFILE_ENFORCE_ABI_VERSIONING`, ensuring that this static assertion passes.
//
// The `HIPFILE_ENFORCE_ABI` macro will also trigger a compiler error if the order of the members in the
// hipFile dispatch API table is altered. Therefore, it is essential to avoid reordering member variables.
//
// Please be aware that `rocprofiler` performs strict compile-time checks to ensure that these versioning
// values are correctly updated. Commenting out this check or merely updating the size field in
// `HIPFILE_ENFORCE_ABI_VERSIONING` will cause the `rocprofiler` to fail during the build process.
#define HIPFILE_ENFORCE_ABI_VERSIONING(TABLE, NUM)                                                           \
    static_assert(sizeof(TABLE) == ComputeTableOffset(NUM),                                                  \
                  "The size of the API table structure has been updated. Please modify the "                 \
                  "STEP_VERSION number (or, in rare cases, the MAJOR_VERSION number) "                       \
                  "in <hipfile/include/hipfile-api-trace.h> for the failing API "                            \
                  "structure before changing the SIZE field passed to HIPFILE_DEFINE_DISPATCH_TABLE_INFO.");

#define HIPFILE_ENFORCE_ABI(TABLE, ENTRY, NUM)                                                               \
    static_assert(offsetof(TABLE, ENTRY) == ComputeTableOffset(NUM),                                         \
                  "ABI broke for " #TABLE "." #ENTRY                                                         \
                  ", only add new function pointers at the end of the struct and do not rearrange them.");

// These ensure that function pointers are not re-ordered
// HIPFILE_RUNTIME_API_TABLE_STEP_VERSION == 0
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_get_op_error_string, 0)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_handle_register, 1)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_handle_deregister, 2)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_buf_register, 3)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_buf_deregister, 4)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_read, 5)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_write, 6)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_driver_open, 7)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_driver_close, 8)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_use_count, 9)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_driver_get_properties, 10)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_driver_set_poll_mode, 11)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_driver_set_max_direct_io_size, 12)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_driver_set_max_cache_size, 13)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_driver_set_max_pinned_mem_size, 14)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_batch_io_set_up, 15)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_batch_io_submit, 16)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_batch_io_get_status, 17)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_batch_io_cancel, 18)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_batch_io_destroy, 19)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_read_async, 20)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_write_async, 21)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_stream_register, 22)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_stream_deregister, 23)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_get_version, 24)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_get_parameter_size_t, 25)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_get_parameter_bool, 26)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_get_parameter_string, 27)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_set_parameter_size_t, 28)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_set_parameter_bool, 29)
HIPFILE_ENFORCE_ABI(hipFileDispatchTable, pfn_hipfile_set_parameter_string, 30)
// HIPFILE_RUNTIME_API_TABLE_STEP_VERSION == 1

// If HIPFILE_ENFORCE_ABI entries are added for each new function pointer in the table,
// the number below will be one greater than the number in the last HIPFILE_ENFORCE_ABI line. For example:
//  HIPFILE_ENFORCE_ABI(<table>, <functor>, 30)
//  HIPFILE_ENFORCE_ABI_VERSIONING(<table>, 31) <- 30 + 1 = 31
HIPFILE_ENFORCE_ABI_VERSIONING(hipFileDispatchTable, 31)

static_assert(HIPFILE_RUNTIME_API_TABLE_MAJOR_VERSION == 0 && HIPFILE_RUNTIME_API_TABLE_STEP_VERSION == 0,
              "If you encounter this error, add the new HIPFILE_ENFORCE_ABI(...) code for the updated "
              "function pointers, "
              "and then modify this check to ensure it evaluates to true.");
#endif // !defined(_WIN32)
