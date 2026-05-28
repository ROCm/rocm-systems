/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include "../../api/amd_detail/rocjpeg_api_trace.h"

#if defined(ROCJPEG_ROCPROFILER_REGISTER) && ROCJPEG_ROCPROFILER_REGISTER > 0
#include <rocprofiler-register/rocprofiler-register.h>

#define ROCJPEG_ROCP_REG_VERSION \
  ROCPROFILER_REGISTER_COMPUTE_VERSION_3(ROCJPEG_ROCP_REG_VERSION_MAJOR, ROCJPEG_ROCP_REG_VERSION_MINOR, \
                                         ROCJPEG_ROCP_REG_VERSION_PATCH)

ROCPROFILER_REGISTER_DEFINE_IMPORT(rocjpeg, ROCJPEG_ROCP_REG_VERSION)
#elif !defined(ROCJPEG_ROCPROFILER_REGISTER)
#define ROCJPEG_ROCPROFILER_REGISTER 0
#endif

namespace rocjpeg {
RocJpegStatus ROCJPEGAPI rocJpegStreamCreate(RocJpegStreamHandle *jpeg_stream_handle);
RocJpegStatus ROCJPEGAPI rocJpegStreamParse(const unsigned char *data, size_t length, RocJpegStreamHandle jpeg_stream_handle);
RocJpegStatus ROCJPEGAPI rocJpegStreamDestroy(RocJpegStreamHandle jpeg_stream_handle);
RocJpegStatus ROCJPEGAPI rocJpegCreate(RocJpegBackend backend, int device_id, RocJpegHandle *handle);
RocJpegStatus ROCJPEGAPI rocJpegDestroy(RocJpegHandle handle);
RocJpegStatus ROCJPEGAPI rocJpegGetImageInfo(RocJpegHandle handle, RocJpegStreamHandle jpeg_stream_handle, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights);
RocJpegStatus ROCJPEGAPI rocJpegDecode(RocJpegHandle handle, RocJpegStreamHandle jpeg_stream_handle, const RocJpegDecodeParams *decode_params, RocJpegImage *destination);
RocJpegStatus ROCJPEGAPI rocJpegDecodeBatched(RocJpegHandle handle, RocJpegStreamHandle *jpeg_stream_handles, int batch_size, const RocJpegDecodeParams *decode_params, RocJpegImage *destinations);
const char* ROCJPEGAPI rocJpegGetErrorName(RocJpegStatus rocjpeg_status);
}

namespace rocjpeg {
namespace {
void UpdateDispatchTable(RocJpegDispatchTable* ptr_dispatch_table) {
    ptr_dispatch_table->size = sizeof(RocJpegDispatchTable);
    ptr_dispatch_table->pfn_rocjpeg_stream_create = rocjpeg::rocJpegStreamCreate;
    ptr_dispatch_table->pfn_rocjpeg_stream_parse = rocjpeg::rocJpegStreamParse;
    ptr_dispatch_table->pfn_rocjpeg_stream_destroy = rocjpeg::rocJpegStreamDestroy;
    ptr_dispatch_table->pfn_rocjpeg_create = rocjpeg::rocJpegCreate;
    ptr_dispatch_table->pfn_rocjpeg_destroy = rocjpeg::rocJpegDestroy;
    ptr_dispatch_table->pfn_rocjpeg_get_image_info = rocjpeg::rocJpegGetImageInfo;
    ptr_dispatch_table->pfn_rocjpeg_decode = rocjpeg::rocJpegDecode;
    ptr_dispatch_table->pfn_rocjpeg_decode_batched = rocjpeg::rocJpegDecodeBatched;
    ptr_dispatch_table->pfn_rocjpeg_get_error_name = rocjpeg::rocJpegGetErrorName;
}

#if ROCJPEG_ROCPROFILER_REGISTER > 0
template <typename Tp> struct dispatch_table_info;

#define ROCJPEG_DEFINE_DISPATCH_TABLE_INFO(TYPE, NAME) \
template <> struct dispatch_table_info<TYPE> { \
    static constexpr auto name = #NAME; \
    static constexpr auto version = ROCJPEG_ROCP_REG_VERSION; \
    static constexpr auto import_func = &ROCPROFILER_REGISTER_IMPORT_FUNC(NAME); \
};

ROCJPEG_DEFINE_DISPATCH_TABLE_INFO(RocJpegDispatchTable, rocjpeg)
#endif

template <typename Tp> void ToolInit(Tp* table) {
#if ROCJPEG_ROCPROFILER_REGISTER > 0
    auto table_array = std::array<void*, 1>{static_cast<void*>(table)};
    auto lib_id = rocprofiler_register_library_indentifier_t{};
    rocprofiler_register_library_api_table(
        dispatch_table_info<Tp>::name, dispatch_table_info<Tp>::import_func,
        dispatch_table_info<Tp>::version, table_array.data(), table_array.size(), &lib_id);
#else
    (void)table;
#endif
}

template <typename Tp> Tp& GetDispatchTableImpl() {
    static auto dispatch_table = Tp{};
    // Update all function pointers to reference the runtime implementation functions of rocJPEG.
    UpdateDispatchTable(&dispatch_table);
    // The profiler registration process may encapsulate the function pointers.
    ToolInit(&dispatch_table);
    return dispatch_table;
}
} //namespace

const RocJpegDispatchTable* GetRocJpegDispatchTable() {
    static auto* rocjpeg_dispatch_table = &GetDispatchTableImpl<RocJpegDispatchTable>();
    return rocjpeg_dispatch_table;
}
} //namespace rocjpeg

#if !defined(_WIN32)
constexpr auto ComputeTableOffset(size_t num_funcs) {
    return (num_funcs * sizeof(void*)) + sizeof(size_t);
}

// The `ROCJPEG_ENFORCE_ABI_VERSIONING` macro will trigger a compiler error if the size of the rocJPEG dispatch API table changes,
// which is most likely due to the addition of a new dispatch table entry. This serves as a reminder for developers to update the table
// versioning value before changing the value in `ROCJPEG_ENFORCE_ABI_VERSIONING`, ensuring that this static assertion passes.
//
// The `ROCJPEG_ENFORCE_ABI` macro will also trigger a compiler error if the order of the members in the rocJPEG dispatch API table
// is altered. Therefore, it is essential to avoid reordering member variables.
//
// Please be aware that `rocprofiler` performs strict compile-time checks to ensure that these versioning values are correctly updated.
// Commenting out this check or merely updating the size field in `ROCJPEG_ENFORCE_ABI_VERSIONING` will cause the `rocprofiler` to fail
// during the build process.
#define ROCJPEG_ENFORCE_ABI_VERSIONING(TABLE, NUM) \
  static_assert( \
      sizeof(TABLE) == ComputeTableOffset(NUM), \
      "The size of the API table structure has been updated. Please modify the " \
      "STEP_VERSION number (or, in rare cases, the MAJOR_VERSION number) " \
      "in <rocJPEG/api/amd_detail/rocjpeg_api_trace.h> for the failing API " \
      "structure before changing the SIZE field passed to ROCJPEG_DEFINE_DISPATCH_TABLE_INFO.");

#define ROCJPEG_ENFORCE_ABI(TABLE, ENTRY, NUM) \
  static_assert(offsetof(TABLE, ENTRY) == ComputeTableOffset(NUM), \
                "ABI broke for " #TABLE "." #ENTRY \
                ", only add new function pointers at the end of the struct and do not rearrange them.");

// These ensure that function pointers are not re-ordered
// ROCJPEG_RUNTIME_API_TABLE_STEP_VERSION == 0
ROCJPEG_ENFORCE_ABI(RocJpegDispatchTable, pfn_rocjpeg_stream_create, 0)
ROCJPEG_ENFORCE_ABI(RocJpegDispatchTable, pfn_rocjpeg_stream_parse, 1)
ROCJPEG_ENFORCE_ABI(RocJpegDispatchTable, pfn_rocjpeg_stream_destroy, 2)
ROCJPEG_ENFORCE_ABI(RocJpegDispatchTable, pfn_rocjpeg_create, 3)
ROCJPEG_ENFORCE_ABI(RocJpegDispatchTable, pfn_rocjpeg_destroy, 4)
ROCJPEG_ENFORCE_ABI(RocJpegDispatchTable, pfn_rocjpeg_get_image_info, 5)
ROCJPEG_ENFORCE_ABI(RocJpegDispatchTable, pfn_rocjpeg_decode, 6)
ROCJPEG_ENFORCE_ABI(RocJpegDispatchTable, pfn_rocjpeg_decode_batched, 7)
ROCJPEG_ENFORCE_ABI(RocJpegDispatchTable, pfn_rocjpeg_get_error_name, 8)

// If ROCJPEG_ENFORCE_ABI entries are added for each new function pointer in the table,
// the number below will be one greater than the number in the last ROCJPEG_ENFORCE_ABI line. For example:
//  ROCJPEG_ENFORCE_ABI(<table>, <functor>, 8)
//  ROCJPEG_ENFORCE_ABI_VERSIONING(<table>, 9) <- 8 + 1 = 9
ROCJPEG_ENFORCE_ABI_VERSIONING(RocJpegDispatchTable, 9)

static_assert(ROCJPEG_RUNTIME_API_TABLE_MAJOR_VERSION == 0 && ROCJPEG_RUNTIME_API_TABLE_STEP_VERSION == 0,
              "If you encounter this error, add the new ROCJPEG_ENFORCE_ABI(...) code for the updated function pointers, "
              "and then modify this check to ensure it evaluates to true.");
#endif