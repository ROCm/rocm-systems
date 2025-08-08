// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#ifndef ROCPROFSYS_CATEGORIES_H_
#define ROCPROFSYS_CATEGORIES_H_

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C"
{
#endif

    /// @typedef rocprofsys_category_t
    /// @brief Identifier for categories
    ///
    typedef enum ROCPROFSYS_CATEGORIES
    {
        // Do not use first enum value
        ROCPROFSYS_CATEGORY_NONE = 0,
        // arrange these in the order most likely to
        // be used since they have to be iterated over
        ROCPROFSYS_CATEGORY_PYTHON,
        ROCPROFSYS_CATEGORY_USER,
        ROCPROFSYS_CATEGORY_HOST,
        ROCPROFSYS_CATEGORY_ROCM,
        ROCPROFSYS_CATEGORY_ROCM_HIP_API,
        ROCPROFSYS_CATEGORY_ROCM_HSA_API,
        ROCPROFSYS_CATEGORY_ROCM_KERNEL_DISPATCH,
        ROCPROFSYS_CATEGORY_ROCM_MEMORY_COPY,
        ROCPROFSYS_CATEGORY_ROCM_MEMORY_ALLOCATE,
        ROCPROFSYS_CATEGORY_ROCM_SCRATCH_MEMORY,
        ROCPROFSYS_CATEGORY_ROCM_HIP_STREAM,
        ROCPROFSYS_CATEGORY_ROCM_PAGE_MIGRATION,
        ROCPROFSYS_CATEGORY_ROCM_COUNTER_COLLECTION,
        ROCPROFSYS_CATEGORY_ROCM_MARKER_API,
        ROCPROFSYS_CATEGORY_ROCM_ROCDECODE_API,
        ROCPROFSYS_CATEGORY_ROCM_ROCJPEG_API,
        ROCPROFSYS_CATEGORY_ROCM_RCCL_API,
        ROCPROFSYS_CATEGORY_AMD_SMI,
        ROCPROFSYS_CATEGORY_AMD_SMI_BUSY_GFX,
        ROCPROFSYS_CATEGORY_AMD_SMI_BUSY_UMC,
        ROCPROFSYS_CATEGORY_AMD_SMI_BUSY_MM,
        ROCPROFSYS_CATEGORY_AMD_SMI_TEMP,
        ROCPROFSYS_CATEGORY_AMD_SMI_POWER,
        ROCPROFSYS_CATEGORY_AMD_SMI_MEMORY_USAGE,
        ROCPROFSYS_CATEGORY_AMD_SMI_VCN_ACTIVITY,
        ROCPROFSYS_CATEGORY_AMD_SMI_JPEG_ACTIVITY,
        ROCPROFSYS_CATEGORY_ROCM_RCCL,
        ROCPROFSYS_CATEGORY_SAMPLING,
        ROCPROFSYS_CATEGORY_PTHREAD,
        ROCPROFSYS_CATEGORY_KOKKOS,
        ROCPROFSYS_CATEGORY_MPI,
        ROCPROFSYS_CATEGORY_OMPT,
        ROCPROFSYS_CATEGORY_PROCESS_SAMPLING,
        ROCPROFSYS_CATEGORY_COMM_DATA,
        ROCPROFSYS_CATEGORY_CAUSAL,
        ROCPROFSYS_CATEGORY_CPU_FREQ,
        ROCPROFSYS_CATEGORY_PROCESS_PAGE,
        ROCPROFSYS_CATEGORY_PROCESS_VIRT,
        ROCPROFSYS_CATEGORY_PROCESS_PEAK,
        ROCPROFSYS_CATEGORY_PROCESS_CONTEXT_SWITCH,
        ROCPROFSYS_CATEGORY_PROCESS_PAGE_FAULT,
        ROCPROFSYS_CATEGORY_PROCESS_USER_MODE_TIME,
        ROCPROFSYS_CATEGORY_PROCESS_KERNEL_MODE_TIME,
        ROCPROFSYS_CATEGORY_THREAD_WALL_TIME,
        ROCPROFSYS_CATEGORY_THREAD_CPU_TIME,
        ROCPROFSYS_CATEGORY_THREAD_PAGE_FAULT,
        ROCPROFSYS_CATEGORY_THREAD_PEAK_MEMORY,
        ROCPROFSYS_CATEGORY_THREAD_CONTEXT_SWITCH,
        ROCPROFSYS_CATEGORY_THREAD_HARDWARE_COUNTER,
        ROCPROFSYS_CATEGORY_KERNEL_HARDWARE_COUNTER,
        ROCPROFSYS_CATEGORY_NUMA,
        ROCPROFSYS_CATEGORY_VAAPI,
        ROCPROFSYS_CATEGORY_TIMER_SAMPLING,
        ROCPROFSYS_CATEGORY_OVERFLOW_SAMPLING,
        ROCPROFSYS_CATEGORY_LAST
        // the value of below enum is used for iterating
        // over the enum in C++ templates. It MUST
        // be the last enumerated id
    } rocprofsys_category_t;

    /// @enum ROCPROFSYS_ANNOTATION_TYPE
    /// @brief Identifier for the data type of the annotation.
    /// if the data type is not a pointer, pass the address of
    /// data.
    /// @typedef ROCPROFSYS_ANNOTATION_TYPE rocprofsys_annotation_type_t
    typedef enum ROCPROFSYS_ANNOTATION_TYPE
    {
        // Do not use first enum value
        ROCPROFSYS_VALUE_NONE = 0,
        // arrange these in the order most likely to
        // be used since they have to be iterated over
        ROCPROFSYS_VALUE_CSTR    = 1,
        ROCPROFSYS_STRING        = ROCPROFSYS_VALUE_CSTR,
        ROCPROFSYS_VALUE_SIZE_T  = 2,
        ROCPROFSYS_SIZE_T        = ROCPROFSYS_VALUE_SIZE_T,
        ROCPROFSYS_VALUE_INT64   = 3,
        ROCPROFSYS_INT64         = ROCPROFSYS_VALUE_INT64,
        ROCPROFSYS_I64           = ROCPROFSYS_VALUE_INT64,
        ROCPROFSYS_VALUE_UINT64  = 4,
        ROCPROFSYS_UINT64        = ROCPROFSYS_VALUE_UINT64,
        ROCPROFSYS_U64           = ROCPROFSYS_VALUE_UINT64,
        ROCPROFSYS_VALUE_FLOAT64 = 5,
        ROCPROFSYS_FLOAT64       = ROCPROFSYS_VALUE_FLOAT64,
        ROCPROFSYS_FP64          = ROCPROFSYS_VALUE_FLOAT64,
        ROCPROFSYS_VALUE_VOID_P  = 6,
        ROCPROFSYS_VOID_P        = ROCPROFSYS_VALUE_VOID_P,
        ROCPROFSYS_PTR           = ROCPROFSYS_VALUE_VOID_P,
        ROCPROFSYS_VALUE_INT32   = 7,
        ROCPROFSYS_INT32         = ROCPROFSYS_VALUE_INT32,
        ROCPROFSYS_I32           = ROCPROFSYS_VALUE_INT32,
        ROCPROFSYS_VALUE_UINT32  = 8,
        ROCPROFSYS_UINT32        = ROCPROFSYS_VALUE_UINT32,
        ROCPROFSYS_U32           = ROCPROFSYS_VALUE_UINT32,
        ROCPROFSYS_VALUE_FLOAT32 = 9,
        ROCPROFSYS_FLOAT32       = ROCPROFSYS_VALUE_FLOAT32,
        ROCPROFSYS_FP32          = ROCPROFSYS_VALUE_FLOAT32,
        ROCPROFSYS_VALUE_INT16   = 10,
        ROCPROFSYS_INT16         = ROCPROFSYS_VALUE_INT16,
        ROCPROFSYS_I16           = ROCPROFSYS_VALUE_INT16,
        ROCPROFSYS_VALUE_UINT16  = 11,
        ROCPROFSYS_UINT16        = ROCPROFSYS_VALUE_UINT16,
        ROCPROFSYS_U16           = ROCPROFSYS_VALUE_UINT16,
        // the value of below enum is used for iterating
        // over the enum in C++ templates. It MUST
        // be the last enumerated id
        ROCPROFSYS_VALUE_LAST
    } rocprofsys_annotation_type_t;

    /// @struct rocprofsys_annotation
    /// @brief A struct containing annotation data to be included in the perfetto trace.
    ///
    /// @code{.cpp}
    /// #include <cstddef>
    /// #include <cstdint>
    ///
    /// double
    /// compute_residual(size_t n, double* data);
    ///
    /// double
    /// compute(size_t n, double* data, size_t nitr, double tolerance)
    /// {
    ///     rocprofsys_annotation_t _annotations[] = {
    ///         { "iteration", ROCPROFSYS_VALUE_SIZE_T, nullptr },
    ///         { "residual", ROCPROFSYS_VALUE_FLOAT64, nullptr },
    ///         { "data", ROCPROFSYS_VALUE_PTR, data },
    ///         { "size", ROCPROFSYS_VALUE_SIZE_T, &n },
    ///         { "tolerance", ROCPROFSYS_VALUE_FLOAT64, &tolerance },
    ///         nullptr
    ///     };
    ///
    ///     double residual = tolerance;
    ///     for(size_t i = 0; i < nitr; ++i)
    ///     {
    ///         rocprofsys_user_push_annotated_region("compute", &_annotations);
    ///
    ///         residual = compute_residual(n, data);
    ///
    ///         _annotations[0].value = &i;
    ///         _annotations[1].value = &residual;
    ///         rocprofsys_user_pop_annotated_region("compute", &_annotations);
    ///     }
    ///
    ///     return residual;
    /// }
    /// @endcode
    /// @typedef rocprofsys_annotation rocprofsys_annotation_t
    typedef struct rocprofsys_annotation
    {
        /// label for annotation
        const char* name;
        /// rocprofsys_annotation_type_t
        uintptr_t type;
        /// data to annotate
        void* value;
    } rocprofsys_annotation_t;

#if defined(__cplusplus)
}
#endif

#endif  // ROCPROFSYS_TYPES_H_
