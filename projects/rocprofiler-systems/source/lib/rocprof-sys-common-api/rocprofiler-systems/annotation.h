// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCPROFSYS_ANNOTATION_H_
#define ROCPROFSYS_ANNOTATION_H_

#include <stdint.h>

#if defined(__cplusplus)
extern "C"
{
#endif

    /// @enum ROCPROFSYS_ANNOTATION_TYPE
    /// @brief Identifier for the data type of the annotation.
    /// if the data type is not a pointer, pass the address of
    /// data.
    /// @typedef ROCPROFSYS_ANNOTATION_TYPE rocprofsys_annotation_type_t
    // C-style enum/typedef for C ABI compatibility
    // NOLINTNEXTLINE(cppcoreguidelines-use-enum-class,modernize-use-using)
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
    ///     size_t iteration = 0;
    ///     double residual  = tolerance;
    ///     rocprofsys_annotation_t _annotations[] = {
    ///         { "iteration", ROCPROFSYS_VALUE_SIZE_T, &iteration },
    ///         { "residual", ROCPROFSYS_VALUE_FLOAT64, &residual },
    ///     };
    ///
    ///     for(iteration = 0; iteration < nitr; ++iteration)
    ///     {
    ///         rocprofsys_push_category_region(ROCPROFSYS_CATEGORY_USER, "compute",
    ///                                         _annotations, 2);
    ///
    ///         residual = compute_residual(n, data);
    ///
    ///         rocprofsys_pop_category_region(ROCPROFSYS_CATEGORY_USER, "compute",
    ///                                        _annotations, 2);
    ///     }
    ///
    ///     return residual;
    /// }
    /// @endcode
    /// @typedef rocprofsys_annotation rocprofsys_annotation_t
    // NOLINTNEXTLINE(modernize-use-using) -- C-style typedef for C ABI compatibility
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

#endif  // ROCPROFSYS_ANNOTATION_H_
