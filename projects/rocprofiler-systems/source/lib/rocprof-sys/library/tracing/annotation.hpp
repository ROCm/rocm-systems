// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocprofiler-systems/annotation.h"  // in rocprof-sys-common-api

#include <cstddef>
#include <cstdint>

namespace rocprofsys::tracing
{
template <size_t Idx>
struct annotation_value_type;

template <size_t Idx>
using annotation_value_type_t = annotation_value_type<Idx>::type;

#define ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ENUM, TYPE)                                    \
    template <>                                                                          \
    struct annotation_value_type<ENUM>                                                   \
    {                                                                                    \
        using type = TYPE;                                                               \
    };

ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_CSTR, const char*)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_SIZE_T, size_t)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_INT16, std::int16_t)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_INT32, std::int32_t)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_INT64, std::int64_t)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_UINT16, std::uint16_t)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_UINT32, std::uint32_t)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_UINT64, std::uint64_t)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_FLOAT32, float)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_FLOAT64, double)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_VOID_P, void*)

#undef ROCPROFSYS_DEFINE_ANNOTATION_TYPE
}  // namespace rocprofsys::tracing
