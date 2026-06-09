/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_HIP_API_SYMBOLS_H
#define HIP_INCLUDE_HIP_HIP_API_SYMBOLS_H

#include <hip/hip_version.h>

#define HIP_ABI_MODE_PUBLIC 0
#define HIP_ABI_MODE_BACKEND 1

#ifndef HIP_ABI_MODE
#define HIP_ABI_MODE HIP_ABI_MODE_PUBLIC
#endif

#ifndef HIP_API_VERSION
#define HIP_API_VERSION HIP_VERSION_MAJOR
#endif

#define HIP_DETAIL_CONCAT2_IMPL(a, b) a##b
#define HIP_DETAIL_CONCAT2(a, b) HIP_DETAIL_CONCAT2_IMPL(a, b)
#define HIP_DETAIL_CONCAT3(a, b, c) HIP_DETAIL_CONCAT2(HIP_DETAIL_CONCAT2(a, b), c)
#define HIP_DETAIL_CONCAT4(a, b, c, d) HIP_DETAIL_CONCAT2(HIP_DETAIL_CONCAT3(a, b, c), d)

#if HIP_ABI_MODE == HIP_ABI_MODE_BACKEND
#define HIP_API_SYMBOL(name) HIP_DETAIL_CONCAT3(hipBackendV, HIP_API_VERSION, name)
#define HIP_PRIVATE_SYMBOL(name) HIP_DETAIL_CONCAT4(hipBackendV, HIP_API_VERSION, Private, name)
#define HIP_COMPILER_API_SYMBOL(name) HIP_DETAIL_CONCAT4(hipBackendV, HIP_API_VERSION, Compiler, name)
#else
#define HIP_API_SYMBOL(name) HIP_DETAIL_CONCAT2(hip, name)
#define HIP_PRIVATE_SYMBOL(name) HIP_DETAIL_CONCAT2(__hip, name)
#define HIP_COMPILER_API_SYMBOL(name) HIP_DETAIL_CONCAT2(__hip, name)
#endif

#endif
