// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <stdint.h>

/// @file enum_definitions.h
/// @brief Shared enum definitions for C and C++ APIs

typedef enum rocprofvis_dm_track_category_t {
    kRocProfVisDmNotATrack = 0,
    kRocProfVisDmPmcTrack = 1,
    kRocProfVisDmRegionTrack = 2,
    kRocProfVisDmKernelDispatchTrack = 3,
    kRocProfVisDmSQTTTrack = 4,
    kRocProfVisDmNICTrack = 5,
    kRocProfVisDmMemoryAllocationTrack = 6,
    kRocProfVisDmMemoryCopyTrack = 7,
    kRocProfVisDmStreamTrack = 8,
    kRocProfVisDmRegionMainTrack = 9,
    kRocProfVisDmRegionSampleTrack = 10,
} rocprofvis_dm_track_category_t;

typedef enum rocprofvis_db_type_t {
    kAutodetect = 0,
    kRocpdSqlite = 1,
    kRocprofSqlite = 2
} rocprofvis_db_type_t;

typedef enum rocprofvis_dm_event_operation_t {
    kRocProfVisDmOperationNoOp = 0,
    kRocProfVisDmOperationLaunch = 1,
    kRocProfVisDmOperationDispatch = 2,
    kRocProfVisDmOperationMemoryAllocate = 3,
    kRocProfVisDmOperationMemoryCopy = 4,
    kRocProfVisDmOperationLaunchSample = 5,
    kRocProfVisDmNumOperation = 6
} rocprofvis_dm_event_operation_t;

typedef enum rocprofvis_dm_event_property_type_t {
    kRPVDMEventFlowTrace = 0,
    kRPVDMEventStackTrace = 1,
    kRPVDMEventExtData = 2,
    kRPVDMNumEventPropertyTypes = 3
} rocprofvis_dm_event_property_type_t;

typedef enum rocprofvis_dm_sort_order_t {
    kRPVDMSortOrderAsc = 0,
    kRPVDMSortOrderDesc = 1
} rocprofvis_dm_sort_order_t;

#ifdef __cplusplus
namespace rocstorage {

using track_category = rocprofvis_dm_track_category_t;
using database_type = rocprofvis_db_type_t;
using event_operation = rocprofvis_dm_event_operation_t;
using event_property_type = rocprofvis_dm_event_property_type_t;
using sort_order = rocprofvis_dm_sort_order_t;

} // namespace rocstorage
#endif
