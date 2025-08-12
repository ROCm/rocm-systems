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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "prestore.hpp"
#include "queue_registration.hpp"
#include "code_object_registration.hpp"

constexpr uint32_t ROCPROFILER_PRESTORE_TABLE_CURRENT_VERSION = 1;

using rocprofiler_prestore_get_version_t = decltype(::rocprofiler_prestore_get_version)*;
using rocprofiler_prestore_set_api_table_t = decltype(::rocprofiler_prestore_set_api_table)*;
using rocprofiler_prestore_export_all_queues_t = decltype(::rocprofiler_prestore_export_all_queues)*;
using rocprofiler_prestore_set_write_interceptor_t = decltype(::rocprofiler_prestore_set_write_interceptor)*;
using rocprofiler_prestore_export_all_code_objects_t = decltype(::rocprofiler_prestore_export_all_code_objects)*;

struct rocprofiler_prestore_dispatch_table_t {
    uint32_t version;
    rocprofiler_prestore_get_version_t rocprofiler_prestore_get_version;
    rocprofiler_prestore_set_api_table_t rocprofiler_prestore_set_api_table;
    rocprofiler_prestore_export_all_queues_t rocprofiler_prestore_export_all_queues;
    rocprofiler_prestore_set_write_interceptor_t rocprofiler_prestore_set_write_interceptor;
    rocprofiler_prestore_export_all_code_objects_t rocprofiler_prestore_export_all_code_objects;    
};
