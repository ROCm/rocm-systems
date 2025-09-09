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

#include "attach.h"
#include "code_object_registration.h"
#include "queue_registration.h"

#define ROCATTACH_API_TABLE_VERSION_MAJOR 0

ROCPROFILER_EXTERN_C_INIT

typedef int (*rocprofiler_attach_get_version_t)();
typedef int (*rocprofiler_attach_iterate_all_queues_t)(rocprof_attach_queue_iterator_t, void*);
typedef int (*rocprofiler_attach_set_write_interceptor_t)(hsa_queue_t*, write_interceptor_t, void*);
typedef int (*rocprofiler_attach_iterate_all_code_objects_t)(rocprof_attach_code_object_iterator_t,
                                                             void*);
typedef void (*rocprofiler_attach_notify_new_queue_t)(hsa_queue_t*, hsa_agent_t, void*);
typedef void (*rocprofiler_attach_notify_new_code_object_t)(hsa_executable_t, void*);

struct RocAttachDispatchTable
{
    uint64_t                                      size;
    rocprofiler_attach_get_version_t              rocprofiler_attach_get_version;
    rocprofiler_attach_iterate_all_queues_t       rocprofiler_attach_iterate_all_queues;
    rocprofiler_attach_set_write_interceptor_t    rocprofiler_attach_set_write_interceptor;
    rocprofiler_attach_iterate_all_code_objects_t rocprofiler_attach_iterate_all_code_objects;
    rocprofiler_attach_notify_new_queue_t         rocprofiler_attach_notify_new_queue;
    rocprofiler_attach_notify_new_code_object_t   rocprofiler_attach_notify_new_code_object;
};

ROCPROFILER_EXTERN_C_FINI
