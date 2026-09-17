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

#include <stddef.h>

#define ROCATTACH_API_TABLE_VERSION_MAJOR 1

ROCPROFILER_EXTERN_C_INIT

typedef int (*rocprofiler_attach_get_version_t)();
typedef int (*rocprofiler_attach_iterate_all_queues_t)(rocprof_attach_queue_iterator_t, void*);
typedef int (*rocprofiler_attach_set_write_interceptor_t)(hsa_queue_t*, write_interceptor_t, void*);
typedef int (*rocprofiler_attach_iterate_all_code_objects_t)(rocprof_attach_code_object_iterator_t,
                                                             void*);
typedef int (*rocprofiler_attach_add_code_object_cb_t)(rocprofiler_attach_code_object_cb_t, void*);
typedef int (*rocprofiler_attach_remove_code_object_cb_t)(rocprofiler_attach_code_object_cb_t);
typedef int (*rocprofiler_attach_add_queue_cb_t)(rocprofiler_attach_queue_cb_t, void*);
typedef int (*rocprofiler_attach_remove_queue_cb_t)(rocprofiler_attach_queue_cb_t);
typedef int (*rocprofiler_attach_lookup_memory_codeobj_data_t)(hsa_loaded_code_object_t,
                                                               const void**,
                                                               uint64_t*);
typedef int (*rocprofiler_attach_is_hsa_interception_active_t)();
typedef int (*rocprofiler_attach_initialize_hsa_interception_t)(struct HsaApiTable*);

struct RocAttachDispatchTable
{
    uint64_t                                         size;
    rocprofiler_attach_get_version_t                 rocprofiler_attach_get_version;
    rocprofiler_attach_iterate_all_queues_t          rocprofiler_attach_iterate_all_queues;
    rocprofiler_attach_set_write_interceptor_t       rocprofiler_attach_set_write_interceptor;
    rocprofiler_attach_iterate_all_code_objects_t    rocprofiler_attach_iterate_all_code_objects;
    rocprofiler_attach_add_code_object_cb_t          rocprofiler_attach_add_code_object_cb;
    rocprofiler_attach_remove_code_object_cb_t       rocprofiler_attach_remove_code_object_cb;
    rocprofiler_attach_add_queue_cb_t                rocprofiler_attach_add_queue_cb;
    rocprofiler_attach_remove_queue_cb_t             rocprofiler_attach_remove_queue_cb;
    rocprofiler_attach_lookup_memory_codeobj_data_t  rocprofiler_attach_lookup_memory_codeobj_data;
    rocprofiler_attach_is_hsa_interception_active_t  rocprofiler_attach_is_hsa_interception_active;
    rocprofiler_attach_initialize_hsa_interception_t rocprofiler_attach_initialize_hsa_interception;
};

static inline int
rocprofiler_attach_table_owns_hsa_interception(const struct RocAttachDispatchTable* table)
{
    const size_t required_size =
        offsetof(struct RocAttachDispatchTable, rocprofiler_attach_is_hsa_interception_active) +
        sizeof(table->rocprofiler_attach_is_hsa_interception_active);
    return (table != NULL && table->size >= required_size &&
            table->rocprofiler_attach_is_hsa_interception_active != NULL &&
            table->rocprofiler_attach_is_hsa_interception_active() != 0);
}

static inline int
rocprofiler_attach_table_initialize_hsa_interception(const struct RocAttachDispatchTable* table,
                                                     struct HsaApiTable* hsa_api_table)
{
    const size_t required_size =
        offsetof(struct RocAttachDispatchTable, rocprofiler_attach_initialize_hsa_interception) +
        sizeof(table->rocprofiler_attach_initialize_hsa_interception);
    if(table == NULL || table->size < required_size ||
       table->rocprofiler_attach_initialize_hsa_interception == NULL)
        return -1;
    return table->rocprofiler_attach_initialize_hsa_interception(hsa_api_table);
}

ROCPROFILER_EXTERN_C_FINI
