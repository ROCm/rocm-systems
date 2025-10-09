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

#include "table.hpp"

#include "lib/common/abi.hpp"
#include "lib/common/static_object.hpp"

namespace rocprofiler
{
namespace attach
{
ROCP_SDK_ENFORCE_ABI_VERSIONING(::RocAttachDispatchTable, ROCPROFILER_ATTACH_DISPATCH_TABLE_LEGNTH);

RocAttachDispatchTable*
get_dispatch_table()
{
    static auto*& dispatch_table =
        rocprofiler::common::static_object<RocAttachDispatchTable>::construct();
    return dispatch_table;
}

void
dispatch_table_init()
{
    auto* table = get_dispatch_table();

    table->size                                     = sizeof(RocAttachDispatchTable);
    table->rocprofiler_attach_get_version           = &rocprofiler_attach_get_version;
    table->rocprofiler_attach_iterate_all_queues    = &rocprofiler_attach_iterate_all_queues;
    table->rocprofiler_attach_set_write_interceptor = &rocprofiler_attach_set_write_interceptor;
    table->rocprofiler_attach_iterate_all_code_objects =
        &rocprofiler_attach_iterate_all_code_objects;
    table->rocprofiler_attach_notify_new_queue       = nullptr;
    table->rocprofiler_attach_notify_new_code_object = nullptr;
}
}  // namespace attach
}  // namespace rocprofiler
