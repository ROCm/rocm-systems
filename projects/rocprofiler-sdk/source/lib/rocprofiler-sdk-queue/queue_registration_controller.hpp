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

#include "queue_registration.hpp"

#include <unordered_map>

extern "C" {
// this is the "hidden" function that rocprofiler-register invokes to pass
// the HSA API table to rocprofiler
int
rocprofiler_queue_set_api_table(
    const char* name,
    uint64_t    lib_version,
    uint64_t    lib_instance,
    void**      tables,
    uint64_t    num_tables) ROCPROFILER_PUBLIC_API;
}

namespace rocprofiler {
namespace hsa{

// Tracks and manages HSA queues to be profiled later when rocprof is loaded
class QueueRegistrationController
{
public:
    QueueRegistrationController()  = default;
    ~QueueRegistrationController() = default;
    // Initializes the QueueRegistrationController. This must be delayed until
    // HSA has been inited.
    void init(CoreApiTable& core_table, AmdExtTable& ext_table);

    // Called to add a queue that was created by the user program
    void add_queue(hsa_queue_t*, std::shared_ptr<QueueRegistration>);
    void destroy_queue(hsa_queue_t*);

    const std::unordered_map<hsa_queue_t*, std::shared_ptr<QueueRegistration>> get_all_registrations() ROCPROFILER_PUBLIC_API;

    const CoreApiTable& get_core_table() const { return _core_table; }
    const AmdExtTable&  get_ext_table() const { return _ext_table; }

private:
    CoreApiTable _core_table = {};
    AmdExtTable  _ext_table  = {};
    std::unordered_map<hsa_queue_t*, std::shared_ptr<QueueRegistration>> _queues;
};

QueueRegistrationController*
get_queue_registration_controller() ROCPROFILER_PUBLIC_API;

void
queue_registration_controller_init(HsaApiTable* table);

}}