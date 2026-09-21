// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Windows stand-ins for the AQLProfile entry points still reached from code that does
// build on Windows (hsa/aql_packet.cpp and counters/id_decode.cpp). AQLProfile emits
// GFXIP-specific PM4 into AQL packets and the ETW tracing path never submits a packet,
// so every entry point here reports the request as unsupported. The whole file goes
// away once the AQLProfile library itself builds on Windows.

#include "lib/aqlprofile/aqlprofile.hpp"

#include <hsa/hsa.h>

extern "C" {
hsa_status_t
aqlprofile_iterate_event_ids(aqlprofile_eventname_callback_t, void*)
{
    return HSA_STATUS_ERROR_INVALID_AGENT;
}

hsa_status_t
aqlprofile_pmc_create_packets(aqlprofile_handle_t*,
                              aqlprofile_pmc_aql_packets_t*,
                              aqlprofile_pmc_profile_t,
                              aqlprofile_memory_alloc_callback_t,
                              aqlprofile_memory_dealloc_callback_t,
                              aqlprofile_memory_copy_t,
                              void*)
{
    return HSA_STATUS_ERROR_INVALID_AGENT;
}

void aqlprofile_pmc_delete_packets(aqlprofile_handle_t) {}

hsa_status_t
aqlprofile_att_create_packets(aqlprofile_handle_t*,
                              aqlprofile_att_control_aql_packets_t*,
                              aqlprofile_att_profile_t,
                              aqlprofile_memory_alloc_callback_t,
                              aqlprofile_memory_dealloc_callback_t,
                              aqlprofile_memory_copy_t,
                              void*)
{
    return HSA_STATUS_ERROR_INVALID_AGENT;
}

void aqlprofile_att_delete_packets(aqlprofile_handle_t) {}

hsa_status_t
aqlprofile_att_iterate_data(aqlprofile_handle_t, aqlprofile_att_data_callback_t, void*)
{
    return HSA_STATUS_ERROR_INVALID_AGENT;
}

hsa_status_t
aqlprofile_att_codeobj_marker(hsa_ext_amd_aql_pm4_packet_t*,
                              aqlprofile_handle_t*,
                              aqlprofile_att_codeobj_data_t,
                              aqlprofile_memory_alloc_callback_t,
                              aqlprofile_memory_dealloc_callback_t,
                              void*)
{
    return HSA_STATUS_ERROR_INVALID_AGENT;
}
}
