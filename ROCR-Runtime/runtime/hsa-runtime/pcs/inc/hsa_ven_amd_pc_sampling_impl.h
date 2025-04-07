////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_VEN_AMD_PC_SAMPLING_IMPL_H
#define HSA_VEN_AMD_PC_SAMPLING_IMPL_H

#include "inc/hsa.h"
#include "inc/hsa_ext_amd.h"
#include "inc/hsa_ven_amd_pc_sampling.h"
#include "core/inc/hsa_ext_interface.h"

//---------------------------------------------------------------------------//
//  APIs that implement PC Sampling functionality
//---------------------------------------------------------------------------//

namespace rocr {
namespace pcs {

hsa_status_t hsa_ven_amd_pcs_iterate_configuration(
    hsa_agent_t agent, hsa_ven_amd_pcs_iterate_configuration_callback_t configuration_callback,
    void* callback_data);

hsa_status_t hsa_ven_amd_pcs_create(hsa_agent_t agent, hsa_ven_amd_pcs_method_kind_t method,
                                    hsa_ven_amd_pcs_units_t units, size_t interval, size_t latency,
                                    size_t buffer_size,
                                    hsa_ven_amd_pcs_data_ready_callback_t data_ready_callback,
                                    void* client_callback_data, hsa_ven_amd_pcs_t* pc_sampling);

hsa_status_t hsa_ven_amd_pcs_create_from_id(
    uint32_t pcs_id, hsa_agent_t agent, hsa_ven_amd_pcs_method_kind_t method,
    hsa_ven_amd_pcs_units_t units, size_t interval, size_t latency, size_t buffer_size,
    hsa_ven_amd_pcs_data_ready_callback_t data_ready_callback, void* client_callback_data,
    hsa_ven_amd_pcs_t* pc_sampling);

hsa_status_t hsa_ven_amd_pcs_destroy(hsa_ven_amd_pcs_t pc_sampling);

hsa_status_t hsa_ven_amd_pcs_start(hsa_ven_amd_pcs_t pc_sampling);

hsa_status_t hsa_ven_amd_pcs_stop(hsa_ven_amd_pcs_t pc_sampling);

hsa_status_t hsa_ven_amd_pcs_flush(hsa_ven_amd_pcs_t pc_sampling);

// Update Api table with func pointers that implement functionality
void LoadPcSampling(core::PcSamplingExtTableInternal* pcs_api);

// Release resources acquired by Image implementation
void ReleasePcSamplingRsrcs();

}  // namespace pcs
}  // namespace rocr

#endif  //  HSA_VEN_AMD_PC_SAMPLING_IMPL_H
