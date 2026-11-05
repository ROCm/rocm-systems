// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Listing of KFD processes, defined in amd_smi_utils.cc. Kept out of the
// installed amd_smi_utils.h because it needs the rocm_smi headers, which are
// not installed; this header is not installed either.

#ifndef AMD_SMI_INCLUDE_AMD_SMI_KFD_PROCESS_LIST_H_
#define AMD_SMI_INCLUDE_AMD_SMI_KFD_PROCESS_LIST_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rocm_smi/rocm_smi.h"

/**
 *  @brief Lists KFD processes into a buffer the way amd::smi::GetProcessInfo()
 *  does: fills up to num_allocated entries, sets how many processes it found,
 *  and returns 0 or an errno.
 */
using SmiKfdProcessLister =
    std::function<int(rsmi_process_info_t* procs, uint32_t num_allocated, uint32_t* num_found)>;

/**
 *  @brief List the KFD processes by PID, listing again with more room while
 *  processes that start during a listing make it longer than the buffer.
 *
 *  @param[in] list Lists the processes; amd::smi::GetProcessInfo() outside tests.
 *  @param[in] expected How many processes the last listing found; sizes the
 *  first buffer.
 *  @param[out] procs The processes found, with only process_id set. Empty when
 *  KFD has no process directory (the listing returns ENOENT).
 *  @return ::AMDSMI_STATUS_SUCCESS, ::AMDSMI_STATUS_RETRY if processes kept
 *  starting through every listing, or the status for the listing's errno.
 */
auto smi_amdgpu_list_kfd_processes(const SmiKfdProcessLister& list, uint32_t expected,
                                   std::vector<rsmi_process_info_t>* procs) -> amdsmi_status_t;

#endif  // AMD_SMI_INCLUDE_AMD_SMI_KFD_PROCESS_LIST_H_
