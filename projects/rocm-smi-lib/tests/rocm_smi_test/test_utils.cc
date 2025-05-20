/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <map>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/test_utils.h"

static const std::map<rsmi_fw_block_t, const char *> kDevFWNameMap = {
    {RSMI_FW_BLOCK_ASD, "asd"},
    {RSMI_FW_BLOCK_CE, "ce"},
    {RSMI_FW_BLOCK_DMCU, "dmcu"},
    {RSMI_FW_BLOCK_MC, "mc"},
    {RSMI_FW_BLOCK_ME, "me"},
    {RSMI_FW_BLOCK_MEC, "mec"},
    {RSMI_FW_BLOCK_MEC2, "mec2"},
    {RSMI_FW_BLOCK_MES, "mes"},
    {RSMI_FW_BLOCK_MES_KIQ, "mes_kiq"},
    {RSMI_FW_BLOCK_PFP, "pfp"},
    {RSMI_FW_BLOCK_RLC, "rlc"},
    {RSMI_FW_BLOCK_RLC_SRLC, "rlc_srlc"},
    {RSMI_FW_BLOCK_RLC_SRLG, "rlc_srlg"},
    {RSMI_FW_BLOCK_RLC_SRLS, "rlc_srls"},
    {RSMI_FW_BLOCK_SDMA, "sdma"},
    {RSMI_FW_BLOCK_SDMA2, "sdma2"},
    {RSMI_FW_BLOCK_SMC, "smc"},
    {RSMI_FW_BLOCK_SOS, "sos"},
    {RSMI_FW_BLOCK_TA_RAS, "ta_ras"},
    {RSMI_FW_BLOCK_TA_XGMI, "ta_xgmi"},
    {RSMI_FW_BLOCK_UVD, "uvd"},
    {RSMI_FW_BLOCK_VCE, "vce"},
    {RSMI_FW_BLOCK_VCN, "vcn"},
};

const char *
NameFromFWEnum(rsmi_fw_block_t blk) {
  return kDevFWNameMap.at(blk);
}

static const std::map<rsmi_evt_notification_type_t, const char *>
                                                      kEvtNotifEvntNameMap = {
    {RSMI_EVT_NOTIF_VMFAULT, "RSMI_EVT_NOTIF_VMFAULT"},
    {RSMI_EVT_NOTIF_THERMAL_THROTTLE, "RSMI_EVT_NOTIF_THERMAL_THROTTLE"},
    {RSMI_EVT_NOTIF_GPU_PRE_RESET, "RSMI_EVT_NOTIF_GPU_PRE_RESET"},
    {RSMI_EVT_NOTIF_GPU_POST_RESET, "RSMI_EVT_NOTIF_GPU_POST_RESET"},
    {RSMI_EVT_NOTIF_EVENT_MIGRATE_START, "RSMI_EVT_NOTIF_EVENT_MIGRATE_START"},
    {RSMI_EVT_NOTIF_EVENT_MIGRATE_END, "RSMI_EVT_NOTIF_EVENT_MIGRATE_END"},
    {RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_START, "RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_START"},
    {RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_END, "RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_END"},
    {RSMI_EVT_NOTIF_EVENT_QUEUE_EVICTION, "RSMI_EVT_NOTIF_EVENT_QUEUE_EVICTION"},
    {RSMI_EVT_NOTIF_EVENT_QUEUE_RESTORE, "RSMI_EVT_NOTIF_EVENT_QUEUE_RESTORE"},
    {RSMI_EVT_NOTIF_EVENT_UNMAP_FROM_GPU, "RSMI_EVT_NOTIF_EVENT_UNMAP_FROM_GPU"},
    {RSMI_EVT_NOTIF_EVENT_ALL_PROCESS, "RSMI_EVT_NOTIF_EVENT_ALL_PROCESS"}
};
const char *
NameFromEvtNotifType(rsmi_evt_notification_type_t evt) {
  return kEvtNotifEvntNameMap.at(evt);
}
