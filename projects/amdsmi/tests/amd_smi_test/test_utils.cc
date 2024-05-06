/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2019, Advanced Micro Devices, Inc.
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

#include "amd_smi/amdsmi.h"
#include "test_utils.h"

static const std::map<amdsmi_fw_block_t, const char *> kDevFWNameMap = {
    {FW_ID_ASD, "asd"},
    {FW_ID_CP_CE, "ce"},
    {FW_ID_DMCU_ERAM, "dmcu"},  // TODO(bliu): double check
    {FW_ID_MC, "mc"},
    {FW_ID_CP_ME, "me"},
    {FW_ID_CP_MEC1, "mec1"},
    {FW_ID_CP_MEC2, "mec2"},
    {FW_ID_CP_MES, "mes"},
    {FW_ID_MES_KIQ, "mes_kiq"}, // TODO: double check
    {FW_ID_CP_PFP, "pfp"},
    {FW_ID_RLC, "rlc"},
    {FW_ID_RLC_SRLG, "rlc_srlg"},
    {FW_ID_RLC_SRLS, "rlc_srls"},
    {FW_ID_SDMA1, "sdma1"},
    {FW_ID_SDMA2, "sdma2"},
    {FW_ID_PM, "pm"},
    {FW_ID_PSP_SOSDRV, "sos"},
    {FW_ID_TA_RAS, "ta_ras"},
    {FW_ID_TA_XGMI, "ta_xgmi"},
    {FW_ID_UVD, "uvd"},
    {FW_ID_VCE, "vce"},
    {FW_ID_VCN, "vcn"},
};

const char *
NameFromFWEnum(amdsmi_fw_block_t blk) {
  return kDevFWNameMap.at(blk);
}

static const std::map<amdsmi_evt_notification_type_t, const char *>
                                                      kEvtNotifEvntNameMap = {
    {AMDSMI_EVT_NOTIF_VMFAULT, "AMDSMI_EVT_NOTIF_VMFAULT"},
    {AMDSMI_EVT_NOTIF_THERMAL_THROTTLE, "AMDSMI_EVT_NOTIF_THERMAL_THROTTLE"},
    {AMDSMI_EVT_NOTIF_GPU_PRE_RESET, "AMDSMI_EVT_NOTIF_GPU_PRE_RESET"},
    {AMDSMI_EVT_NOTIF_GPU_POST_RESET, "AMDSMI_EVT_NOTIF_GPU_POST_RESET"},
    {AMDSMI_EVT_NOTIF_RING_HANG, "AMDSMI_EVT_NOTIF_RING_HANG"},
};
const char *
NameFromEvtNotifType(amdsmi_evt_notification_type_t evt) {
  return kEvtNotifEvntNameMap.at(evt);
}
