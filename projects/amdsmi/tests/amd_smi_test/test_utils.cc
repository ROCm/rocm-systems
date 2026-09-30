// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "test_utils.h"

#include <map>

#include "amd_smi/amdsmi.h"

static const std::map<amdsmi_fw_block_t, const char*> kDevFWNameMap = {
    {AMDSMI_FW_ID_ASD, "asd"},
    {AMDSMI_FW_ID_CP_CE, "ce"},
    {AMDSMI_FW_ID_DMCU_ERAM, "dmcu"},  // TODO(bliu): double check
    {AMDSMI_FW_ID_MC, "mc"},
    {AMDSMI_FW_ID_CP_ME, "me"},
    {AMDSMI_FW_ID_CP_MEC1, "mec1"},
    {AMDSMI_FW_ID_CP_MEC2, "mec2"},
    {AMDSMI_FW_ID_CP_MES, "mes"},
    {AMDSMI_FW_ID_MES_KIQ, "mes_kiq"},  // TODO: double check
    {AMDSMI_FW_ID_CP_PFP, "pfp"},
    {AMDSMI_FW_ID_RLC, "rlc"},
    {AMDSMI_FW_ID_RLC_SRLG, "rlc_srlg"},
    {AMDSMI_FW_ID_RLC_SRLS, "rlc_srls"},
    {AMDSMI_FW_ID_SDMA1, "sdma1"},
    {AMDSMI_FW_ID_SDMA2, "sdma2"},
    {AMDSMI_FW_ID_PM, "pm"},
    {AMDSMI_FW_ID_PSP_SOSDRV, "sos"},
    {AMDSMI_FW_ID_TA_RAS, "ta_ras"},
    {AMDSMI_FW_ID_TA_XGMI, "ta_xgmi"},
    {AMDSMI_FW_ID_UVD, "uvd"},
    {AMDSMI_FW_ID_VCE, "vce"},
    {AMDSMI_FW_ID_VCN, "vcn"},
};

const char* NameFromFWEnum(amdsmi_fw_block_t blk) { return kDevFWNameMap.at(blk); }

static const std::map<amdsmi_evt_notification_type_t, const char*> kEvtNotifEvntNameMap = {
    {AMDSMI_EVT_NOTIF_VMFAULT, "AMDSMI_EVT_NOTIF_VMFAULT"},
    {AMDSMI_EVT_NOTIF_THERMAL_THROTTLE, "AMDSMI_EVT_NOTIF_THERMAL_THROTTLE"},
    {AMDSMI_EVT_NOTIF_GPU_PRE_RESET, "AMDSMI_EVT_NOTIF_GPU_PRE_RESET"},
    {AMDSMI_EVT_NOTIF_GPU_POST_RESET, "AMDSMI_EVT_NOTIF_GPU_POST_RESET"},
    {AMDSMI_EVT_NOTIF_MIGRATE_START, "AMDSMI_EVT_NOTIF_MIGRATE_START"},
    {AMDSMI_EVT_NOTIF_MIGRATE_END, "AMDSMI_EVT_NOTIF_MIGRATE_END"},
    {AMDSMI_EVT_NOTIF_PAGE_FAULT_START, "AMDSMI_EVT_NOTIF_PAGE_FAULT_START"},
    {AMDSMI_EVT_NOTIF_PAGE_FAULT_END, "AMDSMI_EVT_NOTIF_PAGE_FAULT_END"},
    {AMDSMI_EVT_NOTIF_QUEUE_EVICTION, "AMDSMI_EVT_NOTIF_QUEUE_EVICTION"},
    {AMDSMI_EVT_NOTIF_QUEUE_RESTORE, "AMDSMI_EVT_NOTIF_QUEUE_RESTORE"},
    {AMDSMI_EVT_NOTIF_UNMAP_FROM_GPU, "AMDSMI_EVT_NOTIF_UNMAP_FROM_GPU"},
    {AMDSMI_EVT_NOTIF_PROCESS_START, "AMDSMI_EVT_NOTIF_PROCESS_START"},
    {AMDSMI_EVT_NOTIF_PROCESS_END, "AMDSMI_EVT_NOTIF_PROCESS_END"},
};
const char* NameFromEvtNotifType(amdsmi_evt_notification_type_t evt) {
  return kEvtNotifEvntNameMap.at(evt);
}
