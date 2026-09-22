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

#include "rdc_tests/test_utils.h"

#include <map>

#include "amd_smi/amdsmi.h"

static const std::map<amdsmi_fw_block_t, const char*> kDevFWNameMap = {
    {AMDSMI_FW_ID_SMU, "SMU"},
    {AMDSMI_FW_ID_FIRST, "FIRST"},
    {AMDSMI_FW_ID_CP_CE, "CP_CE"},
    {AMDSMI_FW_ID_CP_PFP, "CP_PFP"},
    {AMDSMI_FW_ID_CP_ME, "CP_ME"},
    {AMDSMI_FW_ID_CP_MEC_JT1, "CP_MEC_JT1"},
    {AMDSMI_FW_ID_CP_MEC_JT2, "CP_MEC_JT2"},
    {AMDSMI_FW_ID_CP_MEC1, "CP_MEC1"},
    {AMDSMI_FW_ID_CP_MEC2, "CP_MEC2"},
    {AMDSMI_FW_ID_RLC, "RLC"},
    {AMDSMI_FW_ID_SDMA0, "SDMA0"},
    {AMDSMI_FW_ID_SDMA1, "SDMA1"},
    {AMDSMI_FW_ID_SDMA2, "SDMA2"},
    {AMDSMI_FW_ID_SDMA3, "SDMA3"},
    {AMDSMI_FW_ID_SDMA4, "SDMA4"},
    {AMDSMI_FW_ID_SDMA5, "SDMA5"},
    {AMDSMI_FW_ID_SDMA6, "SDMA6"},
    {AMDSMI_FW_ID_SDMA7, "SDMA7"},
    {AMDSMI_FW_ID_VCN, "VCN"},
    {AMDSMI_FW_ID_UVD, "UVD"},
    {AMDSMI_FW_ID_VCE, "VCE"},
    {AMDSMI_FW_ID_ISP, "ISP"},
    {AMDSMI_FW_ID_DMCU_ERAM, "DMCU_ERAM"},
    {AMDSMI_FW_ID_DMCU_ISR, "DMCU_ISR"},
    {AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM, "RLC_RESTORE_LIST_GPM_MEM"},
    {AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM, "RLC_RESTORE_LIST_SRM_MEM"},
    {AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL, "RLC_RESTORE_LIST_CNTL"},
    {AMDSMI_FW_ID_RLC_V, "RLC_V"},
    {AMDSMI_FW_ID_MMSCH, "MMSCH"},
    {AMDSMI_FW_ID_PSP_SYSDRV, "PSP_SYSDRV"},
    {AMDSMI_FW_ID_PSP_SOSDRV, "PSP_SOSDRV"},
    {AMDSMI_FW_ID_PSP_TOC, "PSP_TOC"},
    {AMDSMI_FW_ID_PSP_KEYDB, "PSP_KEYDB"},
    {AMDSMI_FW_ID_DFC, "DFC"},
    {AMDSMI_FW_ID_PSP_SPL, "PSP_SPL"},
    {AMDSMI_FW_ID_DRV_CAP, "DRV_CAP"},
    {AMDSMI_FW_ID_MC, "MC"},
    {AMDSMI_FW_ID_PSP_BL, "PSP_BL"},
    {AMDSMI_FW_ID_CP_PM4, "CP_PM4"},
    {AMDSMI_FW_ID_RLC_P, "RLC_P"},
    {AMDSMI_FW_ID_SEC_POLICY_STAGE2, "SEC_POLICY_STAGE2"},
    {AMDSMI_FW_ID_REG_ACCESS_WHITELIST, "REG_ACCESS_WHITELIST"},
    {AMDSMI_FW_ID_IMU_DRAM, "IMU_DRAM"},
    {AMDSMI_FW_ID_IMU_IRAM, "IMU_IRAM"},
    {AMDSMI_FW_ID_SDMA_TH0, "SDMA_TH0"},
    {AMDSMI_FW_ID_SDMA_TH1, "SDMA_TH1"},
    {AMDSMI_FW_ID_CP_MES, "CP_MES"},
    {AMDSMI_FW_ID_MES_KIQ, "MES_KIQ"},
    {AMDSMI_FW_ID_MES_STACK, "MES_STACK"},
    {AMDSMI_FW_ID_MES_THREAD1, "MES_THREAD1"},
    {AMDSMI_FW_ID_MES_THREAD1_STACK, "MES_THREAD1_STACK"},
    {AMDSMI_FW_ID_RLX6, "RLX6"},
    {AMDSMI_FW_ID_RLX6_DRAM_BOOT, "RLX6_DRAM_BOOT"},
    {AMDSMI_FW_ID_RS64_ME, "RS64_ME"},
    {AMDSMI_FW_ID_RS64_ME_P0_DATA, "RS64_ME_P0_DATA"},
    {AMDSMI_FW_ID_RS64_ME_P1_DATA, "RS64_ME_P1_DATA"},
    {AMDSMI_FW_ID_RS64_PFP, "RS64_PFP"},
    {AMDSMI_FW_ID_RS64_PFP_P0_DATA, "RS64_PFP_P0_DATA"},
    {AMDSMI_FW_ID_RS64_PFP_P1_DATA, "RS64_PFP_P1_DATA"},
    {AMDSMI_FW_ID_RS64_MEC, "RS64_MEC"},
    {AMDSMI_FW_ID_RS64_MEC_P0_DATA, "RS64_MEC_P0_DATA"},
    {AMDSMI_FW_ID_RS64_MEC_P1_DATA, "RS64_MEC_P1_DATA"},
    {AMDSMI_FW_ID_RS64_MEC_P2_DATA, "RS64_MEC_P2_DATA"},
    {AMDSMI_FW_ID_RS64_MEC_P3_DATA, "RS64_MEC_P3_DATA"},
    {AMDSMI_FW_ID_PPTABLE, "PPTABLE"},
    {AMDSMI_FW_ID_PSP_SOC, "PSP_SOC"},
    {AMDSMI_FW_ID_PSP_DBG, "PSP_DBG"},
    {AMDSMI_FW_ID_PSP_INTF, "PSP_INTF"},
    {AMDSMI_FW_ID_RLX6_CORE1, "RLX6_CORE1"},
    {AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1, "RLX6_DRAM_BOOT_CORE1"},
    {AMDSMI_FW_ID_RLCV_LX7, "RLCV_LX7"},
    {AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST, "RLC_SAVE_RESTORE_LIST"},
    {AMDSMI_FW_ID_ASD, "ASD"},
    {AMDSMI_FW_ID_TA_RAS, "TA_RAS"},
    {AMDSMI_FW_ID_TA_XGMI, "TA_XGMI"},
    {AMDSMI_FW_ID_RLC_SRLG, "RLC_SRLG"},
    {AMDSMI_FW_ID_RLC_SRLS, "RLC_SRLS"},
    {AMDSMI_FW_ID_PM, "PM"},
    {AMDSMI_FW_ID_DMCU, "DMCU"},
};

const char* NameFromFWEnum(amdsmi_fw_block_t blk) { return kDevFWNameMap.at(blk); }
