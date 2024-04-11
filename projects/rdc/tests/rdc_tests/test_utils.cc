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
    {FW_ID_SMU, "SMU"},
    {FW_ID_FIRST, "FIRST"},
    {FW_ID_CP_CE, "CP_CE"},
    {FW_ID_CP_PFP, "CP_PFP"},
    {FW_ID_CP_ME, "CP_ME"},
    {FW_ID_CP_MEC_JT1, "CP_MEC_JT1"},
    {FW_ID_CP_MEC_JT2, "CP_MEC_JT2"},
    {FW_ID_CP_MEC1, "CP_MEC1"},
    {FW_ID_CP_MEC2, "CP_MEC2"},
    {FW_ID_RLC, "RLC"},
    {FW_ID_SDMA0, "SDMA0"},
    {FW_ID_SDMA1, "SDMA1"},
    {FW_ID_SDMA2, "SDMA2"},
    {FW_ID_SDMA3, "SDMA3"},
    {FW_ID_SDMA4, "SDMA4"},
    {FW_ID_SDMA5, "SDMA5"},
    {FW_ID_SDMA6, "SDMA6"},
    {FW_ID_SDMA7, "SDMA7"},
    {FW_ID_VCN, "VCN"},
    {FW_ID_UVD, "UVD"},
    {FW_ID_VCE, "VCE"},
    {FW_ID_ISP, "ISP"},
    {FW_ID_DMCU_ERAM, "DMCU_ERAM"},
    {FW_ID_DMCU_ISR, "DMCU_ISR"},
    {FW_ID_RLC_RESTORE_LIST_GPM_MEM, "RLC_RESTORE_LIST_GPM_MEM"},
    {FW_ID_RLC_RESTORE_LIST_SRM_MEM, "RLC_RESTORE_LIST_SRM_MEM"},
    {FW_ID_RLC_RESTORE_LIST_CNTL, "RLC_RESTORE_LIST_CNTL"},
    {FW_ID_RLC_V, "RLC_V"},
    {FW_ID_MMSCH, "MMSCH"},
    {FW_ID_PSP_SYSDRV, "PSP_SYSDRV"},
    {FW_ID_PSP_SOSDRV, "PSP_SOSDRV"},
    {FW_ID_PSP_TOC, "PSP_TOC"},
    {FW_ID_PSP_KEYDB, "PSP_KEYDB"},
    {FW_ID_DFC, "DFC"},
    {FW_ID_PSP_SPL, "PSP_SPL"},
    {FW_ID_DRV_CAP, "DRV_CAP"},
    {FW_ID_MC, "MC"},
    {FW_ID_PSP_BL, "PSP_BL"},
    {FW_ID_CP_PM4, "CP_PM4"},
    {FW_ID_RLC_P, "RLC_P"},
    {FW_ID_SEC_POLICY_STAGE2, "SEC_POLICY_STAGE2"},
    {FW_ID_REG_ACCESS_WHITELIST, "REG_ACCESS_WHITELIST"},
    {FW_ID_IMU_DRAM, "IMU_DRAM"},
    {FW_ID_IMU_IRAM, "IMU_IRAM"},
    {FW_ID_SDMA_TH0, "SDMA_TH0"},
    {FW_ID_SDMA_TH1, "SDMA_TH1"},
    {FW_ID_CP_MES, "CP_MES"},
    {FW_ID_MES_KIQ, "MES_KIQ"},
    {FW_ID_MES_STACK, "MES_STACK"},
    {FW_ID_MES_THREAD1, "MES_THREAD1"},
    {FW_ID_MES_THREAD1_STACK, "MES_THREAD1_STACK"},
    {FW_ID_RLX6, "RLX6"},
    {FW_ID_RLX6_DRAM_BOOT, "RLX6_DRAM_BOOT"},
    {FW_ID_RS64_ME, "RS64_ME"},
    {FW_ID_RS64_ME_P0_DATA, "RS64_ME_P0_DATA"},
    {FW_ID_RS64_ME_P1_DATA, "RS64_ME_P1_DATA"},
    {FW_ID_RS64_PFP, "RS64_PFP"},
    {FW_ID_RS64_PFP_P0_DATA, "RS64_PFP_P0_DATA"},
    {FW_ID_RS64_PFP_P1_DATA, "RS64_PFP_P1_DATA"},
    {FW_ID_RS64_MEC, "RS64_MEC"},
    {FW_ID_RS64_MEC_P0_DATA, "RS64_MEC_P0_DATA"},
    {FW_ID_RS64_MEC_P1_DATA, "RS64_MEC_P1_DATA"},
    {FW_ID_RS64_MEC_P2_DATA, "RS64_MEC_P2_DATA"},
    {FW_ID_RS64_MEC_P3_DATA, "RS64_MEC_P3_DATA"},
    {FW_ID_PPTABLE, "PPTABLE"},
    {FW_ID_PSP_SOC, "PSP_SOC"},
    {FW_ID_PSP_DBG, "PSP_DBG"},
    {FW_ID_PSP_INTF, "PSP_INTF"},
    {FW_ID_RLX6_CORE1, "RLX6_CORE1"},
    {FW_ID_RLX6_DRAM_BOOT_CORE1, "RLX6_DRAM_BOOT_CORE1"},
    {FW_ID_RLCV_LX7, "RLCV_LX7"},
    {FW_ID_RLC_SAVE_RESTORE_LIST, "RLC_SAVE_RESTORE_LIST"},
    {FW_ID_ASD, "ASD"},
    {FW_ID_TA_RAS, "TA_RAS"},
    {FW_ID_TA_XGMI, "TA_XGMI"},
    {FW_ID_RLC_SRLG, "RLC_SRLG"},
    {FW_ID_RLC_SRLS, "RLC_SRLS"},
    {FW_ID_PM, "PM"},
    {FW_ID_DMCU, "DMCU"},
};

const char* NameFromFWEnum(amdsmi_fw_block_t blk) { return kDevFWNameMap.at(blk); }
