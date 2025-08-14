/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * @file aca_tables.c
 * @brief ACA Decode Tables Implementation
 *
 * This file contains lookup tables and helper functions for mapping ACA error codes
 * to human-readable strings. It includes:
 * - Bank mapping table for hardware IDs and ACA types
 * - Error type mapping table for bank-specific error codes
 * - GFX error mapping tables for XCD and AID errors
 * - Lookup functions to find bank names and error types
 */

#include "aca_tables.h"
#include "aca_constants.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/**
 * @brief Mapping table for hardware IDs and ACA types to bank names
 */
const aca_bank_entry_t bank_table[] = {
    {0x2E, 0x02, "cs"},
    {0x2E, 0x01, "pie"},
    {0x96, 0x00, "umc"},
    {0xFF, 0x01, "psp"},
    {0x01, 0x01, "smu"},
    {0x18, 0x00, "nbio"},
    {0x46, 0x01, "pcie"},
    {0x05, 0x00, "pb"},
    {0x259, 0x00, "kpx_serdes"},
    {0x2E, 0x04, "mall"},
    {0x267, 0x00, "kpx_wafl"},
    {0x50, 0x00, "pcs_xgmi"},
    {0x6C, 0x00, "nbif"},
    {0x80, 0x00, "shub"},
    {0x170, 0x00, "usr_dp"},
    {0x180, 0x00, "usr_cp"}};

/**
 * @brief Mapping table for bank-specific error codes to error types
 */
const aca_error_type_t error_table[] = {
    {"cs", 0x0, "FTI_ILL_REQ"},
    {"cs", 0x1, "FTI_ADDR_VIOL"},
    {"cs", 0x2, "FTI_SEC_VIOL"},
    {"cs", 0x3, "FTI_ILL_RSP"},
    {"cs", 0x4, "FTI_RSP_NO_MTCH"},
    {"cs", 0x5, "FTI_PAR_ERR"},
    {"cs", 0x6, "SDP_PAR_ERR"},
    {"cs", 0x7, "ATM_PAR_ERR"},
    {"cs", 0x8, "SDP_RSP_NO_MTCH"},
    {"cs", 0x9, "SPF_PRT_ERR"},
    {"cs", 0xa, "SPF_ECC_ERR"},
    {"cs", 0xb, "SDP_UNEXP_RETRY"},
    {"cs", 0xc, "CNTR_OVFL"},
    {"cs", 0xd, "CNTR_UNFL"},
    {"cs", 0xe, "FTI_ND_ILL_REQ"},
    {"cs", 0xf, "FTI_ND_ADDR_VIOL"},
    {"cs", 0x10, "FTI_ND_SEC_VIOL"},
    {"cs", 0x11, ACA_ERROR_TYPE_HARDWARE_ASSERTION},
    {"cs", 0x12, "ST_PRT_ERR"},
    {"cs", 0x13, "ST_ECC_ERR"},
    {"cs", 0x14, "ST_TXN_ERR"},
    {"pie", 0x0, ACA_ERROR_TYPE_HARDWARE_ASSERTION},
    {"pie", 0x1, "CSW"},
    {"pie", 0x2, "GMI"},
    {"pie", 0x3, "FTI_DAT_STAT"},
    {"pie", 0x4, "DEF"},
    {"pie", 0x5, ACA_ERROR_TYPE_WATCHDOG_TIMEOUT},
    {"pie", 0x6, "CNLI"},
    {"pie", 0x7, "RSLVFCI"},
    {"umc", 0x0, ACA_ERROR_TYPE_ON_DIE_ECC},
    {"umc", 0x1, "WriteDataPoisonErr"},
    {"umc", 0x2, "SdpParityErr"},
    {"umc", 0x4, "AddressCommandParityErr"},
    {"umc", 0x5, "WriteDataCrcErr"},
    {"umc", 0x6, "SramEccErr"},
    {"umc", 0x9, "EcsErr"},
    {"umc", 0xa, "ThrttlErr"},
    {"umc", 0xb, "RdCrcErr"},
    {"umc", 0xd, "MpFwErr"},
    {"umc", 0xe, "MpParErr"},
    {"umc", 0xf, ACA_ERROR_TYPE_END_TO_END_CRC},
    {"psp", 0x0, "Mp0HighSramError"},
    {"psp", 0x1, "Mp0LowSramError"},
    {"psp", 0x2, "Mp0IDataBank0Error"},
    {"psp", 0x3, "Mp0IDataBank1Error"},
    {"psp", 0x4, "Mp0ITagRam0Error"},
    {"psp", 0x5, "Mp0ITagRam1Error"},
    {"psp", 0x6, "Mp0DDataBank0Error"},
    {"psp", 0x7, "Mp0DDataBank1Error"},
    {"psp", 0x8, "Mp0DDataBank2Error"},
    {"psp", 0x9, "Mp0DDataBank3Error"},
    {"psp", 0xa, "Mp0DTagBank0Error"},
    {"psp", 0xb, "Mp0DTagBank1Error"},
    {"psp", 0xc, "Mp0DTagBank2Error"},
    {"psp", 0xd, "Mp0DTagBank3Error"},
    {"psp", 0xe, "Mp0DDirtyRamError"},
    {"psp", 0xf, "Mp0TlbBank0Error"},
    {"psp", 0x10, "Mp0TlbBank1Error"},
    {"psp", 0x11, "Mp0SHubIfRdBufError"},
    {"psp", 0x12, "PhyRamEccError"},
    {"psp", 0x3a, "PoisonDataConsumption"},
    {"psp", 0x3b, "SRAM_EDC"},
    {"psp", 0x3c, "SMN_Parity"},
    {"psp", 0x3d, "SMN_Timeout"},
    {"psp", 0x3f, ACA_ERROR_TYPE_WAFL},
    {"smu", 0x0, "Mp5HighSramError"},
    {"smu", 0x1, "Mp5LowSramError"},
    {"smu", 0x2, "Mp5DCacheAError"},
    {"smu", 0x3, "Mp5DCacheBError"},
    {"smu", 0x4, "Mp5DTagAError"},
    {"smu", 0x5, "Mp5DTagBError"},
    {"smu", 0x6, "Mp5ICacheAError"},
    {"smu", 0x7, "Mp5ICacheBError"},
    {"smu", 0x8, "Mp5ITagAError"},
    {"smu", 0x9, "Mp5ITagBError"},
    {"smu", 0xb, "PhyRamEccError"},
    {"smu", 0x3a, "GFX_IP_Correctable_Error"},
    {"smu", 0x3b, "GFX_IP_Fatal_Error"},
    {"smu", 0x3d, "Reserved"},
    {"smu", 0x3e, "GFX_IP_Poison_Error"},
    {"nbio", 0x0, "EccParityError"},
    {"nbio", 0x1, "PCIE_Sideband"},
    {"nbio", 0x2, "Ext_ErrEvent"},
    {"nbio", 0x3, "Egress_Poison"},
    {"nbio", 0x4, "IOHC_Internal_Poison"},
    {"nbio", 0x5, "Int_ErrEvent"},
    {"pcie", 0x0, "SDP_PARITY_ERR_LOG"},
    {"pb", 0x0, "EccError"},
    {"kpx_serdes", 0x0, "RAMECC"},
    {"kpx_serdes", 0x1, "ARCIns"},
    {"kpx_serdes", 0x2, "ARCData"},
    {"kpx_serdes", 0x3, "APB"},
    {"mall", 0x0, "CNTR_OVFL"},
    {"mall", 0x1, "CNTR_UNFL"},
    {"mall", 0x2, "CSDP_PAR_ERR"},
    {"mall", 0x3, "USDP_PAR_ERR"},
    {"mall", 0x4, "CACHE_TAG0_ERR"},
    {"mall", 0x5, "CACHE_TAG1_ERR"},
    {"mall", 0x6, "CACHE_DAT_ERR"},
    {"kpx_wafl", 0x0, "RAMECC"},
    {"kpx_wafl", 0x1, "ARCIns"},
    {"kpx_wafl", 0x2, "ARCData"},
    {"kpx_wafl", 0x3, "APB"},
    {"pcs_xgmi", 0x0, "DataLossErr"},
    {"pcs_xgmi", 0x1, "TrainingErr"},
    {"pcs_xgmi", 0x2, "FlowCtrlAckErr"},
    {"pcs_xgmi", 0x3, "RxFifoUnderflowErr"},
    {"pcs_xgmi", 0x4, "RxFifoOverflowErr"},
    {"pcs_xgmi", 0x5, "CRCErr"},
    {"pcs_xgmi", 0x6, "BERExceededErr"},
    {"pcs_xgmi", 0x7, "TxMetaDataErr_TxVcidDataErr"},
    {"pcs_xgmi", 0x8, "ReplayBufParityErr"},
    {"pcs_xgmi", 0x9, "DataParityErr"},
    {"pcs_xgmi", 0xa, "ReplayFifoOverflowErr"},
    {"pcs_xgmi", 0xb, "ReplaFifoUnderflowErr"},
    {"pcs_xgmi", 0xc, "ElasticFifoOverflowErr"},
    {"pcs_xgmi", 0xd, "DeskewErr"},
    {"pcs_xgmi", 0xe, "FlowCtrlCRCErr"},
    {"pcs_xgmi", 0xf, "DataStartupLimitErr"},
    {"pcs_xgmi", 0x10, "FCInitTimeoutErr"},
    {"pcs_xgmi", 0x11, "RecoveryTimeoutErr"},
    {"pcs_xgmi", 0x12, "ReadySerialTimeoutErr"},
    {"pcs_xgmi", 0x13, "ReadySerialAttemptErr"},
    {"pcs_xgmi", 0x14, "RecoveryAttemptErr"},
    {"pcs_xgmi", 0x15, "RecoveryRelockAttemptErr"},
    {"pcs_xgmi", 0x16, "ReplayAttemptErr"},
    {"pcs_xgmi", 0x17, "SyncHdrErr"},
    {"pcs_xgmi", 0x18, "TxReplayTimeoutErr"},
    {"pcs_xgmi", 0x19, "RxReplayTimeoutErr"},
    {"pcs_xgmi", 0x1a, "LinkSubTxTimeoutErr"},
    {"pcs_xgmi", 0x1b, "LinkSubRxTimeoutErr"},
    {"pcs_xgmi", 0x1c, "RxCMDPktErr"},
    {"nbif", 0x0, "TIMEOUT_ERR"},
    {"nbif", 0x1, "SRAM_ECC_ERR"},
    {"nbif", 0x2, "NTB_ERR_EVENT"},
    {"nbif", 0x3, "SDP_PARITY_ERR"},
    {"shub", 0x0, "TIMEOUT_ERR"},
    {"shub", 0x1, "SRAM_ECC_ERR"},
    {"shub", 0x2, "NTB_ERR_EVENT"},
    {"shub", 0x3, "SDP_PARITY_ERR"},
    {"usr_dp", 0x0, "MstCMDErr"},
    {"usr_dp", 0x1, "MstRxFIFOErr"},
    {"usr_dp", 0x2, "MstDeskewErr"},
    {"usr_dp", 0x3, "MstDetectTimeoutErr"},
    {"usr_dp", 0x4, "MstFlowControlErr"},
    {"usr_dp", 0x5, "MstDataValidFifoErr"},
    {"usr_dp", 0x6, "macLinkStateErr"},
    {"usr_dp", 0x7, "DeskewErr"},
    {"usr_dp", 0x8, "InitTimeoutErr"},
    {"usr_dp", 0x9, "InitAttemptErr"},
    {"usr_dp", 0xa, "RecoveryTimeoutErr"},
    {"usr_dp", 0xb, "RecoveryAttemptErr"},
    {"usr_dp", 0xc, "EyeTrainingTimeoutErr"},
    {"usr_dp", 0xd, "DataStartupLimitErr"},
    {"usr_dp", 0xe, "LS0ExitErr"},
    {"usr_dp", 0xf, "PLLpowerStateUpdateTimeoutErr"},
    {"usr_dp", 0x10, "RxFifoErr"},
    {"usr_dp", 0x11, "LcuErr"},
    {"usr_dp", 0x12, "convCECCErr"},
    {"usr_dp", 0x13, "convUECCErr"},
    {"usr_dp", 0x15, "rxDataLossErr"},
    {"usr_dp", 0x16, "ReplayCECCErr"},
    {"usr_dp", 0x17, "ReplayUECCErr"},
    {"usr_dp", 0x18, "CRCErr"},
    {"usr_dp", 0x19, "BERExceededErr"},
    {"usr_dp", 0x1a, "FCInitTimeoutErr"},
    {"usr_dp", 0x1b, "FCInitAttemptErr"},
    {"usr_dp", 0x1c, "ReplayTimoutErr"},
    {"usr_dp", 0x1d, "ReplayAttemptErr"},
    {"usr_dp", 0x1e, "ReplayUnderflowErr"},
    {"usr_dp", 0x1f, "ReplayOverflowErr"},
    {"usr_cp", 0x0, "PacketTypeErr"},
    {"usr_cp", 0x1, "RxFifoErr"},
    {"usr_cp", 0x2, "DeskewErr"},
    {"usr_cp", 0x3, "RxDetectTimeoutErr"},
    {"usr_cp", 0x4, "DataParityErr"},
    {"usr_cp", 0x5, "DataLossErr"},
    {"usr_cp", 0x6, "LcuErr"},
    {"usr_cp", 0x7, "HB1HandshakeTimeoutErr"},
    {"usr_cp", 0x8, "HB2HandshakeTimeoutErr"},
    {"usr_cp", 0x9, "ClkSleepRspTimeoutErr"},
    {"usr_cp", 0xa, "ClkWakeRspTimeoutErr"},
    {"usr_cp", 0xb, "resetAttackErr"},
    {"usr_cp", 0xc, "remoteLinkFatalErr"},
};

/**
 * @brief Error GFX mapping table for XCD errors
 */
const aca_error_entry_t xcd_error_table[] = {
    {0x0, "GfxGcError"},
    {0x1, "GfxGcError"},
    {0x2, "GfxGcError"},
    {0x3, "GfxGcError"},
    {0x4, "GfxGcError"},
    {0x5, "GfxGcError"},
    {0x6, "GfxGcError"},
    {0x7, "GfxGcError"},
    {0x8, "GfxGcError"},
    {0x9, "GfxGcError"},
    {0xa, "GfxGcError"},
    {0xb, "GfxGcError"},
    {0xc, "GfxGcError"},
    {0xd, "GfxGcError"},
    {0xe, "GfxGcError"},
    {0xf, "GfxGcError"},
    {0x10, "GfxGcError"},
    {0x28, "Reserved"},
    {0x2a, "Reserved"}};

/**
 * @brief Error GFX mapping table for AID errors
 */
const aca_error_entry_t aid_error_table[] = {
    {0x0, "GfxGcError"},
    {0x1, "GfxGcError"},
    {0x2, "GfxGcError"},
    {0x3, "GfxGcError"},
    {0x4, "GfxGcError"},
    {0x5, "GfxMmhubError"},
    {0x6, "GfxMmhubError"},
    {0x7, "GfxMmhubError"},
    {0x8, "GfxMmhubError"},
    {0x9, "GfxMmhubError"},
    {0xa, "GfxMmhubError"},
    {0xb, "GfxMmhubError"},
    {0xc, "GfxMmhubError"},
    {0xd, "GfxGcError"},
    {0xe, "GfxVcnError"},
    {0xf, "GfxVcnError"},
    {0x10, "GfxVcnError"},
    {0x11, "GfxVcnError"},
    {0x12, "GfxVcnError"},
    {0x13, "GfxVcnError"},
    {0x14, "GfxVcnError"},
    {0x15, "GfxVcnError"},
    {0x16, "GfxVcnError"},
    {0x17, "GfxVcnError"},
    {0x18, "GfxVcnError"},
    {0x19, "GfxVcnError"},
    {0x1a, "GfxVcnError"},
    {0x1b, "GfxVcnError"},
    {0x1c, "GfxVcnError"},
    {0x1d, "GfxVcnError"},
    {0x1e, "GfxVcnError"},
    {0x1f, "GfxVcnError"},
    {0x20, "GfxVcnError"},
    {0x21, "GfxSdmaError"},
    {0x22, "GfxSdmaError"},
    {0x23, "GfxSdmaError"},
    {0x24, "GfxSdmaError"},
    {0x25, "GfxHdpError"},
    {0x26, "GfxAthubError"},
    {0x27, "GfxGcError"},
    {0x28, "Reserved"},
    {0x29, "Reserved"},
    {0x2a, "Reserved"},
    {0x2b, "Reserved"}};

/**
 * @brief Table mapping instance_id_hi to OAM and AID values
 */
static const oam_aid_map_t oam_aid_table[] = {
    {0, 0}, /* 0x00 */
    {1, 0}, /* 0x01 */
    {2, 0}, /* 0x02 */
    {3, 0}, /* 0x03 */
    {0, 1}, /* 0x04 */
    {1, 1}, /* 0x05 */
    {2, 1}, /* 0x06 */
    {3, 1}, /* 0x07 */
    {0, 2}, /* 0x08 */
    {1, 2}, /* 0x09 */
    {2, 2}, /* 0x0A */
    {3, 2}, /* 0x0B */
    {0, 3}, /* 0x0C */
    {1, 3}, /* 0x0D */
    {2, 3}, /* 0x0E */
    {3, 3}  /* 0x0F */
};

// Constants are now defined as global variables

/**
 * @brief Table mapping bank and instance ID to instance names
 */
static const aca_instance_entry_t instance_table[] = {
    {"cs", 0x1F002000, "cmp0"},
    {"cs", 0x1F000000, "cs0"},
    {"cs", 0x1F000A00, "cs10"},
    {"cs", 0x1F000B00, "cs11"},
    {"cs", 0x1F000C00, "cs12"},
    {"cs", 0x1F000D00, "cs13"},
    {"cs", 0x1F000E00, "cs14"},
    {"cs", 0x1F000F00, "cs15"},
    {"cs", 0x1F001000, "cs16"},
    {"cs", 0x1F001100, "cs17"},
    {"cs", 0x1F001200, "cs18"},
    {"cs", 0x1F001300, "cs19"},
    {"cs", 0x1F000100, "cs1"},
    {"cs", 0x1F001400, "cs20"},
    {"cs", 0x1F001500, "cs21"},
    {"cs", 0x1F001600, "cs22"},
    {"cs", 0x1F001700, "cs23"},
    {"cs", 0x1F001800, "cs24"},
    {"cs", 0x1F001900, "cs25"},
    {"cs", 0x1F001A00, "cs26"},
    {"cs", 0x1F001B00, "cs27"},
    {"cs", 0x1F001C00, "cs28"},
    {"cs", 0x1F001D00, "cs29"},
    {"cs", 0x1F000200, "cs2"},
    {"cs", 0x1F001E00, "cs30"},
    {"cs", 0x1F001F00, "cs31"},
    {"cs", 0x1F000300, "cs3"},
    {"cs", 0x1F000400, "cs4"},
    {"cs", 0x1F000500, "cs5"},
    {"cs", 0x1F000600, "cs6"},
    {"cs", 0x1F000700, "cs7"},
    {"cs", 0x1F000800, "cs8"},
    {"cs", 0x1F000900, "cs9"},
    {"mall", 0x1F005900, "mall0"},
    {"mall", 0x1F006300, "mall10"},
    {"mall", 0x1F006400, "mall11"},
    {"mall", 0x1F006500, "mall12"},
    {"mall", 0x1F006600, "mall13"},
    {"mall", 0x1F006700, "mall14"},
    {"mall", 0x1F006800, "mall15"},
    {"mall", 0x1F006900, "mall16"},
    {"mall", 0x1F006A00, "mall17"},
    {"mall", 0x1F006B00, "mall18"},
    {"mall", 0x1F006C00, "mall19"},
    {"mall", 0x1F005A00, "mall1"},
    {"mall", 0x1F006D00, "mall20"},
    {"mall", 0x1F006E00, "mall21"},
    {"mall", 0x1F006F00, "mall22"},
    {"mall", 0x1F007000, "mall23"},
    {"mall", 0x1F007100, "mall24"},
    {"mall", 0x1F007200, "mall25"},
    {"mall", 0x1F007300, "mall26"},
    {"mall", 0x1F007400, "mall27"},
    {"mall", 0x1F007500, "mall28"},
    {"mall", 0x1F007600, "mall29"},
    {"mall", 0x1F005B00, "mall2"},
    {"mall", 0x1F007700, "mall30"},
    {"mall", 0x1F007800, "mall31"},
    {"mall", 0x1F005C00, "mall3"},
    {"mall", 0x1F005D00, "mall4"},
    {"mall", 0x1F005E00, "mall5"},
    {"mall", 0x1F005F00, "mall6"},
    {"mall", 0x1F006000, "mall7"},
    {"mall", 0x1F006100, "mall8"},
    {"mall", 0x1F006200, "mall9"},
    {"pb", 0x5EA00, "pb"},
    {"pb", 0x30082900, "ccd0 pbccd"},
    {"pb", 0x32082900, "ccd1 pbccd"},
    {"pb", 0x34082900, "ccd2 pbccd"},
    {"pb", 0x36082900, "xcd0 pbccd"},
    {"pb", 0x38082900, "xcd1 pbccd"},
    {"umc", 0x90F00, "ch0 umc0"},
    {"umc", 0x290F00, "ch0 umc1"},
    {"umc", 0x490F00, "ch0 umc2"},
    {"umc", 0x690F00, "ch0 umc3"},
    {"umc", 0x91F00, "ch1 umc0"},
    {"umc", 0x291F00, "ch1 umc1"},
    {"umc", 0x491F00, "ch1 umc2"},
    {"umc", 0x691F00, "ch1 umc3"},
    {"umc", 0x92F00, "ch2 umc0"},
    {"umc", 0x292F00, "ch2 umc1"},
    {"umc", 0x492F00, "ch2 umc2"},
    {"umc", 0x692F00, "ch2 umc3"},
    {"umc", 0x93F00, "ch3 umc0"},
    {"umc", 0x293F00, "ch3 umc1"},
    {"umc", 0x493F00, "ch3 umc2"},
    {"umc", 0x693F00, "ch3 umc3"},
    {"umc", 0x190F00, "ch4 umc0"},
    {"umc", 0x390F00, "ch4 umc1"},
    {"umc", 0x590F00, "ch4 umc2"},
    {"umc", 0x790F00, "ch4 umc3"},
    {"umc", 0x191F00, "ch5 umc0"},
    {"umc", 0x391F00, "ch5 umc1"},
    {"umc", 0x591F00, "ch5 umc2"},
    {"umc", 0x791F00, "ch5 umc3"},
    {"umc", 0x192F00, "ch6 umc0"},
    {"umc", 0x392F00, "ch6 umc1"},
    {"umc", 0x592F00, "ch6 umc2"},
    {"umc", 0x792F00, "ch6 umc3"},
    {"umc", 0x193F00, "ch7 umc0"},
    {"umc", 0x393F00, "ch7 umc1"},
    {"umc", 0x593F00, "ch7 umc2"},
    {"umc", 0x793F00, "ch7 umc3"}};

const size_t NUM_OAM_AID_ENTRIES = sizeof(oam_aid_table) / sizeof(oam_aid_table[0]);
const size_t NUM_BANKS = sizeof(bank_table) / sizeof(bank_table[0]);
const size_t NUM_ERRORS = sizeof(error_table) / sizeof(error_table[0]);
const size_t NUM_XCD_ERRORS = sizeof(xcd_error_table) / sizeof(xcd_error_table[0]);
const size_t NUM_AID_ERRORS = sizeof(aid_error_table) / sizeof(aid_error_table[0]);
const size_t NUM_INSTANCES = sizeof(instance_table) / sizeof(instance_table[0]);

int find_bank_name(uint16_t hw_id, uint16_t aca_type, const char **bank_name)
{
    if (!bank_name)
    {
        return -1;
    }

    for (size_t i = 0; i < NUM_BANKS; i++)
    {
        if (bank_table[i].hw_id == hw_id &&
            bank_table[i].aca_type == aca_type)
        {
            *bank_name = bank_table[i].name;
            return 0;
        }
    }

    *bank_name = ACA_SEVERITY_UNKNOWN;
    return 1;
}

int find_error_type_by_bank(const char *bank, uint32_t error_code, const char **error_type)
{
    if (!bank || !error_type)
    {
        return -1;
    }

    for (size_t i = 0; i < NUM_ERRORS; i++)
    {
        if (error_code == error_table[i].error_code &&
            strcmp(bank, error_table[i].bank) == 0)
        {
            *error_type = error_table[i].type;
            return 0;
        }
    }

    *error_type = ACA_SEVERITY_UNKNOWN;
    return 1;
}

int find_error_in_table(const aca_error_entry_t *table, size_t table_size,
                        uint32_t error_code, const char **error_type)
{
    if (!table || !error_type)
    {
        return -1;
    }

    for (size_t i = 0; i < table_size; i++)
    {
        if (table[i].error_code == error_code)
        {
            *error_type = table[i].type;
            return 0;
        }
    }

    *error_type = ACA_SEVERITY_UNKNOWN;
    return 1;
}

int find_oam_aid(uint8_t instance_id_hi, oam_aid_map_t *oam_aid)
{
    if (!oam_aid || instance_id_hi >= NUM_OAM_AID_ENTRIES)
    {
        return -1;
    }

    oam_aid->oam = oam_aid_table[instance_id_hi].oam;
    oam_aid->aid = oam_aid_table[instance_id_hi].aid;
    return 0;
}

int find_instance_name(const char *bank, uint32_t instance_id_lo, const char **instance_name)
{
    if (!bank || !instance_name)
    {
        return -1;
    }

    // Mask off the lower 2 bits as specified
    uint32_t masked_id = instance_id_lo & 0xFFFFFFFC;

    for (size_t i = 0; i < NUM_INSTANCES; i++)
    {
        if (instance_table[i].instance_id_lo == masked_id &&
            strcmp(bank, instance_table[i].bank) == 0)
        {
            *instance_name = instance_table[i].name;
            return 0;
        }
    }

    *instance_name = ACA_SEVERITY_UNKNOWN;
    return 1;
}
