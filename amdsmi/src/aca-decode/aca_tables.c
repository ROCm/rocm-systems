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
    {"cs", 0x11, "Hardware Assertion (HWA)"},
    {"cs", 0x12, "ST_PRT_ERR"},
    {"cs", 0x13, "ST_ECC_ERR"},
    {"cs", 0x14, "ST_TXN_ERR"},
    {"pie", 0x0, "Hardware Assertion (HWA)"},
    {"pie", 0x1, "CSW"},
    {"pie", 0x2, "GMI"},
    {"pie", 0x3, "FTI_DAT_STAT"},
    {"pie", 0x4, "DEF"},
    {"pie", 0x5, "Watchdog Timeout (WDT)"},
    {"pie", 0x6, "CNLI"},
    {"pie", 0x7, "RSLVFCI"},
    {"umc", 0x0, "On-die ECC"},
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
    {"umc", 0xf, "End-to-end CRC"},
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
    {"psp", 0x3f, "WAFL"},
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

const size_t NUM_BANKS = sizeof(bank_table) / sizeof(bank_table[0]);
const size_t NUM_ERRORS = sizeof(error_table) / sizeof(error_table[0]);
const size_t NUM_XCD_ERRORS = sizeof(xcd_error_table) / sizeof(xcd_error_table[0]);
const size_t NUM_AID_ERRORS = sizeof(aid_error_table) / sizeof(aid_error_table[0]);

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

    *bank_name = "UNKNOWN";
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

    *error_type = "UNKNOWN";
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

    *error_type = "UNKNOWN";
    return 1;
}
