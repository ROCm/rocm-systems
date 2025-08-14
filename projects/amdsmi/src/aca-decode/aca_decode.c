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
 * @file aca_decode.c
 * @brief Implementation of ACA error decoding functions
 *
 * This file contains functions for decoding and analyzing ACA error information from
 * raw register data. It provides functionality to determine error severity, bank
 * information, and specific error types based on hardware-specific error codes.
 */

#include "aca_decode.h"
#include "aca_tables.h"
#include "error_map.h"
#include "aca_constants.h"
#include <string.h>

/**
 * @brief Gets the bank name based on hardware ID and ACA type
 * @param[in] decoder Pointer to the ACA decoder structure
 * @param[out] bank_name Pointer to a string containing the bank name
 * @return 0 on success, -1 on failure
 */
static int
aca_decoder_get_bank(const aca_decoder_t *decoder, const char **bank_name)
{
    if (!decoder || !bank_name)
    {
        return -1;
    }

    const aca_ipid_fields_t *ipid = &decoder->ipid;
    return find_bank_name(ipid->hardware_id, ipid->aca_type, bank_name);
}

/**
 * @brief Determines the error severity based on status fields
 * @param[in] status Pointer to the ACA status fields structure
 * @return String indicating error severity: "Fatal", "Uncorrected, Non-fatal", "Corrected", or "UNKNOWN"
 */
static const char *get_error_severity(const aca_status_fields_t *status)
{
    if (status->poison)
        return ACA_SEVERITY_UNCORRECTED_NON_FATAL;
    if (status->pcc)
        return ACA_SEVERITY_FATAL;
    if (!status->pcc && status->uc && status->tcc)
        return ACA_SEVERITY_FATAL;
    if (!status->pcc && status->uc && !status->tcc)
        return ACA_SEVERITY_UNCORRECTED_NON_FATAL;
    if (!status->pcc && !status->uc && !status->tcc && status->deferred)
        return ACA_SEVERITY_UNCORRECTED_NON_FATAL;
    if (!status->pcc && !status->uc && !status->tcc && !status->deferred)
        return ACA_SEVERITY_CORRECTED;
    return ACA_SEVERITY_UNKNOWN;
}

/**
 * @brief Determines the error category based on bank and error type
 * @param[in] bank Pointer to the bank name
 * @param[in] error_type Pointer to the error type
 * @return String indicating error category: "HBM Errors", "Off-Package Link Errors", or "Device Internal Errors"
 */
static const char *get_error_category(const char *bank, const char *error_type)
{
    if (!bank || !error_type)
    {
        return ACA_SEVERITY_UNKNOWN;
    }

    if (strcmp(bank, ACA_BANK_UMC) == 0)
    {
        if (strcmp(error_type, ACA_ERROR_TYPE_ON_DIE_ECC) == 0 ||
            strcmp(error_type, "WriteDataPoisonErr") == 0 ||
            strcmp(error_type, "AddressCommandParityErr") == 0 ||
            strcmp(error_type, "WriteDataCrcErr") == 0 ||
            strcmp(error_type, "EcsErr") == 0 ||
            strcmp(error_type, "RdCrcErr") == 0 ||
            strcmp(error_type, ACA_ERROR_TYPE_END_TO_END_CRC) == 0)
        {
            return ACA_CATEGORY_HBM_ERRORS;
        }
    }
    else if (strcmp(bank, ACA_BANK_PCS_XGMI) == 0 ||
             strcmp(bank, ACA_BANK_KPX_SERDES) == 0 ||
             strcmp(bank, ACA_BANK_KPX_WAFL) == 0 ||
             (strcmp(bank, ACA_BANK_PSP) == 0 && strcmp(error_type, ACA_ERROR_TYPE_WAFL) == 0))
    {
        return ACA_CATEGORY_OFF_PACKAGE_LINK_ERRORS;
    }

    return ACA_CATEGORY_DEVICE_INTERNAL_ERRORS;
}

/**
 * @brief Determines the service error type from error attributes
 * @param[in] error_category Pointer to the error category string
 * @param[in] error_bank Pointer to the error bank string
 * @param[in] error_type Pointer to the error type string
 * @param[in] error_severity Pointer to the error severity string
 * @param[out] service_error_type Pointer to store the resulting service error type string
 * @return 0 on success, non-zero on failure
 */
static int get_service_error_type(const char *error_category, const char *error_bank, const char *error_type,
                                  const char *error_severity, const char **service_error_type)
{
    if (!error_category || !error_type || !error_severity || !service_error_type ||
        strcmp(error_category, ACA_SEVERITY_UNKNOWN) == 0 ||
        strcmp(error_type, ACA_SEVERITY_UNKNOWN) == 0 ||
        strcmp(error_severity, ACA_SEVERITY_UNKNOWN) == 0)
    {
        return -1;
    }
    if (strcmp(error_type, ACA_ERROR_TYPE_BAD_PAGE_RETIREMENT_THRESHOLD) == 0)
    {
        *service_error_type = ACA_ERROR_TYPE_BAD_PAGE_RETIREMENT_THRESHOLD;
        return 0;
    }
    if ((strcmp(error_category, ACA_CATEGORY_HBM_ERRORS) == 0) && (strcmp(error_severity, ACA_SEVERITY_CORRECTED) == 0))
    {
        *service_error_type = ACA_ERROR_TYPE_ALL;
        return 0;
    }
    if (strcmp(error_type, "RdCrcErr") == 0)
    {
        *service_error_type = ACA_ERROR_TYPE_END_TO_END_CRC;
        return 0;
    }
    if ((strcmp(error_category, ACA_CATEGORY_HBM_ERRORS) == 0) && (strcmp(error_severity, ACA_SEVERITY_FATAL) == 0) &&
        (strcmp(error_type, ACA_ERROR_TYPE_ON_DIE_ECC) != 0) && (strcmp(error_type, ACA_ERROR_TYPE_END_TO_END_CRC) != 0))
    {
        *service_error_type = ACA_ERROR_TYPE_ALL_OTHERS;
        return 0;
    }
    if (strcmp(error_category, ACA_CATEGORY_DEVICE_INTERNAL_ERRORS) == 0)
    {
        if ((strcmp(error_severity, ACA_SEVERITY_UNCORRECTED_NON_FATAL) == 0 ||
             strcmp(error_severity, ACA_SEVERITY_CORRECTED) == 0 ||
             strcmp(error_severity, ACA_SEVERITY_FATAL) == 0) &&
            strcmp(error_type, ACA_ERROR_TYPE_HARDWARE_ASSERTION) != 0 &&
            strcmp(error_type, ACA_ERROR_TYPE_WATCHDOG_TIMEOUT) != 0)
        {
            *service_error_type = ACA_ERROR_TYPE_ALL_OTHERS;
            return 0;
        }
    }
    if (strcmp(error_category, ACA_CATEGORY_OFF_PACKAGE_LINK_ERRORS) == 0)
    {
        if (strcmp(error_bank, ACA_BANK_PCS_XGMI) == 0)
        {
            *service_error_type = ACA_ERROR_TYPE_XGMI;
            return 0;
        }
        if (strcmp(error_bank, ACA_BANK_KPX_WAFL) == 0)
        {
            *service_error_type = ACA_ERROR_TYPE_WAFL;
            return 0;
        }
    }

    return -1;
}

/**
 * @brief Extracts error information from the decoder and populates the info structure
 * @param[in] decoder Pointer to the ACA decoder structure
 * @param[out] info Pointer to the error info structure to be populated
 */
static void aca_decoder_get_error_info(const aca_decoder_t *decoder, aca_error_info_t *info)
{
    const char *bank;
    const char *error_type;
    const char *instance_name;
    int result;
    
    info->raw_status = decoder->aca_status;
    info->raw_addr = decoder->aca_addr;
    info->raw_ipid = decoder->aca_ipid;
    info->raw_synd = decoder->aca_synd;
    
    info->scrub = decoder->status.scrub;
    info->error_code_ext = decoder->status.error_code_ext;


    result = aca_decoder_get_bank(decoder, &bank);
    if (result < 0)
    {
        bank = ACA_SEVERITY_UNKNOWN;
    }
    info->bank_ref = bank;

    if (find_instance_name(bank, decoder->ipid.instance_id_lo, &instance_name) == 0)
    {
        info->instance_ref = instance_name;
    }
    else
    {
        info->instance_ref = ACA_ERROR_TYPE_DECODE_INAPPLICABLE;
    }

    // 0b1000 indicate error threshold has been exceeded, and is always fatal
    if (decoder->flags & ACA_FLAG_THRESHOLD_EXCEEDED)
    {
        info->severity_ref = ACA_SEVERITY_FATAL;
    }
    else
    {
        info->severity_ref = get_error_severity(&decoder->status);
    }

    // Decode OAM and AID from instance_id_lo
    oam_aid_map_t oam_aid = {0};
    uint8_t instance_id_lo = decoder->ipid.instance_id_lo & 0xFF;  // Get lower 8 bits
    if (find_oam_aid(instance_id_lo, &oam_aid) == 0)
    {
        info->oam = oam_aid.oam;
        info->aid = oam_aid.aid;
    }
    else
    {
        info->oam = -1;  // Invalid value
        info->aid = -1;  // Invalid value
    }

    if (decoder->status.error_code_ext >= ACA_ERROR_CODE_EXT_MIN && decoder->status.error_code_ext <= ACA_ERROR_CODE_EXT_MAX)
    {
        uint32_t instance_id = decoder->ipid.instance_id_lo;
        uint32_t error_info = decoder->synd.error_information & 0xFF;

        if ((instance_id == ACA_INSTANCE_ID_XCD0_400 || instance_id == ACA_INSTANCE_ID_XCD1_400 ||
             instance_id == ACA_INSTANCE_ID_XCD0_401 || instance_id == ACA_INSTANCE_ID_XCD1_401) &&
            find_error_in_table(xcd_error_table, NUM_XCD_ERRORS, error_info, &error_type) == 0)
        {
            info->error_type_ref = error_type;
        }
        else if ((instance_id == ACA_INSTANCE_ID_AID_400 || instance_id == ACA_INSTANCE_ID_AID_401) &&
                 find_error_in_table(aid_error_table, NUM_AID_ERRORS, error_info, &error_type) == 0)
        {
            info->error_type_ref = error_type;
        }
        else
        {
            info->error_type_ref = ACA_SEVERITY_UNKNOWN;
        }
    }
    // 0b1000 indicate error threshold has been exceeded
    else if (decoder->flags & ACA_FLAG_THRESHOLD_EXCEEDED)
    {
        info->error_type_ref = ACA_ERROR_TYPE_BAD_PAGE_RETIREMENT_THRESHOLD;
    }
    else
    {
        if (find_error_type_by_bank(bank, decoder->status.error_code_ext, &error_type) == 0)
        {
            info->error_type_ref = error_type;
        }
        else
        {
            info->error_type_ref = ACA_SEVERITY_UNKNOWN;
        }
    }

    // 0b1000 indicate error threshold has been exceeded, and is always a HBM error
    if (decoder->flags & ACA_FLAG_THRESHOLD_EXCEEDED)
    {
        info->category_ref = ACA_CATEGORY_HBM_ERRORS;
    }
    else
    {
        info->category_ref = get_error_category(bank, info->error_type_ref);
    }

    const char *service_error;
    if (get_service_error_type(info->category_ref, info->bank_ref, info->error_type_ref, info->severity_ref, &service_error) != 0)
    {
        service_error = info->error_type_ref;
    }
    
    info->afid = get_error_id(info->category_ref, service_error, info->severity_ref);
}

/**
 * @brief Initializes an ACA decoder structure with raw register values
 * @param[out] decoder Pointer to the decoder structure to initialize
 * @param[in] hw_revision Hardware hw_revision number
 * @param[in] flags Decoder flags
 * @param[in] status_reg Raw status register value
 * @param[in] ipid_reg Raw IPID register value
 * @param[in] synd_reg Raw syndrome register value
 */
static void aca_decoder_init(aca_decoder_t *decoder, uint16_t hw_revision, uint32_t flags,
                             uint64_t status_reg, uint64_t ipid_reg, uint64_t synd_reg)
{
    memset(decoder, 0, sizeof(aca_decoder_t));

    decoder->hw_revision = hw_revision;
    decoder->flags = flags;
    decoder->aca_status = status_reg;
    decoder->aca_ipid = ipid_reg;
    decoder->aca_synd = synd_reg;

    aca_status_init(&decoder->status, status_reg);
    aca_ipid_init(&decoder->ipid, ipid_reg);
    aca_synd_init(&decoder->synd, synd_reg);
}

aca_error_info_t aca_decode(const aca_raw_data_t *raw_data)
{
    aca_decoder_t decoder = {0};
    aca_error_info_t info = {0};

    aca_decoder_init(&decoder,
                     raw_data->hw_revision,
                     raw_data->flags,
                     raw_data->aca_status,
                     raw_data->aca_ipid,
                     raw_data->aca_synd);

    aca_decoder_get_error_info(&decoder, &info);
    return info;
}
