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
#include <stdio.h>
#include <string.h>

/**
 * @brief Gets the bank name based on hardware ID and ACA type
 * @param[in] decoder Pointer to the ACA decoder structure
 * @param[out] bank_name Pointer to a string containing the bank name
 * @return 0 on success, -1 on failure
 */
static int aca_decoder_get_bank(const aca_decoder_t *decoder, const char **bank_name)
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
        return "Uncorrected, Non-fatal";
    if (status->pcc)
        return "Fatal";
    if (!status->pcc && status->uc && status->tcc)
        return "Fatal";
    if (!status->pcc && status->uc && !status->tcc)
        return "Uncorrected, Non-fatal";
    if (!status->pcc && !status->uc && !status->tcc && status->deferred)
        return "Uncorrected, Non-fatal";
    if (!status->pcc && !status->uc && !status->tcc && !status->deferred)
        return "Corrected";
    return "UNKNOWN";
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
        return "UNKNOWN";
    }

    if (strcmp(bank, "umc") == 0)
    {
        if (strcmp(error_type, "On-die ECC") == 0 ||
            strcmp(error_type, "WriteDataPoisonErr") == 0 ||
            strcmp(error_type, "AddressCommandParityErr") == 0 ||
            strcmp(error_type, "WriteDataCrcErr") == 0 ||
            strcmp(error_type, "EcsErr") == 0 ||
            strcmp(error_type, "RdCrcErr") == 0 ||
            strcmp(error_type, "End-to-end CRC") == 0)
        {
            return "HBM Errors";
        }
    }
    else if (strcmp(bank, "pcs_xgmi") == 0 ||
             strcmp(bank, "kpx_serdes") == 0 ||
             strcmp(bank, "kpx_wafl") == 0 ||
             (strcmp(bank, "psp") == 0 && strcmp(error_type, "WAFL") == 0))
    {
        return "Off-Package Link Errors";
    }

    return "Device Internal Errors";
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
        strcmp(error_category, "UNKNOWN") == 0 ||
        strcmp(error_type, "UNKNOWN") == 0 ||
        strcmp(error_severity, "UNKNOWN") == 0)
    {
        return -1;
    }

    if ((strcmp(error_category, "HBM Errors") == 0) && (strcmp(error_severity, "Corrected") == 0))
    {
        *service_error_type = "All";
        return 0;
    }
    if ((strcmp(error_category, "HBM Errors") == 0) && (strcmp(error_severity, "Fatal") == 0) &&
        (strcmp(error_type, "On-die ECC") != 0) && (strcmp(error_type, "End-to-end CRC") != 0))
    {
        *service_error_type = "All Others";
        return 0;
    }
    if (strcmp(error_category, "Device Internal Errors") == 0)
    {
        if ((strcmp(error_severity, "Uncorrected, Non-fatal") == 0 ||
             strcmp(error_severity, "Corrected") == 0 ||
             strcmp(error_severity, "Fatal") == 0) &&
            strcmp(error_type, "Hardware Assertion (HWA)") != 0 &&
            strcmp(error_type, "Watchdog Timeout (WDT)") != 0)
        {
            *service_error_type = "All Others";
            return 0;
        }
    }
    if (strcmp(error_category, "Off-Package Link Errors") == 0)
    {
        if (strcmp(error_bank, "pcs_xgmi") == 0)
        {
            *service_error_type = "XGMI";
            return 0;
        }
        if (strcmp(error_bank, "kpx_wafl") == 0)
        {
            *service_error_type = "WAFL";
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
    int result;

    result = aca_decoder_get_bank(decoder, &bank);
    if (result < 0)
    {
        bank = "UNKNOWN";
    }
    info->bank_ref = bank;

    info->severity_ref = get_error_severity(&decoder->status);

    if (decoder->status.error_code_ext >= 0x3A && decoder->status.error_code_ext <= 0x3E)
    {
        uint32_t instance_id = decoder->ipid.instance_id_lo;
        uint32_t error_info = decoder->synd.error_information & 0xFF;

        if ((instance_id == 0x36430400 || instance_id == 0x38430400 ||
             instance_id == 0x36430401 || instance_id == 0x38430401) &&
            find_error_in_table(xcd_error_table, NUM_XCD_ERRORS, error_info, &error_type) == 0)
        {
            info->error_type_ref = error_type;
        }
        else if ((instance_id == 0x3B30400 || instance_id == 0x3B30401) &&
                 find_error_in_table(aid_error_table, NUM_AID_ERRORS, error_info, &error_type) == 0)
        {
            info->error_type_ref = error_type;
        }
        else
        {
            info->error_type_ref = "UNKNOWN";
        }
    }
    else
    {
        if (find_error_type_by_bank(bank, decoder->status.error_code_ext, &error_type) == 0)
        {
            info->error_type_ref = error_type;
        }
        else
        {
            info->error_type_ref = "UNKNOWN";
        }
    }

    info->category_ref = get_error_category(bank, info->error_type_ref);

    const char *service_error;
    if (get_service_error_type(info->category_ref, info->bank_ref, info->error_type_ref, info->severity_ref, &service_error) != 0)
    {
        service_error = info->error_type_ref;
    }

    printf("Error Category: %s\n", info->category_ref);
    printf("Error Type: %s\n", service_error);
    printf("Error Severity: %s\n", info->severity_ref);

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
