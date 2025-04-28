/**
 * @file aca_fields.c
 * @brief Implementation of ACA register field handling
 *
 * This file contains functions for initializing and reading various ACA register fields
 * including status, IPID, and syndrome registers. Each function
 * extracts specific bit fields from raw register values and populates corresponding
 * field structures.
 */

#include "aca_fields.h"

/**
 * @brief Extracts a bit field from a value
 * @param[in] value The source value to extract bits from
 * @param[in] start Starting bit position
 * @param[in] count Number of bits to extract
 * @return The extracted bits as a value
 */
#define EXTRACT_BITS(value, start, count) (((value) >> (start)) & ((1ULL << (count)) - 1))

uint64_t aca_fields_read(const aca_fields_t *fields)
{
    return fields->raw_value;
}

void aca_status_init(aca_status_fields_t *fields, uint64_t status_reg)
{
    fields->base.raw_value = status_reg;
    fields->error_code = EXTRACT_BITS(status_reg, 0, 16);
    fields->error_code_ext = EXTRACT_BITS(status_reg, 16, 6);
    fields->reserv22 = EXTRACT_BITS(status_reg, 22, 2);
    fields->addr_lsb = EXTRACT_BITS(status_reg, 24, 6);
    fields->reserv30 = EXTRACT_BITS(status_reg, 30, 2);
    fields->err_core_id = EXTRACT_BITS(status_reg, 32, 6);
    fields->reserv38 = EXTRACT_BITS(status_reg, 38, 2);
    fields->scrub = EXTRACT_BITS(status_reg, 40, 1);
    fields->reserv41 = EXTRACT_BITS(status_reg, 41, 2);
    fields->poison = EXTRACT_BITS(status_reg, 43, 1);
    fields->deferred = EXTRACT_BITS(status_reg, 44, 1);
    fields->uecc = EXTRACT_BITS(status_reg, 45, 1);
    fields->cecc = EXTRACT_BITS(status_reg, 46, 1);
    fields->reserv47 = EXTRACT_BITS(status_reg, 47, 5);
    fields->synd_v = EXTRACT_BITS(status_reg, 53, 1);
    fields->reserv54 = EXTRACT_BITS(status_reg, 54, 1);
    fields->tcc = EXTRACT_BITS(status_reg, 55, 1);
    fields->err_core_id_val = EXTRACT_BITS(status_reg, 56, 1);
    fields->pcc = EXTRACT_BITS(status_reg, 57, 1);
    fields->addr_v = EXTRACT_BITS(status_reg, 58, 1);
    fields->misc_v = EXTRACT_BITS(status_reg, 59, 1);
    fields->en = EXTRACT_BITS(status_reg, 60, 1);
    fields->uc = EXTRACT_BITS(status_reg, 61, 1);
    fields->overflow = EXTRACT_BITS(status_reg, 62, 1);
    fields->val = EXTRACT_BITS(status_reg, 63, 1);
}

void aca_ipid_init(aca_ipid_fields_t *fields, uint64_t ipid_reg)
{
    fields->base.raw_value = ipid_reg;
    fields->instance_id_lo = EXTRACT_BITS(ipid_reg, 0, 32);
    fields->hardware_id = EXTRACT_BITS(ipid_reg, 32, 12);
    fields->instance_id_hi = EXTRACT_BITS(ipid_reg, 44, 4);
    fields->aca_type = EXTRACT_BITS(ipid_reg, 48, 16);
}

void aca_synd_init(aca_synd_fields_t *fields, uint64_t synd_reg)
{
    fields->base.raw_value = synd_reg;
    fields->error_information = EXTRACT_BITS(synd_reg, 0, 18);
    fields->length = EXTRACT_BITS(synd_reg, 18, 6);
    fields->error_priority = EXTRACT_BITS(synd_reg, 24, 3);
    fields->reserved27 = EXTRACT_BITS(synd_reg, 27, 5);
    fields->syndrome = EXTRACT_BITS(synd_reg, 32, 7);
    fields->reserved39 = EXTRACT_BITS(synd_reg, 39, 25);
}
