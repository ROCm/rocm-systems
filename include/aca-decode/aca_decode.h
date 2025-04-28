/**
 * @file aca_decode.h
 * @brief Internal decoder interface and data structures
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ACA_DECODE_H
#define ACA_DECODE_H

#include "aca_fields.h"

/**
 * @brief Internal decoder structure with parsed register fields
 */
typedef struct
{
    uint64_t aca_status;  /**< Raw status register value */
    uint64_t aca_ipid;    /**< Raw IPID register value */
    uint64_t aca_synd;    /**< Raw syndrome register value */
    uint32_t flags;       /**< Decoder flags */
    uint16_t hw_revision; /**< Hardware hw_revision */

    aca_status_fields_t status; /**< Parsed status fields */
    aca_ipid_fields_t ipid;     /**< Parsed IPID fields */
    aca_synd_fields_t synd;     /**< Parsed syndrome fields */
} aca_decoder_t;

/**
 * @brief Structure containing raw ACA error data from hardware
 */
typedef struct
{
    uint64_t aca_status;  /**< Raw status register value */
    uint64_t aca_ipid;    /**< Raw IPID register value */
    uint64_t aca_synd;    /**< Raw syndrome register value */
    uint32_t flags;       /**< Flags from descriptor */
    uint16_t hw_revision; /**< Hardware hw_revision number */
} aca_raw_data_t;

/**
 * @brief Structure containing decoded error information
 */
typedef struct
{
    const char *bank_ref;       /**< Reference to bank name string */
    const char *error_type_ref; /**< Reference to error type string */
    const char *severity_ref;   /**< Reference to error severity string */
    const char *category_ref;   /**< Reference to error category string */
    int afid;                   /**< AFID value (AMD Fault ID) */
} aca_error_info_t;

/**
 * @brief Main decode function that processes raw ACA error data
 * @param[in] raw_data Pointer to structure containing raw ACA error data
 * @return Decoded error information structure
 */
aca_error_info_t aca_decode(const aca_raw_data_t *raw_data);

#ifdef __cplusplus
}
#endif
#endif /* ACA_DECODE_H */
