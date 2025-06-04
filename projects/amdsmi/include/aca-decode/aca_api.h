#ifndef ACA_API_H
#define ACA_API_H

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Structure containing decoded error information
 */
typedef struct
{
    const char *bank_ref;       /**< Reference to bank name string */
    const char *error_type_ref; /**< Reference to error type string */
    const char *severity_ref;   /**< Reference to error severity string */
    const char *category_ref;   /**< Reference to error category string */
    const char *instance_ref;   /**< Reference to instance name string */
    int oam;                    /**< OAM value */  
    int aid;                    /**< AID value */
    int afid;                   /**< AFID value (AMD Field ID) */
    uint64_t raw_status;        /**< Raw status register value */ 
    uint64_t raw_addr;          /**< Raw address register value */
    uint64_t raw_ipid;          /**< Raw IPID register value */
    uint64_t raw_synd;          /**< Raw syndrome register value */
    uint8_t scrub;              /**< Scrub bit from status */
    uint8_t error_code_ext;     /**< Extended error code from status */
} aca_error_info_t;

/**
 * @brief Decodes the AFID from a register array
 * @param[in] register_array Pointer to an array of 64-bit register values
 * @param[in] array_len Size of register array in elements
 * @param[in] flag Decoder flags
 * @param[in] hw_revision Hardware revision number
 * @return AFID value or -1 if decoding fails
 */
int decode_afid(const uint64_t *register_array, size_t array_len, uint32_t flag, uint16_t hw_revision);

/**
 * @brief Decodes and returns complete error information from a register array
 * @param[in] register_array Pointer to an array of 64-bit register values
 * @param[in] array_len Size of register array in elements
 * @param[in] flag Decoder flags
 * @param[in] hw_revision Hardware revision number
 * @return Complete error information structure
 */
aca_error_info_t decode_error_info(const uint64_t *register_array, size_t array_len, uint32_t flag, uint16_t hw_revision);

#endif // ACA_API_H
