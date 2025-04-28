/**
 * @file aca_tables.h
 * @brief ACA lookup table definitions and helper functions
 * @details Contains data structures and functions definitions for mapping ACA Registers
 *          into their corresponding names and types.
 */

#ifndef ACA_TABLES_H
#define ACA_TABLES_H

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Structure mapping hardware ID and ACA type to bank names
 */
typedef struct
{
    uint16_t hw_id;    /**< Hardware ID value */
    uint16_t aca_type; /**< ACA type identifier */
    const char *name;  /**< Bank name string */
} aca_bank_entry_t;

/**
 * @brief Structure mapping bank-specific error codes to error types
 */
typedef struct
{
    const char *bank;    /**< Bank name string */
    uint32_t error_code; /**< Error code value */
    const char *type;    /**< Error type string */
} aca_error_type_t;

/**
 * @brief Structure for generic error code to error type mapping
 */
typedef struct
{
    uint32_t error_code; /**< Error code value */
    const char *type;    /**< Error type string */
} aca_error_entry_t;

// External table declarations
extern const aca_bank_entry_t bank_table[];
extern const aca_error_type_t error_table[];
extern const aca_error_entry_t xcd_error_table[];
extern const aca_error_entry_t aid_error_table[];

// Table size constants
extern const size_t NUM_BANKS;
extern const size_t NUM_ERRORS;
extern const size_t NUM_XCD_ERRORS;
extern const size_t NUM_AID_ERRORS;

/**
 * @brief Find bank name based on hardware ID and ACA type
 * @param[in] hw_id Hardware ID value
 * @param[in] aca_type ACA type value
 * @param[out] bank_name Pointer to store result string
 * @return 0 on success, 1 if not found, -1 on parameter error
 */
int find_bank_name(uint16_t hw_id, uint16_t aca_type, const char **bank_name);

/**
 * @brief Find error type for a specific bank and error code
 * @param[in] bank Bank name string
 * @param[in] error_code Error code value
 * @param[out] error_type Pointer to store result string
 * @return 0 on success, 1 if not found, -1 on parameter error
 */
int find_error_type_by_bank(const char *bank, uint32_t error_code, const char **error_type);

/**
 * @brief Generic lookup for error codes in an error table
 * @param[in] table Pointer to error table
 * @param[in] table_size Number of table entries
 * @param[in] error_code Error code to look up
 * @param[out] error_type Pointer to store result string
 * @return 0 on success, 1 if not found, -1 on parameter error
 */
int find_error_in_table(const aca_error_entry_t *table, size_t table_size,
                        uint32_t error_code, const char **error_type);

#endif
