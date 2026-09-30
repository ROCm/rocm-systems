// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ADDC_LIB_C_H
#define ADDC_LIB_C_H

#include "addc/export.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Stable C API contract version. Increment only for an incompatible ABI. */
#define ADDC_API_VERSION 2u

/** Common ADDC result-schema version emitted by this product. */
#define ADDC_SCHEMA_VERSION "3.3.0"

/** AFID used when an event cannot be classified more specifically. */
#define ADDC_UNCLASSIFIED_AFID 16999

typedef struct addc_context addc_context_t;

typedef enum
{
    ADDC_PARSE_MODE_TOLERANT = 0,
    ADDC_PARSE_MODE_STRICT = 1
} addc_parse_mode_t;

/** Versioned context configuration. Zero-initialize before assigning fields. */
typedef struct
{
    uint32_t struct_size;
    addc_parse_mode_t parse_mode;
    uint64_t reserved[4];
} addc_context_options_t;

/**
 * Create a reusable context that caches decoder registries.
 * Use this when processing multiple CPERs to avoid rebuilding
 * registries on every call.
 *
 * Returns NULL on allocation failure.
 * Free with addc_context_destroy().
 * A fully constructed context is read-only and may be used concurrently by
 * multiple decode calls. The caller must not destroy it until all calls finish.
 */
ADDC_API addc_context_t* addc_context_create(void);

/**
 * Create a context with explicit parser behavior. The existing
 * addc_context_create() is equivalent to tolerant mode.
 * Returns NULL for invalid options or allocation failure.
 */
ADDC_API addc_context_t* addc_context_create_with_options(
    const addc_context_options_t* options);

/**
 * Destroy a context created by addc_context_create().
 * Passing NULL is a safe no-op.
 */
ADDC_API void addc_context_destroy(addc_context_t* ctx);

/**
 * Status codes returned by caller-owned APIs.
 * Values are stable ABI and may be extended by appending new values.
 */
typedef enum
{
    ADDC_STATUS_SUCCESS = 0,
    ADDC_STATUS_INVALID_ARGUMENT = 1,
    ADDC_STATUS_PARSE_ERROR = 2,
    ADDC_STATUS_DECODE_ERROR = 3,
    ADDC_STATUS_INSUFFICIENT_SIZE = 4,
    ADDC_STATUS_OUT_OF_RESOURCES = 5,
    ADDC_STATUS_INTERNAL_ERROR = 6,
    ADDC_STATUS_UNSUPPORTED = 7,
    ADDC_STATUS_IO_ERROR = 8
} addc_status_t;

#define ADDC_ERROR_INFO_HAS_OFFSET 0x1u
#define ADDC_ERROR_SOURCE_MAX_LENGTH 32u
#define ADDC_ERROR_MESSAGE_MAX_LENGTH 256u

/**
 * Caller-owned details for a failed safe API operation.
 *
 * Zero-initialize the structure and set struct_size to sizeof(*error_info)
 * before each call. status is the returned addc_status_t value. source names
 * the stage that rejected the input. byte_offset is meaningful only when
 * ADDC_ERROR_INFO_HAS_OFFSET is present in flags. Strings are always
 * NUL-terminated; exceptionally long messages are truncated.
 */
typedef struct
{
    uint32_t struct_size;
    uint32_t status;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t byte_offset;
    char source[ADDC_ERROR_SOURCE_MAX_LENGTH];
    char message[ADDC_ERROR_MESSAGE_MAX_LENGTH];
    uint64_t reserved[4];
} addc_error_info_t;

#define ADDC_CPER_FRU_ID_MAX_LENGTH 64
#define ADDC_CPER_FRU_TEXT_MAX_LENGTH 64
#define ADDC_CPER_ADDITIONAL_CONTEXT_MAX_LENGTH 256

/**
 * One decoded CPER event/AFID result.
 *
 * String fields are always NUL-terminated and are empty when unavailable.
 * additional_context is reserved for future event context and is currently
 * always empty.
 * The reserved fields must be zero-initialized by callers and preserve space
 * for ABI-compatible additions. If the reserved space is ever exhausted, a
 * new versioned type and function will be introduced instead of resizing this
 * structure.
 */
typedef struct
{
    char fru_id[ADDC_CPER_FRU_ID_MAX_LENGTH];
    char fru_text[ADDC_CPER_FRU_TEXT_MAX_LENGTH];
    int32_t afid;
    char additional_context[ADDC_CPER_ADDITIONAL_CONTEXT_MAX_LENGTH];
    uint64_t reserved[4];
} addc_cper_summary_entry_t;

/**
 * Decode a CPER into the full report JSON using caller-owned storage.
 *
 * Two-call contract:
 *   1. Pass json_buffer=NULL to receive the required byte count in
 *      json_buffer_size (including the trailing NUL).
 *   2. Allocate that many bytes and call again.
 *
 * If the supplied buffer is too small, the required size is returned and
 * ADDC_STATUS_INSUFFICIENT_SIZE is reported. The context must remain valid
 * for the duration of the call. Tolerantly repaired reports include top-level
 * recovery metadata. Non-fatal messages, when present, use the documented
 * top-level diagnostics array.
 */
ADDC_API addc_status_t addc_cper_decode_json(
    addc_context_t* ctx, const uint8_t* cper, size_t cper_size,
    char* json_buffer, size_t* json_buffer_size);

/**
 * Detailed form of addc_cper_decode_json(). filename is optional source
 * metadata and is copied into successful report JSON. error_info is optional;
 * when supplied it must be initialized as documented above.
 */
ADDC_API addc_status_t addc_cper_decode_json_ex(
    addc_context_t* ctx, const uint8_t* cper, size_t cper_size,
    const char* filename, char* json_buffer, size_t* json_buffer_size,
    addc_error_info_t* error_info);

/**
 * Decode a CPER into typed event/AFID entries using caller-owned storage.
 *
 * Two-call contract:
 *   1. Pass entries=NULL to receive the required entry count.
 *   2. Allocate that many zero-initialized entries and call again.
 *
 * If entry_count is too small, it is updated to the required count and
 * ADDC_STATUS_INSUFFICIENT_SIZE is returned. ADDC_UNCLASSIFIED_AFID is returned
 * when an event cannot be classified more specifically.
 */
ADDC_API addc_status_t addc_cper_get_summary(
    addc_context_t* ctx, const uint8_t* cper, size_t cper_size,
    addc_cper_summary_entry_t* entries, size_t* entry_count);

/** Detailed form of addc_cper_get_summary(); see addc_cper_decode_json_ex(). */
ADDC_API addc_status_t addc_cper_get_summary_ex(
    addc_context_t* ctx, const uint8_t* cper, size_t cper_size,
    const char* filename, addc_cper_summary_entry_t* entries,
    size_t* entry_count, addc_error_info_t* error_info);

/**
 * Parse a CPER into its intermediate JSON representation using the parse mode
 * configured on ctx. Output and failure details are caller-owned; json_buffer
 * follows the same two-call contract as addc_cper_decode_json().
 */
ADDC_API addc_status_t addc_cper_parse_json(
    addc_context_t* ctx, const uint8_t* cper, size_t cper_size,
    char* json_buffer, size_t* json_buffer_size, addc_error_info_t* error_info);

/** Return a statically allocated description of an ADDC status code. */
ADDC_API const char* addc_status_string(addc_status_t status);

/**
 * Analyze a raw CPER binary buffer and return a JSON string.
 *
 * @param buf       Pointer to the raw CPER record bytes.
 * @param len       Length of the buffer in bytes.
 * @param filename  Optional CPER filename (may be NULL).
 * @param json_out  On success, receives a malloc'd JSON string.
 *                  Caller must free with addc_free_string().
 * @param error_out On failure, receives a malloc'd error message.
 *                  Caller must free with addc_free_string().
 *                  May be NULL if the caller doesn't need the error.
 *
 * @return 0 on success, non-zero on failure.
 */
ADDC_API int addc_analyze_cper(const uint8_t* buf, size_t len,
                               const char* filename, char** json_out,
                               char** error_out);

/**
 * Same as addc_analyze_cper() but reuses a cached context
 * for better performance across multiple calls.
 */
ADDC_API int addc_analyze_cper_ctx(addc_context_t* ctx, const uint8_t* buf,
                                   size_t len, const char* filename,
                                   char** json_out, char** error_out);

/**
 * Analyze a CPER file on disk and return a JSON string.
 *
 * @param path      Path to the CPER file.
 * @param json_out  On success, receives a malloc'd JSON string.
 * @param error_out On failure, receives a malloc'd error message (may be NULL).
 *
 * @return 0 on success, non-zero on failure.
 */
ADDC_API int addc_analyze_cper_file(const char* path, char** json_out,
                                    char** error_out);

/**
 * Extract an error summary from a raw CPER binary buffer.
 *
 * Returns a JSON array containing fru_id, fru_text, afid, and
 * additional_context for every decoded event/AFID pair.
 * An event with no more specific classification is returned with
 * ADDC_UNCLASSIFIED_AFID.
 *
 * @param buf       Pointer to the raw CPER record bytes.
 * @param len       Length of the buffer in bytes.
 * @param filename  Optional CPER filename (may be NULL).
 * @param json_out  On success, receives a malloc'd JSON string.
 *                  Caller must free with addc_free_string().
 * @param error_out On failure, receives a malloc'd error message.
 *                  Caller must free with addc_free_string().
 *                  May be NULL if the caller doesn't need the error.
 *
 * @return 0 on success, non-zero on failure.
 */
ADDC_API int addc_get_error_summary(const uint8_t* buf, size_t len,
                                    const char* filename, char** json_out,
                                    char** error_out);

/**
 * Same as addc_get_error_summary() but reuses a cached context
 * for better performance across multiple calls.
 */
ADDC_API int addc_get_error_summary_ctx(addc_context_t* ctx, const uint8_t* buf,
                                        size_t len, const char* filename,
                                        char** json_out, char** error_out);

/**
 * Extract an error summary from a CPER file on disk.
 *
 * @param path      Path to the CPER file.
 * @param json_out  On success, receives a malloc'd JSON string.
 * @param error_out On failure, receives a malloc'd error message (may be NULL).
 *
 * @return 0 on success, non-zero on failure.
 */
ADDC_API int addc_get_error_summary_file(const char* path, char** json_out,
                                         char** error_out);

/**
 * Parse a raw CPER binary buffer into its intermediate representation (IR).
 * No analysis is performed — returns the raw parsed CPER structure.
 *
 * @param buf       Pointer to the raw CPER record bytes.
 * @param len       Length of the buffer in bytes.
 * @param json_out  On success, receives a malloc'd JSON string.
 *                  Caller must free with addc_free_string().
 * @param error_out On failure, receives a malloc'd error message.
 *                  Caller must free with addc_free_string().
 *                  May be NULL if the caller doesn't need the error.
 *
 * @return 0 on success, non-zero on failure.
 */
ADDC_API int addc_parse_cper(const uint8_t* buf, size_t len, char** json_out,
                             char** error_out);

/**
 * Parse a CPER file on disk into its intermediate representation (IR).
 *
 * @param path      Path to the CPER file.
 * @param json_out  On success, receives a malloc'd JSON string.
 * @param error_out On failure, receives a malloc'd error message (may be NULL).
 *
 * @return 0 on success, non-zero on failure.
 */
ADDC_API int addc_parse_cper_file(const char* path, char** json_out,
                                  char** error_out);

/**
 * Raw MCA register values for addc_mca_decode().
 *
 * Zero-initialize with ``= {0}`` or memset so that future fields
 * (appended at the end) default to 0 without breaking existing callers.
 */
typedef struct
{
    uint64_t status;
    uint64_t ipid;
    uint64_t synd;
    uint64_t addr;
    uint64_t misc0;
    uint64_t misc1;
} addc_mca_regs_t;

/**
 * Decode raw MCA registers using the fixed Base decoder set and caller-owned
 * JSON storage. json_buffer follows the standard two-call contract.
 */
ADDC_API addc_status_t addc_mca_decode_json(
    addc_context_t* ctx, const char* project, const addc_mca_regs_t* regs,
    char* json_buffer, size_t* json_buffer_size, addc_error_info_t* error_info);

/**
 * Decode raw MCA registers for a given platform/project.
 *
 * @param project   Platform name (e.g. "venice").
 * @param regs      Pointer to the register values.  Zero-init the
 *                  struct so future fields default to 0.
 * @param json_out  On success, receives a malloc'd JSON string.
 *                  Caller must free with addc_free_string().
 * @param error_out On failure, receives a malloc'd error message.
 *                  Caller must free with addc_free_string().
 *                  May be NULL if the caller doesn't need the error.
 *
 * @return 0 on success, non-zero on failure.
 */
ADDC_API int addc_mca_decode(const char* project, const addc_mca_regs_t* regs,
                             char** json_out, char** error_out);

/**
 * Same as addc_mca_decode() but reuses a cached context
 * for better performance across multiple calls.
 */
ADDC_API int addc_mca_decode_ctx(addc_context_t* ctx, const char* project,
                                 const addc_mca_regs_t* regs, char** json_out,
                                 char** error_out);

/** List canonical MCA projects supported by this build as JSON. */
ADDC_API int addc_mca_supported_projects(char** json_out, char** error_out);

/** Caller-owned form of addc_mca_supported_projects(). */
ADDC_API addc_status_t addc_mca_supported_projects_json(
    char* json_buffer, size_t* json_buffer_size, addc_error_info_t* error_info);

/**
 * Free a string returned by any addc_*() function that outputs a
 * malloc'd string (addc_analyze_cper, addc_parse_cper, etc.).
 * Passing NULL is a safe no-op.
 */
ADDC_API void addc_free_string(char* s);

/**
 * Return the library version string (statically allocated, do not free).
 */
ADDC_API const char* addc_version(void);

/** Return ADDC_API_VERSION for runtime compatibility checks. */
ADDC_API uint32_t addc_api_version(void);

/** Return ADDC_SCHEMA_VERSION (statically allocated; do not free). */
ADDC_API const char* addc_schema_version(void);

#ifdef __cplusplus
}
#endif

#endif /* ADDC_LIB_C_H */
