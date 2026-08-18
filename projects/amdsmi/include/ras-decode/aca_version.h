// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef RAS_DECODE_VERSION_H
#define RAS_DECODE_VERSION_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief ACA Decoder Library Version Information
 *
 * This header defines version constants and functions for the ACA Decoder library.
 * Version follows Semantic Versioning (SemVer) specification: MAJOR.MINOR.PATCH
 *
 * - MAJOR: Incremented for incompatible API changes
 * - MINOR: Incremented for backward-compatible functionality additions
 * - PATCH: Incremented for backward-compatible fixes
 */

/* Version Components */
#define RAS_DECODE_VERSION_MAJOR 2 /**< Major version number */
#define RAS_DECODE_VERSION_MINOR 0 /**< Minor version number */
#define RAS_DECODE_VERSION_PATCH 0 /**< Patch version number */

/* Helper macros for string concatenation */
#define RAS_DECODE_STRINGIFY(x) #x
#define RAS_DECODE_TOSTRING(x) RAS_DECODE_STRINGIFY(x)

/* Version String - dynamically constructed from components */
#define RAS_DECODE_VERSION_STRING                                            \
  RAS_DECODE_TOSTRING(RAS_DECODE_VERSION_MAJOR)                              \
  "." RAS_DECODE_TOSTRING(RAS_DECODE_VERSION_MINOR) "." RAS_DECODE_TOSTRING( \
      RAS_DECODE_VERSION_PATCH)

/**
 * @brief Structure containing version information
 */
typedef struct {
  int major;          /**< Major version number */
  int minor;          /**< Minor version number */
  int patch;          /**< Patch version number */
  const char* string; /**< Version string (e.g., "1.0.0") */
} aca_version_info_t;

/**
 * @brief Get the major version number
 * @return Major version number
 */
int aca_get_version_major(void);

/**
 * @brief Get the minor version number
 * @return Minor version number
 */
int aca_get_version_minor(void);

/**
 * @brief Get the patch version number
 * @return Patch version number
 */
int aca_get_version_patch(void);

/**
 * @brief Get the version string
 * @return Pointer to version string (e.g., "1.0.0")
 */
const char* aca_get_version_string(void);

/**
 * @brief Get complete version information
 * @return Structure containing all version information
 */
aca_version_info_t aca_get_version_info(void);

#ifdef __cplusplus
}
#endif

#endif /* RAS_DECODE_VERSION_H */
