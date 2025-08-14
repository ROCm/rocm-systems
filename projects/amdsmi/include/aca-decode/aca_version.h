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

#ifndef ACA_VERSION_H
#define ACA_VERSION_H

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief ACA Decoder Library Version Information
 *
 * This header defines version constants and functions for the ACA Decoder library.
 * Version follows Semantic Versioning (SemVer) specification: MAJOR.MINOR.PATCH
 *
 * - MAJOR: Incremented for incompatible API changes
 * - MINOR: Incremented for backward-compatible functionality additions
 * - PATCH: Incremented for backward-compatible bug fixes
 */

/* Version Components */
#define ACA_VERSION_MAJOR 1 /**< Major version number */
#define ACA_VERSION_MINOR 0 /**< Minor version number */
#define ACA_VERSION_PATCH 0 /**< Patch version number */

/* Version String */
#define ACA_VERSION_STRING "1.0.0"

    /**
     * @brief Structure containing version information
     */
    typedef struct
    {
        int major;          /**< Major version number */
        int minor;          /**< Minor version number */
        int patch;          /**< Patch version number */
        const char *string; /**< Version string (e.g., "1.0.0") */
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
    const char *aca_get_version_string(void);

    /**
     * @brief Get complete version information
     * @return Structure containing all version information
     */
    aca_version_info_t aca_get_version_info(void);

#ifdef __cplusplus
}
#endif

#endif /* ACA_VERSION_H */
