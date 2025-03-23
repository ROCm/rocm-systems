// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-register/version.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t (*rocprofiler_register_import_func_t)(void);

typedef struct
{
    uint64_t handle;
} rocprofiler_register_library_indentifier_t;

/// @enum rocprofiler_register_error_code_t
/// @brief enumeration of error codes
///
/// @var ROCP_REG_SUCCESS
/// @brief Successful call to rocprofiler-register
///
/// @var ROCP_REG_NO_TOOLS
/// @brief API table was not passed to rocprofiler because no rocprofiler tool
/// configuration symbols were requested
///
/// @var ROCP_REG_DEADLOCK
/// @brief The thread that is currently communicating the API tables to rocprofiler
/// re-entered \ref rocprofiler_register_library_api_table
///
/// @var ROCP_REG_BAD_API_TABLE_LENGTH
/// @brief Library provided an invalid API table length (likely zero)
///
/// @var ROCP_REG_UNSUPPORTED_API
/// @brief Library is providing an API table name that is not supported
///
/// @var ROCP_REG_ROCPROFILER_ERROR
/// @brief Rocprofiler returned an error when passing the API tables
///
/// @var ROCP_REG_EXCESS_API_INSTANCES
/// @brief The same API has been registered too many times
///
typedef enum rocprofiler_register_error_code_t  // NOLINT(performance-enum-size)
{
    ROCP_REG_SUCCESS = 0,
    ROCP_REG_NO_TOOLS,
    ROCP_REG_DEADLOCK,
    ROCP_REG_BAD_API_TABLE_LENGTH,
    ROCP_REG_UNSUPPORTED_API,
    ROCP_REG_INVALID_API_ADDRESS,
    ROCP_REG_ROCPROFILER_ERROR,
    ROCP_REG_EXCESS_API_INSTANCES,
    ROCP_REG_ERROR_CODE_END,
} rocprofiler_register_error_code_t;

/// @fn rocprofiler_register_error_code_t rocprofiler_register_library_api_table(
///     const char*                                 lib_name,
///     rocprofiler_register_import_func_t          import_func,
///     uint32_t                                    lib_version,
///     void*                                       api_tables,
///     uint64_t                                    api_table_length,
///     rocprofiler_register_library_indentifier_t* register_id)
/// @param[in] lib_name string identifier for the library registering the API table
/// @param[in] import_func function pointer to rocprofiler_register_import_<lib_name>
/// symbol
/// @param[in] lib_version lib version in form of (10000 * major ) + (100 * minor) + patch
/// @param[in] api_tables pointer to one or more API table pointers
/// @param[in] api_table_length number of API tables that are being passed in. Must be > 0
/// @param[out] register_id a unique identifier for this library encoding the API category
/// and the priority in the event that multiple libraries register with the same name
/// @returns value of type @ref rocprofiler_register_error_code_t
///
/// @brief primary function call that needs to be performed when the API table of the
/// library is initialized.
///
rocprofiler_register_error_code_t
rocprofiler_register_library_api_table(
    const char*                                 lib_name,
    rocprofiler_register_import_func_t          import_func,
    uint32_t                                    lib_version,
    void**                                      api_tables,
    uint64_t                                    api_table_length,
    rocprofiler_register_library_indentifier_t* register_id)
    ROCPROFILER_REGISTER_ATTRIBUTE(nonnull(4, 6)) ROCPROFILER_REGISTER_PUBLIC_API;

const char* rocprofiler_register_error_string(rocprofiler_register_error_code_t)
    ROCPROFILER_REGISTER_PUBLIC_API;

/// @brief Struct containing the information about the libraries which have registered
/// with rocprofiler-register. @see rocprofiler_register_iterate_registration_info
typedef struct rocprofiler_register_registration_info_t
{
    size_t      size;              ///< in case of future extensions
    const char* common_name;       ///< name of the library
    uint32_t    lib_version;       ///< version
    uint64_t    api_table_length;  ///< number of API tables
} rocprofiler_register_registration_info_t;

/**
 * @brief Callback function for iterating over the libraries which have registered
 * with rocprofiler-register. @see rocprofiler_register_iterate_registration_info
 *
 * @param [in] info Pointer to library registration instance. Invokee should make a copy
 * for reference outside of callback.
 * @param [in] data User data passed to ::rocprofiler_register_iterate_registration_info
 * @return int
 * @retval 0 If zero is returned from callback, rocprofiler-register will continue to next
 * registration info, if one exists
 * @retval -1 If -1 (or any value != 0) is returned from callback, rocprofiler-register
 * will cease to iterate over the remaining registration info, if any exists
 */
typedef int (*rocprofiler_register_registration_info_cb_t)(
    rocprofiler_register_registration_info_t* info,
    void*                                     data);

/**
 * @brief Iterates over all the (valid) libraries which registered their API tables with
 * rocprofiler-register. Any libraries which do not have an accepted common name, have an
 * invalid import function address (in secure mode), or have registered too many instances
 * are not reported by this function.
 *
 * @param [in] callback Callback function to invoke for each valid registered library
 * @param [in] data User data to pass to the callback function
 * @return ::rocprofiler_register_error_code_t
 * @retval ::ROCP_REG_SUCCESS Always returned
 */
rocprofiler_register_error_code_t
rocprofiler_register_iterate_registration_info(
    rocprofiler_register_registration_info_cb_t callback,
    void*                                       data)
    ROCPROFILER_REGISTER_ATTRIBUTE(nonnull(1)) ROCPROFILER_REGISTER_PUBLIC_API;

#ifdef __cplusplus
}
#endif

/// @def ROCPROFILER_REGISTER_COMPUTE_VERSION_1
/// @param[in] MAJOR_VERSION major version of library (integral)
/// @brief Helper macro for users to generate versioning int expected by
/// rocprofiler-register when the library only maintains a major version number
#define ROCPROFILER_REGISTER_COMPUTE_VERSION_1(MAJOR_VERSION) (10000 * MAJOR_VERSION)

/// @def ROCPROFILER_REGISTER_COMPUTE_VERSION_
/// @param[in] MAJOR_VERSION major version of library (integral)
/// @param[in] MINOR_VERSION minor version of library (integral)
/// @brief Helper macro for users to generate versioning int expected by
/// rocprofiler-register when the library only maintains a major and minor version number
#define ROCPROFILER_REGISTER_COMPUTE_VERSION_2(MAJOR_VERSION, MINOR_VERSION)             \
    (ROCPROFILER_REGISTER_COMPUTE_VERSION_1(MAJOR_VERSION) + (100 * MINOR_VERSION))

/// @def ROCPROFILER_REGISTER_COMPUTE_VERSION_3
/// @param[in] MAJOR_VERSION major version of library (integral)
/// @param[in] MINOR_VERSION minor version of library (integral)
/// @param[in] PATCH_VERSION patch version of library (integral)
/// @brief Helper macro for users to generate versioning int expected by
/// rocprofiler-register when the library maintains a major, minor, and patch version
/// numbers
#define ROCPROFILER_REGISTER_COMPUTE_VERSION_3(                                          \
    MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)                                         \
    (ROCPROFILER_REGISTER_COMPUTE_VERSION_2(MAJOR_VERSION, MINOR_VERSION) +              \
     (PATCH_VERSION))

/// @def ROCPROFILER_REGISTER_COMPUTE_VERSION_4
/// @param[in] MAJOR_VERSION major version of library (integral)
/// @param[in] MINOR_VERSION minor version of library (integral)
/// @param[in] PATCH_VERSION patch version of library (integral)
/// @param[in] TWEAK_VERSION tweak version of library which will be ignored
/// @brief Helper macro for users to generate versioning int expected by
/// rocprofiler-register when the library maintains a major, minor, patch, and tweak
/// version numbers
#define ROCPROFILER_REGISTER_COMPUTE_VERSION_3(                                          \
    MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)                                         \
    (ROCPROFILER_REGISTER_COMPUTE_VERSION_2(MAJOR_VERSION, MINOR_VERSION) +              \
     (PATCH_VERSION))

/// @def ROCPROFILER_REGISTER_IMPORT_FUNC(NAME)
/// @param[in] NAME the string identifier for the library
/// @brief Helper macro for retrieving the name of the import function which is used by
/// rocprofiler-register to identify the location and priority of the registering library
#define ROCPROFILER_REGISTER_IMPORT_FUNC(NAME)                                           \
    ROCPROFILER_REGISTER_PP_COMBINE(rocprofiler_register_import_, NAME)

/// @def ROCPROFILER_REGISTER_DEFINE_IMPORT(NAME, VERSION)
/// @param[in] NAME the unquoted string identifier for the library, e.g. hip
/// @brief Helper macro for declaring a visible import symbol for rocprofiler-register
/// which returns the version.
///
#ifdef __cplusplus
#    define ROCPROFILER_REGISTER_DEFINE_IMPORT(NAME, VERSION)                            \
        extern "C" {                                                                     \
        uint32_t ROCPROFILER_REGISTER_IMPORT_FUNC(NAME)()                                \
            ROCPROFILER_REGISTER_ATTRIBUTE(visibility("default"));                       \
                                                                                         \
        uint32_t ROCPROFILER_REGISTER_IMPORT_FUNC(NAME)() { return VERSION; }            \
        }
#else
#    define ROCPROFILER_REGISTER_DEFINE_IMPORT(NAME, VERSION)                            \
        uint32_t ROCPROFILER_REGISTER_IMPORT_FUNC(NAME)(void)                            \
            ROCPROFILER_REGISTER_ATTRIBUTE(visibility("default"));                       \
                                                                                         \
        uint32_t ROCPROFILER_REGISTER_IMPORT_FUNC(NAME)(void) { return VERSION; }
#endif
