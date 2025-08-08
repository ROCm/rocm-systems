// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler-sdk-rocpd/defines.h>
#include <rocprofiler-sdk-rocpd/types.h>

#include <stdint.h>

ROCPD_EXTERN_C_INIT

/**
 * @defgroup SQL rocPD SQL Utilities
 * @brief Functions for reading rocpd SQL schema
 * @{
 */

/**
 * @brief (experimental) Supported SQL engines. Initial support is for SQLite3, other engines such
 * as MySQL may be added
 */
typedef enum ROCPD_EXPERIMENTAL rocpd_sql_engine_t
{
    ROCPD_SQL_ENGINE_NONE = 0,
    ROCPD_SQL_ENGINE_SQLITE3,  ///< Ensure compatibility with SQLite3
    ROCPD_SQL_ENGINE_LAST,
} rocpd_sql_engine_t;

/**
 * @brief (experimental) Supported SQL schema kinds
 */
typedef enum ROCPD_EXPERIMENTAL rocpd_sql_schema_kind_t
{
    ROCPD_SQL_SCHEMA_NONE = 0,
    ROCPD_SQL_SCHEMA_ROCPD_TABLES,
    ROCPD_SQL_SCHEMA_ROCPD_INDEXES,
    ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
    ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS,
    ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS,
    ROCPD_SQL_SCHEMA_ROCPD_MARKER_VIEWS,
    ROCPD_SQL_SCHEMA_LAST,
} rocpd_sql_schema_kind_t;

/**
 * @brief (experimental) Supported SQL options
 */
typedef enum ROCPD_EXPERIMENTAL rocpd_sql_options_t
{
    ROCPD_SQL_OPTIONS_NONE                        = 0,
    ROCPD_SQL_OPTIONS_PERSIST_STRINGS             = 0x01,  ///< do not delete strings
    ROCPD_SQL_OPTIONS_SQLITE3_PRAGMA_FOREIGN_KEYS = 0x02,  ///< enable SQLite3 foreign keys
    ROCPD_SQL_OPTIONS_LAST                        = 0xFFFFFFFF,
} rocpd_sql_options_t;

/**
 * @brief (experimental) Schema jinja variable substitution values. In general, the struct member
 * variable replaces jinja variables of the same name, e.g. the struct member variable `uuid`
 * replaces `{{uuid}}` in the schema definition.
 *
 * When the schema variable `{{uuid}}` is a non-null and non-empty string, it
 * will be prefixed with a leading underscore (`_`) and all hyphens will be replaced with
 * underscores. This is because this variable is used in SQL table names. The leading
 * underscore improves the readability of the table name and the replacement of hyphens with
 * underscores is reduces problems.
 */
typedef struct ROCPD_EXPERIMENTAL rocpd_sql_schema_jinja_variables_t
{
    uint64_t    size;  ///< Size of this struct (minus reserved padding)
    const char* uuid;  ///< Substitution for `{{uuid}}` (non-empty adds leading underscore)
    const char* guid;  ///< Substitution for `{{guid}}`
} rocpd_sql_schema_jinja_variables_t;

/**
 * @brief (experimental) Callback providing the schema content after variable substitution.
 *
 * @param [in] engine Schema conforms to this SQL database engine
 * @param [in] kind Schema category
 * @param [in] options Schema options included in content
 * @param [in] variables Jinja variables which were substituted
 * @param [in] schema_path Filesystem path to base schema file
 * @param [in] schema_content SQL schema content is pass to database
 * @param [in] user_data User provided data
 */
ROCPD_EXPERIMENTAL typedef void (*rocpd_sql_load_schema_cb_t)(
    rocpd_sql_engine_t                        engine,
    rocpd_sql_schema_kind_t                   kind,
    rocpd_sql_options_t                       options,
    const rocpd_sql_schema_jinja_variables_t* variables,
    const char*                               schema_path,
    const char*                               schema_content,
    void*                                     user_data);

/**
 * @brief (experimental) Invoke the callback which provides the schema kind definition for the given
 * SQL engine + the addition of the requested options + jinja variable substitution.
 *
 * @param [in] engine SQL schema contents should be compatible with this SQL database engine
 * @param [in] kind SQL schema kind (tables, indexes, views, etc.)
 * @param [in] options Options for the callback and schema
 * @param [in] variables Defines variables for jinja substitution. If nullptr, no jinja variables
 * will be substituted. For each member variable that is a nullptr, jinja for that member variable
 * will not be substituted. To replace jinja variables with empty strings, assign all member
 * variables to empty string.
 * @param [in] callback Callback function which provides the schema contents
 * @param [in] schema_path_hints Suggests for where to find the schema templates
 * @param [in] num_schema_path_hints Number of schema path hints
 * @param [in] user_data Pointer to pass back into callback
 * @return ::rocpd_status_t
 * @retval ROCPD_STATUS_SUCCESS if all parameter specifications were applied and callback
 * successfully invoked
 * @retval ROCPD_STATUS_ERROR_INVALID_ARGUMENT Invalid SQL engine
 * @retval ROCPD_STATUS_ERROR_NOT_AVAILABLE Schema file could not be found
 * @retval ROCPD_STATUS_ERROR_PERMISSION_DENIED Schema file could not be found
 * @retval ROCPD_STATUS_ERROR_KIND_NOT_FOUND Invalid SQL schema kind
 * @retval ROCPD_STATUS_ERROR There was some issue with parameters
 * @retval ROCPD_STATUS_ERROR_INVALID_ARGUMENT ::rocpd_sql_schema_jinja_variables_t size parameter
 * not set
 */
ROCPD_EXPERIMENTAL rocpd_status_t
rocpd_sql_load_schema(rocpd_sql_engine_t                        engine,
                      rocpd_sql_schema_kind_t                   kind,
                      rocpd_sql_options_t                       options,
                      const rocpd_sql_schema_jinja_variables_t* variables,
                      rocpd_sql_load_schema_cb_t                callback,
                      const char**                              schema_path_hints,
                      uint64_t                                  num_schema_path_hints,
                      void* user_data) ROCPD_API ROCPD_NONNULL(5);

/** @} */

ROCPD_EXTERN_C_FINI
