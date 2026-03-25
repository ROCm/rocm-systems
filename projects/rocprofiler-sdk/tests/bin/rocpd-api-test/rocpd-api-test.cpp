// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * @file rocpd-api-test.cpp
 * @brief Small C++ test for rocprofiler-sdk-rocpd API: rocpd_get_version and
 *        rocpd_sql_load_schema. Run with ROCPD_SCHEMA_PATH set to the schema
 *        directory (e.g. build/share/rocprofiler-sdk-rocpd).
 */

#include <rocprofiler-sdk-rocpd/rocpd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

/** Extract schema_version from rocpd_metadata INSERT in tables schema (e.g. ("schema_version", "3")). 
    However, in new schema, the INSERT moved to the METADATA file.*/
static std::string
parse_schema_version(const char* content)
{
    if(content == nullptr) return "";
    std::string s(content);
    const std::string key("\"schema_version\"");
    size_t pos = s.find(key);
    if(pos == std::string::npos) return "";
    pos += key.size();
    pos = s.find('"', pos);
    if(pos == std::string::npos) return "";
    ++pos;
    size_t end = s.find('"', pos);
    if(end == std::string::npos) return "";
    return s.substr(pos, end - pos);
}

/** Parse object names from SQL content for a given keyword (e.g. "CREATE TABLE" or "CREATE VIEW"). */
static void
parse_sql_names(const char* content, const char* sql_keyword, std::vector<std::string>& out_names)
{
    if(content == nullptr) return;
    std::string s(content);
    const std::string key(sql_keyword);
    for(size_t pos = 0; (pos = s.find(key, pos)) != std::string::npos; pos += key.size())
    {
        pos += key.size();
        /* Skip past "IF NOT EXISTS" and whitespace until we hit the quoted/backticked name */
        while(pos < s.size() && s[pos] != '"' && s[pos] != '`')
            ++pos;
        if(pos >= s.size()) break;
        char quote = s[pos];
        size_t start = pos + 1;
        size_t end   = s.find(quote, start);
        if(end == std::string::npos) break;
        out_names.push_back(s.substr(start, end - start));
    }
}

struct _callback_data
{
    std::vector<std::string>* names;
    std::string*              schema_version;
};

static void
tables_callback(rocpd_sql_engine_t                        /*engine*/,
                rocpd_sql_schema_kind_t                   /*kind*/,
                rocpd_sql_options_t                       /*options*/,
                rocpd_version_triplet_t                   /*schema_version_triplet*/,
                const rocpd_sql_schema_jinja_variables_t* /*variables*/,
                const char*                               /*schema_path*/,
                const char*                               schema_content,
                void*                                     user_data)
{
    auto* data = static_cast<_callback_data*>(user_data);
    if(schema_content == nullptr || data == nullptr || data->names == nullptr) return;
    std::string content(schema_content);
    if(content.find("CREATE TABLE") == std::string::npos ||
       content.find("rocpd_metadata") == std::string::npos)
        return;
    parse_sql_names(schema_content, "CREATE TABLE", *data->names);
    if(data->schema_version != nullptr)
        *data->schema_version = parse_schema_version(schema_content);
}

static void
views_callback(rocpd_sql_engine_t                        /*engine*/,
               rocpd_sql_schema_kind_t                   kind,
               rocpd_sql_options_t                       /*options*/,
               rocpd_version_triplet_t                   /*schema_version_triplet*/,
               const rocpd_sql_schema_jinja_variables_t* /*variables*/,
               const char*                               /*schema_path*/,
               const char*                               schema_content,
               void*                                     user_data)
{
    auto* data = static_cast<_callback_data*>(user_data);
    if(schema_content == nullptr || data == nullptr || data->names == nullptr) return;
    parse_sql_names(schema_content, "CREATE VIEW", *data->names);
    if(kind == ROCPD_SQL_SCHEMA_ROCPD_METADATA && data->schema_version != nullptr && data->schema_version->empty())
    {
        *data->schema_version = parse_schema_version(schema_content);
    }
}

int
_load_schema(rocpd_version_triplet_t requested_version)
{
    rocpd_status_t status = ROCPD_STATUS_SUCCESS;
    std::vector<std::string> table_names;
    std::string             schema_version;
    _callback_data tables_data{&table_names, &schema_version};

    rocpd_sql_schema_jinja_variables_t variables{};
    variables.size = sizeof(rocpd_sql_schema_jinja_variables_t);
    variables.uuid = "";
    variables.guid = "";

    status = rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3,
                                   ROCPD_SQL_SCHEMA_ROCPD_TABLES,
                                   ROCPD_SQL_OPTIONS_SQLITE3_PRAGMA_FOREIGN_KEYS,
                                   requested_version,
                                   &variables,
                                   tables_callback,
                                   nullptr,
                                   0,
                                   &tables_data);
    if(status != ROCPD_STATUS_SUCCESS)
    {
        std::cerr << "rocpd-api-test: rocpd_sql_load_schema(tables) failed: "
                  << rocpd_get_status_name(status)
                  << " - " << (rocpd_get_status_string(status) ? rocpd_get_status_string(status)
                                                              : "unknown")
                  << "\n";
        return EXIT_FAILURE;
    }
    if(table_names.empty())
    {
        std::cerr << "rocpd-api-test: no tables found in schema\n";
        return EXIT_FAILURE;
    }

    std::vector<std::string> view_names;
    const rocpd_sql_schema_kind_t view_kinds[] = {
        ROCPD_SQL_SCHEMA_ROCPD_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_DATA_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_SUMMARY_VIEWS,
        ROCPD_SQL_SCHEMA_ROCPD_METADATA,
    };
    _callback_data views_data{&view_names, &schema_version};
    for(rocpd_sql_schema_kind_t kind : view_kinds)
    {
        status = rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3,
                                      kind,
                                      ROCPD_SQL_OPTIONS_NONE,
                                      requested_version,
                                      &variables,
                                      views_callback,
                                      nullptr,
                                      0,
                                      &views_data);
        if(status != ROCPD_STATUS_SUCCESS)
        {
            std::cerr << "rocpd-api-test: rocpd_sql_load_schema(views) failed: "
                      << rocpd_get_status_name(status) << "\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "  rocpd-api-test: rocpd_sql_load_schema OK\n";
    std::cout << "  Schema version: " << (schema_version.empty() ? "unknown" : schema_version)
              << "\n";
    std::cout << "  Number of tables: " << table_names.size() << "\n";
    std::cout << "  Tables:";
    for(const auto& name : table_names)
        std::cout << " " << name;
    std::cout << "\n\n";
    std::cout << "  Number of views: " << view_names.size() << "\n";
    std::cout << "  Views:";
    for(const auto& name : view_names)
        std::cout << " " << name;
    std::cout << "\n\n";
    return EXIT_SUCCESS;
}


int
main()
{
    uint32_t major = 0, minor = 0, patch = 0;
    rocpd_status_t status = rocpd_get_version(&major, &minor, &patch);
    if(status != ROCPD_STATUS_SUCCESS)
    {
        std::cerr << "rocpd-api-test: rocpd_get_version failed: " << rocpd_get_status_name(status)
                  << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "rocpd-api-test: rocpd_get_version OK ("
              << major << "." << minor << "." << patch << ")\n";

    rocpd_sql_schema_versions_list_t schema_versions_list;
    std::vector<rocpd_version_triplet_t> local_list_of_schema_versions;

    status = rocpd_sql_list_schema_versions(ROCPD_SQL_ENGINE_SQLITE3,
                                            nullptr,
                                            0,
                                            &schema_versions_list);
    if(status != ROCPD_STATUS_SUCCESS)
    {
        std::cerr << "rocpd-api-test: rocpd_sql_list_schema_versions failed: "
                  << rocpd_get_status_name(status) << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "rocpd-api-test: rocpd_sql_list_schema_versions OK ("
              << schema_versions_list.count << " versions)\n";
    for(uint64_t i = 0; i < schema_versions_list.count; ++i)
    {
        std::cout << "  Version " << i << ": " << schema_versions_list.versions[i].major << "."
                  << schema_versions_list.versions[i].minor << "."
                  << schema_versions_list.versions[i].patch << "\n";
        local_list_of_schema_versions.push_back(schema_versions_list.versions[i]);
    }
    rocpd_sql_free_schema_versions_list(&schema_versions_list);

    // first, load the latest schema version
    std::cout << "\nLoading latest schema version (requesting 0.0.0)...\n";
    rocpd_version_triplet_t latest_schema_version = {0, 0, 0};
    _load_schema(latest_schema_version);

    // then all supported schema versions
    std::cout << "\nNow iterating over the entire list of schema versions:\n";
    for(const auto& version : local_list_of_schema_versions)
    {
        std::cout << "  For schema version: " << version.major << "." << version.minor << "." << version.patch << ", load schema...\n";
        _load_schema(version);
    }

    return EXIT_SUCCESS;
}
