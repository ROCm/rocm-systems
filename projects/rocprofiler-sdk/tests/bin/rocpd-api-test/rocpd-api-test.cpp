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

/** Extract schema_version from rocpd_metadata INSERT in tables schema (e.g. ("schema_version", "3")). */
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

/** Parse object names from SQL content for a given DDL keyword (e.g. "CREATE TABLE" or "CREATE VIEW"). */
static void
parse_ddl_names(const char* content, const char* ddl_keyword, std::vector<std::string>& out_names)
{
    if(content == nullptr) return;
    std::string s(content);
    const std::string key(ddl_keyword);
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

struct tables_callback_data
{
    std::vector<std::string>* table_names;
    std::string*              schema_version;
};

static void
tables_callback(rocpd_sql_engine_t                        /*engine*/,
                rocpd_sql_schema_kind_t                   /*kind*/,
                rocpd_sql_options_t                       /*options*/,
                const rocpd_sql_schema_jinja_variables_t* /*variables*/,
                const char*                               /*schema_path*/,
                const char*                               schema_content,
                void*                                    user_data)
{
    auto* data = static_cast<tables_callback_data*>(user_data);
    if(schema_content == nullptr || data == nullptr || data->table_names == nullptr) return;
    std::string content(schema_content);
    if(content.find("CREATE TABLE") == std::string::npos ||
       content.find("rocpd_metadata") == std::string::npos)
        return;
    parse_ddl_names(schema_content, "CREATE TABLE", *data->table_names);
    if(data->schema_version != nullptr)
        *data->schema_version = parse_schema_version(schema_content);
}

static void
views_callback(rocpd_sql_engine_t                        /*engine*/,
               rocpd_sql_schema_kind_t                   /*kind*/,
               rocpd_sql_options_t                       /*options*/,
               const rocpd_sql_schema_jinja_variables_t* /*variables*/,
               const char*                               /*schema_path*/,
               const char*                               schema_content,
               void*                                    user_data)
{
    auto* view_names = static_cast<std::vector<std::string>*>(user_data);
    if(schema_content == nullptr || view_names == nullptr) return;
    parse_ddl_names(schema_content, "CREATE VIEW", *view_names);
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

    rocpd_sql_schema_jinja_variables_t variables{};
    variables.size = sizeof(rocpd_sql_schema_jinja_variables_t);
    variables.uuid = "";
    variables.guid = "";

    std::vector<std::string> table_names;
    std::string             schema_version;
    tables_callback_data    tables_data{&table_names, &schema_version};
    status = rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3,
                                   ROCPD_SQL_SCHEMA_ROCPD_TABLES,
                                   ROCPD_SQL_OPTIONS_SQLITE3_PRAGMA_FOREIGN_KEYS,
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
        ROCPD_SQL_SCHEMA_ROCPD_MARKER_VIEWS,
    };
    for(rocpd_sql_schema_kind_t kind : view_kinds)
    {
        status = rocpd_sql_load_schema(ROCPD_SQL_ENGINE_SQLITE3,
                                      kind,
                                      ROCPD_SQL_OPTIONS_NONE,
                                      &variables,
                                      views_callback,
                                      nullptr,
                                      0,
                                      &view_names);
        if(status != ROCPD_STATUS_SUCCESS)
        {
            std::cerr << "rocpd-api-test: rocpd_sql_load_schema(views) failed: "
                      << rocpd_get_status_name(status) << "\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "rocpd-api-test: rocpd_sql_load_schema OK\n";
    std::cout << "  Schema version: " << (schema_version.empty() ? "unknown" : schema_version)
              << "\n";
    std::cout << "  Number of tables: " << table_names.size() << "\n";
    std::cout << "  Tables:";
    for(const auto& name : table_names)
        std::cout << " " << name;
    std::cout << "\n";
    std::cout << "  Number of views: " << view_names.size() << "\n";
    std::cout << "  Views:";
    for(const auto& name : view_names)
        std::cout << " " << name;
    std::cout << "\n";
    return EXIT_SUCCESS;
}
