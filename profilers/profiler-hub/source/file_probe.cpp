// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/cpp/reader.hpp"

#include "data_storage/backends/sqlite_api_policy.hpp"
#include "data_storage/schema_version_text.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace profiler_hub
{

using reader_types::file_probe_result_t;
using reader_types::trace_schema_t;

namespace
{

using data_storage::sqlite_api_policy;

constexpr std::string_view k_sqlite_magic{ "SQLite format 3" };
constexpr std::string_view k_metadata_prefix{ "rocpd_metadata_" };

enum class file_kind_t
{
    sqlite,
};

struct schema_support_t
{
    file_kind_t    file_kind;
    trace_schema_t trace_schema;
    bool (*supports_version)(version_t version);
};

file_probe_result_t
result_of(ph_result_t    status,
          std::string    message,
          trace_schema_t schema  = trace_schema_t::unknown,
          version_t      version = {},
          std::string    uuid    = {})
{
    file_probe_result_t result;
    result.status         = status;
    result.trace_schema   = schema;
    result.schema_version = version;
    result.uuid           = std::move(uuid);
    result.message        = std::move(message);
    return result;
}

bool
is_identifier(std::string_view text)
{
    if(text.empty()) return false;
    for(const char ch : text)
    {
        const bool digit = ch >= '0' && ch <= '9';
        const bool upper = ch >= 'A' && ch <= 'Z';
        const bool lower = ch >= 'a' && ch <= 'z';
        if(!digit && !upper && !lower && ch != '_') return false;
    }
    return true;
}

bool
sqlite_magic_matches(const std::string& path)
{
    std::ifstream                           input{ path, std::ios::binary };
    std::array<char, k_sqlite_magic.size()> header{};
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if(input.gcount() != static_cast<std::streamsize>(header.size())) return false;
    return std::string_view{ header.data(), header.size() } == k_sqlite_magic;
}

class sqlite_probe_connection
{
public:
    explicit sqlite_probe_connection(const std::string& path)
    {
        const int rc = sqlite_api_policy::open_readonly(path.c_str(), &m_db);
        if(rc != sqlite_api_policy::result_ok)
        {
            m_error = m_db != nullptr ? sqlite_api_policy::errmsg(m_db)
                                      : sqlite_api_policy::errstr(rc);
            close();
        }
    }

    ~sqlite_probe_connection() { close(); }

    sqlite_probe_connection(const sqlite_probe_connection&)            = delete;
    sqlite_probe_connection& operator=(const sqlite_probe_connection&) = delete;

    [[nodiscard]] bool                          ok() const { return m_db != nullptr; }
    [[nodiscard]] std::string                   error() const { return m_error; }
    [[nodiscard]] sqlite_api_policy::database_t get() const { return m_db; }

private:
    void close()
    {
        if(m_db == nullptr) return;
        sqlite_api_policy::close(m_db);
        m_db = nullptr;
    }

    sqlite_api_policy::database_t m_db{ nullptr };
    std::string                   m_error;
};

class sqlite_statement
{
public:
    sqlite_statement(sqlite_api_policy::database_t db, const std::string& sql)
    {
        if(sqlite_api_policy::prepare(db, sql.c_str(), &m_stmt) !=
           sqlite_api_policy::result_ok)
        {
            m_error = sqlite_api_policy::errmsg(db);
        }
    }

    ~sqlite_statement()
    {
        if(m_stmt != nullptr) sqlite_api_policy::finalize(m_stmt);
    }

    sqlite_statement(const sqlite_statement&)            = delete;
    sqlite_statement& operator=(const sqlite_statement&) = delete;

    [[nodiscard]] bool                           ok() const { return m_stmt != nullptr; }
    [[nodiscard]] std::string                    error() const { return m_error; }
    [[nodiscard]] sqlite_api_policy::statement_t get() const { return m_stmt; }

private:
    sqlite_api_policy::statement_t m_stmt{ nullptr };
    std::string                    m_error;
};

bool
supports_rocpd_schema_v3(version_t version)
{
    return version.major == 3;
}

const std::array<schema_support_t, 1>&
supported_schemas()
{
    static const std::array<schema_support_t, 1> schemas{ {
        { file_kind_t::sqlite, trace_schema_t::rocpd, supports_rocpd_schema_v3 },
    } };
    return schemas;
}

const schema_support_t*
find_support(file_kind_t kind, trace_schema_t schema, version_t version)
{
    for(const schema_support_t& entry : supported_schemas())
    {
        if(entry.file_kind == kind && entry.trace_schema == schema &&
           entry.supports_version(version))
        {
            return &entry;
        }
    }
    return nullptr;
}

file_probe_result_t
probe_sqlite(file_kind_t kind, const std::string& path)
{
    sqlite_probe_connection connection{ path };
    if(!connection.ok())
    {
        return result_of(PH_RESULT_INVALID_FILE,
                         "SQLite database could not be opened read-only: " +
                             connection.error());
    }

    sqlite_statement tables{ connection.get(),
                             "SELECT name FROM sqlite_master WHERE type = 'table' AND "
                             "name LIKE 'rocpd_metadata\\_%' ESCAPE '\\'" };
    if(!tables.ok())
    {
        return result_of(PH_RESULT_INVALID_FILE,
                         "rocpd metadata tables could not be listed: " + tables.error(),
                         trace_schema_t::rocpd);
    }

    std::vector<std::string> names;
    int                      rc = sqlite_api_policy::result_done;
    while((rc = sqlite_api_policy::step(tables.get())) == sqlite_api_policy::result_row)
    {
        names.push_back(sqlite_api_policy::column_text(tables.get(), 0));
    }
    if(rc != sqlite_api_policy::result_done)
    {
        return result_of(PH_RESULT_INVALID_FILE,
                         "rocpd metadata tables could not be read: " +
                             sqlite_api_policy::errmsg(connection.get()),
                         trace_schema_t::rocpd);
    }

    if(names.empty())
    {
        return result_of(PH_RESULT_UNSUPPORTED_FILE,
                         "SQLite database has no rocpd_metadata table");
    }
    if(names.size() != 1)
    {
        return result_of(PH_RESULT_INVALID_FILE,
                         "SQLite database has more than one rocpd_metadata table",
                         trace_schema_t::rocpd);
    }

    const std::string& table = names.front();
    if(table.rfind(k_metadata_prefix, 0) != 0)
    {
        return result_of(PH_RESULT_UNSUPPORTED_FILE,
                         "rocpd metadata table name is malformed",
                         trace_schema_t::rocpd);
    }
    const std::string uuid = table.substr(k_metadata_prefix.size());
    if(!is_identifier(uuid))
    {
        return result_of(PH_RESULT_UNSUPPORTED_FILE,
                         "rocpd metadata uuid is not an identifier",
                         trace_schema_t::rocpd);
    }

    const std::string version_sql =
        "SELECT tag, value FROM \"" + table +
        "\" WHERE tag IN ('schema_version', 'schema_version_major', "
        "'schema_version_minor', 'schema_version_patch')";
    sqlite_statement versions{ connection.get(), version_sql };
    if(!versions.ok())
    {
        return result_of(PH_RESULT_INVALID_FILE,
                         "rocpd schema version could not be read: " + versions.error(),
                         trace_schema_t::rocpd,
                         {},
                         uuid);
    }

    std::optional<std::string> version_text;
    std::optional<std::string> major;
    std::optional<std::string> minor;
    std::optional<std::string> patch;
    while((rc = sqlite_api_policy::step(versions.get())) == sqlite_api_policy::result_row)
    {
        const std::string tag   = sqlite_api_policy::column_text(versions.get(), 0);
        const std::string value = sqlite_api_policy::column_text(versions.get(), 1);
        if(tag == "schema_version")
            version_text = value;
        else if(tag == "schema_version_major")
            major = value;
        else if(tag == "schema_version_minor")
            minor = value;
        else if(tag == "schema_version_patch")
            patch = value;
    }

    const auto text_view = [](const std::optional<std::string>& text) {
        return text.has_value() ? std::optional<std::string_view>{ *text } : std::nullopt;
    };
    const data_storage::schema_version_resolution_t resolved =
        data_storage::resolve_schema_version({ text_view(version_text),
                                               text_view(major),
                                               text_view(minor),
                                               text_view(patch) });
    if(resolved.status == data_storage::schema_version_status_t::component_not_integer)
    {
        return result_of(PH_RESULT_UNSUPPORTED_FILE,
                         "rocpd schema version is not an integer: " +
                             std::string{ data_storage::schema_version_component_name(
                                 resolved.component) },
                         trace_schema_t::rocpd,
                         {},
                         uuid);
    }
    if(rc != sqlite_api_policy::result_done ||
       resolved.status != data_storage::schema_version_status_t::resolved)
    {
        return result_of(PH_RESULT_UNSUPPORTED_FILE,
                         resolved.status ==
                                 data_storage::schema_version_status_t::text_malformed
                             ? "rocpd schema_version text is malformed"
                             : "rocpd schema version is incomplete",
                         trace_schema_t::rocpd,
                         {},
                         uuid);
    }
    if(find_support(kind, trace_schema_t::rocpd, resolved.version) == nullptr)
    {
        return result_of(PH_RESULT_UNSUPPORTED_SCHEMA_VERSION,
                         "rocpd schema version is not supported",
                         trace_schema_t::rocpd,
                         resolved.version,
                         uuid);
    }

    return result_of(
        PH_RESULT_SUCCESS, "", trace_schema_t::rocpd, resolved.version, uuid);
}

struct format_detector_t
{
    file_kind_t kind;
    bool (*matches)(const std::string& path);
    file_probe_result_t (*probe)(file_kind_t kind, const std::string& path);
};

const std::array<format_detector_t, 1>&
format_detectors()
{
    static const std::array<format_detector_t, 1> detectors{ {
        { file_kind_t::sqlite, sqlite_magic_matches, probe_sqlite },
    } };
    return detectors;
}

}  // namespace

file_probe_result_t
probe_file(const std::string& path)
{
    std::error_code error;
    if(path.empty() || !std::filesystem::exists(path, error) || error)
    {
        return result_of(PH_RESULT_INVALID_FILE, "trace file does not exist");
    }
    if(!std::filesystem::is_regular_file(path, error) || error)
    {
        return result_of(PH_RESULT_UNSUPPORTED_FILE, "trace path is not a file");
    }

    std::ifstream input{ path, std::ios::binary };
    if(!input)
    {
        return result_of(PH_RESULT_INVALID_FILE, "trace file could not be read");
    }

    for(const format_detector_t& detector : format_detectors())
    {
        if(detector.matches(path)) return detector.probe(detector.kind, path);
    }
    return result_of(PH_RESULT_UNSUPPORTED_FILE, "trace file is not a supported format");
}

}  // namespace profiler_hub
