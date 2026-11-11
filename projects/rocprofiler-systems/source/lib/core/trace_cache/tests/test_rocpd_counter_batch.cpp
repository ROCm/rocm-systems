// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/agent.hpp"
#include "core/agent_manager.hpp"
#include "core/collection/sample_batch.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/tests/log_capture.hpp"
#include "core/config.hpp"
#include "core/trace_cache/counter_samples.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/rocpd_processor.hpp"

#include <fmt/format.h>
#include <profiler-hub/reader.hpp>
#include <profiler-hub/reader_types.hpp>
#include <profiler-hub/storage.hpp>

#include <gtest/gtest.h>
#include <sqlite3.h>
#include <timemory/settings/settings.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace rocprofsys::trace_cache::detail
{
void
set_force_rocpd_metadata_registration_for_tests(bool enabled);
}

namespace rocprofsys::trace_cache
{
namespace
{
using collection::column_spec;
using collection::k_no_value;
using collection::schema;
using collection::testing::log_capture;

constexpr pid_t         k_pid           = 4200;
constexpr pid_t         k_ppid          = 4100;
constexpr std::uint64_t k_process_start = 1'000'000'000'001'000;
constexpr std::uint64_t k_process_end   = 1'000'000'000'009'000;
constexpr std::uint64_t k_first_ts      = 1'000'000'000'002'000;
constexpr std::uint64_t k_second_ts     = 1'000'000'000'003'000;
constexpr std::uint64_t k_gap_begin     = 1'000'000'000'004'000;
constexpr std::uint64_t k_gap_end       = 1'000'000'000'005'000;
constexpr double        k_busy          = 85.0;
constexpr double        k_temp          = 72.0;
constexpr std::uint32_t k_missing_gpu   = 7;

[[nodiscard]] column_spec
make_column(const std::string& name, const std::string& unit)
{
    return { .symbol              = name,
             .description         = name + " description",
             .unit                = unit,
             .pmc_name            = name,
             .perfetto_track_name = "GPU " + name,
             .rocpd_track_name    = name + "_track",
             .category_id         = 0,
             .block               = {},
             .expression          = {},
             .is_constant         = false,
             .is_derived          = false };
}

[[nodiscard]] schema
make_gpu_schema(std::uint32_t device = 0)
{
    return { .agent          = agent_type::gpu,
             .device         = device,
             .target_arch    = "GPU",
             .event_category = "amd_smi",
             .columns        = { make_column("busy", "%"), make_column("temp", "C") } };
}

[[nodiscard]] agent
make_gpu_agent()
{
    agent result{};
    result.type              = agent_type::gpu;
    result.handle            = 1;
    result.device_type_index = 0;
    result.name              = "gfx942";
    result.model_name        = "MI300X";
    result.vendor_name       = "AMD";
    result.product_name      = "Instinct MI300X";
    return result;
}

[[nodiscard]] info::pmc
make_registry_pmc(const std::string& name)
{
    return { .type             = agent_type::gpu,
             .agent_type_index = 0,
             .target_arch      = "GPU",
             .event_code       = 0,
             .instance_id      = 0,
             .name             = name,
             .symbol           = name,
             .description      = "registry description",
             .long_description = {},
             .component        = {},
             .units            = "%",
             .value_type       = "ABS",
             .block            = {},
             .expression       = {},
             .is_constant      = 0,
             .is_derived       = 0,
             .extdata          = "{}" };
}

/** One pmc_event read back from the database. */
struct read_event
{
    std::string   track;
    std::string   category;
    std::uint64_t timestamp = 0;
    double        value     = 0;
};

class scoped_rocpd_output
{
public:
    explicit scoped_rocpd_output(const std::filesystem::path& out_dir)
    : m_previous{ tim::settings::output_path() }
    {
        detail::set_force_rocpd_metadata_registration_for_tests(true);
        tim::settings::output_path() = out_dir.string();
        reset_database_path_memo();
    }

    ~scoped_rocpd_output()
    {
        tim::settings::output_path() = m_previous;
        reset_database_path_memo();
        detail::set_force_rocpd_metadata_registration_for_tests(false);
    }

    scoped_rocpd_output(const scoped_rocpd_output&)            = delete;
    scoped_rocpd_output& operator=(const scoped_rocpd_output&) = delete;
    scoped_rocpd_output(scoped_rocpd_output&&)                 = delete;
    scoped_rocpd_output& operator=(scoped_rocpd_output&&)      = delete;

private:
    std::string m_previous;
};

class rocpd_counter_batch_test_interface : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_out_dir = std::filesystem::temp_directory_path() /
                    ("rocpd_counter_batch_" + std::to_string(::getpid()) + "_" +
                     ::testing::UnitTest::GetInstance()->current_test_info()->name());
        std::filesystem::create_directories(m_out_dir);
        m_metadata->set_process({ .pid         = k_pid,
                                  .ppid        = k_ppid,
                                  .command     = "test_binary",
                                  .environment = "{}",
                                  .extdata     = "{}",
                                  .start       = k_process_start,
                                  .end         = k_process_end });
        auto gpu = make_gpu_agent();
        m_agents->insert_agent(gpu);
    }

    void TearDown() override
    {
        m_reader.reset();
        std::filesystem::remove_all(m_out_dir);
    }

    /** Runs the processor over `records` (counter_batch / gap_marker) and opens the db.
     */
    void process(const std::function<void(rocpd_processor_t&)>& records = {})
    {
        {
            const scoped_rocpd_output output{ m_out_dir };
            rocpd_processor_t         processor{ m_metadata, m_agents, k_pid, k_ppid };
            processor.prepare_for_processing();
            if(records)
            {
                records(processor);
            }
            processor.finalize_processing();
        }
        open_reader();
    }

    /** Every pmc_event row, by timestamp then track. Read with SQLite: profiler-hub's
     * reader_t::get_pmc_event_details() cannot return pmc events yet. */
    [[nodiscard]] std::vector<read_event> pmc_events() const
    {
        sqlite3* raw_db = nullptr;
        if(sqlite3_open_v2(m_db_path.c_str(), &raw_db, SQLITE_OPEN_READONLY, nullptr) !=
           SQLITE_OK)
        {
            sqlite3_close(raw_db);
            ADD_FAILURE() << "cannot open " << m_db_path;
            return {};
        }
        const std::unique_ptr<sqlite3, int (*)(sqlite3*)> database{ raw_db,
                                                                    &sqlite3_close };
        const auto uuid  = table_suffix(database.get(), "rocpd_pmc_event_");
        const auto query = fmt::format(
            "SELECT track_name.string, category.string, sample.timestamp, pmc.value "
            "FROM rocpd_pmc_event_{0} pmc "
            "JOIN rocpd_event_{0} event ON event.id = pmc.event_id "
            "JOIN rocpd_sample_{0} sample ON sample.event_id = pmc.event_id "
            "JOIN rocpd_track_{0} track ON track.id = sample.track_id "
            "JOIN rocpd_string_{0} track_name ON track_name.id = track.name_id "
            "JOIN rocpd_string_{0} category ON category.id = event.category_id "
            "ORDER BY sample.timestamp, track_name.string",
            uuid);
        sqlite3_stmt* raw_statement = nullptr;
        if(sqlite3_prepare_v2(database.get(), query.c_str(), -1, &raw_statement,
                              nullptr) != SQLITE_OK)
        {
            ADD_FAILURE() << sqlite3_errmsg(database.get());
            return {};
        }
        const std::unique_ptr<sqlite3_stmt, int (*)(sqlite3_stmt*)> statement{
            raw_statement, &sqlite3_finalize
        };
        std::vector<read_event> result;
        while(sqlite3_step(statement.get()) == SQLITE_ROW)
        {
            result.push_back({ .track     = column_text(statement.get(), 0),
                               .category  = column_text(statement.get(), 1),
                               .timestamp = static_cast<std::uint64_t>(
                                   sqlite3_column_int64(statement.get(), 2)),
                               .value = sqlite3_column_double(statement.get(), 3) });
        }
        return result;
    }

    [[nodiscard]] std::vector<std::string> pmc_names() const
    {
        std::vector<std::string> names;
        for(const auto& pmc_row : m_reader->get_all_pmc_info())
        {
            names.push_back(pmc_row->name);
        }
        std::ranges::sort(names);
        return names;
    }

    [[nodiscard]] std::size_t track_count(const std::string& name) const
    {
        const auto tracks = m_reader->get_all_tracks();
        return static_cast<std::size_t>(std::ranges::count_if(
            tracks, [&name](const auto& track) { return track->name == name; }));
    }

    std::shared_ptr<metadata_registry> m_metadata = std::make_shared<metadata_registry>();
    std::shared_ptr<agent_manager>     m_agents   = std::make_shared<agent_manager>();
    std::unique_ptr<profiler_hub::reader_t> m_reader;

private:
    void open_reader()
    {
        std::filesystem::path db_path;
        for(const auto& entry : std::filesystem::directory_iterator(m_out_dir))
        {
            const auto ext = entry.path().extension().string();
            if(ext == ".db" || ext == ".rocpd")
            {
                db_path = entry.path();
            }
        }
        ASSERT_FALSE(db_path.empty()) << "no rocpd database in " << m_out_dir;
        m_db_path       = db_path.string();
        const auto uuid = read_uuid(db_path);
        ASSERT_FALSE(uuid.empty());
        m_reader = std::make_unique<profiler_hub::reader_t>(
            std::make_unique<profiler_hub::storage_t>(m_db_path, uuid));
    }

    /** The suffix of the one table named `prefix` + suffix; empty if none. */
    [[nodiscard]] static std::string table_suffix(sqlite3*           database,
                                                  const std::string& prefix)
    {
        const auto query = "SELECT substr(name, " + std::to_string(prefix.size() + 1) +
                           ") FROM sqlite_master WHERE type = 'table' AND name LIKE '" +
                           prefix + "%'";
        sqlite3_stmt* raw_statement = nullptr;
        if(sqlite3_prepare_v2(database, query.c_str(), -1, &raw_statement, nullptr) !=
           SQLITE_OK)
        {
            return {};
        }
        const std::unique_ptr<sqlite3_stmt, int (*)(sqlite3_stmt*)> statement{
            raw_statement, &sqlite3_finalize
        };
        return sqlite3_step(statement.get()) == SQLITE_ROW
                   ? column_text(statement.get(), 0)
                   : std::string{};
    }

    [[nodiscard]] static std::string column_text(sqlite3_stmt* statement, int column)
    {
        const auto* text = sqlite3_column_text(statement, column);
        return text == nullptr ? std::string{}
                               : std::string{ reinterpret_cast<const char*>(text) };
    }

    /** The profiler-hub uuid is the suffix of the rocpd_info_agent_<uuid> table. */
    [[nodiscard]] static std::string read_uuid(const std::filesystem::path& db_path)
    {
        std::ifstream     db_file{ db_path, std::ios::binary };
        const std::string data{ std::istreambuf_iterator<char>{ db_file },
                                std::istreambuf_iterator<char>{} };
        const std::string key{ "rocpd_info_agent_" };
        const auto        pos = data.find(key);
        if(pos == std::string::npos)
        {
            return {};
        }
        auto end = pos + key.size();
        while(end < data.size() && std::isxdigit(static_cast<unsigned char>(data[end])))
        {
            ++end;
        }
        return data.substr(pos + key.size(), end - pos - key.size());
    }

    std::filesystem::path m_out_dir;
    std::string           m_db_path;
};
}  // namespace

TEST_F(rocpd_counter_batch_test_interface, schema_columns_register_pmc_info)
{
    static_cast<void>(m_metadata->add_counter_schema(make_gpu_schema()));

    process();

    std::vector<std::vector<std::string>> rows;
    for(const auto& pmc_row : m_reader->get_all_pmc_info())
    {
        rows.push_back({ pmc_row->name, pmc_row->symbol, pmc_row->description,
                         pmc_row->units, pmc_row->target_arch,
                         pmc_row->agent_info ? pmc_row->agent_info->agent_type : "" });
    }
    std::ranges::sort(rows);
    EXPECT_EQ(rows, (std::vector<std::vector<std::string>>{
                        { "busy", "busy", "busy description", "%", "GPU", "GPU" },
                        { "temp", "temp", "temp description", "C", "GPU", "GPU" } }));
}

TEST_F(rocpd_counter_batch_test_interface, schema_columns_register_their_tracks)
{
    static_cast<void>(m_metadata->add_counter_schema(make_gpu_schema()));

    process();

    EXPECT_EQ(track_count("busy_track"), 1U);
    EXPECT_EQ(track_count("temp_track"), 1U);
}

TEST_F(rocpd_counter_batch_test_interface,
       pmc_info_and_track_known_to_the_registry_are_registered_once)
{
    m_metadata->add_pmc_info(make_registry_pmc("busy"));
    m_metadata->ensure_track("busy_track", std::nullopt);
    static_cast<void>(m_metadata->add_counter_schema(make_gpu_schema()));

    process();

    EXPECT_EQ(pmc_names(), (std::vector<std::string>{ "busy", "temp" }));
    EXPECT_EQ(track_count("busy_track"), 1U);
}

TEST_F(rocpd_counter_batch_test_interface,
       schema_without_agent_is_skipped_and_warned_once)
{
    static_cast<void>(m_metadata->add_counter_schema(make_gpu_schema(k_missing_gpu)));
    const log_capture logs;

    process([](rocpd_processor_t& processor) {
        processor.handle(counter_batch{
            .schema_id = 0, .timestamp = k_first_ts, .values = { k_busy, k_temp } });
    });

    EXPECT_TRUE(m_reader->get_all_pmc_info().empty());
    EXPECT_EQ(logs.count("warning", "Counter schema skipped"), 1U);
}

TEST_F(rocpd_counter_batch_test_interface,
       batch_values_become_pmc_events_with_the_schema_category)
{
    static_cast<void>(m_metadata->add_counter_schema(make_gpu_schema()));

    process([](rocpd_processor_t& processor) {
        processor.handle(counter_batch{
            .schema_id = 0, .timestamp = k_first_ts, .values = { k_busy, k_temp } });
    });

    const auto events = pmc_events();
    ASSERT_EQ(events.size(), 2U);
    for(const auto& event : events)
    {
        EXPECT_EQ(event.timestamp, k_first_ts);
        EXPECT_EQ(event.category, "amd_smi");
    }
    EXPECT_EQ(events[0].value, k_busy);
    EXPECT_EQ(events[1].value, k_temp);
}

TEST_F(rocpd_counter_batch_test_interface, nan_values_are_skipped)
{
    static_cast<void>(m_metadata->add_counter_schema(make_gpu_schema()));

    process([](rocpd_processor_t& processor) {
        processor.handle(counter_batch{
            .schema_id = 0, .timestamp = k_first_ts, .values = { k_no_value, k_temp } });
    });

    const auto events = pmc_events();
    ASSERT_EQ(events.size(), 1U);
    EXPECT_EQ(events[0].value, k_temp);
}

TEST_F(rocpd_counter_batch_test_interface,
       gap_writes_zero_only_for_columns_that_had_a_value)
{
    static_cast<void>(m_metadata->add_counter_schema(make_gpu_schema()));

    process([](rocpd_processor_t& processor) {
        processor.handle(counter_batch{
            .schema_id = 0, .timestamp = k_first_ts, .values = { k_no_value, k_temp } });
        processor.handle(gap_marker{ .begin_ns = k_gap_begin, .end_ns = k_gap_end });
    });

    const auto events = pmc_events();
    ASSERT_EQ(events.size(), 2U);
    EXPECT_EQ(events[1].timestamp, k_gap_begin);
    EXPECT_EQ(events[1].value, 0.0);
    EXPECT_EQ(events[0].track, "temp_track");
    EXPECT_EQ(events[1].track, "temp_track");
}

TEST_F(rocpd_counter_batch_test_interface,
       value_count_mismatch_drops_the_batch_and_warns_once)
{
    static_cast<void>(m_metadata->add_counter_schema(make_gpu_schema()));
    const log_capture logs;

    process([](rocpd_processor_t& processor) {
        processor.handle(counter_batch{
            .schema_id = 0, .timestamp = k_first_ts, .values = { k_busy } });
        processor.handle(counter_batch{
            .schema_id = 0, .timestamp = k_second_ts, .values = { k_busy } });
    });

    EXPECT_EQ(logs.count("warning", "1 values for 2 columns"), 1U);
}

TEST_F(rocpd_counter_batch_test_interface,
       batch_of_unknown_schema_is_dropped_and_warned_once)
{
    const log_capture logs;

    process([](rocpd_processor_t& processor) {
        processor.handle(counter_batch{
            .schema_id = 0, .timestamp = k_first_ts, .values = { k_busy, k_temp } });
        processor.handle(counter_batch{
            .schema_id = 0, .timestamp = k_first_ts, .values = { k_busy, k_temp } });
    });

    EXPECT_EQ(logs.count("warning", "unknown schema 0"), 1U);
}
}  // namespace rocprofsys::trace_cache
