// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ph_ctx.hpp"
#include "reader_catalog.hpp"
#include "rocpd_v2_fixture.hpp"
#include "trace_fixtures.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <tuple>
#include <vector>

namespace
{

using namespace profiler_hub;

using track_row_t =
    std::tuple<uint32_t, std::string, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t>;

struct loaded_t
{
    std::vector<track_row_t> tracks;
    uint32_t                 agents{};
    uint32_t                 processes{};
    uint32_t                 threads{};
    uint32_t                 queues{};
    uint32_t                 pmc_infos{};
    uint32_t                 streams{};
    uint32_t                 links{};
    bool                     from_cache{};
};

std::vector<char>
read_bytes(const std::filesystem::path& path)
{
    std::ifstream in{ path, std::ios::binary };
    return { std::istreambuf_iterator<char>{ in }, std::istreambuf_iterator<char>{} };
}

void
write_bytes(const std::filesystem::path& path, const std::vector<char>& bytes)
{
    std::ofstream out{ path, std::ios::binary | std::ios::trunc };
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

class catalog_aggregate_store_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_root           = std::filesystem::temp_directory_path() /
                 (std::string{ "catalog_aggregate_store_test_" } + info->name());
        std::filesystem::remove_all(m_root);
        std::filesystem::create_directories(m_root);
        m_cache = m_root / "cache";
    }

    void TearDown() override
    {
        std::error_code error;
        std::filesystem::remove_all(m_root, error);
    }

    [[nodiscard]] loaded_t load(const std::filesystem::path& trace) const
    {
        ph_ctx ctx{ std::vector<std::string>{ trace.string() }, m_cache.string() };
        ctx.load();

        loaded_t   loaded;
        const auto list = ctx.get_track_list();
        for(uint32_t index = 0; index < list.list_size; ++index)
        {
            const ph_track_t& track = list.tracks[index];
            loaded.tracks.emplace_back(static_cast<uint32_t>(track.category),
                                       track.track_name != nullptr ? track.track_name
                                                                   : "",
                                       track.event_count,
                                       track.start_ts,
                                       track.end_ts,
                                       track.agent_id,
                                       track.queue_id);
        }
        const ph_node_t node = ctx.get_node();
        loaded.agents        = node.agents.list_size;
        loaded.processes     = node.process_list.list_size;
        loaded.threads       = node.thread_list.list_size;
        loaded.queues        = node.queue_list.list_size;
        loaded.pmc_infos     = node.pmc_info_list.list_size;
        loaded.streams       = node.stream_list.list_size;
        loaded.links         = node.stream_hardware_links.list_size;
        loaded.from_cache    = ctx.catalog()->derived_aggregates_ready;
        return loaded;
    }

    [[nodiscard]] std::filesystem::path cache_file(std::string_view name) const
    {
        if(!std::filesystem::exists(m_cache)) return {};
        for(const auto& entry : std::filesystem::recursive_directory_iterator{ m_cache })
        {
            if(entry.is_regular_file() && entry.path().filename() == name)
            {
                return entry.path();
            }
        }
        return {};
    }

    static void expect_same(const loaded_t& fresh, const loaded_t& reloaded)
    {
        EXPECT_EQ(reloaded.tracks, fresh.tracks);
        EXPECT_EQ(reloaded.agents, fresh.agents);
        EXPECT_EQ(reloaded.processes, fresh.processes);
        EXPECT_EQ(reloaded.threads, fresh.threads);
        EXPECT_EQ(reloaded.queues, fresh.queues);
        EXPECT_EQ(reloaded.pmc_infos, fresh.pmc_infos);
        EXPECT_EQ(reloaded.streams, fresh.streams);
        EXPECT_EQ(reloaded.links, fresh.links);
    }

    // Loads once to write the cache file, then reloads from it, then damages it
    // twice and checks that each reload falls back to SQL with the same result.
    void check_round_trip(const std::filesystem::path& trace, std::string_view file_name)
    {
        const loaded_t fresh = load(trace);
        EXPECT_FALSE(fresh.from_cache);
        ASSERT_FALSE(fresh.tracks.empty());

        const std::filesystem::path stored = cache_file(file_name);
        ASSERT_FALSE(stored.empty()) << file_name << " was not written";
        const std::vector<char> valid = read_bytes(stored);
        ASSERT_GT(valid.size(), 64U);

        const loaded_t cached = load(trace);
        EXPECT_TRUE(cached.from_cache);
        expect_same(fresh, cached);

        write_bytes(stored, std::vector<char>{ 'X', 'X', 'X', 'X', 0, 0, 0, 0 });
        const loaded_t bad_magic = load(trace);
        EXPECT_FALSE(bad_magic.from_cache);
        expect_same(fresh, bad_magic);
        EXPECT_EQ(read_bytes(stored), valid);

        write_bytes(stored, std::vector<char>(valid.begin(), valid.end() - 4));
        const loaded_t truncated = load(trace);
        EXPECT_FALSE(truncated.from_cache);
        expect_same(fresh, truncated);
        EXPECT_EQ(read_bytes(stored), valid);

        std::vector<char> other_source = valid;
        other_source[8]                = static_cast<char>(other_source[8] + 1);
        write_bytes(stored, other_source);
        EXPECT_FALSE(load(trace).from_cache);
    }

    std::filesystem::path m_root;
    std::filesystem::path m_cache;
};

TEST_F(catalog_aggregate_store_test, schema2_aggregates_round_trip_and_reject_damage)
{
    const auto trace = m_root / "trace.rpd";
    test::write_schema2(trace);
    check_round_trip(trace, "aggregates-0.bin");
}

TEST_F(catalog_aggregate_store_test, schema3_derived_tracks_round_trip_and_reject_damage)
{
    const auto trace = m_root / "trace.db";
    test::write_gpu_trace(trace.string());
    check_round_trip(trace, "derived-0.bin");
}

TEST_F(catalog_aggregate_store_test, no_cache_directory_writes_nothing)
{
    const auto trace = m_root / "trace.rpd";
    test::write_schema2(trace);
    {
        ph_ctx ctx{ std::vector<std::string>{ trace.string() }, std::string{} };
        ctx.load();
        EXPECT_FALSE(ctx.catalog()->derived_aggregates_ready);
    }
    EXPECT_FALSE(std::filesystem::exists(m_cache));
}

}  // namespace
