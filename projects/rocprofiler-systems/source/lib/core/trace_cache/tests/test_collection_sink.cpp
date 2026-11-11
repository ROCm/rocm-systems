// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/agent.hpp"
#include "core/collection/sample_batch.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/tests/log_capture.hpp"
#include "core/trace_cache/collection_sink.hpp"
#include "core/trace_cache/counter_samples.hpp"
#include "policies/trace_cache/buffer_storage.hpp"
#include "policies/trace_cache/metadata_registry.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocprofsys::trace_cache
{
namespace
{
using ::testing::Return;
using ::testing::StrictMock;

constexpr std::uint32_t k_schema_id   = 7;
constexpr std::uint64_t k_timestamp   = 123456;
constexpr std::uint64_t k_gap_begin   = 1000;
constexpr std::uint64_t k_gap_end     = 2000;
constexpr double        k_first_value = 1.5;
constexpr double        k_last_value  = -2.25;

/** What the storage did with a record. */
enum class storage_mode : std::uint8_t
{
    store,
    stopped,
    throw_std,
    throw_other
};

/** A counter row copied out of the writer's buffer. */
struct stored_batch
{
    std::uint32_t       schema_id = 0;
    std::uint64_t       timestamp = 0;
    std::vector<double> values;
};

class fake_storage
{
public:
    storage_mode mode = storage_mode::store;

    std::vector<stored_batch> batches;
    std::vector<gap_marker>   gaps;

    void store(const counter_batch_view& record)
    {
        fail_if_asked();
        batches.push_back({ .schema_id = record.schema_id,
                            .timestamp = record.timestamp,
                            .values = { record.values.begin(), record.values.end() } });
    }

    void store(const gap_marker& record)
    {
        fail_if_asked();
        gaps.push_back(record);
    }

    [[nodiscard]] bool is_running() const { return mode != storage_mode::stopped; }

private:
    void fail_if_asked() const
    {
        if(mode == storage_mode::throw_std)
        {
            throw std::runtime_error("storage full");
        }
        if(mode == storage_mode::throw_other)
        {
            std::rethrow_exception(std::make_exception_ptr(k_schema_id));
        }
    }
};

class mock_registry
{
public:
    MOCK_METHOD(std::uint32_t, add_counter_schema,
                (const collection::schema& description));
    MOCK_METHOD(std::vector<collection::schema>, get_counter_schemas, (), (const));
};

static_assert(
    policies::trace_cache::record_storage<fake_storage, counter_batch_view, gap_marker>);
static_assert(
    policies::trace_cache::counter_schema_registry<mock_registry, collection::schema>);

using sink_t = collection_sink<fake_storage, StrictMock<mock_registry>>;

[[nodiscard]] collection::column_spec
make_column(const std::string& name)
{
    collection::column_spec column;
    column.symbol   = name;
    column.pmc_name = name;
    return column;
}

[[nodiscard]] collection::schema
make_schema()
{
    return { .agent          = agent_type::gpu,
             .device         = 1,
             .target_arch    = "GPU",
             .event_category = "amd_smi",
             .columns        = { make_column("busy") } };
}

class collection_sink_test_interface : public ::testing::Test
{
protected:
    void write_one_batch()
    {
        m_sink.write_batch(
            { .schema = k_schema_id, .timestamp = k_timestamp, .values = m_row });
    }

    fake_storage              m_storage;
    StrictMock<mock_registry> m_registry;
    sink_t                    m_sink{ m_storage, m_registry };
    std::array<double, 3> m_row{ k_first_value, collection::k_no_value, k_last_value };
};
}  // namespace

TEST_F(collection_sink_test_interface, register_schema_returns_the_registry_id)
{
    const auto description = make_schema();
    EXPECT_CALL(m_registry, add_counter_schema(description))
        .WillOnce(Return(k_schema_id));

    EXPECT_EQ(m_sink.register_schema(description), k_schema_id);
}

TEST_F(collection_sink_test_interface,
       schema_without_columns_throws_and_is_not_registered)
{
    auto description = make_schema();
    description.columns.clear();

    EXPECT_THROW(static_cast<void>(m_sink.register_schema(description)),
                 std::invalid_argument);
}

TEST_F(collection_sink_test_interface, batch_is_stored_as_one_record_with_its_values)
{
    write_one_batch();

    ASSERT_EQ(m_storage.batches.size(), 1U);
    const auto& stored = m_storage.batches.front();
    EXPECT_EQ(stored.schema_id, k_schema_id);
    EXPECT_EQ(stored.timestamp, k_timestamp);
    ASSERT_EQ(stored.values.size(), m_row.size());
    EXPECT_EQ(stored.values[0], k_first_value);
    EXPECT_TRUE(std::isnan(stored.values[1]));
    EXPECT_EQ(stored.values[2], k_last_value);
    EXPECT_EQ(m_sink.dropped(), 0U);
}

TEST_F(collection_sink_test_interface, gap_is_stored)
{
    m_sink.write_gap(k_gap_begin, k_gap_end);

    ASSERT_EQ(m_storage.gaps.size(), 1U);
    EXPECT_EQ(m_storage.gaps.front().begin_ns, k_gap_begin);
    EXPECT_EQ(m_storage.gaps.front().end_ns, k_gap_end);
}

TEST_F(collection_sink_test_interface, stopped_storage_drops_and_counts_every_record)
{
    m_storage.mode = storage_mode::stopped;

    write_one_batch();
    write_one_batch();
    m_sink.write_gap(k_gap_begin, k_gap_end);

    EXPECT_TRUE(m_storage.batches.empty());
    EXPECT_TRUE(m_storage.gaps.empty());
    EXPECT_EQ(m_sink.dropped(), 3U);
}

TEST_F(collection_sink_test_interface, throwing_storage_drops_and_counts)
{
    m_storage.mode = storage_mode::throw_std;
    EXPECT_NO_THROW(write_one_batch());
    m_storage.mode = storage_mode::throw_other;
    EXPECT_NO_THROW(m_sink.write_gap(k_gap_begin, k_gap_end));

    EXPECT_EQ(m_sink.dropped(), 2U);
}

TEST_F(collection_sink_test_interface, storing_resumes_after_drops)
{
    m_storage.mode = storage_mode::stopped;
    write_one_batch();
    m_storage.mode = storage_mode::store;

    write_one_batch();

    EXPECT_EQ(m_storage.batches.size(), 1U);
    EXPECT_EQ(m_sink.dropped(), 1U);
}

TEST_F(collection_sink_test_interface, first_drop_is_warned_once_with_its_reason)
{
    const collection::testing::log_capture logs;
    m_storage.mode = storage_mode::throw_std;

    write_one_batch();
    write_one_batch();

    EXPECT_EQ(logs.count("warning", "further drops are only counted: storage full"), 1U);
}

TEST(collection_sink, destructor_logs_the_drop_count_once)
{
    const collection::testing::log_capture logs;
    fake_storage                           storage;
    StrictMock<mock_registry>              registry;
    storage.mode = storage_mode::stopped;
    {
        sink_t sink{ storage, registry };
        sink.write_gap(k_gap_begin, k_gap_end);
        sink.write_gap(k_gap_begin, k_gap_end);
    }

    EXPECT_EQ(logs.count("warning", "collection: 2 record(s) dropped"), 1U);
}

TEST(collection_sink, destructor_without_drops_logs_nothing)
{
    const collection::testing::log_capture logs;
    fake_storage                           storage;
    StrictMock<mock_registry>              registry;
    {
        const sink_t sink{ storage, registry };
    }

    EXPECT_EQ(logs.count("warning", "dropped"), 0U);
}

TEST_F(collection_sink_test_interface, is_usable_through_the_sink_interface)
{
    collection::sink_interface& out = m_sink;

    out.write_gap(k_gap_begin, k_gap_end);

    EXPECT_EQ(m_storage.gaps.size(), 1U);
}
}  // namespace rocprofsys::trace_cache
