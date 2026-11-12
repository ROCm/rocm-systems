// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/agent.hpp"
#include "core/collection/sample_batch.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/tests/log_capture.hpp"
#include "core/trace_cache/counter_samples.hpp"
#include "core/trace_cache/counter_schema_table.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocprofsys::trace_cache
{
namespace
{
using collection::testing::log_capture;

/** Keeps views into the stored schema, as the processors' slots do. */
struct test_slot
{
    const collection::schema*      owner  = nullptr;
    const collection::column_spec* column = nullptr;
    int                            uses   = 0;
};

constexpr std::uint32_t k_unknown_id = 5;
constexpr double        k_value      = 42.5;

[[nodiscard]] collection::schema
make_schema(std::uint32_t device, std::vector<std::string> pmc_names)
{
    collection::schema description{ .agent          = agent_type::gpu,
                                    .device         = device,
                                    .target_arch    = "GPU",
                                    .event_category = "amd_smi",
                                    .columns        = {} };
    for(auto& name : pmc_names)
    {
        collection::column_spec column;
        column.symbol   = name;
        column.pmc_name = std::move(name);
        description.columns.push_back(std::move(column));
    }
    return description;
}

[[nodiscard]] counter_batch
make_batch(std::uint32_t schema_id, std::size_t value_count)
{
    return { .schema_id = schema_id,
             .timestamp = 1,
             .values    = std::vector<double>(value_count) };
}

class counter_schema_table_test_interface : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_table.build({ make_schema(0, { "busy", "temp" }), make_schema(1, { "power" }) },
                      [this](const collection::schema&      description,
                             const collection::column_spec& column) {
                          ++m_make_calls;
                          return test_slot{ .owner = &description, .column = &column };
                      });
    }

    counter_schema_table<test_slot> m_table;
    int                             m_make_calls = 0;
};
}  // namespace

TEST_F(counter_schema_table_test_interface, build_makes_one_slot_per_column)
{
    EXPECT_EQ(m_make_calls, 3);
    EXPECT_EQ(m_table.find(make_batch(0, 2)).size(), 2U);
    EXPECT_EQ(m_table.find(make_batch(1, 1)).size(), 1U);
}

TEST_F(counter_schema_table_test_interface, slots_view_the_stored_schemas)
{
    const auto slots = m_table.find(make_batch(0, 2));

    ASSERT_EQ(slots.size(), 2U);
    EXPECT_EQ(slots[0].owner, slots[1].owner);
    EXPECT_EQ(slots[0].owner->columns.size(), 2U);
    EXPECT_EQ(slots[0].column->pmc_name, "busy");
    EXPECT_EQ(slots[1].column->pmc_name, "temp");
}

TEST_F(counter_schema_table_test_interface, find_returns_the_slots_of_the_batch_schema)
{
    const auto slots = m_table.find(make_batch(1, 1));

    ASSERT_EQ(slots.size(), 1U);
    EXPECT_EQ(slots[0].owner->device, 1U);
    EXPECT_EQ(slots[0].column->pmc_name, "power");
}

TEST_F(counter_schema_table_test_interface, slot_changes_persist_between_batches)
{
    ++m_table.find(make_batch(1, 1))[0].uses;

    EXPECT_EQ(m_table.find(make_batch(1, 1))[0].uses, 1);
}

TEST_F(counter_schema_table_test_interface,
       unknown_schema_id_gives_no_slots_and_one_warning)
{
    const log_capture logs;

    EXPECT_TRUE(m_table.find(make_batch(k_unknown_id, 1)).empty());
    EXPECT_TRUE(m_table.find(make_batch(k_unknown_id, 1)).empty());

    EXPECT_EQ(logs.count("warning", "unknown schema 5"), 1U);
}

TEST_F(counter_schema_table_test_interface,
       value_count_mismatch_gives_no_slots_and_one_warning)
{
    const log_capture logs;

    EXPECT_TRUE(m_table.find(make_batch(0, 1)).empty());
    EXPECT_TRUE(m_table.find(make_batch(0, 3)).empty());

    EXPECT_EQ(logs.count("warning", "Counter batch of schema 0 dropped"), 1U);
}

TEST_F(counter_schema_table_test_interface,
       for_each_slot_visits_every_column_of_every_schema)
{
    int visited = 0;
    m_table.for_each_slot([&visited](test_slot& slot) {
        ++slot.uses;
        ++visited;
    });

    EXPECT_EQ(visited, 3);
    EXPECT_EQ(m_table.find(make_batch(0, 2))[1].uses, 1);
}

TEST_F(counter_schema_table_test_interface, for_each_value_skips_nan_values)
{
    const counter_batch                         batch{ .schema_id = 0,
                                                       .timestamp = 1,
                                                       .values = { collection::k_no_value, k_value } };
    std::vector<std::pair<std::string, double>> visited;

    m_table.for_each_value(batch, [&visited](test_slot& slot, double value) {
        visited.emplace_back(slot.column->pmc_name, value);
    });

    EXPECT_EQ(visited,
              (std::vector<std::pair<std::string, double>>{ { "temp", k_value } }));
}

TEST_F(counter_schema_table_test_interface, for_each_value_of_a_dropped_batch_visits_none)
{
    const log_capture logs;
    int               visited = 0;

    m_table.for_each_value(make_batch(0, 1),
                           [&visited](test_slot&, double) { ++visited; });

    EXPECT_EQ(visited, 0);
}

// The slots view into the table: a copy or move would leave them dangling
static_assert(!std::is_copy_constructible_v<counter_schema_table<test_slot>>);
static_assert(!std::is_move_constructible_v<counter_schema_table<test_slot>>);

TEST_F(counter_schema_table_test_interface, rebuild_replaces_the_content_and_the_warnings)
{
    {
        const log_capture logs;
        static_cast<void>(m_table.find(make_batch(k_unknown_id, 1)));
    }

    m_table.build({}, [](const collection::schema&, const collection::column_spec&) {
        return test_slot{};
    });
    const log_capture logs;

    EXPECT_TRUE(m_table.find(make_batch(0, 2)).empty());
    EXPECT_TRUE(m_table.find(make_batch(k_unknown_id, 1)).empty());
    EXPECT_EQ(logs.count("warning", "unknown schema 5"), 1U);
}
}  // namespace rocprofsys::trace_cache
