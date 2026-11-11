// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/agent.hpp"
#include "core/collection/schema.hpp"
#include "core/trace_cache/metadata_registry.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <rocprofiler-sdk/callback_tracing.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════
// metadata_registry save_to_file → load_from_file
// ═══════════════════════════════════════════════════════════════════════════

namespace
{
using rocprofsys::agent;
using rocprofsys::trace_cache::metadata_registry;
using rocprofsys::trace_cache::info::thread;

// Above UINT32_MAX, so 32-bit truncation is caught.
constexpr std::uint64_t k_start = 1'000'000'000'001'000;
constexpr std::uint64_t k_end   = 1'000'000'000'009'000;
static_assert(k_start > std::numeric_limits<std::uint32_t>::max());

// Not a realistic TID (Linux caps TIDs at 2^22). Tests that the JSON keeps 64 bits field
constexpr std::uint64_t k_thread_id = 5'000'000'300;
static_assert(k_thread_id > std::numeric_limits<std::uint32_t>::max());

constexpr pid_t k_pid  = 200;
constexpr pid_t k_ppid = 100;

void
set_test_process(metadata_registry& registry)
{
    registry.set_process({ .pid         = k_pid,
                           .ppid        = k_ppid,
                           .command     = "test_binary",
                           .environment = "{}",
                           .extdata     = "{}",
                           .start       = k_start,
                           .end         = k_end });
}
}  // namespace

class metadata_registry_json_test_interface : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_file_path =
            std::filesystem::temp_directory_path() /
            ("metadata_registry_json_test_" + std::to_string(::getpid()) + ".json");
    }

    void TearDown() override { std::filesystem::remove(m_file_path); }

    [[nodiscard]] bool round_trip(const metadata_registry& saved,
                                  metadata_registry&       loaded) const
    {
        const std::vector<std::shared_ptr<agent>> saved_agents;
        std::vector<std::shared_ptr<agent>>       loaded_agents;
        return saved.save_to_file(m_file_path.string(), saved_agents) &&
               loaded.load_from_file(m_file_path.string(), loaded_agents);
    }

    [[nodiscard]] bool round_trip(
        const metadata_registry&                   saved,
        const std::vector<std::shared_ptr<agent>>& saved_agents,
        metadata_registry&                         loaded,
        std::vector<std::shared_ptr<agent>>&       loaded_agents) const
    {
        return saved.save_to_file(m_file_path.string(), saved_agents) &&
               loaded.load_from_file(m_file_path.string(), loaded_agents);
    }

    /** save_to_file, then edit(json) the file content, then load_from_file. */
    template <typename EditFn>
    [[nodiscard]] bool round_trip_edited(const metadata_registry& saved,
                                         metadata_registry& loaded, EditFn&& edit) const
    {
        const std::vector<std::shared_ptr<agent>> saved_agents;
        std::vector<std::shared_ptr<agent>>       loaded_agents;
        if(!saved.save_to_file(m_file_path.string(), saved_agents))
        {
            return false;
        }
        auto content = nlohmann::json::parse(std::ifstream{ m_file_path });
        std::forward<EditFn>(edit)(content);
        std::ofstream{ m_file_path } << content.dump();
        return loaded.load_from_file(m_file_path.string(), loaded_agents);
    }

private:
    std::filesystem::path m_file_path;
};

TEST_F(metadata_registry_json_test_interface, process_timestamps_survive_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    auto const process = loaded.get_process_info();
    EXPECT_EQ(process.start, k_start);
    EXPECT_EQ(process.end, k_end);
}

TEST_F(metadata_registry_json_test_interface, thread_timestamps_survive_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);
    saved.ensure_thread(k_thread_id, [] {
        return thread{ .parent_process_id = k_ppid,
                       .process_id        = k_pid,
                       .thread_id         = k_thread_id,
                       .start             = k_start,
                       .end               = k_end,
                       .extdata           = "{}" };
    });

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    auto const threads = loaded.get_thread_info_list();
    ASSERT_EQ(threads.size(), 1U);
    EXPECT_EQ(threads[0].thread_id, k_thread_id);
    EXPECT_EQ(threads[0].start, k_start);
    EXPECT_EQ(threads[0].end, k_end);
}

TEST_F(metadata_registry_json_test_interface, track_thread_id_survives_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);
    saved.ensure_track("test_track", k_thread_id);

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    auto const tracks = loaded.get_track_info_list();
    ASSERT_EQ(tracks.size(), 1U);
    EXPECT_EQ(tracks[0].thread_id, k_thread_id);
}

TEST_F(metadata_registry_json_test_interface, track_without_thread_id_survives_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);
    saved.ensure_track("test_track", std::nullopt);

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    auto const tracks = loaded.get_track_info_list();
    ASSERT_EQ(tracks.size(), 1U);
    EXPECT_FALSE(tracks[0].thread_id.has_value());
}

// Every field of every registry member survives save_to_file -> load_from_file.

namespace
{
using rocprofsys::agent_type;
using rocprofsys::collection::column_spec;
using rocprofsys::collection::schema;
using rocprofsys::trace_cache::info::pmc;
using rocprofsys::trace_cache::info::process;
using code_object_t = rocprofiler_callback_tracing_code_object_load_data_t;
using kernel_symbol_t =
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

constexpr std::uint64_t k_queue        = 0x7f00'0000'1000;
constexpr std::uint64_t k_stream       = 0x7f00'0000'2000;
constexpr std::uint64_t k_code_object  = 11;
constexpr std::uint64_t k_kernel_id    = 12;
constexpr std::uint64_t k_agent_handle = 13;
constexpr std::uint64_t k_load_base    = 0x1000;
constexpr std::uint64_t k_load_size    = 0x2000;
constexpr std::int64_t  k_load_delta   = -0x30;
constexpr std::uint64_t k_memory_base  = 0x4000;
constexpr std::uint64_t k_memory_size  = 0x500;
constexpr std::uint64_t k_kernel_obj   = 0x6000;
constexpr std::uint32_t k_kernarg_size = 64;
constexpr std::uint32_t k_kernarg_algn = 16;
constexpr std::uint32_t k_group_size   = 128;
constexpr std::uint32_t k_private_size = 32;
constexpr std::uint32_t k_sgprs        = 24;
constexpr std::uint32_t k_arch_vgprs   = 40;
constexpr std::uint32_t k_accum_vgprs  = 8;
constexpr std::int64_t  k_entry_offset = 256;
constexpr std::uint64_t k_kernel_addr  = 0x7000;
constexpr std::size_t   k_pmc_index    = 2;
constexpr std::size_t   k_event_code   = 17;
constexpr std::size_t   k_instance_id  = 5;
constexpr std::uint32_t k_category_id  = 9;
constexpr std::uint32_t k_device       = 3;
constexpr std::int32_t  k_node_id      = 6;
constexpr std::int32_t  k_logical_node = 7;
constexpr std::int32_t  k_logical_type = 8;
constexpr std::size_t   k_type_index   = 1;
constexpr std::uint32_t k_location_id  = 0x2600;
constexpr std::uint32_t k_domain       = 1;
// above INT32_MAX, as real device ids are
constexpr std::uint64_t k_device_id = 0x1'0000'0042ULL;

/** Why the DISABLED_ tests below stay disabled. */
constexpr std::string_view k_known_json_gap =
    "pre-existing gap on develop: the metadata JSON does not write these fields";

[[nodiscard]] pmc
make_full_pmc()
{
    return { .type             = agent_type::gpu,
             .agent_type_index = k_pmc_index,
             .target_arch      = "gfx942",
             .event_code       = k_event_code,
             .instance_id      = k_instance_id,
             .name             = "SQ_WAVES",
             .symbol           = "sq_waves",
             .description      = "waves",
             .long_description = "number of waves",
             .component        = "SQ",
             .units            = "count",
             .value_type       = "ABS",
             .block            = "SQ",
             .expression       = "reduce(SQ_WAVES,sum)",
             .is_constant      = 1,
             .is_derived       = 1,
             .extdata          = R"({"pmc":1})" };
}

[[nodiscard]] column_spec
make_column(std::string_view base, std::uint32_t category, bool hardware)
{
    const std::string name{ base };
    return { .symbol              = name + "_symbol",
             .description         = name + " description",
             .unit                = name + "_unit",
             .pmc_name            = name + "_pmc",
             .perfetto_track_name = name + " [perfetto]",
             .rocpd_track_name    = name + "_track",
             .category_id         = category,
             .block               = hardware ? "SQ" : "",
             .expression          = hardware ? "a+b" : "",
             .is_constant         = hardware,
             .is_derived          = !hardware };
}

[[nodiscard]] std::vector<schema>
make_schemas()
{
    return { { .agent          = agent_type::gpu,
               .device         = k_device,
               .target_arch    = "GPU",
               .event_category = "amd_smi",
               .columns        = { make_column("busy", k_category_id, false),
                                   make_column("waves", k_category_id + 1, true) } },
             { .agent          = agent_type::nic,
               .device         = 0,
               .target_arch    = "NIC",
               .event_category = "ainic",
               .columns        = { make_column("rx", 1, false) } } };
}

[[nodiscard]] code_object_t
make_code_object()
{
    code_object_t code_object{};
    code_object.size            = sizeof(code_object_t);
    code_object.code_object_id  = k_code_object;
    code_object.agent_id.handle = k_agent_handle;
    code_object.uri             = "file:///tmp/a.out#offset=0&size=8";
    code_object.load_base       = k_load_base;
    code_object.load_size       = k_load_size;
    code_object.load_delta      = k_load_delta;
    code_object.storage_type    = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY;
    code_object.memory_base     = k_memory_base;
    code_object.memory_size     = k_memory_size;
    return code_object;
}

[[nodiscard]] kernel_symbol_t
make_kernel_symbol()
{
    kernel_symbol_t symbol{};
    symbol.size                          = sizeof(kernel_symbol_t);
    symbol.kernel_id                     = k_kernel_id;
    symbol.code_object_id                = k_code_object;
    symbol.kernel_name                   = "kernel_a";
    symbol.kernel_object                 = k_kernel_obj;
    symbol.kernarg_segment_size          = k_kernarg_size;
    symbol.kernarg_segment_alignment     = k_kernarg_algn;
    symbol.group_segment_size            = k_group_size;
    symbol.private_segment_size          = k_private_size;
    symbol.sgpr_count                    = k_sgprs;
    symbol.arch_vgpr_count               = k_arch_vgprs;
    symbol.accum_vgpr_count              = k_accum_vgprs;
    symbol.kernel_code_entry_byte_offset = k_entry_offset;
    symbol.kernel_address.value          = k_kernel_addr;
    return symbol;
}

[[nodiscard]] std::shared_ptr<agent>
make_agent()
{
    auto result                  = std::make_shared<agent>();
    result->type                 = agent_type::gpu;
    result->handle               = k_agent_handle;
    result->device_id            = k_device;
    result->node_id              = k_node_id;
    result->logical_node_id      = k_logical_node;
    result->logical_node_type_id = k_logical_type;
    result->name                 = "gfx942";
    result->model_name           = "MI300X";
    result->vendor_name          = "AMD";
    result->product_name         = "Instinct MI300X";
    result->device_type_index    = k_type_index;
    return result;
}

[[nodiscard]] auto
fields(const pmc& value)
{
    return std::tuple{ value.type,
                       value.agent_type_index,
                       value.target_arch,
                       value.event_code,
                       value.instance_id,
                       value.name,
                       value.symbol,
                       value.description,
                       value.long_description,
                       value.component,
                       value.units,
                       value.value_type,
                       value.block,
                       value.expression,
                       value.is_constant,
                       value.is_derived,
                       value.extdata };
}

[[nodiscard]] auto
fields(const thread& value)
{
    return std::tuple{ value.parent_process_id,
                       value.process_id,
                       value.thread_id,
                       value.start,
                       value.end,
                       value.extdata };
}

/** The code object fields the JSON mirror writes. */
[[nodiscard]] auto
mirrored_fields(const code_object_t& value)
{
    return std::tuple{
        value.code_object_id, value.agent_id.handle, std::string_view{ value.uri },
        value.load_base,      value.load_size,       value.load_delta,
        value.storage_type
    };
}

/** The kernel symbol fields the JSON mirror writes. */
[[nodiscard]] auto
mirrored_fields(const kernel_symbol_t& value)
{
    return std::tuple{ value.kernel_id,
                       value.code_object_id,
                       std::string_view{ value.kernel_name },
                       value.kernel_object,
                       value.kernarg_segment_size,
                       value.kernarg_segment_alignment,
                       value.group_segment_size,
                       value.private_segment_size,
                       value.sgpr_count,
                       value.arch_vgpr_count,
                       value.accum_vgpr_count };
}

/** The agent fields the JSON mirror writes. */
[[nodiscard]] auto
mirrored_fields(const agent& value)
{
    return std::tuple{
        value.type,         value.handle,           value.device_id,
        value.node_id,      value.logical_node_id,  value.logical_node_type_id,
        value.name,         value.model_name,       value.vendor_name,
        value.product_name, value.device_type_index
    };
}

/** The agent fields the JSON mirror does not write (or truncates). */
[[nodiscard]] auto
unmirrored_fields(const agent& value)
{
    return std::tuple{ value.device_id, value.agent_info, value.location_id, value.domain,
                       value.hip_visible };
}
}  // namespace

TEST_F(metadata_registry_json_test_interface, process_mirrored_fields_survive_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    const process original = saved.get_process_info();
    const process restored = loaded.get_process_info();
    EXPECT_EQ(restored.pid, original.pid);
    EXPECT_EQ(restored.ppid, original.ppid);
    EXPECT_EQ(restored.command, original.command);
    EXPECT_EQ(restored.start, original.start);
    EXPECT_EQ(restored.end, original.end);
}

// to_json(info::process) writes neither environment nor extdata.
TEST_F(metadata_registry_json_test_interface,
       DISABLED_process_environment_and_extdata_survive_round_trip)
{
    SCOPED_TRACE(k_known_json_gap);
    metadata_registry saved;
    saved.set_process({ .pid         = k_pid,
                        .ppid        = k_ppid,
                        .command     = "test_binary",
                        .environment = R"({"HOME":"/root"})",
                        .extdata     = R"({"rank":0})",
                        .start       = k_start,
                        .end         = k_end });

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    EXPECT_EQ(loaded.get_process_info().environment, R"({"HOME":"/root"})");
    EXPECT_EQ(loaded.get_process_info().extdata, R"({"rank":0})");
}

TEST_F(metadata_registry_json_test_interface, pmc_info_survives_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);
    saved.add_pmc_info(make_full_pmc());

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    const auto pmcs = loaded.get_pmc_info_list();
    ASSERT_EQ(pmcs.size(), 1U);
    EXPECT_EQ(fields(pmcs.front()), fields(make_full_pmc()));
}

TEST_F(metadata_registry_json_test_interface, thread_survives_round_trip)
{
    const thread      original{ .parent_process_id = k_ppid,
                                .process_id        = k_pid,
                                .thread_id         = k_thread_id,
                                .start             = k_start,
                                .end               = k_end,
                                .extdata           = R"({"name":"worker"})" };
    metadata_registry saved;
    set_test_process(saved);
    saved.ensure_thread(k_thread_id, [&original] { return original; });

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    const auto threads = loaded.get_thread_info_list();
    ASSERT_EQ(threads.size(), 1U);
    EXPECT_EQ(fields(threads[0]), fields(original));
}

TEST_F(metadata_registry_json_test_interface, track_extdata_survives_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);
    saved.ensure_track("track", k_thread_id, R"({"kind":"pmc"})");

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    const auto tracks = loaded.get_track_info_list();
    ASSERT_EQ(tracks.size(), 1U);
    EXPECT_EQ(tracks[0].track_name, "track");
    EXPECT_EQ(tracks[0].thread_id, k_thread_id);
    EXPECT_EQ(tracks[0].extdata, R"({"kind":"pmc"})");
}

TEST_F(metadata_registry_json_test_interface, queues_streams_strings_survive_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);
    saved.add_queue(k_queue);
    saved.add_stream(k_stream);
    saved.add_string("kernel_a");

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    EXPECT_EQ(loaded.get_queue_list(), std::vector<std::uint64_t>{ k_queue });
    EXPECT_EQ(loaded.get_stream_list(), std::vector<std::uint64_t>{ k_stream });
    const auto strings = loaded.get_string_list();
    ASSERT_EQ(strings.size(), 1U);
    EXPECT_EQ(strings[0], "kernel_a");
}

TEST_F(metadata_registry_json_test_interface,
       code_object_mirrored_fields_survive_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);
    saved.add_code_object(make_code_object());

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    const auto code_objects = loaded.get_code_object_list();
    ASSERT_EQ(code_objects.size(), 1U);
    EXPECT_EQ(mirrored_fields(code_objects[0]), mirrored_fields(make_code_object()));
}

// The struct size and the memory_base/memory_size (or storage_file) union are not
// written.
TEST_F(metadata_registry_json_test_interface,
       DISABLED_code_object_storage_location_survives_round_trip)
{
    SCOPED_TRACE(k_known_json_gap);
    metadata_registry saved;
    set_test_process(saved);
    saved.add_code_object(make_code_object());

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    const auto code_objects = loaded.get_code_object_list();
    ASSERT_EQ(code_objects.size(), 1U);
    EXPECT_EQ(code_objects[0].size, sizeof(code_object_t));
    EXPECT_EQ(code_objects[0].memory_base, k_memory_base);
    EXPECT_EQ(code_objects[0].memory_size, k_memory_size);
}

TEST_F(metadata_registry_json_test_interface,
       kernel_symbol_mirrored_fields_survive_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);
    saved.add_kernel_symbol(make_kernel_symbol());

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    const auto symbols = loaded.get_kernel_symbol_list();
    ASSERT_EQ(symbols.size(), 1U);
    EXPECT_EQ(mirrored_fields(symbols[0]), mirrored_fields(make_kernel_symbol()));
}

// size, kernel_code_entry_byte_offset and kernel_address are not written.
TEST_F(metadata_registry_json_test_interface,
       DISABLED_kernel_symbol_entry_and_address_survive_round_trip)
{
    SCOPED_TRACE(k_known_json_gap);
    metadata_registry saved;
    set_test_process(saved);
    saved.add_kernel_symbol(make_kernel_symbol());

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    const auto symbols = loaded.get_kernel_symbol_list();
    ASSERT_EQ(symbols.size(), 1U);
    EXPECT_EQ(symbols[0].size, sizeof(kernel_symbol_t));
    EXPECT_EQ(symbols[0].kernel_code_entry_byte_offset, k_entry_offset);
    EXPECT_EQ(symbols[0].kernel_address.value, k_kernel_addr);
}

TEST_F(metadata_registry_json_test_interface, agent_mirrored_fields_survive_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);
    const std::vector<std::shared_ptr<agent>> saved_agents{ make_agent() };

    metadata_registry                   loaded;
    std::vector<std::shared_ptr<agent>> loaded_agents;
    ASSERT_TRUE(round_trip(saved, saved_agents, loaded, loaded_agents));

    ASSERT_EQ(loaded_agents.size(), 1U);
    ASSERT_NE(loaded_agents[0], nullptr);
    EXPECT_EQ(mirrored_fields(*loaded_agents[0]), mirrored_fields(*saved_agents[0]));
}

// agent_info, location_id, domain and hip_visible are not written, and device_id is
// read back as int32.
TEST_F(metadata_registry_json_test_interface,
       DISABLED_agent_identity_and_visibility_survive_round_trip)
{
    SCOPED_TRACE(k_known_json_gap);
    metadata_registry saved;
    set_test_process(saved);
    const auto original   = make_agent();
    original->device_id   = k_device_id;
    original->agent_info  = R"({"gfx":"942"})";
    original->location_id = k_location_id;
    original->domain      = k_domain;
    original->hip_visible = false;
    const std::vector<std::shared_ptr<agent>> saved_agents{ original };

    metadata_registry                   loaded;
    std::vector<std::shared_ptr<agent>> loaded_agents;
    ASSERT_TRUE(round_trip(saved, saved_agents, loaded, loaded_agents));

    ASSERT_EQ(loaded_agents.size(), 1U);
    ASSERT_NE(loaded_agents[0], nullptr);
    EXPECT_EQ(unmirrored_fields(*loaded_agents[0]), unmirrored_fields(*original));
}

TEST_F(metadata_registry_json_test_interface, counter_schemas_survive_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);
    for(const auto& description : make_schemas())
    {
        static_cast<void>(saved.add_counter_schema(description));
    }

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    // schema::operator== compares every field of the schema and of each column
    EXPECT_EQ(loaded.get_counter_schemas(), make_schemas());
}

TEST_F(metadata_registry_json_test_interface, counter_schema_ids_survive_round_trip)
{
    metadata_registry saved;
    set_test_process(saved);
    const auto schemas = make_schemas();
    for(const auto& description : schemas)
    {
        static_cast<void>(saved.add_counter_schema(description));
    }

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    EXPECT_EQ(loaded.add_counter_schema(schemas[1]), 1U);
    EXPECT_EQ(loaded.add_counter_schema(schemas[0]), 0U);
}

TEST_F(metadata_registry_json_test_interface,
       counter_schema_that_gets_another_id_fails_the_load)
{
    metadata_registry saved;
    set_test_process(saved);
    for(const auto& description : make_schemas())
    {
        static_cast<void>(saved.add_counter_schema(description));
    }
    metadata_registry loaded;
    static_cast<void>(loaded.add_counter_schema(make_schemas()[1]));

    EXPECT_FALSE(round_trip(saved, loaded));
}

TEST_F(metadata_registry_json_test_interface,
       counter_column_with_a_missing_field_fails_the_load)
{
    metadata_registry saved;
    set_test_process(saved);
    static_cast<void>(saved.add_counter_schema(make_schemas()[0]));

    metadata_registry loaded;
    EXPECT_FALSE(round_trip_edited(saved, loaded, [](nlohmann::json& content) {
        content["counter_schemas"][0]["columns"][0].erase("unit");
    }));
}

TEST_F(metadata_registry_json_test_interface, registry_without_counter_schemas_loads_none)
{
    metadata_registry saved;
    set_test_process(saved);

    metadata_registry loaded;
    ASSERT_TRUE(round_trip(saved, loaded));

    EXPECT_TRUE(loaded.get_counter_schemas().empty());
}
