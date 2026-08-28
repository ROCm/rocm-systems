// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/reader.hpp"
#include "profiler-hub/storage.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{

using namespace profiler_hub;

class writer_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_db_path             = (std::filesystem::temp_directory_path() /
                     (std::string{ "writer_test_" } + test_info->name() + ".db"))
                        .string();
        std::filesystem::remove(m_db_path);
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    [[nodiscard]] std::unique_ptr<writer_t> make_writer() const
    {
        return std::make_unique<writer_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    }

    // Registers a minimal node + process pair that most register_*/insert_* calls
    // require via insert_validator::require_node/require_process.
    static void register_node_and_process(writer_t&                  writer,
                                          writer_types::node_id_t    node_id = 1,
                                          writer_types::process_id_t pid     = 100)
    {
        const writer_types::node_info_t node_info{ node_id, 42, "machine-1" };
        writer.register_node_info(node_info);

        writer_types::process_info_t process_info;
        process_info.pid     = pid;
        process_info.node_id = node_id;
        writer.register_process_info(process_info);
    }

    std::string m_db_path;
    // Embedded verbatim into unquoted SQL table names (e.g. rocpd_string_<uuid>) by
    // insert_statements, so it must be a valid identifier fragment - no hyphens.
    std::string m_uuid = "testuuid0000";
};

TEST_F(writer_test, construct_with_null_storage_throws_invalid_argument)
{
    EXPECT_THROW((void) writer_t(nullptr), std::invalid_argument);
}

TEST_F(writer_test, construct_with_valid_storage_succeeds)
{
    EXPECT_NO_THROW({ auto writer = make_writer(); });
}

TEST_F(writer_test, construct_with_empty_uuid_throws_invalid_argument)
{
    EXPECT_THROW((void) writer_t(std::make_unique<storage_t>(m_db_path, "")),
                 std::invalid_argument);
}

TEST_F(writer_test, register_node_info_is_readable_after_flush)
{
    auto writer = make_writer();

    const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
    writer->register_node_info(node_info);
    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader =
        std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    const auto nodes = reader->get_all_nodes();

    ASSERT_EQ(nodes.size(), 1);
    EXPECT_EQ(nodes[0]->node_id, 1);
    EXPECT_EQ(nodes[0]->hash, 42);
    EXPECT_EQ(nodes[0]->machine_id, "machine-1");
}

TEST_F(writer_test, register_process_info_is_readable_after_flush)
{
    auto writer = make_writer();

    const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
    writer->register_node_info(node_info);

    writer_types::process_info_t process_info;
    process_info.pid     = 100;
    process_info.node_id = node_info.node_id;
    writer->register_process_info(process_info);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader =
        std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    const auto processes = reader->get_all_processes();

    ASSERT_EQ(processes.size(), 1);
    EXPECT_EQ(processes[0]->pid, 100);
    ASSERT_NE(processes[0]->node_info, nullptr);
    EXPECT_EQ(processes[0]->node_info->node_id, node_info.node_id);
}

TEST_F(writer_test, register_process_info_with_unregistered_node_throws_runtime_error)
{
    auto writer = make_writer();

    writer_types::process_info_t process_info;
    process_info.pid     = 100;
    process_info.node_id = 999;

    EXPECT_THROW(writer->register_process_info(process_info), std::runtime_error);
}

TEST_F(writer_test, register_agent_info_is_readable_after_flush)
{
    auto writer = make_writer();

    const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
    writer->register_node_info(node_info);

    writer_types::process_info_t process_info;
    process_info.pid     = 100;
    process_info.node_id = node_info.node_id;
    writer->register_process_info(process_info);

    writer_types::agent_info_t agent_info;
    agent_info.unique_id.agent_type = "GPU";
    agent_info.unique_id.type_index = 0;
    agent_info.absolute_index       = 0;
    agent_info.logical_index        = 0;
    agent_info.uuid                 = 123;
    agent_info.node_id              = node_info.node_id;
    agent_info.process_id           = process_info.pid;
    writer->register_agent_info(agent_info);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader =
        std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    const auto agents = reader->get_all_agents();

    ASSERT_EQ(agents.size(), 1);
    EXPECT_EQ(agents[0]->agent_type, "GPU");
    EXPECT_EQ(agents[0]->type_index, 0);
    ASSERT_NE(agents[0]->node_info, nullptr);
    EXPECT_EQ(agents[0]->node_info->node_id, node_info.node_id);
    ASSERT_NE(agents[0]->process_info, nullptr);
    EXPECT_EQ(agents[0]->process_info->pid, process_info.pid);
}

TEST_F(writer_test, register_agent_info_with_invalid_agent_type_throws_invalid_argument)
{
    auto writer = make_writer();

    const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
    writer->register_node_info(node_info);

    writer_types::process_info_t process_info;
    process_info.pid     = 100;
    process_info.node_id = node_info.node_id;
    writer->register_process_info(process_info);

    writer_types::agent_info_t agent_info;
    agent_info.unique_id.agent_type = "TPU";
    agent_info.unique_id.type_index = 0;
    agent_info.node_id              = node_info.node_id;
    agent_info.process_id           = process_info.pid;

    EXPECT_THROW(writer->register_agent_info(agent_info), std::invalid_argument);
}

TEST_F(writer_test, insert_region_data_is_readable_after_flush)
{
    auto writer = make_writer();

    const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
    writer->register_node_info(node_info);

    writer_types::process_info_t process_info;
    process_info.pid     = 100;
    process_info.node_id = node_info.node_id;
    writer->register_process_info(process_info);

    writer_types::thread_info_t thread_info;
    thread_info.thread_id  = 200;
    thread_info.node_id    = node_info.node_id;
    thread_info.process_id = process_info.pid;
    writer->register_thread_info(thread_info);

    writer_types::trace_environment_t trace_environment;
    trace_environment.node_id    = node_info.node_id;
    trace_environment.process_id = process_info.pid;
    trace_environment.thread_id  = thread_info.thread_id;

    writer_types::region_data_t region_data;
    region_data.name            = "test-region";
    region_data.start_timestamp = 1000;
    region_data.end_timestamp   = 2000;
    // region_statements() inner-joins on rocpd_event, so a region with no event data
    // is written but invisible to timeline queries (get_events/get_event_count).
    region_data.event = writer_types::event_data_t{};
    writer->insert_region_data(region_data, trace_environment);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader =
        std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    const auto events = reader->get_events();

    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0].display_name, "test-region");
    EXPECT_EQ(events[0].start_timestamp, 1000);
    EXPECT_EQ(events[0].end_timestamp, 2000);
    EXPECT_EQ(reader->get_event_count(), 1);
}

TEST_F(writer_test, insert_region_data_with_unregistered_thread_throws_runtime_error)
{
    auto writer = make_writer();

    writer_types::trace_environment_t trace_environment;
    trace_environment.node_id    = 1;
    trace_environment.process_id = 100;
    trace_environment.thread_id  = 999;

    writer_types::region_data_t region_data;
    region_data.name            = "test-region";
    region_data.start_timestamp = 1000;
    region_data.end_timestamp   = 2000;

    EXPECT_THROW(writer->insert_region_data(region_data, trace_environment),
                 std::runtime_error);
}

TEST_F(writer_test, flush_persists_data_to_disk_path)
{
    auto writer = make_writer();

    const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
    writer->register_node_info(node_info);

    ASSERT_FALSE(std::filesystem::exists(m_db_path));

    writer->flush_in_memory_data_to_disk();

    ASSERT_TRUE(std::filesystem::exists(m_db_path));
    EXPECT_GT(std::filesystem::file_size(m_db_path), 0);
}

TEST_F(writer_test, flush_called_twice_throws_runtime_error)
{
    auto writer = make_writer();

    writer->flush_in_memory_data_to_disk();

    EXPECT_THROW(writer->flush_in_memory_data_to_disk(), std::runtime_error);
}

TEST_F(writer_test, register_stream_info_is_readable_after_flush)
{
    auto writer = make_writer();
    register_node_and_process(*writer);

    writer_types::stream_info_t stream_info;
    stream_info.stream_id  = 10;
    stream_info.node_id    = 1;
    stream_info.process_id = 100;
    writer->register_stream_info(stream_info);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader =
        std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    const auto streams = reader->get_all_streams();

    ASSERT_EQ(streams.size(), 1);
    ASSERT_NE(streams[0]->node_info, nullptr);
    EXPECT_EQ(streams[0]->node_info->node_id, 1);
    ASSERT_NE(streams[0]->process_info, nullptr);
    EXPECT_EQ(streams[0]->process_info->pid, 100);
}

TEST_F(writer_test, register_queue_info_is_readable_after_flush)
{
    auto writer = make_writer();
    register_node_and_process(*writer);

    writer_types::queue_info_t queue_info;
    queue_info.queue_id   = 10;
    queue_info.node_id    = 1;
    queue_info.process_id = 100;
    writer->register_queue_info(queue_info);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader =
        std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    const auto queues = reader->get_all_queues();

    ASSERT_EQ(queues.size(), 1);
    ASSERT_NE(queues[0]->node_info, nullptr);
    EXPECT_EQ(queues[0]->node_info->node_id, 1);
    ASSERT_NE(queues[0]->process_info, nullptr);
    EXPECT_EQ(queues[0]->process_info->pid, 100);
}

TEST_F(writer_test, register_code_object_info_is_readable_after_flush)
{
    auto writer = make_writer();
    register_node_and_process(*writer);

    writer_types::code_object_info_t code_object_info;
    code_object_info.id         = 10;
    code_object_info.node_id    = 1;
    code_object_info.process_id = 100;
    writer->register_code_object_info(code_object_info);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader =
        std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    const auto code_objects = reader->get_all_code_objects();

    ASSERT_EQ(code_objects.size(), 1);
    EXPECT_EQ(code_objects[0]->id, 10);
    ASSERT_NE(code_objects[0]->node_info, nullptr);
    EXPECT_EQ(code_objects[0]->node_info->node_id, 1);
}

TEST_F(writer_test, register_kernel_symbol_info_is_readable_after_flush)
{
    auto writer = make_writer();
    register_node_and_process(*writer);

    writer_types::code_object_info_t code_object_info;
    code_object_info.id         = 10;
    code_object_info.node_id    = 1;
    code_object_info.process_id = 100;
    writer->register_code_object_info(code_object_info);

    writer_types::kernel_symbol_info_t kernel_symbol_info;
    kernel_symbol_info.id          = 20;
    kernel_symbol_info.node_id     = 1;
    kernel_symbol_info.process_id  = 100;
    kernel_symbol_info.code_obj_id = code_object_info.id;
    writer->register_kernel_symbol_info(kernel_symbol_info);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader =
        std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    const auto kernel_symbols = reader->get_all_kernel_symbols();

    ASSERT_EQ(kernel_symbols.size(), 1);
    EXPECT_EQ(kernel_symbols[0]->id, 20);
    ASSERT_NE(kernel_symbols[0]->code_object_info, nullptr);
    EXPECT_EQ(kernel_symbols[0]->code_object_info->id, code_object_info.id);
}

TEST_F(writer_test, register_track_info_is_readable_after_flush)
{
    auto writer = make_writer();

    const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
    writer->register_node_info(node_info);

    writer_types::track_info_t track_info;
    track_info.node_id = node_info.node_id;
    writer->register_track_info(track_info);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader =
        std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    const auto tracks = reader->get_all_tracks();

    ASSERT_EQ(tracks.size(), 1);
    ASSERT_NE(tracks[0]->node_info, nullptr);
    EXPECT_EQ(tracks[0]->node_info->node_id, node_info.node_id);
}

TEST_F(writer_test, register_string_does_not_throw)
{
    auto writer = make_writer();

    EXPECT_NO_THROW(writer->register_string("a-symbol-name"));
}

TEST_F(writer_test, register_pmc_info_is_readable_after_flush)
{
    auto writer = make_writer();
    register_node_and_process(*writer);

    writer_types::pmc_info_t pmc_info;
    pmc_info.unique_id.name = "occupancy";
    pmc_info.symbol         = "occupancy_symbol";
    pmc_info.node_id        = 1;
    pmc_info.process_id     = 100;
    writer->register_pmc_info(pmc_info);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader =
        std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    const auto pmc_infos = reader->get_all_pmc_info();

    ASSERT_EQ(pmc_infos.size(), 1);
    EXPECT_EQ(pmc_infos[0]->name, "occupancy");
    EXPECT_EQ(pmc_infos[0]->symbol, "occupancy_symbol");
}

TEST_F(writer_test, insert_pmc_event_data_does_not_throw)
{
    auto writer = make_writer();
    register_node_and_process(*writer);

    writer_types::pmc_info_t pmc_info;
    pmc_info.unique_id.name = "occupancy";
    pmc_info.symbol         = "occupancy_symbol";
    pmc_info.node_id        = 1;
    pmc_info.process_id     = 100;
    writer->register_pmc_info(pmc_info);

    writer_types::pmc_event_data_t pmc_event_data;
    pmc_event_data.value            = 1.5;
    pmc_event_data.sample.timestamp = 1000;

    EXPECT_NO_THROW(writer->insert_pmc_event_data(pmc_event_data, pmc_info.unique_id));
}

TEST_F(writer_test, insert_memory_copy_data_is_readable_after_flush)
{
    auto writer = make_writer();
    register_node_and_process(*writer);

    writer_types::trace_environment_t trace_environment;
    trace_environment.node_id    = 1;
    trace_environment.process_id = 100;

    writer_types::memory_copy_data_t memory_copy_data;
    memory_copy_data.name = "hipMemcpy";
    // Timeline display_name for memory copies comes from region_name_id, not name_id
    // (see initialize_memory_copy_timeline_event_statements in read_statements.hpp).
    memory_copy_data.region_name     = "hipMemcpy";
    memory_copy_data.start_timestamp = 1000;
    memory_copy_data.end_timestamp   = 2000;
    memory_copy_data.size            = 4096;
    memory_copy_data.event           = writer_types::event_data_t{};
    writer->insert_memory_copy_data(memory_copy_data, trace_environment);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader =
        std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    const auto events = reader->get_events();

    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0].display_name, "hipMemcpy");
}

TEST_F(writer_test, insert_memory_alloc_data_is_readable_after_flush)
{
    auto writer = make_writer();
    register_node_and_process(*writer);

    writer_types::trace_environment_t trace_environment;
    trace_environment.node_id    = 1;
    trace_environment.process_id = 100;

    writer_types::memory_alloc_data_t memory_alloc_data;
    memory_alloc_data.type            = "ALLOC";
    memory_alloc_data.level           = "REAL";
    memory_alloc_data.start_timestamp = 1000;
    memory_alloc_data.end_timestamp   = 2000;
    memory_alloc_data.size            = 4096;
    memory_alloc_data.event           = writer_types::event_data_t{};
    writer->insert_memory_alloc_data(memory_alloc_data, trace_environment);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader =
        std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    const auto events = reader->get_events();

    ASSERT_EQ(events.size(), 1);
}

TEST_F(writer_test, insert_memory_alloc_data_with_invalid_type_throws_runtime_error)
{
    auto writer = make_writer();
    register_node_and_process(*writer);

    writer_types::trace_environment_t trace_environment;
    trace_environment.node_id    = 1;
    trace_environment.process_id = 100;

    writer_types::memory_alloc_data_t memory_alloc_data;
    memory_alloc_data.type            = "BOGUS";
    memory_alloc_data.start_timestamp = 1000;
    memory_alloc_data.end_timestamp   = 2000;

    EXPECT_THROW(writer->insert_memory_alloc_data(memory_alloc_data, trace_environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_kernel_dispatch_data_is_readable_after_flush)
{
    auto writer = make_writer();
    register_node_and_process(*writer);

    writer_types::agent_info_t agent_info;
    agent_info.unique_id.agent_type = "GPU";
    agent_info.unique_id.type_index = 0;
    agent_info.node_id              = 1;
    agent_info.process_id           = 100;
    writer->register_agent_info(agent_info);

    // kernel_dispatch_writer::insert_impl requires thread/agent/queue/stream to be
    // registered (insert_validator::require_*), unlike region/memory writers which
    // treat them as optional (validate_optional_*).
    writer_types::thread_info_t thread_info;
    thread_info.thread_id  = 200;
    thread_info.node_id    = 1;
    thread_info.process_id = 100;
    writer->register_thread_info(thread_info);

    writer_types::queue_info_t queue_info;
    queue_info.queue_id   = 30;
    queue_info.node_id    = 1;
    queue_info.process_id = 100;
    writer->register_queue_info(queue_info);

    writer_types::stream_info_t stream_info;
    stream_info.stream_id  = 40;
    stream_info.node_id    = 1;
    stream_info.process_id = 100;
    writer->register_stream_info(stream_info);

    writer_types::code_object_info_t code_object_info;
    code_object_info.id         = 10;
    code_object_info.node_id    = 1;
    code_object_info.process_id = 100;
    writer->register_code_object_info(code_object_info);

    writer_types::kernel_symbol_info_t kernel_symbol_info;
    kernel_symbol_info.id          = 20;
    kernel_symbol_info.node_id     = 1;
    kernel_symbol_info.process_id  = 100;
    kernel_symbol_info.code_obj_id = code_object_info.id;
    writer->register_kernel_symbol_info(kernel_symbol_info);

    writer_types::trace_environment_t trace_environment;
    trace_environment.node_id    = 1;
    trace_environment.process_id = 100;
    trace_environment.thread_id  = thread_info.thread_id;
    trace_environment.agent_id   = agent_info.unique_id;
    trace_environment.queue_id   = queue_info.queue_id;
    trace_environment.stream_id  = stream_info.stream_id;

    writer_types::kernel_dispatch_data_t kernel_dispatch_data;
    kernel_dispatch_data.kernel_symbol_id = kernel_symbol_info.id;
    kernel_dispatch_data.start_timestamp  = 1000;
    kernel_dispatch_data.end_timestamp    = 2000;
    kernel_dispatch_data.event            = writer_types::event_data_t{};
    writer->insert_kernel_dispatch_data(kernel_dispatch_data, trace_environment);

    writer->flush_in_memory_data_to_disk();
    writer.reset();

    auto reader =
        std::make_unique<reader_t>(std::make_unique<storage_t>(m_db_path, m_uuid));
    const auto events = reader->get_events();

    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0].start_timestamp, 1000);
    EXPECT_EQ(events[0].end_timestamp, 2000);
}

}  // namespace
