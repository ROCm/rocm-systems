// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// prepare_for_processing / finalize_processing of rocpd_processor_t against a StrictMock
// profiler-hub writer.

#include "library/thread_info.hpp"
#include "rocpd_processor_mock_fixture.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

// rocprof-sys-unit-tests does not link library/thread_info.cpp (see
// source/tests/CMakeLists.txt). post_process_metadata() calls thread_info::get when
// metadata has thread rows; return empty so thread start/end come from metadata_registry
// only.
namespace rocprofsys
{
const std::optional<thread_info>&
thread_info::get(std::int64_t, ThreadIdType)
{
    static const std::optional<thread_info> k_none{};
    return k_none;
}

std::uint64_t
thread_info::get_start() const
{
    return 0;
}

std::uint64_t
thread_info::get_stop() const
{
    return 0;
}
}  // namespace rocprofsys

namespace rocprofsys::trace_cache::test
{
namespace
{

using ::testing::A;
using ::testing::Eq;
using ::testing::Throw;

struct nic_pmc_spec
{
    std::string_view name;
    std::string_view units;
};

constexpr std::array k_nic_pmcs{
    nic_pmc_spec{ .name = "nic_rx_ucast_pkts", .units = "packets" },
    nic_pmc_spec{ .name = "nic_tx_ucast_pkts", .units = "packets" },
    nic_pmc_spec{ .name = "nic_rx_cnp_pkts", .units = "packets" },
    nic_pmc_spec{ .name = "nic_tx_cnp_pkts", .units = "packets" },
    nic_pmc_spec{ .name = "nic_rx_ucast_bytes", .units = "bytes" },
    nic_pmc_spec{ .name = "nic_tx_ucast_bytes", .units = "bytes" },
    nic_pmc_spec{ .name = "nic_tx_rdma_ack_timeout", .units = "timeouts" },
    nic_pmc_spec{ .name = "nic_resp_tx_pkt_seq_err", .units = "errors" },
    nic_pmc_spec{ .name = "nic_req_rx_pkt_seq_err", .units = "errors" },
    nic_pmc_spec{ .name = "nic_req_rx_impl_nak_seq_err", .units = "errors" },
};

MATCHER(IsExpectedNode, "node registered with the node_info id")
{
    return arg.node_id == node_id();
}

MATCHER(IsExpectedProcess, "process registered from metadata")
{
    return arg.pid == static_cast<std::size_t>(k_pid) &&
           arg.ppid == static_cast<std::size_t>(k_ppid) && arg.node_id == node_id() &&
           arg.command == "test_binary" && arg.start == k_process_start &&
           arg.end == k_process_end;
}

struct expected_agent
{
    std::string_view type;
    std::size_t      absolute_index;
    std::string_view name;
    std::string_view model_name;
    std::string_view vendor_name;
    std::string_view product_name;
};

MATCHER_P(IsAgentInfo, expected, "agent registered with its unique id and names")
{
    return arg.unique_id == make_uid(expected.type, 0) &&
           arg.absolute_index == expected.absolute_index && arg.name == expected.name &&
           arg.model_name == expected.model_name &&
           arg.vendor_name == expected.vendor_name &&
           arg.product_name == expected.product_name && arg.node_id == node_id() &&
           arg.process_id == static_cast<std::size_t>(k_pid);
}

struct expected_pmc_info
{
    std::string_view                               name;
    std::string_view                               target_arch;
    std::optional<writer_types::agent_unique_id_t> agent_id;
    std::string_view                               description;
    std::string_view                               units;
};

MATCHER_P(IsPmcInfo, expected, "pmc info registered with its agent")
{
    return arg.unique_id.name == expected.name &&
           arg.unique_id.agent_id == expected.agent_id && arg.symbol == expected.name &&
           arg.target_arch == expected.target_arch &&
           arg.description == expected.description && arg.units == expected.units &&
           arg.node_id == node_id() && arg.process_id == static_cast<std::size_t>(k_pid);
}

MATCHER_P(IsTrack, expected_name, "track registered for the process")
{
    return arg.name == expected_name && arg.node_id == node_id() &&
           arg.process_id == static_cast<std::size_t>(k_pid);
}

[[nodiscard]] agent
make_agent(agent_type type, std::string_view name, std::string_view model,
           std::string_view vendor, std::string_view product)
{
    agent result{};
    result.type              = type;
    result.device_type_index = 0;
    result.name              = name;
    result.model_name        = model;
    result.vendor_name       = vendor;
    result.product_name      = product;
    return result;
}

class rocpd_processor_metadata_test : public rocpd_processor_mock_test
{
protected:
    void expect_node_and_process_registered()
    {
        EXPECT_CALL(*g_mock_profiler_hub_writer, register_node_info(IsExpectedNode()))
            .Times(1);
        EXPECT_CALL(*g_mock_profiler_hub_writer,
                    register_process_info(IsExpectedProcess()))
            .Times(1);
    }

    void expect_baseline_registered() { expect_node_and_process_registered(); }
};

TEST_F(rocpd_processor_metadata_test, finalize_processing_flushes_writer_exactly_once)
{
    EXPECT_CALL(*g_mock_profiler_hub_writer, flush_in_memory_data_to_disk()).Times(1);

    make_processor()->finalize_processing();
}

TEST_F(rocpd_processor_metadata_test, prepare_registers_gpu_cpu_and_nic_agents_in_order)
{
    add_agent(make_agent(agent_type::gpu, "gfx90a", "MI210", "AMD", "Instinct MI210"));
    add_agent(make_agent(agent_type::cpu, "CPU0", "EPYC", "AMD", "EPYC 7763"));
    add_agent(make_agent(agent_type::nic, "NIC0", "CX7", "AI NIC", "AI NIC"));

    expect_baseline_registered();
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_agent_info(IsAgentInfo(expected_agent{
                    "GPU", 0, "gfx90a", "MI210", "AMD", "Instinct MI210" })))
        .Times(1);
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_agent_info(IsAgentInfo(
                    expected_agent{ "CPU", 1, "CPU0", "EPYC", "AMD", "EPYC 7763" })))
        .Times(1);
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_agent_info(IsAgentInfo(
                    expected_agent{ "NIC", 2, "NIC0", "CX7", "AI NIC", "AI NIC" })))
        .Times(1);

    make_processor()->prepare_for_processing();
}

TEST_F(rocpd_processor_metadata_test, prepare_registers_thread_queue_stream_and_tracks)
{
    info::thread thread{};
    thread.parent_process_id = k_ppid;
    thread.process_id        = k_pid;
    thread.thread_id         = k_thread_id;
    thread.start             = k_process_start;
    thread.end               = k_process_end;
    m_metadata->add_thread_info(thread);
    m_metadata->add_queue(k_queue_id);
    m_metadata->add_stream(k_stream_id);
    m_metadata->add_track(info::track{
        .track_name = "Sampling [CPU 0]", .thread_id = k_thread_id, .extdata = {} });
    m_metadata->add_track(info::track{
        .track_name = "process_track", .thread_id = std::nullopt, .extdata = {} });

    expect_baseline_registered();
    EXPECT_CALL(
        *g_mock_profiler_hub_writer,
        register_thread_info(::testing::AllOf(
            ::testing::Field(&writer_types::thread_info_t::thread_id, Eq(k_thread_id)),
            ::testing::Field(&writer_types::thread_info_t::parent_process_id,
                             Eq(static_cast<std::size_t>(k_ppid))),
            ::testing::Field(&writer_types::thread_info_t::name,
                             Eq(std::optional<std::string_view>{ "Thread 300" })),
            ::testing::Field(&writer_types::thread_info_t::start, Eq(k_process_start)),
            ::testing::Field(&writer_types::thread_info_t::end, Eq(k_process_end)),
            ::testing::Field(&writer_types::thread_info_t::node_id, Eq(node_id())),
            ::testing::Field(&writer_types::thread_info_t::process_id,
                             Eq(static_cast<std::size_t>(k_pid))))))
        .Times(1);
    EXPECT_CALL(
        *g_mock_profiler_hub_writer,
        register_queue_info(::testing::AllOf(
            ::testing::Field(&writer_types::queue_info_t::queue_id, Eq(k_queue_id)),
            ::testing::Field(&writer_types::queue_info_t::name,
                             Eq(std::optional<std::string_view>{ "Queue 10" })),
            ::testing::Field(&writer_types::queue_info_t::node_id, Eq(node_id())),
            ::testing::Field(&writer_types::queue_info_t::process_id,
                             Eq(static_cast<std::size_t>(k_pid))))))
        .Times(1);
    EXPECT_CALL(
        *g_mock_profiler_hub_writer,
        register_stream_info(::testing::AllOf(
            ::testing::Field(&writer_types::stream_info_t::stream_id, Eq(k_stream_id)),
            ::testing::Field(&writer_types::stream_info_t::name,
                             Eq(std::optional<std::string_view>{ "Stream 20" })),
            ::testing::Field(&writer_types::stream_info_t::node_id, Eq(node_id())),
            ::testing::Field(&writer_types::stream_info_t::process_id,
                             Eq(static_cast<std::size_t>(k_pid))))))
        .Times(1);
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_track_info(IsTrack(std::string_view{ "Sampling [CPU 0]" })))
        .Times(1);
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_track_info(IsTrack(std::string_view{ "process_track" })))
        .Times(1);

    make_processor()->prepare_for_processing();
}

TEST_F(rocpd_processor_metadata_test, prepare_registers_code_object_and_kernel_symbol)
{
    add_agent(agent_type::gpu, 0, k_gpu_handle);
    seed_kernel_symbol(9, "my_test_kernel");

    expect_baseline_registered();
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_agent_info(::testing::Field(
                    &writer_types::agent_info_t::unique_id, Eq(make_uid("GPU", 0)))))
        .Times(1);
    EXPECT_CALL(*g_mock_profiler_hub_writer, register_string(Eq("my_test_kernel")))
        .Times(1);
    EXPECT_CALL(
        *g_mock_profiler_hub_writer,
        register_code_object_info(::testing::AllOf(
            ::testing::Field(&writer_types::code_object_info_t::id, Eq(1U)),
            ::testing::Field(
                &writer_types::code_object_info_t::uri,
                Eq(std::optional<std::string_view>{ "file:///test_code_object.co" })),
            ::testing::Field(&writer_types::code_object_info_t::storage_type,
                             Eq(std::optional<std::string_view>{ "MEMORY" })),
            ::testing::Field(
                &writer_types::code_object_info_t::agent_id,
                Eq(std::optional<writer_types::agent_unique_id_t>{ make_uid("GPU", 0) })),
            ::testing::Field(&writer_types::code_object_info_t::node_id, Eq(node_id())),
            ::testing::Field(&writer_types::code_object_info_t::process_id,
                             Eq(static_cast<std::size_t>(k_pid))))))
        .Times(1);
    EXPECT_CALL(
        *g_mock_profiler_hub_writer,
        register_kernel_symbol_info(::testing::AllOf(
            ::testing::Field(&writer_types::kernel_symbol_info_t::id, Eq(9U)),
            ::testing::Field(&writer_types::kernel_symbol_info_t::name,
                             Eq(std::optional<std::string_view>{ "my_test_kernel" })),
            ::testing::Field(&writer_types::kernel_symbol_info_t::display_name,
                             Eq(std::optional<std::string_view>{ "my_test_kernel" })),
            ::testing::Field(&writer_types::kernel_symbol_info_t::code_obj_id, Eq(1U)),
            ::testing::Field(&writer_types::kernel_symbol_info_t::node_id, Eq(node_id())),
            ::testing::Field(&writer_types::kernel_symbol_info_t::process_id,
                             Eq(static_cast<std::size_t>(k_pid))))))
        .Times(1);

    make_processor()->prepare_for_processing();
}
TEST_F(rocpd_processor_metadata_test, prepare_registers_every_nic_pmc_on_the_nic_agent)
{
    add_agent(agent_type::nic, 0, 0);
    for(const auto& spec : k_nic_pmcs)
    {
        add_pmc(agent_type::nic, spec.name, "NIC", {}, spec.units);
    }

    expect_baseline_registered();
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_agent_info(::testing::Field(
                    &writer_types::agent_info_t::unique_id, Eq(make_uid("NIC", 0)))))
        .Times(1);
    for(const auto& spec : k_nic_pmcs)
    {
        EXPECT_CALL(*g_mock_profiler_hub_writer,
                    register_pmc_info(IsPmcInfo(expected_pmc_info{
                        spec.name, "NIC", make_uid("NIC", 0), spec.name, spec.units })))
            .Times(1);
    }

    make_processor()->prepare_for_processing();
}

TEST_F(rocpd_processor_metadata_test, prepare_keeps_distinct_target_arch_per_agent_type)
{
    add_agent(agent_type::gpu, 0, k_gpu_handle);
    add_agent(agent_type::nic, 0, 0);
    add_pmc(agent_type::gpu, "gfx_busy", "GPU");
    add_pmc(agent_type::nic, "nic_rx_ucast_bytes", "NIC", {}, "bytes");

    expect_baseline_registered();
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_agent_info(::testing::Field(
                    &writer_types::agent_info_t::unique_id, Eq(make_uid("GPU", 0)))))
        .Times(1);
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_agent_info(::testing::Field(
                    &writer_types::agent_info_t::unique_id, Eq(make_uid("NIC", 0)))))
        .Times(1);
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_pmc_info(IsPmcInfo(expected_pmc_info{
                    "gfx_busy", "GPU", make_uid("GPU", 0), "gfx_busy", "" })))
        .Times(1);
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_pmc_info(IsPmcInfo(
                    expected_pmc_info{ "nic_rx_ucast_bytes", "NIC", make_uid("NIC", 0),
                                       "nic_rx_ucast_bytes", "bytes" })))
        .Times(1);

    make_processor()->prepare_for_processing();
}

TEST_F(rocpd_processor_metadata_test,
       prepare_registers_pmc_without_agent_when_lookup_fails)
{
    constexpr std::size_t k_unregistered_agent_index = 999;
    add_agent(agent_type::gpu, 0, k_gpu_handle);
    add_pmc(agent_type::gpu, "my_track", "GPU", "IN_TIME", {},
            k_unregistered_agent_index);

    expect_baseline_registered();
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_agent_info(::testing::Field(
                    &writer_types::agent_info_t::unique_id, Eq(make_uid("GPU", 0)))))
        .Times(1);
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_pmc_info(IsPmcInfo(
                    expected_pmc_info{ "my_track", "GPU", std::nullopt, "IN_TIME", "" })))
        .Times(1);

    make_processor()->prepare_for_processing();
}

TEST_F(rocpd_processor_metadata_test, prepare_propagates_writer_rejecting_a_pmc_info)
{
    add_agent(agent_type::nic, 0, 0);
    add_pmc(agent_type::nic, "nic_rx_ucast_bytes", "AINIC", {}, "bytes");

    expect_baseline_registered();
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_agent_info(::testing::Field(
                    &writer_types::agent_info_t::unique_id, Eq(make_uid("NIC", 0)))))
        .Times(1);
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                register_pmc_info(
                    ::testing::Field(&writer_types::pmc_info_t::target_arch,
                                     Eq(std::optional<std::string_view>{ "AINIC" }))))
        .WillOnce(Throw(std::invalid_argument{ "Invalid PMC target_arch: AINIC" }));

    auto processor = make_processor();
    EXPECT_THAT([&] { processor->prepare_for_processing(); },
                ::testing::ThrowsMessage<std::invalid_argument>(
                    ::testing::HasSubstr("Invalid PMC target_arch: AINIC")));
}

TEST_F(rocpd_processor_metadata_test, prepare_propagates_writer_rejecting_an_empty_string)
{
    m_metadata->add_string("");

    expect_node_and_process_registered();
    EXPECT_CALL(*g_mock_profiler_hub_writer, register_string(Eq("")))
        .WillOnce(Throw(std::runtime_error{ "Trying to register empty string" }));

    auto processor = make_processor();
    EXPECT_THAT([&] { processor->prepare_for_processing(); },
                ::testing::ThrowsMessage<std::runtime_error>(
                    ::testing::HasSubstr("Trying to register empty string")));
}

}  // namespace
}  // namespace rocprofsys::trace_cache::test
