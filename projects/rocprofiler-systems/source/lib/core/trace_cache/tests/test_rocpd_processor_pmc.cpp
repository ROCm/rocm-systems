// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// rocpd_processor_t::handle() for PMC/counter samples against a StrictMock profiler-hub
// writer.

#include "library/pmc/collectors/cpu/sample.hpp"
#include "library/pmc/collectors/cpu/types.hpp"
#include "library/pmc/collectors/gpu/sample.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "library/pmc/collectors/gpu_perf_counter/sample.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/collectors/nic/sample.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "rocpd_processor_mock_fixture.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::trace_cache::test
{
namespace
{

using ::testing::Throw;

class rocpd_processor_pmc_test : public rocpd_processor_mock_test
{
protected:
    static std::optional<writer_types::agent_unique_id_t> no_agent() { return {}; }

    static expected_pmc_event make_pmc(expected_event event, double value,
                                       std::size_t timestamp, std::string_view track_name)
    {
        expected_pmc_event result{};
        result.event      = event;
        result.value      = value;
        result.timestamp  = timestamp;
        result.track_name = track_name;
        return result;
    }

    static expected_event make_event(std::string_view category)
    {
        expected_event result{};
        result.stack_id        = 0;
        result.parent_stack_id = 0;
        result.correlation_id  = 0;
        result.category        = category;
        return result;
    }

    void expect_pmc_insert(const expected_pmc_event& event, std::string_view pmc_name,
                           std::optional<writer_types::agent_unique_id_t> agent_id)
    {
        EXPECT_CALL(*g_mock_profiler_hub_writer,
                    insert_pmc_event_data(IsPmcEvent(event), IsPmcId(pmc_name, agent_id)))
            .Times(1);
    }
};

// ---------------------------------------------------------------------------
// in_time_sample / pmc_event_with_sample
// ---------------------------------------------------------------------------

TEST_F(rocpd_processor_pmc_test, handle_in_time_sample_inserts_event_named_after_track)
{
    constexpr std::size_t      k_timestamp = 9500;
    constexpr std::string_view k_meta      = R"({"metadata": "test"})";

    auto event            = make_event("my_track");
    event.stack_id        = 5;
    event.parent_stack_id = 3;
    event.extdata         = k_meta;
    expect_pmc_insert(make_pmc(event, 0.0, k_timestamp, "my_track"), "my_track",
                      no_agent());

    make_processor()->handle(
        in_time_sample{ 0, "my_track", k_timestamp, k_meta, 5, 3, 0, "", "" });
}

TEST_F(rocpd_processor_pmc_test, handle_in_time_sample_propagates_writer_failure)
{
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_pmc_event_data(
                    IsPmcEvent(make_pmc(
                        [] {
                            auto event            = make_event("unregistered_track");
                            event.stack_id        = 5;
                            event.parent_stack_id = 3;
                            return event;
                        }(),
                        0.0, 9500, "unregistered_track")),
                    IsPmcId(std::string_view{ "unregistered_track" }, no_agent())))
        .WillOnce(Throw(std::runtime_error{ "PMC Info not registered" }));

    auto processor = make_processor();
    EXPECT_THAT(
        [&] {
            processor->handle(
                in_time_sample{ 0, "unregistered_track", 9500, "{}", 5, 3, 0, "", "" });
        },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("PMC Info not registered")));
}

TEST_F(rocpd_processor_pmc_test,
       handle_pmc_event_inserts_value_on_gpu_agent_and_thread_track)
{
    add_agent(agent_type::gpu, 0, k_gpu_handle);

    auto event               = make_event("SQ_WAVES [GPU 0]");
    event.stack_id           = 10;
    event.parent_stack_id    = 5;
    event.correlation_id     = 42;
    auto expected            = make_pmc(event, 1024.0, 10000, "SQ_WAVES [GPU 0]");
    expected.track_thread_id = k_thread_id;
    expect_pmc_insert(expected, "SQ_WAVES", make_uid("GPU", 0));

    make_processor()->handle(
        pmc_event_with_sample{ 0, "SQ_WAVES [GPU 0]", 10000, "{}", 10, 5, 42, "", "", 0,
                               static_cast<std::uint8_t>(agent_type::gpu), "SQ_WAVES",
                               1024.0, static_cast<std::int64_t>(k_thread_id) });
}

// ---------------------------------------------------------------------------
// gpu_pmc_sample / gpu_perf_counter_sample
// ---------------------------------------------------------------------------

TEST_F(rocpd_processor_pmc_test, handle_gpu_pmc_inserts_one_event_per_enabled_metric)
{
    constexpr std::size_t k_timestamp = 11000;
    add_agent(agent_type::gpu, 0, k_gpu_handle);

    const auto amd_smi = category_name<category::amd_smi>();
    expect_pmc_insert(make_pmc(make_event(amd_smi), 85.0, k_timestamp, "device_busy_gfx"),
                      "device_busy_gfx", make_uid("GPU", 0));
    expect_pmc_insert(make_pmc(make_event(amd_smi), 42.0, k_timestamp, "device_busy_umc"),
                      "device_busy_umc", make_uid("GPU", 0));
    expect_pmc_insert(make_pmc(make_event(amd_smi), 72.0, k_timestamp, "device_temp"),
                      "device_temp", make_uid("GPU", 0));

    rocprofsys::pmc::collectors::gpu::enabled_metrics enabled{};
    enabled.bits.gfx_activity        = 1;
    enabled.bits.umc_activity        = 1;
    enabled.bits.hotspot_temperature = 1;
    rocprofsys::pmc::collectors::gpu::metrics metrics{};
    metrics.gfx_activity        = 85.0;
    metrics.umc_activity        = 42.0;
    metrics.hotspot_temperature = 72.0;

    make_processor()->handle(gpu_pmc_sample{ enabled, 0, k_timestamp, metrics });
}

TEST_F(rocpd_processor_pmc_test, handle_gpu_perf_counter_inserts_named_counter_value)
{
    constexpr std::uint32_t k_counter_id = 42;
    constexpr std::size_t   k_timestamp  = 11500;
    const std::string       pmc_name{ "SQ_WAVES[WGP=0,SA=0]" };
    const std::string       track_name{ "GPU [0] SQ_WAVES (S)" };

    add_agent(agent_type::gpu, 0, k_gpu_handle);
    info::gpu_perf_counter_name_entry name_entry{};
    name_entry.counter_id    = k_counter_id;
    name_entry.pmc_info_name = pmc_name;
    name_entry.track_name    = track_name;
    m_metadata->set_gpu_perf_counter_counter_names(0, { name_entry });

    expect_pmc_insert(
        make_pmc(make_event("rocm_counter_collection"), 2048.0, k_timestamp, track_name),
        pmc_name, make_uid("GPU", 0));

    make_processor()->handle(gpu_perf_counter_sample{
        0, k_timestamp,
        std::vector<rocprofsys::pmc::collectors::gpu_perf_counter::counter_value>{
            { .counter_id = k_counter_id, .value = 2048.0 } } });
}

TEST_F(rocpd_processor_pmc_test, handle_gpu_perf_counter_with_no_entries_inserts_nothing)
{
    add_agent(agent_type::gpu, 0, k_gpu_handle);

    make_processor()->handle(gpu_perf_counter_sample{ 0, 12000, {} });
}

// ---------------------------------------------------------------------------
// cpu_pmc_sample
// ---------------------------------------------------------------------------

TEST_F(rocpd_processor_pmc_test,
       handle_cpu_pmc_inserts_process_rss_and_per_core_frequency)
{
    constexpr std::size_t k_timestamp = 13000;
    add_agent(agent_type::cpu, 0, k_cpu_handle);

    const auto cpu_freq = category_name<category::cpu_freq>();
    expect_pmc_insert(
        make_pmc(make_event(cpu_freq), 256.5, k_timestamp, "process_physical_memory"),
        "process_physical_memory", make_uid("CPU", 0));
    expect_pmc_insert(
        make_pmc(make_event(cpu_freq), 3200.0, k_timestamp, "cpu_frequency [0] Core [0]"),
        "cpu_frequency", make_uid("CPU", 0));
    expect_pmc_insert(
        make_pmc(make_event(cpu_freq), 3100.0, k_timestamp, "cpu_frequency [0] Core [1]"),
        "cpu_frequency", make_uid("CPU", 0));

    rocprofsys::pmc::collectors::cpu::enabled_metrics enabled{};
    enabled.bits.page_rss  = 1;
    enabled.bits.frequency = 1;
    rocprofsys::pmc::collectors::cpu::process_metrics process{};
    process.page_rss = 256'500'000;
    const std::vector<rocprofsys::pmc::collectors::cpu::per_cpu_metrics> cores{
        { .cpu_id = 0, .frequency = 3200.0F, .load = 0.0 },
        { .cpu_id = 1, .frequency = 3100.0F, .load = 0.0 },
    };

    make_processor()->handle(
        cpu_pmc_sample{ enabled,
                        0,
                        k_timestamp,
                        process,
                        rocprofsys::pmc::collectors::cpu::serialize_frequencies(cores),
                        {} });
}

// ---------------------------------------------------------------------------
// ainic_pmc_sample
// ---------------------------------------------------------------------------

TEST_F(rocpd_processor_pmc_test, handle_ainic_inserts_one_event_per_enabled_metric)
{
    constexpr std::size_t k_timestamp = 12000;
    agent                 nic{};
    nic.type              = agent_type::nic;
    nic.device_type_index = 0;
    nic.name              = "NIC0";
    add_agent(nic);

    const auto nic_category = category_name<category::amd_smi_nic>();
    expect_pmc_insert(make_pmc(make_event(nic_category), 1048576.0, k_timestamp,
                               "ainic_rx_rdma_ucast_bytes"),
                      category_name<category::amd_smi_nic_rx_ucast_bytes>(),
                      make_uid("NIC", 0));
    expect_pmc_insert(make_pmc(make_event(nic_category), 524288.0, k_timestamp,
                               "ainic_tx_rdma_ucast_bytes"),
                      category_name<category::amd_smi_nic_tx_ucast_bytes>(),
                      make_uid("NIC", 0));

    rocprofsys::pmc::collectors::nic::enabled_metrics enabled{};
    enabled.bits.rx_rdma_ucast_bytes = 1;
    enabled.bits.tx_rdma_ucast_bytes = 1;
    rocprofsys::pmc::collectors::nic::metrics metrics{};
    metrics.rx_rdma_ucast_bytes = 1048576;
    metrics.tx_rdma_ucast_bytes = 524288;

    make_processor()->handle(
        ainic_pmc_sample{ enabled, 0, "NIC0", k_timestamp, metrics });
}

// ---------------------------------------------------------------------------
// kfd_sample
// ---------------------------------------------------------------------------

constexpr std::size_t      k_kfd_start = 14000;
constexpr std::size_t      k_kfd_end   = 14500;
constexpr std::string_view k_kfd_track = "KFD Events [GPU 0]";

MATCHER_P2(IsKfdRegion, event_meta, args, "kfd_sample mapped to a region with arguments")
{
    if(arg.name != "KFD_PAGE_FAULT" || arg.start_timestamp != k_kfd_start ||
       arg.end_timestamp != k_kfd_end || !arg.event.has_value() ||
       arg.event->event_category != "kfd" || arg.event->extdata != event_meta ||
       arg.args.size() != args.size())
    {
        return false;
    }
    for(std::size_t i = 0; i < args.size(); ++i)
    {
        if(arg.args[i].position != i || arg.args[i].type != args[i][0] ||
           arg.args[i].name != args[i][1] || arg.args[i].value != args[i][2])
        {
            return false;
        }
    }
    return true;
}

kfd_sample
make_kfd_sample(std::uint32_t device_id, std::string_view args_str)
{
    return kfd_sample{ k_thread_id,
                       "KFD_PAGE_FAULT",
                       k_kfd_start,
                       k_kfd_end,
                       args_str,
                       "kfd",
                       k_kfd_track,
                       R"({"source": "kfd"})",
                       device_id,
                       static_cast<std::uint8_t>(agent_type::gpu),
                       "kfd_page_fault",
                       1.0,
                       static_cast<std::int64_t>(k_thread_id) };
}

TEST_F(rocpd_processor_pmc_test, handle_kfd_inserts_region_with_arguments_then_pmc_event)
{
    add_agent(agent_type::gpu, 0, k_gpu_handle);
    const auto kfd_args =
        get_args_string(function_args_t{ { .arg_number = 0U,
                                           .arg_type   = "std::uint64_t",
                                           .arg_name   = "address",
                                           .arg_value  = "0x7f4a00001000" },
                                         { .arg_number = 1U,
                                           .arg_type   = "string",
                                           .arg_name   = "agent",
                                           .arg_value  = "5" } });

    auto event               = make_event("kfd");
    event.extdata            = R"({"source": "kfd"})";
    auto expected            = make_pmc(event, 1.0, k_kfd_start, k_kfd_track);
    expected.track_thread_id = k_thread_id;

    const ::testing::InSequence sequence;
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_region_data(
                    IsKfdRegion(std::string_view{ R"({"source": "kfd"})" },
                                std::vector<std::vector<std::string_view>>{
                                    { "std::uint64_t", "address", "0x7f4a00001000" },
                                    { "string", "agent", "5" } }),
                    IsEnv(expected_env{})))
        .Times(1);
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_pmc_event_data(
                    IsPmcEvent(expected),
                    IsPmcId(std::string_view{ "kfd_page_fault" }, make_uid("GPU", 0))))
        .Times(1);

    make_processor()->handle(make_kfd_sample(0, kfd_args));
}

TEST_F(rocpd_processor_pmc_test, handle_kfd_with_unknown_agent_throws_before_any_insert)
{
    add_agent(agent_type::gpu, 0, k_gpu_handle);

    auto processor = make_processor();
    EXPECT_THAT([&] { processor->handle(make_kfd_sample(999, "")); },
                ::testing::ThrowsMessage<std::out_of_range>(
                    ::testing::HasSubstr("Agent not found for type index")));
}

}  // namespace
}  // namespace rocprofsys::trace_cache::test
