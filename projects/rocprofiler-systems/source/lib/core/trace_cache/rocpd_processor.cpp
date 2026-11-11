// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/rocpd_processor.hpp"
#include "agent.hpp"
#include "common/md5sum.hpp"
#include "core/agent_manager.hpp"
#include "core/collection/schema.hpp"
#include "core/common_types.hpp"
#include "core/config.hpp"
#include "core/demangler.hpp"
#include "core/node_info.hpp"
#include "core/output/output_summary.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/counter_samples.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/rocpd_helpers.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <array>
#include <profiler-hub/storage.hpp>
#include <profiler-hub/writer.hpp>
#include <profiler-hub/writer_types.hpp>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/version.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocprofsys::trace_cache
{
namespace
{

// Tests construct rocpd_processor_t without rocprofsys_init_library / USE_ROCPD.
// post_process_metadata() normally returns early when get_use_rocpd() is false; tests
// toggle this via detail::set_force_rocpd_metadata_registration_for_tests() (see
// test_rocpd_processor.cpp run_processor_and_open_reader) so agent/PMC metadata still
// registers on the production code path.
bool g_force_rocpd_metadata_registration = false;

using rocpd_helpers::make_agent_uid;
using rocpd_helpers::make_event;
using rocpd_helpers::make_trace_env;
using rocpd_helpers::make_trace_env_with_agent;
using rocpd_helpers::make_trace_env_with_agent_queue_stream;
using rocpd_helpers::parse_memory_operation_name;

auto
get_handle_from_code_object(
    const rocprofiler_callback_tracing_code_object_load_data_t& code_object)
{
#if(ROCPROFILER_VERSION >= 600)
    return code_object.agent_id.handle;
#else
    return code_object.rocp_agent.handle;
#endif
}

std::string
generate_db_output_path(int pid)
{
    auto const tag     = std::to_string(pid);
    auto const db_name = std::string{ "rocpd" };
    return rocprofsys::get_database_absolute_path(db_name, tag);
}

info::pmc
make_counter_pmc_info(const collection::schema&      description,
                      const collection::column_spec& column)
{
    return { .type             = description.agent,
             .agent_type_index = description.device,
             .target_arch      = description.target_arch,
             .event_code       = 0,
             .instance_id      = 0,
             .name             = column.pmc_name,
             .symbol           = column.symbol,
             .description      = column.description,
             .long_description = {},
             .component        = {},
             .units            = column.unit,
             .value_type       = k_absolute,
             .block            = column.block,
             .expression       = column.expression,
             .is_constant      = column.is_constant ? 1U : 0U,
             .is_derived       = column.is_derived ? 1U : 0U,
             .extdata          = "{}" };
}

// Appends the counter-schema PMCs the registry does not have; registry order is kept.
std::vector<info::pmc>
merge_counter_pmc_infos(std::vector<info::pmc>                 pmc_infos,
                        const std::vector<collection::schema>& schemas)
{
    std::unordered_set<info::pmc, info::pmc_info_hash, info::pmc_info_equal> known{
        pmc_infos.begin(), pmc_infos.end()
    };
    const auto add_if_new = [&known, &pmc_infos](info::pmc pmc_info) {
        if(known.insert(pmc_info).second)
        {
            pmc_infos.push_back(std::move(pmc_info));
        }
    };
    for(const auto& description : schemas)
    {
        for(const auto& column : description.columns)
        {
            add_if_new(make_counter_pmc_info(description, column));
        }
    }
    return pmc_infos;
}

std::set<info::track, info::track_less>
merge_counter_tracks(const std::vector<info::track>&        tracks,
                     const std::vector<collection::schema>& schemas)
{
    std::set<info::track, info::track_less> merged{ tracks.begin(), tracks.end() };
    for(const auto& description : schemas)
    {
        for(const auto& column : description.columns)
        {
            merged.insert(info::track{ .track_name = column.rocpd_track_name,
                                       .thread_id  = std::nullopt,
                                       .extdata    = "{}" });
        }
    }
    return merged;
}
}  // namespace

namespace detail
{
void
set_force_rocpd_metadata_registration_for_tests(bool enabled)
{
    g_force_rocpd_metadata_registration = enabled;
}
}  // namespace detail

void
rocpd_processor_t::handle(const kernel_dispatch_sample& kds)
{
    const auto& n_info    = node_info::get_instance();
    auto const  process   = m_metadata->get_process_info();
    const auto& agent_ref = m_agent_manager->get_agent_by_handle(kds.agent_id_handle);

    auto kernel_symbol = m_metadata->get_kernel_symbol(kds.kernel_id);
    if(!kernel_symbol.has_value())
    {
        throw std::runtime_error("Kernel symbol is missing for kernel dispatch");
    }

    auto const kernel_name = rocprofsys::utility::demangle(kernel_symbol->kernel_name);

    auto const event =
        make_event(kds.correlation_id_internal, kds.correlation_id_ancestor, 0,
                   trait::name<category::rocm_kernel_dispatch>::value);

    profiler_hub::writer_types::kernel_dispatch_data_t kernel_dispatch;
    kernel_dispatch.event                = event;
    kernel_dispatch.dispatch_id          = kds.dispatch_id;
    kernel_dispatch.start_timestamp      = kds.start_timestamp;
    kernel_dispatch.end_timestamp        = kds.end_timestamp;
    kernel_dispatch.kernel_symbol_id     = kds.kernel_id;
    kernel_dispatch.private_segment_size = kds.private_segment_size;
    kernel_dispatch.group_segment_size   = kds.group_segment_size;
    kernel_dispatch.workgroup_size_x     = kds.workgroup_size_x;
    kernel_dispatch.workgroup_size_y     = kds.workgroup_size_y;
    kernel_dispatch.workgroup_size_z     = kds.workgroup_size_z;
    kernel_dispatch.grid_size_x          = kds.grid_size_x;
    kernel_dispatch.grid_size_y          = kds.grid_size_y;
    kernel_dispatch.grid_size_z          = kds.grid_size_z;
    kernel_dispatch.name                 = kernel_name.c_str();

    auto const env = make_trace_env_with_agent_queue_stream(
        n_info.id, process.pid, kds.thread_id, agent_ref, kds.queue_id_handle,
        kds.stream_handle);

    m_writer->insert_kernel_dispatch_data(kernel_dispatch, env);
}

void
rocpd_processor_t::handle(const scratch_memory_sample& sms)
{
    auto const& n_info  = node_info::get_instance();
    auto const  process = m_metadata->get_process_info();

    const auto& agent_ref = m_agent_manager->get_agent_by_handle(sms.agent_id_handle);

    auto [memory_operation, memory_type_val] = parse_memory_operation_name(sms.name);
    auto extdata_json_str = fmt::format("{{\"flags\": {}}}", sms.flags);

    auto const event =
        make_event(sms.correlation_id_internal, sms.correlation_id_ancestor, 0,
                   trait::name<category::rocm_scratch_memory>::value);

    profiler_hub::writer_types::memory_alloc_data_t ma;
    ma.event           = event;
    ma.type            = memory_operation.c_str();
    ma.level           = memory_type_val.c_str();
    ma.start_timestamp = sms.start_timestamp;
    ma.end_timestamp   = sms.end_timestamp;
    ma.address         = 0;
    ma.size            = sms.allocation_size;
    ma.extdata         = extdata_json_str;

    auto const env = make_trace_env_with_agent_queue_stream(
        n_info.id, process.pid, sms.thread_id, agent_ref, sms.queue_id_handle,
        sms.stream_handle);

    m_writer->insert_memory_alloc_data(ma, env);
}

void
rocpd_processor_t::handle(const memory_copy_sample& mcs)
{
    auto const& n_info  = node_info::get_instance();
    auto const  process = m_metadata->get_process_info();

    const auto& dst_agent = m_agent_manager->get_agent_by_handle(mcs.dst_agent_id_handle);
    const auto& src_agent = m_agent_manager->get_agent_by_handle(mcs.src_agent_id_handle);

    auto const event =
        make_event(mcs.correlation_id_internal, mcs.correlation_id_ancestor, 0,
                   trait::name<category::rocm_memory_copy>::value);

    profiler_hub::writer_types::memory_copy_data_t memory_copy;
    memory_copy.event           = event;
    memory_copy.start_timestamp = mcs.start_timestamp;
    memory_copy.end_timestamp   = mcs.end_timestamp;
    memory_copy.dst_agent_id    = make_agent_uid(dst_agent);
    memory_copy.dst_address     = mcs.dst_address_value;
    memory_copy.src_agent_id    = make_agent_uid(src_agent);
    memory_copy.src_address     = mcs.src_address_value;
    memory_copy.size            = mcs.bytes;
    memory_copy.name            = mcs.name;
    memory_copy.region_name     = mcs.name;

    auto env      = make_trace_env(n_info.id, process.pid, mcs.thread_id);
    env.stream_id = mcs.stream_handle;
    env.queue_id  = 0;

    m_writer->insert_memory_copy_data(memory_copy, env);
}

void
rocpd_processor_t::handle([[maybe_unused]] const memory_allocate_sample& mas)
{
#if(ROCPROFILER_VERSION >= 600)
    auto const& n_info  = node_info::get_instance();
    auto const  process = m_metadata->get_process_info();

    const auto invalid_context = ROCPROFILER_CONTEXT_NONE;
    if(mas.agent_id_handle != invalid_context.handle)
    {
        const auto& agent_ref = m_agent_manager->get_agent_by_handle(mas.agent_id_handle);

        auto [memory_operation, memory_type_val] = parse_memory_operation_name(mas.name);

        auto const event =
            make_event(mas.correlation_id_internal, mas.correlation_id_ancestor, 0,
                       trait::name<category::rocm_memory_allocate>::value);

        profiler_hub::writer_types::memory_alloc_data_t ma;
        ma.event           = event;
        ma.type            = memory_operation;
        ma.level           = memory_type_val;
        ma.start_timestamp = mas.start_timestamp;
        ma.end_timestamp   = mas.end_timestamp;
        ma.address         = mas.address_value;
        ma.size            = mas.allocation_size;

        auto env =
            make_trace_env_with_agent(n_info.id, process.pid, mas.thread_id, agent_ref);
        env.stream_id = mas.stream_handle;
        env.queue_id  = 0;

        m_writer->insert_memory_alloc_data(ma, env);
    }
#endif
}

void
rocpd_processor_t::handle(const region_sample& reg_sample)
{
    auto const& n_info  = node_info::get_instance();
    auto const  process = m_metadata->get_process_info();

    auto event =
        make_event(reg_sample.correlation_id_internal, reg_sample.correlation_id_ancestor,
                   0, reg_sample.category.data());
    event.call_stack.push_back({});
    // call_stack and line_info are serialized JSON in the old code; in profiler-hub
    // they are structured types. For now pass the raw JSON via extdata.
    event.extdata = reg_sample.call_stack;

    profiler_hub::writer_types::region_data_t region;
    region.event           = event;
    region.start_timestamp = reg_sample.start_timestamp;
    region.end_timestamp   = reg_sample.end_timestamp;
    region.name            = reg_sample.name;

    auto const parsed_args = process_arguments_string(reg_sample.args_str);
    for(const auto& arg : parsed_args)
    {
        profiler_hub::writer_types::arg_data_t arg_data;
        arg_data.position = arg.arg_number;
        arg_data.type     = arg.arg_type;
        arg_data.name     = arg.arg_name;
        arg_data.value    = arg.arg_value;
        region.args.push_back(arg_data);
    }

    auto const env = make_trace_env(n_info.id, process.pid, reg_sample.thread_id);
    m_writer->insert_region_data(region, env);
}

void
rocpd_processor_t::handle(const backtrace_region_sample& bts)
{
    auto const& n_info  = node_info::get_instance();
    auto const  process = m_metadata->get_process_info();

    auto event = make_event(0, 0, 0, bts.category.data());
    event.call_stack.push_back({});
    // call_stack and line_info are serialized JSON in the old code; in profiler-hub
    // they are structured types. For now pass the raw JSON via extdata.
    event.extdata = bts.call_stack;

    profiler_hub::writer_types::region_data_t region;
    region.event           = event;
    region.start_timestamp = bts.start_timestamp;
    region.end_timestamp   = bts.end_timestamp;
    region.name            = bts.name;

    auto env       = make_trace_env(n_info.id, process.pid, bts.thread_id);
    env.track_name = bts.track_name;

    m_writer->insert_region_data(region, env);
}

void
rocpd_processor_t::handle(const in_time_sample& its)
{
    auto event    = make_event(its.stack_id, its.parent_stack_id, its.correlation_id,
                               its.track_name.data());
    event.extdata = its.event_metadata;

    profiler_hub::writer_types::pmc_event_data_t pmc_data;
    pmc_data.event = event;
    pmc_data.value = 0.0;

    profiler_hub::writer_types::track_info_t track;
    track.name = its.track_name;

    profiler_hub::writer_types::sample_data_t sample;
    sample.timestamp = its.timestamp_ns;
    sample.track     = track;
    pmc_data.sample  = sample;

    profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
    pmc_uid.name = its.track_name;

    try_insert_pmc_event(pmc_data, pmc_uid, "In-time sample");
}

void
rocpd_processor_t::handle(const pmc_event_with_sample& pmc)
{
    const auto& process_info = m_metadata->get_process_info();

    const agent* agent_ptr = nullptr;
    try
    {
        agent_ptr = &m_agent_manager->get_agent_by_type_index(
            pmc.device_id, static_cast<agent_type>(pmc.device_type));
    } catch(const std::out_of_range& e)
    {
        LOG_WARNING("PMC event skipped: agent lookup failed for device_id={}, "
                    "device_type={}: {}",
                    pmc.device_id, pmc.device_type, e.what());
        return;
    }

    const auto& agent_ref = *agent_ptr;
    auto        event = make_event(pmc.stack_id, pmc.parent_stack_id, pmc.correlation_id,
                                   pmc.track_name.data());
    event.extdata     = pmc.event_metadata;

    profiler_hub::writer_types::pmc_event_data_t pmc_data;
    pmc_data.event   = event;
    pmc_data.value   = pmc.value;
    pmc_data.extdata = pmc.event_metadata;

    profiler_hub::writer_types::track_info_t track;
    track.name       = pmc.track_name;
    track.node_id    = node_info::get_instance().id;
    track.process_id = process_info.pid;
    track.thread_id  = pmc.system_tid;

    profiler_hub::writer_types::sample_data_t sample;
    sample.timestamp = pmc.timestamp_ns;
    sample.track     = track;
    pmc_data.sample  = sample;

    profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
    pmc_uid.name     = pmc.pmc_info_name;
    pmc_uid.agent_id = make_agent_uid(agent_ref);

    try_insert_pmc_event(pmc_data, pmc_uid, "PMC event");
}

void
rocpd_processor_t::handle(const counter_batch& batch)
{
    m_counters.for_each_value(batch, [this, &batch](counter_slot& slot, double value) {
        if(slot.skip)
        {
            return;
        }
        insert_counter_value(slot, batch.timestamp, value);
        slot.had_value = true;
    });
}

void
rocpd_processor_t::handle(const gap_marker& gap)
{
    // columns that never had a value get no zero
    m_counters.for_each_slot([this, &gap](const counter_slot& slot) {
        if(!slot.skip && slot.had_value)
        {
            insert_counter_value(slot, gap.begin_ns, 0.0);
        }
    });
}

void
rocpd_processor_t::insert_counter_value(const counter_slot& slot, std::uint64_t timestamp,
                                        double value)
{
    profiler_hub::writer_types::pmc_event_data_t pmc_data;
    pmc_data.event            = make_event(0, 0, 0, slot.category);
    pmc_data.value            = value;
    pmc_data.sample.timestamp = timestamp;
    pmc_data.sample.track     = slot.track;
    try_insert_pmc_event(pmc_data, slot.uid, "Counter sample");
}

void
rocpd_processor_t::build_counter_slots()
{
    const auto process_id = m_metadata->get_process_info().pid;
    m_counters.build(m_metadata->get_counter_schemas(),
                     [this, process_id](const collection::schema&      description,
                                        const collection::column_spec& column) {
                         return make_counter_slot(description, column, process_id);
                     });
}

rocpd_processor_t::counter_slot
rocpd_processor_t::make_counter_slot(
    const collection::schema& description, const collection::column_spec& column,
    profiler_hub::writer_types::process_id_t process_id) const
{
    const agent* pmc_agent = nullptr;
    try
    {
        pmc_agent = &find_pmc_agent(description.agent, description.device);
    } catch(const std::out_of_range& error)
    {
        if(&column == &description.columns.front())
        {
            LOG_WARNING(
                "Counter schema skipped: no agent for agent_type_index={}, type={}: {}",
                description.device, to_string(description.agent), error.what());
        }
        return counter_slot{ .uid = {}, .track = {}, .skip = true };
    }

    counter_slot slot;
    slot.uid.name         = column.pmc_name;
    slot.uid.agent_id     = make_agent_uid(*pmc_agent);
    slot.track.name       = column.rocpd_track_name;
    slot.track.node_id    = node_info::get_instance().id;
    slot.track.process_id = process_id;
    slot.category         = description.event_category.c_str();
    return slot;
}

const agent&
rocpd_processor_t::find_pmc_agent(agent_type type, std::size_t index) const
{
    try
    {
        return m_agent_manager->get_agent_by_type_index(index, type);
    } catch(const std::out_of_range&)
    {
        return m_agent_manager->get_agent_by_id(index, type);
    }
}

void
rocpd_processor_t::handle(const kfd_sample& kfd)
{
    auto const& n_info       = node_info::get_instance();
    auto const  process_info = m_metadata->get_process_info();

    auto event    = make_event(0, 0, 0, kfd.category.data());
    event.extdata = kfd.event_metadata;

    profiler_hub::writer_types::region_data_t region;
    region.event           = event;
    region.start_timestamp = kfd.start_timestamp;
    region.end_timestamp   = kfd.end_timestamp;
    region.name            = kfd.name;

    auto const parsed_args = process_arguments_string(kfd.args_str);
    for(const auto& arg : parsed_args)
    {
        profiler_hub::writer_types::arg_data_t arg_data;
        arg_data.position = arg.arg_number;
        arg_data.type     = arg.arg_type;
        arg_data.name     = arg.arg_name;
        arg_data.value    = arg.arg_value;
        region.args.push_back(arg_data);
    }

    auto const env = make_trace_env(n_info.id, process_info.pid, kfd.thread_id);
    m_writer->insert_region_data(region, env);

    try
    {
        const auto& agent_ref = m_agent_manager->get_agent_by_type_index(
            kfd.device_id, static_cast<agent_type>(kfd.device_type));

        profiler_hub::writer_types::pmc_event_data_t pmc_data;
        pmc_data.event = event;
        pmc_data.value = kfd.value;

        profiler_hub::writer_types::track_info_t track;
        track.name       = kfd.track_name;
        track.node_id    = n_info.id;
        track.process_id = process_info.pid;
        if(kfd.system_tid.has_value())
        {
            track.thread_id = kfd.system_tid.value();
        }

        profiler_hub::writer_types::sample_data_t sample;
        sample.timestamp = kfd.start_timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
        pmc_uid.name     = kfd.pmc_info_name;
        pmc_uid.agent_id = make_agent_uid(agent_ref);

        try_insert_pmc_event(pmc_data, pmc_uid, "KFD PMC event");
    } catch(const std::out_of_range& e)
    {
        LOG_WARNING("KFD PMC event skipped: agent lookup failed for device_id={}, "
                    "device_type={}: {}",
                    kfd.device_id, kfd.device_type, e.what());
    }
}

rocpd_processor_t::rocpd_processor_t(const std::shared_ptr<metadata_registry>& md,
                                     const std::shared_ptr<agent_manager>&     agent_mngr,
                                     pid_t pid, pid_t ppid)
: sample_processor_interface()
, m_metadata(md)
, m_agent_manager(agent_mngr)
, m_pid(pid)
, m_db_output_path(generate_db_output_path(pid))
{
    auto const n_info = node_info::get_instance();
    auto const uuid   = common::md5sum{ n_info.id, pid, ppid }.hexdigest();

    // Remove any existing database file so the writer starts from a clean
    // in-memory state. This handles multiple rocprofsys_init/finalize cycles
    // in the same process (e.g. with ROCPROFSYS_USE_PID=OFF) where successive
    // sessions would otherwise find the file left by the previous session and
    // fail with "Database already initialized!".
    if(std::filesystem::exists(m_db_output_path))
    {
        LOG_WARNING("rocpd output file already exists and will be overwritten: {}. "
                    "Previous profiling data in that file will be lost. "
                    "Set ROCPROFSYS_USE_PID=ON to give each session a unique path.",
                    m_db_output_path);
        std::filesystem::remove(m_db_output_path);
    }

    auto storage = std::make_unique<profiler_hub::storage_t>(m_db_output_path, uuid);
    m_writer     = std::make_unique<profiler_hub::writer_t>(std::move(storage));
}

void
rocpd_processor_t::try_insert_pmc_event(
    const profiler_hub::writer_types::pmc_event_data_t&     event_data,
    const profiler_hub::writer_types::pmc_info_unique_id_t& unique_id,
    std::string_view                                        context)
{
    try
    {
        m_writer->insert_pmc_event_data(event_data, unique_id);
    } catch(const std::runtime_error& e)
    {
        ++m_dropped_pmc_events_count;

        auto key = std::string{ unique_id.name };

        // Build the key for the PMC info. Two agents missing the same PMC info will
        // warn separately.
        if(unique_id.agent_id.has_value())
        {
            const auto& agent_id = unique_id.agent_id.value();
            key += fmt::format(" [{}:{}]", agent_id.agent_type.value_or("unknown"),
                               agent_id.type_index);
        }

        if(m_unregistered_pmcs_already_warned.emplace(std::move(key)).second)
        {
            LOG_WARNING("{} skipped: PMC info not registered for name={} - {}. "
                        "Further samples for this PMC will be dropped without warning.",
                        context, unique_id.name, e.what());
        }
    }
}

void
rocpd_processor_t::prepare_for_processing()
{
    LOG_DEBUG("Preparing rocpd processor for processing");
    post_process_metadata();
    build_counter_slots();
    LOG_TRACE("Rocpd processor prepared for processing");
}

void
rocpd_processor_t::finalize_processing()
{
    LOG_DEBUG("Finalizing rocpd processor");
    try
    {
        m_writer->flush_in_memory_data_to_disk();
    } catch(const std::exception& e)
    {
        // This can happen in multi-process scenarios when two processes attempt
        // to flush their in-memory databases to the same output file at the same
        // time (SQLITE_BUSY). The underlying fix belongs in profiler-hub (retry
        // with back-off or per-PID filenames), but crashing is never appropriate.
        LOG_ERROR("Failed to flush rocpd database to disk ({}): {}. "
                  "Profile data for this process may be incomplete.",
                  m_db_output_path, e.what());
        return;
    }

    output::registry::instance().register_file(m_db_output_path, m_pid);

    if(m_dropped_pmc_events_count > 0)
    {
        // Sorted so the message is reproducible across runs.
        auto counters =
            std::vector<std::string>{ m_unregistered_pmcs_already_warned.begin(),
                                      m_unregistered_pmcs_already_warned.end() };
        std::ranges::sort(counters);

        LOG_WARNING("Rocpd processor finalized with {} PMC event(s) dropped across {} "
                    "unregistered counter(s); {} is incomplete. Counters: {}",
                    m_dropped_pmc_events_count, counters.size(), m_db_output_path,
                    fmt::join(counters, ", "));
    }
    else
    {
        LOG_INFO("Rocpd processor finalized successfully");
    }
}

void
rocpd_processor_t::post_process_metadata()
{
    if(!g_force_rocpd_metadata_registration && !get_use_rocpd())
    {
        LOG_TRACE("Rocpd not enabled, skipping metadata post-processing");
        return;
    }
    LOG_DEBUG("Post-processing metadata for rocpd");
    auto const n_info = node_info::get_instance();

    // Register node info
    profiler_hub::writer_types::node_info_t node;
    node.node_id       = n_info.id;
    node.hash          = n_info.hash;
    node.machine_id    = n_info.machine_id;
    node.system_name   = n_info.system_name;
    node.hostname      = n_info.node_name;
    node.release       = n_info.release;
    node.version       = n_info.version;
    node.hardware_name = n_info.machine;
    node.domain_name   = n_info.domain_name;
    m_writer->register_node_info(node);

    // Register process info
    auto const process_info = m_metadata->get_process_info();
    profiler_hub::writer_types::process_info_t proc;
    proc.ppid    = process_info.ppid;
    proc.pid     = process_info.pid;
    proc.init    = 0;
    proc.fini    = 0;
    proc.start   = process_info.start;
    proc.end     = process_info.end;
    proc.command = process_info.command.c_str();
    proc.environment =
        process_info.environment.empty() ? "{}" : process_info.environment.c_str();
    proc.extdata = process_info.extdata.empty() ? "{}" : process_info.extdata.c_str();
    proc.node_id = n_info.id;
    m_writer->register_process_info(proc);

    // Register agents
    const auto& agents  = m_agent_manager->get_agents();
    int         counter = 0;
    for(const auto& rocpd_agent : agents)
    {
        profiler_hub::writer_types::agent_info_t agent_info;
        agent_info.unique_id      = make_agent_uid(*rocpd_agent);
        agent_info.absolute_index = static_cast<size_t>(counter++);

        agent_info.logical_index = rocpd_agent->logical_node_id;
        agent_info.uuid          = rocpd_agent->device_id;
        agent_info.name          = rocpd_agent->name;
        agent_info.model_name    = rocpd_agent->model_name;
        agent_info.vendor_name   = rocpd_agent->vendor_name;
        agent_info.product_name  = rocpd_agent->product_name;
        agent_info.user_name     = rocpd_agent->product_name;
        agent_info.extdata       = rocpd_agent->agent_info;
        agent_info.node_id       = n_info.id;
        agent_info.process_id    = process_info.pid;
        m_writer->register_agent_info(agent_info);
    }

    // Register strings
    auto const string_list = m_metadata->get_string_list();
    for(auto const& str : string_list)
    {
        m_writer->register_string(str);
    }

    // Register thread info
    auto thread_info_list = m_metadata->get_thread_info_list();
    for(auto& t_info : thread_info_list)
    {
        const auto& extended_info =
            thread_info::get(static_cast<std::int64_t>(t_info.thread_id), SystemTID);
        if(extended_info.has_value())
        {
            t_info.start = extended_info->get_start();
            t_info.end   = extended_info->get_stop();
        }

        auto const thread_name = fmt::format("Thread {}", t_info.thread_id);

        profiler_hub::writer_types::thread_info_t thread_info;
        thread_info.parent_process_id = process_info.ppid;
        thread_info.thread_id         = t_info.thread_id;
        thread_info.name              = thread_name;
        thread_info.start             = t_info.start;
        thread_info.end               = t_info.end;
        thread_info.node_id           = n_info.id;
        thread_info.process_id        = process_info.pid;
        m_writer->register_thread_info(thread_info);
    }

    // Tracks and PMC info of the counter schemas are registered with the registry ones
    auto const counter_schemas = m_metadata->get_counter_schemas();
    register_tracks(counter_schemas, process_info.pid);

    // Register code objects
    auto const code_object_list = m_metadata->get_code_object_list();
    for(const auto& code_object : code_object_list)
    {
        const auto& code_agent = m_agent_manager->get_agent_by_handle(
            get_handle_from_code_object(code_object));

        const char* strg_type = "UNKNOWN";
        switch(code_object.storage_type)
        {
            case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE: strg_type = "FILE"; break;
            case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY: strg_type = "MEMORY"; break;
            default: break;
        }

        profiler_hub::writer_types::code_object_info_t co;
        co.id           = code_object.code_object_id;
        co.uri          = code_object.uri;
        co.load_base    = code_object.load_base;
        co.load_size    = code_object.load_size;
        co.load_delta   = code_object.load_delta;
        co.storage_type = strg_type;
        co.node_id      = n_info.id;
        co.process_id   = process_info.pid;
        co.agent_id     = make_agent_uid(code_agent);
        m_writer->register_code_object_info(co);
    }

    // Register kernel symbols
    auto const kernel_symbols_list = m_metadata->get_kernel_symbol_list();
    for(const auto& kernel_symbol : kernel_symbols_list)
    {
        auto const kernel_name = rocprofsys::utility::demangle(kernel_symbol.kernel_name);

        profiler_hub::writer_types::kernel_symbol_info_t ksi;
        ksi.id                        = kernel_symbol.kernel_id;
        ksi.name                      = kernel_symbol.kernel_name;
        ksi.display_name              = kernel_name;
        ksi.kernel_object             = kernel_symbol.kernel_object;
        ksi.kernarg_segment_size      = kernel_symbol.kernarg_segment_size;
        ksi.kernarg_segment_alignment = kernel_symbol.kernarg_segment_alignment;
        ksi.group_segment_size        = kernel_symbol.group_segment_size;
        ksi.private_segment_size      = kernel_symbol.private_segment_size;
        ksi.sgpr_count                = kernel_symbol.sgpr_count;
        ksi.arch_vgpr_count           = kernel_symbol.arch_vgpr_count;
        ksi.accum_vgpr_count          = kernel_symbol.accum_vgpr_count;
        ksi.node_id                   = n_info.id;
        ksi.process_id                = process_info.pid;
        ksi.code_obj_id               = kernel_symbol.code_object_id;
        m_writer->register_kernel_symbol_info(ksi);

        m_writer->register_string(kernel_name);
    }

    // Register queue info
    auto const queue_list = m_metadata->get_queue_list();
    for(const auto& queue_handle : queue_list)
    {
        auto const queue_name = fmt::format("Queue {}", queue_handle);

        profiler_hub::writer_types::queue_info_t qi;
        qi.queue_id   = queue_handle;
        qi.name       = queue_name;
        qi.node_id    = n_info.id;
        qi.process_id = process_info.pid;
        m_writer->register_queue_info(qi);
    }

    // Register stream info
    auto const stream_list = m_metadata->get_stream_list();
    for(const auto& stream_handle : stream_list)
    {
        auto const stream_name = fmt::format("Stream {}", stream_handle);

        profiler_hub::writer_types::stream_info_t str_info;
        str_info.stream_id  = stream_handle;
        str_info.name       = stream_name;
        str_info.node_id    = n_info.id;
        str_info.process_id = process_info.pid;
        m_writer->register_stream_info(str_info);
    }

    register_pmc_infos(counter_schemas, process_info.pid);
}

void
rocpd_processor_t::register_tracks(const std::vector<collection::schema>& counter_schemas,
                                   profiler_hub::writer_types::process_id_t process_id)
{
    const auto node_id = node_info::get_instance().id;
    auto const track_info_list =
        merge_counter_tracks(m_metadata->get_track_info_list(), counter_schemas);
    for(auto const& track : track_info_list)
    {
        profiler_hub::writer_types::track_info_t track_info;
        track_info.name       = track.track_name;
        track_info.node_id    = node_id;
        track_info.process_id = process_id;
        track_info.thread_id  = track.thread_id;
        m_writer->register_track_info(track_info);
    }
}

void
rocpd_processor_t::register_pmc_infos(
    const std::vector<collection::schema>&   counter_schemas,
    profiler_hub::writer_types::process_id_t process_id)
{
    const auto node_id = node_info::get_instance().id;
    auto const pmc_info_list =
        merge_counter_pmc_infos(m_metadata->get_pmc_info_list(), counter_schemas);
    for(const auto& pmc_info : pmc_info_list)
    {
        const agent* pmc_agent = nullptr;
        try
        {
            pmc_agent = &find_pmc_agent(pmc_info.type, pmc_info.agent_type_index);
        } catch(const std::out_of_range& error)
        {
            LOG_WARNING("PMC info registration skipped: agent lookup failed for "
                        "agent_type_index={}, type={}: {}",
                        pmc_info.agent_type_index, to_string(pmc_info.type),
                        error.what());
            continue;
        }

        auto pmc_agent_uid = make_agent_uid(*pmc_agent);

        LOG_TRACE("Inserting PMC description: agent_uid: {}, pmc_info: {}",
                  pmc_agent_uid.type_index, pmc_info.name);

        profiler_hub::writer_types::pmc_info_t           pmc_info_data;
        profiler_hub::writer_types::pmc_info_unique_id_t uid;
        uid.name                = pmc_info.name;
        uid.agent_id            = pmc_agent_uid;
        pmc_info_data.unique_id = uid;
        // pmc_info_t uses string_view fields throughout, this is safe here because
        // register_pmc_info() consumes the data synchronously (SQLite bind + step)
        // and does NOT retain the string_view after returning. If that contract
        // ever changes, target_arch (and the other string_view fields in pmc_info_t)
        // must be changed to std::string.
        pmc_info_data.target_arch =
            pmc_info.target_arch.empty()
                ? std::nullopt
                : std::optional<std::string_view>{ pmc_info.target_arch };
        pmc_info_data.event_code       = pmc_info.event_code;
        pmc_info_data.instance_id      = pmc_info.instance_id;
        pmc_info_data.symbol           = pmc_info.symbol;
        pmc_info_data.description      = pmc_info.description;
        pmc_info_data.long_description = pmc_info.long_description;
        pmc_info_data.component        = pmc_info.component;
        pmc_info_data.units            = pmc_info.units;
        pmc_info_data.value_type       = pmc_info.value_type;
        pmc_info_data.block            = pmc_info.block;
        pmc_info_data.expression       = pmc_info.expression;
        pmc_info_data.is_constant      = pmc_info.is_constant;
        pmc_info_data.is_derived       = pmc_info.is_derived;
        pmc_info_data.node_id          = node_id;
        pmc_info_data.process_id       = process_id;
        m_writer->register_pmc_info(pmc_info_data);
    }
}

}  // namespace rocprofsys::trace_cache
