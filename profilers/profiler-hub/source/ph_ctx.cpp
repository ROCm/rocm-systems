#include "ph_ctx.hpp"
#include "debug.hpp"
#include "file_probe_failure.hpp"
#include "ph_future.hpp"
#include "populate_reader_catalog.hpp"
#include "profiler-hub/cpp/reader.hpp"
#include "reader_catalog.hpp"
#include "track_read_options.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace
{

ph_track_category_t
to_c_track_category(profiler_hub::reader_types::track_kind_t kind)
{
    using profiler_hub::reader_types::track_kind_t;
    switch(kind)
    {
        case track_kind_t::thread: return PH_TRACK_CATEGORY_THREAD;
        case track_kind_t::thread_sample: return PH_TRACK_CATEGORY_THREAD_SAMPLE;
        case track_kind_t::pmc_agent: return PH_TRACK_CATEGORY_PMC_AGENT;
        case track_kind_t::kernel_dispatch_agent_queue:
            return PH_TRACK_CATEGORY_KERNEL_DISPATCH_AGENT_QUEUE;
        case track_kind_t::memory_allocate_agent_queue:
            return PH_TRACK_CATEGORY_MEMORY_ALLOCATE_AGENT_QUEUE;
        case track_kind_t::memory_copy_agent_queue:
            return PH_TRACK_CATEGORY_MEMORY_COPY_AGENT_QUEUE;
        case track_kind_t::stream: return PH_TRACK_CATEGORY_STREAM;
    }
    return PH_TRACK_CATEGORY_THREAD;
}

ph_optional_uint64_t
optional_uint64(const std::optional<size_t>& value)
{
    if(!value.has_value())
    {
        return ph_optional_uint64_t{ .value = 0, .is_valid = 0 };
    }
    return ph_optional_uint64_t{ .value    = static_cast<uint64_t>(value.value()),
                                 .is_valid = 1 };
}

ph_optional_uint64_t
optional_track_id(const profiler_hub::reader_types::track_info_ptr_t& track)
{
    if(!track)
    {
        return ph_optional_uint64_t{ .value = 0, .is_valid = 0 };
    }
    return ph_optional_uint64_t{ .value    = static_cast<uint64_t>(track->id),
                                 .is_valid = 1 };
}

std::optional<profiler_hub::reader_types::event_type_t>
to_reader_event_type(ph_event_type_t type)
{
    using profiler_hub::reader_types::event_type_t;
    switch(type)
    {
        case PH_EVENT_TYPE_REGION: return event_type_t::region;
        case PH_EVENT_TYPE_KERNEL_DISPATCH: return event_type_t::kernel_dispatch;
        case PH_EVENT_TYPE_MEMORY_COPY: return event_type_t::memory_copy;
        case PH_EVENT_TYPE_MEMORY_ALLOCATE: return event_type_t::memory_allocate;
    }
    return std::nullopt;
}

ph_event_type_t
to_c_event_type(profiler_hub::reader_types::event_type_t type)
{
    using profiler_hub::reader_types::event_type_t;
    switch(type)
    {
        case event_type_t::region: return PH_EVENT_TYPE_REGION;
        case event_type_t::kernel_dispatch: return PH_EVENT_TYPE_KERNEL_DISPATCH;
        case event_type_t::memory_copy: return PH_EVENT_TYPE_MEMORY_COPY;
        case event_type_t::memory_allocate: return PH_EVENT_TYPE_MEMORY_ALLOCATE;
        case event_type_t::sample:
        case event_type_t::pmc_event: break;
    }
    throw std::invalid_argument{ "event type is not a duration event" };
}

bool
track_holds_event(const profiler_hub::reader_types::track_info_ptr_t& track,
                  profiler_hub::reader_types::event_type_t            type)
{
    using profiler_hub::reader_types::event_type_t;
    using profiler_hub::reader_types::track_kind_t;
    if(!track) return false;
    switch(track->category)
    {
        case track_kind_t::kernel_dispatch_agent_queue:
            return type == event_type_t::kernel_dispatch;
        case track_kind_t::memory_allocate_agent_queue:
            return type == event_type_t::memory_allocate;
        case track_kind_t::memory_copy_agent_queue:
            return type == event_type_t::memory_copy;
        case track_kind_t::stream: return type != event_type_t::region;
        case track_kind_t::thread:
        case track_kind_t::thread_sample: return true;
        case track_kind_t::pmc_agent: return false;
    }
    return false;
}

const char*
text_or_empty(const std::string& text)
{
    return text.c_str();
}

ph_optional_uint64_t
optional_size(const std::optional<size_t>& value)
{
    if(!value.has_value())
    {
        return ph_optional_uint64_t{ .value = 0, .is_valid = 0 };
    }
    return ph_optional_uint64_t{ .value    = static_cast<uint64_t>(value.value()),
                                 .is_valid = 1 };
}

ph_event_data_t
to_c_event_data(const profiler_hub::reader_types::event_data_ptr_t& event)
{
    ph_event_data_t view{};
    if(!event) return view;
    view.is_valid        = 1;
    view.stack_id        = event->stack_id;
    view.parent_stack_id = event->parent_stack_id;
    view.correlation_id  = event->correlation_id;
    view.event_category  = text_or_empty(event->event_category);
    view.extdata         = text_or_empty(event->extdata);
    return view;
}

ph_detail_node_t
to_c_detail_node(const profiler_hub::reader_types::node_info_ptr_t& node)
{
    ph_detail_node_t view{};
    if(!node) return view;
    view.is_valid      = 1;
    view.node_id       = node->node_id;
    view.hash          = node->hash;
    view.machine_id    = text_or_empty(node->machine_id);
    view.system_name   = text_or_empty(node->system_name);
    view.hostname      = text_or_empty(node->hostname);
    view.release       = text_or_empty(node->release);
    view.version       = text_or_empty(node->version);
    view.hardware_name = text_or_empty(node->hardware_name);
    view.domain_name   = text_or_empty(node->domain_name);
    return view;
}

ph_detail_process_t
to_c_detail_process(const profiler_hub::reader_types::process_info_ptr_t& process)
{
    ph_detail_process_t view{};
    if(!process) return view;
    view.is_valid    = 1;
    view.id          = process->id;
    view.pid         = process->pid;
    view.ppid        = optional_size(process->ppid);
    view.init        = optional_size(process->init);
    view.fini        = optional_size(process->fini);
    view.start       = optional_size(process->start);
    view.end         = optional_size(process->end);
    view.command     = text_or_empty(process->command);
    view.environment = text_or_empty(process->environment);
    view.extdata     = text_or_empty(process->extdata);
    return view;
}

ph_detail_thread_t
to_c_detail_thread(const profiler_hub::reader_types::thread_info_ptr_t& thread)
{
    ph_detail_thread_t view{};
    if(!thread) return view;
    view.is_valid          = 1;
    view.id                = thread->id;
    view.thread_id         = thread->thread_id;
    view.parent_process_id = optional_size(thread->parent_process_id);
    view.start             = optional_size(thread->start);
    view.end               = optional_size(thread->end);
    view.name              = text_or_empty(thread->name);
    view.extdata           = text_or_empty(thread->extdata);
    return view;
}

ph_detail_agent_t
to_c_detail_agent(const profiler_hub::reader_types::agent_info_ptr_t& agent)
{
    ph_detail_agent_t view{};
    if(!agent) return view;
    view.is_valid       = 1;
    view.id             = agent->id;
    view.type_index     = agent->type_index;
    view.absolute_index = optional_size(agent->absolute_index);
    view.logical_index  = optional_size(agent->logical_index);
    view.uuid           = optional_size(agent->uuid);
    view.agent_type     = text_or_empty(agent->agent_type);
    view.name           = text_or_empty(agent->name);
    view.model_name     = text_or_empty(agent->model_name);
    view.vendor_name    = text_or_empty(agent->vendor_name);
    view.product_name   = text_or_empty(agent->product_name);
    view.user_name      = text_or_empty(agent->user_name);
    view.extdata        = text_or_empty(agent->extdata);
    return view;
}

ph_detail_stream_t
to_c_detail_stream(const profiler_hub::reader_types::stream_info_ptr_t& stream)
{
    ph_detail_stream_t view{};
    if(!stream) return view;
    view.is_valid  = 1;
    view.stream_id = stream->stream_id;
    view.nid       = stream->nid;
    view.pid       = stream->pid;
    view.name      = text_or_empty(stream->name);
    view.extdata   = text_or_empty(stream->extdata);
    return view;
}

ph_detail_queue_t
to_c_detail_queue(const profiler_hub::reader_types::queue_info_ptr_t& queue)
{
    ph_detail_queue_t view{};
    if(!queue) return view;
    view.is_valid = 1;
    view.queue_id = queue->queue_id;
    view.nid      = queue->nid;
    view.pid      = queue->pid;
    view.name     = text_or_empty(queue->name);
    view.extdata  = text_or_empty(queue->extdata);
    return view;
}

ph_detail_code_object_t
to_c_detail_code_object(const profiler_hub::reader_types::code_object_info_ptr_t& object)
{
    ph_detail_code_object_t view{};
    if(!object) return view;
    view.is_valid     = 1;
    view.id           = object->id;
    view.load_base    = optional_size(object->load_base);
    view.load_size    = optional_size(object->load_size);
    view.load_delta   = optional_size(object->load_delta);
    view.uri          = text_or_empty(object->uri);
    view.storage_type = text_or_empty(object->storage_type);
    view.extdata      = text_or_empty(object->extdata);
    return view;
}

ph_detail_kernel_symbol_t
to_c_detail_kernel_symbol(
    const profiler_hub::reader_types::kernel_symbol_info_ptr_t& symbol)
{
    ph_detail_kernel_symbol_t view{};
    if(!symbol) return view;
    view.is_valid                  = 1;
    view.id                        = symbol->id;
    view.kernel_object             = optional_size(symbol->kernel_object);
    view.kernarg_segment_size      = optional_size(symbol->kernarg_segment_size);
    view.kernarg_segment_alignment = optional_size(symbol->kernarg_segment_alignment);
    view.group_segment_size        = optional_size(symbol->group_segment_size);
    view.private_segment_size      = optional_size(symbol->private_segment_size);
    view.sgpr_count                = optional_size(symbol->sgpr_count);
    view.arch_vgpr_count           = optional_size(symbol->arch_vgpr_count);
    view.accum_vgpr_count          = optional_size(symbol->accum_vgpr_count);
    view.name                      = text_or_empty(symbol->name);
    view.display_name              = text_or_empty(symbol->display_name);
    view.extdata                   = text_or_empty(symbol->extdata);
    return view;
}

ph_address_range_t
to_c_address_range(
    const std::optional<profiler_hub::reader_types::address_range_info_t>& range)
{
    ph_address_range_t view{};
    if(!range.has_value()) return view;
    view.is_valid     = 1;
    view.address_base = static_cast<uint64_t>(range->address_base);
    view.address_low  = static_cast<uint64_t>(range->address_low);
    view.address_high = static_cast<uint64_t>(range->address_high);
    view.extdata      = text_or_empty(range->extdata);
    return view;
}

ph_program_counter_t
to_c_program_counter(
    const std::optional<profiler_hub::reader_types::program_counter_info_t>& counter)
{
    ph_program_counter_t view{};
    if(!counter.has_value()) return view;
    view.is_valid    = 1;
    view.event_id    = static_cast<uint64_t>(counter->event_id);
    view.function    = text_or_empty(counter->function);
    view.filename    = text_or_empty(counter->filename);
    view.line_number = optional_size(counter->line_number);
    view.extdata     = text_or_empty(counter->extdata);
    return view;
}

ph_stack_frame_t
to_c_stack_frame(const profiler_hub::reader_types::stack_frame_t& frame)
{
    return ph_stack_frame_t{
        .program_counter = to_c_program_counter(frame.program_counter),
        .address_range   = to_c_address_range(frame.address_range),
        .extdata         = text_or_empty(frame.extdata),
    };
}

profiler_hub::reader_types::timeline_event_t
detail_source(uint64_t event_id, profiler_hub::reader_types::event_type_t type)
{
    profiler_hub::reader_types::timeline_event_t source;
    source.unique_identifier = { .id = static_cast<size_t>(event_id), .type = type };
    return source;
}

std::uint32_t
to_list_size(std::size_t size)
{
    if(size > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::length_error{ "result has more elements than the C API can report" };
    }
    return static_cast<std::uint32_t>(size);
}

}  // namespace

size_t
ph_ctx::default_thread_pool_size()
{
    return std::max<size_t>(1, std::thread::hardware_concurrency() / 2);
}

size_t
ph_ctx::default_connection_count()
{
    return 8;
}

ph_ctx::accepted_trace_t
ph_ctx::accept_trace(std::string_view trace_path)
{
    const auto probe = profiler_hub::probe_file(std::string{ trace_path });
    if(probe.status != PH_RESULT_SUCCESS)
    {
        throw profiler_hub::file_probe_failure{ probe };
    }
    return accepted_trace_t{ std::string{ trace_path },
                             ph_schema_version_t{ .major = probe.schema_version.major,
                                                  .minor = probe.schema_version.minor,
                                                  .patch = probe.schema_version.patch } };
}

ph_ctx::ph_ctx(std::string_view trace_path)
: ph_ctx{ accept_trace(trace_path) }
{}

ph_ctx::ph_ctx(accepted_trace_t accepted)
: m_file_path{ std::move(accepted.path) }
, m_schema_version{ accepted.version }
, m_catalog{ std::make_shared<profiler_hub::reader_catalog_t>() }
{}

ph_ctx::~ph_ctx()
{
    if(!m_load_task) return;

    std::ignore = m_load_task->cancel();
    m_load_task->wait();

    // A load that never got a worker did not run, so nobody ended its future.
    set_load_state_if_loading(load_state_t::cancelled);
    m_load_future->finish(PH_FUTURE_CANCELLED, PH_RESULT_CANCELLED);
}

void
ph_ctx::load()
{
    try
    {
        std::ignore = read_trace([] { return false; }, {});
    } catch(...)
    {
        set_load_state(load_state_t::failed);
        throw;
    }
    set_load_state(load_state_t::ready);
}

void
ph_ctx::load_async(std::shared_ptr<ph_future> future)
{
    m_load_future = future;
    m_load_task   = m_thread_pool.submit([this, future](const std::stop_token& token) {
        const auto stop_requested = [&] {
            return token.stop_requested() || future->stop_token().stop_requested();
        };
        const auto progress = [&](double value, ph_progress_description_t description) {
            future->report_progress(value, description);
        };

        try
        {
            if(read_trace(stop_requested, progress))
            {
                set_load_state(load_state_t::ready);
                future->finish(PH_FUTURE_FINISHED, PH_RESULT_SUCCESS);
            }
            else
            {
                set_load_state(load_state_t::cancelled);
                future->finish(PH_FUTURE_CANCELLED, PH_RESULT_CANCELLED);
            }
        } catch(...)
        {
            set_load_state(load_state_t::failed);
            future->finish(PH_FUTURE_ERROR, PH_RESULT_CONTEXT_ALLOCATION_FAILED);
        }
    });
}

void
ph_ctx::run_async(std::shared_ptr<ph_future>   future,
                  std::function<ph_result_t()> operation)
{
    m_operations.add(future);
    try
    {
        std::ignore =
            m_thread_pool.submit([this, future, operation = std::move(operation)](
                                     const std::stop_token& token) {
                if(token.stop_requested() || future->stop_token().stop_requested())
                {
                    future->finish(PH_FUTURE_CANCELLED, PH_RESULT_CANCELLED);
                }
                else
                {
                    const auto result = operation();
                    future->finish(result == PH_RESULT_SUCCESS ? PH_FUTURE_FINISHED
                                                               : PH_FUTURE_ERROR,
                                   result);
                }
                m_operations.remove(future);
            });
    } catch(...)
    {
        m_operations.remove(future);
        throw;
    }
}

ph_result_t
ph_ctx::wait_until_ready()
{
    std::unique_lock lock{ m_state_mutex };
    m_state_cv.wait(lock, [this] { return m_state != load_state_t::loading; });

    switch(m_state)
    {
        case load_state_t::ready: return PH_RESULT_SUCCESS;
        case load_state_t::cancelled: return PH_RESULT_CANCELLED;
        case load_state_t::loading:
        case load_state_t::failed: break;
    }
    return PH_RESULT_INVALID_CONTEXT;
}

void
ph_ctx::set_load_state(load_state_t state)
{
    {
        const std::scoped_lock lock{ m_state_mutex };
        m_state = state;
    }
    m_state_cv.notify_all();
}

void
ph_ctx::set_load_state_if_loading(load_state_t state)
{
    {
        const std::scoped_lock lock{ m_state_mutex };
        if(m_state != load_state_t::loading) return;
        m_state = state;
    }
    m_state_cv.notify_all();
}

bool
ph_ctx::read_trace(const stop_requested_fn& stop_requested, const progress_fn& progress)
{
    if(stop_requested()) return false;

    if(progress) progress(0.0, "Reading schema version");

    if(progress) progress(0.0, "Building reader catalog");
    populate_reader_catalog(m_thread_pool, m_connection_pool, *m_catalog);
    if(stop_requested()) return false;

    if(progress) progress(0.0, "Building track list");
    initialize_track_list();
    if(!load_all_tracks(stop_requested, progress)) return false;

    if(progress) progress(1.0, "Reading node agents");
    initialize_node_agents();
    if(progress) progress(1.0, "Reading processes");
    initialize_node_processes();
    if(progress) progress(1.0, "Reading threads");
    initialize_node_threads();
    if(progress) progress(1.0, "Reading streams");
    initialize_node_streams();
    if(progress) progress(1.0, "Reading queues");
    initialize_node_queues();
    if(progress) progress(1.0, "Reading PMC info");
    initialize_node_pmc_infos();
    if(progress) progress(1.0, "Reading stream hardware links");
    initialize_stream_hardware_links();
    if(progress) progress(1.0, "Reading node info");
    initialize_node_info();
    return true;
}

ph_schema_version_t
ph_ctx::get_schema_version()
{
    return m_schema_version;
}

ph_track_list_t
ph_ctx::get_track_list()
{
    LOG_DEBUG("[Profiler-Hub] Get track list");
    return ph_track_list_t{ .list_size = static_cast<std::uint32_t>(m_c_tracks.size()),
                            .tracks    = m_c_tracks.data() };
}

ph_node_t
ph_ctx::get_node()
{
    return *m_c_node;
}

ph_event_list_t
ph_ctx::get_track_events(uint32_t track_id, uint64_t start_ts, uint64_t end_ts)
{
    LOG_DEBUG("[Profiler-Hub] Get track events. Track id {}, time slice [{} - {}]",
              track_id,
              start_ts,
              end_ts);
    const auto track_it = m_track_by_id.find(track_id);
    if(track_it == m_track_by_id.end())
    {
        return ph_event_list_t{ .list_size = 0, .events = nullptr };
    }

    return m_track_reader.events(track_it->second, start_ts, end_ts);
}

std::optional<ph_event_list_t>
ph_ctx::get_correlated_events(uint64_t event_id, ph_event_type_t event_type)
{
    LOG_DEBUG("[Profiler-Hub] Get correlated events. Event id {}, type {}",
              event_id,
              static_cast<int>(event_type));

    const auto type = to_reader_event_type(event_type);
    if(!type.has_value())
    {
        return std::nullopt;
    }

    profiler_hub::reader_types::timeline_event_t source;
    source.unique_identifier = { .id   = static_cast<size_t>(event_id),
                                 .type = type.value() };

    const auto correlated = m_connection_pool.with_connection(
        [&source](profiler_hub::common::connection& connection) {
            return connection.reader().get_correlated_events(source);
        });

    std::vector<ph_event_t> converted;
    converted.reserve(correlated.size());
    for(const auto& event : correlated)
    {
        const ph_event_type_t correlated_type =
            to_c_event_type(event.unique_identifier.type);
        const uint64_t          correlated_id = event.unique_identifier.id;
        std::optional<uint32_t> depth;
        profiler_hub::reader_types::track_info_ptr_t placed_track;
        if(event.track)
        {
            depth = m_track_reader.nesting_depth_of(
                event.track, correlated_id, correlated_type);
            if(depth.has_value()) placed_track = event.track;
        }
        if(!depth.has_value())
        {
            for(const auto& track : m_catalog->tracks)
            {
                if(track == event.track ||
                   !track_holds_event(track, event.unique_identifier.type))
                {
                    continue;
                }
                depth = m_track_reader.nesting_depth_of(
                    track, correlated_id, correlated_type);
                if(depth.has_value())
                {
                    placed_track = track;
                    break;
                }
            }
        }
        if(!depth.has_value() || !placed_track)
        {
            return std::nullopt;
        }

        converted.push_back(ph_event_t{
            .id            = correlated_id,
            .start         = event.start_timestamp,
            .end           = event.end_timestamp,
            .name          = event.display_name.empty() ? "" : event.display_name.data(),
            .type          = correlated_type,
            .nesting_depth = depth.value(),
            .track_id      = static_cast<uint32_t>(placed_track->id),
        });
    }

    if(converted.empty())
    {
        return ph_event_list_t{ .list_size = 0, .events = nullptr };
    }

    const std::lock_guard<std::mutex> lock{ m_correlated_events_mutex };
    m_correlated_events.push_back(std::move(converted));
    auto& stored = m_correlated_events.back();
    return ph_event_list_t{ .list_size = to_list_size(stored.size()),
                            .events    = stored.data() };
}

ph_call_stack_t
ph_ctx::get_call_stack(uint64_t event_id, ph_event_type_t event_type)
{
    LOG_DEBUG("[Profiler-Hub] Get call stack. Event id {}, type {}",
              event_id,
              static_cast<int>(event_type));

    const auto type = to_reader_event_type(event_type);
    if(!type.has_value())
    {
        return ph_call_stack_t{ .list_size = 0, .frames = nullptr };
    }

    const auto source = detail_source(event_id, type.value());
    auto       stack  = m_connection_pool.with_connection(
        [&source](profiler_hub::common::connection& connection) {
            return connection.reader().get_call_stack(source);
        });
    if(stack.empty())
    {
        return ph_call_stack_t{ .list_size = 0, .frames = nullptr };
    }

    auto held  = std::make_unique<call_stack_hold>();
    held->data = std::move(stack);
    held->frames.reserve(held->data.size());
    for(const auto& frame : held->data)
    {
        held->frames.push_back(to_c_stack_frame(frame));
    }

    const ph_call_stack_t view{ .list_size = to_list_size(held->frames.size()),
                                .frames    = held->frames.data() };
    const std::lock_guard<std::mutex> lock{ m_call_stack_mutex };
    m_call_stacks.push_back(std::move(held));
    return view;
}

std::optional<uint32_t>
ph_ctx::track_id_of(uint64_t event_id, ph_event_type_t event_type)
{
    const auto type = to_reader_event_type(event_type);
    if(!type.has_value()) return std::nullopt;

    profiler_hub::reader_types::timeline_event_t source;
    source.unique_identifier = { .id   = static_cast<size_t>(event_id),
                                 .type = type.value() };
    const auto track = m_connection_pool.with_connection(
        [&source](profiler_hub::common::connection& connection) {
            return connection.reader().track_of(source);
        });
    if(!track) return std::nullopt;
    return static_cast<uint32_t>(track->id);
}

ph_region_details_t
ph_ctx::get_region_details(uint64_t event_id)
{
    const auto source =
        detail_source(event_id, profiler_hub::reader_types::event_type_t::region);
    auto details = m_connection_pool.with_connection(
        [&source](profiler_hub::common::connection& connection) {
            return connection.reader().get_region_details(source);
        });
    if(!details.has_value())
    {
        return ph_region_details_t{};
    }

    auto held  = std::make_unique<region_detail_hold>();
    held->data = std::move(details.value());
    held->args.reserve(held->data.args.size());
    for(const auto& arg : held->data.args)
    {
        held->args.push_back(ph_arg_data_t{ .position = arg.position,
                                            .type     = text_or_empty(arg.type),
                                            .name     = text_or_empty(arg.name),
                                            .value    = text_or_empty(arg.value),
                                            .extdata  = text_or_empty(arg.extdata) });
    }
    held->view = ph_region_details_t{
        .is_valid        = 1,
        .start_timestamp = held->data.start_timestamp,
        .end_timestamp   = held->data.end_timestamp,
        .name            = text_or_empty(held->data.name),
        .extdata         = text_or_empty(held->data.extdata),
        .event           = to_c_event_data(held->data.event),
        .arg_count       = to_list_size(held->args.size()),
        .args            = held->args.empty() ? nullptr : held->args.data(),
    };
    const ph_region_details_t         view = held->view;
    const std::lock_guard<std::mutex> lock{ m_detail_mutex };
    m_region_details.push_back(std::move(held));
    return view;
}

ph_kernel_dispatch_details_t
ph_ctx::get_kernel_dispatch_details(uint64_t event_id)
{
    const auto source = detail_source(
        event_id, profiler_hub::reader_types::event_type_t::kernel_dispatch);
    auto details = m_connection_pool.with_connection(
        [&source](profiler_hub::common::connection& connection) {
            return connection.reader().get_kernel_dispatch_details(source);
        });
    if(!details.has_value())
    {
        return ph_kernel_dispatch_details_t{};
    }

    auto held        = std::make_unique<kernel_detail_hold>();
    held->data       = std::move(details.value());
    const auto& data = held->data;
    held->view       = ph_kernel_dispatch_details_t{
              .is_valid             = 1,
              .dispatch_id          = data.dispatch_id,
              .start_timestamp      = data.start_timestamp,
              .end_timestamp        = data.end_timestamp,
              .private_segment_size = data.private_segment_size.value_or(0),
              .group_segment_size   = data.group_segment_size.value_or(0),
              .workgroup_size_x     = data.workgroup_size_x,
              .workgroup_size_y     = data.workgroup_size_y,
              .workgroup_size_z     = data.workgroup_size_z,
              .grid_size_x          = data.grid_size_x,
              .grid_size_y          = data.grid_size_y,
              .grid_size_z          = data.grid_size_z,
              .name                 = text_or_empty(data.name),
              .extdata              = text_or_empty(data.extdata),
              .event                = to_c_event_data(data.event),
              .node_info            = to_c_detail_node(data.node_info),
              .process_info         = to_c_detail_process(data.process_info),
              .thread_info          = to_c_detail_thread(data.thread_info),
              .agent_info           = to_c_detail_agent(data.agent_info),
              .kernel_symbol_info   = to_c_detail_kernel_symbol(data.kernel_symbol_info),
              .code_object_info     = to_c_detail_code_object(data.code_object_info),
              .stream_info          = to_c_detail_stream(data.stream_info),
              .queue_info           = to_c_detail_queue(data.queue_info),
    };
    const ph_kernel_dispatch_details_t view = held->view;
    const std::lock_guard<std::mutex>  lock{ m_detail_mutex };
    m_kernel_details.push_back(std::move(held));
    return view;
}

ph_memory_copy_details_t
ph_ctx::get_memory_copy_details(uint64_t event_id)
{
    const auto source =
        detail_source(event_id, profiler_hub::reader_types::event_type_t::memory_copy);
    auto details = m_connection_pool.with_connection(
        [&source](profiler_hub::common::connection& connection) {
            return connection.reader().get_memory_copy_details(source);
        });
    if(!details.has_value())
    {
        return ph_memory_copy_details_t{};
    }

    auto held        = std::make_unique<copy_detail_hold>();
    held->data       = std::move(details.value());
    const auto& data = held->data;
    held->view       = ph_memory_copy_details_t{
              .is_valid        = 1,
              .start_timestamp = data.start_timestamp,
              .end_timestamp   = data.end_timestamp,
              .dst_address     = optional_size(data.dst_address),
              .src_address     = optional_size(data.src_address),
              .size            = data.size,
              .name            = text_or_empty(data.name),
              .region_name     = text_or_empty(data.region_name),
              .extdata         = text_or_empty(data.extdata),
              .event           = to_c_event_data(data.event),
              .dst_agent_id    = to_c_detail_agent(data.dst_agent_id),
              .src_agent_id    = to_c_detail_agent(data.src_agent_id),
              .node_info       = to_c_detail_node(data.node_info),
              .process_info    = to_c_detail_process(data.process_info),
              .thread_info     = to_c_detail_thread(data.thread_info),
              .stream_info     = to_c_detail_stream(data.stream_info),
              .queue_info      = to_c_detail_queue(data.queue_info),
    };
    const ph_memory_copy_details_t    view = held->view;
    const std::lock_guard<std::mutex> lock{ m_detail_mutex };
    m_copy_details.push_back(std::move(held));
    return view;
}

ph_memory_alloc_details_t
ph_ctx::get_memory_alloc_details(uint64_t event_id)
{
    const auto source = detail_source(
        event_id, profiler_hub::reader_types::event_type_t::memory_allocate);
    auto details = m_connection_pool.with_connection(
        [&source](profiler_hub::common::connection& connection) {
            return connection.reader().get_memory_alloc_details(source);
        });
    if(!details.has_value())
    {
        return ph_memory_alloc_details_t{};
    }

    auto held        = std::make_unique<alloc_detail_hold>();
    held->data       = std::move(details.value());
    const auto& data = held->data;
    held->view       = ph_memory_alloc_details_t{
              .is_valid        = 1,
              .start_timestamp = data.start_timestamp,
              .end_timestamp   = data.end_timestamp,
              .address         = optional_size(data.address),
              .size            = data.size,
              .type            = text_or_empty(data.type),
              .level           = text_or_empty(data.level),
              .extdata         = text_or_empty(data.extdata),
              .event           = to_c_event_data(data.event),
              .node_info       = to_c_detail_node(data.node_info),
              .process_info    = to_c_detail_process(data.process_info),
              .thread_info     = to_c_detail_thread(data.thread_info),
              .agent_info      = to_c_detail_agent(data.agent_info),
              .stream_info     = to_c_detail_stream(data.stream_info),
              .queue_info      = to_c_detail_queue(data.queue_info),
    };
    const ph_memory_alloc_details_t   view = held->view;
    const std::lock_guard<std::mutex> lock{ m_detail_mutex };
    m_alloc_details.push_back(std::move(held));
    return view;
}

ph_sample_list_t
ph_ctx::get_track_samples(uint32_t track_id, uint64_t start_ts, uint64_t end_ts)
{
    LOG_DEBUG("[Profiler-Hub] Get track samples. Track id {}, time slice [{} - {}]",
              track_id,
              start_ts,
              end_ts);
    const auto track_it = m_track_by_id.find(track_id);
    if(track_it == m_track_by_id.end())
    {
        return ph_sample_list_t{ .list_size = 0, .samples = nullptr };
    }

    return m_track_reader.samples(track_it->second, start_ts, end_ts);
}

namespace
{
std::uint32_t
narrow_histogram_count(std::size_t value)
{
    if(value > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::overflow_error{ "histogram value does not fit uint32_t" };
    }
    return static_cast<std::uint32_t>(value);
}
}  // namespace

void
ph_ctx::build_histogram(uint32_t max_buckets, ph_histogram_t& output)
{
    const auto source = m_connection_pool.with_connection(
        [max_buckets](profiler_hub::common::connection& connection) {
            return connection.reader().get_histogram(max_buckets);
        });

    auto storage = std::make_unique<c_histogram_storage_t>();
    storage->tracks.resize(source.tracks.size());
    storage->buckets.resize(source.tracks.size());

    for(std::size_t track_index = 0; track_index < source.tracks.size(); ++track_index)
    {
        const auto& source_track = source.tracks[track_index];
        auto&       destination  = storage->buckets[track_index];
        destination.reserve(source_track.buckets.size());
        for(const auto& bucket : source_track.buckets)
        {
            destination.push_back(
                ph_histogram_bucket_t{ narrow_histogram_count(bucket.bucket_index),
                                       static_cast<std::uint64_t>(bucket.event_count),
                                       bucket.value });
        }
        storage->tracks[track_index] =
            ph_track_histogram_t{ narrow_histogram_count(source_track.track->id),
                                  narrow_histogram_count(destination.size()),
                                  destination.empty() ? nullptr : destination.data() };
    }

    storage->histogram =
        ph_histogram_t{ static_cast<std::uint64_t>(source.origin_ts),
                        static_cast<std::uint64_t>(source.bucket_size),
                        narrow_histogram_count(source.bucket_count),
                        narrow_histogram_count(storage->tracks.size()),
                        storage->tracks.empty() ? nullptr : storage->tracks.data() };
    output = storage->histogram;

    const std::lock_guard<std::mutex> lock{ m_histogram_mutex };
    m_histograms.push_back(std::move(storage));
}

void
ph_ctx::initialize_track_list()
{
    const auto& all_tracks = m_catalog->tracks;

    m_c_tracks.reserve(all_tracks.size());

    for(const auto& track : all_tracks)
    {
        if(track->event_count == 0)
        {
            continue;
        }

        m_track_by_id.emplace(static_cast<std::uint32_t>(track->id), track);
        m_c_tracks.push_back(ph_track_t{
            .id          = static_cast<std::uint32_t>(track->id),
            .track_name  = track->name.c_str(),
            .nid         = track->node_info
                               ? static_cast<std::uint32_t>(track->node_info->node_id)
                               : 0,
            .pid         = track->process_info
                               ? static_cast<std::uint32_t>(track->process_info->pid)
                               : 0,
            .tid         = track->thread_info  // Some tracks have no thread (e.g. per
                                       // agent+queue category tracks)
                               ? static_cast<std::uint32_t>(track->thread_info->thread_id)
                               : 0,
            .event_count = static_cast<std::uint32_t>(std::min<size_t>(
                track->event_count, std::numeric_limits<std::uint32_t>::max())),
            .agent_id    = static_cast<std::uint32_t>(track->agent_id),
            .category    = to_c_track_category(track->category),
            .queue_id    = static_cast<std::uint32_t>(track->queue_id),
            .stream_id   = static_cast<std::uint32_t>(track->stream_id),
            .start_ts    = static_cast<std::uint64_t>(track->start_ts),
            .end_ts      = static_cast<std::uint64_t>(track->end_ts),
            .value_range = track->value_range ? ph_value_range_t{ track->value_range->min,
                                                                  track->value_range->max,
                                                                  1 }
                                              : ph_value_range_t{},
            .process_row_id = track->process_info
                                  ? static_cast<std::uint64_t>(track->process_info->id)
                                  : 0,
            .thread_row_id  = track->thread_info
                                  ? static_cast<std::uint64_t>(track->thread_info->id)
                                  : 0,
        });
    }
}

bool
ph_ctx::load_all_tracks(const stop_requested_fn& stop_requested,
                        const progress_fn&       progress)
{
    const auto total_events = std::accumulate(
        m_c_tracks.begin(),
        m_c_tracks.end(),
        uint64_t{ 0 },
        [](uint64_t sum, const auto& track) { return sum + track.event_count; });

    uint64_t loaded_events = 0;
    for(auto& c_track : m_c_tracks)
    {
        if(stop_requested()) return false;

        const auto& track = m_track_by_id.at(c_track.id);
        if(track->category == profiler_hub::reader_types::track_kind_t::pmc_agent)
        {
            std::ignore = m_track_reader.samples(track, 0, 0);
        }
        else
        {
            const auto loaded     = m_track_reader.events(track, 0, 0);
            c_track.nesting_depth = m_track_reader.nesting_depth(track);
            c_track.event_count   = loaded.list_size;
        }

        loaded_events += c_track.event_count;
        if(progress && total_events != 0)
        {
            progress(static_cast<double>(loaded_events) /
                         static_cast<double>(total_events),
                     "Loading tracks");
        }
    }
    return true;
}

void
ph_ctx::initialize_node_info()
{
    m_c_node = std::make_unique<ph_node_t>();
    m_nodes  = m_catalog->nodes;

    m_c_node->info = ph_node_info_t{ .id            = 0,
                                     .machine_id    = "",
                                     .system_name   = "",
                                     .hostname      = "",
                                     .release       = "",
                                     .version       = "",
                                     .hardware_name = "",
                                     .domain_name   = "" };

    if(!m_nodes.empty())
    {
        const auto& node = m_nodes[0];
        m_c_node->info   = {
              .id            = static_cast<uint32_t>(node->node_id),
              .machine_id    = node->machine_id.c_str(),
              .system_name   = node->system_name.c_str(),
              .hostname      = node->hostname.c_str(),
              .release       = node->release.c_str(),
              .version       = node->version.c_str(),
              .hardware_name = node->hardware_name.c_str(),
              .domain_name   = node->domain_name.c_str(),
        };
    }

    m_c_node->agents =
        ph_agent_list_t{ .list_size = static_cast<std::uint32_t>(m_c_agents.size()),
                         .agents    = m_c_agents.data() };

    m_c_node->process_list =
        ph_process_list_t{ .list_size = static_cast<std::uint32_t>(m_c_processes.size()),
                           .processes = m_c_processes.data() };

    m_c_node->track_list = get_track_list();

    m_c_node->thread_list =
        ph_thread_list_t{ .list_size = static_cast<std::uint32_t>(m_c_threads.size()),
                          .threads   = m_c_threads.data() };
    m_c_node->stream_list =
        ph_stream_list_t{ .list_size = static_cast<std::uint32_t>(m_c_streams.size()),
                          .streams   = m_c_streams.data() };
    m_c_node->queue_list =
        ph_queue_list_t{ .list_size = static_cast<std::uint32_t>(m_c_queues.size()),
                         .queues    = m_c_queues.data() };
    m_c_node->pmc_info_list =
        ph_pmc_info_list_t{ .list_size = static_cast<std::uint32_t>(m_c_pmc_infos.size()),
                            .pmc_infos = m_c_pmc_infos.data() };

    m_c_node->stream_hardware_links = ph_stream_hardware_link_list_t{
        .list_size = static_cast<std::uint32_t>(m_c_stream_hardware_links.size()),
        .links     = m_c_stream_hardware_links.data()
    };
}

void
ph_ctx::initialize_node_agents()
{
    m_agents = m_catalog->agents;
    m_c_agents.reserve(m_agents.size());

    for(const auto& agent : m_agents)
    {
        m_c_agents.push_back(ph_agent_t{
            .id             = static_cast<std::uint32_t>(agent->type_index),
            .agent_type     = agent->agent_type.c_str(),
            .name           = agent->name.c_str(),
            .model_name     = agent->model_name.c_str(),
            .vendor_name    = agent->vendor_name.c_str(),
            .product_name   = agent->product_name.c_str(),
            .user_name      = agent->user_name.c_str(),
            .extdata        = agent->extdata.c_str(),
            .absolute_index = optional_uint64(agent->absolute_index),
            .logical_index  = optional_uint64(agent->logical_index),
            .uuid           = optional_uint64(agent->uuid),
            .row_id         = static_cast<std::uint64_t>(agent->id),
        });
    }
}

void
ph_ctx::initialize_node_processes()
{
    m_processes = m_catalog->processes;
    m_c_processes.reserve(m_processes.size());

    for(const auto& process : m_processes)
    {
        m_c_processes.push_back(ph_process_t{
            .id          = static_cast<std::uint32_t>(process->pid),
            .command     = process->command.c_str(),
            .environment = process->environment.c_str(),
            .extdata     = process->extdata.c_str(),
            .ppid        = optional_uint64(process->ppid),
            .init        = optional_uint64(process->init),
            .fini        = optional_uint64(process->fini),
            .start       = optional_uint64(process->start),
            .end         = optional_uint64(process->end),
            .row_id      = static_cast<std::uint64_t>(process->id),
        });
    }
}

void
ph_ctx::initialize_node_threads()
{
    m_threads = m_catalog->threads;
    m_c_threads.reserve(m_threads.size());

    for(const auto& thread : m_threads)
    {
        m_c_threads.push_back(ph_thread_t{
            .thread_id         = static_cast<uint64_t>(thread->thread_id),
            .name              = thread->name.c_str(),
            .extdata           = thread->extdata.c_str(),
            .parent_process_id = optional_uint64(thread->parent_process_id),
            .start             = optional_uint64(thread->start),
            .end               = optional_uint64(thread->end),
            .row_id            = static_cast<uint64_t>(thread->id),
        });
    }
}

void
ph_ctx::initialize_node_streams()
{
    m_streams = m_catalog->streams;
    m_c_streams.reserve(m_streams.size());

    for(const auto& stream : m_streams)
    {
        m_c_streams.push_back(ph_stream_t{
            .stream_id = static_cast<uint64_t>(stream->stream_id),
            .name      = stream->name.c_str(),
            .extdata   = stream->extdata.c_str(),
            .nid       = static_cast<uint32_t>(stream->nid),
            .pid       = static_cast<uint32_t>(stream->pid),
        });
    }
}

void
ph_ctx::initialize_node_queues()
{
    m_queues = m_catalog->queues;
    m_c_queues.reserve(m_queues.size());

    for(const auto& queue : m_queues)
    {
        m_c_queues.push_back(ph_queue_t{
            .queue_id = static_cast<uint64_t>(queue->queue_id),
            .name     = queue->name.c_str(),
            .extdata  = queue->extdata.c_str(),
            .nid      = static_cast<uint32_t>(queue->nid),
            .pid      = static_cast<uint32_t>(queue->pid),
        });
    }
}

void
ph_ctx::initialize_node_pmc_infos()
{
    m_pmc_infos = m_catalog->pmc_infos;
    m_c_pmc_infos.reserve(m_pmc_infos.size());

    for(const auto& pmc_info : m_pmc_infos)
    {
        m_c_pmc_infos.push_back(ph_pmc_info_t{
            .name             = pmc_info->name.c_str(),
            .target_arch      = pmc_info->target_arch.c_str(),
            .symbol           = pmc_info->symbol.c_str(),
            .description      = pmc_info->description.c_str(),
            .long_description = pmc_info->long_description.c_str(),
            .component        = pmc_info->component.c_str(),
            .units            = pmc_info->units.c_str(),
            .value_type       = pmc_info->value_type.c_str(),
            .block            = pmc_info->block.c_str(),
            .expression       = pmc_info->expression.c_str(),
            .extdata          = pmc_info->extdata.c_str(),
            .event_code       = optional_uint64(pmc_info->event_code),
            .instance_id      = optional_uint64(pmc_info->instance_id),
            .is_constant      = optional_uint64(pmc_info->is_constant),
            .is_derived       = optional_uint64(pmc_info->is_derived),
            .id               = static_cast<uint64_t>(pmc_info->id),
        });
    }
}

void
ph_ctx::initialize_stream_hardware_links()
{
    const auto& links = m_catalog->stream_hardware_links;
    m_c_stream_hardware_links.reserve(links.size());

    for(const auto& link : links)
    {
        m_c_stream_hardware_links.push_back(ph_stream_hardware_link_t{
            .category          = to_c_track_category(link.category),
            .nid               = static_cast<std::uint32_t>(link.nid),
            .pid               = static_cast<std::uint32_t>(link.pid),
            .stream_id         = static_cast<std::uint64_t>(link.stream_id),
            .agent_id          = optional_uint64(link.agent_id),
            .queue_id          = optional_uint64(link.queue_id),
            .stream_track_id   = optional_track_id(link.stream_track),
            .hardware_track_id = optional_track_id(link.hardware_track),
        });
    }
}
