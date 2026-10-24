#include "event_query.hpp"
#include "file_probe_failure.hpp"
#include "ph_ctx.hpp"
#include "ph_future.hpp"
#include "profiler-hub/c/profiler_hub.h"
#include "profiler-hub/cpp/reader.hpp"

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

namespace
{
template <typename Fn>
ph_result_t
guard_call(Fn&& fn)
{
    try
    {
        return std::forward<Fn>(fn)();
    } catch(...)
    {
        return PH_RESULT_INTERNAL_ERROR;
    }
}

ph_result_t
run_operation(ph_ctx_t ctx, ph_future_t future, std::function<ph_result_t()> operation)
{
    if(future == nullptr)
    {
        return guard_call(operation);
    }

    if(!future->try_attach())
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    try
    {
        ctx->run_async(future->shared_from_this(), std::move(operation));
    } catch(...)
    {
        future->finish(PH_FUTURE_ERROR, PH_RESULT_FUTURE_ALLOCATION_FAILED);
        return PH_RESULT_FUTURE_ALLOCATION_FAILED;
    }
    return PH_RESULT_SUCCESS;
}

void
copy_probe_result(const profiler_hub::reader_types::file_probe_result_t& result,
                  ph_file_probe_t*                                       probe)
{
    probe->trace_schema =
        result.trace_schema == profiler_hub::reader_types::trace_schema_t::rocpd
            ? PH_TRACE_SCHEMA_ROCPD
            : PH_TRACE_SCHEMA_UNKNOWN;
    probe->schema_version = { result.schema_version.major,
                              result.schema_version.minor,
                              result.schema_version.patch };
}

bool
is_duration_event_type(ph_event_type_t type)
{
    switch(type)
    {
        case PH_EVENT_TYPE_REGION:
        case PH_EVENT_TYPE_KERNEL_DISPATCH:
        case PH_EVENT_TYPE_MEMORY_COPY:
        case PH_EVENT_TYPE_MEMORY_ALLOCATE: return true;
    }
    return false;
}

template <typename Details, typename Fetch>
ph_result_t
get_event_details(ph_ctx_t        ctx,
                  ph_event_t*     event,
                  Details*        details,
                  ph_future_t     future,
                  ph_event_type_t expected_type,
                  Fetch           fetch)
{
    if(details != nullptr)
    {
        *details = {};
    }
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }
    if(details == nullptr || event == nullptr || event->type != expected_type)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    const uint64_t event_id = event->id;
    return run_operation(
        ctx, future, [ctx, event, event_id, expected_type, details, fetch]() -> ph_result_t {
            if(const auto ready = ctx->wait_until_ready(); ready != PH_RESULT_SUCCESS)
            {
                return ready;
            }
            *details = fetch(ctx, event_id);
            if(const auto track_id = ctx->track_id_of(event_id, expected_type);
               track_id.has_value())
            {
                event->track_id = track_id.value();
            }
            return PH_RESULT_SUCCESS;
        });
}
}  // namespace

ph_result_t
ph_probe_file(const char* file_path, ph_file_probe_t* probe)
{
    if(file_path == nullptr || probe == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    *probe = {};
    return guard_call([file_path, probe]() {
        const profiler_hub::reader_types::file_probe_result_t result =
            profiler_hub::probe_file(file_path);
        copy_probe_result(result, probe);
        return result.status;
    });
}

ph_result_t
ph_ctx_create(ph_ctx_t* ctx, const char* file_path, ph_future_t future)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    *ctx = nullptr;
    if(file_path == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    std::unique_ptr<ph_ctx> created;
    try
    {
        created = std::make_unique<ph_ctx>(file_path);
        if(future == nullptr)
        {
            created->load();
        }
        else if(future->try_attach())
        {
            created->load_async(future->shared_from_this());
        }
        else
        {
            return PH_RESULT_INVALID_ARGUMENT;
        }
    } catch(const profiler_hub::file_probe_failure& failure)
    {
        return failure.result().status;
    } catch(...)
    {
        return PH_RESULT_CONTEXT_ALLOCATION_FAILED;
    }

    *ctx = created.release();
    return PH_RESULT_SUCCESS;
}

ph_result_t
ph_ctx_free(ph_ctx_t ctx)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    return guard_call([ctx]() {
        delete ctx;
        return PH_RESULT_SUCCESS;
    });
}

ph_result_t
ph_get_library_version(ph_ctx_t ctx, ph_library_version_t* version)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(version == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return guard_call([version]() {
        version->major = PROFILER_HUB_VERSION_MAJOR;
        version->minor = PROFILER_HUB_VERSION_MINOR;
        version->patch = PROFILER_HUB_VERSION_PATCH;
        return PH_RESULT_SUCCESS;
    });
}

ph_result_t
ph_get_schema_version(ph_ctx_t ctx, ph_schema_version_t* version)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(version == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return guard_call([ctx, version]() {
        if(const auto ready = ctx->wait_until_ready(); ready != PH_RESULT_SUCCESS)
        {
            return ready;
        }
        const auto ctx_version = ctx->get_schema_version();
        version->major         = ctx_version.major;
        version->minor         = ctx_version.minor;
        version->patch         = ctx_version.patch;
        return PH_RESULT_SUCCESS;
    });
}

ph_result_t
ph_get_track_list(ph_ctx_t ctx, ph_track_list_t* track_list, ph_future_t future)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(track_list == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return run_operation(ctx, future, [ctx, track_list]() -> ph_result_t {
        if(const auto ready = ctx->wait_until_ready(); ready != PH_RESULT_SUCCESS)
        {
            return ready;
        }
        *track_list = ctx->get_track_list();
        return PH_RESULT_SUCCESS;
    });
}

ph_result_t
ph_get_node(ph_ctx_t ctx, ph_node_t* node, ph_future_t future)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(node == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return run_operation(ctx, future, [ctx, node]() -> ph_result_t {
        if(const auto ready = ctx->wait_until_ready(); ready != PH_RESULT_SUCCESS)
        {
            return ready;
        }
        *node = ctx->get_node();
        return PH_RESULT_SUCCESS;
    });
}

ph_result_t
ph_get_track_events(ph_ctx_t         ctx,
                    uint32_t         track_id,
                    uint64_t         start_ts,
                    uint64_t         end_ts,
                    ph_event_list_t* events,
                    ph_future_t      future)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(events == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    *events = ph_event_list_t{ .list_size = 0, .events = nullptr };
    return run_operation(
        ctx, future, [ctx, track_id, start_ts, end_ts, events]() -> ph_result_t {
            if(const auto ready = ctx->wait_until_ready(); ready != PH_RESULT_SUCCESS)
            {
                return ready;
            }
            if(!ctx->has_track(track_id))
            {
                return PH_RESULT_INVALID_ARGUMENT;
            }

            *events = ctx->get_track_events(track_id, start_ts, end_ts);

            return PH_RESULT_SUCCESS;
        });
}

ph_result_t
ph_get_correlated_events(ph_ctx_t          ctx,
                         const ph_event_t* event,
                         ph_event_list_t*  events,
                         ph_future_t       future)
{
    if(events != nullptr)
    {
        *events = ph_event_list_t{ .list_size = 0, .events = nullptr };
    }

    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(events == nullptr || event == nullptr || !is_duration_event_type(event->type))
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    const uint64_t        event_id   = event->id;
    const ph_event_type_t event_type = event->type;
    return run_operation(
        ctx, future, [ctx, event_id, event_type, events]() -> ph_result_t {
            if(const auto ready = ctx->wait_until_ready(); ready != PH_RESULT_SUCCESS)
            {
                return ready;
            }

            const auto correlated = ctx->get_correlated_events(event_id, event_type);
            if(!correlated.has_value())
            {
                return PH_RESULT_INTERNAL_ERROR;
            }
            *events = correlated.value();
            return PH_RESULT_SUCCESS;
        });
}

ph_result_t
ph_get_call_stack(ph_ctx_t          ctx,
                  const ph_event_t* event,
                  ph_call_stack_t*  call_stack,
                  ph_future_t       future)
{
    if(call_stack != nullptr)
    {
        *call_stack = ph_call_stack_t{ .list_size = 0, .frames = nullptr };
    }

    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(call_stack == nullptr || event == nullptr || !is_duration_event_type(event->type))
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    const uint64_t        event_id   = event->id;
    const ph_event_type_t event_type = event->type;
    return run_operation(
        ctx, future, [ctx, event_id, event_type, call_stack]() -> ph_result_t {
            if(const auto ready = ctx->wait_until_ready(); ready != PH_RESULT_SUCCESS)
            {
                return ready;
            }

            *call_stack = ctx->get_call_stack(event_id, event_type);
            return PH_RESULT_SUCCESS;
        });
}

ph_result_t
ph_get_region_details(ph_ctx_t             ctx,
                      ph_event_t*          event,
                      ph_region_details_t* details,
                      ph_future_t          future)
{
    return get_event_details(
        ctx,
        event,
        details,
        future,
        PH_EVENT_TYPE_REGION,
        [](ph_ctx_t context, uint64_t id) { return context->get_region_details(id); });
}

ph_result_t
ph_get_kernel_dispatch_details(ph_ctx_t                      ctx,
                               ph_event_t*                   event,
                               ph_kernel_dispatch_details_t* details,
                               ph_future_t                   future)
{
    return get_event_details(ctx,
                             event,
                             details,
                             future,
                             PH_EVENT_TYPE_KERNEL_DISPATCH,
                             [](ph_ctx_t context, uint64_t id) {
                                 return context->get_kernel_dispatch_details(id);
                             });
}

ph_result_t
ph_get_memory_copy_details(ph_ctx_t                  ctx,
                           ph_event_t*               event,
                           ph_memory_copy_details_t* details,
                           ph_future_t               future)
{
    return get_event_details(ctx,
                             event,
                             details,
                             future,
                             PH_EVENT_TYPE_MEMORY_COPY,
                             [](ph_ctx_t context, uint64_t id) {
                                 return context->get_memory_copy_details(id);
                             });
}

ph_result_t
ph_get_memory_alloc_details(ph_ctx_t                   ctx,
                            ph_event_t*                event,
                            ph_memory_alloc_details_t* details,
                            ph_future_t                future)
{
    return get_event_details(ctx,
                             event,
                             details,
                             future,
                             PH_EVENT_TYPE_MEMORY_ALLOCATE,
                             [](ph_ctx_t context, uint64_t id) {
                                 return context->get_memory_alloc_details(id);
                             });
}

ph_result_t
ph_get_track_samples(ph_ctx_t          ctx,
                     uint32_t          track_id,
                     uint64_t          start_ts,
                     uint64_t          end_ts,
                     ph_sample_list_t* samples,
                     ph_future_t       future)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(samples == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    *samples = ph_sample_list_t{ .list_size = 0, .samples = nullptr };
    return run_operation(
        ctx, future, [ctx, track_id, start_ts, end_ts, samples]() -> ph_result_t {
            if(const auto ready = ctx->wait_until_ready(); ready != PH_RESULT_SUCCESS)
            {
                return ready;
            }
            if(!ctx->has_track(track_id))
            {
                return PH_RESULT_INVALID_ARGUMENT;
            }

            *samples = ctx->get_track_samples(track_id, start_ts, end_ts);

            return PH_RESULT_SUCCESS;
        });
}

ph_result_t
ph_get_histogram(ph_ctx_t        ctx,
                 uint32_t        max_buckets,
                 ph_histogram_t* histogram,
                 ph_future_t     future)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }
    if(histogram == nullptr || max_buckets == 0)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }
    *histogram = {};

    return run_operation(ctx, future, [ctx, max_buckets, histogram]() -> ph_result_t {
        if(const auto ready = ctx->wait_until_ready(); ready != PH_RESULT_SUCCESS)
        {
            return ready;
        }
        ctx->build_histogram(max_buckets, *histogram);
        return PH_RESULT_SUCCESS;
    });
}

ph_result_t
ph_future_create(ph_progress_fn on_progress,
                 ph_finished_fn on_finished,
                 ph_future_t*   future)
{
    if(future == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    *future = nullptr;
    try
    {
        *future = ph_future::create(on_progress, on_finished).get();
    } catch(...)
    {
        return PH_RESULT_FUTURE_ALLOCATION_FAILED;
    }

    return PH_RESULT_SUCCESS;
}

ph_result_t
ph_future_wait(ph_future_t future)
{
    if(future == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return guard_call([future]() { return future->wait(); });
}

ph_result_t
ph_future_cancel(ph_future_t future)
{
    if(future == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return guard_call([future]() { return future->cancel(); });
}

ph_result_t
ph_future_result(ph_future_t future, ph_result_t* result)
{
    if(future == nullptr || result == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return guard_call([future, result]() { return future->result(*result); });
}

ph_result_t
ph_future_free(ph_future_t future)
{
    if(future == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return guard_call([future]() {
        return future->release_user_reference() ? PH_RESULT_SUCCESS
                                                : PH_RESULT_INVALID_ARGUMENT;
    });
}

extern "C" struct ph_event_query
{
    ph_ctx*                            ctx;
    profiler_hub::resolved_event_query query;
};

namespace
{
bool
known_event_type(uint32_t type)
{
    return type == PH_EVENT_TYPE_REGION || type == PH_EVENT_TYPE_REGION_SAMPLE ||
           type == PH_EVENT_TYPE_KERNEL_DISPATCH || type == PH_EVENT_TYPE_MEMORY_COPY ||
           type == PH_EVENT_TYPE_MEMORY_ALLOCATE;
}

std::optional<uint64_t>
copied_id(ph_optional_uint64_t value)
{
    if(value.is_valid == 0)
    {
        return std::nullopt;
    }
    return value.value;
}

ph_result_t
copy_event_filter(const ph_event_filter_t& filter, profiler_hub::resolved_event_query& query)
{
    if(filter.size < sizeof(ph_event_filter_t))
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }
    if((filter.track_count > 0 && filter.tracks == nullptr) ||
       (filter.event_type_count > 0 && filter.event_types == nullptr) ||
       (filter.string_filter.search_strings_count > 0 &&
        filter.string_filter.search_strings == nullptr))
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    if(filter.time.is_valid != 0)
    {
        query.time = profiler_hub::time_window{ filter.time.start, filter.time.end };
    }
    query.node_id          = copied_id(filter.node_id);
    query.agent_id         = copied_id(filter.agent_id);
    query.process_id       = copied_id(filter.process_id);
    query.thread_id        = copied_id(filter.thread_id);
    query.include_category = filter.string_filter.include_category != 0;

    if(filter.track_count > 0)
    {
        query.tracks.assign(filter.tracks, filter.tracks + filter.track_count);
    }
    query.event_types.reserve(filter.event_type_count);
    for(uint32_t index = 0; index < filter.event_type_count; ++index)
    {
        if(!known_event_type(filter.event_types[index]))
        {
            return PH_RESULT_INVALID_ARGUMENT;
        }
        query.event_types.push_back(filter.event_types[index]);
    }

    query.search_strings.reserve(filter.string_filter.search_strings_count);
    for(uint32_t index = 0; index < filter.string_filter.search_strings_count; ++index)
    {
        const char* text = filter.string_filter.search_strings[index];
        if(text == nullptr || text[0] == '\0')
        {
            return PH_RESULT_INVALID_ARGUMENT;
        }
        query.search_strings.emplace_back(text);
    }
    return PH_RESULT_SUCCESS;
}
}  // namespace

ph_result_t
ph_event_query_create(ph_ctx_t ctx, const ph_event_filter_t* filter, ph_event_query_t* query)
{
    if(query != nullptr)
    {
        *query = nullptr;
    }
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }
    if(filter == nullptr || query == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return guard_call([ctx, filter, query]() {
        if(const auto ready = ctx->wait_until_ready(); ready != PH_RESULT_SUCCESS)
        {
            return ready;
        }
        if(ctx->catalog() == nullptr)
        {
            return PH_RESULT_INVALID_CONTEXT;
        }

        auto resolved = std::make_unique<ph_event_query>();
        resolved->ctx = ctx;
        if(const auto copied = copy_event_filter(*filter, resolved->query);
           copied != PH_RESULT_SUCCESS)
        {
            return copied;
        }
        profiler_hub::resolve_event_query_strings(
            *ctx->catalog(),
            resolved->query,
            filter->string_filter.include_substring != 0,
            filter->string_filter.partial_matching != 0);
        *query = resolved.release();
        return PH_RESULT_SUCCESS;
    });
}

ph_result_t
ph_event_query_free(ph_event_query_t query)
{
    if(query == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }
    delete query;
    return PH_RESULT_SUCCESS;
}

ph_result_t
ph_event_query_read(ph_event_query_t  query,
                    uint32_t          event_type,
                    uint64_t          after_id,
                    ph_table_event_t* rows,
                    uint32_t          capacity,
                    uint32_t*         count,
                    ph_future_t       future)
{
    if(count != nullptr)
    {
        *count = 0;
    }
    if(query == nullptr || query->ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }
    const bool known = event_type == PH_EVENT_TYPE_REGION ||
                       event_type == PH_EVENT_TYPE_REGION_SAMPLE ||
                       event_type == PH_EVENT_TYPE_KERNEL_DISPATCH ||
                       event_type == PH_EVENT_TYPE_MEMORY_COPY ||
                       event_type == PH_EVENT_TYPE_MEMORY_ALLOCATE;
    if(rows == nullptr || count == nullptr || capacity == 0 || !known)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return guard_call([query, event_type, after_id, rows, capacity, count, future]() {
        if(future != nullptr && future->stop_token().stop_requested())
        {
            return PH_RESULT_CANCELLED;
        }
        *count = query->ctx->read_table_events(
            query->query, event_type, after_id, rows, capacity);
        if(future != nullptr && future->stop_token().stop_requested())
        {
            return PH_RESULT_CANCELLED;
        }
        return PH_RESULT_SUCCESS;
    });
}

ph_result_t
ph_lookup_text(ph_ctx_t ctx, uint64_t id, uint32_t kind, const char** text)
{
    if(text != nullptr)
    {
        *text = nullptr;
    }
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }
    if(text == nullptr ||
       (kind != PH_TEXT_KIND_STRING && kind != PH_TEXT_KIND_KERNEL_SYMBOL))
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return guard_call([ctx, id, kind, text]() {
        if(const auto ready = ctx->wait_until_ready(); ready != PH_RESULT_SUCCESS)
        {
            return ready;
        }
        if(ctx->catalog() == nullptr)
        {
            return PH_RESULT_INVALID_CONTEXT;
        }
        const auto& catalog = *ctx->catalog();
        if(kind == PH_TEXT_KIND_STRING)
        {
            const auto it = catalog.strings_by_id.find(static_cast<size_t>(id));
            if(it == catalog.strings_by_id.end())
            {
                return PH_RESULT_INVALID_ARGUMENT;
            }
            *text = it->second.c_str();
            return PH_RESULT_SUCCESS;
        }

        const auto it = catalog.kernel_symbols_by_id.find(static_cast<size_t>(id));
        if(it == catalog.kernel_symbols_by_id.end() || !it->second)
        {
            return PH_RESULT_INVALID_ARGUMENT;
        }
        const auto& symbol = *it->second;
        *text = symbol.display_name.empty() ? symbol.name.c_str() : symbol.display_name.c_str();
        return PH_RESULT_SUCCESS;
    });
}
