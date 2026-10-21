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
