#include "ph_ctx.hpp"
#include "ph_future.hpp"
#include "profiler-hub/c/profiler_hub.h"

#include <memory>
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
}  // namespace

ph_result_t
ph_ctx_create(ph_ctx_t* ctx, const char* file_path)
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

    try
    {
        *ctx = new ph_ctx(file_path);
    } catch(...)
    {
        return PH_RESULT_CONTEXT_ALLOCATION_FAILED;
    }

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
        const auto ctx_version = ctx->get_schema_version();
        version->major         = ctx_version.major;
        version->minor         = ctx_version.minor;
        version->patch         = ctx_version.patch;
        return PH_RESULT_SUCCESS;
    });
}

ph_result_t
ph_get_track_list(ph_ctx_t ctx, ph_track_list_t* track_list)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(track_list == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return guard_call([ctx, track_list]() {
        *track_list = ctx->get_track_list();
        return PH_RESULT_SUCCESS;
    });
}

ph_result_t
ph_get_node(ph_ctx_t ctx, ph_node_t* node)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(node == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return guard_call([ctx, node]() {
        *node = ctx->get_node();
        return PH_RESULT_SUCCESS;
    });
}

ph_result_t
ph_get_track_events(ph_ctx_t         ctx,
                    uint32_t         track_id,
                    uint64_t         start_ts,
                    uint64_t         end_ts,
                    ph_event_list_t* events)
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
    return guard_call([ctx, track_id, start_ts, end_ts, events]() {
        if(!ctx->has_track(track_id))
        {
            return PH_RESULT_INVALID_ARGUMENT;
        }

        *events = ctx->get_track_events(track_id, start_ts, end_ts);

        return PH_RESULT_SUCCESS;
    });
}

ph_result_t
ph_get_track_samples(ph_ctx_t          ctx,
                     uint32_t          track_id,
                     uint64_t          start_ts,
                     uint64_t          end_ts,
                     ph_sample_list_t* samples)
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
    return guard_call([ctx, track_id, start_ts, end_ts, samples]() {
        if(!ctx->has_track(track_id))
        {
            return PH_RESULT_INVALID_ARGUMENT;
        }

        *samples = ctx->get_track_samples(track_id, start_ts, end_ts);

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
