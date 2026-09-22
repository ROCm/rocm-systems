#include "profiler-hub/c/profiler_hub.h"
#include "profiler_hub_ctx.hpp"
#include "profiler_hub_future.hpp"

#include <tuple>
#include <utility>

ph_result_t
ph_ctx_create(ph_ctx_t* ctx, const char* file_path)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
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
    delete ctx;
    return PH_RESULT_SUCCESS;
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

    version->major = 0;
    version->minor = 1;
    version->patch = 0;
    return PH_RESULT_SUCCESS;
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

    const auto ctx_version = ctx->get_storage_version();
    version->major         = ctx_version.major;
    version->minor         = ctx_version.minor;
    version->patch         = ctx_version.patch;
    return PH_RESULT_SUCCESS;
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

    *track_list = ctx->get_track_list();

    return PH_RESULT_SUCCESS;
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

    *node = ctx->get_node();

    return PH_RESULT_SUCCESS;
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

    if(events == nullptr || !ctx->has_track(track_id))
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    *events = ctx->get_track_events(track_id, start_ts, end_ts);

    return PH_RESULT_SUCCESS;
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

    if(samples == nullptr || !ctx->has_track(track_id))
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    *samples = ctx->get_track_samples(track_id, start_ts, end_ts);

    return PH_RESULT_SUCCESS;
}

namespace
{
ph_result_t
submit_and_wrap_future(ph_ctx_t                                   ctx,
                       ph_future_t*                               future,
                       profiler_hub::common::thread_pool::task_fn task)
{
    auto handle = ctx->get_thread_pool().submit(std::move(task));

    try
    {
        *future = new ph_future(std::move(handle));
    } catch(...)
    {
        return PH_RESULT_FUTURE_ALLOCATION_FAILED;
    }

    ctx->register_future(*future);
    return PH_RESULT_SUCCESS;
}
}  // namespace

ph_result_t
ph_future_get(ph_ctx_t ctx, ph_future_t* future, ph_task_fn task_fn, void* user_data)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(future == nullptr || task_fn == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return submit_and_wrap_future(
        ctx, future, [task_fn, user_data](const std::stop_token&) {
            task_fn(user_data);
        });
}

ph_result_t
ph_future_wait(ph_ctx_t ctx, ph_future_t future)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(future == nullptr || !ctx->owns_future(future))
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    future->m_handle.wait();
    return PH_RESULT_SUCCESS;
}

ph_result_t
ph_future_cancel(ph_ctx_t ctx, ph_future_t future)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(future == nullptr || !ctx->owns_future(future))
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    std::ignore = future->m_handle.cancel();
    return PH_RESULT_SUCCESS;
}

ph_result_t
ph_future_free(ph_ctx_t ctx, ph_future_t future)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(future == nullptr || !ctx->owns_future(future))
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    ctx->unregister_future(future);
    delete future;
    return PH_RESULT_SUCCESS;
}
