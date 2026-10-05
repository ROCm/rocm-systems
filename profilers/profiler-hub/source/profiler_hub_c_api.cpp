#include "profiler-hub/c/profiler_hub.h"
#include "profiler_hub_ctx.hpp"
#include "profiler_hub_future.hpp"

#include <memory>
#include <optional>
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
        const auto ctx_version = ctx->get_storage_version();
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

    return guard_call([ctx, track_id, start_ts, end_ts, events]() {
        if(events == nullptr || !ctx->has_track(track_id))
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

    return guard_call([ctx, track_id, start_ts, end_ts, samples]() {
        if(samples == nullptr || !ctx->has_track(track_id))
        {
            return PH_RESULT_INVALID_ARGUMENT;
        }

        *samples = ctx->get_track_samples(track_id, start_ts, end_ts);

        return PH_RESULT_SUCCESS;
    });
}

namespace
{
void
abandon_task(profiler_hub::common::thread_pool::task_handle& handle)
{
    std::ignore = handle.cancel();
    handle.wait();
}

ph_result_t
submit_and_wrap_future(ph_ctx_t                                   ctx,
                       ph_future_t*                               future,
                       profiler_hub::common::thread_pool::task_fn task)
{
    std::optional<profiler_hub::common::thread_pool::task_handle> handle;
    try
    {
        handle.emplace(ctx->get_thread_pool().submit(std::move(task)));
    } catch(...)
    {
        return PH_RESULT_FUTURE_ALLOCATION_FAILED;
    }

    std::unique_ptr<ph_future> wrapper;
    bool                       registered = false;
    try
    {
        wrapper    = std::make_unique<ph_future>(*handle);
        registered = ctx->register_future(wrapper.get());
    } catch(...)
    {
        abandon_task(*handle);
        return PH_RESULT_FUTURE_ALLOCATION_FAILED;
    }

    if(!registered)
    {
        abandon_task(*handle);
        return PH_RESULT_INVALID_CONTEXT;
    }

    *future = wrapper.release();
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

    if(future == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    *future = nullptr;
    if(task_fn == nullptr)
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

    return guard_call([future]() {
        future->m_handle.wait();
        return PH_RESULT_SUCCESS;
    });
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

    return guard_call([future]() {
        std::ignore = future->m_handle.cancel();
        return PH_RESULT_SUCCESS;
    });
}

ph_result_t
ph_future_free(ph_ctx_t ctx, ph_future_t future)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(future == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return guard_call([ctx, future]() {
        if(!ctx->unregister_future(future))
        {
            return PH_RESULT_INVALID_ARGUMENT;
        }
        delete future;
        return PH_RESULT_SUCCESS;
    });
}
