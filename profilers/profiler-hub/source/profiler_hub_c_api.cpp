#include "database_probe_failure.hpp"
#include "ph_ctx.hpp"
#include "ph_future.hpp"
#include "profiler-hub/c/profiler_hub.h"
#include "profiler-hub/cpp/database_probe.hpp"

#include <algorithm>
#include <cstring>
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
copy_probe_text(const std::string& text, char* buffer, uint64_t capacity)
{
    if(buffer == nullptr || capacity == 0) return;

    const uint64_t text_size = static_cast<uint64_t>(text.size());
    const uint64_t copy_size = std::min(text_size, capacity - 1);
    std::memcpy(buffer, text.data(), static_cast<size_t>(copy_size));
    buffer[copy_size] = '\0';
}

void
copy_probe_result(const profiler_hub::database_probe_result_t& result,
                  ph_database_probe_t*                         probe)
{
    switch(result.status)
    {
        case profiler_hub::probe_status_t::supported:
            probe->status = PH_PROBE_STATUS_SUPPORTED;
            break;
        case profiler_hub::probe_status_t::file_not_found:
            probe->status = PH_PROBE_STATUS_FILE_NOT_FOUND;
            break;
        case profiler_hub::probe_status_t::unreadable:
            probe->status = PH_PROBE_STATUS_UNREADABLE;
            break;
        case profiler_hub::probe_status_t::unknown_format:
            probe->status = PH_PROBE_STATUS_UNKNOWN_FORMAT;
            break;
        case profiler_hub::probe_status_t::metadata_missing:
            probe->status = PH_PROBE_STATUS_METADATA_MISSING;
            break;
        case profiler_hub::probe_status_t::metadata_malformed:
            probe->status = PH_PROBE_STATUS_METADATA_MALFORMED;
            break;
        case profiler_hub::probe_status_t::ambiguous_trace:
            probe->status = PH_PROBE_STATUS_AMBIGUOUS_TRACE;
            break;
        case profiler_hub::probe_status_t::unsupported_version:
            probe->status = PH_PROBE_STATUS_UNSUPPORTED_VERSION;
            break;
    }

    probe->database_kind  = result.database_kind == profiler_hub::database_kind_t::sqlite
                                ? PH_DATABASE_KIND_SQLITE
                                : PH_DATABASE_KIND_UNKNOWN;
    probe->trace_schema   = result.trace_schema == profiler_hub::trace_schema_t::rocpd
                                ? PH_TRACE_SCHEMA_ROCPD
                                : PH_TRACE_SCHEMA_UNKNOWN;
    probe->schema_version = { result.schema_version.major,
                              result.schema_version.minor,
                              result.schema_version.patch };
}
}  // namespace

ph_result_t
ph_probe_database(const char* file_path, ph_database_probe_t* probe)
{
    return ph_probe_database_details(file_path, probe, nullptr, 0, nullptr, 0);
}

ph_result_t
ph_probe_database_details(const char*          file_path,
                          ph_database_probe_t* probe,
                          char*                uuid,
                          uint64_t             uuid_capacity,
                          char*                message,
                          uint64_t             message_capacity)
{
    if(file_path == nullptr || probe == nullptr)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }
    if((uuid == nullptr && uuid_capacity != 0) ||
       (message == nullptr && message_capacity != 0))
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    *probe = {};
    copy_probe_text({}, uuid, uuid_capacity);
    copy_probe_text({}, message, message_capacity);
    return guard_call(
        [file_path, probe, uuid, uuid_capacity, message, message_capacity]() {
            const auto result = profiler_hub::probe_database(file_path);
            copy_probe_result(result, probe);
            copy_probe_text(result.uuid, uuid, uuid_capacity);
            copy_probe_text(result.message, message, message_capacity);
            return PH_RESULT_SUCCESS;
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
    } catch(const profiler_hub::database_probe_failure& failure)
    {
        return profiler_hub::to_ph_result(failure.result().status);
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
                 uint32_t        desired_bins,
                 ph_histogram_t* histogram,
                 ph_future_t     future)
{
    if(histogram != nullptr)
    {
        *histogram = {};
    }

    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }

    if(histogram == nullptr || desired_bins == 0)
    {
        return PH_RESULT_INVALID_ARGUMENT;
    }

    return run_operation(ctx, future, [ctx, desired_bins, histogram]() -> ph_result_t {
        if(const auto ready = ctx->wait_until_ready(); ready != PH_RESULT_SUCCESS)
        {
            return ready;
        }
        ctx->build_histogram(desired_bins, *histogram);
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
