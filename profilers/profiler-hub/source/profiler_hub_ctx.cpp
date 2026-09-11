#include "profiler_hub_ctx.hpp"
#include "profiler-hub/c_interface/profiler_hub.h"
#include "profiler-hub/c_interface/profiler_hub_types.h"

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

    return PH_RESULT_SUCCES;
}

ph_result_t
ph_ctx_free(ph_ctx_t ctx)
{
    if(ctx == nullptr)
    {
        return PH_RESULT_INVALID_CONTEXT;
    }
    delete ctx;
    return PH_RESULT_SUCCES;
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
    return PH_RESULT_SUCCES;
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
    return PH_RESULT_SUCCES;
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

    return PH_RESULT_SUCCES;
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

    const auto x = ctx->get_node();

    return PH_RESULT_SUCCES;
}
