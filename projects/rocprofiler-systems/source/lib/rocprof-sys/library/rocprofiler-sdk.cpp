// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk.hpp"
#include <cstdint>

// ─── C registration entry points (must remain non-template) ─────────────────
//
// These are C symbols loaded by name by the rocprofiler-sdk loader.
// They delegate immediately to the fully-instantiated library_sdk<backend>.

namespace
{
using lib_sdk = rocprofsys::rocprofiler_sdk::library_sdk_t;
}

extern "C"
{
    rocprofiler_tool_configure_result_t* rocprofiler_configure(
        std::uint32_t version, const char* runtime_version,
        [[maybe_unused]] std::uint32_t priority, rocprofiler_client_id_t* id)
    {
        // activate only once
        {
            static std::atomic<bool> _first{ true };
            if(!_first.exchange(false)) return nullptr;
        }

        if(!rocprofsys::get_env(rocprofsys::env_vars::INIT_TOOLING, true)) return nullptr;
        if(!tim::settings::enabled()) return nullptr;

        if(!lib_sdk::sdk_tool_configure(version, runtime_version, id)) return nullptr;

        static auto cfg = rocprofiler_tool_configure_result_t{
            sizeof(rocprofiler_tool_configure_result_t), &lib_sdk::tool_init,
            &lib_sdk::tool_fini, lib_sdk::tool_data
        };
        return &cfg;
    }

#if ROCPROFILER_VERSION >= 10200
    rocprofiler_tool_configure_attach_result_t* rocprofiler_configure_attach(
        std::uint32_t version, const char* runtime_version,
        [[maybe_unused]] std::uint32_t priority, rocprofiler_client_id_t* id)
    {
        if(!lib_sdk::sdk_tool_configure(version, runtime_version, id)) return nullptr;

        static auto cfg = rocprofiler_tool_configure_attach_result_t{
            sizeof(rocprofiler_tool_configure_attach_result_t),
            &lib_sdk::tool_attach_init, &lib_sdk::tool_attach_fini, lib_sdk::tool_data
        };
        return &cfg;
    }
#endif
}
