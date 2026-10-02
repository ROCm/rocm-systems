// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include "logger/debug.hpp"

#include <cstdint>

namespace rocprofsys::domains::callback::rccl
{

template <policies::domain_service::backend SdkBackend>
struct device_resolver
{
    template <policies::domain_service::externals Externals>
    static void configure_comm_cu_device_function()
    {
        auto func                 = Externals::dlsym("ncclCommCuDevice");
        s_nccl_comm_cu_device_ptr = reinterpret_cast<nccl_comm_cu_device_fn>(func);
        if(s_nccl_comm_cu_device_ptr == nullptr)
        {
            const char* error = Externals::dlerror();
            LOG_DEBUG(
                "ncclCommCuDevice not found via dlsym ({}), using default device_id",
                error ? error : "unknown error");
        }
    }

    [[nodiscard]] static std::uint32_t resolve_device_id(
        SdkBackend::nccl_comm_t comm) noexcept
    {
        constexpr std::uint32_t k_default_device_id = 0;

        if(comm == nullptr)
        {
            return k_default_device_id;
        }

        if(s_nccl_comm_cu_device_ptr == nullptr)
        {
            return k_default_device_id;
        }

        int                                device_id = k_default_device_id;
        typename SdkBackend::nccl_result_t result =
            s_nccl_comm_cu_device_ptr(comm, &device_id);
        if(result != SdkBackend::NCCL_SUCCESS)
        {
            LOG_DEBUG("ncclCommCuDevice failed with error {}, using default device_id",
                      static_cast<int>(result));
            return k_default_device_id;
        }
        return static_cast<std::uint32_t>(device_id);
    }

private:
    using nccl_comm_cu_device_fn =
        SdkBackend::nccl_result_t (*)(typename SdkBackend::nccl_comm_t, int*);

    static inline nccl_comm_cu_device_fn s_nccl_comm_cu_device_ptr{ nullptr };
};
}  // namespace rocprofsys::domains::callback::rccl
