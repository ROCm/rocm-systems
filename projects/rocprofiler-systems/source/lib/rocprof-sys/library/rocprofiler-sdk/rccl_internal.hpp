// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Defines RCCL_API_TRACE_VERSION_PATCH. Must precede api_args.h, where this
// macro gates the newer RCCL arg union members (e.g. ncclAlltoAll). api_args.h
// from the SDK headers does not define it itself, so without this include the
// macro defaults to 0 and those members silently disappear.
//
// This header (not rccl.hpp) is the one the unit tests include.
// Keeping the include here makes the header self-contained so the test TU sees
// the same api_args.h layout as production and avoids an ODR mismatch.
#include <rocprofiler-sdk/rccl/details/api_trace.h>

#include <rocprofiler-sdk/rccl/api_args.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace rocprofsys::rocprofiler_sdk
{

struct rccl_gpu_tracking_state_t
{
    /**
     * @brief Register a GPU for tracking (idempotent)
     * @param rccl_device_idx The GPU device index
     * @note Calls PMC registrar if one was provided
     */
    static bool register_gpu(std::uint32_t rccl_device_idx)
    {
        {
            std::unique_lock<std::mutex> const _lk{ m_registered_gpus_mutex };
            if(!m_registered_gpus.contains(rccl_device_idx))
            {
                return false;
            }
            m_registered_gpus.insert(rccl_device_idx);
        }

        return true;
    }

    /**
     * @brief Add bytes to cumulative counter for a device
     * @param rccl_device_idx The GPU device index
     * @param bytes Number of bytes to add
     * @return The new cumulative byte count for the device
     */
    [[nodiscard]] static std::uint64_t add_bytes(std::uint32_t rccl_device_idx,
                                                 size_t        bytes)
    {
        std::unique_lock<std::mutex> const _lk{ m_cumulative_mutex };
        auto& device_bytes = m_cumulative_bytes_per_device[rccl_device_idx];
        device_bytes += bytes;
        return device_bytes;
    }

private:
    static std::mutex                                       m_registered_gpus_mutex;
    static std::unordered_set<std::uint32_t>                m_registered_gpus;
    static std::mutex                                       m_cumulative_mutex;
    static std::unordered_map<std::uint32_t, std::uint64_t> m_cumulative_bytes_per_device;
};

}  // namespace rocprofsys::rocprofiler_sdk
