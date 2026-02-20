// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <rocprofiler-sdk/rccl/api_args.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

/**
 * @brief Get the size in bytes for an NCCL data type (constexpr for compile-time
 * computation)
 *
 * @param datatype The NCCL data type enum value
 * @return The size in bytes for the given data type, or 0 if unsupported
 */
[[nodiscard]] constexpr size_t
rccl_type_size(ncclDataType_t datatype) noexcept
{
    switch(datatype)
    {
        case ncclInt8:
        case ncclUint8: return 1;
        case ncclFloat16:
        case ncclBfloat16: return 2;
        case ncclInt32:
        case ncclUint32:
        case ncclFloat32: return 4;
        case ncclInt64:
        case ncclUint64:
        case ncclFloat64: return 8;
#if defined(ncclFp8E4M3) && defined(ncclFp8E5M2)
        case ncclFp8E4M3:
        case ncclFp8E5M2: return 1;
#endif
        default: return 0;  // Invalid type - caller should handle
    }
}

/**
 * @brief Get device ID from RCCL communicator
 *
 * Dynamically loads ncclCommCuDevice() to query the device associated with
 * the communicator. Falls back to device 0 if the function is unavailable
 * or the query fails.
 *
 * @param comm The RCCL communicator
 * @return The device ID associated with the communicator
 */
[[nodiscard]] uint32_t
rccl_get_device_id(ncclComm_t comm) noexcept;

/**
 * @brief Information extracted from an RCCL API event
 */
struct rccl_event_info
{
    size_t     size    = 0;        ///< Transfer size in bytes
    bool       is_send = false;    ///< True if send operation, false if recv
    ncclComm_t comm    = nullptr;  ///< RCCL communicator handle
};

/**
 * @brief Thread-safe GPU tracking state for RCCL PMC registration and byte counting
 *
 * This template class manages per-GPU registration and cumulative byte tracking.
 * It uses template-based dependency injection for PMC registration to enable
 * zero-overhead testing (no virtual function calls).
 *
 * @tparam PmcRegistrar Type that implements register_gpu_pmc(uint32_t) - duck typing
 *
 * Usage:
 *   Production: rccl_gpu_tracking_state_t<production_pmc_registrar> state(registrar);
 *   Testing:    rccl_gpu_tracking_state_t<mock_pmc_registrar> state(mock);
 *               rccl_gpu_tracking_state_t<mock_pmc_registrar> state(nullptr);
 */
template <typename PmcRegistrar>
class rccl_gpu_tracking_state_t
{
public:
    /**
     * @brief Construct tracking state with PMC registrar
     * @param registrar Shared pointer to PMC registrar (can be nullptr to disable PMC
     * registration)
     */
    explicit rccl_gpu_tracking_state_t(std::shared_ptr<PmcRegistrar> registrar = nullptr)
    : m_pmc_registrar(std::move(registrar))
    {}

    /**
     * @brief Register a GPU for tracking (idempotent)
     * @param rccl_device_idx The GPU device index
     * @note Calls PMC registrar if one was provided
     */
    inline void register_gpu(uint32_t rccl_device_idx)
    {
        bool newly_registered = false;
        {
            std::unique_lock<std::mutex> _lk{ m_registered_gpus_mutex };
            if(m_registered_gpus.count(rccl_device_idx) == 0)
            {
                m_registered_gpus.insert(rccl_device_idx);
                newly_registered = true;
            }
        }

        if(newly_registered && m_pmc_registrar)
        {
            m_pmc_registrar->register_gpu_pmc(rccl_device_idx);
        }
    }

    /**
     * @brief Add bytes to cumulative counter for a device
     * @param rccl_device_idx The GPU device index
     * @param bytes Number of bytes to add
     * @return The new cumulative byte count for the device
     */
    [[nodiscard]] inline uint64_t add_bytes(uint32_t rccl_device_idx, size_t bytes)
    {
        std::unique_lock<std::mutex> _lk{ m_cumulative_mutex };
        auto& device_bytes = m_cumulative_bytes_per_device[rccl_device_idx];
        device_bytes += bytes;
        return device_bytes;
    }

    /**
     * @brief Check if a GPU is already registered
     * @param rccl_device_idx The GPU device index
     * @return True if registered
     */
    [[nodiscard]] inline bool is_registered(uint32_t rccl_device_idx) const
    {
        std::unique_lock<std::mutex> _lk{ m_registered_gpus_mutex };
        return m_registered_gpus.count(rccl_device_idx) > 0;
    }

    /**
     * @brief Get the cumulative byte count for a device
     * @param rccl_device_idx The GPU device index
     * @return Cumulative bytes (0 if not tracked)
     */
    [[nodiscard]] inline uint64_t get_bytes(uint32_t rccl_device_idx) const
    {
        std::unique_lock<std::mutex> _lk{ m_cumulative_mutex };
        auto it = m_cumulative_bytes_per_device.find(rccl_device_idx);
        return (it != m_cumulative_bytes_per_device.end()) ? it->second : 0;
    }

    /**
     * @brief Reset all tracking state (for testing)
     */
    inline void reset()
    {
        {
            std::unique_lock<std::mutex> _lk{ m_registered_gpus_mutex };
            m_registered_gpus.clear();
        }
        {
            std::unique_lock<std::mutex> _lk{ m_cumulative_mutex };
            m_cumulative_bytes_per_device.clear();
        }
    }

private:
    std::shared_ptr<PmcRegistrar>          m_pmc_registrar;
    mutable std::mutex                     m_registered_gpus_mutex{};
    std::unordered_set<uint32_t>           m_registered_gpus{};
    mutable std::mutex                     m_cumulative_mutex{};
    std::unordered_map<uint32_t, uint64_t> m_cumulative_bytes_per_device{};
};

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
