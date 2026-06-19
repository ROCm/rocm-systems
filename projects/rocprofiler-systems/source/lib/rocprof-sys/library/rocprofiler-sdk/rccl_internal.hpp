// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// NCCL and rocprofiler-sdk/rccl headers come transitively through the backend shim.
#include "library/rocprofiler-sdk/fwd.hpp"

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
 * @brief Get the size in bytes for an NCCL data type.
 *
 * @tparam Wrapper Wrapper policy providing nccl_data_type_t.
 * @param datatype The NCCL data type enum value.
 * @return The size in bytes for the given data type, or 0 if unsupported.
 */
template <typename Wrapper>
[[nodiscard]] constexpr size_t
rccl_type_size(typename Wrapper::nccl_data_type_t datatype) noexcept
{
    switch(datatype)
    {
        case Wrapper::NCCL_INT8:
        case Wrapper::NCCL_UINT8: return 1;
        case Wrapper::NCCL_FLOAT16:
        case Wrapper::NCCL_BFLOAT16: return 2;
        case Wrapper::NCCL_INT32:
        case Wrapper::NCCL_UINT32:
        case Wrapper::NCCL_FLOAT32: return 4;
        case Wrapper::NCCL_INT64:
        case Wrapper::NCCL_UINT64:
        case Wrapper::NCCL_FLOAT64: return 8;
#if defined(ncclFp8E4M3) && defined(ncclFp8E5M2)
        case Wrapper::NCCL_FP8_E4M3:
        case Wrapper::NCCL_FP8_E5M2: return 1;
#endif
        default: return 0;
    }
}

/**
 * @brief Get device ID from RCCL communicator.
 *
 * @tparam Wrapper Wrapper policy providing nccl_comm_t.
 */
template <typename Wrapper>
[[nodiscard]] std::uint32_t
rccl_get_device_id(typename Wrapper::nccl_comm_t comm) noexcept;

/**
 * @brief Information extracted from an RCCL API event.
 *
 * @tparam Wrapper Wrapper policy providing nccl_comm_t.
 */
template <typename Wrapper>
struct rccl_event_info
{
    size_t                        size    = 0;
    bool                          is_send = false;
    typename Wrapper::nccl_comm_t comm    = nullptr;
};

/**
 * @brief Thread-safe GPU tracking state for RCCL PMC registration and byte counting.
 *
 * @tparam PmcRegistrar Type implementing register_gpu_pmc(std::uint32_t).
 */
template <typename T>
concept pmc_registrar =
    requires(T& _r, std::uint32_t _idx) { _r.register_gpu_pmc(_idx); };

template <pmc_registrar PmcRegistrar>
class rccl_gpu_tracking_state_t
{
public:
    explicit rccl_gpu_tracking_state_t(std::shared_ptr<PmcRegistrar> registrar = nullptr)
    : m_pmc_registrar(std::move(registrar))
    {}

    inline void register_gpu(std::uint32_t rccl_device_idx)
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
            m_pmc_registrar->register_gpu_pmc(rccl_device_idx);
    }

    [[nodiscard]] inline std::uint64_t add_bytes(std::uint32_t rccl_device_idx,
                                                 size_t        bytes)
    {
        std::unique_lock<std::mutex> _lk{ m_cumulative_mutex };
        auto& device_bytes = m_cumulative_bytes_per_device[rccl_device_idx];
        device_bytes += bytes;
        return device_bytes;
    }

    [[nodiscard]] inline bool is_registered(std::uint32_t rccl_device_idx) const
    {
        std::unique_lock<std::mutex> _lk{ m_registered_gpus_mutex };
        return m_registered_gpus.count(rccl_device_idx) > 0;
    }

    [[nodiscard]] inline std::uint64_t get_bytes(std::uint32_t rccl_device_idx) const
    {
        std::unique_lock<std::mutex> _lk{ m_cumulative_mutex };
        auto itr = m_cumulative_bytes_per_device.find(rccl_device_idx);
        return (itr != m_cumulative_bytes_per_device.end()) ? itr->second : 0;
    }

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
    std::shared_ptr<PmcRegistrar>                    m_pmc_registrar;
    mutable std::mutex                               m_registered_gpus_mutex{};
    std::unordered_set<std::uint32_t>                m_registered_gpus{};
    mutable std::mutex                               m_cumulative_mutex{};
    std::unordered_map<std::uint32_t, std::uint64_t> m_cumulative_bytes_per_device{};
};

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
