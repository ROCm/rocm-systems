// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace rocprofsys::domains::callback::rccl
{
template <policies::domain_service::externals Externals>
struct transferred_bytes_tracker
{
    struct add_bytes_result
    {
        std::uint64_t cumulative_bytes;
        bool          is_first_registration;
    };

    [[nodiscard]] static add_bytes_result add_bytes(std::uint32_t rccl_device_idx,
                                                    size_t        bytes)
    {
        auto thread_state_guard =
            Externals::state_thread::scoped(Externals::state_thread::Internal);

        std::atomic<std::uint64_t>* counter  = nullptr;
        bool                        is_first = false;
        {
            const std::unique_lock<std::mutex> lock{ m_mutex };
            auto [iter, inserted] =
                m_cumulative_bytes_per_device.try_emplace(rccl_device_idx);
            counter  = &iter->second;
            is_first = inserted;
        }

        const auto cumulative =
            counter->fetch_add(bytes, std::memory_order_relaxed) + bytes;
        return { cumulative, is_first };
    }

private:
    static inline std::mutex m_mutex;
    static inline std::unordered_map<std::uint32_t, std::atomic<std::uint64_t>>
        m_cumulative_bytes_per_device;
};

}  // namespace rocprofsys::domains::callback::rccl
