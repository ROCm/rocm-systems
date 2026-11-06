// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/hipfile/types.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace rocprofsys::pmc::device_providers::hipfile
{

/**
 * @brief Device provider for hipFile per-GPU telemetry.
 *
 * hipFile needs no enumeration API: its stats are indexed by HIP ordinal, so the
 * provider hands out one device per mapped ordinal. @c type_indices[k] is the
 * profiler GPU index of HIP ordinal k. All devices share a single backend session,
 * which is what lets one sampling interval cost one hipFile query and gives every
 * device in that interval the same coherent snapshot.
 *
 * @tparam BackendFactory Factory for creating hipFile backend instances.
 */
template <typename BackendFactory>
class provider
{
public:
    using backend_t = BackendFactory::backend_t;

    provider()
    : m_backend(BackendFactory::create_backend())
    {}

    ~provider() = default;

    provider(const provider&)            = delete;
    provider& operator=(const provider&) = delete;
    provider(provider&&)                 = default;
    provider& operator=(provider&&)      = default;

    /**
     * @brief Create one device per HIP ordinal, tagged with the profiler GPU index.
     *
     * @param type_indices Element k is the profiler @c device_type_index of HIP ordinal
     *                    k. Clamped to the snapshot capacity.
     */
    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_devices(
        const std::vector<std::size_t>& type_indices)
    {
        const auto slots =
            std::min<std::size_t>(type_indices.size(), backends::hipfile::MAX_GPUS);

        std::vector<std::shared_ptr<Device>> devices;
        devices.reserve(slots);
        for(std::size_t slot = 0; slot < slots; ++slot)
        {
            devices.push_back(
                std::make_shared<Device>(m_backend, slot, type_indices[slot]));
        }
        return devices;
    }

    void init() {}

    void shutdown()
    {
        if(m_backend)
        {
            m_backend->shutdown();
        }
    }

private:
    std::shared_ptr<backend_t> m_backend;
};

/**
 * @brief Factory for creating hipFile provider instances.
 */
template <typename BackendFactory>
struct provider_factory
{
    using provider_t = provider<BackendFactory>;

    static std::shared_ptr<provider_t> create() { return std::make_shared<provider_t>(); }
};

}  // namespace rocprofsys::pmc::device_providers::hipfile
