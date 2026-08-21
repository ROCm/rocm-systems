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
 * hipFile needs no enumeration API: its stats are indexed by GPU ordinal, so the
 * provider hands out one device per requested ordinal. All devices share a single
 * backend session, which is what lets one sampling interval cost one hipFile query
 * and gives every device in that interval the same coherent snapshot.
 *
 * @tparam BackendFactory Factory for creating hipFile backend instances.
 */
template <typename BackendFactory>
class provider
{
public:
    using backend_t = typename BackendFactory::backend_t;

    provider()
    : m_backend(BackendFactory::create_backend())
    {}

    ~provider() = default;

    provider(const provider&)            = delete;
    provider& operator=(const provider&) = delete;
    provider(provider&&)                 = default;
    provider& operator=(provider&&)      = default;

    /**
     * @brief Create one device per GPU ordinal in [0, gpu_count).
     *
     * @param gpu_count Number of ordinals to expose; clamped to the snapshot capacity.
     */
    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_devices(std::size_t gpu_count)
    {
        const auto slots = std::min<std::size_t>(gpu_count, backends::hipfile::MAX_GPUS);

        std::vector<std::shared_ptr<Device>> devices;
        devices.reserve(slots);
        for(std::size_t ordinal = 0; ordinal < slots; ++ordinal)
        {
            devices.push_back(std::make_shared<Device>(m_backend, ordinal));
        }
        return devices;
    }

    void init() {}

    void shutdown()
    {
        if(m_backend) m_backend->shutdown();
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
