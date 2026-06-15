// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/common/types.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::device_providers::amd_smi
{

/**
 * @brief AMD SMI device provider for initialization and device enumeration.
 *
 * Manages AMD SMI backend lifecycle (init/shutdown) and provides device
 * enumeration. Shared between GPU and NIC collectors.
 *
 * @tparam BackendFactory  Provides @c backend_t, a type that exposes static
 *                         lifecycle, versioning, and enumeration methods.
 */
template <typename BackendFactory>
class provider
{
    version m_version{};
    bool    m_initialized{ false };

public:
    using backend_t = BackendFactory::backend_t;

    /**
     * @brief Initialize the AMD SMI backend and retrieve version information.
     * @throws std::runtime_error on AMD SMI failure.
     */
    provider()
    {
        backend_t::initialize();
        m_initialized = true;

        auto ver                                 = backend_t::get_lib_version();
        m_version.numeric_representation.major   = ver.major;
        m_version.numeric_representation.minor   = ver.minor;
        m_version.numeric_representation.release = ver.release;
        m_version.string_representation          = ver.build ? ver.build : "";
    }

    ~provider() noexcept
    {
        if(m_initialized) backend_t::shutdown();
    }

    // Non-copyable, but movable
    provider(const provider&)            = delete;
    provider& operator=(const provider&) = delete;

    provider(provider&& other) noexcept
    : m_version(std::move(other.m_version))
    , m_initialized(std::exchange(other.m_initialized, false))
    {}

    provider& operator=(provider&& other) noexcept
    {
        if(this != &other)
        {
            if(m_initialized) backend_t::shutdown();
            m_version     = std::move(other.m_version);
            m_initialized = std::exchange(other.m_initialized, false);
        }
        return *this;
    }

    /**
     * @brief Get AMD SMI library version.
     */
    [[nodiscard]] const version& get_version() const noexcept { return m_version; }

    /**
     * @brief Shutdown the AMD SMI backend. Safe to call multiple times.
     */
    void shutdown()
    {
        if(m_initialized)
        {
            backend_t::shutdown();
            m_initialized = false;
        }
    }

    /**
     * @brief Get all GPU devices.
     *
     * @tparam Device  Concrete device type; exposes @c backend_type used to
     *                 construct per-device backend instances from handles.
     */
    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_gpu_devices()
    {
        auto handles = backend_t::enumerate_gpu_handles();

        std::vector<std::shared_ptr<Device>> result;
        result.reserve(handles.size());

        size_t index = 0;
        for(auto handle : handles)
        {
            auto dev_backend = std::make_shared<typename Device::backend_type>(handle);
            result.push_back(std::make_shared<Device>(std::move(dev_backend), index++));
        }

        return result;
    }

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    /**
     * @brief Get all NIC devices.
     *
     * @tparam Device  Concrete NIC device type; exposes @c backend_type.
     */
    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_nic_devices()
    {
        auto handles = backend_t::enumerate_nic_handles();

        std::vector<std::shared_ptr<Device>> result;
        result.reserve(handles.size());

        size_t index = 0;
        for(auto handle : handles)
        {
            auto dev_backend = std::make_shared<typename Device::backend_type>(handle);
            result.push_back(std::make_shared<Device>(std::move(dev_backend), index++));
        }

        return result;
    }
#endif
};

/**
 * @brief Factory for creating AMD SMI provider instances.
 */
template <typename BackendFactory>
struct provider_factory
{
    using provider_t = provider<BackendFactory>;

    static std::shared_ptr<provider_t> create() { return std::make_shared<provider_t>(); }
};

}  // namespace rocprofsys::pmc::device_providers::amd_smi
