// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/common/types.hpp"

#include <concepts>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::device_providers::amd_smi
{

/**
 * @brief Concept that a BackendFactory template argument must satisfy.
 *
 * Enumerates every expression @c provider<F> evaluates on the factory and on
 * the session it produces, so a mismatch is caught at the template boundary
 * rather than deep inside the provider body.
 *
 * Checked expressions:
 *  - @c F::backend_t          — the session type
 *  - @c F::create_backend()   — returns @c shared_ptr<backend_t>
 *  - @c sess.initialize()     — lifecycle init (may throw)
 *  - @c sess.shutdown()       — lifecycle teardown
 *  - @c sess.get_lib_version() — version query used in constructor
 *  - @c sess.enumerate_gpu_handles() — GPU handle enumeration
 *  - @c sess.enumerate_nic_handles() — NIC handle enumeration (AINIC builds only)
 */
template <typename F>
concept backend_factory_contract =
    requires { typename F::backend_t; } &&
    requires {
        { F::create_backend() } -> std::same_as<std::shared_ptr<typename F::backend_t>>;
    } &&
    requires(typename F::backend_t sess) {
        { sess.initialize() };
        { sess.shutdown() };
        { sess.get_lib_version() };
        { sess.enumerate_gpu_handles() };
    }
#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    &&
    requires(typename F::backend_t sess) {
        { sess.enumerate_nic_handles() };
    }
#endif
;

/**
 * @brief AMD SMI device provider for initialization and device enumeration.
 *
 * Manages AMD SMI backend lifecycle (init/shutdown) and provides device
 * enumeration. Shared between GPU and NIC collectors.
 *
 * Follows the same factory pattern as procfs and rocprofiler_sdk providers:
 * BackendFactory::create_backend() instantiates the backend; the provider
 * stores it as @c m_backend_api and dispatches lifecycle and enumeration
 * calls through that instance.
 *
 * @tparam BackendFactory  Provides @c backend_t and @c create_backend().
 */
template <backend_factory_contract BackendFactory>
class provider
{
    std::shared_ptr<typename BackendFactory::backend_t> m_backend_api;
    version                                             m_version{};

public:
    using backend_t = BackendFactory::backend_t;

    /**
     * @brief Initialize the AMD SMI backend and retrieve version information.
     * @throws std::runtime_error on AMD SMI failure.
     */
    provider()
    : m_backend_api(BackendFactory::create_backend())
    {
        m_backend_api->initialize();

        auto ver                                 = m_backend_api->get_lib_version();
        m_version.numeric_representation.major   = ver.major;
        m_version.numeric_representation.minor   = ver.minor;
        m_version.numeric_representation.release = ver.release;
        m_version.string_representation          = ver.build ? ver.build : "";
    }

    ~provider() noexcept
    {
        if(m_backend_api) m_backend_api->shutdown();
    }

    // Non-copyable, movable
    provider(const provider&)            = delete;
    provider& operator=(const provider&) = delete;

    provider(provider&& other) noexcept
    : m_backend_api(std::move(other.m_backend_api))
    , m_version(std::move(other.m_version))
    {
        other.m_backend_api.reset();
    }

    provider& operator=(provider&& other) noexcept
    {
        if(this != &other)
        {
            if(m_backend_api) m_backend_api->shutdown();
            m_backend_api = std::move(other.m_backend_api);
            m_version     = std::move(other.m_version);
            other.m_backend_api.reset();
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
        if(m_backend_api)
        {
            m_backend_api->shutdown();
            m_backend_api.reset();
        }
    }

    /**
     * @brief Get all GPU devices.
     *
     * Enumerates GPU handles via the shared session, then wraps each handle in
     * a @c Device::backend_type (i.e. @c device_backend<backend_t>) constructed
     * with the session and the handle.
     *
     * @tparam Device  Concrete device type; exposes @c backend_type used to
     *                 construct per-device backend instances from handles.
     */
    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_gpu_devices()
    {
        auto handles = m_backend_api->enumerate_gpu_handles();

        std::vector<std::shared_ptr<Device>> result;
        result.reserve(handles.size());

        std::size_t index = 0;
        for(auto handle : handles)
        {
            auto proxy =
                std::make_shared<typename Device::backend_type>(m_backend_api, handle);
            result.push_back(std::make_shared<Device>(std::move(proxy), index++));
        }

        return result;
    }

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    /**
     * @brief Get all NIC devices.
     */
    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_nic_devices()
    {
        auto handles = m_backend_api->enumerate_nic_handles();

        std::vector<std::shared_ptr<Device>> result;
        result.reserve(handles.size());

        std::size_t index = 0;
        for(auto handle : handles)
        {
            auto proxy =
                std::make_shared<typename Device::backend_type>(m_backend_api, handle);
            result.push_back(std::make_shared<Device>(std::move(proxy), index++));
        }

        return result;
    }
#endif
};

/**
 * @brief Factory for creating AMD SMI provider instances.
 */
template <backend_factory_contract BackendFactory>
struct provider_factory
{
    using provider_t = provider<BackendFactory>;

    static std::shared_ptr<provider_t> create() { return std::make_shared<provider_t>(); }
};

}  // namespace rocprofsys::pmc::device_providers::amd_smi
