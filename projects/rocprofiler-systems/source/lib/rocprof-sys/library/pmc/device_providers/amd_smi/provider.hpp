// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/common/types.hpp"
#include "library/pmc/device_providers/amd_smi/drivers/driver.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <amd_smi/amdsmi.h>

namespace rocprofsys::pmc::device_providers::amd_smi
{

/**
 * @brief AMD SMI device provider for initialization and device enumeration.
 *
 * This class manages the AMD SMI driver initialization/shutdown and provides
 * access to raw device handles. It is designed to be shared by collectors
 * (GPU and NIC). Device object creation and filtering is the responsibility
 * of the collector.
 *
 * @tparam DriverFactory Factory for creating AMD SMI driver instances.
 */
template <typename DriverFactory>
class provider
{
private:
    /**
     * @brief Check AMD SMI status and throw on error.
     * @param status AMD SMI status code.
     * @param error_message Error message to include in exception.
     */
    static void check_amd_smi_status(amdsmi_status_t status, const char* error_message)
    {
        if(status != AMDSMI_STATUS_SUCCESS)
        {
            std::stringstream ss;
            ss << error_message << " AMD SMI Error code: " << status;
            throw std::runtime_error(ss.str());
        }
    }

    /**
     * @brief Get all socket handles.
     *
     * Queries the AMD SMI driver for all available socket handles in the system.
     *
     * @return Vector of socket handles.
     * @throws std::runtime_error If querying socket handles fails.
     */
    [[nodiscard]] std::vector<amdsmi_socket_handle> get_socket_handles()
    {
        uint32_t count = 0;
        check_amd_smi_status(m_driver_api->get_socket_handles(&count, nullptr),
                             "Failed to get socket count!");

        std::vector<amdsmi_socket_handle> handles(count);
        if(count > 0)
        {
            check_amd_smi_status(m_driver_api->get_socket_handles(&count, handles.data()),
                                 "Failed to get socket handles!");
        }

        return handles;
    }

    /**
     * @brief Get processor handles of a specific type for a socket.
     *
     * @param socket_handle Socket to query.
     * @param processor_type Type of processor to enumerate.
     * @return Vector of processor handles of the specified type (empty if none found).
     */
    [[nodiscard]] std::vector<amdsmi_processor_handle> get_processor_handles_for_socket(
        amdsmi_socket_handle socket_handle, processor_type_t processor_type)
    {
        uint32_t count  = 0;
        auto     status = m_driver_api->get_processor_handles_by_type(
            socket_handle, processor_type, nullptr, &count);

        if(status != AMDSMI_STATUS_SUCCESS || count == 0)
        {
            return {};
        }

        std::vector<amdsmi_processor_handle> handles(count);
        status = m_driver_api->get_processor_handles_by_type(
            socket_handle, processor_type, handles.data(), &count);

        if(status != AMDSMI_STATUS_SUCCESS)
        {
            return {};
        }

        return handles;
    }

    /**
     * @brief Enumerate devices of a specific processor type.
     *
     * @tparam Device The device type to create.
     * @param processor_type AMD SMI processor type.
     * @return Vector of shared pointers to device objects.
     */
    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> enumerate_devices(
        processor_type_t processor_type)
    {
        std::vector<std::shared_ptr<Device>> devices;

        auto   socket_handles = get_socket_handles();
        size_t index          = 0;

        for(auto& socket_handle : socket_handles)
        {
            auto handles =
                get_processor_handles_for_socket(socket_handle, processor_type);
            for(auto& handle : handles)
            {
                devices.push_back(std::make_shared<Device>(m_driver_api, handle,
                                                           processor_type, index));
                index++;
            }
        }

        return devices;
    }

    std::shared_ptr<typename DriverFactory::driver_t>
            m_driver_api;  ///< Driver API instance
    version m_version{};   ///< AMD SMI library version

public:
    using driver_t = typename DriverFactory::driver_t;

    /**
     * @brief Construct and initialize the AMD SMI device provider.
     *
     * Creates the driver instance, initializes the AMD SMI driver, and retrieves version
     * information.
     *
     * @throws std::runtime_error If AMD SMI initialization fails or version retrieval
     * fails.
     */
    provider()
    : m_driver_api(DriverFactory::create_driver())
    {
        // Initialize AMD SMI driver
        check_amd_smi_status(m_driver_api->init(),
                             "Failed to initialize AMD SMI driver!");

        // Get and store version information
        amdsmi_version_t ver;
        check_amd_smi_status(m_driver_api->get_version(&ver),
                             "Failed to get AMD SMI driver version!");

        m_version.numeric_representation.major   = ver.major;
        m_version.numeric_representation.minor   = ver.minor;
        m_version.numeric_representation.release = ver.release;
        m_version.string_representation          = ver.build;
    }

    ~provider() noexcept
    {
        if(m_driver_api)
        {
            m_driver_api->shutdown();
        }
    }

    // Non-copyable, but movable
    provider(const provider&)            = delete;
    provider& operator=(const provider&) = delete;

    provider(provider&& other) noexcept
    : m_driver_api(std::move(other.m_driver_api))
    , m_version(std::move(other.m_version))
    {
        other.m_driver_api.reset();  // Prevent double-shutdown
    }

    provider& operator=(provider&& other) noexcept
    {
        if(this != &other)
        {
            if(m_driver_api)
            {
                m_driver_api->shutdown();
            }
            m_driver_api = std::move(other.m_driver_api);
            m_version    = std::move(other.m_version);
            other.m_driver_api.reset();
        }
        return *this;
    }

    /**
     * @brief Get AMD SMI library version.
     * @return Const reference to the version information.
     */
    [[nodiscard]] const version& get_version() const noexcept { return m_version; }

    /**
     * @brief Shutdown the AMD SMI driver.
     *
     * Releases the driver API and cleans up resources. Safe to call multiple times.
     */
    void shutdown()
    {
        if(m_driver_api)
        {
            m_driver_api->shutdown();
            m_driver_api.reset();
        }
    }

    /**
     * @brief Get all devices of a specific type.
     *
     * Enumerates all devices of the specified type across all sockets.
     *
     * @tparam Device The device type to create.
     * @param type The device type to enumerate (GPU or NIC).
     * @return Vector of shared pointers to device objects.
     */
    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_devices(device_type type)
    {
        processor_type_t proc_type = (type == device_type::GPU)
                                         ? AMDSMI_PROCESSOR_TYPE_AMD_GPU
                                         : AMDSMI_PROCESSOR_TYPE_AMD_NIC;
        return enumerate_devices<Device>(proc_type);
    }
};

/**
 * @brief Factory for creating AMD SMI provider instances.
 *
 * @tparam DriverFactory Factory type for creating AMD SMI driver instances.
 */
template <typename DriverFactory>
struct provider_factory
{
    using provider_t = provider<DriverFactory>;

    /**
     * @brief Create a new provider instance.
     * @return Shared pointer to a newly created provider.
     */
    static std::shared_ptr<provider_t> create() { return std::make_shared<provider_t>(); }
};

}  // namespace rocprofsys::pmc::device_providers::amd_smi
