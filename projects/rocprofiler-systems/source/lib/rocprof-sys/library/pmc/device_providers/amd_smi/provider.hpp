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

namespace rocprofsys
{
namespace pmc
{
namespace device_providers
{
namespace amd_smi
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

    ~provider() = default;

    // Non-copyable, but movable
    provider(const provider&)            = delete;
    provider& operator=(const provider&) = delete;
    provider(provider&&)                 = default;
    provider& operator=(provider&&)      = default;

    /**
     * @brief Get AMD SMI library version.
     * @return Const reference to the version information.
     */
    [[nodiscard]] const version& get_version() const noexcept { return m_version; }

    /**
     * @brief Get driver instance.
     * @return Shared pointer to driver (used by collectors for low-level API calls).
     */
    [[nodiscard]] std::shared_ptr<driver_t> get_driver() const noexcept
    {
        return m_driver_api;
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
     * @brief Get processor handles for a specific socket (GPUs only).
     *
     * Queries the AMD SMI driver for GPU processor handles associated with
     * the specified socket.
     *
     * @param socket_handle Socket to query.
     * @return Vector of GPU processor handles for the specified socket.
     * @throws std::runtime_error If querying processor handles fails.
     *
     * @note This function only returns GPUs. For NICs, use
     * get_processor_handles_by_type() with AMDSMI_PROCESSOR_TYPE_AMD_NIC.
     */
    [[nodiscard]] std::vector<amdsmi_processor_handle> get_processor_handles(
        amdsmi_socket_handle socket_handle)
    {
        uint32_t count = 0;
        check_amd_smi_status(
            m_driver_api->get_processor_handles(socket_handle, &count, nullptr),
            "Failed to get processor count!");

        std::vector<amdsmi_processor_handle> handles(count);
        if(count > 0)
        {
            check_amd_smi_status(m_driver_api->get_processor_handles(
                                     socket_handle, &count, handles.data()),
                                 "Failed to get processor handles!");
        }

        return handles;
    }

    /**
     * @brief Get processor handles of a specific type for a socket.
     *
     * Queries the AMD SMI driver for processor handles of a specific type
     * (GPU, NIC, CPU) associated with the specified socket.
     *
     * @param socket_handle Socket to query.
     * @param processor_type Type of processor to enumerate.
     * @return Vector of processor handles of the specified type (empty if none found
     *         or if the socket doesn't support the requested processor type).
     *
     * @note This is required for enumerating NICs. get_processor_handles()
     * only returns GPUs.
     */
    [[nodiscard]] std::vector<amdsmi_processor_handle> get_processor_handles_by_type(
        amdsmi_socket_handle socket_handle, processor_type_t processor_type)
    {
        uint32_t count  = 0;
        auto     status = m_driver_api->get_processor_handles_by_type(
            socket_handle, processor_type, nullptr, &count);

        // Return empty vector if no processors of this type exist on this socket
        // or if the query is not supported for this socket type
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
     * @brief Shutdown AMD SMI driver.
     *
     * @note Should be called once during application shutdown.
     */
    void shutdown() { m_driver_api->shutdown(); }

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

    std::shared_ptr<driver_t> m_driver_api;  ///< Driver API instance
    version                   m_version{};   ///< AMD SMI library version
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

}  // namespace amd_smi
}  // namespace device_providers
}  // namespace pmc
}  // namespace rocprofsys
