// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include "common.hpp"
#include "processor.hpp"

#include <amd_smi/amdsmi.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace rocprofsys
{
namespace amd_smi
{
/**
 * @class service
 * @tparam driver_factory The factory type used to create the driver interface.
 * @brief Manages AMD SMI driver initialization and processor enumeration.
 *
 * The service class provides methods to initialize the AMD SMI driver, retrieve
 * its version, and enumerate all available processors. It uses a driver factory
 * to create the driver instance and wraps lower-level driver calls with error
 * checking.
 */
template <typename driver_factory>
struct service
{
    using driver_t           = typename driver_factory::driver_t;
    using processor_vector_t = std::vector<std::shared_ptr<processor<driver_t>>>;

    /**
     * @brief Constructs a service object and initializes the AMD SMI driver.
     * @throws std::runtime_error if driver initialization or version retrieval
     * fails.
     */
    service()
    : m_driver_api(driver_factory::create_driver())
    {
        check_status(m_driver_api->init(), "Fail to initialize AMD SMI driver!");

        amdsmi_version_t version;
        check_status(m_driver_api->get_version(&version),
                     "Fail to get AMD SMI driver version!");
        m_version = { .numeric_representation = { .major   = version.major,
                                                  .minor   = version.minor,
                                                  .release = version.release },
                      .string_representation  = std::string{ version.build } };
    }

    // ~service() { m_driver_api->shutdown(); }
    /**
     * @brief Returns the AMD SMI driver version information.
     * @return Reference to the version struct.
     */
    const version& get_version() { return m_version; }

    /**
     * @brief Enumerates all available processors managed by the AMD SMI driver.
     * @return Vector of shared pointers to processor objects.
     * @throws std::runtime_error if processor enumeration fails.
     */
    processor_vector_t get_processors(
        const std::function<processor_vector_t(processor_vector_t& processors)> filter =
            nullptr)
    {
        std::vector<std::shared_ptr<processor<driver_t>>> processors{};
        auto   socket_handles = get_socket_handles();
        size_t index          = 0;
        for(auto& socket_handle : socket_handles)
        {
            auto processor_handles = get_processor_handles(socket_handle);
            for(auto& processor_handle : processor_handles)
            {
                processor_type_t processor_type;
                check_status(
                    m_driver_api->get_processor_type(processor_handle, &processor_type),
                    "Failed to get processor type!");
                // We can implement filter later
                processors.emplace_back(std::make_shared<processor<driver_t>>(
                    m_driver_api, processor_handle, processor_type, index++));
            }
        };

        return filter == nullptr ? processors : filter(processors);
    }

private:
    /**
     * @brief Retrieves all socket handles from the AMD SMI driver.
     * @return Vector of socket handles.
     * @throws std::runtime_error if socket handle retrieval fails.
     */
    std::vector<amdsmi_socket_handle> get_socket_handles()
    {
        uint32_t count;
        check_status(m_driver_api->get_socket_handles(&count, nullptr),
                     "Failed to get socket handles!");
        std::vector<amdsmi_socket_handle> socket_handles(count);
        if(count > 0)
        {
            check_status(m_driver_api->get_socket_handles(&count, socket_handles.data()),
                         "Failed to get socket handles!");
        }

        return socket_handles;
    }

    /**
     * @brief Retrieves all processor handles for a given socket.
     * @param socket_handle The socket handle to query.
     * @return Vector of processor handles.
     * @throws std::runtime_error if processor handle retrieval fails.
     */
    std::vector<amdsmi_processor_handle> get_processor_handles(
        amdsmi_socket_handle& socket_handle)
    {
        uint32_t count;
        check_status(m_driver_api->get_processor_handles(socket_handle, &count, nullptr),
                     "Fail to get processor count for provided socket!");
        std::vector<amdsmi_processor_handle> processor_handles(count);
        if(count > 0)
        {
            check_status(m_driver_api->get_processor_handles(socket_handle, &count,
                                                             processor_handles.data()),
                         "Failed to get processor handles for provided socket!");
        }

        return processor_handles;
    }

private:
    /** Shared pointer to the driver interface used for SMI operations. */
    std::shared_ptr<driver_t> m_driver_api;
    /** AMD SMI driver version information. */
    version m_version;
};

}  // namespace amd_smi
}  // namespace rocprofsys
