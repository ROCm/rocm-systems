// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <memory>

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

namespace rocprofsys
{
namespace pmc
{
namespace drivers
{
namespace amd_smi
{

#if ROCPROFSYS_USE_ROCM > 0

/**
 * @brief Thin wrapper around AMD SMI C API for dependency injection and testing.
 *
 * This struct provides static methods that directly forward to the AMD SMI library.
 * It serves as an abstraction layer that can be mocked in tests through the
 * driver_factory.
 */
struct driver
{
    /**
     * @brief Initialize the AMD SMI library.
     * @param init_flags Initialization flags (default: AMDSMI_INIT_AMD_GPUS).
     * @return AMD SMI status code indicating success or failure.
     */
    static amdsmi_status_t init(uint64_t init_flags = AMDSMI_INIT_AMD_GPUS)
    {
        return amdsmi_init(init_flags);
    }

    /**
     * @brief Shutdown the AMD SMI library.
     * @return AMD SMI status code indicating success or failure.
     */
    static amdsmi_status_t shutdown() { return amdsmi_shut_down(); }

    /**
     * @brief Get AMD SMI library version information.
     * @param version Pointer to structure to receive version information.
     * @return AMD SMI status code indicating success or failure.
     */
    static amdsmi_status_t get_version(amdsmi_version_t* version)
    {
        return amdsmi_get_lib_version(version);
    }

    /**
     * @brief Get all socket handles in the system.
     * @param socket_count Pointer to receive the number of sockets (input/output).
     * @param socket_handles Pointer to array to receive socket handles (can be nullptr
     * for count query).
     * @return AMD SMI status code indicating success or failure.
     */
    static amdsmi_status_t get_socket_handles(uint32_t*             socket_count,
                                              amdsmi_socket_handle* socket_handles)
    {
        return amdsmi_get_socket_handles(socket_count, socket_handles);
    }

    /**
     * @brief Get processor handles for a specific socket.
     * @param socket_handle Socket to query.
     * @param processor_count Pointer to receive the number of processors (input/output).
     * @param processor_handles Pointer to array to receive processor handles (can be
     * nullptr for count query).
     * @return AMD SMI status code indicating success or failure.
     */
    static amdsmi_status_t get_processor_handles(
        amdsmi_socket_handle socket_handle, uint32_t* processor_count,
        amdsmi_processor_handle* processor_handles)
    {
        return amdsmi_get_processor_handles(socket_handle, processor_count,
                                            processor_handles);
    }

    /**
     * @brief Get the type of a processor (GPU, NIC, etc.).
     * @param processor_handle Processor to query.
     * @param processor_type Pointer to receive the processor type.
     * @return AMD SMI status code indicating success or failure.
     */
    static amdsmi_status_t get_processor_type(amdsmi_processor_handle processor_handle,
                                              processor_type_t*       processor_type)
    {
        return amdsmi_get_processor_type(processor_handle, processor_type);
    }

    /**
     * @brief Get GPU memory usage for a specific memory type.
     * @param processor_handle GPU processor to query.
     * @param type Memory type (e.g., VRAM, GTT).
     * @param usage Pointer to receive memory usage in bytes.
     * @return AMD SMI status code indicating success or failure.
     */
    static amdsmi_status_t get_memory_usage(amdsmi_processor_handle processor_handle,
                                            amdsmi_memory_type_t type, uint64_t* usage)
    {
        return amdsmi_get_gpu_memory_usage(processor_handle, type, usage);
    }

    /**
     * @brief Get GPU metrics information (temperature, power, clocks, etc.).
     * @param processor_handle GPU processor to query.
     * @param metrics Pointer to structure to receive GPU metrics.
     * @return AMD SMI status code indicating success or failure.
     */
    static amdsmi_status_t get_metrics_info(amdsmi_processor_handle processor_handle,
                                            amdsmi_gpu_metrics_t*   metrics)
    {
        return amdsmi_get_gpu_metrics_info(processor_handle, metrics);
    }
};

/**
 * @brief Factory for creating driver instances.
 *
 * Provides a factory method for creating driver instances. This enables
 * dependency injection and allows for substituting mock drivers in tests.
 */
struct driver_factory
{
    using driver_t = driver;

    /**
     * @brief Create a new driver instance.
     * @return Shared pointer to the driver instance.
     */
    static std::shared_ptr<driver_t> create_driver()
    {
        return std::make_shared<driver_t>();
    }
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace amd_smi
}  // namespace drivers
}  // namespace pmc
}  // namespace rocprofsys
