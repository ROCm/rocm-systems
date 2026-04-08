// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <memory>

#include <amd_smi/amdsmi.h>

namespace rocprofsys::pmc::drivers::amd_smi
{

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
     *
     * @note This function only returns GPU processors. For NICs, use
     * get_processor_handles_by_type() with AMDSMI_PROCESSOR_TYPE_AMD_NIC.
     */
    static amdsmi_status_t get_processor_handles(
        amdsmi_socket_handle socket_handle, uint32_t* processor_count,
        amdsmi_processor_handle* processor_handles)
    {
        return amdsmi_get_processor_handles(socket_handle, processor_count,
                                            processor_handles);
    }

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    /**
     * @brief Get processor handles of a specific type for a socket.
     * @param socket_handle Socket to query.
     * @param processor_type Type of processor to enumerate (GPU, NIC, CPU).
     * @param processor_handles Pointer to array to receive processor handles (can be
     * nullptr for count query).
     * @param processor_count Pointer to receive the number of processors (input/output).
     * @return AMD SMI status code indicating success or failure.
     *
     * @note This is required for enumerating NICs. amdsmi_get_processor_handles()
     * only returns GPUs. Requires AMD SMI >= 26.3.
     */
    static amdsmi_status_t get_processor_handles_by_type(
        amdsmi_socket_handle socket_handle, processor_type_t processor_type,
        amdsmi_processor_handle* processor_handles, uint32_t* processor_count)
    {
        return amdsmi_get_processor_handles_by_type(socket_handle, processor_type,
                                                    processor_handles, processor_count);
    }
#endif

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

    /**
     * @brief Get GPU ASIC information including vendor and product names.
     * @param processor_handle GPU processor to query.
     * @param asic_info Pointer to structure to receive ASIC information.
     * @return AMD SMI status code indicating success or failure.
     */
    static amdsmi_status_t get_gpu_asic_info(amdsmi_processor_handle processor_handle,
                                             amdsmi_asic_info_t*     asic_info)
    {
        return amdsmi_get_gpu_asic_info(processor_handle, asic_info);
    }

    /**
     * @brief Get GPU process list with per-process SDMA usage.
     *
     * Returns the list of processes using the GPU, including cumulative SDMA
     * usage in microseconds per process. Used for computing SDMA utilization.
     *
     * @param processor_handle GPU processor to query.
     * @param max_processes Pointer to max process count (input/output).
     * @param list Pointer to array to receive process info (can be nullptr for count
     * query).
     * @return AMD SMI status code indicating success or failure.
     *
     * @note Requires AMD SMI >= 26.3. Guarded by AMD_SMI_SDMA_SUPPORTED.
     */
#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
    static amdsmi_status_t get_gpu_process_list(amdsmi_processor_handle processor_handle,
                                                uint32_t*               max_processes,
                                                amdsmi_proc_info_t*     list)
    {
        return amdsmi_get_gpu_process_list(processor_handle, max_processes, list);
    }
#endif

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    /**
     * @brief Get NIC ASIC information including vendor and product names.
     * @param processor_handle NIC processor to query.
     * @param asic_info Pointer to structure to receive ASIC information.
     * @return AMD SMI status code indicating success or failure.
     */
    static amdsmi_status_t get_nic_asic_info(amdsmi_processor_handle processor_handle,
                                             amdsmi_nic_asic_info_t* asic_info)
    {
        return amdsmi_get_nic_asic_info(processor_handle, asic_info);
    }

    /**
     * @brief Get NIC port information.
     * @param processor_handle NIC processor to query.
     * @param port_info Pointer to structure to receive port information.
     * @return AMD SMI status code indicating success or failure.
     */
    static amdsmi_status_t get_nic_port_info(amdsmi_processor_handle processor_handle,
                                             amdsmi_nic_port_info_t* port_info)
    {
        return amdsmi_get_nic_port_info(processor_handle, port_info);
    }

    /**
     * @brief Get NIC RDMA device information.
     * @param processor_handle NIC processor to query.
     * @param rdma_info Pointer to structure to receive RDMA device info.
     * @return AMD SMI status code indicating success or failure.
     */
    static amdsmi_status_t get_nic_rdma_dev_info(
        amdsmi_processor_handle         processor_handle,
        amdsmi_nic_rdma_devices_info_t* rdma_info)
    {
        return amdsmi_get_nic_rdma_dev_info(processor_handle, rdma_info);
    }

    /**
     * @brief Get NIC RDMA port statistics.
     * @param processor_handle NIC processor to query.
     * @param rdma_port_idx RDMA port index.
     * @param num_stats Pointer to number of stats (input/output).
     * @param stats Pointer to array to receive statistics (can be nullptr for count).
     * @return AMD SMI status code indicating success or failure.
     */
    static amdsmi_status_t get_nic_rdma_port_statistics(
        amdsmi_processor_handle processor_handle, uint8_t rdma_port_idx,
        uint32_t* num_stats, amdsmi_nic_stat_t* stats)
    {
        return amdsmi_get_nic_rdma_port_statistics(processor_handle, rdma_port_idx,
                                                   num_stats, stats);
    }
#endif
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

}  // namespace rocprofsys::pmc::drivers::amd_smi
