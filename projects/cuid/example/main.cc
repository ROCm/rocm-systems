/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include "include/amd_cuid.h"
#include <climits>

int main() {
    amdcuid_status_t err;
    uint32_t handle_count = 0;
    uint32_t available_handle_count = 0;
    std::vector<amdcuid_id_t> handles;

    // Retry until the available_handle_count matches the handle_count
    do {
        handle_count = available_handle_count;
        handles.resize(handle_count);
        err = amdcuid_get_all_handles(
            handles.data(),
            &available_handle_count);
        if (err != AMDCUID_STATUS_SUCCESS && err != AMDCUID_STATUS_INSUFFICIENT_SIZE) {
            std::cerr << "Failed to get device handles. Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
            return 1;
        }
    } while (handle_count != available_handle_count);

    // sort handles for GPUs and CPUs separately
    std::vector<amdcuid_id_t> cpu_handles;
    std::vector<amdcuid_id_t> gpu_handles;
    for (uint32_t i = 0; i < handles.size(); ++i) {
        amdcuid_device_type_t device_type;
        uint32_t device_type_size = sizeof(device_type);
        err = amdcuid_query_device_property(handles[i], AMDCUID_QUERY_DEVICE_TYPE, &device_type, &device_type_size);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get device type for handle #" << i << ". Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
            continue;
        }

        if (device_type == AMDCUID_DEVICE_TYPE_GPU) {
            // copy the GPU handle to gpu_handles vector
            gpu_handles.push_back(handles[i]);
        }
        else if (device_type == AMDCUID_DEVICE_TYPE_CPU) {
            // copy the CPU handle to cpu_handles vector
            cpu_handles.push_back(handles[i]);
        }
    }

    std::cout << "Discovered " << gpu_handles.size() << " GPU(s):" << std::endl;
    for (uint32_t i = 0; i < gpu_handles.size(); ++i) {
        uint16_t vendor_id = 0;
        uint32_t data_size = sizeof(vendor_id);
        err = amdcuid_query_device_property(gpu_handles[i], AMDCUID_QUERY_VENDOR_ID, &vendor_id, &data_size);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get vendor_id for GPU #" << i << ". Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
        }

        char device_node[128] = {0};
        uint32_t device_node_len = sizeof(device_node);
        err = amdcuid_query_device_property(gpu_handles[i], AMDCUID_QUERY_DEVICE_PATH, device_node, &device_node_len);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get device node for GPU #" << i << ". Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
            device_node[0] = '\0';
        }

        amdcuid_id_t derived_id = {};
        uint32_t derived_id_size = sizeof(derived_id);
        err = amdcuid_query_device_property(gpu_handles[i], AMDCUID_QUERY_DERIVED_CUID, &derived_id, &derived_id_size);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get derived CUID for GPU #" << i << ". Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
        }

        std::cout << "GPU #" << i
                  << std::dec
                  << " vendor_id: " << vendor_id
                  << " DeviceNode: " << device_node
                  << "  CUID: " << amdcuid_id_to_string(derived_id) << std::endl;
        std::cout << " Handle: " << amdcuid_id_to_string(gpu_handles[i]) << std::endl;
        std::cout << std::endl;
    }

    // filter handles for CPU devices

    std::cout << "Discovered " << cpu_handles.size() << " CPU(s):" << std::endl;

    for (uint32_t i = 0; i < cpu_handles.size(); ++i) {
        uint16_t vendor_id = 0;
        uint32_t data_size = sizeof(vendor_id);
        err = amdcuid_query_device_property(cpu_handles[i], AMDCUID_QUERY_VENDOR_ID, &vendor_id, &data_size);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get vendor ID for CPU #" << i << ". Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
            vendor_id = 0;
        }

        uint16_t core = 0;
        data_size = sizeof(core);
        err = amdcuid_query_device_property(cpu_handles[i], AMDCUID_QUERY_CORE_ID, &core, &data_size);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get core for CPU #" << i << ". Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
            core = 0;
        }

        amdcuid_id_t derived_id = {};
        uint32_t derived_id_size = sizeof(derived_id);
        err = amdcuid_query_device_property(cpu_handles[i], AMDCUID_QUERY_DERIVED_CUID, &derived_id, &derived_id_size);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get derived CUID for CPU #" << i << ". Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
        }

        std::cout << "CPU #" << i
                  << std::dec
                  << " Core: " << core
                  << " VendorID: " << vendor_id
                  << "  CUID: " << amdcuid_id_to_string(derived_id) << std::endl;
        std::cout << " Handle: " << amdcuid_id_to_string(cpu_handles[i]) << std::endl;
        std::cout << std::endl;
    }
    std::string example_device_path;
    uint32_t path_length = PATH_MAX;
    amdcuid_status_t status = amdcuid_query_device_property(gpu_handles[0], AMDCUID_QUERY_DEVICE_PATH, &example_device_path, &path_length);

    // example for getting specific device handle by device path and querying its properties
    amdcuid_id_t device_handle = {};
    err = amdcuid_get_handle_by_dev_path(example_device_path.c_str(), AMDCUID_DEVICE_TYPE_GPU, &device_handle);
    if (err != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Failed to get device handle for path " << example_device_path
                  << ". Error code: " << err << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
        return 1;
    }

    // handle itself is also the derived CUID, so we can print it directly
    std::cout << "Device at path " << example_device_path << " has derived CUID: " << amdcuid_id_to_string(device_handle) << std::endl;

    std::string example_bdf;
    uint32_t bdf_length = 64;
    status = amdcuid_query_device_property(gpu_handles[0], AMDCUID_QUERY_DEVICE_PATH, &example_bdf, &bdf_length);

    // example for getting a specific device handle by BDF and querying its properties
    amdcuid_id_t bdf_device_handle = {};
    err = amdcuid_get_handle_by_bdf(example_bdf.c_str(), AMDCUID_DEVICE_TYPE_GPU, &bdf_device_handle);
    if (err != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Failed to get device handle for BDF " << example_bdf
                  << ". Error code: " << err << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
        return 1;
    }

    // handle itself is also the derived CUID, so we can print it directly
    std::cout << "Device at BDF " << example_bdf << " has derived CUID: " << amdcuid_id_to_string(bdf_device_handle) << std::endl;

    return 0;
}