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

int main() {
    amdcuid_status_t err;
    uint32_t gpu_count = 0;
    uint32_t available_gpu_count = 0;
    std::vector<amdcuid_id_t> gpu_handles;

    // Retry until the available_gpu_count matches the gpu_count
    do {
        gpu_count = available_gpu_count;
        gpu_handles.resize(gpu_count);
        err = amdcuid_get_handles(
            AMDCUID_DEVICE_TYPE_SET_GPU,
            &gpu_count,
            gpu_handles.data(),
            &available_gpu_count);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get GPU handles. Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
            return 1;
        }
    } while (gpu_count != available_gpu_count);

    std::cout << "Discovered " << gpu_count << " GPU(s):" << std::endl;
    for (uint32_t i = 0; i < gpu_count; ++i) {
        char bdf[64] = {0};
        uint32_t bdf_len = sizeof(bdf);
        err = amdcuid_get_bdf(gpu_handles[i], bdf, &bdf_len);
        if (err != AMDCUID_STATUS_SUCCESS) {
            // skip for now due to gpu partitioning issues
            continue;
            // std::cerr << "Failed to get BDF for GPU #" << i << ". Error code: " << err
            //           << " (" << cuid_status_to_string(err) << ")" << std::endl;
            // bdf[0] = '\0';
        }

        char device_node[128] = {0};
        uint32_t device_node_len = sizeof(device_node);
        err = amdcuid_get_render_node(gpu_handles[i], device_node, &device_node_len);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get device node for GPU #" << i << ". Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
            device_node[0] = '\0';
        }

        amdcuid_id_t derived_id = {};
        err = amdcuid_get_derived_cuid(gpu_handles[i], &derived_id);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get derived CUID for GPU #" << i << ". Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
        }

        std::cout << "GPU #" << i
                  << std::dec
                  << " BDF: " << bdf
                  << " DeviceNode: " << device_node
                  << "  CUID: ";
        for (int j = 0; j < 16; ++j) {
            printf("%02x", derived_id.bytes[j]);
        }
        std::cout << std::endl;
    }

    // Same as above but for CPUs
    uint32_t cpu_count = 0;
    uint32_t available_cpu_count = 0;
    std::vector<amdcuid_id_t> cpu_handles;

    // Retry until the available_cpu_count matches the cpu_count
    do {
        cpu_count = available_cpu_count;
        cpu_handles.resize(cpu_count);
        err = amdcuid_get_handles(
            AMDCUID_DEVICE_TYPE_SET_CPU,
            &cpu_count,
            cpu_handles.data(),
            &available_cpu_count);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get CPU handles. Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
            return 1;
        }
    } while (cpu_count != available_cpu_count);

    std::cout << "Discovered " << cpu_count << " CPU(s):" << std::endl;

    for (uint32_t i = 0; i < cpu_count; ++i) {
        uint16_t vendor_id = 0;
        err = amdcuid_get_vendor_id(cpu_handles[i], &vendor_id);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get vendor ID for CPU #" << i << ". Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
            vendor_id = 0;
        }

        uint16_t core = 0;
        err = amdcuid_get_cpu_core(cpu_handles[i], &core);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get core for CPU #" << i << ". Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
            core = 0;
        }

        amdcuid_id_t derived_id = {};
        err = amdcuid_get_derived_cuid(cpu_handles[i], &derived_id);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get derived CUID for CPU #" << i << ". Error code: " << err
                      << " (" << amdcuid_status_to_string(err) << ")" << std::endl;
        }

        std::cout << "CPU #" << i
                  << std::dec
                  << " Core: " << core
                  << " VendorID: " << vendor_id
                  << "  CUID: ";
        for (int j = 0; j < 16; ++j) {
            printf("%02x", derived_id.bytes[j]);
        }
        std::cout << std::endl;
    }
    return 0;
}