
#include <iostream>
#include <vector>
#include <cstdint>
#include "cuid.h"

inline const char* cuid_status_to_string(amdcuid_status_t status) {
    switch (status) {
        case AMDCUID_STATUS_SUCCESS: return "SUCCESS";
        case AMDCUID_STATUS_FILE_NOT_FOUND: return "FILE_NOT_FOUND";
        case AMDCUID_STATUS_DEVICE_NOT_FOUND: return "DEVICE_NOT_FOUND";
        case AMDCUID_STATUS_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case AMDCUID_STATUS_PERMISSION_DENIED: return "PERMISSION_DENIED";
        case AMDCUID_STATUS_UNSUPPORTED: return "UNSUPPORTED";
        case AMDCUID_STATUS_NOT_INIT: return "NOT_INIT";
        case AMDCUID_STATUS_WRONG_DEVICE_TYPE: return "WRONG_DEVICE_TYPE";
        case AMDCUID_STATUS_INSUFFICIENT_SIZE: return "INSUFFICIENT_SIZE";
        case AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND: return "AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND";
        case AMDCUID_STATUS_HW_FINGERPRINT_FORMAT_ERROR: return "AMDCUID_STATUS_HW_FINGERPRINT_FORMAT_ERROR";
        case AMDCUID_STATUS_HW_FINGERPRINT_PERMISSION_DENIED: return "AMDCUID_STATUS_HW_FINGERPRINT_PERMISSION_DENIED";
        default: return "UNKNOWN_ERROR";
    }
}

int main() {
    amdcuid_status_t err;

    uint32_t gpu_count = 0;
    uint32_t available_gpu_count = 0;
    std::vector<amdcuid_handle> gpu_handles;

    // Retry until the available_gpu_count matches the gpu_count
    do {
        gpu_count = available_gpu_count;
        gpu_handles.resize(gpu_count);
        err = amdcuid_get_handles(
            AMDCUID_DEVICE_TYPE_SET_GPU,
            gpu_count,
            gpu_handles.data(),
            &available_gpu_count);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get GPU handles. Error code: " << err
                      << " (" << cuid_status_to_string(err) << ")" << std::endl;
            return 1;
        }
    } while (gpu_count != available_gpu_count);

    std::cout << "Discovered " << gpu_count << " GPU(s):" << std::endl;
    for (uint32_t i = 0; i < gpu_count; ++i) {
        char bdf[64] = {0};
        uint32_t bdf_len = sizeof(bdf);
        err = amdcuid_get_bdf(gpu_handles[i], bdf, &bdf_len);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get BDF for GPU #" << i << ". Error code: " << err
                      << " (" << cuid_status_to_string(err) << ")" << std::endl;
            bdf[0] = '\0';
        }

        char device_node[128] = {0};
        uint32_t device_node_len = sizeof(device_node);
        err = amdcuid_get_render_node(gpu_handles[i], device_node, &device_node_len);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get device node for GPU #" << i << ". Error code: " << err
                      << " (" << cuid_status_to_string(err) << ")" << std::endl;
            device_node[0] = '\0';
        }

        amdcuid secondary_id = {};
        err = amdcuid_get_secondary_cuid(gpu_handles[i], &secondary_id);
        if (err != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Failed to get secondary CUID for GPU #" << i << ". Error code: " << err
                      << " (" << cuid_status_to_string(err) << ")" << std::endl;
        }

        std::cout << "GPU #" << i
                  << std::dec
                  << " BDF: " << bdf
                  << " DeviceNode: " << device_node
                  << "  CUID: ";
        for (int j = 0; j < 16; ++j) {
            printf("%02x", secondary_id.bytes[j]);
        }
        std::cout << std::endl;
    }
    return 0;
}