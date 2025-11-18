////////////////////////////////////////////////////////////////////////////////
// HSA Runtime Test Application
// 
// Example application demonstrating how to use ROCm Systems find_package
// to access HSA Runtime functionality.
//
// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cstdint>

#ifdef __has_include
#  if __has_include(<hsa/hsa.h>)
#    include <hsa/hsa.h>
#    define HAVE_HSA_RUNTIME
#    include "hsa/hsa_ext_amd.h"
#  endif
#endif

void print_separator(const std::string& title = "") {
    std::cout << "\n";
    std::cout << "========================================";
    if (!title.empty()) {
        std::cout << "\n" << title;
        std::cout << "\n========================================";
    }
    std::cout << "\n";
}

#ifdef HAVE_HSA_RUNTIME

// Callback to iterate over HSA agents
hsa_status_t agent_callback(hsa_agent_t agent, void* data) {
    char name[64] = {0};
    hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
    
    hsa_device_type_t device_type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
    
    std::cout << "  Agent: " << name;
    
    if (device_type == HSA_DEVICE_TYPE_CPU) {
        std::cout << " (CPU)";
    } else if (device_type == HSA_DEVICE_TYPE_GPU) {
        std::cout << " (GPU)";
        
        // Get additional GPU info
        uint32_t cu_count = 0;
        hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &cu_count);
        std::cout << " - Compute Units: " << cu_count;
    } else if (device_type == HSA_DEVICE_TYPE_DSP) {
        std::cout << " (DSP)";
    }
    
    std::cout << "\n";
    
    int* agent_count = static_cast<int*>(data);
    (*agent_count)++;
    
    return HSA_STATUS_SUCCESS;
}

int test_hsa_runtime() {
    print_separator("HSA Runtime Test");
    
    // Initialize HSA runtime
    hsa_status_t status = hsa_init();
    if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "✗ Failed to initialize HSA runtime\n";
        std::cerr << "  Status code: " << status << "\n";
        std::cerr << "\nThis might be because:\n";
        std::cerr << "  - No ROCm-compatible devices are available\n";
        std::cerr << "  - ROCm drivers are not properly installed\n";
        std::cerr << "  - Running in an environment without GPU access\n";
        return 1;
    }
    
    std::cout << "✓ HSA Runtime initialized successfully\n";
    
    // Get HSA system info
    print_separator("System Information");
    
    uint16_t major, minor;
    status = hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &major);
    if (status == HSA_STATUS_SUCCESS) {
        hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, &minor);
        std::cout << "HSA Runtime Version: " << major << "." << minor << "\n";
    }
    
    // Enumerate agents
    print_separator("Available HSA Agents");
    
    int agent_count = 0;
    status = hsa_iterate_agents(agent_callback, &agent_count);
    
    if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "✗ Failed to iterate agents\n";
    } else if (agent_count == 0) {
        std::cout << "No HSA agents found\n";
    } else {
        std::cout << "\nTotal agents found: " << agent_count << "\n";
    }
    
    // Shutdown HSA runtime
    hsa_shut_down();
    std::cout << "\n✓ HSA Runtime shutdown successfully\n";
    
    return 0;
}

#else

int test_hsa_runtime() {
    print_separator("HSA Runtime Test");
    std::cout << "✗ HSA Runtime headers not available\n";
    std::cout << "\nTo enable HSA Runtime support:\n";
    std::cout << "  1. Configure rocm-systems CMAKE config\n";
    std::cout << "  2. Install rocm-systems\n";
    std::cout << "  3. Rebuild this test application\n";
    return 1;
}

#endif

int main(int argc, char* argv[]) {
    print_separator("ROCm Systems Test: HSA Runtime");
    
    std::cout << "This application demonstrates using find_package(rocm-systems)\n";
    std::cout << "to access HSA Runtime functionality.\n";
    
    int result = test_hsa_runtime();
    
    print_separator("Test Summary");
    
    if (result == 0) {
        std::cout << "✓ HSA Runtime test completed successfully\n";
        std::cout << "✓ Application built with find_package(rocm-systems)\n";
    } else {
        std::cout << "✗ HSA Runtime test encountered issues\n";
        std::cout << "  (This may be expected in environments without GPU access)\n";
    }
    
    print_separator();
    
    std::cout << "\nFor more information about ROCm Systems, see:\n";
    std::cout << "  https://github.com/ROCm/rocm-systems\n\n";
    
    return result;
}


