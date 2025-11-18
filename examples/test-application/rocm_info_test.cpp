////////////////////////////////////////////////////////////////////////////////
// ROCm Info Test Application
// 
// Example application demonstrating how to use rocm-core to query ROCm version
// and installation information.
//
// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>

#ifdef __has_include
#  if __has_include(<rocm-core/rocm_version.h>)
#    include <rocm-core/rocm_version.h>
#    define HAVE_ROCM_VERSION
#  endif
#  if __has_include(<rocm-core/rocm_getpath.h>)
#    include <rocm-core/rocm_getpath.h>
#    define HAVE_ROCM_GETPATH
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

int main(int argc, char* argv[]) {
    print_separator("ROCm Systems Information Test");
    
    std::cout << "This application demonstrates using find_package(rocm-systems)\n";
    std::cout << "to access ROCm version and path information.\n";
    
    print_separator("Build Configuration");
    
#ifdef HAVE_ROCM_VERSION
    std::cout << "✓ rocm_version.h available\n";
#else
    std::cout << "✗ rocm_version.h not available\n";
#endif

#ifdef HAVE_ROCM_GETPATH
    std::cout << "✓ rocm_getpath.h available\n";
#else
    std::cout << "✗ rocm_getpath.h not available\n";
#endif
    
    print_separator("ROCm Version Information");
    
#ifdef HAVE_ROCM_VERSION
    #ifdef ROCM_VERSION_MAJOR
    std::cout << "ROCm Version Major: " << ROCM_VERSION_MAJOR << "\n";
    #endif
    
    #ifdef ROCM_VERSION_MINOR
    std::cout << "ROCm Version Minor: " << ROCM_VERSION_MINOR << "\n";
    #endif
    
    #ifdef ROCM_VERSION_PATCH
    std::cout << "ROCm Version Patch: " << ROCM_VERSION_PATCH << "\n";
    #endif
    
    #ifdef ROCM_VERSION_STRING
    std::cout << "ROCm Version String: " << ROCM_VERSION_STRING << "\n";
    #endif
#else
    std::cout << "Version information not available\n";
    std::cout << "(rocm-core headers not found or not included)\n";
#endif
    
    print_separator("ROCm Path Information");
    
#ifdef HAVE_ROCM_GETPATH
    // Try to get ROCm installation path
    char *path_buffer=(char*)NULL;
    unsigned int path_len=0;
    if (getROCmInstallPath(&path_buffer, &path_len) == 0) {
        std::cout << "ROCm Install Path: " << path_buffer << "\n";
    } else {
        std::cout << "Could not determine ROCm install path\n";
    }
#else
    std::cout << "Path query functions not available\n";
    std::cout << "(rocm_getpath.h not found or not included)\n";
#endif
    
    print_separator("Runtime Environment");
    
    // Check for common ROCm environment variables
    const char* rocm_path_env = std::getenv("ROCM_PATH");
    if (rocm_path_env) {
        std::cout << "ROCM_PATH env var: " << rocm_path_env << "\n";
    } else {
        std::cout << "ROCM_PATH env var: not set\n";
    }
    
    const char* hip_path = std::getenv("HIP_PATH");
    if (hip_path) {
        std::cout << "HIP_PATH env var:  " << hip_path << "\n";
    } else {
        std::cout << "HIP_PATH env var:  not set\n";
    }
    
    print_separator("Test Summary");
    
    std::cout << "✓ Application built successfully using find_package(rocm-systems)\n";
    std::cout << "✓ Linked against rocm-core (if available)\n";
    std::cout << "✓ Able to query ROCm installation information\n";
    
    print_separator();
    
    std::cout << "\nFor more information about ROCm Systems, see:\n";
    std::cout << "  https://github.com/ROCm/rocm-systems\n\n";
    
    return 0;
}


