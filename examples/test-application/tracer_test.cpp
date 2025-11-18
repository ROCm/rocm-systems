////////////////////////////////////////////////////////////////////////////////
// ROCtracer Test Application
// 
// Example application demonstrating how to use ROCm Systems find_package
// to optionally enable roctracer support.
//
// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>

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
    print_separator("ROCm Systems Test: roctracer");
    
    std::cout << "This application demonstrates using find_package(rocm-systems)\n";
    std::cout << "with optional component support.\n";
    
    print_separator("Build Configuration");
    
#ifdef HAVE_ROCTRACER
    std::cout << "✓ Built with roctracer support\n";
    std::cout << "  (roctracer was found and enabled)\n";
    
    // In a real application, you would use roctracer API here
    // For this example, we just demonstrate that the library is available
    
    print_separator("ROCtracer Information");
    
    std::cout << "roctracer is available for use.\n";
    std::cout << "\nIn a real application, you would:\n";
    std::cout << "  - Initialize roctracer\n";
    std::cout << "  - Enable tracing for specific APIs\n";
    std::cout << "  - Register callbacks for traced events\n";
    std::cout << "  - Process trace data\n";
    std::cout << "  - Shutdown roctracer\n";
    
    print_separator("Example roctracer Usage");
    
    std::cout << "Typical roctracer initialization:\n\n";
    std::cout << "  #include <roctracer/roctracer.h>\n";
    std::cout << "  \n";
    std::cout << "  // Initialize roctracer\n";
    std::cout << "  roctracer_properties_t properties {};\n";
    std::cout << "  roctracer_open_pool(&properties);\n";
    std::cout << "  \n";
    std::cout << "  // Enable tracing\n";
    std::cout << "  roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API);\n";
    std::cout << "  \n";
    std::cout << "  // Your application code here\n";
    std::cout << "  \n";
    std::cout << "  // Disable and cleanup\n";
    std::cout << "  roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API);\n";
    std::cout << "  roctracer_close_pool();\n";
    
#else
    std::cout << "✗ Built without roctracer support\n";
    std::cout << "  (roctracer was not found or not enabled)\n";
    
    print_separator("How to Enable roctracer");
    
    std::cout << "To build this application with roctracer support:\n\n";
    std::cout << "1. Build ROCm Systems with roctracer:\n";
    std::cout << "   cd rocm-systems/build\n";
    std::cout << "   cmake .. -DBUILD_ROCTRACER=ON\n";
    std::cout << "   cmake --build .\n";
    std::cout << "   sudo cmake --install .\n\n";
    std::cout << "2. Rebuild this test application:\n";
    std::cout << "   cd examples/test-application/build\n";
    std::cout << "   cmake ..\n";
    std::cout << "   cmake --build .\n\n";
    std::cout << "The CMakeLists.txt will automatically detect and use roctracer.\n";
#endif
    
    print_separator("Test Summary");
    
#ifdef HAVE_ROCTRACER
    std::cout << "✓ roctracer support is enabled\n";
    std::cout << "✓ Application built with optional component\n";
    std::cout << "✓ Ready for roctracer API usage\n";
#else
    std::cout << "✓ Application built without roctracer\n";
    std::cout << "✓ Demonstrates conditional component support\n";
    std::cout << "ℹ roctracer can be added later by following steps above\n";
#endif
    
    print_separator();
    
    std::cout << "\nKey Takeaways:\n";
    std::cout << "  • find_package(rocm-systems) supports optional COMPONENTS\n";
    std::cout << "  • Components can be conditionally enabled at build time\n";
    std::cout << "  • Applications can adapt based on available components\n";
    std::cout << "  • No code changes needed when components become available\n";
    
    std::cout << "\nFor more information:\n";
    std::cout << "  https://github.com/ROCm/rocm-systems\n";
    std::cout << "  https://rocm.docs.amd.com/\n\n";
    
    return 0;
}


