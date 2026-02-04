// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * @file librocprofiler_python_preload.cpp
 * @brief LD_PRELOAD shim for early rocprofiler registration
 *
 * This library provides a constructor that runs before the main program
 * (and before HSA/HIP runtime initialization) to register the rocprofiler
 * tool callbacks. This enables dispatch interception for Python profiling.
 *
 * IMPORTANT: This library uses dlopen/dlsym to avoid link-time dependencies
 * on rocprofiler-sdk, which would cause HSA to initialize immediately when
 * the preload library is loaded, before PyTorch can initialize.
 *
 * Usage:
 *   LD_PRELOAD=/path/to/librocprofiler-python-preload.so python3 script.py
 */

#include <dlfcn.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

// Define minimal rocprofiler types needed for configuration
// These must match the rocprofiler-sdk definitions exactly

extern "C" {
// Status enum - we only need SUCCESS
enum rocprofiler_status_t
{
    ROCPROFILER_STATUS_SUCCESS = 0
};

// Client finalization callback type
typedef void (*rocprofiler_client_finalize_t)(void*);

// Client identifier
struct rocprofiler_client_id_t
{
    const char* name;
    uint64_t    handle;
};

// Tool configuration result
struct rocprofiler_tool_configure_result_t
{
    size_t size;
    int (*tool_init)(rocprofiler_client_finalize_t, void*);
    void (*tool_fini)(void*);
    void* tool_data;
};

// Function pointer type for rocprofiler_force_configure
typedef rocprofiler_status_t (*rocprofiler_force_configure_fn)(
    rocprofiler_tool_configure_result_t* (*) (uint32_t,
                                              const char*,
                                              uint32_t,
                                              rocprofiler_client_id_t*) );

// Function pointer type for rocprofiler_configure
typedef rocprofiler_tool_configure_result_t* (*rocprofiler_configure_fn)(
    uint32_t                 version,
    const char*              runtime_version,
    uint32_t                 priority,
    rocprofiler_client_id_t* id);
}

namespace
{
// Global state
static void*                    g_core_library_handle        = nullptr;
static void*                    g_rocprofiler_library_handle = nullptr;
static rocprofiler_configure_fn g_configure_func             = nullptr;
static rocprofiler_client_id_t* g_client_id                  = nullptr;

/**
 * @brief Minimal rocprofiler_configure implementation for this preload library
 *
 * This function is called by rocprofiler_force_configure to get our tool configuration.
 * It forwards to the real rocprofiler_configure in the core library if available.
 */
extern "C" rocprofiler_tool_configure_result_t*
preload_rocprofiler_configure(uint32_t                 version,
                              const char*              runtime_version,
                              uint32_t                 priority,
                              rocprofiler_client_id_t* id)
{
    // Store the client ID for later reference
    g_client_id = id;

    // If we have the core library loaded, forward to its rocprofiler_configure
    if(g_configure_func)
    {
        return g_configure_func(version, runtime_version, priority, id);
    }

    // Core library not loaded yet - return a minimal configuration
    // The real configuration will happen when the Python bindings load
    id->name = "rocprofiler-python-preload";

    static int (*dummy_init)(rocprofiler_client_finalize_t,
                             void*) = [](rocprofiler_client_finalize_t, void*) -> int { return 0; };

    static void (*dummy_fini)(void*) = [](void*) {};

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), dummy_init, dummy_fini, nullptr};

    return &cfg;
}

/**
 * @brief Constructor function that runs at library load time
 *
 * This constructor uses priority 101 (default is 65535, lower runs earlier)
 * to ensure it runs before most other constructors.
 *
 * Strategy:
 * 1. Try to dlopen the core library (librocprofiler-python-core.so) with RTLD_LAZY
 *    to get the rocprofiler_configure function without initializing HSA
 * 2. dlopen librocprofiler-sdk.so with RTLD_LAZY to get rocprofiler_force_configure
 * 3. Call rocprofiler_force_configure with our configure callback
 *
 * Using RTLD_LAZY means symbols are resolved on demand, so HSA won't initialize
 * until the symbols are actually called later (after PyTorch initializes).
 */
__attribute__((constructor(101))) void
rocprofiler_python_early_init()
{
    // Check if we should skip initialization (for debugging)
    if(std::getenv("ROCPROFILER_PYTHON_SKIP_PRELOAD"))
    {
        std::cerr << "[rocprofiler-python-preload] Skipping initialization (env var set)"
                  << std::endl;
        return;
    }

    // Try to load the core library first to get rocprofiler_configure
    // Use RTLD_LAZY to defer symbol resolution
    g_core_library_handle =
        dlopen("librocprofiler-python-core.so", RTLD_LAZY | RTLD_GLOBAL | RTLD_NOLOAD);
    if(!g_core_library_handle)
    {
        // Not already loaded, try to load it
        g_core_library_handle = dlopen("librocprofiler-python-core.so", RTLD_LAZY | RTLD_GLOBAL);
    }

    if(g_core_library_handle)
    {
        g_configure_func = reinterpret_cast<rocprofiler_configure_fn>(
            dlsym(g_core_library_handle, "rocprofiler_configure"));
        if(!g_configure_func)
        {
            std::cerr << "[rocprofiler-python-preload] Warning: Could not find "
                         "rocprofiler_configure in core library"
                      << std::endl;
        }
    }
    else
    {
        // Core library not found - that's okay, we'll use our minimal configure
        // This can happen if the library isn't in LD_LIBRARY_PATH yet
        // The real configuration will happen when Python imports the module
    }

    // Now load rocprofiler-sdk to get rocprofiler_force_configure
    // Use RTLD_LAZY to avoid initializing HSA immediately
    g_rocprofiler_library_handle =
        dlopen("librocprofiler-sdk.so", RTLD_LAZY | RTLD_GLOBAL | RTLD_NOLOAD);
    if(!g_rocprofiler_library_handle)
    {
        g_rocprofiler_library_handle = dlopen("librocprofiler-sdk.so", RTLD_LAZY | RTLD_GLOBAL);
    }

    if(!g_rocprofiler_library_handle)
    {
        std::cerr << "[rocprofiler-python-preload] Warning: Could not load librocprofiler-sdk.so: "
                  << dlerror() << std::endl;
        return;
    }

    // Get the rocprofiler_force_configure function
    auto force_configure_fn = reinterpret_cast<rocprofiler_force_configure_fn>(
        dlsym(g_rocprofiler_library_handle, "rocprofiler_force_configure"));

    if(!force_configure_fn)
    {
        std::cerr << "[rocprofiler-python-preload] Warning: Could not find "
                     "rocprofiler_force_configure: "
                  << dlerror() << std::endl;
        return;
    }

    // Register our configuration
    auto status = force_configure_fn(&preload_rocprofiler_configure);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "[rocprofiler-python-preload] Warning: force_configure returned error code "
                  << static_cast<int>(status) << std::endl;
    }
}

/**
 * @brief Destructor to clean up dlopen handles
 */
__attribute__((destructor)) void
rocprofiler_python_cleanup()
{
    // Don't actually close the libraries - they may still be in use
    // The OS will clean them up on process exit
}

}  // namespace
