// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * @file tests/hip-fatbin-unload/main.cpp
 *
 * @brief Loads a kernel after a HIP library was unloaded without launching any of its kernels
 *
 * The test app's fat binary is registered at startup, so the library's fat binary, registered by
 * the dlopen below, is the most recently registered one. The dlclose unregisters it and unmaps it.
 * With deferred loading, no code object is loaded in between, so the profiler has not parsed the
 * library's fat binary while it was mapped. Each time an HSA executable is frozen, the profiler
 * parses the most recently registered HIP fat binary if it has not done so yet. The first freeze
 * after the unmap comes from the HIP calls below: the HIP runtime loads its own blit kernels when
 * it creates its first device queue, and the kernel launch loads the test app's code object. A
 * profiler which still references the unregistered fat binary reads unmapped memory at that
 * first freeze.
 *
 * The HIP runtime maps memory of its own after the unload, which can reuse the library's old
 * address range and turn the stale read into a silent read of unrelated memory. The library's
 * old address range is therefore reserved with no access rights, so any stale read faults.
 */

#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <link.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (call);                                                                  \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            std::cerr << "ERROR: HIP error '" << hipGetErrorString(_err) << "' at " << __FILE__    \
                      << ":" << __LINE__ << "\n";                                                  \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while(0)

constexpr int expected_value = 42;

namespace
{
using address_range_t = std::pair<uintptr_t, size_t>;

struct library_ranges
{
    const char*                  name   = nullptr;
    std::vector<address_range_t> ranges = {};
};

int
collect_load_segments(dl_phdr_info* info, size_t /* size */, void* data)
{
    auto* lib = static_cast<library_ranges*>(data);
    if(!info->dlpi_name || std::strcmp(info->dlpi_name, lib->name) != 0) return 0;

    const auto page_size = static_cast<uintptr_t>(::sysconf(_SC_PAGESIZE));
    for(int i = 0; i < info->dlpi_phnum; ++i)
    {
        const auto& phdr = info->dlpi_phdr[i];
        if(phdr.p_type != PT_LOAD || phdr.p_memsz == 0) continue;

        auto begin = (info->dlpi_addr + phdr.p_vaddr) & ~(page_size - 1);
        auto end =
            (info->dlpi_addr + phdr.p_vaddr + phdr.p_memsz + page_size - 1) & ~(page_size - 1);
        lib->ranges.emplace_back(begin, end - begin);
    }
    return 1;
}
}  // namespace

__global__ void
fatbin_unload_app_kernel(int* out)
{
    *out = expected_value;
}

int
main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cerr << "ERROR: usage: " << argv[0] << " <hip-library>\n";
        return EXIT_FAILURE;
    }

    const char* library = argv[1];

    void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if(!handle)
    {
        std::cerr << "ERROR: dlopen(" << library << ") failed: " << dlerror() << "\n";
        return EXIT_FAILURE;
    }

    using marker_t = int (*)();
    auto marker    = reinterpret_cast<marker_t>(dlsym(handle, "fatbin_unload_library_marker"));
    if(!marker || marker() != 1)
    {
        std::cerr << "ERROR: library marker function not found in " << library << "\n";
        return EXIT_FAILURE;
    }

    auto lib = library_ranges{library, {}};
    dl_iterate_phdr(collect_load_segments, &lib);
    if(lib.ranges.empty())
    {
        std::cerr << "ERROR: no loaded segments found for " << library << "\n";
        return EXIT_FAILURE;
    }

    if(dlclose(handle) != 0)
    {
        std::cerr << "ERROR: dlclose(" << library << ") failed: " << dlerror() << "\n";
        return EXIT_FAILURE;
    }

    // the scenario requires the library (and its fat binary) to be unmapped
    if(dlopen(library, RTLD_NOW | RTLD_NOLOAD) != nullptr)
    {
        std::cout << "SKIP: " << library << " was not unloaded by dlclose\n";
        return EXIT_SUCCESS;
    }

    for(const auto& [begin, length] : lib.ranges)
    {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        void* addr     = reinterpret_cast<void*>(begin);
        void* reserved = ::mmap(
            addr, length, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if(reserved != addr)
        {
            std::cout << "SKIP: the unloaded library's address range at " << addr
                      << " could not be reserved: " << std::strerror(errno) << "\n";
            return EXIT_SUCCESS;
        }
    }

    int* d_out = nullptr;
    int  h_out = 0;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));
    HIP_CHECK(hipMemset(d_out, 0, sizeof(int)));
    hipLaunchKernelGGL(fatbin_unload_app_kernel, dim3(1), dim3(1), 0, 0, d_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_out));

    if(h_out != expected_value)
    {
        std::cerr << "ERROR: kernel wrote " << h_out << ", expected " << expected_value << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Test PASSED: Successfully launched a kernel after unloading a HIP library!\n";
    return EXIT_SUCCESS;
}
