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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _e = (call);                                                                    \
        if(_e != hipSuccess)                                                                       \
        {                                                                                          \
            std::cerr << "HIP error " << hipGetErrorName(_e) << " (" << static_cast<int>(_e)       \
                      << ") at " << __FILE__ << ":" << __LINE__ << " -> " << hipGetErrorString(_e) \
                      << std::endl;                                                                \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while(0)

extern "C" {
extern void
__hipUnregisterFatBinary(void** modules);
}

namespace
{
struct dtor
{
    dtor() = default;
    ~dtor()
    {
        std::cerr << "\nCalling __hipUnregisterFatBinary(nullptr)... \n" << std::flush;
        __hipUnregisterFatBinary(nullptr);
        std::cerr << "Calling __hipUnregisterFatBinary(nullptr)... Done\n" << std::flush;
    }
};

const char*
funcCacheToStr(hipFuncCache_t v)
{
    switch(v)
    {
        case hipFuncCachePreferNone: return "PreferNone";
        case hipFuncCachePreferShared: return "PreferShared";
        case hipFuncCachePreferL1: return "PreferL1";
        case hipFuncCachePreferEqual: return "PreferEqual";
        default: return "Unknown";
    }
}

const char*
shmemBankToStr(hipSharedMemConfig v)
{
    switch(v)
    {
        case hipSharedMemBankSizeDefault: return "Default";
        case hipSharedMemBankSizeFourByte: return "FourByte";
        case hipSharedMemBankSizeEightByte: return "EightByte";
        default: return "Unknown";
    }
}

struct LimitDesc
{
    hipLimit_t  limit;
    const char* name;
};

const std::vector<LimitDesc> kLimits = {
    {hipLimitStackSize, "hipLimitStackSize"},
    {hipLimitPrintfFifoSize, "hipLimitPrintfFifoSize"},
    {hipLimitMallocHeapSize, "hipLimitMallocHeapSize"},
};

void
printBytes(const char* label, size_t bytes, int width = 28)
{
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double      v       = static_cast<double>(bytes);
    int         u       = 0;
    while(v >= 1024.0 && u < 4)
    {
        v /= 1024.0;
        ++u;
    }
    std::cout << std::left << std::setw(width) << label << ": " << bytes << " (" << std::fixed
              << std::setprecision(2) << v << " " << units[u] << ")\n";
}

void
printArray3(const char* label, int v[3], int width = 28)
{
    std::cout << std::left << std::setw(width) << label << ": " << v[0] << ", " << v[1] << ", "
              << v[2] << "\n";
}

void
printBool(const char* label, int v, int width = 28)
{
    std::cout << std::left << std::setw(width) << label << ": " << (v ? "true" : "false") << "\n";
}

void
printIfNonzero(const char* label, int v, int width = 28)
{
    if(v != 0) std::cout << std::left << std::setw(width) << label << ": " << v << "\n";
}
}  // namespace

auto _dtor = std::unique_ptr<dtor>{};

int
main(int argc, char** argv)
{
    _dtor = std::make_unique<dtor>();

    const auto help_flags     = std::set<std::string_view>{"-h", "--help", "-?"};
    auto       report_devices = std::set<int>{};
    for(int i = 1; i < argc; ++i)
    {
        if(std::string_view{argv[i]}.find_first_not_of("0123456789") == std::string_view::npos)
        {
            report_devices.insert(std::atoi(argv[++i]));
        }
        else if(help_flags.count(std::string_view{argv[i]}) != 0)
        {
            std::cout << "Usage: " << argv[0] << "-h / --help / -? / [<device_id> ...]\n";
            return EXIT_SUCCESS;
        }
        else
        {
            std::cerr << "Usage: " << argv[0] << " [--skip-device N ...]\n";
            return EXIT_FAILURE;
        }
    }

    int        deviceCount = 0;
    hipError_t e           = hipGetDeviceCount(&deviceCount);
    if(e == hipErrorNoDevice)
    {
        std::cout << "No HIP devices found.\n";
        return EXIT_FAILURE;
    }
    HIP_CHECK(e);

    std::cout << "Found " << deviceCount << " HIP device(s)\n\n";
    if(report_devices.empty())
    {
        for(int i = 0; i < deviceCount; ++i)
            report_devices.insert(i);
    }

    for(int dev : report_devices)
    {
        HIP_CHECK(hipSetDevice(dev));

        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, dev));

        std::cout << "============================================================\n";
        std::cout << "Device " << dev << ": " << prop.name << "\n";
        std::cout << "============================================================\n";

        // --- Device Properties ---
        std::cout << "\n[Device Properties]\n";
        std::cout << std::left << std::setw(28) << "major.minor"
                  << ": " << prop.major << "." << prop.minor << "\n";

// AMD-specific helpful fields if present in your HIP version:
// prop.gcnArchName may exist; print defensively using std::string
#if defined(__HIP_PLATFORM_AMD__)
        if(!std::string_view{prop.gcnArchName}.empty())
        {
            std::cout << std::left << std::setw(28) << "gcnArchName"
                      << ": " << prop.gcnArchName << "\n";
        }
#endif

        printBytes("totalGlobalMem", prop.totalGlobalMem);
        printBytes("sharedMemPerBlock", prop.sharedMemPerBlock);
        printBytes("sharedMemPerMultiprocessor", prop.sharedMemPerMultiprocessor);
        std::cout << std::left << std::setw(28) << "regsPerBlock"
                  << ": " << prop.regsPerBlock << "\n";
        std::cout << std::left << std::setw(28) << "warpSize/wavefront"
                  << ": " << prop.warpSize << "\n";
        std::cout << std::left << std::setw(28) << "maxThreadsPerBlock"
                  << ": " << prop.maxThreadsPerBlock << "\n";
        printArray3("maxThreadsDim", prop.maxThreadsDim);
        printArray3("maxGridSize", prop.maxGridSize);
        std::cout << std::left << std::setw(28) << "clockRate (kHz)"
                  << ": " << prop.clockRate << "\n";
        std::cout << std::left << std::setw(28) << "memoryClockRate (kHz)"
                  << ": " << prop.memoryClockRate << "\n";
        std::cout << std::left << std::setw(28) << "memoryBusWidth (bits)"
                  << ": " << prop.memoryBusWidth << "\n";
        printBytes("totalConstMem", prop.totalConstMem);
        std::cout << std::left << std::setw(28) << "multiProcessorCount"
                  << ": " << prop.multiProcessorCount << "\n";
        std::cout << std::left << std::setw(28) << "l2CacheSize (bytes)"
                  << ": " << prop.l2CacheSize << "\n";
        std::cout << std::left << std::setw(28) << "maxThreadsPerMultiProcessor"
                  << ": " << prop.maxThreadsPerMultiProcessor << "\n";
        printBool("concurrentKernels", prop.concurrentKernels);
        printBool("cooperativeLaunch", prop.cooperativeLaunch);
        printBool("cooperativeMultiDeviceLaunch", prop.cooperativeMultiDeviceLaunch);
        printBool("managedMemory", prop.managedMemory);
        printBool("pageableMemoryAccess", prop.pageableMemoryAccess);
        printBool("concurrentManagedAccess", prop.concurrentManagedAccess);
        printIfNonzero("pciDomainID", prop.pciDomainID);
        printIfNonzero("pciBusID", prop.pciBusID);
        printIfNonzero("pciDeviceID", prop.pciDeviceID);
        std::cout << std::left << std::setw(28) << "isMultiGpuBoard"
                  << ": " << (prop.isMultiGpuBoard != 0 ? "true" : "false") << "\n";

        // --- Device Flags / Cache / SharedMem config are per current device context ---
        std::cout << "\n[Device Flags]\n";
        unsigned int flags = 0;
        HIP_CHECK(hipGetDeviceFlags(&flags));
        std::cout << "flags (hex): 0x" << std::hex << std::setw(8) << std::setfill('0') << flags
                  << std::dec << std::setfill(' ') << "\n";

// Optionally decode some common bits if present in your HIP:
#ifdef hipDeviceScheduleAuto
        if((flags & hipDeviceScheduleAuto) != 0u) std::cout << " - hipDeviceScheduleAuto\n";
#endif
#ifdef hipDeviceScheduleSpin
        if((flags & hipDeviceScheduleSpin) != 0u) std::cout << " - hipDeviceScheduleSpin\n";
#endif
#ifdef hipDeviceScheduleBlockingSync
        if((flags & hipDeviceScheduleBlockingSync) != 0u)
            std::cout << " - hipDeviceScheduleBlockingSync\n";
#endif
#ifdef hipDeviceMapHost
        if((flags & hipDeviceMapHost) != 0u) std::cout << " - hipDeviceMapHost\n";
#endif
#ifdef hipDeviceLmemResizeToMax
        if((flags & hipDeviceLmemResizeToMax) != 0u) std::cout << " - hipDeviceLmemResizeToMax\n";
#endif

        std::cout << "\n[Device Cache Config]\n";
        hipFuncCache_t cacheCfg{};
        HIP_CHECK(hipDeviceGetCacheConfig(&cacheCfg));
        std::cout << "cache config: " << funcCacheToStr(cacheCfg) << "\n";

        std::cout << "\n[Shared Memory Config]\n";
        hipSharedMemConfig shCfg{};
        HIP_CHECK(hipDeviceGetSharedMemConfig(&shCfg));
        std::cout << "shared mem bank size: " << shmemBankToStr(shCfg) << "\n";

        // --- Device Limits ---
        std::cout << "\n[Device Limits]\n";
        for(const auto& ld : kLimits)
        {
            size_t     value = 0;
            hipError_t le    = hipDeviceGetLimit(&value, ld.limit);
            if(le == hipSuccess)
            {
                if(ld.limit == hipLimitPrintfFifoSize || ld.limit == hipLimitMallocHeapSize ||
                   ld.limit == hipLimitStackSize)
                {
                    printBytes(ld.name, value);
                }
                else
                {
                    std::cout << std::left << std::setw(28) << ld.name << ": " << value << "\n";
                }
            }
            else if(le == hipErrorUnsupportedLimit)
            {
                std::cout << std::left << std::setw(28) << ld.name << ": unsupported\n";
            }
            else
            {
                std::cout << std::left << std::setw(28) << ld.name << ": error ("
                          << hipGetErrorName(le) << ")\n";
            }
        }

        std::cout << std::endl;
    }

    return 0;
}
