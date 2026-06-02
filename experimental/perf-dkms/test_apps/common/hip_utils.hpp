#ifndef HIP_UTILS_HPP
#define HIP_UTILS_HPP

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <vector>

// HIP error checking macro
#define HIP_CHECK(cmd)                                                         \
    do {                                                                       \
        hipError_t error = (cmd);                                              \
        if (error != hipSuccess) {                                             \
            std::cerr << "HIP Error: '" << hipGetErrorString(error)            \
                      << "' (" << error << ") at " << __FILE__ << ":"         \
                      << __LINE__ << std::endl;                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// HIP kernel launch error checking
#define HIP_CHECK_LAST()                                                       \
    do {                                                                       \
        hipError_t error = hipGetLastError();                                  \
        if (error != hipSuccess) {                                             \
            std::cerr << "HIP Kernel Launch Error: '"                          \
                      << hipGetErrorString(error) << "' (" << error            \
                      << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace hip_utils {

// Get device properties
inline void printDeviceInfo() {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No HIP devices found!" << std::endl;
        exit(EXIT_FAILURE);
    }

    int device;
    HIP_CHECK(hipGetDevice(&device));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    std::cout << "Using device " << device << ": " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total global memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Warp size: " << prop.warpSize << std::endl;
}

// Allocate device memory
template<typename T>
T* allocateDevice(size_t count) {
    T* ptr = nullptr;
    HIP_CHECK(hipMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

// Allocate and initialize device memory
template<typename T>
T* allocateDeviceWithValue(size_t count, T value) {
    T* ptr = allocateDevice<T>(count);
    std::vector<T> host_data(count, value);
    HIP_CHECK(hipMemcpy(ptr, host_data.data(), count * sizeof(T), hipMemcpyHostToDevice));
    return ptr;
}

// Free device memory
template<typename T>
void freeDevice(T* ptr) {
    if (ptr) {
        HIP_CHECK(hipFree(ptr));
    }
}

// Copy from device to host
template<typename T>
void copyFromDevice(T* host, const T* device, size_t count) {
    HIP_CHECK(hipMemcpy(host, device, count * sizeof(T), hipMemcpyDeviceToHost));
}

// Copy from host to device
template<typename T>
void copyToDevice(T* device, const T* host, size_t count) {
    HIP_CHECK(hipMemcpy(device, host, count * sizeof(T), hipMemcpyHostToDevice));
}

} // namespace hip_utils

#endif // HIP_UTILS_HPP
