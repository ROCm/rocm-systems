/**
 * Padding for device linker.
 * 
 * This file provides padding to ensure the .hip_fatbin section is large enough
 * to hold the merged device code from the device linker.
 */

#include <hip/hip_runtime.h>

// Padding to make .hip_fatbin section large enough for merged device code
// The device linker replaces .hip_fatbin with the merged specialized kernels
__device__ __attribute__((used)) char __device_linker_padding[2*1024*1024];
