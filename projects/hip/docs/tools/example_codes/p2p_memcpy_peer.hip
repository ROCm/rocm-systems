// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// [sphinx-start]
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#define HIP_CHECK(expression)                                                  \
  {                                                                            \
    const hipError_t status = expression;                                      \
    if (status != hipSuccess) {                                                \
      std::cerr << "HIP error " << status << ": " << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

__global__ void fillKernel(double *data, std::size_t elems) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < elems)
    data[idx] = idx * 2.0;
}

int main() {
  int deviceCount;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    std::cout << "This example requires at least two HIP devices." << std::endl;
    return EXIT_SUCCESS;
  }

  constexpr std::size_t elems = 1024;
  constexpr std::size_t size = elems * sizeof(double);

  int srcDevice = 0;
  int dstDevice = 1;

  // Enable peer access in both directions if the devices support it.
  // The peer copy functions also work when peer access is not available.
  int canAccessPeerSrcDst = 0;
  int canAccessPeerDstSrc = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeerSrcDst, srcDevice, dstDevice));
  HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeerDstSrc, dstDevice, srcDevice));
  const bool peerAccess = canAccessPeerSrcDst && canAccessPeerDstSrc;
  if (peerAccess) {
    HIP_CHECK(hipSetDevice(srcDevice));
    HIP_CHECK(hipDeviceEnablePeerAccess(dstDevice, 0));
    HIP_CHECK(hipSetDevice(dstDevice));
    HIP_CHECK(hipDeviceEnablePeerAccess(srcDevice, 0));
  }
  std::cout << "Peer access between device " << srcDevice << " and device "
            << dstDevice << (peerAccess ? " is enabled." : " is not supported.")
            << std::endl;

  // Fill a buffer on the source device. A second buffer on the source device
  // receives the data when it is copied back.
  double *srcData;
  double *returnData;
  HIP_CHECK(hipSetDevice(srcDevice));
  HIP_CHECK(hipMalloc(&srcData, size));
  HIP_CHECK(hipMalloc(&returnData, size));
  fillKernel<<<(elems + 127) / 128, 128>>>(srcData, elems);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // Allocate the destination buffer on the destination device.
  double *dstData;
  HIP_CHECK(hipSetDevice(dstDevice));
  HIP_CHECK(hipMalloc(&dstData, size));

  // Copy from the source device to the destination device.
  HIP_CHECK(hipMemcpyPeer(dstData, dstDevice, srcData, srcDevice, size));

  // Wait for the copy to finish before another stream reads dstData.
  HIP_CHECK(hipDeviceSynchronize());

  // Copy the data back to the source device in a stream.
  hipStream_t stream;
  HIP_CHECK(hipSetDevice(srcDevice));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMemcpyPeerAsync(returnData, srcDevice, dstData, dstDevice, size,
                               stream));

  // The asynchronous copy can still be running when hipMemcpyPeerAsync
  // returns. Wait for the stream before using the result.
  HIP_CHECK(hipStreamSynchronize(stream));

  // Check the data that made the round trip.
  std::vector<double> hostData(elems);
  HIP_CHECK(
      hipMemcpy(hostData.data(), returnData, size, hipMemcpyDeviceToHost));

  std::size_t errors = 0;
  for (std::size_t i = 0; i < elems; ++i) {
    if (hostData[i] != i * 2.0)
      ++errors;
  }

  // Free resources
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(returnData));
  HIP_CHECK(hipFree(srcData));
  HIP_CHECK(hipFree(dstData));

  if (errors != 0) {
    std::cout << "Validation failed: " << errors << " of " << elems
              << " elements are wrong." << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Copied " << elems << " elements from device " << srcDevice
            << " to device " << dstDevice << " and back. Validation passed."
            << std::endl;

  return EXIT_SUCCESS;
}
// [sphinx-end]
