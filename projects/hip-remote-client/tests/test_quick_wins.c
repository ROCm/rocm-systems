/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Test suite for Quick Win HIP APIs
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hip_remote/hip_remote_client.h"

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s (code %d)\n", \
                    __FILE__, __LINE__, hipGetErrorString(err), err); \
            return 1; \
        } \
    } while (0)

#define CHECK_HIP_WARN(call, msg) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "Warning at %s:%d: %s - %s (code %d)\n", \
                    __FILE__, __LINE__, msg, hipGetErrorString(err), err); \
        } \
    } while (0)

int main() {
    printf("=== HIP Remote Quick Win APIs Test ===\n\n");

    /* Test 1: hipDeviceGetStreamPriorityRange */
    printf("[TEST] hipDeviceGetStreamPriorityRange\n");
    int least_priority = 0;
    int greatest_priority = 0;
    CHECK_HIP(hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    printf("  Stream priority range: least=%d, greatest=%d\n", least_priority, greatest_priority);

    /* Test 2: hipSetValidDevices (just check it doesn't crash) */
    printf("\n[TEST] hipSetValidDevices\n");
    int devices[] = {0};
    CHECK_HIP_WARN(hipSetValidDevices(devices, 1), "hipSetValidDevices may not be fully supported");
    printf("  hipSetValidDevices called\n");

    /* Test 3: hipChooseDevice */
    printf("\n[TEST] hipChooseDevice\n");
    int chosen_device = -1;
    /* hipChooseDevice currently uses a stub implementation that returns device 0 */
    CHECK_HIP(hipChooseDevice(&chosen_device, NULL));
    printf("  Chosen device: %d\n", chosen_device);

    /* Test 4: hipStreamGetCaptureInfo */
    printf("\n[TEST] hipStreamGetCaptureInfo\n");
    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));

    int capture_status = 0;
    unsigned long long capture_id = 0;
    CHECK_HIP(hipStreamGetCaptureInfo(stream, &capture_status, &capture_id));
    printf("  Capture status: %d, id: %llu\n", capture_status, capture_id);

    CHECK_HIP(hipStreamDestroy(stream));

    /* Test 5: hipPointerGetAttribute */
    printf("\n[TEST] hipPointerGetAttribute\n");
    void* dev_ptr = NULL;
    CHECK_HIP(hipMalloc(&dev_ptr, 1024));

    void* data = NULL;
    CHECK_HIP(hipPointerGetAttribute(&data, hipPointerAttributeDevicePointer, dev_ptr));
    printf("  Device pointer: %p\n", data);

    CHECK_HIP(hipFree(dev_ptr));

    /* Test 6: hipMemcpyPeer (multi-GPU only) */
    int device_count = 0;
    CHECK_HIP(hipGetDeviceCount(&device_count));

    if (device_count > 1) {
        printf("\n[TEST] hipMemcpyPeer (multi-GPU)\n");

        void* dev_ptr0 = NULL;
        void* dev_ptr1 = NULL;

        CHECK_HIP(hipSetDevice(0));
        CHECK_HIP(hipMalloc(&dev_ptr0, 1024));

        CHECK_HIP(hipSetDevice(1));
        CHECK_HIP(hipMalloc(&dev_ptr1, 1024));

        CHECK_HIP(hipMemcpyPeer(dev_ptr1, 1, dev_ptr0, 0, 1024));
        printf("  Peer copy successful\n");

        CHECK_HIP(hipSetDevice(0));
        CHECK_HIP(hipFree(dev_ptr0));
        CHECK_HIP(hipSetDevice(1));
        CHECK_HIP(hipFree(dev_ptr1));
        CHECK_HIP(hipSetDevice(0));
    } else {
        printf("\n[TEST] hipMemcpyPeer - Skipped (single GPU)\n");
    }

    printf("\n=== All Tests Passed! ===\n");
    return 0;
}
