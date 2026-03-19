/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file test_extended.c
 * @brief Extended tests for new remote HIP APIs
 *
 * Tests for: hipMallocAsync, hipFreeAsync, hipMemcpy2D,
 *            hipStreamGetFlags, hipStreamGetPriority, hipPointerGetAttributes
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hip_remote/hip_remote_client.h"

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n  %s:%d\n", msg, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

#define TEST_HIP_OK(call) do { \
    hipError_t _err = (call); \
    if (_err != hipSuccess) { \
        fprintf(stderr, "FAIL: %s returned %d (%s)\n  %s:%d\n", \
                #call, _err, hipGetErrorString(_err), __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

static int test_malloc_async(void) {
    printf("Test: hipMallocAsync / hipFreeAsync\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    /* Create a stream for async operations */
    void* stream = NULL;
    TEST_HIP_OK(hipStreamCreate(&stream));

    void* ptr = NULL;
    size_t size = 1024 * 1024;  /* 1MB */

    TEST_HIP_OK(hipMallocAsync(&ptr, size, stream));
    TEST_ASSERT(ptr != NULL, "Pointer should not be NULL");

    TEST_HIP_OK(hipFreeAsync(ptr, stream));

    /* Synchronize to ensure operations complete */
    TEST_HIP_OK(hipStreamSynchronize(stream));
    TEST_HIP_OK(hipStreamDestroy(stream));

    printf("  Async allocated and freed 1MB: OK\n");
    return 0;
}

static int test_stream_get_flags(void) {
    printf("Test: hipStreamGetFlags\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    /* Create stream with non-blocking flag */
    void* stream = NULL;
    unsigned int create_flags = 1;  /* hipStreamNonBlocking */
    TEST_HIP_OK(hipStreamCreateWithFlags(&stream, create_flags));
    TEST_ASSERT(stream != NULL, "Stream should not be NULL");

    /* Get flags back */
    unsigned int flags = 0;
    TEST_HIP_OK(hipStreamGetFlags(stream, &flags));
    printf("  Stream flags: 0x%x\n", flags);

    TEST_HIP_OK(hipStreamDestroy(stream));

    printf("  Stream get flags: OK\n");
    return 0;
}

static int test_stream_get_priority(void) {
    printf("Test: hipStreamGetPriority\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    /* Create stream with priority */
    void* stream = NULL;
    int create_priority = 0;  /* Normal priority */
    TEST_HIP_OK(hipStreamCreateWithPriority(&stream, 0, create_priority));
    TEST_ASSERT(stream != NULL, "Stream should not be NULL");

    /* Get priority back */
    int priority = -999;
    TEST_HIP_OK(hipStreamGetPriority(stream, &priority));
    printf("  Stream priority: %d\n", priority);

    TEST_HIP_OK(hipStreamDestroy(stream));

    printf("  Stream get priority: OK\n");
    return 0;
}

static int test_memcpy2d(void) {
    printf("Test: hipMemcpy2D\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    /* Create a 2D array: 64 rows x 128 columns of floats */
    size_t width = 128 * sizeof(float);  /* Width in bytes */
    size_t height = 64;                   /* Number of rows */
    size_t spitch = 256 * sizeof(float); /* Source pitch (with padding) */
    size_t dpitch = 256 * sizeof(float); /* Dest pitch */

    /* Allocate host memory with pitch */
    float* host_src = (float*)malloc(spitch * height);
    float* host_dst = (float*)malloc(dpitch * height);
    TEST_ASSERT(host_src != NULL, "Host source allocation failed");
    TEST_ASSERT(host_dst != NULL, "Host destination allocation failed");

    /* Initialize source - only the first 128 floats per row */
    for (size_t row = 0; row < height; row++) {
        float* row_ptr = (float*)((char*)host_src + row * spitch);
        for (size_t col = 0; col < 128; col++) {
            row_ptr[col] = (float)(row * 128 + col);
        }
    }
    memset(host_dst, 0, dpitch * height);

    /* Allocate device memory */
    void* device_ptr = NULL;
    TEST_HIP_OK(hipMalloc(&device_ptr, dpitch * height));

    /* Copy H2D using 2D copy */
    TEST_HIP_OK(hipMemcpy2D(device_ptr, dpitch, host_src, spitch, width, height,
                            hipMemcpyHostToDevice));

    /* Copy D2H */
    TEST_HIP_OK(hipMemcpy2D(host_dst, dpitch, device_ptr, dpitch, width, height,
                            hipMemcpyDeviceToHost));

    /* Verify */
    int match = 1;
    for (size_t row = 0; row < height && match; row++) {
        float* src_row = (float*)((char*)host_src + row * spitch);
        float* dst_row = (float*)((char*)host_dst + row * dpitch);
        for (size_t col = 0; col < 128; col++) {
            if (src_row[col] != dst_row[col]) {
                fprintf(stderr, "Mismatch at row %zu col %zu: %f != %f\n",
                        row, col, src_row[col], dst_row[col]);
                match = 0;
                break;
            }
        }
    }
    TEST_ASSERT(match, "Data mismatch after 2D round-trip");

    /* Cleanup */
    TEST_HIP_OK(hipFree(device_ptr));
    free(host_src);
    free(host_dst);

    printf("  2D memcpy round-trip 64x128 floats: OK\n");
    return 0;
}

static int test_pointer_get_attributes(void) {
    printf("Test: hipPointerGetAttributes\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    /* Allocate device memory */
    void* device_ptr = NULL;
    TEST_HIP_OK(hipMalloc(&device_ptr, 1024));
    TEST_ASSERT(device_ptr != NULL, "Device pointer should not be NULL");

    /* Get attributes */
    hipPointerAttribute_t attrs;
    memset(&attrs, 0, sizeof(attrs));
    TEST_HIP_OK(hipPointerGetAttributes(&attrs, device_ptr));

    printf("  Memory type: %d (expected: %d = hipMemoryTypeDevice)\n",
           (int)attrs.type, (int)hipMemoryTypeDevice);
    printf("  Device: %d\n", attrs.device);
    printf("  Device pointer: %p\n", attrs.devicePointer);
    printf("  Is managed: %d\n", attrs.isManaged);

    TEST_ASSERT(attrs.type == hipMemoryTypeDevice,
                "Memory type should be hipMemoryTypeDevice");
    TEST_ASSERT(attrs.device == 0, "Device should be 0");

    TEST_HIP_OK(hipFree(device_ptr));

    printf("  Pointer attributes: OK\n");
    return 0;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("=== Remote HIP Extended Tests ===\n\n");

    int failures = 0;

    /* Check if worker is configured */
    const char* host = getenv("TF_WORKER_HOST");
    if (!host || host[0] == '\0') {
        printf("NOTE: TF_WORKER_HOST not set, using localhost\n");
    }

    /* Run tests */
    failures += test_malloc_async();
    failures += test_stream_get_flags();
    failures += test_stream_get_priority();
    failures += test_memcpy2d();
    failures += test_pointer_get_attributes();

    printf("\n=== Results ===\n");
    if (failures == 0) {
        printf("All extended tests PASSED\n");
        return 0;
    } else {
        printf("%d extended test(s) FAILED\n", failures);
        return 1;
    }
}
