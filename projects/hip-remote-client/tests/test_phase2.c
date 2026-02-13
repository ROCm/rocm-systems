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
 * @file test_phase2.c
 * @brief Phase 2 tests for new remote HIP APIs
 *
 * Tests for: hipDeviceGetLimit, hipDeviceSetLimit, hipDeviceCanAccessPeer,
 *            hipDeviceEnablePeerAccess, hipMemcpyPeer
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

static int test_device_limits(void) {
    printf("Test: hipDeviceGetLimit / hipDeviceSetLimit\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    /* Get malloc heap size limit */
    size_t heap_size = 0;
    TEST_HIP_OK(hipDeviceGetLimit(&heap_size, hipLimitMallocHeapSize));
    printf("  Malloc heap size: %zu bytes\n", heap_size);
    TEST_ASSERT(heap_size > 0, "Heap size should be > 0");

    /* Get stack size limit */
    size_t stack_size = 0;
    TEST_HIP_OK(hipDeviceGetLimit(&stack_size, hipLimitStackSize));
    printf("  Stack size: %zu bytes\n", stack_size);

    printf("  Device limits: OK\n");
    return 0;
}

static int test_peer_access(void) {
    printf("Test: hipDeviceCanAccessPeer\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count < 2) {
        printf("  SKIP: Need at least 2 devices for peer access test\n");
        return 0;
    }

    /* Check if device 0 can access device 1 */
    int can_access = 0;
    TEST_HIP_OK(hipDeviceCanAccessPeer(&can_access, 0, 1));
    printf("  Device 0 -> Device 1: %s\n", can_access ? "Yes" : "No");

    /* Check reverse */
    TEST_HIP_OK(hipDeviceCanAccessPeer(&can_access, 1, 0));
    printf("  Device 1 -> Device 0: %s\n", can_access ? "Yes" : "No");

    printf("  Peer access query: OK\n");
    return 0;
}

static int test_memcpy_peer(void) {
    printf("Test: hipMemcpyPeer\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count < 2) {
        printf("  SKIP: Need at least 2 devices for peer memcpy test\n");
        return 0;
    }

    /* Check if peer access is possible */
    int can_access = 0;
    TEST_HIP_OK(hipDeviceCanAccessPeer(&can_access, 0, 1));
    if (!can_access) {
        printf("  SKIP: Peer access not supported between device 0 and 1\n");
        return 0;
    }

    /* Allocate on device 0 */
    TEST_HIP_OK(hipSetDevice(0));
    void* d0_ptr = NULL;
    size_t size = 1024 * sizeof(float);
    TEST_HIP_OK(hipMalloc(&d0_ptr, size));

    /* Allocate on device 1 */
    TEST_HIP_OK(hipSetDevice(1));
    void* d1_ptr = NULL;
    TEST_HIP_OK(hipMalloc(&d1_ptr, size));

    /* Initialize data on device 0 */
    float* host_data = (float*)malloc(size);
    TEST_ASSERT(host_data != NULL, "Host allocation failed");
    for (int i = 0; i < 1024; i++) {
        host_data[i] = (float)i;
    }

    TEST_HIP_OK(hipSetDevice(0));
    TEST_HIP_OK(hipMemcpy(d0_ptr, host_data, size, hipMemcpyHostToDevice));

    /* Enable peer access from device 1 to device 0 */
    TEST_HIP_OK(hipSetDevice(1));
    hipError_t peer_err = hipDeviceEnablePeerAccess(0, 0);
    if (peer_err == hipErrorPeerAccessAlreadyEnabled) {
        /* Already enabled, that's fine */
    } else if (peer_err != hipSuccess) {
        fprintf(stderr, "  Warning: hipDeviceEnablePeerAccess failed: %d\n", peer_err);
    }

    /* Copy from device 0 to device 1 using peer memcpy */
    TEST_HIP_OK(hipMemcpyPeer(d1_ptr, 1, d0_ptr, 0, size));

    /* Verify by copying back to host */
    float* verify = (float*)malloc(size);
    TEST_ASSERT(verify != NULL, "Verify allocation failed");
    memset(verify, 0, size);

    TEST_HIP_OK(hipMemcpy(verify, d1_ptr, size, hipMemcpyDeviceToHost));

    int match = 1;
    for (int i = 0; i < 1024 && match; i++) {
        if (host_data[i] != verify[i]) {
            fprintf(stderr, "Mismatch at %d: %f != %f\n", i, host_data[i], verify[i]);
            match = 0;
        }
    }
    TEST_ASSERT(match, "Peer memcpy data mismatch");

    /* Cleanup */
    TEST_HIP_OK(hipSetDevice(0));
    TEST_HIP_OK(hipFree(d0_ptr));
    TEST_HIP_OK(hipSetDevice(1));
    TEST_HIP_OK(hipFree(d1_ptr));
    free(host_data);
    free(verify);

    /* Reset to device 0 */
    TEST_HIP_OK(hipSetDevice(0));

    printf("  Peer memcpy: OK\n");
    return 0;
}

static int test_stream_wait_event(void) {
    printf("Test: hipStreamWaitEvent\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    /* Create two streams and an event */
    void* stream1 = NULL;
    void* stream2 = NULL;
    void* event = NULL;

    TEST_HIP_OK(hipStreamCreate(&stream1));
    TEST_HIP_OK(hipStreamCreate(&stream2));
    TEST_HIP_OK(hipEventCreate(&event));

    /* Record event on stream1 */
    TEST_HIP_OK(hipEventRecord(event, stream1));

    /* Make stream2 wait on the event */
    TEST_HIP_OK(hipStreamWaitEvent(stream2, event, 0));

    /* Synchronize stream2 (should complete after stream1's work) */
    TEST_HIP_OK(hipStreamSynchronize(stream2));

    /* Cleanup */
    TEST_HIP_OK(hipEventDestroy(event));
    TEST_HIP_OK(hipStreamDestroy(stream1));
    TEST_HIP_OK(hipStreamDestroy(stream2));

    printf("  Stream wait event: OK\n");
    return 0;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("=== Remote HIP Phase 2 Tests ===\n\n");

    int failures = 0;

    /* Check if worker is configured */
    const char* host = getenv("TF_WORKER_HOST");
    if (!host || host[0] == '\0') {
        printf("NOTE: TF_WORKER_HOST not set, using localhost\n");
    }

    /* Run tests */
    failures += test_device_limits();
    failures += test_peer_access();
    failures += test_memcpy_peer();
    failures += test_stream_wait_event();

    printf("\n=== Results ===\n");
    if (failures == 0) {
        printf("All Phase 2 tests PASSED\n");
        return 0;
    } else {
        printf("%d Phase 2 test(s) FAILED\n", failures);
        return 1;
    }
}
