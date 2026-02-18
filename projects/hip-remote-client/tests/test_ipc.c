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
 * @file test_ipc.c
 * @brief Tests for IPC (Inter-Process Communication) APIs
 *
 * Note: Full IPC testing requires multiple processes. These tests verify
 * the basic API functionality within a single process, which is useful for
 * testing the remote protocol implementation even though the IPC handles
 * cannot actually be shared across processes in this test.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hip_remote/hip_remote_client.h"

#define TEST_CHECK(name, condition) do { \
    if (!(condition)) { \
        printf("FAIL: %s\n", name); \
        printf("  %s:%d\n", __FILE__, __LINE__); \
        return 1; \
    } \
    printf("  PASS: %s\n", name); \
} while (0)

static int test_ipc_get_mem_handle(void) {
    printf("\nTest: hipIpcGetMemHandle\n");

    /* Allocate some device memory */
    void* devPtr = NULL;
    hipError_t err = hipMalloc(&devPtr, 1024 * 1024);  /* 1MB */
    TEST_CHECK("hipMalloc for IPC test", err == hipSuccess && devPtr != NULL);

    /* Get IPC handle for the allocation */
    hipIpcMemHandle_t handle;
    memset(&handle, 0, sizeof(handle));
    err = hipIpcGetMemHandle(&handle, devPtr);
    TEST_CHECK("hipIpcGetMemHandle succeeds", err == hipSuccess);

    /* Verify handle is not all zeros (indicates it was filled) */
    int has_nonzero = 0;
    for (size_t i = 0; i < sizeof(handle.reserved); i++) {
        if (handle.reserved[i] != 0) {
            has_nonzero = 1;
            break;
        }
    }
    TEST_CHECK("IPC handle is populated", has_nonzero);

    /* Test error case: NULL pointer */
    err = hipIpcGetMemHandle(&handle, NULL);
    TEST_CHECK("hipIpcGetMemHandle with NULL ptr fails",
               err == hipErrorInvalidDevicePointer || err == hipErrorInvalidValue);

    /* Test error case: NULL handle */
    err = hipIpcGetMemHandle(NULL, devPtr);
    TEST_CHECK("hipIpcGetMemHandle with NULL handle fails", err == hipErrorInvalidValue);

    hipFree(devPtr);
    return 0;
}

static int test_ipc_open_close_mem_handle(void) {
    printf("\nTest: hipIpcOpenMemHandle / hipIpcCloseMemHandle\n");

    /* Allocate device memory */
    void* devPtr = NULL;
    hipError_t err = hipMalloc(&devPtr, 1024 * 1024);
    TEST_CHECK("hipMalloc for IPC test", err == hipSuccess);

    /* Get IPC handle */
    hipIpcMemHandle_t handle;
    err = hipIpcGetMemHandle(&handle, devPtr);
    TEST_CHECK("hipIpcGetMemHandle succeeds", err == hipSuccess);

    /* Open the IPC handle (in the same process - limited test) */
    void* openedPtr = NULL;
    err = hipIpcOpenMemHandle(&openedPtr, handle, hipIpcMemLazyEnablePeerAccess);
    /* Note: This may fail with hipErrorInvalidContext or hipErrorMapFailed
     * when called in the same process that created the allocation.
     * We accept success or specific expected errors. */
    if (err == hipSuccess) {
        TEST_CHECK("hipIpcOpenMemHandle returned valid pointer", openedPtr != NULL);
        printf("  IPC open succeeded (same-process): opened=%p, original=%p\n",
               openedPtr, devPtr);

        /* Close the opened handle */
        err = hipIpcCloseMemHandle(openedPtr);
        TEST_CHECK("hipIpcCloseMemHandle succeeds", err == hipSuccess);
    } else {
        /* Expected failure in same-process scenario */
        printf("  IPC open returned %d (expected in same-process scenario)\n", err);
        TEST_CHECK("hipIpcOpenMemHandle returns expected error",
                   err == hipErrorInvalidContext ||
                   err == hipErrorMapFailed ||
                   err == hipErrorAlreadyMapped ||
                   err == hipErrorPeerAccessAlreadyEnabled);
    }

    hipFree(devPtr);
    return 0;
}

static int test_ipc_event_handles(void) {
    printf("\nTest: hipIpcGetEventHandle / hipIpcOpenEventHandle\n");

    /* Create an event with interprocess flag */
    hipEvent_t event = NULL;
    hipError_t err = hipEventCreateWithFlags(&event, hipEventInterprocess | hipEventDisableTiming);
    TEST_CHECK("hipEventCreateWithFlags with interprocess", err == hipSuccess);

    /* Get IPC handle for the event */
    hipIpcEventHandle_t handle;
    memset(&handle, 0, sizeof(handle));
    err = hipIpcGetEventHandle(&handle, event);
    TEST_CHECK("hipIpcGetEventHandle succeeds", err == hipSuccess);

    /* Verify handle is populated */
    int has_nonzero = 0;
    for (size_t i = 0; i < sizeof(handle.reserved); i++) {
        if (handle.reserved[i] != 0) {
            has_nonzero = 1;
            break;
        }
    }
    TEST_CHECK("IPC event handle is populated", has_nonzero);

    /* Open the IPC event handle (in same process - limited test) */
    hipEvent_t openedEvent = NULL;
    err = hipIpcOpenEventHandle(&openedEvent, handle);
    if (err == hipSuccess) {
        TEST_CHECK("hipIpcOpenEventHandle returned valid event", openedEvent != NULL);
        printf("  IPC event open succeeded: opened=%p, original=%p\n",
               openedEvent, event);

        /* We don't destroy the opened event - it's the same underlying event */
    } else {
        /* May fail in same-process scenario */
        printf("  IPC event open returned %d (may be expected in same-process)\n", err);
    }

    /* Test error case: NULL event */
    err = hipIpcGetEventHandle(&handle, NULL);
    TEST_CHECK("hipIpcGetEventHandle with NULL event fails",
               err == hipErrorInvalidResourceHandle || err == hipErrorInvalidValue);

    hipEventDestroy(event);
    return 0;
}

static int test_ipc_null_checks(void) {
    printf("\nTest: IPC NULL parameter checks\n");

    hipError_t err;

    /* hipIpcGetMemHandle NULL checks */
    void* devPtr = (void*)0x12345678;  /* Fake pointer for NULL check test */
    err = hipIpcGetMemHandle(NULL, devPtr);
    TEST_CHECK("hipIpcGetMemHandle(NULL, ptr) fails", err == hipErrorInvalidValue);

    /* hipIpcOpenMemHandle NULL checks */
    hipIpcMemHandle_t memHandle;
    memset(&memHandle, 0, sizeof(memHandle));
    err = hipIpcOpenMemHandle(NULL, memHandle, 0);
    TEST_CHECK("hipIpcOpenMemHandle(NULL, ...) fails", err == hipErrorInvalidValue);

    /* hipIpcCloseMemHandle NULL check */
    err = hipIpcCloseMemHandle(NULL);
    TEST_CHECK("hipIpcCloseMemHandle(NULL) fails", err == hipErrorInvalidValue);

    /* hipIpcGetEventHandle NULL checks */
    err = hipIpcGetEventHandle(NULL, (hipEvent_t)0x12345678);
    TEST_CHECK("hipIpcGetEventHandle(NULL, event) fails", err == hipErrorInvalidValue);

    /* hipIpcOpenEventHandle NULL check */
    hipIpcEventHandle_t eventHandle;
    memset(&eventHandle, 0, sizeof(eventHandle));
    err = hipIpcOpenEventHandle(NULL, eventHandle);
    TEST_CHECK("hipIpcOpenEventHandle(NULL, ...) fails", err == hipErrorInvalidValue);

    return 0;
}

int main(void) {
    printf("=== Remote HIP IPC Tests ===\n");

    const char* host = getenv("TF_WORKER_HOST");
    if (!host || strlen(host) == 0) {
        printf("\nNOTE: TF_WORKER_HOST not set, using localhost\n");
    }

    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count == 0) {
        printf("SKIP: No GPUs available (err=%d, count=%d)\n", err, count);
        return 77;  /* Skip exit code */
    }
    printf("Found %d GPU(s)\n", count);

    int failures = 0;
    failures += test_ipc_null_checks();
    failures += test_ipc_get_mem_handle();
    failures += test_ipc_open_close_mem_handle();
    failures += test_ipc_event_handles();

    printf("\n=== Results ===\n");
    if (failures > 0) {
        printf("%d IPC test(s) FAILED\n", failures);
        return 1;
    }
    printf("All IPC tests PASSED\n");
    return 0;
}
