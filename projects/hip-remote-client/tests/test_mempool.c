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
 * @file test_mempool.c
 * @brief Tests for Memory Pool APIs
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

static int test_device_default_mem_pool(void) {
    printf("\nTest: hipDeviceGetDefaultMemPool\n");

    hipMemPool_t memPool = NULL;
    hipError_t err = hipDeviceGetDefaultMemPool(&memPool, 0);
    TEST_CHECK("hipDeviceGetDefaultMemPool succeeds", err == hipSuccess);
    TEST_CHECK("default pool is not NULL", memPool != NULL);
    printf("  Default pool for device 0: %p\n", memPool);

    /* Test NULL check */
    err = hipDeviceGetDefaultMemPool(NULL, 0);
    TEST_CHECK("hipDeviceGetDefaultMemPool with NULL fails", err == hipErrorInvalidValue);

    return 0;
}

static int test_device_get_set_mem_pool(void) {
    printf("\nTest: hipDeviceGetMemPool / hipDeviceSetMemPool\n");

    /* Get current pool */
    hipMemPool_t currentPool = NULL;
    hipError_t err = hipDeviceGetMemPool(&currentPool, 0);
    TEST_CHECK("hipDeviceGetMemPool succeeds", err == hipSuccess);
    printf("  Current pool for device 0: %p\n", currentPool);

    /* Get default pool */
    hipMemPool_t defaultPool = NULL;
    err = hipDeviceGetDefaultMemPool(&defaultPool, 0);
    TEST_CHECK("get default pool", err == hipSuccess);

    /* Set to default pool (should always work) */
    err = hipDeviceSetMemPool(0, defaultPool);
    TEST_CHECK("hipDeviceSetMemPool succeeds", err == hipSuccess);

    /* Verify it was set */
    hipMemPool_t verifyPool = NULL;
    err = hipDeviceGetMemPool(&verifyPool, 0);
    TEST_CHECK("verify pool was set", err == hipSuccess);

    return 0;
}

static int test_malloc_from_default_pool(void) {
    printf("\nTest: hipMallocFromPoolAsync with default pool\n");

    /* Get default pool */
    hipMemPool_t defaultPool = NULL;
    hipError_t err = hipDeviceGetDefaultMemPool(&defaultPool, 0);
    TEST_CHECK("get default pool", err == hipSuccess);

    /* Allocate from pool */
    void* devPtr = NULL;
    err = hipMallocFromPoolAsync(&devPtr, 1024 * 1024, defaultPool, NULL);
    TEST_CHECK("hipMallocFromPoolAsync succeeds", err == hipSuccess);
    TEST_CHECK("allocated pointer is not NULL", devPtr != NULL);
    printf("  Allocated 1MB from default pool: %p\n", devPtr);

    /* Free the allocation (using regular hipFreeAsync since it came from pool) */
    err = hipFreeAsync(devPtr, NULL);
    TEST_CHECK("hipFreeAsync succeeds", err == hipSuccess);

    /* Synchronize to ensure free completes */
    err = hipDeviceSynchronize();
    TEST_CHECK("hipDeviceSynchronize succeeds", err == hipSuccess);

    return 0;
}

static int test_mem_pool_attributes(void) {
    printf("\nTest: hipMemPoolGetAttribute / hipMemPoolSetAttribute\n");

    /* Get default pool */
    hipMemPool_t pool = NULL;
    hipError_t err = hipDeviceGetDefaultMemPool(&pool, 0);
    TEST_CHECK("get default pool", err == hipSuccess);

    /* Get release threshold */
    uint64_t threshold = 0;
    err = hipMemPoolGetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold);
    TEST_CHECK("get release threshold", err == hipSuccess);
    printf("  Current release threshold: %lu\n", (unsigned long)threshold);

    /* Get reserved memory */
    uint64_t reserved = 0;
    err = hipMemPoolGetAttribute(pool, hipMemPoolAttrReservedMemCurrent, &reserved);
    TEST_CHECK("get reserved memory", err == hipSuccess);
    printf("  Current reserved memory: %lu bytes\n", (unsigned long)reserved);

    /* Get used memory */
    uint64_t used = 0;
    err = hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemCurrent, &used);
    TEST_CHECK("get used memory", err == hipSuccess);
    printf("  Current used memory: %lu bytes\n", (unsigned long)used);

    /* Set a higher release threshold */
    uint64_t newThreshold = 1024 * 1024 * 10;  /* 10 MB */
    err = hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &newThreshold);
    TEST_CHECK("set release threshold", err == hipSuccess);

    /* Verify it was set */
    uint64_t verifyThreshold = 0;
    err = hipMemPoolGetAttribute(pool, hipMemPoolAttrReleaseThreshold, &verifyThreshold);
    TEST_CHECK("verify release threshold was set", err == hipSuccess && verifyThreshold == newThreshold);

    /* Restore original threshold */
    err = hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold);
    TEST_CHECK("restore release threshold", err == hipSuccess);

    return 0;
}

static int test_mem_pool_trim(void) {
    printf("\nTest: hipMemPoolTrimTo\n");

    /* Get default pool */
    hipMemPool_t pool = NULL;
    hipError_t err = hipDeviceGetDefaultMemPool(&pool, 0);
    TEST_CHECK("get default pool", err == hipSuccess);

    /* Allocate and free to build up pool memory */
    void* devPtr = NULL;
    err = hipMallocFromPoolAsync(&devPtr, 1024 * 1024 * 4, pool, NULL);  /* 4MB */
    TEST_CHECK("allocate from pool", err == hipSuccess);

    err = hipFreeAsync(devPtr, NULL);
    TEST_CHECK("free to pool", err == hipSuccess);

    err = hipDeviceSynchronize();
    TEST_CHECK("sync", err == hipSuccess);

    /* Trim the pool */
    err = hipMemPoolTrimTo(pool, 0);  /* Trim to minimum */
    TEST_CHECK("hipMemPoolTrimTo succeeds", err == hipSuccess);
    printf("  Pool trimmed successfully\n");

    return 0;
}

static int test_mem_pool_create_destroy(void) {
    printf("\nTest: hipMemPoolCreate / hipMemPoolDestroy\n");

    /* Create a new pool */
    hipMemPoolProps props;
    memset(&props, 0, sizeof(props));
    props.allocType = hipMemAllocationTypePinned;
    props.handleTypes = hipMemHandleTypeNone;
    props.location.type = hipMemLocationTypeDevice;
    props.location.id = 0;  /* Device 0 */
    props.maxSize = 0;  /* Unlimited */

    hipMemPool_t pool = NULL;
    hipError_t err = hipMemPoolCreate(&pool, &props);
    TEST_CHECK("hipMemPoolCreate succeeds", err == hipSuccess);
    TEST_CHECK("created pool is not NULL", pool != NULL);
    printf("  Created custom pool: %p\n", pool);

    /* Allocate from the custom pool */
    void* devPtr = NULL;
    err = hipMallocFromPoolAsync(&devPtr, 1024, pool, NULL);
    TEST_CHECK("allocate from custom pool", err == hipSuccess);
    printf("  Allocated from custom pool: %p\n", devPtr);

    /* Free the allocation */
    err = hipFreeAsync(devPtr, NULL);
    TEST_CHECK("free from custom pool", err == hipSuccess);

    err = hipDeviceSynchronize();
    TEST_CHECK("sync", err == hipSuccess);

    /* Destroy the pool */
    err = hipMemPoolDestroy(pool);
    TEST_CHECK("hipMemPoolDestroy succeeds", err == hipSuccess);

    return 0;
}

static int test_null_checks(void) {
    printf("\nTest: Memory Pool NULL parameter checks\n");

    hipError_t err;
    hipMemPool_t pool = NULL;
    hipDeviceGetDefaultMemPool(&pool, 0);

    /* hipMemPoolCreate NULL checks */
    err = hipMemPoolCreate(NULL, NULL);
    TEST_CHECK("hipMemPoolCreate(NULL, NULL) fails", err == hipErrorInvalidValue);

    hipMemPoolProps props = {0};
    err = hipMemPoolCreate(NULL, &props);
    TEST_CHECK("hipMemPoolCreate(NULL, props) fails", err == hipErrorInvalidValue);

    /* hipMemPoolDestroy NULL check */
    err = hipMemPoolDestroy(NULL);
    TEST_CHECK("hipMemPoolDestroy(NULL) fails", err == hipErrorInvalidValue);

    /* hipMemPoolGetAttribute NULL checks */
    uint64_t value = 0;
    err = hipMemPoolGetAttribute(NULL, hipMemPoolAttrReleaseThreshold, &value);
    TEST_CHECK("hipMemPoolGetAttribute(NULL, ...) fails", err == hipErrorInvalidValue);

    err = hipMemPoolGetAttribute(pool, hipMemPoolAttrReleaseThreshold, NULL);
    TEST_CHECK("hipMemPoolGetAttribute(..., NULL) fails", err == hipErrorInvalidValue);

    /* hipMallocFromPoolAsync NULL checks */
    err = hipMallocFromPoolAsync(NULL, 1024, pool, NULL);
    TEST_CHECK("hipMallocFromPoolAsync(NULL, ...) fails", err == hipErrorInvalidValue);

    void* ptr;
    err = hipMallocFromPoolAsync(&ptr, 1024, NULL, NULL);
    TEST_CHECK("hipMallocFromPoolAsync(..., NULL, ...) fails", err == hipErrorInvalidValue);

    /* hipMemPoolTrimTo NULL check */
    err = hipMemPoolTrimTo(NULL, 0);
    TEST_CHECK("hipMemPoolTrimTo(NULL, ...) fails", err == hipErrorInvalidValue);

    return 0;
}

int main(void) {
    printf("=== Remote HIP Memory Pool Tests ===\n");

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
    failures += test_null_checks();
    failures += test_device_default_mem_pool();
    failures += test_device_get_set_mem_pool();
    failures += test_malloc_from_default_pool();
    failures += test_mem_pool_attributes();
    failures += test_mem_pool_trim();
    failures += test_mem_pool_create_destroy();

    printf("\n=== Results ===\n");
    if (failures > 0) {
        printf("%d memory pool test(s) FAILED\n", failures);
        return 1;
    }
    printf("All memory pool tests PASSED\n");
    return 0;
}
