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
 * @file test_graphs.c
 * @brief Tests for HIP Graph APIs
 *
 * Tests for: hipGraphCreate, hipGraphDestroy, hipGraphInstantiate,
 *            hipGraphLaunch, hipStreamBeginCapture, hipStreamEndCapture
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

static int test_graph_create_destroy(void) {
    printf("Test: hipGraphCreate / hipGraphDestroy\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    /* Create an empty graph */
    hipGraph_t graph = NULL;
    TEST_HIP_OK(hipGraphCreate(&graph, 0));
    TEST_ASSERT(graph != NULL, "Graph should not be NULL");

    /* Destroy the graph */
    TEST_HIP_OK(hipGraphDestroy(graph));

    printf("  Graph create/destroy: OK\n");
    return 0;
}

static int test_stream_capture_status(void) {
    printf("Test: hipStreamIsCapturing\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    /* Create a stream */
    hipStream_t stream = NULL;
    TEST_HIP_OK(hipStreamCreate(&stream));

    /* Check capture status - should be none */
    hipStreamCaptureStatus status = (hipStreamCaptureStatus)99;
    TEST_HIP_OK(hipStreamIsCapturing(stream, &status));
    TEST_ASSERT(status == hipStreamCaptureStatusNone,
                "Stream should not be capturing initially");
    printf("  Initial capture status: %d (None)\n", status);

    TEST_HIP_OK(hipStreamDestroy(stream));

    printf("  Stream capture status query: OK\n");
    return 0;
}

static int test_stream_capture_basic(void) {
    printf("Test: hipStreamBeginCapture / hipStreamEndCapture\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    /* Create a non-blocking stream for capture */
    hipStream_t stream = NULL;
    TEST_HIP_OK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    /* Begin capture with relaxed mode to avoid affecting global state */
    TEST_HIP_OK(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));

    /* Check status - should be active */
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    TEST_HIP_OK(hipStreamIsCapturing(stream, &status));
    TEST_ASSERT(status == hipStreamCaptureStatusActive,
                "Stream should be capturing");
    printf("  Capture status after begin: %d (Active)\n", status);

    /*
     * Note: hipMalloc/hipFree are NOT captured because they are sync operations.
     * During capture, we should use async operations like hipMemsetAsync,
     * hipMemcpyAsync, etc. For this basic test, we capture an empty graph.
     */

    /* End capture - creates an empty graph */
    hipGraph_t graph = NULL;
    TEST_HIP_OK(hipStreamEndCapture(stream, &graph));
    TEST_ASSERT(graph != NULL, "Captured graph should not be NULL");

    /* Status should be none again */
    TEST_HIP_OK(hipStreamIsCapturing(stream, &status));
    TEST_ASSERT(status == hipStreamCaptureStatusNone,
                "Stream should not be capturing after end");

    /* Cleanup */
    TEST_HIP_OK(hipGraphDestroy(graph));
    TEST_HIP_OK(hipStreamDestroy(stream));

    printf("  Stream capture basic: OK\n");
    return 0;
}

static int test_graph_instantiate_launch(void) {
    printf("Test: hipGraphInstantiate / hipGraphLaunch\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    /* Create an empty graph directly (no stream capture) */
    hipGraph_t graph = NULL;
    TEST_HIP_OK(hipGraphCreate(&graph, 0));
    TEST_ASSERT(graph != NULL, "Graph should not be NULL");

    /* Instantiate the empty graph */
    hipGraphExec_t graphExec = NULL;
    TEST_HIP_OK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    TEST_ASSERT(graphExec != NULL, "GraphExec should not be NULL");

    /* Create a stream for launch */
    hipStream_t launch_stream = NULL;
    TEST_HIP_OK(hipStreamCreate(&launch_stream));

    /* Launch the empty graph (should be a no-op) */
    TEST_HIP_OK(hipGraphLaunch(graphExec, launch_stream));
    TEST_HIP_OK(hipStreamSynchronize(launch_stream));

    /* Cleanup */
    TEST_HIP_OK(hipGraphExecDestroy(graphExec));
    TEST_HIP_OK(hipGraphDestroy(graph));
    TEST_HIP_OK(hipStreamDestroy(launch_stream));

    printf("  Graph instantiate/launch: OK\n");
    return 0;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("=== Remote HIP Graph Tests ===\n\n");

    int failures = 0;

    /* Check if worker is configured */
    const char* host = getenv("TF_WORKER_HOST");
    if (!host || host[0] == '\0') {
        printf("NOTE: TF_WORKER_HOST not set, using localhost\n");
    }

    /* Run tests */
    failures += test_graph_create_destroy();
    failures += test_stream_capture_status();
    failures += test_stream_capture_basic();
    failures += test_graph_instantiate_launch();

    printf("\n=== Results ===\n");
    if (failures == 0) {
        printf("All graph tests PASSED\n");
        return 0;
    } else {
        printf("%d graph test(s) FAILED\n", failures);
        return 1;
    }
}
