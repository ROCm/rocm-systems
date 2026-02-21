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
 * @file hip_api_stream.c
 * @brief Stream and event API implementations for remote HIP
 */

#include "hip_remote/hip_remote_client.h"
#include "hip_remote/hip_remote_protocol.h"

/* ============================================================================
 * Pre-allocated Handle Pools
 *
 * Inspired by NX/NoMachine's pre-allocated X11 resource IDs: request a batch
 * of handles from the worker upfront so that subsequent create calls can
 * return immediately from the local pool without a network round-trip.
 * ============================================================================ */

#define EVENT_POOL_SIZE HIP_REMOTE_MAX_BATCH_HANDLES
static hipEvent_t g_event_pool[EVENT_POOL_SIZE];
static int g_event_pool_count = 0;
static hip_mutex_t g_event_pool_lock = HIP_MUTEX_INIT;

static hipError_t refill_event_pool(unsigned int flags) {
    HipRemoteHandleBatchRequest req = { .count = EVENT_POOL_SIZE, .flags = flags };
    HipRemoteHandleBatchResponse resp;
    memset(&resp, 0, sizeof(resp));

    hipError_t err = hip_remote_request(
        HIP_OP_EVENT_CREATE_BATCH,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess && resp.count > 0) {
        for (uint32_t i = 0; i < resp.count && g_event_pool_count < EVENT_POOL_SIZE; i++) {
            g_event_pool[g_event_pool_count++] = (hipEvent_t)(uintptr_t)resp.handles[i];
        }
    }
    return err;
}

/* ============================================================================
 * Stream Operations
 * ============================================================================
 * Types (hipStream_t, hipEvent_t) are defined in hip_remote_client.h
 */

hipError_t hipStreamCreate(hipStream_t* stream) {
    if (!stream) {
        return hipErrorInvalidValue;
    }

    HipRemoteStreamCreateRequest req = {
        .flags = 0,
        .priority = 0
    };
    HipRemoteStreamCreateResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_CREATE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *stream = (hipStream_t)(uintptr_t)resp.stream;
    } else {
        *stream = NULL;
    }
    return err;
}

hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
    if (!stream) {
        return hipErrorInvalidValue;
    }

    HipRemoteStreamCreateRequest req = {
        .flags = flags,
        .priority = 0
    };
    HipRemoteStreamCreateResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_CREATE_WITH_FLAGS,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *stream = (hipStream_t)(uintptr_t)resp.stream;
    } else {
        *stream = NULL;
    }
    return err;
}

hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags,
                                        int priority) {
    if (!stream) {
        return hipErrorInvalidValue;
    }

    HipRemoteStreamCreateRequest req = {
        .flags = flags,
        .priority = priority
    };
    HipRemoteStreamCreateResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_CREATE_WITH_PRIORITY,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *stream = (hipStream_t)(uintptr_t)resp.stream;
    } else {
        *stream = NULL;
    }
    return err;
}

hipError_t hipStreamDestroy(hipStream_t stream) {
    if (!stream) {
        return hipSuccess;
    }

    HipRemoteStreamRequest req = {
        .stream = (uint64_t)(uintptr_t)stream
    };

    return hip_remote_request_fire_and_forget(
        HIP_OP_STREAM_DESTROY, &req, sizeof(req)
    );
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
    HipRemoteStreamRequest req = {
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_STREAM_SYNCHRONIZE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipStreamQuery(hipStream_t stream) {
    HipRemoteStreamRequest req = {
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_STREAM_QUERY,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
    if (!flags) {
        return hipErrorInvalidValue;
    }

    HipRemoteStreamRequest req = {
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteStreamGetFlagsResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_GET_FLAGS,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *flags = resp.flags;
    }
    return err;
}

hipError_t hipStreamGetPriority(hipStream_t stream, int* priority) {
    if (!priority) {
        return hipErrorInvalidValue;
    }

    HipRemoteStreamRequest req = {
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteStreamGetPriorityResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_GET_PRIORITY,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *priority = resp.priority;
    }
    return err;
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                               unsigned int flags) {
    HipRemoteStreamWaitEventRequest req = {
        .stream = (uint64_t)(uintptr_t)stream,
        .event = (uint64_t)(uintptr_t)event,
        .flags = flags
    };

    return hip_remote_request_fire_and_forget(
        HIP_OP_STREAM_WAIT_EVENT, &req, sizeof(req)
    );
}

/* ============================================================================
 * Event Operations
 * ============================================================================ */

hipError_t hipEventCreate(hipEvent_t* event) {
    if (!event) {
        return hipErrorInvalidValue;
    }

    HipRemoteEventCreateRequest req = {
        .flags = 0
    };
    HipRemoteEventCreateResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_EVENT_CREATE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *event = (hipEvent_t)(uintptr_t)resp.event;
    } else {
        *event = NULL;
    }
    return err;
}

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags) {
    if (!event) {
        return hipErrorInvalidValue;
    }

    /* Only timing-disabled events use the pool. Timing-enabled events
     * (hipEventDefault) must be created individually because MIOpen's
     * profiling requires accurate per-event timing semantics. */
    if (flags == hipEventDisableTiming) {
        hip_mutex_lock(&g_event_pool_lock);
        if (g_event_pool_count == 0) {
            refill_event_pool(flags);
        }
        if (g_event_pool_count > 0) {
            *event = g_event_pool[--g_event_pool_count];
            hip_mutex_unlock(&g_event_pool_lock);
            return hipSuccess;
        }
        hip_mutex_unlock(&g_event_pool_lock);
    }

    HipRemoteEventCreateRequest req = {
        .flags = flags
    };
    HipRemoteEventCreateResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_EVENT_CREATE_WITH_FLAGS,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *event = (hipEvent_t)(uintptr_t)resp.event;
    } else {
        *event = NULL;
    }
    return err;
}

hipError_t hipEventDestroy(hipEvent_t event) {
    if (!event) {
        return hipErrorInvalidValue;
    }

    HipRemoteEventRequest req = {
        .event = (uint64_t)(uintptr_t)event
    };

    return hip_remote_request_fire_and_forget(
        HIP_OP_EVENT_DESTROY, &req, sizeof(req)
    );
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    if (!event) {
        return hipErrorInvalidValue;
    }

    HipRemoteEventRecordRequest req = {
        .event = (uint64_t)(uintptr_t)event,
        .stream = (uint64_t)(uintptr_t)stream
    };

    return hip_remote_request_fire_and_forget(
        HIP_OP_EVENT_RECORD, &req, sizeof(req)
    );
}

hipError_t hipEventSynchronize(hipEvent_t event) {
    if (!event) {
        return hipErrorInvalidValue;
    }

    HipRemoteEventRequest req = {
        .event = (uint64_t)(uintptr_t)event
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_EVENT_SYNCHRONIZE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipEventQuery(hipEvent_t event) {
    if (!event) {
        return hipErrorInvalidValue;
    }

    HipRemoteEventRequest req = {
        .event = (uint64_t)(uintptr_t)event
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_EVENT_QUERY,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
    hip_remote_log_debug("hipEventElapsedTime: ms=%p start=%p stop=%p", (void*)ms, (void*)start, (void*)stop);

    if (!ms || !start || !stop) {
        hip_remote_log_error("hipEventElapsedTime: NULL arg ms=%p start=%p stop=%p",
                             (void*)ms, (void*)start, (void*)stop);
        return hipErrorInvalidValue;
    }

    HipRemoteEventElapsedTimeRequest req = {
        .start_event = (uint64_t)(uintptr_t)start,
        .end_event = (uint64_t)(uintptr_t)stop
    };
    HipRemoteEventElapsedTimeResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_EVENT_ELAPSED_TIME,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    hip_remote_log_debug("hipEventElapsedTime: err=%d ms=%.4f", err, err == hipSuccess ? resp.milliseconds : 0.0f);

    if (err == hipSuccess) {
        *ms = resp.milliseconds;
    }
    return err;
}

/* ============================================================================
 * Graph Operations
 * ============================================================================
 * Types (hipGraph_t, hipGraphExec_t, hipGraphNode_t, hipStreamCaptureStatus,
 *        hipStreamCaptureMode, hipStreamCallback_t) are defined in hip_remote_client.h
 */

hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags) {
    if (!pGraph) {
        return hipErrorInvalidValue;
    }

    HipRemoteGraphCreateRequest req = {
        .flags = flags
    };
    HipRemoteGraphCreateResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_CREATE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *pGraph = (hipGraph_t)(uintptr_t)resp.graph;
    } else {
        *pGraph = NULL;
    }
    return err;
}

hipError_t hipGraphDestroy(hipGraph_t graph) {
    if (!graph) {
        return hipErrorInvalidValue;
    }

    HipRemoteGraphDestroyRequest req = {
        .graph = (uint64_t)(uintptr_t)graph
    };

    return hip_remote_request_fire_and_forget(
        HIP_OP_GRAPH_DESTROY, &req, sizeof(req)
    );
}

hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                hipGraphNode_t* pErrorNode, char* pLogBuffer,
                                size_t bufferSize) {
    if (!pGraphExec || !graph) {
        return hipErrorInvalidValue;
    }

    if (pErrorNode) *pErrorNode = NULL;
    if (pLogBuffer && bufferSize > 0) pLogBuffer[0] = '\0';

    HipRemoteGraphInstantiateRequest req = {
        .graph = (uint64_t)(uintptr_t)graph,
        .flags = 0
    };
    HipRemoteGraphInstantiateResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_INSTANTIATE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *pGraphExec = (hipGraphExec_t)(uintptr_t)resp.graph_exec;
    } else {
        *pGraphExec = NULL;
    }
    return err;
}

hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
    if (!graphExec) {
        return hipErrorInvalidValue;
    }

    HipRemoteGraphLaunchRequest req = {
        .graph_exec = (uint64_t)(uintptr_t)graphExec,
        .stream = (uint64_t)(uintptr_t)stream
    };
    return hip_remote_request_fire_and_forget(
        HIP_OP_GRAPH_LAUNCH, &req, sizeof(req)
    );
}

hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) {
    if (!graphExec) {
        return hipErrorInvalidValue;
    }

    HipRemoteGraphExecDestroyRequest req = {
        .graph_exec = (uint64_t)(uintptr_t)graphExec
    };

    return hip_remote_request_fire_and_forget(
        HIP_OP_GRAPH_EXEC_DESTROY, &req, sizeof(req)
    );
}

hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
    HipRemoteStreamBeginCaptureRequest req = {
        .stream = (uint64_t)(uintptr_t)stream,
        .mode = mode
    };

    return hip_remote_request_fire_and_forget(
        HIP_OP_STREAM_BEGIN_CAPTURE, &req, sizeof(req)
    );
}
hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
    if (!pGraph) {
        return hipErrorInvalidValue;
    }

    HipRemoteStreamEndCaptureRequest req = {
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteStreamEndCaptureResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_END_CAPTURE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *pGraph = (hipGraph_t)(uintptr_t)resp.graph;
    } else {
        *pGraph = NULL;
    }
    return err;
}

hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
    if (!pCaptureStatus) {
        return hipErrorInvalidValue;
    }

    /* For the default stream (NULL), we know it's not capturing */
    if (!stream) {
        *pCaptureStatus = 0; /* hipStreamCaptureStatusNone */
        return hipSuccess;
    }

    HipRemoteStreamIsCapturingRequest req;
    memset(&req, 0, sizeof(req));
    req.stream = (uint64_t)(uintptr_t)stream;
    HipRemoteStreamIsCapturingResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_IS_CAPTURING,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *pCaptureStatus = resp.capture_status;
    } else {
        /* Treat capture query errors as "not capturing" */
        *pCaptureStatus = 0;
        err = hipSuccess;
    }
    return err;
}

hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback,
                                 void* userData, unsigned int flags) {
    /*
     * Stream callbacks cannot be fully supported in remote mode because
     * the callback function pointer is meaningless on the remote worker.
     *
     * For now, we return hipErrorNotSupported. Applications should use
     * events (hipEventRecord + hipEventSynchronize) for synchronization.
     */
    (void)stream;
    (void)callback;
    (void)userData;
    (void)flags;

    hip_remote_log_error("hipStreamAddCallback: not supported in remote mode");
    hip_remote_log_error("Use hipEventRecord + hipEventSynchronize instead");
    return hipErrorNotSupported;
}

/* ============================================================================
 * Additional Stream/Graph Stubs
 * ============================================================================ */

hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream, uint32_t cuMaskSize, const uint32_t* cuMask) {
    hip_remote_log_debug("hipExtStreamCreateWithCUMask: cuMaskSize=%u (CU mask not forwarded, creating default stream)",
                         cuMaskSize);
    return hipStreamCreate(stream);
}

hipError_t hipGetStreamDeviceId(hipStream_t stream) {
    hip_remote_log_debug("hipGetStreamDeviceId: stream=%p (returning device 0)", (void*)stream);
    return 0;
}

hipError_t hipStreamGetCaptureInfo(hipStream_t stream, int* captureStatus, unsigned long long* id) {
    hip_remote_log_debug("hipStreamGetCaptureInfo: stream=%p (not supported remotely)", (void*)stream);
    if (captureStatus) *captureStatus = 0;
    if (id) *id = 0;
    return hipSuccess;
}

hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, int* captureStatus, unsigned long long* id, void** graph, const void** dependencies, size_t* numDependencies) {
    hip_remote_log_debug("hipStreamGetCaptureInfo_v2: stream=%p (not supported remotely)", (void*)stream);
    if (captureStatus) *captureStatus = 0;
    if (id) *id = 0;
    if (graph) *graph = NULL;
    if (dependencies) *dependencies = NULL;
    if (numDependencies) *numDependencies = 0;
    return hipSuccess;
}

hipError_t hipThreadExchangeStreamCaptureMode(int* mode) {
    hip_remote_log_debug("hipThreadExchangeStreamCaptureMode: mode=%p (not supported remotely)", (void*)mode);
    return hipSuccess;
}

hipError_t hipGraphInstantiateWithFlags(void** pGraphExec, void* graph, unsigned long long flags) {
    return hipGraphInstantiate(pGraphExec, graph, NULL, NULL, (unsigned int)flags);
}

hipError_t hipGraphGetNodes(void* graph, void** nodes, size_t* numNodes) {
    hip_remote_log_debug("hipGraphGetNodes: graph=%p (not supported remotely)", graph);
    return hipErrorNotSupported;
}

hipError_t hipGraphNodeGetDependencies(void* node, void** dependencies, size_t* numDependencies) {
    hip_remote_log_debug("hipGraphNodeGetDependencies: node=%p (not supported remotely)", node);
    return hipErrorNotSupported;
}

hipError_t hipGraphDebugDotPrint(void* graph, const char* path, unsigned int flags) {
    hip_remote_log_debug("hipGraphDebugDotPrint: graph=%p path=%s (not supported remotely)",
                         graph, path ? path : "(null)");
    return hipErrorNotSupported;
}
