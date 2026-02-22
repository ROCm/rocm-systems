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

#include <stdlib.h>

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
        return hipSuccess;  /* NULL stream is default stream, don't destroy */
    }

    HipRemoteStreamRequest req = {
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_STREAM_DESTROY,
        &req, sizeof(req),
        &resp, sizeof(resp)
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
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_STREAM_WAIT_EVENT,
        &req, sizeof(req),
        &resp, sizeof(resp)
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
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_EVENT_DESTROY,
        &req, sizeof(req),
        &resp, sizeof(resp)
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
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_EVENT_RECORD,
        &req, sizeof(req),
        &resp, sizeof(resp)
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
    if (!ms || !start || !stop) {
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
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_GRAPH_DESTROY,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                hipGraphNode_t* pErrorNode, char* pLogBuffer,
                                size_t bufferSize) {
    if (!pGraphExec || !graph) {
        return hipErrorInvalidValue;
    }

    /* Note: pErrorNode and pLogBuffer are not supported in remote mode */
    (void)pErrorNode;
    (void)pLogBuffer;
    (void)bufferSize;

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
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_GRAPH_LAUNCH,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) {
    if (!graphExec) {
        return hipErrorInvalidValue;
    }

    HipRemoteGraphExecDestroyRequest req = {
        .graph_exec = (uint64_t)(uintptr_t)graphExec
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_GRAPH_EXEC_DESTROY,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
    HipRemoteStreamBeginCaptureRequest req = {
        .stream = (uint64_t)(uintptr_t)stream,
        .mode = mode
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_STREAM_BEGIN_CAPTURE,
        &req, sizeof(req),
        &resp, sizeof(resp)
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

    HipRemoteStreamIsCapturingRequest req = {
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteStreamIsCapturingResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_IS_CAPTURING,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *pCaptureStatus = resp.capture_status;
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

hipError_t hipStreamGetCaptureInfo(void* stream, int* captureStatus, unsigned long long* id) {
    if (!captureStatus) {
        return hipErrorInvalidValue;
    }

    HipRemoteStreamGetCaptureInfoRequest req = {
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteStreamGetCaptureInfoResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_GET_CAPTURE_INFO,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *captureStatus = resp.capture_status;
        if (id) {
            *id = resp.graph;
        }
    }
    return err;
}

hipError_t hipStreamUpdateCaptureDependencies(void* stream, void** dependencies,
                                                size_t numDependencies, unsigned int flags) {
    if (!dependencies && numDependencies > 0) {
        return hipErrorInvalidValue;
    }

    /* Allocate variable-length request buffer */
    size_t req_size = sizeof(HipRemoteStreamUpdateCaptureDependenciesRequest) +
                      numDependencies * sizeof(uint64_t);
    HipRemoteStreamUpdateCaptureDependenciesRequest* req =
        (HipRemoteStreamUpdateCaptureDependenciesRequest*)malloc(req_size);
    if (!req) {
        return hipErrorOutOfMemory;
    }

    req->stream = (uint64_t)(uintptr_t)stream;
    req->num_dependencies = (uint32_t)numDependencies;
    req->flags = flags;

    /* Copy dependency node handles */
    uint64_t* nodes = (uint64_t*)(req + 1);
    for (size_t i = 0; i < numDependencies; i++) {
        nodes[i] = (uint64_t)(uintptr_t)dependencies[i];
    }

    HipRemoteResponseHeader resp;
    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_UPDATE_CAPTURE_DEPENDENCIES,
        req, req_size,
        &resp, sizeof(resp)
    );

    free(req);
    return err;
}

/* ============================================================================
 * Extended Stream Operations (New APIs)
 * ============================================================================ */

hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device) {
    if (!device) {
        return hipErrorInvalidValue;
    }

    HipRemoteStreamHandleRequest req = {
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteStreamGetDeviceResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_GET_DEVICE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *device = resp.device;
    }
    return err;
}

hipError_t hipStreamGetId(hipStream_t stream, unsigned long long* streamId) {
    if (!streamId) {
        return hipErrorInvalidValue;
    }

    HipRemoteStreamHandleRequest req = {
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteStreamGetIdResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_STREAM_GET_ID,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *streamId = resp.streamId;
    }
    return err;
}

hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value,
                                 unsigned int flags, uint32_t mask) {
    HipRemoteStreamWaitValue32Request req = {
        .stream = (uint64_t)(uintptr_t)stream,
        .ptr = (uint64_t)(uintptr_t)ptr,
        .value = value,
        .flags = flags,
        .mask = mask
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_STREAM_WAIT_VALUE_32,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value,
                                 unsigned int flags, uint64_t mask) {
    HipRemoteStreamWaitValue64Request req = {
        .stream = (uint64_t)(uintptr_t)stream,
        .ptr = (uint64_t)(uintptr_t)ptr,
        .value = value,
        .flags = flags,
        .mask = mask
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_STREAM_WAIT_VALUE_64,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, uint32_t value,
                                  unsigned int flags) {
    HipRemoteStreamWriteValue32Request req = {
        .stream = (uint64_t)(uintptr_t)stream,
        .ptr = (uint64_t)(uintptr_t)ptr,
        .value = value,
        .flags = flags
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_STREAM_WRITE_VALUE_32,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value,
                                  unsigned int flags) {
    HipRemoteStreamWriteValue64Request req = {
        .stream = (uint64_t)(uintptr_t)stream,
        .ptr = (uint64_t)(uintptr_t)ptr,
        .value = value,
        .flags = flags
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_STREAM_WRITE_VALUE_64,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipStreamAttachMemAsync(hipStream_t stream, void* dev_ptr, size_t length,
                                    unsigned int flags) {
    HipRemoteStreamAttachMemAsyncRequest req = {
        .stream = (uint64_t)(uintptr_t)stream,
        .dev_ptr = (uint64_t)(uintptr_t)dev_ptr,
        .length = length,
        .flags = flags
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_STREAM_ATTACH_MEM_ASYNC,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

/* Stubs for complex APIs that need more work */
hipError_t hipStreamSetAttribute(hipStream_t stream, hipStreamAttrID attr,
                                  const hipStreamAttrValue* value) {
    /* TODO: Implement stream attribute setting */
    (void)stream;
    (void)attr;
    (void)value;
    return hipErrorNotSupported;
}

hipError_t hipStreamGetAttribute(hipStream_t stream, hipStreamAttrID attr,
                                  hipStreamAttrValue* value_out) {
    /* TODO: Implement stream attribute getting */
    (void)stream;
    (void)attr;
    (void)value_out;
    return hipErrorNotSupported;
}

hipError_t hipStreamCopyAttributes(hipStream_t dst, hipStream_t src) {
    /* TODO: Implement stream attribute copying */
    (void)dst;
    (void)src;
    return hipErrorNotSupported;
}

hipError_t hipStreamBeginCaptureToGraph(hipStream_t stream, hipGraph_t graph,
                                         const hipGraphNode_t* dependencies,
                                         const hipGraphEdgeData* dependencyData,
                                         size_t numDependencies, hipStreamCaptureMode mode) {
    /* TODO: Implement stream capture to existing graph */
    (void)stream;
    (void)graph;
    (void)dependencies;
    (void)dependencyData;
    (void)numDependencies;
    (void)mode;
    return hipErrorNotSupported;
}

hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream,
                                       hipStreamCaptureStatus* captureStatus_out,
                                       unsigned long long* id_out,
                                       hipGraph_t* graph_out,
                                       const hipGraphNode_t** dependencies_out,
                                       size_t* numDependencies_out) {
    /* TODO: Implement v2 capture info */
    (void)stream;
    (void)captureStatus_out;
    (void)id_out;
    (void)graph_out;
    (void)dependencies_out;
    (void)numDependencies_out;
    return hipErrorNotSupported;
}

hipError_t hipStreamBatchMemOp(hipStream_t stream, unsigned int count,
                                hipStreamBatchMemOpParams* paramArray, unsigned int flags) {
    /* TODO: Implement batch memory operations */
    (void)stream;
    (void)count;
    (void)paramArray;
    (void)flags;
    return hipErrorNotSupported;
}
