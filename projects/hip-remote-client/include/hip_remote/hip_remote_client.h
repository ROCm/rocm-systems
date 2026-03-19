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
 * @file hip_remote_client.h
 * @brief HIP types and hip-remote-specific API extensions
 *
 * All standard HIP types and function declarations come from
 * <hip/hip_runtime_api.h> (provided by the ROCm SDK via ROCM_PATH).
 *
 * This header adds hip-remote-specific extensions only.
 * Internal hip-remote API (connection, messaging) is in hip_remote_internal.h.
 */

#ifndef HIP_REMOTE_CLIENT_H
#define HIP_REMOTE_CLIENT_H

#include "hip_remote/hip_remote_platform.h"
#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Hip-remote extensions (not in standard HIP API)
 * ============================================================================ */

/** Allocate N device buffers in a single round-trip. */
hipError_t hipMallocBatch(void** ptrs, const size_t* sizes, uint32_t count);

#ifdef __cplusplus
}
#endif

#endif /* HIP_REMOTE_CLIENT_H */
