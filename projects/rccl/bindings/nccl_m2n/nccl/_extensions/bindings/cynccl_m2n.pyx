# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This code was automatically generated with version 0.2.0. Do not modify it directly.

from ._internal cimport nccl_m2n as _nccl_m2n


###############################################################################
# Wrapper functions
###############################################################################

cdef ncclResult_t ncclM2nInit(ncclM2nHandle_t* handle, const ncclM2nConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_m2n._ncclM2nInit(handle, config)


cdef ncclResult_t ncclM2nFinalize(ncclM2nHandle_t handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_m2n._ncclM2nFinalize(handle)


cdef ncclResult_t ncclM2nGroupStart() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_m2n._ncclM2nGroupStart()


cdef ncclResult_t ncclM2nGroupEnd() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_m2n._ncclM2nGroupEnd()


cdef ncclResult_t ncclM2nGroupAbort() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_m2n._ncclM2nGroupAbort()


cdef ncclResult_t ncclReshardWithWindow(ncclM2nHandle_t handle, ncclComm_t comm, ncclWindow_t window, const ncclDistTensor_t* src, const ncclDistTensor_t* dst, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_m2n._ncclReshardWithWindow(handle, comm, window, src, dst, stream)


cdef ncclResult_t ncclReshard(ncclM2nHandle_t handle, ncclComm_t comm, const ncclDistTensor_t* src, const ncclDistTensor_t* dst, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_m2n._ncclReshard(handle, comm, src, dst, stream)


cdef const char* ncclM2nGetLastError() noexcept nogil:
    return _nccl_m2n._ncclM2nGetLastError()
