# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

from ..cynccl_m2n cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef ncclResult_t _ncclM2nInit(ncclM2nHandle_t* handle, const ncclM2nConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclM2nFinalize(ncclM2nHandle_t handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclM2nGroupStart() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclM2nGroupEnd() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclM2nGroupAbort() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef const char* _ncclM2nGetLastError() noexcept nogil
cdef ncclResult_t _ncclReshardWithWindow(ncclM2nHandle_t handle, ncclComm_t comm, ncclWindow_t window, const ncclDistTensor_t* src, const ncclDistTensor_t* dst, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclReshard(ncclM2nHandle_t handle, ncclComm_t comm, const ncclDistTensor_t* src, const ncclDistTensor_t* dst, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
