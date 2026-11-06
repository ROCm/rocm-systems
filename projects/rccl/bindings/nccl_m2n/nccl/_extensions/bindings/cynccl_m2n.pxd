# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This code was automatically generated with version 0.2.0. Do not modify it directly.

from nccl.bindings.cynccl cimport ncclResult_t, ncclDataType_t, ncclComm_t, ncclWindow_t, _NCCLRESULT_T_INTERNAL_LOADING_ERROR



###############################################################################
# Types and constants
###############################################################################

# The generated declarations encode the M2N ABI, including these public
# constants, so wheel builds do not require nccl_m2n.h at build time.
cdef extern from *:
    """
    #include <limits.h>
    enum {
      NCCL_RESHARD_MESH_NDIMS = 2,
      NCCL_RESHARD_MAX_MESH_DIMS = 2,
      NCCL_RESHARD_MAX_TENSOR_DIMS = 3,
      NCCL_RESHARD_REPLICATE = -1,
      NCCL_M2N_CONFIG_UNDEF_INT = INT_MIN,
      NCCL_M2N_API_MAGIC = 0x4d324e32u,
      NCCL_M2N_API_VERSION = 2u
    };
    """
    enum:
        NCCL_RESHARD_MESH_NDIMS
        NCCL_RESHARD_MAX_MESH_DIMS
        NCCL_RESHARD_MAX_TENSOR_DIMS
        NCCL_RESHARD_REPLICATE
        NCCL_M2N_CONFIG_UNDEF_INT
        NCCL_M2N_API_MAGIC
        NCCL_M2N_API_VERSION

# enums


# types
# Opaque on purpose (as in nccl4py's cynccl.pxd): streams are only passed
# through to libnccl_m2n, so no CUDA/HIP headers are needed at build time.
ctypedef void* cudaStream_t
ctypedef int cudaError_t

ctypedef void* ncclM2nHandle_t 'ncclM2nHandle_t'

ctypedef struct ncclMesh_t 'ncclMesh_t':
    size_t size
    unsigned int version
    int ndims
    int* dims
    int startRank

ctypedef struct ncclM2nConfig_t 'ncclM2nConfig_t':
    size_t size
    unsigned int magic
    unsigned int version
    int maxCta

ctypedef struct ncclDistTensor_t 'ncclDistTensor_t':
    size_t size
    unsigned int version
    void* dataPtr
    size_t* localShape
    int ndims
    ncclDataType_t dtype
    const ncclMesh_t* mesh
    int* placements


###############################################################################
# Functions
###############################################################################

cdef ncclResult_t ncclM2nInit(ncclM2nHandle_t* handle, const ncclM2nConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclM2nFinalize(ncclM2nHandle_t handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclM2nGroupStart() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclM2nGroupEnd() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclM2nGroupAbort() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclReshardWithWindow(ncclM2nHandle_t handle, ncclComm_t comm, ncclWindow_t window, const ncclDistTensor_t* src, const ncclDistTensor_t* dst, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclReshard(ncclM2nHandle_t handle, ncclComm_t comm, const ncclDistTensor_t* src, const ncclDistTensor_t* dst, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil

# Keep the error-detail query on the original no-throw Cython contract.
cdef const char* ncclM2nGetLastError() noexcept nogil
