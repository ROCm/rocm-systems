/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_H_
#define NCCL_H_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define NCCL_MAJOR  2
#define NCCL_MINOR  11
#define NCCL_PATCH  4
#define NCCL_SUFFIX ""

#define NCCL_VERSION_CODE 21104
#define NCCL_VERSION(X, Y, Z)                                                            \
    (((X) <= 2 && (Y) <= 8) ? (X) *1000 + (Y) *100 + (Z) : (X) *10000 + (Y) *100 + (Z))

#define RCCL_BFLOAT16       1
#define RCCL_GATHER_SCATTER 1
#define RCCL_ALLTOALLV      1

#ifdef __cplusplus
extern "C"
{
#endif

    /*! @brief Opaque handle to communicator */
    typedef struct ncclComm* ncclComm_t;

#define NCCL_UNIQUE_ID_BYTES 128
    typedef struct
    {
        char internal[NCCL_UNIQUE_ID_BYTES];
    } ncclUniqueId;

    /*! @brief Error type */
    typedef enum
    {
        ncclSuccess            = 0,
        ncclUnhandledCudaError = 1,
        ncclSystemError        = 2,
        ncclInternalError      = 3,
        ncclInvalidArgument    = 4,
        ncclInvalidUsage       = 5,
        ncclNumResults         = 6
    } ncclResult_t;

    /*! @brief Return the NCCL_VERSION_CODE of the NCCL library in the supplied integer.
     *
     * @details This integer is coded with the MAJOR, MINOR and PATCH level of the
     * NCCL library
     */
    ncclResult_t ncclGetVersion(int* version);
    /// @cond include_hidden
    ncclResult_t pncclGetVersion(int* version);
    /// @endcond

    /*! @brief Generates an ID for ncclCommInitRank

        @details
        Generates an ID to be used in ncclCommInitRank. ncclGetUniqueId should be
        called once and the Id should be distributed to all ranks in the
        communicator before calling ncclCommInitRank.

        @param[in]
        uniqueId     ncclUniqueId*
                     pointer to uniqueId

    */
    ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
    /// @cond include_hidden
    ncclResult_t pncclGetUniqueId(ncclUniqueId* uniqueId);
    /// @endcond

    /*! @brief Creates a new communicator (multi thread/process version).

        @details
        rank must be between 0 and nranks-1 and unique within a communicator clique.
        Each rank is associated to a CUDA device, which has to be set before calling
        ncclCommInitRank.
        ncclCommInitRank implicitly syncronizes with other ranks, so it must be
        called by different threads/processes or use ncclGroupStart/ncclGroupEnd.

        @param[in]
        comm        ncclComm_t*
                    communicator struct pointer
        */
    ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId,
                                  int rank);
    /// @cond include_hidden
    ncclResult_t pncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId,
                                   int rank);
    /// @endcond

    /*! @brief Creates a clique of communicators (single process version).
     *
     * @details This is a convenience function to create a single-process communicator
     * clique. Returns an array of ndev newly initialized communicators in comm. comm
     * should be pre-allocated with size at least ndev*sizeof(ncclComm_t). If devlist is
     * NULL, the first ndev HIP devices are used. Order of devlist defines user-order of
     * processors within the communicator.
     * */
    ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
    /// @cond include_hidden
    ncclResult_t pncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
    /// @endcond

    /*! @brief Frees resources associated with communicator object, but waits for any
     * operations that might still be running on the device */
    ncclResult_t ncclCommDestroy(ncclComm_t comm);
    /// @cond include_hidden
    ncclResult_t pncclCommDestroy(ncclComm_t comm);
    /// @endcond

    /*! @brief Frees resources associated with communicator object and aborts any
     * operations that might still be running on the device. */
    ncclResult_t ncclCommAbort(ncclComm_t comm);
    /// @cond include_hidden
    ncclResult_t pncclCommAbort(ncclComm_t comm);
    /// @endcond

    /*! @brief Returns a human-readable error message. */
    const char* ncclGetErrorString(ncclResult_t result);
    const char* pncclGetErrorString(ncclResult_t result);

    /*! @brief Checks whether the comm has encountered any asynchronous errors */
    ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);
    /// @cond include_hidden
    ncclResult_t pncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);
    /// @endcond

    /*! @brief Gets the number of ranks in the communicator clique. */
    ncclResult_t ncclCommCount(const ncclComm_t comm, int* count);
    /// @cond include_hidden
    ncclResult_t pncclCommCount(const ncclComm_t comm, int* count);
    /// @endcond

    /*! @brief Returns the rocm device number associated with the communicator. */
    ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device);
    /// @cond include_hidden
    ncclResult_t pncclCommCuDevice(const ncclComm_t comm, int* device);
    /// @endcond

    /*! @brief Returns the user-ordered "rank" associated with the communicator. */
    ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank);
    /// @cond include_hidden
    ncclResult_t pncclCommUserRank(const ncclComm_t comm, int* rank);
    /// @endcond

    /*! @brief Reduction operation selector */
    /* Reduction operation selector */
    typedef enum
    {
        ncclNumOps_dummy = 5
    } ncclRedOp_dummy_t;
    typedef enum
    {
        ncclSum  = 0,
        ncclProd = 1,
        ncclMax  = 2,
        ncclMin  = 3,
        ncclAvg  = 4,
        /* ncclNumOps: The number of built-in ncclRedOp_t values. Also
         * serves as the least possible value for dynamic ncclRedOp_t's
         * as constructed by ncclRedOpCreate*** functions. */
        ncclNumOps = 5,
        /* ncclMaxRedOp: The largest valid value for ncclRedOp_t.
         * It is defined to be the largest signed value (since compilers
         * are permitted to use signed enums) that won't grow
         * sizeof(ncclRedOp_t) when compared to previous NCCL versions to
         * maintain ABI compatibility. */
        ncclMaxRedOp = 0x7fffffff >> (32 - 8 * sizeof(ncclRedOp_dummy_t))
    } ncclRedOp_t;

    /*! @brief Data types */
    typedef enum
    {
        ncclInt8     = 0,
        ncclChar     = 0,
        ncclUint8    = 1,
        ncclInt32    = 2,
        ncclInt      = 2,
        ncclUint32   = 3,
        ncclInt64    = 4,
        ncclUint64   = 5,
        ncclFloat16  = 6,
        ncclHalf     = 6,
        ncclFloat32  = 7,
        ncclFloat    = 7,
        ncclFloat64  = 8,
        ncclDouble   = 8,
        ncclBfloat16 = 9,
        ncclNumTypes = 10
    } ncclDataType_t;

    /* ncclScalarResidence_t: Location and dereferencing logic for scalar arguments. */
    typedef enum
    {
        /* ncclScalarDevice: The scalar is in device-visible memory and will be
         * dereferenced while the collective is running. */
        ncclScalarDevice = 0,

        /* ncclScalarHostImmediate: The scalar is in host-visible memory and will be
         * dereferenced before the ncclRedOpCreate***() function returns. */
        ncclScalarHostImmediate = 1
    } ncclScalarResidence_t;

    /*
     * ncclRedOpCreatePreMulSum
     *
     * Creates a new reduction operator which pre-multiplies input values by a given
     * scalar locally before reducing them with peer values via summation. For use
     * only with collectives launched against *comm* and *datatype*. The
     * *residence* argument indicates how/when the memory pointed to by *scalar*
     * will be dereferenced. Upon return, the newly created operator's handle
     * is stored in *op*.
     */
    ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar,
                                          ncclDataType_t        datatype,
                                          ncclScalarResidence_t residence,
                                          ncclComm_t            comm);
    ncclResult_t pncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar,
                                           ncclDataType_t        datatype,
                                           ncclScalarResidence_t residence,
                                           ncclComm_t            comm);

    /*
     * ncclRedOpDestroy
     *
     * Destroys the reduction operator *op*. The operator must have been created by
     * ncclRedOpCreatePreMul with the matching communicator *comm*. An operator may be
     * destroyed as soon as the last NCCL function which is given that operator returns.
     */
    ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);
    ncclResult_t pncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);

    /*
     * Collective communication operations
     *
     * Collective communication operations must be called separately for each
     * communicator in a communicator clique.
     *
     * They return when operations have been enqueued on the CUDA stream.
     *
     * Since they may perform inter-CPU synchronization, each call has to be done
     * from a different thread or process, or need to use Group Semantics (see
     * below).
     */

    /*!
     * @brief Reduce
     *
     * @details Reduces data arrays of length count in sendbuff into recvbuff using op
     * operation.
     * recvbuff may be NULL on all calls except for root device.
     * root is the rank (not the CUDA device) where data will reside after the
     * operation is complete.
     *
     * In-place operation will happen if sendbuff == recvbuff.
     */
    ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op, int root,
                            ncclComm_t comm, hipStream_t stream);
    /// @cond include_hidden
    ncclResult_t pncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                             ncclDataType_t datatype, ncclRedOp_t op, int root,
                             ncclComm_t comm, hipStream_t stream);
    /// @endcond

    /*! @brief (deprecated) Broadcast (in-place)
     *
     * @details Copies count values from root to all other devices.
     * root is the rank (not the CUDA device) where data resides before the
     * operation is started.
     *
     * This operation is implicitely in place.
     */
    ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
                           ncclComm_t comm, hipStream_t stream);
    /// @cond include_hidden
    ncclResult_t pncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
                            ncclComm_t comm, hipStream_t stream);
    /// @endcond

    /*! @brief Broadcast
     *
     * @details Copies count values from root to all other devices.
     * root is the rank (not the HIP device) where data resides before the
     * operation is started.
     *
     * In-place operation will happen if sendbuff == recvbuff.
     */
    ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, int root, ncclComm_t comm,
                               hipStream_t stream);
    /// @cond include_hidden
    ncclResult_t pncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                                ncclDataType_t datatype, int root, ncclComm_t comm,
                                hipStream_t stream);
    /// @endcond

    /*! @brief All-Reduce
     *
     * @details Reduces data arrays of length count in sendbuff using op operation, and
     * leaves identical copies of result on each recvbuff.
     *
     * In-place operation will happen if sendbuff == recvbuff.
     */
    ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                               hipStream_t stream);
    /// @cond include_hidden
    ncclResult_t pncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                                ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                hipStream_t stream);
    /// @endcond

    /*!
     * @brief Reduce-Scatter
     *
     * @details Reduces data in sendbuff using op operation and leaves reduced result
     * scattered over the devices so that recvbuff on rank i will contain the i-th
     * block of the result.
     * Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
     * should have a size of at least nranks*recvcount elements.
     *
     * In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
     */
    ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                                   ncclDataType_t datatype, ncclRedOp_t op,
                                   ncclComm_t comm, hipStream_t stream);
    /// @cond include_hidden
    ncclResult_t pncclReduceScatter(const void* sendbuff, void* recvbuff,
                                    size_t recvcount, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm_t comm, hipStream_t stream);
    /// @endcond

    /*! @brief All-Gather
     *
     * @details Each device gathers sendcount values from other GPUs into recvbuff,
     * receiving data from rank i at offset i*sendcount.
     * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
     * should have a size of at least nranks*sendcount elements.
     *
     * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
     */
    ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                               ncclDataType_t datatype, ncclComm_t comm,
                               hipStream_t stream);
    /// @cond include_hidden
    ncclResult_t pncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                                ncclDataType_t datatype, ncclComm_t comm,
                                hipStream_t stream);
    /// @endcond

    /*! @brief Send
     *
     * @details Send data from sendbuff to rank peer.
     * Rank peer needs to call ncclRecv with the same datatype and the same count from
     * this rank.
     *
     * This operation is blocking for the GPU. If multiple ncclSend and ncclRecv
     * operations need to progress concurrently to complete, they must be fused within a
     * ncclGroupStart/ ncclGroupEnd section.
     */
    ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                          int peer, ncclComm_t comm, hipStream_t stream);
    /// @cond include_hidden
    ncclResult_t pncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                           int peer, ncclComm_t comm, hipStream_t stream);
    /// @endcond

    /*! @brief Receive
     *
     * @details Receive data from rank peer into recvbuff.
     * Rank peer needs to call ncclSend with the same datatype and the same count to this
     * rank.
     *
     * This operation is blocking for the GPU. If multiple ncclSend and ncclRecv
     * operations need to progress concurrently to complete, they must be fused within a
     * ncclGroupStart/ ncclGroupEnd section.
     */
    ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
                          ncclComm_t comm, hipStream_t stream);
    /// @cond include_hidden
    ncclResult_t pncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                           int peer, ncclComm_t comm, hipStream_t stream);
    /// @endcond

    /*! @brief Gather
     *
     * @details Root device gathers sendcount values from other GPUs into recvbuff,
     * receiving data from rank i at offset i*sendcount.
     *
     * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
     * should have a size of at least nranks*sendcount elements.
     *
     * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
     */
    ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                            ncclDataType_t datatype, int root, ncclComm_t comm,
                            hipStream_t stream);
    /// @cond include_hidden
    ncclResult_t pncclGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                             ncclDataType_t datatype, int root, ncclComm_t comm,
                             hipStream_t stream);
    /// @endcond

    /*! @brief Scatter
     *
     * @details Scattered over the devices so that recvbuff on rank i will contain the
     * i-th block of the data on root.
     *
     * Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
     * should have a size of at least nranks*recvcount elements.
     *
     * In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
     */
    ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                             ncclDataType_t datatype, int root, ncclComm_t comm,
                             hipStream_t stream);
    /// @cond include_hidden
    ncclResult_t pncclScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                              ncclDataType_t datatype, int root, ncclComm_t comm,
                              hipStream_t stream);
    /// @endcond

    /*! @brief All-To-All
     *
     * @details Device (i) send (j)th block of data to device (j) and be placed as (i)th
     * block. Each block for sending/receiving has count elements, which means
     * that recvbuff and sendbuff should have a size of nranks*count elements.
     *
     * In-place operation will happen if sendbuff == recvbuff.
     */
    ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
                              ncclDataType_t datatype, ncclComm_t comm,
                              hipStream_t stream);
    /// @cond include_hidden
    ncclResult_t pncclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclComm_t comm,
                               hipStream_t stream);
    /// @endcond

    /*! @brief All-To-Allv
     *
     * @details Device (i) sends sendcounts[j] of data from offset sdispls[j]
     * to device (j). In the same time, device (i) receives recvcounts[j] of data
     * from device (j) to be placed at rdispls[j].

     * sendcounts, sdispls, recvcounts and rdispls are all measured in the units
     * of datatype, not bytes.
     *
     * In-place operation will happen if sendbuff == recvbuff.
     */
    ncclResult_t ncclAllToAllv(const void* sendbuff, const size_t sendcounts[],
                               const size_t sdispls[], void* recvbuff,
                               const size_t recvcounts[], const size_t rdispls[],
                               ncclDataType_t datatype, ncclComm_t comm,
                               hipStream_t stream);
    /// @cond include_hidden
    ncclResult_t pncclAllToAllv(const void* sendbuff, const size_t sendcounts[],
                                const size_t sdispls[], void* recvbuff,
                                const size_t recvcounts[], const size_t rdispls[],
                                ncclDataType_t datatype, ncclComm_t comm,
                                hipStream_t stream);
    /// @endcond

    /*
     * Group semantics
     *
     * When managing multiple GPUs from a single thread, and since NCCL collective
     * calls may perform inter-CPU synchronization, we need to "group" calls for
     * different ranks/devices into a single call.
     *
     * Grouping NCCL calls as being part of the same collective operation is done
     * using ncclGroupStart and ncclGroupEnd. ncclGroupStart will enqueue all
     * collective calls until the ncclGroupEnd call, which will wait for all calls
     * to be complete. Note that for collective communication, ncclGroupEnd only
     * guarantees that the operations are enqueued on the streams, not that
     * the operation is effectively done.
     *
     * Both collective communication and ncclCommInitRank can be used in conjunction
     * of ncclGroupStart/ncclGroupEnd, but not together.
     *
     * Group semantics also allow to fuse multiple operations on the same device
     * to improve performance (for aggregated collective calls), or to permit
     * concurrent progress of multiple send/receive operations.
     */

    /*! @brief Group Start
     *
     * Start a group call. All calls to NCCL until ncclGroupEnd will be fused into
     * a single NCCL operation. Nothing will be started on the CUDA stream until
     * ncclGroupEnd.
     */
    ncclResult_t ncclGroupStart();
    /// @cond include_hidden
    ncclResult_t pncclGroupStart();
    /// @endcond

    /*! @brief Group End
     *
     * End a group call. Start a fused NCCL operation consisting of all calls since
     * ncclGroupStart. Operations on the CUDA stream depending on the NCCL operations
     * need to be called after ncclGroupEnd.
     */
    ncclResult_t ncclGroupEnd();
    /// @cond include_hidden
    ncclResult_t pncclGroupEnd();
    /// @endcond

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // end include guard
