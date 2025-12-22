
#include "api_trace.h"
#include "core.h"
#include "nccl.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#if defined(RCCL_ROCPROFILER_REGISTER) && RCCL_ROCPROFILER_REGISTER > 0
#    include <rocprofiler-register/rocprofiler-register.h>

#    define ROCP_REG_VERSION                                                             \
        ROCPROFILER_REGISTER_COMPUTE_VERSION_3(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH)

ROCPROFILER_REGISTER_DEFINE_IMPORT(rccl, ROCP_REG_VERSION)
#endif

ncclResult_t
ncclAllGather_impl(const void* sendbuff, void* recvbuff, size_t sendcount,
                   ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t
ncclAllReduce_impl(const void* sendbuff, void* recvbuff, size_t count,
                   ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
                   cudaStream_t stream);

ncclResult_t
ncclAllToAll_impl(const void* sendbuff, void* recvbuff, size_t count,
                  ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);

ncclResult_t
ncclAllToAllv_impl(const void* sendbuff, const size_t sendcounts[],
                   const size_t sdispls[], void* recvbuff, const size_t recvcounts[],
                   const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm,
                   hipStream_t stream);

ncclResult_t
ncclBroadcast_impl(const void* sendbuff, void* recvbuff, size_t count,
                   ncclDataType_t datatype, int root, ncclComm_t comm,
                   cudaStream_t stream);

ncclResult_t
ncclGather_impl(const void* sendbuff, void* recvbuff, size_t sendcount,
                ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);

ncclResult_t
ncclReduce_impl(const void* sendbuff, void* recvbuff, size_t count,
                ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm,
                cudaStream_t stream);

ncclResult_t
ncclReduceScatter_impl(const void* sendbuff, void* recvbuff, size_t recvcount,
                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
                       cudaStream_t stream);

ncclResult_t
ncclScatter_impl(const void* sendbuff, void* recvbuff, size_t recvcount,
                 ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);

ncclResult_t
ncclSend_impl(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
              ncclComm_t comm, cudaStream_t stream);

ncclResult_t
ncclRecv_impl(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
              ncclComm_t comm, cudaStream_t stream);

ncclResult_t
ncclRedOpCreatePreMulSum_impl(ncclRedOp_t* op, void* scalar, ncclDataType_t datatype,
                              ncclScalarResidence_t residence, ncclComm_t comm);

ncclResult_t
ncclRedOpDestroy_impl(ncclRedOp_t op, ncclComm_t comm);

ncclResult_t
ncclGroupStart_impl();

ncclResult_t
ncclGroupEnd_impl();

ncclResult_t
ncclGetVersion_impl(int* version);

ncclResult_t
ncclGetUniqueId_impl(ncclUniqueId* out);

ncclResult_t
ncclCommInitRank_impl(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);

ncclResult_t
ncclCommInitAll_impl(ncclComm_t* comms, int ndev, const int* devlist);

ncclResult_t
ncclCommInitRankConfig_impl(ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank,
                            ncclConfig_t* config);

ncclResult_t
ncclCommFinalize_impl(ncclComm_t comm);

ncclResult_t
ncclCommDestroy_impl(ncclComm_t comm);

ncclResult_t
ncclCommAbort_impl(ncclComm_t comm);

ncclResult_t
ncclCommShrink_impl(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t *newcomm,
                    ncclConfig_t* config, int shrinkFlags);

ncclResult_t
ncclCommSplit_impl(ncclComm_t comm, int color, int key, ncclComm_t* newcomm,
                   ncclConfig_t* config);

const char*
ncclGetErrorString_impl(ncclResult_t code);

const char*
ncclGetLastError_impl(const ncclComm_t comm);

ncclResult_t
ncclCommGetAsyncError_impl(ncclComm_t comm, ncclResult_t* asyncError);

ncclResult_t
ncclCommCount_impl(const ncclComm_t comm, int* count);

ncclResult_t
ncclCommCuDevice_impl(const ncclComm_t comm, int* devid);

ncclResult_t
ncclCommUserRank_impl(const ncclComm_t comm, int* rank);

ncclResult_t
ncclMemAlloc_impl(void** ptr, size_t size);

ncclResult_t
ncclMemFree_impl(void* ptr);

ncclResult_t
mscclLoadAlgo_impl(const char* mscclAlgoFilePath, mscclAlgoHandle_t* mscclAlgoHandle,
                   int rank);

ncclResult_t
mscclRunAlgo_impl(const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
                  void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
                  size_t count, ncclDataType_t dataType, int root, int peer,
                  ncclRedOp_t op, mscclAlgoHandle_t mscclAlgoHandle, ncclComm_t comm,
                  hipStream_t stream);

ncclResult_t
mscclUnloadAlgo_impl(mscclAlgoHandle_t mscclAlgoHandle);

ncclResult_t
ncclCommRegister_impl(const ncclComm_t comm, void* buff, size_t size, void** handle);

ncclResult_t
ncclCommDeregister_impl(const ncclComm_t comm, void* handle);

ncclResult_t
ncclCommWindowRegister_impl(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags);

ncclResult_t
ncclCommWindowDeregister_impl(ncclComm_t comm, ncclWindow_t win);

ncclResult_t
ncclAllReduceWithBias_impl(const void* sendbuff, void* recvbuff, size_t count,
                   ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
                   cudaStream_t stream, const void* acc);

namespace rccl
{
namespace
{

constexpr size_t
compute_table_offset(size_t n)
{
    return (sizeof(uint64_t) + (n * sizeof(void*)));
}

constexpr size_t
compute_table_size(size_t nmembers)
{
    return (sizeof(uint64_t) + (nmembers * sizeof(void*)));
}

#define RCCL_ASSERT_OFFSET(TABLE, MEMBER, IDX)                                           \
    static_assert(offsetof(TABLE, MEMBER) == compute_table_offset(IDX),                  \
                  "Do not re-arrange the table members")

// DO NOT REORDER, ADD NEW ITEMS TO BOTTOM
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclAllGather_fn, 0);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclAllReduce_fn, 1);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclAllToAll_fn, 2);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclAllToAllv_fn, 3);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclBroadcast_fn, 4);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclGather_fn, 5);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclReduce_fn, 6);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclReduceScatter_fn, 7);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclScatter_fn, 8);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclSend_fn, 9);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclRecv_fn, 10);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclRedOpCreatePreMulSum_fn, 11);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclRedOpDestroy_fn, 12);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclGroupStart_fn, 13);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclGroupEnd_fn, 14);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclGetVersion_fn, 15);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclGetUniqueId_fn, 16);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommInitRank_fn, 17);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommInitAll_fn, 18);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommInitRankConfig_fn, 19);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommFinalize_fn, 20);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommDestroy_fn, 21);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommAbort_fn, 22);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommSplit_fn, 23);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclGetErrorString_fn, 24);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclGetLastError_fn, 25);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommGetAsyncError_fn, 26);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommCount_fn, 27);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommCuDevice_fn, 28);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommUserRank_fn, 29);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclMemAlloc_fn, 30);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclMemFree_fn, 31);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, mscclLoadAlgo_fn, 32);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, mscclRunAlgo_fn, 33);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, mscclUnloadAlgo_fn, 34);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommRegister_fn, 35);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommDeregister_fn, 36);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclAllReduceWithBias_fn, 37);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommShrink_fn, 38);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommWindowRegister_fn, 39);
RCCL_ASSERT_OFFSET(rcclApiFuncTable, ncclCommWindowDeregister_fn, 40);
// DO NOT REORDER, ADD NEW ITEMS HERE

#undef RCCL_ASSERT_OFFSET

static_assert(sizeof(rcclApiFuncTable) == compute_table_size(41),
              "Update table major/step version and add a new offset assertion if this "
              "fails to compile");

std::array<unsigned char, sizeof(rcclApiFuncTable)> m_buffer = {};

rcclApiFuncTable*
RcclGetFunctionTable_impl()
{
    static auto* tbl =
        new(m_buffer.data()) rcclApiFuncTable{ sizeof(rcclApiFuncTable),
                                               &ncclAllGather_impl,
                                               &ncclAllReduce_impl,
                                               &ncclAllToAll_impl,
                                               &ncclAllToAllv_impl,
                                               &ncclBroadcast_impl,
                                               &ncclGather_impl,
                                               &ncclReduce_impl,
                                               &ncclReduceScatter_impl,
                                               &ncclScatter_impl,
                                               &ncclSend_impl,
                                               &ncclRecv_impl,
                                               &ncclRedOpCreatePreMulSum_impl,
                                               &ncclRedOpDestroy_impl,
                                               &ncclGroupStart_impl,
                                               &ncclGroupEnd_impl,
                                               &ncclGetVersion_impl,
                                               &ncclGetUniqueId_impl,
                                               &ncclCommInitRank_impl,
                                               &ncclCommInitAll_impl,
                                               &ncclCommInitRankConfig_impl,
                                               &ncclCommFinalize_impl,
                                               &ncclCommDestroy_impl,
                                               &ncclCommAbort_impl,
                                               &ncclCommSplit_impl,
                                               &ncclGetErrorString_impl,
                                               &ncclGetLastError_impl,
                                               &ncclCommGetAsyncError_impl,
                                               &ncclCommCount_impl,
                                               &ncclCommCuDevice_impl,
                                               &ncclCommUserRank_impl,
                                               &ncclMemAlloc_impl,
                                               &ncclMemFree_impl,
                                               &mscclLoadAlgo_impl,
                                               &mscclRunAlgo_impl,
                                               &mscclUnloadAlgo_impl,
                                               &ncclCommRegister_impl,
                                               &ncclCommDeregister_impl,
                                               &ncclAllReduceWithBias_impl,
                                               &ncclCommShrink_impl,
                                               &ncclCommWindowRegister_impl,
                                               &ncclCommWindowDeregister_impl
                                               // DO NOT REORDER, ADD NEW ITEMS HERE
                                             };

#if defined(RCCL_ROCPROFILER_REGISTER) && RCCL_ROCPROFILER_REGISTER > 0
    std::array<void*, 1>                       table_array{ tbl };
    rocprofiler_register_library_indentifier_t lib_id =
        rocprofiler_register_library_indentifier_t{};
    rocprofiler_register_error_code_t rocp_reg_status =
        rocprofiler_register_library_api_table(
            "rccl", &ROCPROFILER_REGISTER_IMPORT_FUNC(rccl), ROCP_REG_VERSION,
            table_array.data(), table_array.size(), &lib_id);

    INFO(NCCL_COLL,
         "[rocprofiler-sdk-rccl][ = %d ] rocprofiler-register returned code = %d : %s",
         getpid(), rocp_reg_status, rocprofiler_register_error_string(rocp_reg_status));

    if(rocp_reg_status != ROCP_REG_SUCCESS && rocp_reg_status != ROCP_REG_NO_TOOLS)
        WARN("[rocprofiler-sdk-rccl][%d] rocprofiler-register failed with error code %d "
             ": %s",
             getpid(), rocp_reg_status,
             rocprofiler_register_error_string(rocp_reg_status));
#endif

    return tbl;
}
}  // end of namespace

const rcclApiFuncTable*
RcclGetFunctionTable()
{
    static const auto* tbl = RcclGetFunctionTable_impl();
    return tbl;
}
}  // end of namespace rccl

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff,
         size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
         ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclAllReduceWithBias, const void* sendbuff, void* recvbuff, size_t count,
         ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, hipStream_t stream, const void* acc);

NCCL_API(ncclResult_t, ncclAllToAll, const void* sendbuff, void* recvbuff, size_t count,
         ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclAllToAllv, const void* sendbuff, const size_t sendcounts[],
         const size_t sdispls[], void* recvbuff, const size_t recvcounts[],
         const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm,
         hipStream_t stream);

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count,
         ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t sendcount,
         ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
         ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm,
         hipStream_t stream);

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff,
         size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
         hipStream_t stream);

NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff,
         size_t recvcount, ncclDataType_t datatype, int root, ncclComm_t comm,
         hipStream_t stream);

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count,
         ncclDataType_t datatype, int peer, ncclComm_t comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype,
         int peer, ncclComm_t comm, hipStream_t stream);

NCCL_API(ncclResult_t, ncclRedOpCreatePreMulSum, ncclRedOp_t* op, void* scalar,
         ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);

NCCL_API(ncclResult_t, ncclRedOpDestroy, ncclRedOp_t op, ncclComm_t comm);

NCCL_API(ncclResult_t, ncclGroupStart);

NCCL_API(ncclResult_t, ncclGroupEnd);

NCCL_API(ncclResult_t, ncclGetVersion, int* version);

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks,
         ncclUniqueId commId, int myrank);

NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);

NCCL_API(ncclResult_t, ncclCommInitRankConfig, ncclComm_t* comm, int nranks,
         ncclUniqueId commId, int myrank, ncclConfig_t* config);

NCCL_API(ncclResult_t, ncclCommFinalize, ncclComm_t comm);

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);

NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);

NCCL_API(ncclResult_t, ncclCommShrink, ncclComm_t comm, int* excludeRanksList, int excludeRanksCount,
         ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);

NCCL_API(ncclResult_t, ncclCommSplit, ncclComm_t comm, int color, int key,
         ncclComm_t* newcomm, ncclConfig_t* config);

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);

NCCL_API(const char*, ncclGetLastError, const ncclComm_t comm);

NCCL_API(ncclResult_t, ncclCommGetAsyncError, ncclComm_t comm, ncclResult_t* asyncError);

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);

NCCL_API(ncclResult_t, ncclMemAlloc, void** ptr, size_t size);

NCCL_API(ncclResult_t, ncclMemFree, void* ptr);

NCCL_API(ncclResult_t, mscclLoadAlgo, const char* mscclAlgoFilePath,
         mscclAlgoHandle_t* mscclAlgoHandle, int rank);

NCCL_API(ncclResult_t, mscclRunAlgo, const void* sendBuff, const size_t sendCounts[],
         const size_t sDisPls[], void* recvBuff, const size_t recvCounts[],
         const size_t rDisPls[], size_t count, ncclDataType_t dataType, int root,
         int peer, ncclRedOp_t op, mscclAlgoHandle_t mscclAlgoHandle, ncclComm_t comm,
         hipStream_t stream);

NCCL_API(ncclResult_t, mscclUnloadAlgo, mscclAlgoHandle_t mscclAlgoHandle);

NCCL_API(ncclResult_t, ncclCommRegister, const ncclComm_t comm, void* buff, size_t size,
         void** handle);

NCCL_API(ncclResult_t, ncclCommDeregister, const ncclComm_t comm, void* handle);

NCCL_API(ncclResult_t, ncclCommWindowRegister, ncclComm_t comm, void* buff, size_t size, 
         ncclWindow_t* win, int winFlags);

NCCL_API(ncclResult_t, ncclCommWindowDeregister, ncclComm_t comm, ncclWindow_t win);

ncclResult_t
ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
              ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)
{
    return ::rccl::RcclGetFunctionTable()->ncclAllGather_fn(sendbuff, recvbuff, sendcount,
                                                            datatype, comm, stream);
}

ncclResult_t
ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
              ncclRedOp_t op, ncclComm* comm, cudaStream_t stream)
{
    return ::rccl::RcclGetFunctionTable()->ncclAllReduce_fn(sendbuff, recvbuff, count,
                                                            datatype, op, comm, stream);
}

ncclResult_t
ncclAllReduceWithBias(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
              ncclRedOp_t op, ncclComm* comm, cudaStream_t stream, const void* acc)
{
    return ::rccl::RcclGetFunctionTable()->ncclAllReduceWithBias_fn(sendbuff, recvbuff, count,
                                                            datatype, op, comm, stream, acc);
}

ncclResult_t
ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
             ncclComm_t comm, hipStream_t stream)
{
    return ::rccl::RcclGetFunctionTable()->ncclAllToAll_fn(sendbuff, recvbuff, count,
                                                           datatype, comm, stream);
}

ncclResult_t
ncclAllToAllv(const void* sendbuff, const size_t sendcounts[], const size_t sdispls[],
              void* recvbuff, const size_t recvcounts[], const size_t rdispls[],
              ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream)
{
    return ::rccl::RcclGetFunctionTable()->ncclAllToAllv_fn(sendbuff, sendcounts, sdispls,
                                                            recvbuff, recvcounts, rdispls,
                                                            datatype, comm, stream);
}

ncclResult_t
ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
              int root, ncclComm_t comm, cudaStream_t stream)
{
    return ::rccl::RcclGetFunctionTable()->ncclBroadcast_fn(sendbuff, recvbuff, count,
                                                            datatype, root, comm, stream);
}

ncclResult_t
ncclGather(const void* sendbuff, void* recvbuff, size_t sendcount,
           ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream)
{
    return ::rccl::RcclGetFunctionTable()->ncclGather_fn(sendbuff, recvbuff, sendcount,
                                                         datatype, root, comm, stream);
}

ncclResult_t
ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
           ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
{
    return ::rccl::RcclGetFunctionTable()->ncclReduce_fn(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

ncclResult_t
ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
                  cudaStream_t stream)
{
    return ::rccl::RcclGetFunctionTable()->ncclReduceScatter_fn(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

ncclResult_t
ncclScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
            ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream)
{
    return ::rccl::RcclGetFunctionTable()->ncclScatter_fn(sendbuff, recvbuff, recvcount,
                                                          datatype, root, comm, stream);
}

ncclResult_t
ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
         ncclComm_t comm, cudaStream_t stream)
{
    return ::rccl::RcclGetFunctionTable()->ncclSend_fn(sendbuff, count, datatype, peer,
                                                       comm, stream);
}

ncclResult_t
ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
         cudaStream_t stream)
{
    return ::rccl::RcclGetFunctionTable()->ncclRecv_fn(recvbuff, count, datatype, peer,
                                                       comm, stream);
}

ncclResult_t
ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar, ncclDataType_t datatype,
                         ncclScalarResidence_t residence, ncclComm_t comm)
{
    return ::rccl::RcclGetFunctionTable()->ncclRedOpCreatePreMulSum_fn(
        op, scalar, datatype, residence, comm);
}

ncclResult_t
ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm)
{
    return ::rccl::RcclGetFunctionTable()->ncclRedOpDestroy_fn(op, comm);
}

ncclResult_t
ncclGroupStart()
{
    return ::rccl::RcclGetFunctionTable()->ncclGroupStart_fn();
}

ncclResult_t
ncclGroupEnd()
{
    return ::rccl::RcclGetFunctionTable()->ncclGroupEnd_fn();
}

ncclResult_t
ncclGetVersion(int* version)
{
    return ::rccl::RcclGetFunctionTable()->ncclGetVersion_fn(version);
}

ncclResult_t
ncclGetUniqueId(ncclUniqueId* out)
{
    return ::rccl::RcclGetFunctionTable()->ncclGetUniqueId_fn(out);
}

ncclResult_t
ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommInitRank_fn(newcomm, nranks, commId,
                                                               myrank);
}

ncclResult_t
ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommInitAll_fn(comms, ndev, devlist);
}

ncclResult_t
ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank,
                       ncclConfig_t* config)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommInitRankConfig_fn(comm, nranks, commId,
                                                                     myrank, config);
}

ncclResult_t
ncclCommFinalize(ncclComm_t comm)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommFinalize_fn(comm);
}

ncclResult_t
ncclCommDestroy(ncclComm_t comm)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommDestroy_fn(comm);
}

ncclResult_t
ncclCommAbort(ncclComm_t comm)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommAbort_fn(comm);
}

ncclResult_t
ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm,
               ncclConfig_t* config, int shrinkFlags)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommShrink_fn(comm, excludeRanksList, excludeRanksCount, 
                                                             newcomm, config, shrinkFlags);
}

ncclResult_t
ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm,
              ncclConfig_t* config)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommSplit_fn(comm, color, key, newcomm,
                                                            config);
}

const char*
ncclGetErrorString(ncclResult_t code)
{
    return ::rccl::RcclGetFunctionTable()->ncclGetErrorString_fn(code);
}

const char*
ncclGetLastError(const ncclComm_t comm)
{
    return ::rccl::RcclGetFunctionTable()->ncclGetLastError_fn(comm);
}

ncclResult_t
ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommGetAsyncError_fn(comm, asyncError);
}

ncclResult_t
ncclCommCount(const ncclComm_t comm, int* count)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommCount_fn(comm, count);
}

ncclResult_t
ncclCommCuDevice(const ncclComm_t comm, int* devid)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommCuDevice_fn(comm, devid);
}

ncclResult_t
ncclCommUserRank(const ncclComm_t comm, int* rank)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommUserRank_fn(comm, rank);
}

ncclResult_t
ncclMemAlloc(void** ptr, size_t size)
{
    return ::rccl::RcclGetFunctionTable()->ncclMemAlloc_fn(ptr, size);
}

ncclResult_t
ncclMemFree(void* ptr)
{
    return ::rccl::RcclGetFunctionTable()->ncclMemFree_fn(ptr);
}

ncclResult_t
mscclLoadAlgo(const char* mscclAlgoFilePath, mscclAlgoHandle_t* mscclAlgoHandle, int rank)
{
    return ::rccl::RcclGetFunctionTable()->mscclLoadAlgo_fn(mscclAlgoFilePath,
                                                            mscclAlgoHandle, rank);
}

ncclResult_t
mscclRunAlgo(const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
             void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
             size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
             mscclAlgoHandle_t mscclAlgoHandle, ncclComm_t comm, hipStream_t stream)
{
    return ::rccl::RcclGetFunctionTable()->mscclRunAlgo_fn(
        sendBuff, sendCounts, sDisPls, recvBuff, recvCounts, rDisPls, count, dataType,
        root, peer, op, mscclAlgoHandle, comm, stream);
}

ncclResult_t
mscclUnloadAlgo(mscclAlgoHandle_t mscclAlgoHandle)
{
    return ::rccl::RcclGetFunctionTable()->mscclUnloadAlgo_fn(mscclAlgoHandle);
}

ncclResult_t
ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommRegister_fn(comm, buff, size, handle);
}

ncclResult_t
ncclCommDeregister(const ncclComm_t comm, void* handle)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommDeregister_fn(comm, handle);
}

ncclResult_t
ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommWindowRegister_fn(comm, buff, size, win, winFlags);
}

ncclResult_t
ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win)
{
    return ::rccl::RcclGetFunctionTable()->ncclCommWindowDeregister_fn(comm, win);
}
