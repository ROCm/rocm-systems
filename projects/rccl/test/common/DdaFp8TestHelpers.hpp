/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

#include <hip/hip_runtime.h>

#include "comm.h"
#include "gtest/gtest.h"
#include "nccl.h"

namespace RcclUnitTesting
{

inline const std::vector<ncclDataType_t> kDdaFp8Types{ncclFloat8e4m3, ncclFloat8e5m2};

// One 16-byte line of fp8 elements.
constexpr size_t kDdaFp8LineCount = 16;

// Eligibility gates and launchers share the reducing collectives' signature so
// one table can hold every collective. The adapters drop the op for AllGather
// and AlltoAll.
using DdaGateFn = bool (*)(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t);
using DdaLaunchFn =
    ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm*, hipStream_t);

template <bool (*Gate)(ncclComm*, const void*, void*, size_t, ncclDataType_t)>
bool ddaGateIgnoringOp(
    ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t)
{
    return Gate(comm, sendbuff, recvbuff, count, datatype);
}

template <ncclResult_t (*Launch)(const void*, void*, size_t, ncclDataType_t, ncclComm*, hipStream_t)>
ncclResult_t ddaLaunchIgnoringOp(const void*    sendbuff,
                                 void*          recvbuff,
                                 size_t         count,
                                 ncclDataType_t datatype,
                                 ncclRedOp_t,
                                 ncclComm*      comm,
                                 hipStream_t    stream)
{
    return Launch(sendbuff, recvbuff, count, datatype, comm, stream);
}

struct DdaGateCase
{
    const char* name;
    DdaGateFn   gate;
    size_t      count;
};

struct DdaLaunchCase
{
    const char* name;
    DdaLaunchFn launch;
};

// Every gate must return `expected` for every datatype, with op ncclSum.
inline void expectDdaGates(std::initializer_list<DdaGateCase>  cases,
                           const std::vector<ncclDataType_t>& datatypes,
                           bool                               expected,
                           ncclComm*                          comm,
                           const void*                        sendbuff,
                           void*                              recvbuff)
{
    for (const DdaGateCase& c : cases)
        for (ncclDataType_t datatype : datatypes)
            EXPECT_EQ(c.gate(comm, sendbuff, recvbuff, c.count, datatype, ncclSum), expected)
                << c.name << ", count " << c.count << ", datatype " << datatype;
}

// Every launcher must return `expected` for every datatype, with op ncclSum and
// a null stream.
inline void expectDdaLaunches(std::initializer_list<DdaLaunchCase> cases,
                              const std::vector<ncclDataType_t>&   datatypes,
                              size_t                               count,
                              ncclResult_t                         expected,
                              ncclComm*                            comm,
                              const void*                          sendbuff,
                              void*                                recvbuff)
{
    for (const DdaLaunchCase& c : cases)
        for (ncclDataType_t datatype : datatypes)
            EXPECT_EQ(c.launch(sendbuff, recvbuff, count, datatype, ncclSum, comm, nullptr), expected)
                << c.name << ", count " << count << ", datatype " << datatype;
}

} // namespace RcclUnitTesting
