/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "device.h"

// Verifies the LL128 registered / non-registered kernel split.
//
// For the "reg-variant" collectives (AllReduce, AllGather, Broadcast) the LL128
// protocol is generated as TWO separate device functions selected at launch by
// the compile-time UserRegMode template parameter:
//   UserRegMode == 1 : registered user buffer     (Direct path, system-scope
//                      cache-bypassing load/store)
//   UserRegMode == 2 : non-registered user buffer  (plain / non-temporal path)
// Every other collective / protocol stays a single kernel (UserRegMode == 0).
//
// Source of truth: ncclDevFuncIsLL128RegVariant() and the func-id key packing in
// src/include/device.h, mirrored by `ll128_reg_variant_colls` / `reg_values_of`
// in src/device/generate.py. These tests pin that contract so the two code
// paths cannot silently collapse back into one kernel.

namespace RcclUnitTesting
{

namespace
{
constexpr int kRegMode0        = 0;  // not-applicable / single kernel
constexpr int kRegistered      = 1;  // UserRegMode = registered user buffer
constexpr int kNonRegistered   = 2;  // UserRegMode = non-registered user buffer

// Resolve the generated device function-table row for a single kernel key.
// Returns -1 when no such kernel was generated.
int FuncId(int coll, int devRedOp, int type, int algo, int proto, int reg)
{
    return ncclDevFuncId(coll, devRedOp, type, algo, proto,
                         /*acc=*/0, /*pipeline=*/0, reg);
}
} // namespace

class LL128RegVariantKernelTests : public ::testing::Test
{
};

// The core guarantee: each reg-variant collective emits two *distinct* LL128
// kernels (registered vs non-registered), both present in the device table.
TEST_F(LL128RegVariantKernelTests, TwoDistinctLL128KernelsPerRegVariantCollective)
{
    struct Case
    {
        const char* name;
        int         coll;
        int         redop;
        int         type;
        int         algo;
    };

    const Case cases[] = {
        {"AllReduce/RING/Sum/f32", ncclFuncAllReduce, ncclDevSum, ncclFloat32, NCCL_ALGO_RING},
        {"AllReduce/TREE/Sum/f32", ncclFuncAllReduce, ncclDevSum, ncclFloat32, NCCL_ALGO_TREE},
        {"AllGather/RING/Sum/i8",  ncclFuncAllGather, ncclDevSum, ncclInt8,    NCCL_ALGO_RING},
        {"Broadcast/RING/Sum/i8",  ncclFuncBroadcast, ncclDevSum, ncclInt8,    NCCL_ALGO_RING},
    };

    for (const auto& c : cases)
    {
        SCOPED_TRACE(c.name);

        int idReg    = FuncId(c.coll, c.redop, c.type, c.algo, NCCL_PROTO_LL128, kRegistered);
        int idNonReg = FuncId(c.coll, c.redop, c.type, c.algo, NCCL_PROTO_LL128, kNonRegistered);

        // Both variants must have been generated into the device function table.
        EXPECT_GE(idReg, 0)    << "registered (UserRegMode=1) LL128 kernel not generated";
        EXPECT_GE(idNonReg, 0) << "non-registered (UserRegMode=2) LL128 kernel not generated";

        // ...and they must be two different kernels -- the entire point of the split.
        EXPECT_NE(idReg, idNonReg)
            << "registered and non-registered LL128 map to the same kernel id";
    }
}

// Pins the registration -> UserRegMode selection performed at launch in
// enqueue.cc (ncclTasksRegAndEnqueue): a registered user buffer must resolve to
// the reg=1 kernel and a non-registered one to the reg=2 kernel. Because the
// generated ids are only asserted to be *distinct* elsewhere, inverting the
// `1 : 2` mapping there would still pass those tests -- this one catches it.
TEST_F(LL128RegVariantKernelTests, RegModeSelectionResolvesToMatchingKernel)
{
    const int coll  = ncclFuncAllReduce;
    const int redop = ncclDevSum;
    const int type  = ncclFloat32;
    const int algo  = NCCL_ALGO_RING;

    const int idRegistered    = FuncId(coll, redop, type, algo, NCCL_PROTO_LL128, kRegistered);
    const int idNonRegistered = FuncId(coll, redop, type, algo, NCCL_PROTO_LL128, kNonRegistered);
    ASSERT_GE(idRegistered, 0)    << "registered (reg=1) LL128 kernel not generated";
    ASSERT_GE(idNonRegistered, 0) << "non-registered (reg=2) LL128 kernel not generated";
    ASSERT_NE(idRegistered, idNonRegistered);

    // The mapping enqueue.cc applies: a registered IPC/NVLS buffer (regUsed), a
    // registered network buffer (netRegUsed), or both -> reg=1; nothing -> reg=2.
    EXPECT_EQ(ncclDevFuncLL128RegMode(/*regUsed=*/true, /*netRegUsed=*/false), kRegistered);
    EXPECT_EQ(ncclDevFuncLL128RegMode(/*regUsed=*/false, /*netRegUsed=*/true), kRegistered);
    EXPECT_EQ(ncclDevFuncLL128RegMode(/*regUsed=*/true, /*netRegUsed=*/true), kRegistered);
    EXPECT_EQ(ncclDevFuncLL128RegMode(/*regUsed=*/false, /*netRegUsed=*/false), kNonRegistered);

    // End-to-end: the selected mode must resolve to the matching kernel id, so an
    // inverted 1:2 mapping would land on the wrong (registered<->non-registered) id.
    auto resolve = [&](bool regUsed, bool netRegUsed) {
        return FuncId(coll, redop, type, algo, NCCL_PROTO_LL128,
                      ncclDevFuncLL128RegMode(regUsed, netRegUsed));
    };
    EXPECT_EQ(resolve(true, false), idRegistered)     << "registered buffer must select the reg=1 kernel";
    EXPECT_EQ(resolve(false, true), idRegistered)     << "net-registered buffer must select the reg=1 kernel";
    EXPECT_EQ(resolve(false, false), idNonRegistered) << "non-registered buffer must select the reg=2 kernel";
}

// The classifier that gates the split must agree with the generated set.
TEST_F(LL128RegVariantKernelTests, ClassificationMatchesRegVariantSet)
{
    // Reg-variant collectives, only at LL128.
    EXPECT_TRUE(ncclDevFuncIsLL128RegVariant(ncclFuncAllReduce, NCCL_PROTO_LL128));
    EXPECT_TRUE(ncclDevFuncIsLL128RegVariant(ncclFuncAllGather, NCCL_PROTO_LL128));
    EXPECT_TRUE(ncclDevFuncIsLL128RegVariant(ncclFuncBroadcast, NCCL_PROTO_LL128));

    // Same collectives are NOT split at other protocols.
    EXPECT_FALSE(ncclDevFuncIsLL128RegVariant(ncclFuncAllReduce, NCCL_PROTO_SIMPLE));
    EXPECT_FALSE(ncclDevFuncIsLL128RegVariant(ncclFuncAllReduce, NCCL_PROTO_LL));

    // Non-reg-variant collectives are not split even at LL128.
    EXPECT_FALSE(ncclDevFuncIsLL128RegVariant(ncclFuncReduceScatter, NCCL_PROTO_LL128));
    EXPECT_FALSE(ncclDevFuncIsLL128RegVariant(ncclFuncReduce, NCCL_PROTO_LL128));
}

// Negative control: kernels outside the reg-variant set are a single (reg=0)
// kernel, confirming the split is scoped and did not leak into other paths.
TEST_F(LL128RegVariantKernelTests, NonRegVariantEmitsSingleKernel)
{
    // Non-reg-variant collective at LL128: single kernel keyed with reg=0.
    int rsLL128 = FuncId(ncclFuncReduceScatter, ncclDevSum, ncclFloat32,
                         NCCL_ALGO_RING, NCCL_PROTO_LL128, kRegMode0);
    EXPECT_GE(rsLL128, 0) << "ReduceScatter RING LL128 kernel not generated";
    // ...and its reg-variant kernels must NOT exist -- this is what confirms the split is scoped.
    EXPECT_LT(FuncId(ncclFuncReduceScatter, ncclDevSum, ncclFloat32,
                     NCCL_ALGO_RING, NCCL_PROTO_LL128, kRegistered), 0)
        << "ReduceScatter must not have a registered (reg=1) LL128 kernel";
    EXPECT_LT(FuncId(ncclFuncReduceScatter, ncclDevSum, ncclFloat32,
                     NCCL_ALGO_RING, NCCL_PROTO_LL128, kNonRegistered), 0)
        << "ReduceScatter must not have a non-registered (reg=2) LL128 kernel";

    // Reg-variant collective at SIMPLE: also a single kernel keyed with reg=0.
    int arSimple = FuncId(ncclFuncAllReduce, ncclDevSum, ncclFloat32,
                          NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, kRegMode0);
    EXPECT_GE(arSimple, 0) << "AllReduce RING SIMPLE kernel not generated";
}

} // namespace RcclUnitTesting
