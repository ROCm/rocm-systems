/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <string>
#include <vector>
#include <memory>

#include "gtest/gtest.h"
#include "suites/functional/agent_props.h"
#include "suites/functional/debug_basic.h"
#include "suites/functional/memory_basic.h"
#include "suites/functional/memory_access.h"
#include "suites/functional/ipc.h"
#include "suites/functional/memory_alignment.h"
#include "suites/functional/memory_atomics.h"
#include "suites/functional/memory_allocation.h"
#include "suites/functional/memory_fill.h"
#include "suites/functional/deallocation_notifier.h"
#include "suites/functional/virtual_memory.h"
#include "suites/functional/svm_memory.h"
#include "suites/functional/time_stamp.h"
#include "suites/performance/dispatch_time.h"
#include "suites/performance/memory_async_copy.h"
#if ENABLE_COPY_NUMA
#include "suites/performance/memory_async_copy_numa.h"
#endif
#include "suites/performance/memory_async_copy_on_engine.h"
#include "suites/performance/enqueueLatency.h"
#include "suites/performance/agent_preload.h"
#include "suites/negative/memory_allocate_negative_tests.h"
#include "suites/negative/queue_validation.h"
#include "suites/stress/memory_concurrent_tests.h"
#include "suites/stress/queue_write_index_concurrent_tests.h"
#include "suites/test_common/test_case_template.h"
#include "suites/functional/test_fault_example.h"
#include "suites/test_common/main.h"
#include "suites/test_common/test_common.h"
#include "suites/functional/concurrent_init.h"
#include "suites/functional/concurrent_init_shutdown.h"
#include "suites/functional/concurrent_shutdown.h"
#include "suites/functional/reference_count.h"
#include "suites/functional/signal_concurrent.h"
#include "suites/functional/signal_allocation_validation.h"
#include "suites/functional/metadata_prefetch.h"
#include "suites/functional/aql_barrier_bit.h"
#include "suites/functional/signal_kernel.h"
#include "suites/functional/cu_masking.h"
#include "suites/functional/filter_devices.h"
#include "suites/functional/fp_exception_shutdown.h"
#include "suites/functional/gpu_coredump.h"
#include "amd_smi/amdsmi.h"
#include "common/common.h"
#include "suites/functional/counted_queues.h"
#include "suites/functional/queue_create.h"
#include "suites/functional/cuid.h"
#include "common/os.h"
#include "common/platform_filter.h"
#include "common/base_rocr_utils.h"
#include "common/env_config.h"

static RocrTstGlobals *sRocrtstGlvalues = nullptr;

static void SetFlags(TestBase *test) {
  assert(sRocrtstGlvalues != nullptr);

  test->set_num_iteration(sRocrtstGlvalues->num_iterations);
  test->set_verbosity(sRocrtstGlvalues->verbosity);
  test->set_monitor_verbosity(sRocrtstGlvalues->monitor_verbosity);
}

static bool RunCustomTestProlog(TestBase *test) {
  SetFlags(test);

  test->DisplayTestInfo();
  test->SetUp();
  if (test->isTestSkipped()) {
    return false;  // Test was skipped, don't run test method
  }
  test->Run();
  return true;  // Test ran successfully, OK to run test method
}
static void RunCustomTestEpilog(TestBase *test) {
  test->DisplayResults();
  test->Close();
  return;
}

// If the test case one big test, you should use RunGenericTest()
// to run the test case. OTOH, if the test case consists of multiple
// functions to be run as separate tests, follow this pattern:
//   * RunCustomTestProlog(test)  // Run() should contain minimal code
//   * <insert call to actual test function within test case>
//   * RunCustomTestEpilog(test)
static void RunGenericTest(TestBase *test) {
  if (!RunCustomTestProlog(test)) {
    return;  // Test was skipped, don't run epilog
  }
  RunCustomTestEpilog(test);
  return;
}

// TEST ENTRY TEMPLATE:
// TEST(rocrtst, Perf_<test name>) {
//  <Test Implementation class> <test_obj>;
//
//  // Copy and modify implementation of RunGenericTest() if you need to deviate
//  // from the standard pattern implemented there.
//  RunGenericTest(&<test_obj>);
// }

TEST(rocrtst, Test_Example) {
  TestExample tst;

  RunGenericTest(&tst);
}

TEST(rocrtst, Test_Example_InterruptDisabled) {
  TestExample tst;
  rocrtst::SetEnv("HSA_ENABLE_INTERRUPT", "0");
  RunGenericTest(&tst);
}

TEST(rocrtst, Test_MetadataPrefetchPacket) {
  MetadataPrefetch tst;

  RunGenericTest(&tst);
}

TEST(rocrtstFunc, MemoryAccessTests) {
  MemoryAccessTest mt;
  if (!RunCustomTestProlog(&mt)) return;
  mt.CPUAccessToGPUMemoryTest();
  mt.GPUAccessToCPUMemoryTest();
  RunCustomTestEpilog(&mt);
}

TEST(rocrtstFunc, MemoryAccessCoherent) {
  MemoryAccessTest mt;
  if (!RunCustomTestProlog(&mt)) return;
  mt.MemoryAccessCoherentTest();
  RunCustomTestEpilog(&mt);
}

TEST(rocrtstFunc, GroupMemoryAllocationTest) {
  MemoryAllocationTest ma(true, false);
  if (!RunCustomTestProlog(&ma)) return;
  ma.GroupMemoryDynamicAllocation();
  RunCustomTestEpilog(&ma);
}

TEST(rocrtstFunc, MemoryAllocateAndFreeTest) {
  MemoryAllocationTest ma(false, true);
  if (!RunCustomTestProlog(&ma)) return;
  ma.MemoryBasicAllocationAndFree();
  RunCustomTestEpilog(&ma);
}

TEST(rocrtstFunc, MemoryAllocateContiguousTest) {
  MemoryAllocationTest ma(false, true);
  if (!RunCustomTestProlog(&ma)) return;
  ma.MemoryAllocateContiguousTest();
  RunCustomTestEpilog(&ma);
}

TEST(rocrtstFunc, MemoryFillTest) {
  MemoryFill mf;
  if (!RunCustomTestProlog(&mf)) return;
  mf.MemoryFillTest();
  RunCustomTestEpilog(&mf);
}

TEST(rocrtstFunc, Concurrent_Init_Test) {
  ConcurrentInitTest ci;
  if (!RunCustomTestProlog(&ci)) return;
  ci.TestConcurrentInit();
  RunCustomTestEpilog(&ci);
}

TEST(rocrtstFunc, Concurrent_Init_Shutdown_Test) {
  ConcurrentInitShutdownTest ci;
  if (!RunCustomTestProlog(&ci)) return;
  ci.TestConcurrentInitShutdown();
  RunCustomTestEpilog(&ci);
}
TEST(rocrtstFunc, Concurrent_Shutdown) {
  ConcurrentShutdownTest cs;
  if (!RunCustomTestProlog(&cs)) return;
  cs.TestConcurrentShutdown();
  RunCustomTestEpilog(&cs);
}

TEST(rocrtstFunc, Reference_Count) {
  ReferenceCountTest rc(true, false);
  if (!RunCustomTestProlog(&rc)) return;
  rc.TestReferenceCount();
  RunCustomTestEpilog(&rc);
}

TEST(rocrtstFunc, Max_Reference_Count) {
  ReferenceCountTest rc(false, true);
  if (!RunCustomTestProlog(&rc)) return;
  rc.TestMaxReferenceCount();
  RunCustomTestEpilog(&rc);
}

TEST(rocrtstFunc, Signal_Destroy_Concurrently) {
  SignalConcurrentTest sd(true, false, false, false);
  if (!RunCustomTestProlog(&sd)) return;
  sd.TestSignalDestroyConcurrent();
  RunCustomTestEpilog(&sd);
}

TEST(rocrtstFunc, Signal_Max_Consumer) {
  SignalConcurrentTest sd(false, true, false, false);
  if (!RunCustomTestProlog(&sd)) return;
  sd.TestSignalCreateMaxConsumers();
  RunCustomTestEpilog(&sd);
}

TEST(rocrtstFunc, Signal_Create_Concurrently) {
  SignalConcurrentTest sd(false, false, false, true);
  if (!RunCustomTestProlog(&sd)) return;
  sd.TestSignalCreateConcurrent();
  RunCustomTestEpilog(&sd);
}

TEST(rocrtstFunc, Signal_Allocation_Validation) {
  SignalAllocationValidationTest sav;
  RunCustomTestProlog(&sav);
  sav.TestSignalAllocationValidation();
  RunCustomTestEpilog(&sav);
}

/* Temporary: Disable CU Masking until it is fixed */
TEST(rocrtstFunc, DISABLED_CU_Masking) {
  CU_Masking sd;
  RunGenericTest(&sd);
}

TEST(rocrtstFunc, IPC) {
    IPCTest ipc;
    RunGenericTest(&ipc);
}

TEST(rocrtstFunc, DISABLED_Signal_Kernel_Set) {
    SignalKernelTest sk(SET);
    if (!RunCustomTestProlog(&sk)) return;
    sk.TestSignalKernelSet();
    RunCustomTestEpilog(&sk);
}

TEST(rocrtstFunc, DISABLED_Signal_Kernel_Multi_Set) {
    SignalKernelTest sk(MULTISET);
    if (!RunCustomTestProlog(&sk)) return;
    sk.TestSignalKernelMultiSet();
    RunCustomTestEpilog(&sk);
}

TEST(rocrtstFunc, DISABLED_Signal_Kernel_Wait) {
    SignalKernelTest sw(WAIT);
    if (!RunCustomTestProlog(&sw)) return;
    sw.TestSignalKernelWait();
    RunCustomTestEpilog(&sw);
}

TEST(rocrtstFunc, DISABLED_Signal_Kernel_Multi_Wait) {
    SignalKernelTest sw(MULTIWAIT);
    if (!RunCustomTestProlog(&sw)) return;
    sw.TestSignalKernelMultiWait();
    RunCustomTestEpilog(&sw);
}

TEST(rocrtstFunc, DISABLED_Aql_Barrier_Bit_Set) {
    AqlBarrierBitTest ab(true, false);
    if (!RunCustomTestProlog(&ab)) return;
    ab.BarrierBitSet();
    RunCustomTestEpilog(&ab);
}

TEST(rocrtstFunc, DISABLED_Aql_Barrier_Bit_Not_Set) {
    AqlBarrierBitTest ab(false, true);
    if (!RunCustomTestProlog(&ab)) return;
    ab.BarrierBitNotSet();
    RunCustomTestEpilog(&ab);
}

TEST(rocrtstFunc, Memory_Max_Mem) {
    MemoryTest mt;

    if (!RunCustomTestProlog(&mt)) return;
    mt.MaxSingleAllocationTest();
    RunCustomTestEpilog(&mt);
}

TEST(rocrtstFunc, Memory_Available) {
    MemoryTest mt;

    if (!RunCustomTestProlog(&mt)) return;
    mt.MemAvailableTest();
    RunCustomTestEpilog(&mt);
}

TEST(rocrtstFunc, Time_Stamp) {
  TimeStamp ts;
  if (!RunCustomTestProlog(&ts)) return;
  ts.TimeStampTest();
  RunCustomTestEpilog(&ts);
}

TEST(rocrtstFunc, BarrierPkt_TimeStamp) {
    TimeStamp ts;
    RunCustomTestProlog(&ts);
    ts.BarrierPacketTimestampValidationTest();
    RunCustomTestEpilog(&ts);
}

TEST(rocrtstFunc, GpuCoreDump_DefaultPattern) {
    GpuCoreDumpTest gcd;
    if (!RunCustomTestProlog(&gcd)) return;
    gcd.TestDefaultPattern();
    RunCustomTestEpilog(&gcd);
}

TEST(rocrtstFunc, GpuCoreDump_CustomPattern) {
    GpuCoreDumpTest gcd;
    if (!RunCustomTestProlog(&gcd)) return;
    gcd.TestCustomPattern();
    RunCustomTestEpilog(&gcd);
}

TEST(rocrtstFunc, GpuCoreDump_DisableFlag) {
    GpuCoreDumpTest gcd;
    if (!RunCustomTestProlog(&gcd)) return;
    gcd.TestDisableFlag();
    RunCustomTestEpilog(&gcd);
}

TEST(rocrtstFunc, GpuCoreDump_PatternSubstitution) {
    GpuCoreDumpTest gcd;
    if (!RunCustomTestProlog(&gcd)) return;
    gcd.TestPatternSubstitution();
    RunCustomTestEpilog(&gcd);
}

TEST(rocrtstFunc, GpuCoreDump_InvalidPath) {
    GpuCoreDumpTest gcd;
    if (!RunCustomTestProlog(&gcd)) return;
    gcd.TestInvalidPath();
    RunCustomTestEpilog(&gcd);
}

TEST(rocrtstFunc, GpuCoreDump_ContentIntegrity) {
    GpuCoreDumpTest gcd;
    if (!RunCustomTestProlog(&gcd)) return;
    gcd.TestCoreDumpContentIntegrity();
    RunCustomTestEpilog(&gcd);
}

TEST(rocrtstFunc, GpuCoreDump_PipePattern) {
    GpuCoreDumpTest gcd;
    if (!RunCustomTestProlog(&gcd)) return;
    gcd.TestPipePattern();
    RunCustomTestEpilog(&gcd);
}

TEST(rocrtstFunc, FP_Exception_Shutdown) {
    FpExceptionShutdownTest fpx;
    if (!RunCustomTestProlog(&fpx)) return;
    fpx.TestShutdownSurvivesStrictFpEnv();
    RunCustomTestEpilog(&fpx);
}


TEST(rocrtstFunc, Memory_Atomic_Add_Test) {
    MemoryAtomic ma(ADD);
    if (!RunCustomTestProlog(&ma)) return;
    ma.MemoryAtomicTest();
    RunCustomTestEpilog(&ma);
}

TEST(rocrtstFunc, Memory_Atomic_Sub_Test) {
    MemoryAtomic ma(SUB);
    if (!RunCustomTestProlog(&ma)) return;
    ma.MemoryAtomicTest();
    RunCustomTestEpilog(&ma);
}

TEST(rocrtstFunc, Memory_Atomic_And_Test) {
    MemoryAtomic ma(AND);
    if (!RunCustomTestProlog(&ma)) return;
    ma.MemoryAtomicTest();
    RunCustomTestEpilog(&ma);
}

TEST(rocrtstFunc, Memory_Atomic_Or_Test) {
    MemoryAtomic ma(OR);
    if (!RunCustomTestProlog(&ma)) return;
    ma.MemoryAtomicTest();
    RunCustomTestEpilog(&ma);
}

TEST(rocrtstFunc, Memory_Atomic_Xor_Test) {
    MemoryAtomic ma(XOR);
    if (!RunCustomTestProlog(&ma)) return;
    ma.MemoryAtomicTest();
    RunCustomTestEpilog(&ma);
}

TEST(rocrtstFunc, Memory_Atomic_Min_Test) {
    MemoryAtomic ma(MIN);
    if (!RunCustomTestProlog(&ma)) return;
    ma.MemoryAtomicTest();
    RunCustomTestEpilog(&ma);
}

TEST(rocrtstFunc, Memory_Atomic_Max_Test) {
    MemoryAtomic ma(MAX);
    if (!RunCustomTestProlog(&ma)) return;
    ma.MemoryAtomicTest();
    RunCustomTestEpilog(&ma);
}

TEST(rocrtstFunc, Memory_Atomic_Inc_Test) {
    MemoryAtomic ma(INC);
    if (!RunCustomTestProlog(&ma)) return;
    ma.MemoryAtomicTest();
    RunCustomTestEpilog(&ma);
}

TEST(rocrtstFunc, Memory_Atomic_Dec_Test) {
    MemoryAtomic ma(DEC);
    if (!RunCustomTestProlog(&ma)) return;
    ma.MemoryAtomicTest();
    RunCustomTestEpilog(&ma);
}

TEST(rocrtstFunc, Memory_Atomic_Xchg_Test) {
    MemoryAtomic ma(XCHG);
    if (!RunCustomTestProlog(&ma)) return;
    ma.MemoryAtomicTest();
    RunCustomTestEpilog(&ma);
}

TEST(rocrtstFunc, DISABLED_DebugBasicTests) {
    DebugBasicTest mt;
    if (!RunCustomTestProlog(&mt)) return;
    mt.VectorAddDebugTrapTest();
    RunCustomTestEpilog(&mt);
}

TEST(rocrtstFunc, Memory_Alignment_Test) {
    MemoryAlignmentTest ma;
    if (!RunCustomTestProlog(&ma)) return;
    ma.MemoryPoolAlignment();
    RunCustomTestEpilog(&ma);
}

TEST(rocrtstFunc, Deallocation_Notifier_Test) {
    DeallocationNotifierTest notifier;
    RunGenericTest(&notifier);
}

TEST(rocrtstFunc, AgentPropertiesTests) {
    AgentPropTest propTest;
    if (!RunCustomTestProlog(&propTest)) return;
    propTest.QueryAgentUUID();
    propTest.QueryAgentClockCounters();
    RunCustomTestEpilog(&propTest);
}

TEST(rocrtstFunc, SvmMemory_Basic_Test) {
    SvmMemoryTestBasic smt;

    if (!RunCustomTestProlog(&smt)) return;
    smt.TestCreateDestroy();
    smt.TestSVMPrefetch();
    smt.TestSVMBatchDiscard();
    RunCustomTestEpilog(&smt);
}

TEST(rocrtstFunc, SvmMemory_Negative_Test) {
    SvmMemoryTestBasic smt;
    if (!RunCustomTestProlog(&smt)) return;
    smt.TestSVMDiscardNegative();
    RunCustomTestEpilog(&smt);
}

TEST(rocrtstFunc, SvmMemory_AccessedBy_All_Devices_Test) {
    SvmMemoryTestBasic smt;
    if (!RunCustomTestProlog(&smt)) return;
    smt.TestAccessedByAllDevices();
    RunCustomTestEpilog(&smt);
}

TEST(rocrtstFunc, VirtMemory_Basic_Test) {
    VirtMemoryTestBasic vmt;

    if (!RunCustomTestProlog(&vmt)) return;
    vmt.TestCreateDestroy();
    vmt.TestRefCount();
    vmt.TestPartialMapping();
    RunCustomTestEpilog(&vmt);
}

TEST(rocrtstFunc, VirtMemory_Access_Test) {
    VirtMemoryTestBasic vmt;

    if (!RunCustomTestProlog(&vmt)) return;
    vmt.CPUAccessToGPUMemoryTest();
    vmt.GPUAccessToCPUMemoryTest();
    vmt.GPUAccessToGPUMemoryTest();
    vmt.ImportedShareableHandleSetAccessAfterFdClose();
    RunCustomTestEpilog(&vmt);
}

TEST(rocrtstFunc, VirtMemory_Accounting_Test) {
    VirtMemoryTestBasic vmt;

    if (!RunCustomTestProlog(&vmt)) return;
    vmt.MemoryAccountingTest();
    RunCustomTestEpilog(&vmt);
}

TEST(rocrtstFunc, VirtMemory_Aliasing_Test) {
    VirtMemoryTestBasic vmt;

    if (!RunCustomTestProlog(&vmt)) return;
    vmt.TestVirtAddressAlias();
    RunCustomTestEpilog(&vmt);
}

TEST(rocrtstFunc, VirtMemory_NonContiguousChunks_Test) {
  VirtMemoryTestBasic vmt;

  if (!RunCustomTestProlog(&vmt)) return;
  vmt.NonContiguousChunks();
  RunCustomTestEpilog(&vmt);
}

TEST(rocrtstFunc, VirtMemory_GPUtoHostAccess_Test) {
  VirtMemoryTestBasic vmt;

  if (!RunCustomTestProlog(&vmt)) return;
  vmt.TestGpuAccessToHostMemoryAllocation();
  RunCustomTestEpilog(&vmt);
}

TEST(rocrtstFunc, VirtMemory_Interprocess_DevicePool_Test) {
    VirtMemoryTestInterProcess vmt(PoolType::kDevicePool);
    if (!RunCustomTestProlog(&vmt)) return;
    RunCustomTestEpilog(&vmt);
}

TEST(rocrtstFunc, VirtMemory_Interprocess_HostPool_Test) {
    VirtMemoryTestInterProcess vmt(PoolType::kCpuPool);
    if (!RunCustomTestProlog(&vmt)) return;
    RunCustomTestEpilog(&vmt);
}

TEST(rocrtstFunc, Filter_Devices_Test) {
    FilterDevicesTest fd;
    if (!RunCustomTestProlog(&fd)) return;
    fd.TestRocrVisibleDevicesFiltering();
    RunCustomTestEpilog(&fd);
}

TEST(rocrtstFunc, Counted_Queue_Basic_Test) {
  CountedQueuesTest cq;
  if (!RunCustomTestProlog(&cq)) return;
  cq.CountedQueueBasicApiTest();
  RunCustomTestEpilog(&cq);
}

TEST(rocrtstFunc, Counted_Queue_Same_Priority_Max_Limit_Test) {
  CountedQueuesTest cq;
  if (!RunCustomTestProlog(&cq)) return;
  cq.CountedQueues_SamePriority_MaxLimitTest();
  RunCustomTestEpilog(&cq);
}

TEST(rocrtstFunc, Counted_Queue_Invalid_Args_Test) {
  CountedQueuesTest cq;
  if (!RunCustomTestProlog(&cq)) return;
  cq.InvalidArgsTest();
  RunCustomTestEpilog(&cq);
}

TEST(rocrtstFunc, Counted_Queue_Multiple_Priorities_Limit_Test) {
  CountedQueuesTest cq;
  if (!RunCustomTestProlog(&cq)) return;
  cq.CountedQueuesAllPrioritiesLimitTest();
  RunCustomTestEpilog(&cq);
}

TEST(rocrtstFunc, Counted_Queue_Set_Priority_Nack_Test) {
  CountedQueuesTest cq;
  if (!RunCustomTestProlog(&cq)) return;
  cq.CountedQueuesSetPriorityNackTest();
  RunCustomTestEpilog(&cq);
}

TEST(rocrtstFunc, Counted_Queue_Set_CUMask_Nack_Test) {
  CountedQueuesTest cq;
  if (!RunCustomTestProlog(&cq)) return;
  cq.CountedQueuesSetCUMaskNackTest();
  RunCustomTestEpilog(&cq);
}

TEST(rocrtstFunc, Counted_Queue_Dispatch_Test) {
  CountedQueuesTest cq;
  if (!RunCustomTestProlog(&cq)) return;
  cq.CountedQueuesDispatchTest();
  RunCustomTestEpilog(&cq);
}

TEST(rocrtstFunc, Counted_Queue_Multithreaded_Dispatch_Test) {
  CountedQueuesTest cq;
  if (!RunCustomTestProlog(&cq)) return;
  cq.CountedQueuesMultithreadedDispatchTest();
  RunCustomTestEpilog(&cq);
}

TEST(rocrtstFunc, Counted_Queue_Overflow_And_Wraparound_Test) {
  CountedQueuesTest cq;
  if (!RunCustomTestProlog(&cq)) return;
  cq.CountedQueuesOverflowWrapAroundTest();
  RunCustomTestEpilog(&cq);
}

TEST(rocrtstFunc, Queue_Create_SystemMem_Test) {
  QueueCreateTest qt;
  if (!RunCustomTestProlog(&qt)) return;
  qt.SystemMemQueueTest();
  RunCustomTestEpilog(&qt);
}

TEST(rocrtstFunc, Queue_Create_DeviceMem_RingBuf_Test) {
  QueueCreateTest qt;
  if (!RunCustomTestProlog(&qt)) return;
  qt.DeviceMemRingBufQueueTest();
  RunCustomTestEpilog(&qt);
}

TEST(rocrtstFunc, Queue_Create_Batch_Test) {
  QueueCreateTest qt;
  if (!RunCustomTestProlog(&qt)) return;
  qt.BatchQueueCreateTest();
  RunCustomTestEpilog(&qt);
}

TEST(rocrtstFunc, Queue_Create_SDMA_Create_Destroy_Test) {
  QueueCreateTest qt;
  if (!RunCustomTestProlog(&qt)) return;
  qt.SdmaQueueCreateDestroyTest();
  RunCustomTestEpilog(&qt);
}

TEST(rocrtstFunc, Queue_Create_Invalid_Args_Test) {
  QueueCreateTest qt;
  if (!RunCustomTestProlog(&qt)) return;
  qt.InvalidArgsTest();
  RunCustomTestEpilog(&qt);
}

#ifdef HSA_ENABLE_AMDCUID_SUPPORT
TEST(rocrtstFunc, Cuid_GPU_Validation_Test) {
  CuidTest ct;
  if (!RunCustomTestProlog(&ct)) return;
  ct.ValidateGpuCuidTest();
  RunCustomTestEpilog(&ct);
}
#endif

TEST(rocrtstNeg, Memory_Negative_Tests) {
    MemoryAllocateNegativeTest mt;
    if (!RunCustomTestProlog(&mt)) return;
    mt.ZeroMemoryAllocateTest();
    mt.MaxMemoryAllocateTest();

    // Disabled temporarily - Renable this test only
    // on recent GPUs - gfx94x+
    // mt.FreeQueueRingBufferTest();

    RunCustomTestEpilog(&mt);
}

TEST(rocrtstNeg, Queue_Validation_InvalidDimension) {
    QueueValidation qv(true, false, false, false, false);
    if (!RunCustomTestProlog(&qv)) return;
    qv.QueueValidationForInvalidDimension();
    RunCustomTestEpilog(&qv);
}

TEST(rocrtstNeg, Queue_Validation_InvalidGroupMemory) {
    QueueValidation qv(false, true, false, false, false);
    if (!RunCustomTestProlog(&qv)) return;
    qv.QueueValidationInvalidGroupMemory();
    RunCustomTestEpilog(&qv);
}

TEST(rocrtstNeg, Queue_Validation_InvalidKernelObject) {
    QueueValidation qv(false, false, true, false, false);
    if (!RunCustomTestProlog(&qv)) return;
    qv.QueueValidationForInvalidKernelObject();
    RunCustomTestEpilog(&qv);
}

TEST(rocrtstNeg, Queue_Validation_InvalidPacket) {
    QueueValidation qv(false, false, false, true, false);
    if (!RunCustomTestProlog(&qv)) return;
    qv.QueueValidationForInvalidPacket();
    RunCustomTestEpilog(&qv);
}

TEST(rocrtstNeg, DISABLED_Queue_Validation_InvalidWorkGroupSize) {
    QueueValidation qv(false, false, false, false, true);
    if (!RunCustomTestProlog(&qv)) return;
    qv.QueueValidationForInvalidWorkGroupSize();
    RunCustomTestEpilog(&qv);
}

TEST(rocrtstStress, Memory_Concurrent_Allocate_Test) {
    MemoryConcurrentTest mt(true, false, false);
    if (!RunCustomTestProlog(&mt)) return;
    mt.MemoryConcurrentAllocate();
    RunCustomTestEpilog(&mt);
}

TEST(rocrtstStress, Memory_Concurrent_Free_Test) {
    MemoryConcurrentTest mt(false, true, false);
    if (!RunCustomTestProlog(&mt)) return;
    mt.MemoryConcurrentFree();
    RunCustomTestEpilog(&mt);
}

TEST(rocrtstStress, Memory_Concurrent_Pool_Info_Test) {
    MemoryConcurrentTest mt(false, false, true);
    if (!RunCustomTestProlog(&mt)) return;
    mt.MemoryConcurrentPoolGetInfo();
    RunCustomTestEpilog(&mt);
}

TEST(rocrtstStress, Queue_Add_Write_Index_ConcurrentTest) {
    QueueWriteIndexConcurrentTest Qw(true, false, false);
    if (!RunCustomTestProlog(&Qw)) return;
    Qw.QueueAddWriteIndexAtomic();
    RunCustomTestEpilog(&Qw);
}

TEST(rocrtstStress, Queue_CAS_Write_Index_ConcurrentTest) {
    QueueWriteIndexConcurrentTest Qw(false, true, false);
    if (!RunCustomTestProlog(&Qw)) return;
    Qw.QueueCasWriteIndexAtomic();
    RunCustomTestEpilog(&Qw);
}

TEST(rocrtstStress, Queue_LoadStore_Write_Index_ConcurrentTest) {
    QueueWriteIndexConcurrentTest Qw(false, false, true);
    if (!RunCustomTestProlog(&Qw)) return;
    Qw.QueueLoadStoreWriteIndexAtomic();
    RunCustomTestEpilog(&Qw);
}

TEST(rocrtstPerf, Memory_Async_Copy) {
  MemoryAsyncCopy mac;
  // To do full test, uncomment this:
  //  mac.set_full_test(true);
  // To test only 1 path, add lines like this:
  //  mac.set_src_pool(<src pool id>);
  //  mac.set_dst_pool(<dst pool id>);
  // The default is to and from the cpu to 1 gpu, and to/from a gpu to
  // another gpu
  RunGenericTest(&mac);
}

TEST(rocrtstPerf, Memory_Async_Copy_On_Engine) {
    MemoryAsyncCopyOnEngine mac;
    RunGenericTest(&mac);
}

TEST(rocrtstPerf, ENQUEUE_LATENCY) {
  EnqueueLatency singlePacketequeue(true);
  EnqueueLatency multiPacketequeue(false);
  RunGenericTest(&singlePacketequeue);
  RunGenericTest(&multiPacketequeue);
}

#if ENABLE_COPY_NUMA
TEST(rocrtstPerf, DISABLED_Memory_Async_Copy_NUMA) {
  MemoryAsyncCopyNUMA numa;
  RunGenericTest(&numa);
}
#endif

TEST(rocrtstPerf, AQL_Dispatch_Time_Single_SpinWait) {
  DispatchTime dt(true, true);
  RunGenericTest(&dt);
}

TEST(rocrtstPerf, AQL_Dispatch_Time_Single_Interrupt) {
  DispatchTime dt(false, true);
  RunGenericTest(&dt);
}

TEST(rocrtstPerf, AQL_Dispatch_Time_Multi_SpinWait) {
  DispatchTime dt(true, false);
  RunGenericTest(&dt);
}

TEST(rocrtstPerf, AQL_Dispatch_Time_Multi_Interrupt) {
  DispatchTime dt(false, false);
  RunGenericTest(&dt);
}

TEST(rocrtstPerf, Agent_Preload_Latency) {
  AgentPreloadTest apt;
  RunGenericTest(&apt);
}


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Initialize environment configuration (must be done first)
  rocrtst::EnvironmentConfig::getInstance().initialize();

  // Initialize platform-aware test filtering
  rocrtst::TestFilterManager& filterMgr =
      rocrtst::TestFilterManager::getInstance();

  // Locate config file (matches hsaco file search pattern)
  std::string configPath = rocrtst::LocateConfigFile();
  filterMgr.initialize(configPath);

  // Display platform information
  rocrtst::PlatformType platform = filterMgr.getPlatform();
  std::cout << "========================================================\n";
  std::cout << "ROC Runtime Test Suite\n";
  std::cout << "Platform detected: "
            << rocrtst::PlatformDetector::platformName(platform) << '\n';
  std::cout << "Configuration: " << filterMgr.getConfigPath() << '\n';

  std::vector<std::string> activeGroups = filterMgr.getActiveGroups();
  if (!activeGroups.empty()) {
    std::cout << "Active groups: ";
    for (size_t i = 0; i < activeGroups.size(); ++i) {
      std::cout << activeGroups[i];
      if (i < activeGroups.size() - 1) std::cout << ", ";
    }
    std::cout << '\n';
  }
  std::cout << "========================================================\n";

  if (rocrtst::isEmuModeEnabled()) {
    std::cout << "--- Emulation build ---" << std::endl;
  }

  RocrTstGlobals settings;

  // Set some default values
  settings.verbosity = 1;
  settings.monitor_verbosity = 0;
  settings.num_iterations = 5;

  if (ProcessCmdline(&settings, argc, argv)) {
    return 1;
  }
  sRocrtstGlvalues = &settings;

  if (settings.monitor_verbosity > 0) {
    amdsmi_status_t amdsmi_ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    if (amdsmi_ret != AMDSMI_STATUS_SUCCESS) {
      std::cout << "Failed to initialize AMD smi" << std::endl;
      return 1;
    }
    DumpMonitorInfo();
  }
  return RUN_ALL_TESTS();
}
