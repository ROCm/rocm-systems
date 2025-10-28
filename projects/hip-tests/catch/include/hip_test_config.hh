/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_TEST_CONFIG_HH
#define HIP_TEST_CONFIG_HH

/**
 * @file hip_test_config.hh
 * @brief Centralized test parameter configuration for HIP tests
 * 
 * Test Levels (set via CMake TEST_LEVEL):
 *   - quick: Faster execution with reduced parameters for development/iteration
 *   - full:  Comprehensive testing with full parameters (default)
 * 
 * For each new change that needs constant, add constant with below naming format in this file.
 * TEST_<CATEGORY>_<TESTFILE>_<PARAMNAME>
 */

// =============================================================================
// GRAPH TEST PARAMETERS
// =============================================================================

#ifdef QUICK_TESTS
  #define TEST_GRAPH_HIPGRAPH_LAUNCH_ITERATIONS 5
  #define TEST_GRAPH_LOOP_SIZE 5
  #define TEST_GRAPH_EVENT_RECORD_ITERATIONS 5
  #define TEST_GRAPH_EVENT_WAIT_ITERATIONS 5
  #define TEST_GRAPH_DEVICE_GET_GRAPH_MEM_ATTR_ELEMENT_COUNT (100 * 1024)
  #define TEST_GRAPH_ADD_MEM_ALLOC_NODE_ELEMENT_COUNT (100 * 1024)
  #define TEST_GRAPH_ADD_NODE_BEGIN_CAPTURE_SIZE (100 * 1024)
  #define TEST_GRAPH_CLONE_NUM_THREADS 3
  #define TEST_GRAPH_DESTROY_NODE_NUM_OF_DUMMY_NODES 3
  #define TEST_GRAPH_INSTANTIATE_NUM_OF_INSTANCES 5
  #define TEST_GRAPH_CYCLE_N_TEST4 (1024 * 100)
  #define TEST_GRAPH_CYCLE_N_TEST5 (100 * 1024)
  #define TEST_GRAPH_GET_NODES_N 10000
  #define TEST_GRAPH_GET_ROOT_NODES_NUM_OF_DUMMY_NODES 3
  #define TEST_GRAPH_INSTANTIATE_WITH_FLAGS_N 10000
  #define TEST_GRAPH_INSTANTIATE_WITH_FLAGS_SIZE (100 * 1024)
  #define TEST_GRAPH_INSTANTIATE_WITH_FLAGS_SIZE_2 (100 * 1024)
  #define TEST_GRAPH_INSTANTIATE_WITH_FLAGS_NBYTES (100 * 1024)
  #define TEST_GRAPH_INSTANTIATE_WITH_FLAGS_LOOP 3
  #define TEST_GRAPH_INSTANTIATE_WITH_FLAGS_LAUNCH 3
  #define TEST_GRAPH_INSTANTIATE_WITH_PARAMS_N 10000
  #define TEST_GRAPH_MEM_ALLOC_NODE_GET_PARAMS_N (100 * 1024)
  #define TEST_GRAPH_CLONE_COMPLX_LOOP 5
  #define TEST_GRAPH_SIMPLE_GRAPH_WITH_KERNEL_N (100 * 1024)
  #define TEST_GRAPH_SIMPLE_GRAPH_WITH_KERNEL_NSTEP 3
  #define TEST_GRAPH_SIMPLE_GRAPH_WITH_KERNEL_NKERNEL 3
  #define TEST_GRAPH_STREAM_BEGIN_CAPTURE_N 10000
  #define TEST_GRAPH_STREAM_BEGIN_CAPTURE_OLD_N 100000
  #define TEST_GRAPH_STREAM_BEGIN_CAPTURE_OLD_LAUNCH_ITERS 5
  #define TEST_GRAPH_STREAM_END_CAPTURE_N 10000
  #define TEST_GRAPH_STREAM_GET_CAPTURE_INFO_N 10000
  #define TEST_GRAPH_STREAM_IS_CAPTURING_N 10000
  #define TEST_GRAPH_STREAM_UPDATE_CAPTURE_DEPENDENCIES_N 10000
  #define TEST_GRAPH_STREAM_CAPTURE_COMMON_KLAUNCH_ITERS 3
#else
  #define TEST_GRAPH_HIPGRAPH_LAUNCH_ITERATIONS 1000
  #define TEST_GRAPH_LOOP_SIZE 50
  #define TEST_GRAPH_EVENT_RECORD_ITERATIONS 100
  #define TEST_GRAPH_EVENT_WAIT_ITERATIONS 100
  #define TEST_GRAPH_DEVICE_GET_GRAPH_MEM_ATTR_ELEMENT_COUNT (64 * 1024 * 1024)
  #define TEST_GRAPH_ADD_MEM_ALLOC_NODE_ELEMENT_COUNT (512 * 1024 * 1024)
  #define TEST_GRAPH_ADD_NODE_BEGIN_CAPTURE_SIZE (1024 * 1024)
  #define TEST_GRAPH_CLONE_NUM_THREADS 10
  #define TEST_GRAPH_DESTROY_NODE_NUM_OF_DUMMY_NODES 8
  #define TEST_GRAPH_INSTANTIATE_NUM_OF_INSTANCES 10
  #define TEST_GRAPH_CYCLE_N_TEST4 (1024 * 1024)
  #define TEST_GRAPH_CYCLE_N_TEST5 (1024 * 1024)
  #define TEST_GRAPH_GET_NODES_N 1000000
  #define TEST_GRAPH_GET_ROOT_NODES_NUM_OF_DUMMY_NODES 8
  #define TEST_GRAPH_INSTANTIATE_WITH_FLAGS_N 1000000
  #define TEST_GRAPH_INSTANTIATE_WITH_FLAGS_SIZE (1024 * 1024)
  #define TEST_GRAPH_INSTANTIATE_WITH_FLAGS_SIZE_2 (512 * 1024 * 1024)
  #define TEST_GRAPH_INSTANTIATE_WITH_FLAGS_NBYTES (1024 * 1024 * 1024)
  #define TEST_GRAPH_INSTANTIATE_WITH_FLAGS_LOOP 100
  #define TEST_GRAPH_INSTANTIATE_WITH_FLAGS_LAUNCH 10
  #define TEST_GRAPH_INSTANTIATE_WITH_PARAMS_N 1000000
  #define TEST_GRAPH_MEM_ALLOC_NODE_GET_PARAMS_N (1024 * 1024)
  #define TEST_GRAPH_CLONE_COMPLX_LOOP 100
  #define TEST_GRAPH_SIMPLE_GRAPH_WITH_KERNEL_N (1024 * 1024)
  #define TEST_GRAPH_SIMPLE_GRAPH_WITH_KERNEL_NSTEP 1000
  #define TEST_GRAPH_SIMPLE_GRAPH_WITH_KERNEL_NKERNEL 25
  #define TEST_GRAPH_STREAM_BEGIN_CAPTURE_N 1000000
  #define TEST_GRAPH_STREAM_BEGIN_CAPTURE_OLD_N 1000000
  #define TEST_GRAPH_STREAM_BEGIN_CAPTURE_OLD_LAUNCH_ITERS 50
  #define TEST_GRAPH_STREAM_END_CAPTURE_N 1000000
  #define TEST_GRAPH_STREAM_GET_CAPTURE_INFO_N 1000000
  #define TEST_GRAPH_STREAM_IS_CAPTURING_N 1000000
  #define TEST_GRAPH_STREAM_UPDATE_CAPTURE_DEPENDENCIES_N 1000000
  #define TEST_GRAPH_STREAM_CAPTURE_COMMON_KLAUNCH_ITERS 10
#endif

// =============================================================================
// MEMORY TEST PARAMETERS
// =============================================================================

#ifdef QUICK_TESTS
  // hipMemset.cc parameters
  #define TEST_MEMORY_MEMSET_LOOP_ITERATIONS 8
  #define TEST_MEMORY_MEMSET_USE_OFFSET_LOOP 0  // 0 = no loop, 1 = use loop

  // hipHostRegister.cc parameters
  #define TEST_MEMORY_HOST_REGISTER_OFFSET 8
  #define TEST_MEMORY_HOST_REGISTER_ITERATION 10
  #define TEST_MEMORY_HOST_REGISTER_LOOP_MULTIPLIER 16  // i*16 instead of i
  
  // hipHostRegister_exe.cc parameters  
  #define TEST_MEMORY_HOST_REGISTER_EXE_ITERATION 10
  #define TEST_MEMORY_HOST_REGISTER_EXE_SIZE (1024*1024)
  
  // hipMallocConcurrency.cc parameters
  #define TEST_MEMORY_MALLOC_CONCURRENCY_BUFF_SIZE_BC (5 * 100 * 1024)
  #define TEST_MEMORY_MALLOC_CONCURRENCY_MAX_ALLOC_FREE_SMALL_CHUNKS (5000 / NumDiv)
  #define TEST_MEMORY_MALLOC_CONCURRENCY_MAX_ALLOC_FREE_BIG_CHUNKS 10
  #define TEST_MEMORY_MALLOC_CONCURRENCY_MAX_ALLOC_POOL_ITER (2000 / NumDiv)
  #define TEST_MEMORY_MALLOC_CONCURRENCY_N (4 * 100 * 1024)
  
  // hipMallocFromPoolAsync.cc parameters
  #define TEST_MEMORY_MALLOC_FROM_POOL_ASYNC_N (1 << 12)
  
  // hipMallocManaged_MultiScenario.cc parameters
  #define TEST_MEMORY_MALLOC_MANAGED_MULTI_SCENARIO_NUM_ELMS (100 * 1024)
  
  // hipMemPoolApi.cc parameters
  #define TEST_MEMORY_MEM_POOL_API_NUM_ELEMENTS (8 * 100 * 1024)
  
  // hipMemPoolTrimTo.cc parameters
  #define TEST_MEMORY_MEM_POOL_TRIM_TO_N (1 << 12)
  
  // hipMemcpy2DAsync_old.cc and hipMemcpy2D_old.cc parameters
  #define TEST_MEMORY_MEMCPY2D_OLD_INPUT (1 << 12)
  
  // hipMemcpyDeviceToDeviceNoCU.cc parameters
  #define TEST_MEMORY_MEMCPY_DEVICE_TO_DEVICE_NO_CU_N (1 << 12)
  
  // hipMemcpyWithStreamMultiThread.cc parameters
  #define TEST_MEMORY_MEMCPY_WITH_STREAM_MULTI_THREAD_THREADCOUNT 4
  
  // hipMemcpy_EdgeCases.cc parameters
  #define TEST_MEMORY_MEMCPY_EDGE_CASES_SIZE (100 * 1024)
  
  // hipMemset2DAsyncMultiThreadAndKernel.cc parameters
  #define TEST_MEMORY_MEMSET2D_ASYNC_MULTI_THREAD_NUM_THREADS 10
  #define TEST_MEMORY_MEMSET2D_ASYNC_MULTI_THREAD_ITER 5
  
  // hipMemset3DRegressMultiThread.cc parameters
  #define TEST_MEMORY_MEMSET3D_REGRESS_MULTI_THREAD_MAX_THREADS 4
  
  // hipMemsetAsyncAndKernel.cc parameters
  #define TEST_MEMORY_MEMSET_ASYNC_AND_KERNEL_ITER 3
  #define TEST_MEMORY_MEMSET_ASYNC_AND_KERNEL_N (100 * 1024)
  
  // hipMemsetD32.cc parameters
  #define TEST_MEMORY_MEMSET_D32_BUFFER_NELEMS { 1024, 1024 * 8, 1024 * 32 }
  #define TEST_MEMORY_MEMSET_D32_BUFFER_NELEMS_SIZE 3
  
  // hipMemset2D.cc parameters (tableItems)
  #define TEST_MEMORY_MEMSET2D_TABLE_ITEMS \
    std::make_tuple(20, 20, 20, 20), \
    std::make_tuple(10, 10, 4, 4), \
    std::make_tuple(100, 100, 20, 40), \
    std::make_tuple(100, 100, 0, 20)
  
  // memcpy2d_tests_common.hh parameters (used by multiple tests)
  #define TEST_MEMORY_MEMCPY2D_COMMON_COLS 7
  #define TEST_MEMORY_MEMCPY2D_COMMON_ROWS 8
  
  // mempool_common.hh parameters
  #define TEST_MEMORY_MEMPOOL_COMMON_LAUNCH_ITERATIONS 3
  #define TEST_MEMORY_MEMPOOL_COMMON_NUMBER_OF_THREADS 3
#else
  // hipMemset.cc parameters (full test mode)
  #define TEST_MEMORY_MEMSET_LOOP_ITERATIONS 256
  #define TEST_MEMORY_MEMSET_USE_OFFSET_LOOP 1  // 0 = no loop, 1 = use loop
  
  // hipHostRegister.cc parameters
  #define TEST_MEMORY_HOST_REGISTER_OFFSET 128
  #define TEST_MEMORY_HOST_REGISTER_ITERATION 100
  #define TEST_MEMORY_HOST_REGISTER_LOOP_MULTIPLIER 1  // i*1 (just i)
  
  // hipHostRegister_exe.cc parameters
  #define TEST_MEMORY_HOST_REGISTER_EXE_ITERATION 1000
  #define TEST_MEMORY_HOST_REGISTER_EXE_SIZE (64 * 1024 * 1024)
  
  // hipMallocConcurrency.cc parameters
  #define TEST_MEMORY_MALLOC_CONCURRENCY_BUFF_SIZE_BC (5 * 1024 * 1024)
  #define TEST_MEMORY_MALLOC_CONCURRENCY_MAX_ALLOC_FREE_SMALL_CHUNKS (5000000 / NumDiv)
  #define TEST_MEMORY_MALLOC_CONCURRENCY_MAX_ALLOC_FREE_BIG_CHUNKS 10000
  #define TEST_MEMORY_MALLOC_CONCURRENCY_MAX_ALLOC_POOL_ITER (2000000 / NumDiv)
  #define TEST_MEMORY_MALLOC_CONCURRENCY_N (4 * 1024 * 1024)
  
  // hipMallocFromPoolAsync.cc parameters
  #define TEST_MEMORY_MALLOC_FROM_POOL_ASYNC_N (1 << 20)
  
  // hipMallocManaged_MultiScenario.cc parameters
  #define TEST_MEMORY_MALLOC_MANAGED_MULTI_SCENARIO_NUM_ELMS (100 * 1024)
  
  // hipMemPoolApi.cc parameters
  #define TEST_MEMORY_MEM_POOL_API_NUM_ELEMENTS (8 * 100 * 1024)
  
  // hipMemPoolTrimTo.cc parameters
  #define TEST_MEMORY_MEM_POOL_TRIM_TO_N (1 << 20)
  
  // hipMemcpy2DAsync_old.cc and hipMemcpy2D_old.cc parameters
  #define TEST_MEMORY_MEMCPY2D_OLD_INPUT (1 << 20)
  
  // hipMemcpyDeviceToDeviceNoCU.cc parameters
  #define TEST_MEMORY_MEMCPY_DEVICE_TO_DEVICE_NO_CU_N (1 << 18)
  
  // hipMemcpyWithStreamMultiThread.cc parameters
  #define TEST_MEMORY_MEMCPY_WITH_STREAM_MULTI_THREAD_THREADCOUNT 10
  
  // hipMemcpy_EdgeCases.cc parameters
  #define TEST_MEMORY_MEMCPY_EDGE_CASES_SIZE (1024 * 1024)
  
  // hipMemset2DAsyncMultiThreadAndKernel.cc parameters
  #define TEST_MEMORY_MEMSET2D_ASYNC_MULTI_THREAD_NUM_THREADS 1000
  #define TEST_MEMORY_MEMSET2D_ASYNC_MULTI_THREAD_ITER 10
  
  // hipMemset3DRegressMultiThread.cc parameters
  #define TEST_MEMORY_MEMSET3D_REGRESS_MULTI_THREAD_MAX_THREADS 10
  
  // hipMemsetAsyncAndKernel.cc parameters
  #define TEST_MEMORY_MEMSET_ASYNC_AND_KERNEL_ITER 6
  #define TEST_MEMORY_MEMSET_ASYNC_AND_KERNEL_N (1024 * 1024)
  
  // hipMemsetD32.cc parameters
  #define TEST_MEMORY_MEMSET_D32_BUFFER_NELEMS { 4096, 4096 * 8, 4096 * 32, 4096 * 128, 4096 * 256 }
  #define TEST_MEMORY_MEMSET_D32_BUFFER_NELEMS_SIZE 5
  
  // hipMemset2D.cc parameters (tableItems)
  #define TEST_MEMORY_MEMSET2D_TABLE_ITEMS \
    std::make_tuple(20, 20, 20, 20),   std::make_tuple(10, 10, 4, 4), \
    std::make_tuple(100, 100, 20, 40), std::make_tuple(256, 256, 39, 19), \
    std::make_tuple(100, 100, 20, 0),  std::make_tuple(100, 100, 0, 20), \
    std::make_tuple(100, 100, 0, 0)
  
  // memcpy2d_tests_common.hh parameters (used by multiple tests)
  #define TEST_MEMORY_MEMCPY2D_COMMON_COLS 127
  #define TEST_MEMORY_MEMCPY2D_COMMON_ROWS 128
  
  // mempool_common.hh parameters
  #define TEST_MEMORY_MEMPOOL_COMMON_LAUNCH_ITERATIONS 5
  #define TEST_MEMORY_MEMPOOL_COMMON_NUMBER_OF_THREADS 5
#endif

// =============================================================================
// KERNEL TEST PARAMETERS
// =============================================================================

#ifdef QUICK_TESTS
  // hipGridLaunch.cc parameters
  #define TEST_KERNEL_GRID_LAUNCH_N (100 * 1024)
  
  // hipMemFaultStackAllocation.cc parameters
  #define TEST_KERNEL_MEM_FAULT_STACK_ALLOCATION_N 1024
#else
  // hipGridLaunch.cc parameters
  #define TEST_KERNEL_GRID_LAUNCH_N (4 * 1024 * 1024)
  
  // hipMemFaultStackAllocation.cc parameters
  #define TEST_KERNEL_MEM_FAULT_STACK_ALLOCATION_N 100000
#endif

// =============================================================================
// EVENT TEST PARAMETERS
// =============================================================================

#ifdef QUICK_TESTS
  // Unit_hipEvent.cc parameters
  #define TEST_EVENT_UNIT_HIP_EVENT_ITERATIONS 1000
  
  // Unit_hipEventMGpuMThreads.cc parameters
  #define TEST_EVENT_UNIT_HIP_EVENT_MGPU_MTHREADS_WIDTH 256
  
  // hipEventCreateWithFlags.cc parameters
  #define TEST_EVENT_CREATE_WITH_FLAGS_BUFFER_SIZE (100 * 1024)
#else
  // Unit_hipEvent.cc parameters
  #define TEST_EVENT_UNIT_HIP_EVENT_ITERATIONS 10000000
  
  // Unit_hipEventMGpuMThreads.cc parameters
  #define TEST_EVENT_UNIT_HIP_EVENT_MGPU_MTHREADS_WIDTH 1024
  
  // hipEventCreateWithFlags.cc parameters
  #define TEST_EVENT_CREATE_WITH_FLAGS_BUFFER_SIZE (1024 * 1024)
#endif

// =============================================================================
// STREAM TEST PARAMETERS
// =============================================================================

#ifdef QUICK_TESTS
  // hipAPIStreamDisable.cc parameters
  #define TEST_STREAM_API_STREAM_DISABLE_NUM_STREAMS 4
  #define TEST_STREAM_API_STREAM_DISABLE_NN 10000
  
  // hipLaunchHostFunc.cc parameters
  #define TEST_STREAM_LAUNCH_HOST_FUNC_GRAPH_LAUNCH_ITERATIONS 10
  #define TEST_STREAM_LAUNCH_HOST_FUNC_NN 10000
  
  // hipMultiStream.cc parameters
  #define TEST_STREAM_MULTI_STREAM_N_LOOPS 50
  
  // hipStreamCreateWithPriority.cc parameters
  #define TEST_STREAM_CREATE_WITH_PRIORITY_MEMCPYSIZE1 (100 * 1024)
  #define TEST_STREAM_CREATE_WITH_PRIORITY_MEMCPYSIZE2 (100 * 1024)
  #define TEST_STREAM_CREATE_WITH_PRIORITY_TOTALTHREADS 4
  
  // hipStreamGetDevice.cc parameters
  #define TEST_STREAM_GET_DEVICE_NUMBER_OF_THREADS 4
  
  // hipStreamLegacy_Ext.cc parameters
  #define TEST_STREAM_LEGACY_EXT_N (100 * 1024)
  #define TEST_STREAM_LEGACY_EXT_NUMBER_OF_THREADS 4
  
  // hipStreamWithCUMask.cc parameters
  #define TEST_STREAM_WITH_CU_MASK_N_MULTIPLIER 1
#else
  // hipAPIStreamDisable.cc parameters
  #define TEST_STREAM_API_STREAM_DISABLE_NUM_STREAMS 8
  #define TEST_STREAM_API_STREAM_DISABLE_NN (1 << 21)
  
  // hipLaunchHostFunc.cc parameters
  #define TEST_STREAM_LAUNCH_HOST_FUNC_GRAPH_LAUNCH_ITERATIONS 1000
  #define TEST_STREAM_LAUNCH_HOST_FUNC_NN (1 << 21)
  
  // hipMultiStream.cc parameters
  #define TEST_STREAM_MULTI_STREAM_N_LOOPS 50000
  
  // hipStreamCreateWithPriority.cc parameters
  #define TEST_STREAM_CREATE_WITH_PRIORITY_MEMCPYSIZE1 (64 * 1024 * 1024)
  #define TEST_STREAM_CREATE_WITH_PRIORITY_MEMCPYSIZE2 (1024 * 1024)
  #define TEST_STREAM_CREATE_WITH_PRIORITY_TOTALTHREADS 16
  
  // hipStreamGetDevice.cc parameters
  #define TEST_STREAM_GET_DEVICE_NUMBER_OF_THREADS 10
  
  // hipStreamLegacy_Ext.cc parameters
  #define TEST_STREAM_LEGACY_EXT_N (2 * 1024 * 1024)
  
  // hipStreamWithCUMask.cc parameters
  #define TEST_STREAM_WITH_CU_MASK_N_MULTIPLIER 25
#endif

// =============================================================================
// PRINTF TEST PARAMETERS
// =============================================================================

#ifdef QUICK_TESTS
  // printfNonHost.cc parameters
  #define TEST_PRINTF_NON_HOST_KERNEL_ITERATIONS 3
#else
  // printfNonHost.cc parameters
  #define TEST_PRINTF_NON_HOST_KERNEL_ITERATIONS 15
#endif

#endif // HIP_TEST_CONFIG_HH


