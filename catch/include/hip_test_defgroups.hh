/*
Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_common.hh>

// Test groups are named based on the group names from hip_api_runtime.h, with adding "Test" suffix

/**
 * @defgroup CallbackTest Callback Activity APIs
 * @{
 * This section describes tests for the callback/Activity of HIP runtime API.
 * @}
 */

/**
 * @defgroup ContextTest Context Management
 * @{
 * This section describes tests for the context management functions of HIP runtime API.
 * @warning All Context Management APIs are **deprecated** and shall not be implemented.
 * @}
 */

/**
 * @defgroup AtomicsTest Device Atomics
 * @{
 * This section describes tests for the Device Atomic APIs.
 * @}
 */

/**
 * @defgroup DeviceLanguageTest Device Language
 * @{
 * This section describes tests for the Device Language API.
 * @}
 */

/**
 * @defgroup DeviceTest Device Management
 * @{
 * This section describes tests for device management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup DriverTest Initialization and Version
 * @{
 * This section describes tests for the initialization and version functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup DynamicLoadingTest Kernel Loading Management
 * @{
 * This section describes the different kernel launch approaches.
 * @}
 */

/**
 * @defgroup ErrorTest Error Handling
 * @{
 * This section describes tests for the error handling functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup EventTest Event Management
 * @{
 * This section describes tests for the event management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup ExecutionTest Execution Control
 * @{
 * This section describes tests for the execution control functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup GraphTest Graph Management
 * @{
 * This section describes tests for the graph management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup KernelTest Kernel Functions Management
 * @{
 * This section describes the various kernel functions invocation.
 * @}
 */

/**
 * @defgroup SyncthreadsTest Synchronization Functions
 * @{
 * This section describes tests for Synchronization Functions.
 * @}
 */

/**
 * @defgroup ThreadfenceTest Memory Fence Functions
 * @{
 * This section describes tests for Memory Fence Functions.
 * @}
 */

/**
 * @defgroup MemoryTest memory Management APIs
 * @{
 * This section describes the memory management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup PeerToPeerTest PeerToPeer Device Memory Access
 * @{
 * This section describes tests for the PeerToPeer device memory access functions of HIP runtime
 * API.
 * @warning PeerToPeer support is experimental.
 * @}
 */

/**
 * @defgroup PerformanceTest Performance tests
 * @{
 * This section describes performance tests for the target API groups and use-cases.
 * @}
 */

/**
 * @defgroup ShflTest warp shuffle function Management
 * @{
 * This section describes the warp shuffle types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup p2pTest P2P Management
 * @{
 * This section describes the P2P management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup StreamOTest Ordered Memory Allocator
 * @{
 * This section describes the tests for Stream Ordered Memory Allocator functions of HIP runtime
 * API.
 * @}
 */

/**
 * @defgroup StreamTest Stream Management
 * @{
 * This section describes the stream management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup ModuleTest Module Management
 * @{
 * This section describes the module management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup TextureTest Texture Management
 * @{
 * This section describes tests for the texture management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup VectorTypeTest Vector types
 * @{
 * This section describes tests for the Vector type functions and operators.
 * @}
 */

 /**
 * @defgroup MathTest Math Device Functions
 * @{
 * This section describes tests for device math functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup PrintfTest Printf API Management
 * @{
 * This section describes the various Printf use case Scenarios.
 * @}
 */

/**
 * @defgroup SurfaceTest Surface Management
 * @{
 * This section describes tests for the surface management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup ComplexTest Complex type
 * @{
 * This section describes tests for the Complex type functions.
 * @}
 */

/**
 * @defgroup VirtualMemoryManagementTest Virtual Memory Management APIs
 * @{
 * This section describes the virtual memory management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup ModuleTest Module Functions Management
 * @{
 * This section describes the loading of modules from code object files and invocation of different kernels.
 * @}
 */
