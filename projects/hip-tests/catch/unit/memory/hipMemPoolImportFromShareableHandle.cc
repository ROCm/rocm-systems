/**
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipMemPoolImportFromShareableHandle hipMemPoolImportFromShareableHandle
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemPoolImportFromShareableHandle(
                                                 hipMemPool_t*              mem_pool,
                                                 void*                      shared_handle,
                                                 hipMemAllocationHandleType handle_type,
                                                 unsigned int               flags) ` -
 * Imports a memory pool from a shared handle.
 */

#include "mempool_common.hh"

/**
 * Test Description
 * ------------------------
 *    - Negative Tests for hipMemPoolImportFromShareableHandle.
 * ------------------------
 *    - unit/memory/hipMemPoolImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolImportFromShareableHandle_Negative") {
  hipShareableHdl sharedHandle;
  hipMemPoolProps pool_props{};
  hipMemPool_t mempoolPfd;
  checkMempoolSupported(0)

      // Create mempool with Posix File Descriptor
      pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.handleTypes = hipMemHandleTypePosixFileDescriptor;
  HIP_CHECK(hipMemPoolCreate(&mempoolPfd, &pool_props));

  HIP_CHECK(hipMemPoolExportToShareableHandle(&sharedHandle, mempoolPfd,
                                              hipMemHandleTypePosixFileDescriptor, 0));
  hipMemPool_t mempoolImp;
  SECTION("Passing nullptr as imported mempool") {
    HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(nullptr, (void*)sharedHandle,
                                                        hipMemHandleTypePosixFileDescriptor, 0),
                    hipErrorInvalidValue);
  }
  SECTION("Passing nullptr as handle") {
    HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(&mempoolImp, nullptr,
                                                        hipMemHandleTypePosixFileDescriptor, 0),
                    hipErrorInvalidValue);
  }
  SECTION("Passing invalid handle type") {
    HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(&mempoolImp, (void*)sharedHandle,
                                                        hipMemHandleTypeNone, 0),
                    hipErrorInvalidValue);
  }
  HIP_CHECK(hipMemPoolDestroy(mempoolPfd));
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
