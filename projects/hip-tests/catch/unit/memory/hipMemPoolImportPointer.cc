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
 * @addtogroup hipMemPoolImportPointer hipMemPoolImportPointer
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemPoolImportPointer(
                                      void**                   dev_ptr,
                                      hipMemPool_t             mem_pool,
                                      hipMemPoolPtrExportData* export_data) ` -
 * Import a memory pool allocation from another process.
 */

#include "mempool_common.hh"

constexpr int DATA_SIZE = 1024 * 1024;
constexpr size_t byte_size = DATA_SIZE * sizeof(int);

/**
 * Test Description
 * ------------------------
 *    - Negative Tests for hipMemPoolImportPointer.
 * ------------------------
 *    - unit/memory/hipMemPoolImportPointer.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolImportPointer_Negative") {
  hipMemPoolPtrExportData ptrExp;
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
  int* A_d;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), byte_size, mempoolPfd, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  HIP_CHECK(hipMemPoolExportToShareableHandle(&sharedHandle, mempoolPfd,
                                              hipMemHandleTypePosixFileDescriptor, 0));
  HIP_CHECK(hipMemPoolExportPointer(&ptrExp, A_d));
  hipMemPool_t mempoolImp;
  HIP_CHECK(hipMemPoolImportFromShareableHandle(&mempoolImp, (void*)sharedHandle,
                                                hipMemHandleTypePosixFileDescriptor, 0));
  void* ptrImp;
  SECTION("Passing nullptr as import data") {
    HIP_CHECK_ERROR(hipMemPoolImportPointer(nullptr, mempoolImp, &ptrExp), hipErrorInvalidValue);
  }
  SECTION("Passing nullptr as imported mempool") {
    HIP_CHECK_ERROR(hipMemPoolImportPointer(&ptrImp, nullptr, &ptrExp), hipErrorInvalidValue);
  }
  SECTION("Passing nullptr as exported pointer") {
    HIP_CHECK_ERROR(hipMemPoolImportPointer(&ptrImp, mempoolImp, nullptr), hipErrorInvalidValue);
  }
  HIP_CHECK(hipFree(reinterpret_cast<void*>(A_d)));
  HIP_CHECK(hipMemPoolDestroy(mempoolPfd));
  HIP_CHECK(hipMemPoolDestroy(mempoolImp));
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
