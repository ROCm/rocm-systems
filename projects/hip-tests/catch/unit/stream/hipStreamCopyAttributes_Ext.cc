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

#include <hip_test_common.hh>

TEST_CASE("Unit_hipStreamCopyAttributes_Negative") {
  hipStream_t srcStream = nullptr;
  HIP_CHECK(hipStreamCreate(&srcStream));
  hipStream_t dstStream = nullptr;
  HIP_CHECK(hipStreamCreate(&dstStream));

  std::cout << "At line " << __LINE__ << std::endl;

  SECTION("Invalid src Stream") {
    HIP_CHECK_ERROR(hipStreamCopyAttributes(dstStream, reinterpret_cast<hipStream_t>(-1)),
                    hipErrorInvalidResourceHandle);
    std::cout << "At line " << __LINE__ << std::endl;
  }

  SECTION("Invalid dst Stream") {
    HIP_CHECK_ERROR(hipStreamCopyAttributes(reinterpret_cast<hipStream_t>(-1), srcStream),
                    hipErrorInvalidResourceHandle);
    std::cout << "At line " << __LINE__ << std::endl;
  }

  SECTION("Invalid src & dst Streams") {
    HIP_CHECK_ERROR(hipStreamCopyAttributes(reinterpret_cast<hipStream_t>(-1),
                    reinterpret_cast<hipStream_t>(-1)),
	  			    hipErrorInvalidResourceHandle);
    std::cout << "At line " << __LINE__ << std::endl;
  }

  HIP_CHECK(hipStreamDestroy(srcStream)); 
  HIP_CHECK(hipStreamDestroy(dstStream)); 
}

TEST_CASE("Unit_hipStreamCopyAttributes_Negative_InTwoContexts") {
  //hipInit(0);
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  std::cout << "At line " << __LINE__ << std::endl;

  hipCtx_t context1, context2;
  HIP_CHECK(hipCtxCreate(&context1, 0, device)); 
  std::cout << "At line " << __LINE__ << std::endl;

  HIP_CHECK(hipCtxSetCurrent(context1));
  std::cout << "At line " << __LINE__ << std::endl;

  hipStream_t srcStream = nullptr;
  HIP_CHECK(hipStreamCreate(&srcStream));
  std::cout << "At line " << __LINE__ << std::endl;

  HIP_CHECK(hipCtxCreate(&context2, 0, device)); 
  std::cout << "At line " << __LINE__ << std::endl;
  HIP_CHECK(hipCtxSetCurrent(context2));
  std::cout << "At line " << __LINE__ << std::endl;

  hipStream_t dstStream = nullptr;
  HIP_CHECK(hipStreamCreate(&dstStream));
  std::cout << "At line " << __LINE__ << std::endl;

  HIP_CHECK_ERROR(hipStreamCopyAttributes(dstStream, srcStream), hipErrorInvalidValue);
  std::cout << "At line " << __LINE__ << std::endl;

  HIP_CHECK(hipCtxDestroy(context1));
  HIP_CHECK(hipCtxDestroy(context2));
  
}

TEST_CASE("Unit_hipStreamCopyAttributes_WithAllSyncPolicyValues") {
  hipStream_t srcStream = nullptr;
  HIP_CHECK(hipStreamCreate(&srcStream));
  hipStream_t dstStream = nullptr;
  HIP_CHECK(hipStreamCreate(&dstStream));

  hipStreamAttrID attr = hipStreamAttributeSynchronizationPolicy;
  hipStreamAttrValue valueToSet;
  hipSynchronizationPolicy syncPolicy =
      GENERATE(hipSynchronizationPolicy::hipSyncPolicyAuto,
               hipSynchronizationPolicy::hipSyncPolicySpin,
               hipSynchronizationPolicy::hipSyncPolicyYield,
               hipSynchronizationPolicy::hipSyncPolicyBlockingSync);
  valueToSet.syncPolicy = syncPolicy;
  HIP_CHECK(hipStreamSetAttribute(srcStream, attr, &valueToSet));

  HIP_CHECK(hipStreamCopyAttributes(dstStream, srcStream));

  hipStreamAttrValue valueOut;
  HIP_CHECK(hipStreamGetAttribute(dstStream, attr, &valueOut));
  REQUIRE(valueOut.syncPolicy == syncPolicy);

  HIP_CHECK(hipStreamDestroy(srcStream));
  HIP_CHECK(hipStreamDestroy(dstStream)); 
}

TEST_CASE("Unit_hipStreamCopyAttributes_ValuesSetAtStart_InSameContext") {
  hipStream_t srcStream = nullptr;
  HIP_CHECK(hipStreamCreate(&srcStream));
  hipStream_t dstStream = nullptr;
  HIP_CHECK(hipStreamCreate(&dstStream));

  hipStreamAttrID attr = hipStreamAttributeSynchronizationPolicy;

  hipStreamAttrValue valueToSetForSrc;
  valueToSetForSrc.syncPolicy = hipSynchronizationPolicy::hipSyncPolicyAuto;
  HIP_CHECK(hipStreamSetAttribute(srcStream, attr, &valueToSetForSrc));

  hipStreamAttrValue valueToSetForDst;
  valueToSetForDst.syncPolicy = hipSynchronizationPolicy::hipSyncPolicySpin;
  HIP_CHECK(hipStreamSetAttribute(dstStream, attr, &valueToSetForDst));

  HIP_CHECK(hipStreamCopyAttributes(dstStream, srcStream));

  hipStreamAttrValue valueOut;
  HIP_CHECK(hipStreamGetAttribute(dstStream, attr, &valueOut));
  REQUIRE(valueOut.syncPolicy == hipSynchronizationPolicy::hipSyncPolicyAuto);

  HIP_CHECK(hipStreamDestroy(srcStream));
  HIP_CHECK(hipStreamDestroy(dstStream)); 
}

TEST_CASE("Unit_hipStreamCopyAttributes_ValuesSetAtStart_InDifferentContexts") {
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  std::cout << "At line " << __LINE__ << std::endl;

  hipCtx_t context1, context2;
  HIP_CHECK(hipCtxCreate(&context1, 0, device)); 
  std::cout << "At line " << __LINE__ << std::endl;

  HIP_CHECK(hipCtxSetCurrent(context1));
  std::cout << "At line " << __LINE__ << std::endl;

  hipStream_t srcStream = nullptr;
  HIP_CHECK(hipStreamCreate(&srcStream));

  hipStreamAttrID attr = hipStreamAttributeSynchronizationPolicy;

  hipStreamAttrValue valueToSetForSrc;
  valueToSetForSrc.syncPolicy = hipSynchronizationPolicy::hipSyncPolicyAuto;
  HIP_CHECK(hipStreamSetAttribute(srcStream, attr, &valueToSetForSrc));

  HIP_CHECK(hipCtxCreate(&context2, 0, device)); 
  std::cout << "At line " << __LINE__ << std::endl;
  HIP_CHECK(hipCtxSetCurrent(context2));
  std::cout << "At line " << __LINE__ << std::endl;

  hipStream_t dstStream = nullptr;
  HIP_CHECK(hipStreamCreate(&dstStream));

  hipStreamAttrValue valueToSetForDst;
  valueToSetForDst.syncPolicy = hipSynchronizationPolicy::hipSyncPolicySpin;
  HIP_CHECK(hipStreamSetAttribute(dstStream, attr, &valueToSetForDst));

  HIP_CHECK(hipStreamCopyAttributes(dstStream, srcStream));

  hipStreamAttrValue valueOut;
  HIP_CHECK(hipStreamGetAttribute(dstStream, attr, &valueOut));
  REQUIRE(valueOut.syncPolicy == hipSynchronizationPolicy::hipSyncPolicyAuto);

  HIP_CHECK(hipStreamDestroy(dstStream));
  HIP_CHECK(hipCtxDestroy(context2));

  HIP_CHECK(hipCtxSetCurrent(context1));
  HIP_CHECK(hipCtxDestroy(context2));
  HIP_CHECK(hipStreamDestroy(srcStream));
}

