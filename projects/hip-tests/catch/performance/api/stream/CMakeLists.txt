# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

if(HIP_PLATFORM MATCHES "amd")
set(TEST_SRC
    hipStreamWaitEvent.cc
    hipStreamGetFlags.cc
    hipStreamGetPriority.cc
    hipExtStreamCreateWithCUMask.cc
    hipExtStreamGetCUMask.cc
    hipStreamAddCallback.cc
    hipStreamWaitValue.cc
    hipStreamWriteValue.cc
    hipMallocAsync.cc
    hipFreeAsync.cc
    hipMemPoolCreate.cc
    hipMemPoolDestroy.cc
    hipMemPoolTrimTo.cc
    hipMemPoolSetAttribute.cc
    hipMemPoolGetAttribute.cc
    hipMemPoolSetAccess.cc
    hipMallocFromPoolAsync.cc
    hipMemPoolExportToShareableHandle.cc
    hipMemPoolImportFromShareableHandle.cc
    hipMemPoolExportPointer.cc
    hipMemPoolImportPointer.cc
    hipStreamBasic.cc
)
else()
set(TEST_SRC
    hipStreamWaitEvent.cc
    hipStreamGetFlags.cc
    hipStreamGetPriority.cc
    hipStreamAddCallback.cc
    hipStreamWaitValue.cc
    hipStreamWriteValue.cc
    hipMallocAsync.cc
    hipFreeAsync.cc
    hipStreamBasic.cc
)
endif()

hip_add_exe_to_target(NAME StreamPerformance
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
