# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(COMMON_SHARED_SRC vulkan_test.cc)

if(WIN32)
    set(TEST_SRC
        hipExternalMemoryGetMappedBuffer.cc
        hipExternalMemoryGetMappedMipmappedArray.cc
        hipImportExternalMemory.cc
        hipDestroyExternalMemory.cc
        hipWaitExternalSemaphoresAsync.cc
        hipSignalExternalSemaphoresAsync.cc
        hipImportExternalSemaphore.cc
        hipDestroyExternalSemaphore.cc
        hipVulkanImageDataVerification.cc
       )
else()
    # On Linux only the image data verification test is supported for now. The other Vulkan
    # interop tests (external memory/semaphore, graph nodes) are buggy or not implemented on
    # Linux and need further investigation later.
    set(TEST_SRC
        hipVulkanImageDataVerification.cc
       )
endif()

if(WIN32)
  set(Vulkan_LIBRARY $ENV{VULKAN_SDK}/Lib/vulkan-l)
else()
  # The code can be compiled in Linux, but some required instance extentions
  # aren't supported in Linux, thus test will fail at VulkanTest::CreateInstance()
  # with VK_ERROR_EXTENSION_NOT_PRESENT(-7).
  # Now temperally ignore it in Linux
  message(STATUS "Run vulkan_interop test in Linux")
endif()
find_package(Vulkan)
message(STATUS "Vulkan_FOUND: ${Vulkan_FOUND}")
message(STATUS "Vulkan_LIBRARIES: ${Vulkan_LIBRARIES}")
message(STATUS "Vulkan_INCLUDE_DIRS: ${Vulkan_INCLUDE_DIRS}")

if(NOT Vulkan_FOUND)
  if (EXISTS "$ENV{VULKAN_SDK}")
    message(STATUS "FOUND VULKAN SDK: $ENV{VULKAN_SDK}")
    message(STATUS "Please check Vulkan_LIBRARY if it is missing. Ignore vulkan test!")
    return()
  else()
    message(STATUS "Vulkan sdk not found, interop test not enabled. Please set ENV VULKAN_SDK to enable the test")
    return()
  endif()
endif()

if(WIN32)
  set(LINKER_LIBS vulkan-1 advapi32)
else()
  set(LINKER_LIBS vulkan)
endif()

#set(CMAKE_BUILD_TYPE DEBUG)
include_directories(AFTER ${Vulkan_INCLUDE_DIRS})
link_directories(${Vulkan_INCLUDE_DIRS}/../bin;${Vulkan_INCLUDE_DIRS}/../lib)
hip_add_exe_to_target(NAME VulkanInteropTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMMON_SHARED_SRC ${COMMON_SHARED_SRC}
                      LINKER_LIBS ${LINKER_LIBS})
