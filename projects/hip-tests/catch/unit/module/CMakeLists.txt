# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# All Module API Tests
set(TEST_SRC
  hip_module_common.cc
  hipModuleLoad.cc
  hipModuleLoadData.cc
  hipModuleLoadDataEx.cc
  hipModuleUnload.cc
  hipModuleGetFunction.cc
  hipModuleLaunchKernel.cc
  hipModuleGetGlobal.cc
  hipModuleGetTexRef.cc
  hipModuleLaunchCooperativeKernel.cc
  hipModuleLaunchCooperativeKernelMultiDevice.cc
  hipModuleGetFunctionCount.cc
  hipModuleLoadFatBinary.cc
  hipFuncGetAttribute.cc
  hipFuncSetAttribute.cc
  hipFuncGetAttributes.cc
  hipFuncSetSharedMemConfig.cc
  hipGetFuncBySymbol.cc
  hipDrvLaunchKernelEx.cc
  hipManagedKeyword.cc
  hipModule.cc
  hipModuleLoadMultProcessOnMultGPU.cc
)

if(HIP_PLATFORM MATCHES "amd")
  list(APPEND TEST_SRC
    hipExtModuleLaunchKernel.cc
    hipHccModuleLaunchKernel.cc
    hipExtLaunchKernelGGL.cc
    hipExtLaunchMultiKernelMultiDevice.cc
    hipLinkCreate.cc)

  if(BUILD_SHARED_LIBS)
    list(APPEND TEST_SRC hipGetProcAddressModuleApis.cc)
  endif()

  if(UNIX)
    list(APPEND TEST_SRC hipKerArgOptimization.cc)
  endif()

  set(RTCLIB "hiprtc")
else()
  set(RTCLIB "nvrtc")
endif()

function(create_device_code_targets DEVICE_CODE_TARGETS)
  foreach(TARGET IN LISTS DEVICE_CODE_TARGETS)
    hip_add_device_code(
      TARGET ${TARGET}
      SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cc)
  endforeach()
endfunction()

set(ALL_DEVICE_TARGETS
  get_function_module
  launch_kernel_module
  get_global_test_module
  get_tex_ref_module
  coopKernel
  empty_module
  managed_kernel
  vcpy_kernel
  matmul
  kernel_composite_test
  emptyModuleCount
  kernel_count
)

if(HIP_PLATFORM MATCHES "amd")
  list(APPEND ALL_DEVICE_TARGETS
    copyKernel
    addKernel)
endif()

create_device_code_targets("${ALL_DEVICE_TARGETS}")

if(HIP_PLATFORM MATCHES "amd")
  hip_add_device_code(
    TARGET copyKernelCompressed
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
    COMPILE_OPTIONS --offload-compress)

  hip_add_device_code(
    TARGET copyKernel_assembly
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
    OUTPUT_FORMAT assembly
    OUTPUT_NAME copyKernel)

  hip_add_device_code(
    TARGET addKernel_spv
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/addKernel.cc
    OUTPUT_FORMAT spv
    OUTPUT_NAME addKernel)

  hip_add_device_code(
    TARGET addKernel_bundle_spv
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/addKernel.cc
    OUTPUT_FORMAT spv-bundle
    OUTPUT_NAME addKernel-bundle)

  # Define generic offload architectures for testing
  set(OFFLOAD_ARCH_GENERIC_STR
    --offload-arch=gfx9-generic
    --offload-arch=gfx9-4-generic:sramecc+:xnack-
    --offload-arch=gfx9-4-generic:sramecc-:xnack-
    --offload-arch=gfx9-4-generic:xnack+
    --offload-arch=gfx10-1-generic
    --offload-arch=gfx10-3-generic
    --offload-arch=gfx11-generic
    --offload-arch=gfx12-generic)

  hip_add_device_code(
    TARGET copyKernelGenericTarget
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
    OFFLOAD_ARCH_OVERRIDE ${OFFLOAD_ARCH_GENERIC_STR})

  hip_add_device_code(
    TARGET copyKernelGenericTargetCompressed
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
    OFFLOAD_ARCH_OVERRIDE ${OFFLOAD_ARCH_GENERIC_STR}
    COMPILE_OPTIONS --offload-compress)

  list(APPEND ALL_DEVICE_TARGETS
    copyKernelCompressed
    copyKernel_assembly
    addKernel_spv
    addKernel_bundle_spv
    copyKernelGenericTarget
    copyKernelGenericTargetCompressed)

  if(UNIX)
    set(PRELOAD_COUNTS 0 1 2 3 16 17)
    foreach(COUNT IN LISTS PRELOAD_COUNTS)
      set(TARGET_NAME "copiousArgKernel${COUNT}")

      hip_add_device_code(
        TARGET ${TARGET_NAME}
        SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
        COMPILE_OPTIONS -mllvm -amdgpu-kernarg-preload-count=${COUNT})

      list(APPEND ALL_DEVICE_TARGETS ${TARGET_NAME})
    endforeach()

    hip_add_device_code(
        TARGET copiousArgKernel
        SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc)

    list(APPEND ALL_DEVICE_TARGETS copiousArgKernel)
  endif()
endif()

hip_add_exe_to_target(NAME ModuleTest
  TEST_SRC ${TEST_SRC}
  TEST_TARGET_NAME build_tests
  LINKER_LIBS ${RTCLIB}
  COMPILE_OPTIONS -std=c++17)

hip_add_executable(NAME hipGetFuncBySymbol_exe
                   SOURCE hipGetFuncBySymbol_exe.cc
                   ADD_TO_TARGET build_tests)

hip_add_executable(NAME testhipModuleLoadUnloadFunc_exe
                   SOURCE testhipModuleLoadUnloadFunc_exe.cc
                   ADD_TO_TARGET build_tests)

foreach(TARGET IN LISTS ALL_DEVICE_TARGETS)
  add_dependencies(build_tests ${TARGET})
endforeach()

add_dependencies(ModuleTest testhipModuleLoadUnloadFunc_exe)
