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

# Common Tests - Test independent of all platforms
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
    hipFuncGetAttribute.cc
    hipGetFuncBySymbol.cc
    hipDrvLaunchKernelEx.cc
    hipModuleGetFunctionCount.cc
    hipModuleLoadFatBinary.cc
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/get_function_module.code
                   COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only ${OFFLOAD_ARCH_LIST}
                   -x hip ${CMAKE_CURRENT_SOURCE_DIR}/get_function_module.cc
                    -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                   -o get_function_module.code
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/get_function_module.cc)
add_custom_target(get_function_module ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/get_function_module.code)
#target_link_libraries(get_function_module.code hip::host hip::device)

if (NOT WIN32)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/launch_kernel_module.code
                     COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only ${OFFLOAD_ARCH_LIST}
                     -x hip ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel_module.cc
                     -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                     -o launch_kernel_module.code
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel_module.cc)
  #target_link_libraries(launch_kernel_module.code hip::host hip::device)
  add_custom_target(launch_kernel_module ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/launch_kernel_module.code)

  add_custom_target(coopKernel.code
                    COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only ${OFFLOAD_ARCH_LIST}
                    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/coopKernel.cpp
                    -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/coopKernel.code
                    -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
  #target_link_libraries(coopKernel.code hip::host hip::device)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS 
  ${CMAKE_CURRENT_BINARY_DIR}/launch_kernel_module.code
  ${CMAKE_CURRENT_BINARY_DIR}/coopKernel.code)
endif()



add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/get_global_test_module.code
                   COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only ${OFFLOAD_ARCH_LIST}
                   -x hip ${CMAKE_CURRENT_SOURCE_DIR}/get_global_test_module.cc
                   -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                   -o get_global_test_module.code
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/get_global_test_module.cc)
add_custom_target(get_global_test_module ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/get_global_test_module.code)
#target_link_libraries(get_global_test_module.code hip::host hip::device)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/get_tex_ref_module.code
                   COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only ${OFFLOAD_ARCH_LIST}
                   -x hip ${CMAKE_CURRENT_SOURCE_DIR}/get_tex_ref_module.cc
                   -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                   -o get_tex_ref_module.code
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/get_tex_ref_module.cc)
add_custom_target(get_tex_ref_module ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/get_tex_ref_module.code)
#target_link_libraries(get_tex_ref_module.code hip::host hip::device)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/not_a_module.txt
                   COMMAND ${CMAKE_COMMAND} -E copy
                   ${CMAKE_CURRENT_SOURCE_DIR}/not_a_module.txt
                   ${CMAKE_CURRENT_BINARY_DIR}/not_a_module.txt
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/not_a_module.txt)
add_custom_target(not_a_module ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/not_a_module.txt)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/empty_file.txt
                   COMMAND ${CMAKE_COMMAND} -E copy
                   ${CMAKE_CURRENT_SOURCE_DIR}/empty_file.txt
                   ${CMAKE_CURRENT_BINARY_DIR}/empty_file.txt
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/empty_file.txt)
add_custom_target(empty_file ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/empty_file.txt)

add_custom_target(empty_module
                  COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only ${OFFLOAD_ARCH_LIST}
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/empty_module.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/empty_module.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT})

set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS
  ${CMAKE_CURRENT_BINARY_DIR}/get_function_module.code
  ${CMAKE_CURRENT_BINARY_DIR}/get_global_test_module.code
  ${CMAKE_CURRENT_BINARY_DIR}/get_tex_ref_module.code
  ${CMAKE_CURRENT_BINARY_DIR}/not_a_module.txt
  ${CMAKE_CURRENT_BINARY_DIR}/empty_file.txt
  ${CMAKE_CURRENT_BINARY_DIR}/empty_module.code
)
# Note to pass arch use format like -DOFFLOAD_ARCH_STR="--offload-arch=gfx900  --offload-arch=gfx906"
# having space at the start/end of OFFLOAD_ARCH_STR can cause build failures

if(HIP_PLATFORM MATCHES "amd")
set(TEST_SRC
    ${TEST_SRC}
    hipExtModuleLaunchKernel.cc
    hipHccModuleLaunchKernel.cc
    hipLinkCreate.cc)

if(BUILD_SHARED_LIBS)
  set(TEST_SRC
      ${TEST_SRC}
      hipGetProcAddressModuleApis.cc)
endif()

add_custom_target(copyKernel.code
                  COMMAND ${CMAKE_HIP_COMPILER} -mcode-object-version=5 --cuda-device-only
                  ${OFFLOAD_ARCH_LIST} -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copyKernel.code
                  -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
#target_link_libraries(copyKernel.code hip::host hip::device)

add_custom_target(copyKernel.s
                  COMMAND ${CMAKE_HIP_COMPILER} -mcode-object-version=5 -S -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copyKernel.s
                  -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
#target_link_libraries(copyKernel.s hip::host hip::device)

add_custom_target(addKernel.code
                  COMMAND ${CMAKE_HIP_COMPILER} -mcode-object-version=5 --cuda-device-only -x hip ${OFFLOAD_ARCH_LIST}
                  ${CMAKE_CURRENT_SOURCE_DIR}/addKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/addKernel.code
                  -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
#target_link_libraries(addKernel.code hip::host hip::device)  

# TODO - Rock build failure from clang
if(NOT WIN32)
  add_custom_target(addKernel.spv
                    COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only --offload-arch=amdgcnspirv
                    --no-gpu-bundle-output
                    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/addKernel.cc
                    -o ${CMAKE_CURRENT_BINARY_DIR}/addKernel.spv
                    -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
  #target_link_libraries(addKernel.spv hip::host hip::device)

  add_custom_target(addKernel-bundle.spv
                    COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only --offload-arch=amdgcnspirv
                    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/addKernel.cc
                    -o ${CMAKE_CURRENT_BINARY_DIR}/addKernel-bundle.spv
                    -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
  #target_link_libraries(addKernel-bundle.spv hip::host hip::device)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS
    ${CMAKE_CURRENT_BINARY_DIR}/addKernel.spv
    ${CMAKE_CURRENT_BINARY_DIR}/addKernel-bundle.spv)
endif()


add_custom_target(copyKernelCompressed.code
                  COMMAND ${CMAKE_HIP_COMPILER} -mcode-object-version=5 --offload-compress --cuda-device-only ${OFFLOAD_ARCH_LIST}
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copyKernelCompressed.code
                  -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
#target_link_libraries(copyKernelCompressed.code hip::host hip::device)

set(OFFLOAD_ARCH_GENERIC_STR
  --offload-arch=gfx9-generic
  --offload-arch=gfx9-4-generic:sramecc+:xnack-
  --offload-arch=gfx9-4-generic:sramecc-:xnack-
  --offload-arch=gfx9-4-generic:xnack+
  --offload-arch=gfx10-1-generic
  --offload-arch=gfx10-3-generic
  --offload-arch=gfx11-generic
  --offload-arch=gfx12-generic
)
#set(OFFLOAD_ARCH_GENERIC_STR "--offload-arch=gfx9-generic --offload-arch=gfx9-4-generic:sramecc+:xnack- --offload-arch=gfx9-4-generic:sramecc-:xnack- --offload-arch=gfx9-4-generic:xnack+ --offload-arch=gfx10-1-generic --offload-arch=gfx10-3-generic --offload-arch=gfx11-generic --offload-arch=gfx12-generic")
add_custom_target(copyKernelGenericTarget.code
                  COMMAND ${CMAKE_HIP_COMPILER} -mcode-object-version=6 --cuda-device-only ${OFFLOAD_ARCH_GENERIC_STR}
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copyKernelGenericTarget.code
                  -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
#target_link_libraries(copyKernelGenericTarget.code hip::host hip::device)

add_custom_target(copyKernelGenericTargetCompressed.code
                  COMMAND ${CMAKE_HIP_COMPILER} -mcode-object-version=6 --offload-compress --cuda-device-only
                  ${OFFLOAD_ARCH_GENERIC_STR}
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copyKernelGenericTargetCompressed.code
                  -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
#target_link_libraries(copyKernelGenericTargetCompressed.code hip::host hip::device)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS
  ${CMAKE_CURRENT_BINARY_DIR}/copyKernel.code
  ${CMAKE_CURRENT_BINARY_DIR}/copyKernel.s
  ${CMAKE_CURRENT_BINARY_DIR}/addKernel.code
  ${CMAKE_CURRENT_BINARY_DIR}/copyKernelCompressed.code
  ${CMAKE_CURRENT_BINARY_DIR}/copyKernelGenericTarget.code
  ${CMAKE_CURRENT_BINARY_DIR}/copyKernelGenericTargetCompressed.code
)

if(UNIX)
set(TEST_SRC
    ${TEST_SRC}
    hipKerArgOptimization.cc)

add_custom_target(copiousArgKernel.code
                  COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only ${OFFLOAD_ARCH_LIST}
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copiousArgKernel.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT})
#target_link_libraries(copiousArgKernel.code hip::host hip::device)

add_custom_target(copiousArgKernel0.code
                  COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only ${OFFLOAD_ARCH_LIST}
                  -mllvm -amdgpu-kernarg-preload-count=0
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copiousArgKernel0.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT})
#target_link_libraries(copiousArgKernel0.code hip::host hip::device)

add_custom_target(copiousArgKernel1.code
                  COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only ${OFFLOAD_ARCH_LIST}
                  -mllvm -amdgpu-kernarg-preload-count=1
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copiousArgKernel1.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT})
#target_link_libraries(copiousArgKernel1.code hip::host hip::device)

add_custom_target(copiousArgKernel2.code
                  COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only ${OFFLOAD_ARCH_LIST}
                  -mllvm -amdgpu-kernarg-preload-count=2
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copiousArgKernel2.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT})
#target_link_libraries(copiousArgKernel2.code hip::host hip::device)

add_custom_target(copiousArgKernel3.code
                  COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only ${OFFLOAD_ARCH_LIST}
                  -mllvm -amdgpu-kernarg-preload-count=3
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copiousArgKernel3.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT})
#target_link_libraries(copiousArgKernel3.code hip::host hip::device)

add_custom_target(copiousArgKernel16.code
                  COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only ${OFFLOAD_ARCH_LIST}
                  -mllvm -amdgpu-kernarg-preload-count=16
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copiousArgKernel16.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT})
#target_link_libraries(copiousArgKernel16.code hip::host hip::device)

add_custom_target(copiousArgKernel17.code
                  COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only ${OFFLOAD_ARCH_LIST}
                  -mllvm -amdgpu-kernarg-preload-count=17
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copiousArgKernel17.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT})
#target_link_libraries(copiousArgKernel17.code hip::host hip::device)

set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS
  ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel.code
  ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel0.code
  ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel1.code
  ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel2.code
  ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel3.code
  ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel16.code
  ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel17.code
)
endif()
endif()

if(HIP_PLATFORM MATCHES "amd")
    set(RTCLIB "hiprtc::hiprtc")
else()
    set(RTCLIB "nvrtc")
endif()

hip_add_exe_to_target(NAME ModuleTest
  TEST_SRC ${TEST_SRC}
  TEST_TARGET_NAME build_tests
  LINKER_LIBS ${RTCLIB}
  COMMON_SHARED_SRC ${COMMON_SHARED_SRC})

if(NOT WIN32)
  add_dependencies(ModuleTest coopKernel.code)
  add_dependencies(ModuleTest launch_kernel_module)
endif()

add_dependencies(ModuleTest get_function_module)
add_dependencies(ModuleTest get_global_test_module)
add_dependencies(ModuleTest get_tex_ref_module)
add_dependencies(ModuleTest not_a_module)
add_dependencies(ModuleTest empty_file)
add_dependencies(ModuleTest empty_module)

if(HIP_PLATFORM MATCHES "amd")
add_dependencies(build_tests copyKernel.code copyKernel.s)
add_dependencies(build_tests addKernel.code)
# TODO - Rock build failure from clang
if(NOT WIN32)
  add_dependencies(build_tests addKernel.spv)
  add_dependencies(build_tests addKernel-bundle.spv)
endif()
add_dependencies(build_tests copyKernelCompressed.code)
add_dependencies(build_tests copyKernelGenericTarget.code)
add_dependencies(build_tests copyKernelGenericTargetCompressed.code)

if(UNIX)
add_dependencies(build_tests copiousArgKernel.code copiousArgKernel0.code copiousArgKernel1.code copiousArgKernel2.code
copiousArgKernel3.code copiousArgKernel16.code copiousArgKernel17.code)
endif()
endif()

add_executable(hipGetFuncBySymbol_exe EXCLUDE_FROM_ALL hipGetFuncBySymbol_exe.cc)
set_source_files_properties(hipGetFuncBySymbol_exe.cc PROPERTIES LANGUAGE HIP)
set_target_properties(hipGetFuncBySymbol_exe PROPERTIES LINKER_LANGUAGE HIP)
add_dependencies(build_tests hipGetFuncBySymbol_exe)
target_link_libraries(hipGetFuncBySymbol_exe hip::host hip::device)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hipGetFuncBySymbol_exe)

# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipFuncSetAttribute.cc
  hipFuncGetAttributes.cc
  hipFuncSetSharedMemConfig.cc
  hipManagedKeywordBasic.cc
  hipModule.cc
  hipModuleLoadMultProcessOnMultGPU.cc
)
set(AMD_TEST_SRC
    hipExtLaunchKernelGGL.cc
    hipExtLaunchMultiKernelMultiDevFunctional.cc
)

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
endif()
hip_add_exe_to_target(NAME module
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
add_custom_target(managed_kernel.code COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only
  ${OFFLOAD_ARCH_LIST} -x hip ${CMAKE_CURRENT_SOURCE_DIR}/managed_kernel.cpp
  -o ${CMAKE_CURRENT_BINARY_DIR}/../module/managed_kernel.code
  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT})
#target_link_libraries(managed_kernel.code hip::host hip::device)

add_custom_target(vcpy_kernel.code COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only
  ${OFFLOAD_ARCH_LIST} -x hip ${CMAKE_CURRENT_SOURCE_DIR}/vcpy_kernel.cpp
  -o ${CMAKE_CURRENT_BINARY_DIR}/../module/vcpy_kernel.code
  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT})
#target_link_libraries(vcpy_kernel.code hip::host hip::device)

add_custom_target(matmul.code COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only
  -x hip ${OFFLOAD_ARCH_LIST} ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp -o
  ${CMAKE_CURRENT_BINARY_DIR}/../module/matmul.code
  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT})
#target_link_libraries(matmul.code hip::host hip::device)

add_custom_target(kernel_count.code COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only
  ${OFFLOAD_ARCH_LIST} -x hip ${CMAKE_CURRENT_SOURCE_DIR}/kernel_count.cpp -o
  ${CMAKE_CURRENT_BINARY_DIR}/../module/kernel_count.code
  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT})
#target_link_libraries(kernel_count.code hip::host hip::device)

add_custom_target(emptyModuleCount.code COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only
  ${OFFLOAD_ARCH_LIST} -x hip ${CMAKE_CURRENT_SOURCE_DIR}/emptyModuleCount.cpp -o
  ${CMAKE_CURRENT_BINARY_DIR}/../module/emptyModuleCount.code
  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT})
#target_link_libraries(emptyModuleCount.code hip::host hip::device)

add_dependencies(ModuleTest managed_kernel.code)
add_dependencies(ModuleTest vcpy_kernel.code)
add_dependencies(ModuleTest matmul.code)
add_dependencies(ModuleTest kernel_count.code)
add_dependencies(ModuleTest emptyModuleCount.code)

add_custom_target(kernel_composite_test.code COMMAND ${CMAKE_HIP_COMPILER} --cuda-device-only
  ${OFFLOAD_ARCH_LIST} -x hip ${CMAKE_CURRENT_SOURCE_DIR}/kernel_composite_test.cpp -o
  ${CMAKE_CURRENT_BINARY_DIR}/../module/kernel_composite_test.code
  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT})
#target_link_libraries(kernel_composite_test.code hip::host hip::device)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS
  ${CMAKE_CURRENT_BINARY_DIR}/kernel_composite_test.code
  ${CMAKE_CURRENT_BINARY_DIR}/matmul.code
  ${CMAKE_CURRENT_BINARY_DIR}/vcpy_kernel.code
  ${CMAKE_CURRENT_BINARY_DIR}/managed_kernel.code
  ${CMAKE_CURRENT_BINARY_DIR}/kernel_count.code
  ${CMAKE_CURRENT_BINARY_DIR}/emptyModuleCount.code
)
add_executable(testhipModuleLoadUnloadFunc_exe EXCLUDE_FROM_ALL testhipModuleLoadUnloadFunc_exe.cc)
set_source_files_properties(testhipModuleLoadUnloadFunc_exe.cc PROPERTIES LANGUAGE HIP)
set_target_properties(testhipModuleLoadUnloadFunc_exe PROPERTIES LINKER_LANGUAGE HIP)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS testhipModuleLoadUnloadFunc_exe)
target_link_libraries(testhipModuleLoadUnloadFunc_exe hip::host hip::device)

add_dependencies(module managed_kernel.code vcpy_kernel.code matmul.code kernel_composite_test.code
testhipModuleLoadUnloadFunc_exe)
