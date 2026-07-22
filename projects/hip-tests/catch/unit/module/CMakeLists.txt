# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

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

if (HIP_PLATFORM MATCHES "amd")
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/get_function_module.code
                   COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                   -x hip ${CMAKE_CURRENT_SOURCE_DIR}/get_function_module.cc
                    -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                   -o get_function_module.code
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/get_function_module.cc)
else()
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/get_function_module.code
                   COMMAND ${CMAKE_HIP_COMPILER} --fatbin ${OFFLOAD_ARCH_LIST}
                   -x cu ${CMAKE_CURRENT_SOURCE_DIR}/get_function_module.cc
                    -I${HIP_INCLUDE_DIR}
                   -o get_function_module.code
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/get_function_module.cc)
endif()
add_custom_target(get_function_module DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/get_function_module.code)

if (NOT WIN32)
  if (HIP_PLATFORM MATCHES "amd")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/launch_kernel_module.code
                     COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                     -x hip ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel_module.cc
                     -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                     -o launch_kernel_module.code
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel_module.cc)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/coopKernel.code
                    COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/coopKernel.cpp
                    -o ${CMAKE_CURRENT_BINARY_DIR}/coopKernel.code
                    -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/coopKernel.cpp
                    COMMENT "Compiling coopKernel.code")
  else()
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/launch_kernel_module.code
                     COMMAND ${CMAKE_HIP_COMPILER} --fatbin ${OFFLOAD_ARCH_LIST}
                     -x cu ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel_module.cc
                     -I${HIP_INCLUDE_DIR}
                     -o launch_kernel_module.code
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel_module.cc)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/coopKernel.code
                    COMMAND ${CMAKE_HIP_COMPILER} --fatbin ${OFFLOAD_ARCH_LIST}
                    -x cu ${CMAKE_CURRENT_SOURCE_DIR}/coopKernel.cpp
                    -o ${CMAKE_CURRENT_BINARY_DIR}/coopKernel.code
                    -I${HIP_INCLUDE_DIR}
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/coopKernel.cpp
                    COMMENT "Compiling coopKernel.code")
  endif()
  add_custom_target(launch_kernel_module DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/launch_kernel_module.code)
  add_custom_target(coopKernel_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/coopKernel.code)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS
  ${CMAKE_CURRENT_BINARY_DIR}/launch_kernel_module.code
  ${CMAKE_CURRENT_BINARY_DIR}/coopKernel.code)
endif()


if (HIP_PLATFORM MATCHES "amd")
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/get_global_test_module.code
                   COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                   -x hip ${CMAKE_CURRENT_SOURCE_DIR}/get_global_test_module.cc
                   -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                   -o get_global_test_module.code
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/get_global_test_module.cc)
  add_custom_target(get_global_test_module DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/get_global_test_module.code)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/get_tex_ref_module.code
                   COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                   -x hip ${CMAKE_CURRENT_SOURCE_DIR}/get_tex_ref_module.cc
                   -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                   -o get_tex_ref_module.code
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/get_tex_ref_module.cc)
  add_custom_target(get_tex_ref_module DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/get_tex_ref_module.code)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/not_a_module.txt
                   COMMAND ${CMAKE_COMMAND} -E copy
                   ${CMAKE_CURRENT_SOURCE_DIR}/not_a_module.txt
                   ${CMAKE_CURRENT_BINARY_DIR}/not_a_module.txt
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/not_a_module.txt)
  add_custom_target(not_a_module DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/not_a_module.txt)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/empty_file.txt
                   COMMAND ${CMAKE_COMMAND} -E copy
                   ${CMAKE_CURRENT_SOURCE_DIR}/empty_file.txt
                   ${CMAKE_CURRENT_BINARY_DIR}/empty_file.txt
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/empty_file.txt)
  add_custom_target(empty_file DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/empty_file.txt)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/empty_module.code
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/empty_module.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/empty_module.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/empty_module.cc
                  COMMENT "Compiling empty_module.code")
  add_custom_target(empty_module DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/empty_module.code)
else()
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/get_global_test_module.code
                   COMMAND ${CMAKE_HIP_COMPILER} --fatbin ${OFFLOAD_ARCH_LIST}
                   -x cu ${CMAKE_CURRENT_SOURCE_DIR}/get_global_test_module.cc
                   -I${HIP_INCLUDE_DIR}
                   -o get_global_test_module.code
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/get_global_test_module.cc)
  add_custom_target(get_global_test_module DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/get_global_test_module.code)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/get_tex_ref_module.code
                   COMMAND ${CMAKE_HIP_COMPILER} --fatbin ${OFFLOAD_ARCH_LIST}
                   -x cu ${CMAKE_CURRENT_SOURCE_DIR}/get_tex_ref_module.cc
                   -I${HIP_INCLUDE_DIR}
                   -o get_tex_ref_module.code
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/get_tex_ref_module.cc)
  add_custom_target(get_tex_ref_module DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/get_tex_ref_module.code)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/not_a_module.txt
                   COMMAND ${CMAKE_COMMAND} -E copy
                   ${CMAKE_CURRENT_SOURCE_DIR}/not_a_module.txt
                   ${CMAKE_CURRENT_BINARY_DIR}/not_a_module.txt
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/not_a_module.txt)
  add_custom_target(not_a_module DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/not_a_module.txt)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/empty_file.txt
                   COMMAND ${CMAKE_COMMAND} -E copy
                   ${CMAKE_CURRENT_SOURCE_DIR}/empty_file.txt
                   ${CMAKE_CURRENT_BINARY_DIR}/empty_file.txt
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/empty_file.txt)
  add_custom_target(empty_file DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/empty_file.txt)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/empty_module.code
                  COMMAND ${CMAKE_HIP_COMPILER} --fatbin ${OFFLOAD_ARCH_LIST}
                  -x cu ${CMAKE_CURRENT_SOURCE_DIR}/empty_module.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/empty_module.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/empty_module.cc
                  COMMENT "Compiling empty_module.code")
  add_custom_target(empty_module DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/empty_module.code)
endif()
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

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copyKernel.code
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} -mcode-object-version=5 --cuda-device-only
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/copyKernel.code
                  -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  COMMENT "Compiling copyKernel.code")
  add_custom_target(copyKernel_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/copyKernel.code)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copyKernel.s
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} -mcode-object-version=5 -S -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/copyKernel.s
                  -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  COMMENT "Compiling copyKernel.s")
  add_custom_target(copyKernel_s DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/copyKernel.s)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/addKernel.code
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} -mcode-object-version=5 --cuda-device-only -x hip
                  ${CMAKE_CURRENT_SOURCE_DIR}/addKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/addKernel.code
                  -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/addKernel.cc
                  COMMENT "Compiling addKernel.code")
  add_custom_target(addKernel_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/addKernel.code)

  # TODO - Rock build failure from clang
  if(NOT WIN32)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/addKernel.spv
                    COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} --cuda-device-only --offload-arch=amdgcnspirv
                    --no-gpu-bundle-output
                    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/addKernel.cc
                    -o ${CMAKE_CURRENT_BINARY_DIR}/addKernel.spv
                    -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/addKernel.cc
                    COMMENT "Compiling addKernel.spv")
    add_custom_target(addKernel_spv DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/addKernel.spv)

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/addKernel-bundle.spv
                    COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} --cuda-device-only --offload-arch=amdgcnspirv
                    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/addKernel.cc
                    -o ${CMAKE_CURRENT_BINARY_DIR}/addKernel-bundle.spv
                    -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/addKernel.cc
                    COMMENT "Compiling addKernel-bundle.spv")
    add_custom_target(addKernel-bundle_spv DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/addKernel-bundle.spv)
    set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS
    ${CMAKE_CURRENT_BINARY_DIR}/addKernel.spv
    ${CMAKE_CURRENT_BINARY_DIR}/addKernel-bundle.spv)
  endif()


  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copyKernelCompressed.code
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} -mcode-object-version=5 --offload-compress --cuda-device-only
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/copyKernelCompressed.code
                  -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  COMMENT "Compiling copyKernelCompressed.code")
  add_custom_target(copyKernelCompressed_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/copyKernelCompressed.code)

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
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copyKernelGenericTarget.code
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} -mcode-object-version=6 --cuda-device-only ${OFFLOAD_ARCH_GENERIC_STR}
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/copyKernelGenericTarget.code
                  -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  COMMENT "Compiling copyKernelGenericTarget.code")
  add_custom_target(copyKernelGenericTarget_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/copyKernelGenericTarget.code)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copyKernelGenericTargetCompressed.code
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} -mcode-object-version=6 --offload-compress --cuda-device-only
                  ${OFFLOAD_ARCH_GENERIC_STR}
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/copyKernelGenericTargetCompressed.code
                  -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  COMMENT "Compiling copyKernelGenericTargetCompressed.code")
  add_custom_target(copyKernelGenericTargetCompressed_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/copyKernelGenericTargetCompressed.code)
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

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel.code
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  COMMENT "Compiling copiousArgKernel.code")
    add_custom_target(copiousArgKernel_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel.code)

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel0.code
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                  -mllvm -amdgpu-kernarg-preload-count=0
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel0.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  COMMENT "Compiling copiousArgKernel0.code")
    add_custom_target(copiousArgKernel0_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel0.code)

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel1.code
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                  -mllvm -amdgpu-kernarg-preload-count=1
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel1.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  COMMENT "Compiling copiousArgKernel1.code")
    add_custom_target(copiousArgKernel1_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel1.code)

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel2.code
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                  -mllvm -amdgpu-kernarg-preload-count=2
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel2.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  COMMENT "Compiling copiousArgKernel2.code")
    add_custom_target(copiousArgKernel2_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel2.code)

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel3.code
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                  -mllvm -amdgpu-kernarg-preload-count=3
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel3.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  COMMENT "Compiling copiousArgKernel3.code")
    add_custom_target(copiousArgKernel3_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel3.code)

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel16.code
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                  -mllvm -amdgpu-kernarg-preload-count=16
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel16.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  COMMENT "Compiling copiousArgKernel16.code")
    add_custom_target(copiousArgKernel16_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel16.code)

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel17.code
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                  -mllvm -amdgpu-kernarg-preload-count=17
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel17.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  COMMENT "Compiling copiousArgKernel17.code")
    add_custom_target(copiousArgKernel17_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/copiousArgKernel17.code)

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
  add_dependencies(ModuleTest coopKernel_code)
  add_dependencies(ModuleTest launch_kernel_module)
endif()

add_dependencies(ModuleTest get_function_module)
add_dependencies(ModuleTest get_global_test_module)
add_dependencies(ModuleTest get_tex_ref_module)
add_dependencies(ModuleTest not_a_module)
add_dependencies(ModuleTest empty_file)
add_dependencies(ModuleTest empty_module)

if(HIP_PLATFORM MATCHES "amd")
  # Wire kernel artifacts to ModuleTest so `make ModuleTest` builds them.
  # These are loaded at runtime by tests in the ModuleTest binary.
  add_dependencies(ModuleTest copyKernel_code copyKernel_s)
  add_dependencies(ModuleTest addKernel_code)
  # TODO - Rock build failure from clang
  if(NOT WIN32)
    add_dependencies(ModuleTest addKernel_spv)
    add_dependencies(ModuleTest addKernel-bundle_spv)
  endif()
  add_dependencies(ModuleTest copyKernelCompressed_code)
  add_dependencies(ModuleTest copyKernelGenericTarget_code)
  add_dependencies(ModuleTest copyKernelGenericTargetCompressed_code)

  if(UNIX)
    add_dependencies(ModuleTest copiousArgKernel_code copiousArgKernel0_code copiousArgKernel1_code copiousArgKernel2_code
copiousArgKernel3_code copiousArgKernel16_code copiousArgKernel17_code)
  endif()
endif()

add_executable(hipGetFuncBySymbol_exe EXCLUDE_FROM_ALL hipGetFuncBySymbol_exe.cc)
set_source_files_properties(hipGetFuncBySymbol_exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
target_link_libraries(hipGetFuncBySymbol_exe ${GPGPU_LINKER_LIBRARIES})
add_dependencies(build_tests hipGetFuncBySymbol_exe)
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

if (HIP_PLATFORM MATCHES "amd")
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/managed_kernel.code
    COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/managed_kernel.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/managed_kernel.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/managed_kernel.cpp
    COMMENT "Compiling managed_kernel.code")
  add_custom_target(managed_kernel_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/managed_kernel.code)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vcpy_kernel.code
    COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/vcpy_kernel.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/vcpy_kernel.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vcpy_kernel.cpp
    COMMENT "Compiling vcpy_kernel.code")
  add_custom_target(vcpy_kernel_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vcpy_kernel.code)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/matmul.code
    COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/matmul.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
    COMMENT "Compiling matmul.code")
  add_custom_target(matmul_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/matmul.code)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel_count.code
    COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/kernel_count.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/kernel_count.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel_count.cpp
    COMMENT "Compiling kernel_count.code")
  add_custom_target(kernel_count_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernel_count.code)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/emptyModuleCount.code
    COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/emptyModuleCount.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/emptyModuleCount.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/emptyModuleCount.cpp
    COMMENT "Compiling emptyModuleCount.code")
  add_custom_target(emptyModuleCount_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/emptyModuleCount.code)
else()
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/managed_kernel.code
    COMMAND ${CMAKE_HIP_COMPILER} --fatbin
    ${OFFLOAD_ARCH_LIST} -x cu ${CMAKE_CURRENT_SOURCE_DIR}/managed_kernel.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/managed_kernel.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include -I${HIP_INCLUDE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/managed_kernel.cpp
    COMMENT "Compiling managed_kernel.code")
  add_custom_target(managed_kernel_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/managed_kernel.code)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vcpy_kernel.code
    COMMAND ${CMAKE_HIP_COMPILER} --fatbin
    ${OFFLOAD_ARCH_LIST} -x cu ${CMAKE_CURRENT_SOURCE_DIR}/vcpy_kernel.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/vcpy_kernel.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include -I${HIP_INCLUDE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vcpy_kernel.cpp
    COMMENT "Compiling vcpy_kernel.code")
  add_custom_target(vcpy_kernel_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vcpy_kernel.code)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/matmul.code
    COMMAND ${CMAKE_HIP_COMPILER} --fatbin
    -x cu ${OFFLOAD_ARCH_LIST} ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/matmul.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include -I${HIP_INCLUDE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
    COMMENT "Compiling matmul.code")
  add_custom_target(matmul_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/matmul.code)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel_count.code
    COMMAND ${CMAKE_HIP_COMPILER} --fatbin
    ${OFFLOAD_ARCH_LIST} -x cu ${CMAKE_CURRENT_SOURCE_DIR}/kernel_count.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/kernel_count.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include -I${HIP_INCLUDE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel_count.cpp
    COMMENT "Compiling kernel_count.code")
  add_custom_target(kernel_count_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernel_count.code)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/emptyModuleCount.code
    COMMAND ${CMAKE_HIP_COMPILER} --fatbin
    ${OFFLOAD_ARCH_LIST} -x cu ${CMAKE_CURRENT_SOURCE_DIR}/emptyModuleCount.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/emptyModuleCount.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include -I${HIP_INCLUDE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/emptyModuleCount.cpp
    COMMENT "Compiling emptyModuleCount.code")
  add_custom_target(emptyModuleCount_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/emptyModuleCount.code)
endif()

add_dependencies(ModuleTest managed_kernel_code)
add_dependencies(ModuleTest vcpy_kernel_code)
add_dependencies(ModuleTest matmul_code)
add_dependencies(ModuleTest kernel_count_code)
add_dependencies(ModuleTest emptyModuleCount_code)

if (HIP_PLATFORM MATCHES "amd")
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel_composite_test.code
    COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/kernel_composite_test.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/kernel_composite_test.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel_composite_test.cpp
    COMMENT "Compiling kernel_composite_test.code")
else()
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel_composite_test.code
    COMMAND ${CMAKE_HIP_COMPILER} --fatbin
    ${OFFLOAD_ARCH_LIST} -x cu ${CMAKE_CURRENT_SOURCE_DIR}/kernel_composite_test.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/kernel_composite_test.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include -I${HIP_INCLUDE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel_composite_test.cpp
    COMMENT "Compiling kernel_composite_test.code")
endif()
add_custom_target(kernel_composite_test_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernel_composite_test.code)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS
  ${CMAKE_CURRENT_BINARY_DIR}/kernel_composite_test.code
  ${CMAKE_CURRENT_BINARY_DIR}/matmul.code
  ${CMAKE_CURRENT_BINARY_DIR}/vcpy_kernel.code
  ${CMAKE_CURRENT_BINARY_DIR}/managed_kernel.code
  ${CMAKE_CURRENT_BINARY_DIR}/kernel_count.code
  ${CMAKE_CURRENT_BINARY_DIR}/emptyModuleCount.code
)
add_executable(testhipModuleLoadUnloadFunc_exe EXCLUDE_FROM_ALL testhipModuleLoadUnloadFunc_exe.cc)
set_source_files_properties(testhipModuleLoadUnloadFunc_exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
target_link_libraries(testhipModuleLoadUnloadFunc_exe ${GPGPU_LINKER_LIBRARIES})
set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS testhipModuleLoadUnloadFunc_exe)

add_dependencies(module managed_kernel_code vcpy_kernel_code matmul_code kernel_composite_test_code
testhipModuleLoadUnloadFunc_exe)
