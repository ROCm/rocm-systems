# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

if(UNIX AND BUILD_SHARED_LIBS)
  set(TEST_SRC ${TEST_SRC}
      complex_loading_behavior.cc)
  set(AMD_TEST_SRC
      hipApiDynamicLoad.cc)

  if(HIP_PLATFORM MATCHES "amd")
    set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
  endif()

  hip_add_exe_to_target(NAME dynamicLoading
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

  if(HIP_PLATFORM MATCHES "amd")
    add_custom_target(libLazyLoad.so COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} -fPIC -lpthread -shared
    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/liblazyLoad.cc
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include ${HIP_PATH_OPT}
    -I${Catch2_SOURCE_DIR}/src -I${Catch2_BINARY_DIR}/generated-includes
    -o libLazyLoad.so)
    add_custom_target(bit_extract_kernel.code COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/bit_extract_kernel.cpp ${HIP_PATH_OPT}
    -o ${CMAKE_CURRENT_BINARY_DIR}/../dynamicLoading/bit_extract_kernel.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
  elseif(HIP_PLATFORM MATCHES "nvidia")
    add_custom_target(libLazyLoad.so COMMAND ${CMAKE_CUDA_COMPILER} -Xcompiler -fPIC -lpthread -shared
    -x cu ${CMAKE_CURRENT_SOURCE_DIR}/liblazyLoad.cc -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
    -I${Catch2_SOURCE_DIR}/src -I${Catch2_BINARY_DIR}/generated-includes -I${HIP_PATH}/include/ -o libLazyLoad.so)
    add_custom_target(bit_extract_kernel.code COMMAND ${CMAKE_HIP_COMPILER} --fatbin
      -x cu ${CMAKE_CURRENT_SOURCE_DIR}/bit_extract_kernel.cpp -I${HIP_PATH}/include/
      ${OFFLOAD_ARCH_LIST} -o ${CMAKE_CURRENT_BINARY_DIR}/../dynamicLoading/bit_extract_kernel.code
      -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
  endif()


  add_custom_target(vecadd.cc COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/vecadd.cc ${CMAKE_CURRENT_BINARY_DIR}/../dynamicLoading/)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS
  ${CMAKE_CURRENT_BINARY_DIR}/vecadd.cc
  ${CMAKE_CURRENT_BINARY_DIR}/libLazyLoad.so
  ${CMAKE_CURRENT_BINARY_DIR}/bit_extract_kernel.code
)
add_dependencies(build_tests bit_extract_kernel.code libLazyLoad.so vecadd.cc)
endif()
