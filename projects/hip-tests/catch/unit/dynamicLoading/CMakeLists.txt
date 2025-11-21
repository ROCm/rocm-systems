# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
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
add_custom_target(libLazyLoad.so COMMAND ${CMAKE_CXX_COMPILER} -fPIC -lpthread -shared ${OFFLOAD_ARCH_STR} ${CMAKE_CURRENT_SOURCE_DIR}/liblazyLoad.cc -I${CMAKE_CURRENT_SOURCE_DIR}/../../include -I${CMAKE_CURRENT_SOURCE_DIR}/../../external/Catch2 -L${HIP_PATH}/${CMAKE_INSTALL_LIBDIR} --hip-path=${HIP_PATH} -o libLazyLoad.so)
elseif(HIP_PLATFORM MATCHES "nvidia")
add_custom_target(libLazyLoad.so COMMAND ${CMAKE_CXX_COMPILER} -Xcompiler -fPIC -lpthread -shared ${CMAKE_CURRENT_SOURCE_DIR}/liblazyLoad.cc -I${CMAKE_CURRENT_SOURCE_DIR}/../../include -I${CMAKE_CURRENT_SOURCE_DIR}/../../external/Catch2 -I${HIP_PATH}/include/ -o libLazyLoad.so)
endif()

add_custom_target(bit_extract_kernel.code COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR} ${CMAKE_CURRENT_SOURCE_DIR}/bit_extract_kernel.cpp -o ${CMAKE_CURRENT_BINARY_DIR}/../dynamicLoading/bit_extract_kernel.code -I${HIP_PATH}/include/ -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --hip-path=${HIP_PATH} -L${HIP_PATH}/${CMAKE_INSTALL_LIBDIR})

add_custom_target(vecadd.cc COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/vecadd.cc ${CMAKE_CURRENT_BINARY_DIR}/../dynamicLoading/)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS
  ${CMAKE_CURRENT_BINARY_DIR}/vecadd.cc
  ${CMAKE_CURRENT_BINARY_DIR}/libLazyLoad.so
  ${CMAKE_CURRENT_BINARY_DIR}/bit_extract_kernel.code
)

if(HIP_PLATFORM MATCHES "amd")
hip_add_exe_to_target(NAME Dynamic
                      TEST_SRC ${LINUX_TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS ${CMAKE_DL_LIBS})
endif()
add_dependencies(build_tests bit_extract_kernel.code libLazyLoad.so vecadd.cc)
endif()
