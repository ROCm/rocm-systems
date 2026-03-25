# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

add_executable(ocltst)

target_sources(ocltst PRIVATE
    ${OCLTST_DIR}/env/oclTestLog.cpp
    ${OCLTST_DIR}/env/oclsysinfo.cpp
    ${OCLTST_DIR}/env/ocltst.cpp
    ${OCLTST_DIR}/env/pfm.cpp
    ${OCLTST_DIR}/env/Timer.cpp
    ${OCLTST_DIR}/module/common/BaseTestImp.cpp
    ${OCLTST_DIR}/module/common/OCLTestImp.cpp
    ${OCLTST_DIR}/module/common/OCLThread.cpp
    ${OCLTST_DIR}/module/common/OCLWrapper.cpp)

# Windows compatibilty logic
if (WIN32)
    target_sources(ocltst PRIVATE
        ${OCLTST_DIR}/env/getopt.cpp
        ${OCLTST_DIR}/env/ServiceCode.cpp
        ${OCLTST_DIR}/env/window.cpp)
endif()


set_target_properties(ocltst PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst)

target_compile_definitions(ocltst
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_COMPILE_DEFINITIONS>)

target_include_directories(ocltst
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(ocltst PRIVATE ${OPENCL_ICD_LIB})

if (NOT WIN32)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  target_link_libraries(ocltst PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
  # ASAN with -shared-libsan requires explicit pthread linkage
  if(THEROCK_SANITIZER MATCHES "ASAN" OR ENABLE_ADDRESS_SANITIZER)
    target_link_options(ocltst PRIVATE -lpthread)
  endif()
endif()

set_target_properties(ocltst PROPERTIES INSTALL_RPATH "$ORIGIN")

INSTALL(TARGETS ocltst DESTINATION ${OCLTST_INSTALL_DIR} COMPONENT ocltst)
