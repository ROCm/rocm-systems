# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# D3D11 interop tests. Windows only: the sources include <d3d11.h> and the
# TestList is empty on other platforms, so there is nothing to build there.

set(TESTS
    OCLDX11YUY2
)

add_library(ocldx SHARED
    TestList.cpp
    OCLDX11Common.cpp
    $<TARGET_OBJECTS:Common>)

foreach(TEST ${TESTS})
    target_sources(ocldx
        PRIVATE
            ${TEST}.cpp)
endforeach()

set_target_properties(ocldx PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst)

target_compile_definitions(ocldx
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_COMPILE_DEFINITIONS>)

target_include_directories(ocldx
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(ocldx PRIVATE ${OPENCL_ICD_LIB} d3d11 dxgi)

add_custom_command(
    TARGET ocldx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/ocldx.exclude
            ${CMAKE_BINARY_DIR}/tests/ocltst/ocldx.exclude)

add_custom_target(test.ocltst.ocldx
    COMMAND
        ${CMAKE_COMMAND} -E env "OCL_ICD_FILENAMES=$<TARGET_FILE:amdocl>"
        $<TARGET_FILE:ocltst> -p 0 -m $<TARGET_FILE:ocldx> -A ocldx.exclude
    DEPENDS
        ocltst ocldx amdocl
    WORKING_DIRECTORY
        ${CMAKE_BINARY_DIR}/tests/ocltst
    USES_TERMINAL)

foreach(TEST ${TESTS})
    add_custom_target(test.ocltst.ocldx.${TEST}
        COMMAND
            ${CMAKE_COMMAND} -E env "OCL_ICD_FILENAMES=$<TARGET_FILE:amdocl>"
            $<TARGET_FILE:ocltst> -p 0 -m $<TARGET_FILE:ocldx> -t ${TEST}
        DEPENDS
            ocltst ocldx amdocl
        WORKING_DIRECTORY
            ${CMAKE_BINARY_DIR}/tests/ocltst
        USES_TERMINAL)
endforeach()

INSTALL(TARGETS ocldx
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT ocltst)
INSTALL(FILES ocldx.exclude DESTINATION ${OCLTST_INSTALL_DIR} COMPONENT ocltst)
