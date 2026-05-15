# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(TARGET rocprofiler-systems-attach)

find_package(rocprofiler-sdk-rocattach QUIET) #QUIET to avoid error if not found

if(rocprofiler-sdk-rocattach_FOUND)
    add_executable(${TARGET} ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-attach.cpp)

    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)

    target_link_libraries(
        ${TARGET}
        PRIVATE
            rocprofiler-sdk::rocprofiler-sdk-rocattach-shared-library
            rocprofiler-systems::rocprofiler-systems-headers
            rocprofiler-systems::rocprofiler-systems-common-library
    )

    set_target_properties(
        ${TARGET}
        PROPERTIES
            BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
            INSTALL_RPATH "${ROCPROFSYS_EXE_INSTALL_RPATH}"
            OUTPUT_NAME ${BINARY_NAME_PREFIX}-attach
    )

    rocprofiler_systems_strip_target(${TARGET})

    install(TARGETS ${TARGET} DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
else()
    message(
        STATUS
        "rocprofiler-sdk-rocattach not found, skipping rocprof-sys-attach target"
    )
endif()
