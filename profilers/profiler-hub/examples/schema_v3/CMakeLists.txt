# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(PROFILER_HUB_SCHEMA_V3_EXAMPLES
    memory_alloc_writer_reader
    kernel_dispatch_writer_reader
)

foreach(_example IN LISTS PROFILER_HUB_SCHEMA_V3_EXAMPLES)
    set(_target profiler-hub_${_example})

    add_executable(${_target} ${_example}.cpp)

    target_link_libraries(${_target} PRIVATE ${PROFILER_HUB_EXAMPLE_LIBRARY})
    target_compile_features(${_target} PRIVATE cxx_std_17)

    # Scratch directory for the rocpd database created during the round-trip.
    target_compile_definitions(
        ${_target}
        PRIVATE PHUB_INTEGRATION_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    )

    # Keep the executables under the profiler-hub build tree at
    # <build>/bin/examples.
    set_target_properties(
        ${_target}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROFILER_HUB_EXAMPLE_OUTPUT_DIR}"
    )

    # Locate the shared library relative to the binary
    set_target_properties(${_target} PROPERTIES BUILD_RPATH "$ORIGIN/../../lib")

    if(PROFILER_HUB_INSTALL_EXAMPLES)
        install(
            TARGETS ${_target}
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/profiler-hub/examples
            COMPONENT profiler-hub-examples
        )
    endif()
endforeach()
