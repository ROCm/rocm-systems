# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

find_program(JULIA_EXECUTABLE NAMES julia)

# # Install Julia
# curl -fsSL https://install.julialang.org | sh -s -- -y
# # Get AMDGPU.jl
# julia -e 'import Pkg; Pkg.add("AMDGPU")'
# # To run
# julia vecadd.jl

check_rocprofsys_disable_examples("julia" _JULIA_DISABLED)
if(_JULIA_DISABLED)
    return()
endif()

if(NOT JULIA_EXECUTABLE)
    message(AUTHOR_WARNING "Julia not found - Please install... Skipping Julia examples")
    return()
endif()

# Check if AMDGPU.jl is installed
execute_process(
    COMMAND
        ${JULIA_EXECUTABLE} --startup-file=no -e
        "import Pkg; exit(\"AMDGPU\" in keys(Pkg.project().dependencies) ? 0 : 1)"
    RESULT_VARIABLE AMDGPU_CHECK_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
)

if(NOT AMDGPU_CHECK_RESULT EQUAL 0)
    message(
        AUTHOR_WARNING
        "Julia AMDGPU.jl package not found. Skipping Julia examples.\n"
        "To install, run: julia -e 'import Pkg; Pkg.add(\"AMDGPU\")'"
    )
    return()
endif()

# Julia is JIT compilation
set(JULIA_TARGET_NAME "vecadd-julia")
add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/vecadd.jl"
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_LIST_DIR}/vecadd.jl
        ${CMAKE_BINARY_DIR}/vecadd.jl
    VERBATIM
    COMMENT "Copying vecadd.jl to ${CMAKE_BINARY_DIR}"
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/vecadd.jl
)

# Build target
add_custom_target(
    "${JULIA_TARGET_NAME}-build"
    ALL
    DEPENDS "${CMAKE_BINARY_DIR}/vecadd.jl"
)

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET "${JULIA_TARGET_NAME}-build")
    install(
        PROGRAMS "${CMAKE_BINARY_DIR}/vecadd.jl"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()
