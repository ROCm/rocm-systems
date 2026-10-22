# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# The binding tests. The HIP ones build anywhere but need a GPU to pass.

# The Intel compilers skip .f03 and .f08 sources unless -Tf precedes them.
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    string(
        REPLACE
        "<SOURCE>"
        "-Tf <SOURCE>"
        CMAKE_Fortran_COMPILE_OBJECT
        "${CMAKE_Fortran_COMPILE_OBJECT}"
    )
endif()

# Keep only the binding wrappers a test calls (the archive has one section per
# procedure), so a test links even when another wrapper names a C symbol the
# installed library does not export. Spelled -Wl, rather than LINKER:, which
# CMake 3.27 passes to LLVMFlang as a bare --gc-sections.
if(NOT APPLE AND NOT WIN32)
    add_link_options(-Wl,--gc-sections)
endif()

# ROCTx: one test, no GPU needed. The parent found rocprofiler-sdk-roctx.
if(TARGET roctx_fortran AND _ROCTX_clients)
    if(ROCTX_FORTRAN_NEEDS_C_PACKAGE)
        set_source_files_properties(roctx.f03 PROPERTIES LANGUAGE Fortran)
        add_executable(roctx_fortran_test_roctx roctx.f03)
        set_target_properties(
            roctx_fortran_test_roctx
            PROPERTIES LINKER_LANGUAGE Fortran
        )
        target_link_libraries(roctx_fortran_test_roctx PRIVATE roctx_fortran)
        add_test(NAME roctx_fortran_roctx COMMAND roctx_fortran_test_roctx)
    else()
        message(
            STATUS
            "ROCTx: rocprofiler-sdk-roctx was not found, skipping the Fortran binding "
            "test (it links it). Point CMAKE_PREFIX_PATH at an installed ROCm to "
            "build it."
        )
    endif()
endif()

if(NOT TARGET hip_fortran OR NOT _HIP_clients)
    return()
endif()

# The parent found hip; the tests link its runtime.
if(NOT TARGET hip::host)
    message(
        STATUS
        "HIP: no HIP runtime target, skipping the Fortran binding tests "
        "(they link it). Point CMAKE_PREFIX_PATH at an installed ROCm "
        "to build them."
    )
    return()
endif()
set(HIP_FORTRAN_RUNTIME hip::host)

# hip_fortran_add_test(<name> [F2008]): <name>.f03 by default, which uses the
# C-pointer interfaces; with F2008, <name>.f08, which uses the array overloads,
# so its target and test names end in _f2008.
function(hip_fortran_add_test name)
    if("F2008" IN_LIST ARGN)
        set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.f08")
        set(id "${name}_f2008")
    else()
        set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.f03")
        set(id "${name}")
    endif()
    if(NOT EXISTS "${src}")
        message(FATAL_ERROR "Missing test source: ${src}")
    endif()
    set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
    add_executable(hip_fortran_test_${id} "${src}")
    set_target_properties(
        hip_fortran_test_${id}
        PROPERTIES LINKER_LANGUAGE Fortran
    )
    target_link_libraries(
        hip_fortran_test_${id}
        PRIVATE hip_fortran ${HIP_FORTRAN_RUNTIME}
    )
    # The tests call EXIT, a GNU extension in gfortran.
    target_compile_options(
        hip_fortran_test_${id}
        PRIVATE $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>
    )
    add_test(NAME hip_fortran_${id} COMMAND hip_fortran_test_${id})
    # A test exits 77 when the device lacks the feature it exercises.
    set_tests_properties(
        hip_fortran_${id}
        PROPERTIES LABELS "gpu" SKIP_RETURN_CODE 77
    )
endfunction()

set(HIP_FORTRAN_TESTS
    cache_config
    device_management
    device_properties
    error_version
    event
    event_timing
    graph
    graph_empty_node
    graph_memset_node
    graph_nodes
    host_malloc
    host_register
    malloc_managed
    mem_advise
    mem_pool
    memcpy2d
    memcpy_async
    memory_ops
    memset
    peer_access
    peer_copy
    pointer_attributes
    stream
    stream_callback
    stream_capture_info
    stream_flags
    virtual_memory
)

foreach(_t IN LISTS HIP_FORTRAN_TESTS)
    hip_fortran_add_test(${_t})
endforeach()

# The array overloads exist in both the assumed-shape and assumed-rank tiers.
if(NOT FORTRAN_ARRAY_INTERFACES STREQUAL "none")
    foreach(
        _t
        IN
        ITEMS
            host_malloc
            malloc_failure
            malloc_lbounds
            malloc_managed
            mem_advise
            memcpy2d
            pointer_attributes
    )
        hip_fortran_add_test(${_t} F2008)
    endforeach()
endif()

# hipCheck on a failing status must name the status, then stop. The pass
# expression replaces the exit-status check, so the stop is not a failure here.
hip_fortran_add_test(check_message)
set_tests_properties(
    hip_fortran_check_message
    PROPERTIES
        PASS_REGULAR_EXPRESSION "HIP ERROR: hipErrorInvalidValue \\(code 1\\)"
        FAIL_REGULAR_EXPRESSION "FAILED!"
)

# occupancy and cooperative_launch take a kernel's host stub, so the kernel is
# compiled into the test executable.
if(NOT CMAKE_HIP_COMPILER_LOADED)
    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
    endif()
endif()

if(CMAKE_HIP_COMPILER_LOADED)
    foreach(_kt IN ITEMS occupancy cooperative_launch)
        hip_fortran_add_test(${_kt})
        target_sources(hip_fortran_test_${_kt} PRIVATE vecadd_kernel.hip.cpp)
        set_source_files_properties(
            vecadd_kernel.hip.cpp
            PROPERTIES LANGUAGE HIP
        )
        set_target_properties(
            hip_fortran_test_${_kt}
            PROPERTIES LINKER_LANGUAGE Fortran POSITION_INDEPENDENT_CODE ON
        )
    endforeach()
else()
    message(
        STATUS
        "HIP: no HIP compiler, skipping the Fortran binding tests that embed a "
        "GPU kernel (occupancy, cooperative_launch)"
    )
endif()

# module_kernel loads the same kernel as a standalone code object at runtime.
find_program(
    HIP_FORTRAN_HIPCC
    hipcc
    HINTS "${HIPCC_BIN_DIR}" "${ROCM_PATH}/bin" /opt/rocm/bin
)

set(_archs "")
foreach(_v CMAKE_HIP_ARCHITECTURES GPU_TARGETS AMDGPU_TARGETS)
    if(NOT _archs AND ${_v})
        set(_archs "${${_v}}")
    endif()
endforeach()
if(NOT _archs)
    # Bare gfxNNN: the :xnack suffix of -name pins the code object to one mode.
    find_program(
        HIP_FORTRAN_AGENT_ENUM
        rocm_agent_enumerator
        HINTS "${ROCM_PATH}/bin" /opt/rocm/bin
    )
    if(HIP_FORTRAN_AGENT_ENUM)
        execute_process(
            COMMAND ${HIP_FORTRAN_AGENT_ENUM}
            OUTPUT_VARIABLE _agents
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        string(REPLACE "\n" ";" _agents "${_agents}")
        list(REMOVE_ITEM _agents "gfx000")
        if(_agents)
            list(GET _agents 0 _archs)
        endif()
    endif()
endif()

if(NOT HIP_FORTRAN_HIPCC)
    message(
        STATUS
        "HIP: hipcc not found, skipping the module_kernel Fortran binding test "
        "(it loads a code object built with hipcc --genco)"
    )
elseif(NOT _archs)
    message(
        STATUS
        "HIP: no GPU architecture known, skipping the module_kernel Fortran binding "
        "test. Set CMAKE_HIP_ARCHITECTURES or GPU_TARGETS to build it."
    )
else()
    set(_arch_flags "")
    foreach(_a IN LISTS _archs)
        list(APPEND _arch_flags "--offload-arch=${_a}")
    endforeach()

    set(_code_object "${CMAKE_CURRENT_BINARY_DIR}/vecadd_kernel.co")
    add_custom_command(
        OUTPUT "${_code_object}"
        COMMAND
            ${HIP_FORTRAN_HIPCC} --genco ${_arch_flags}
            "${CMAKE_CURRENT_SOURCE_DIR}/vecadd_kernel.hip.cpp" -o
            "${_code_object}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/vecadd_kernel.hip.cpp"
        COMMENT "Building the HIP code object used by the module_kernel test"
    )
    add_custom_target(hip_fortran_test_code_object DEPENDS "${_code_object}")

    hip_fortran_add_test(module_kernel)
    add_dependencies(
        hip_fortran_test_module_kernel
        hip_fortran_test_code_object
    )
    set_tests_properties(
        hip_fortran_module_kernel
        PROPERTIES ENVIRONMENT "HIP_FORTRAN_TEST_CODE_OBJECT=${_code_object}"
    )
endif()
