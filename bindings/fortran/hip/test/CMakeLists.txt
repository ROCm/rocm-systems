# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Runtime tests: they build anywhere but need a GPU to pass.

# The tests are free form whatever their extension: with the Intel compilers,
# which read .f03 and .f08 as fixed form, this adds -free.
set(CMAKE_Fortran_FORMAT FREE)

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

include(GNUInstallDirs)

# Installed tests: the executables go to bin/, and a CTestTestfile.cmake with
# the same tests and properties to bin/hip-fortran/, for ctest --test-dir.
# fortran_binding_install_test(<ctest file> <test> <target> [<properties>])
function(fortran_binding_install_test file test target)
    install(
        TARGETS ${target}
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT tests
    )
    file(APPEND "${file}" "add_test(${test} \"../${target}\")\n")
    if(ARGN)
        file(
            APPEND "${file}"
            "set_tests_properties(${test} PROPERTIES ${ARGN})\n"
        )
    endif()
endfunction()

set(_hip_ctest "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-CTestTestfile.cmake")
file(WRITE "${_hip_ctest}" "")

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
    fortran_binding_install_test(
        "${_hip_ctest}"
        hip_fortran_${id}
        hip_fortran_test_${id}
        "LABELS gpu SKIP_RETURN_CODE 77"
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
    rdma_flush
    stream
    stream_callback
    stream_capture_info
    stream_capture_info_spt_opt
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
file(
    APPEND "${_hip_ctest}"
    [=[set_tests_properties(hip_fortran_check_message PROPERTIES PASS_REGULAR_EXPRESSION "HIP ERROR: hipErrorInvalidValue \\(code 1\\)" FAIL_REGULAR_EXPRESSION "FAILED!")
]=]
)

# The GPU architecture of the tests that build GPU code: occupancy,
# cooperative_launch, and the code object of module_kernel and library_module.
# An explicit CMAKE_HIP_ARCHITECTURES, GPU_TARGETS or AMDGPU_TARGETS wins, and
# otherwise the first GPU that rocm_agent_enumerator reports. With neither, as
# on a login node, these tests are skipped: built for the compiler's default
# architecture, they would fail on the GPU they later run on.
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

# Keep the bare gfxNNN: a target ID such as gfx942:sramecc+:xnack-, which
# GPU_TARGETS can carry from the GPU detection, pins the code object to the
# XNACK mode of the node it was configured on, and the tests run in both.
list(TRANSFORM _archs REPLACE ":.*$" "")
list(REMOVE_DUPLICATES _archs)
if(_archs)
    message(
        STATUS
        "HIP: the Fortran binding tests build GPU code for ${_archs}"
    )
endif()
if(_archs AND NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${_archs}")
endif()

# occupancy and cooperative_launch take a kernel's host stub, so the kernel is
# compiled into the test executable.
if(_archs AND NOT CMAKE_HIP_COMPILER_LOADED)
    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
    endif()
endif()

if(NOT _archs)
    message(
        STATUS
        "HIP: no GPU architecture known, skipping the Fortran binding tests that "
        "build GPU code. Set CMAKE_HIP_ARCHITECTURES or GPU_TARGETS to build them."
    )
elseif(CMAKE_HIP_COMPILER_LOADED)
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

# module_kernel loads the same kernel as a standalone code object at runtime,
# built by the HIP compiler alone: --offload-device-only makes the object the
# clang offload bundle that hipModuleLoad takes, as hipcc --genco would.
if(_archs AND NOT CMAKE_HIP_COMPILER_LOADED)
    message(
        STATUS
        "HIP: no HIP compiler, skipping the module_kernel and library_module "
        "Fortran binding tests (they load a code object of a GPU kernel)"
    )
elseif(_archs)
    add_library(hip_fortran_test_kernel_object OBJECT vecadd_kernel.hip.cpp)
    target_compile_options(
        hip_fortran_test_kernel_object
        PRIVATE --offload-device-only
    )

    set(_code_object "${CMAKE_CURRENT_BINARY_DIR}/vecadd_kernel.co")
    add_custom_command(
        OUTPUT "${_code_object}"
        COMMAND
            ${CMAKE_COMMAND} -E copy
            "$<TARGET_OBJECTS:hip_fortran_test_kernel_object>" "${_code_object}"
        DEPENDS "$<TARGET_OBJECTS:hip_fortran_test_kernel_object>"
        COMMENT "Building the HIP code object used by the module_kernel test"
    )
    add_custom_target(hip_fortran_test_code_object DEPENDS "${_code_object}")
    install(
        FILES "${_code_object}"
        DESTINATION "${CMAKE_INSTALL_BINDIR}/hip-fortran"
        COMPONENT tests
    )

    hip_fortran_add_test(module_kernel)
    add_dependencies(
        hip_fortran_test_module_kernel
        hip_fortran_test_code_object
    )
    set_tests_properties(
        hip_fortran_module_kernel
        PROPERTIES ENVIRONMENT "HIP_FORTRAN_TEST_CODE_OBJECT=${_code_object}"
    )
    file(
        APPEND "${_hip_ctest}"
        "set_tests_properties(hip_fortran_module_kernel PROPERTIES ENVIRONMENT HIP_FORTRAN_TEST_CODE_OBJECT=vecadd_kernel.co)\n"
    )

    # hipLibrary* on the same code object, and the module behind the library.
    hip_fortran_add_test(library_module)
    add_dependencies(
        hip_fortran_test_library_module
        hip_fortran_test_code_object
    )
    set_tests_properties(
        hip_fortran_library_module
        PROPERTIES ENVIRONMENT "HIP_FORTRAN_TEST_CODE_OBJECT=${_code_object}"
    )
    file(
        APPEND "${_hip_ctest}"
        "set_tests_properties(hip_fortran_library_module PROPERTIES ENVIRONMENT HIP_FORTRAN_TEST_CODE_OBJECT=vecadd_kernel.co)\n"
    )
endif()

install(
    FILES "${_hip_ctest}"
    DESTINATION "${CMAKE_INSTALL_BINDIR}/hip-fortran"
    COMPONENT tests
    RENAME CTestTestfile.cmake
)
