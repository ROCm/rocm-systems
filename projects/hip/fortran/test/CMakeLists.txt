# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Runtime tests: they build anywhere but need a GPU to pass.

# The Intel compilers skip .f03 sources unless -Tf precedes them.
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  string(REPLACE "<SOURCE>" "-Tf <SOURCE>"
         CMAKE_Fortran_COMPILE_OBJECT "${CMAKE_Fortran_COMPILE_OBJECT}")
endif()

# Same order as the parent: in-tree, find_package(hip) is circular.
if(NOT TARGET hip::host AND NOT TARGET amdhip64)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(hip QUIET)
endif()

if(TARGET hip::host)
  set(HIP_FORTRAN_RUNTIME hip::host)
elseif(TARGET amdhip64)
  set(HIP_FORTRAN_RUNTIME amdhip64)
else()
  message(STATUS
    "HIP: no HIP runtime target, skipping the Fortran binding tests "
    "(they link it). Point CMAKE_PREFIX_PATH at an installed ROCm "
    "to build them standalone.")
  return()
endif()

function(hip_fortran_add_test name)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.f03")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "Missing test source: ${src}")
  endif()
  set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
  add_executable(hip_fortran_test_${name} "${src}")
  set_target_properties(hip_fortran_test_${name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(hip_fortran_test_${name} PRIVATE hip_fortran ${HIP_FORTRAN_RUNTIME})
  # The tests call EXIT, a GNU extension in gfortran.
  target_compile_options(hip_fortran_test_${name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME hip_fortran_${name} COMMAND hip_fortran_test_${name})
  # A test exits 77 when the device lacks the feature it exercises.
  set_tests_properties(hip_fortran_${name} PROPERTIES LABELS "gpu"
                       SKIP_RETURN_CODE 77)
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
    pointer_attributes
    stream
    stream_callback
    stream_capture_info
    stream_flags
    virtual_memory)

foreach(_t IN LISTS HIP_FORTRAN_TESTS)
  hip_fortran_add_test(${_t})
endforeach()

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
    set_source_files_properties(vecadd_kernel.hip.cpp PROPERTIES LANGUAGE HIP)
    set_target_properties(hip_fortran_test_${_kt} PROPERTIES
      LINKER_LANGUAGE Fortran
      POSITION_INDEPENDENT_CODE ON)
  endforeach()
else()
  message(STATUS
    "HIP: no HIP compiler, skipping the Fortran binding tests that embed a "
    "GPU kernel (occupancy, cooperative_launch)")
endif()

# module_kernel loads the same kernel as a standalone code object at runtime.
find_program(HIP_FORTRAN_HIPCC hipcc
             HINTS "${HIPCC_BIN_DIR}" "${ROCM_PATH}/bin" /opt/rocm/bin)

set(_archs "")
foreach(_v CMAKE_HIP_ARCHITECTURES GPU_TARGETS AMDGPU_TARGETS)
  if(NOT _archs AND ${_v})
    set(_archs "${${_v}}")
  endif()
endforeach()
if(NOT _archs)
  # Bare gfxNNN: the :xnack suffix of -name pins the code object to one mode.
  find_program(HIP_FORTRAN_AGENT_ENUM rocm_agent_enumerator
               HINTS "${ROCM_PATH}/bin" /opt/rocm/bin)
  if(HIP_FORTRAN_AGENT_ENUM)
    execute_process(COMMAND ${HIP_FORTRAN_AGENT_ENUM}
                    OUTPUT_VARIABLE _agents OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    string(REPLACE "\n" ";" _agents "${_agents}")
    list(REMOVE_ITEM _agents "gfx000")
    if(_agents)
      list(GET _agents 0 _archs)
    endif()
  endif()
endif()

if(NOT HIP_FORTRAN_HIPCC)
  message(STATUS
    "HIP: hipcc not found, skipping the module_kernel Fortran binding test "
    "(it loads a code object built with hipcc --genco)")
elseif(NOT _archs)
  message(STATUS
    "HIP: no GPU architecture known, skipping the module_kernel Fortran binding "
    "test. Set CMAKE_HIP_ARCHITECTURES or GPU_TARGETS to build it.")
else()
  set(_arch_flags "")
  foreach(_a IN LISTS _archs)
    list(APPEND _arch_flags "--offload-arch=${_a}")
  endforeach()

  set(_code_object "${CMAKE_CURRENT_BINARY_DIR}/vecadd_kernel.co")
  add_custom_command(
    OUTPUT "${_code_object}"
    COMMAND ${HIP_FORTRAN_HIPCC} --genco ${_arch_flags}
            "${CMAKE_CURRENT_SOURCE_DIR}/vecadd_kernel.hip.cpp"
            -o "${_code_object}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/vecadd_kernel.hip.cpp"
    COMMENT "Building the HIP code object used by the module_kernel test")
  add_custom_target(hip_fortran_test_code_object DEPENDS "${_code_object}")

  hip_fortran_add_test(module_kernel)
  add_dependencies(hip_fortran_test_module_kernel hip_fortran_test_code_object)
  set_tests_properties(hip_fortran_module_kernel PROPERTIES
    ENVIRONMENT "HIP_FORTRAN_TEST_CODE_OBJECT=${_code_object}")
endif()
