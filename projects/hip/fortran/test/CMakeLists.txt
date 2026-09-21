# Tests for the HIP Fortran bindings.
#
# One executable per test, driven by CTest. These are runtime tests: they call
# into the HIP runtime and need a GPU, so they build anywhere but only pass on a
# machine with one.
#
# Reached only from the parent CMakeLists, under BUILD_FORTRAN_TESTS.

# The Intel compilers (ifx/ifort) do not recognize the .f03 source extension and
# skip such files ("no action performed"). Force Fortran compilation by inserting
# -Tf immediately before the source in the compile rule; the LANGUAGE Fortran
# source property alone is not enough.
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  string(REPLACE "<SOURCE>" "-Tf <SOURCE>"
         CMAKE_Fortran_COMPILE_OBJECT "${CMAKE_Fortran_COMPILE_OBJECT}")
endif()

# The HIP runtime is a link-time dependency of every test. In-tree it comes from
# clr through the hip_fortran link interface; standalone it has to be found.
#
# hip-config.cmake reaches FindThreads, which hard-errors unless C or CXX is
# enabled. The bindings are Fortran-only and standalone this project declares
# LANGUAGES NONE, so C is enabled here purely to satisfy that dependency: it
# compiles none of the sources below. Without it, a standalone test build stops
# at "FindThreads only works if either C or CXX language is enabled", which says
# nothing about what is actually wrong.
if(NOT TARGET hip::host)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(hip QUIET)
endif()
if(NOT TARGET hip::host)
  message(STATUS
    "HIP: no hip::host target, skipping the Fortran binding tests "
    "(they link the HIP runtime). Point CMAKE_PREFIX_PATH at an installed ROCm "
    "to build them standalone.")
  return()
endif()

function(hip_fortran_add_test name)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.f03")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "Missing test source: ${src}")
  endif()
  # Force language detection for .f03 in toolchains where CMake misses it.
  set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
  add_executable(hip_fortran_test_${name} "${src}")
  set_target_properties(hip_fortran_test_${name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(hip_fortran_test_${name} PRIVATE hip_fortran hip::host)
  # The tests call EXIT, a GNU extension in gfortran.
  target_compile_options(hip_fortran_test_${name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME hip_fortran_${name} COMMAND hip_fortran_test_${name})
  # Every one of these calls into the HIP runtime and fails without a GPU, which
  # is not the same thing as the binding being broken. The label lets a build
  # machine with no GPU still run the suite meaningfully: ctest -LE gpu.
  set_tests_properties(hip_fortran_${name} PROPERTIES LABELS "gpu")
endfunction()

# Pure-Fortran tests: the binding plus the HIP runtime is all they need to link.
set(HIP_FORTRAN_TESTS
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
    memcpy2d
    memcpy_async
    memory_ops
    memset
    peer_access
    pointer_attributes
    stream
    stream_callback
    stream_flags
    virtual_memory)

foreach(_t IN LISTS HIP_FORTRAN_TESTS)
  hip_fortran_add_test(${_t})
endforeach()

# The remaining three are NOT pure Fortran: they need an actual GPU kernel, which
# only a HIP compiler can produce. vecadd_kernel.hip.cpp carries it. Each of the
# two ways a kernel reaches the runtime is guarded on its own tool, and a site
# without it simply does not get these tests, which is why they are listed apart
# rather than in HIP_FORTRAN_TESTS, where they would break the link for everyone.

# 1. hipOccupancy* and hipLaunchCooperativeKernel take the HOST STUB of a kernel,
#    so the kernel is compiled into the test executable itself.
if(CMAKE_HIP_COMPILER OR CMAKE_HIP_COMPILER_LOADED)
  set(_have_hip_lang TRUE)
else()
  include(CheckLanguage)
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(_have_hip_lang TRUE)
  else()
    set(_have_hip_lang FALSE)
  endif()
endif()

if(_have_hip_lang)
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

# 2. The hipModule* entry points load a STANDALONE code object at runtime, so the
#    same kernel is compiled a second time with --genco and handed to the test
#    through the environment.
find_program(HIP_FORTRAN_HIPCC hipcc HINTS "${ROCM_PATH}/bin" /opt/rocm/bin)
if(HIP_FORTRAN_HIPCC)
  # --offload-arch=native and `rocm_agent_enumerator -name` both append the
  # :xnack suffix, which pins the code object to one XNACK mode and makes
  # hipModuleLoad fail in the other. The bare gfxNNN loads under both.
  set(_arch "native")
  find_program(HIP_FORTRAN_AGENT_ENUM rocm_agent_enumerator
               HINTS "${ROCM_PATH}/bin" /opt/rocm/bin)
  if(HIP_FORTRAN_AGENT_ENUM)
    execute_process(COMMAND ${HIP_FORTRAN_AGENT_ENUM}
                    OUTPUT_VARIABLE _agents OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REPLACE "\n" ";" _agents "${_agents}")
    list(REMOVE_ITEM _agents "gfx000")
    if(_agents)
      list(GET _agents 0 _arch)
    endif()
  endif()
  if(_arch STREQUAL "native")
    message(STATUS
      "HIP: could not enumerate a GPU, so the module_kernel code object falls "
      "back to --offload-arch=native, which does not compile unless a GPU is "
      "visible at build time.")
  endif()

  set(_code_object "${CMAKE_CURRENT_BINARY_DIR}/vecadd_kernel.co")
  add_custom_command(
    OUTPUT "${_code_object}"
    COMMAND ${HIP_FORTRAN_HIPCC} --genco --offload-arch=${_arch}
            "${CMAKE_CURRENT_SOURCE_DIR}/vecadd_kernel.hip.cpp"
            -o "${_code_object}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/vecadd_kernel.hip.cpp"
    COMMENT "Building the HIP code object used by the module_kernel test")
  add_custom_target(hip_fortran_test_code_object DEPENDS "${_code_object}")

  hip_fortran_add_test(module_kernel)
  add_dependencies(hip_fortran_test_module_kernel hip_fortran_test_code_object)
  set_tests_properties(hip_fortran_module_kernel PROPERTIES
    ENVIRONMENT "HIPFORT_TEST_CODE_OBJECT=${_code_object}")
else()
  message(STATUS
    "HIP: hipcc not found, skipping the module_kernel Fortran binding test "
    "(it loads a code object built with hipcc --genco)")
endif()

# Exhaustive "symbol" test, generated by the rocm-fortran generator alongside the
# binding. It references EVERY bind(C) routine of the module, guarded so nothing
# executes: the point is purely to prove each interface compiles and each C
# symbol links. Because it references every symbol, it turns a symbol missing
# from the installed ROCm into a link error, which is useful to audit a ROCm
# version but too strict for the default test build, hence the separate option.
# The name carries hipFORT's HIPFORT_EXTENDED_TESTS forward, because this is the
# same test under a new home and anyone who knew the old flag looks for it.
option(BUILD_FORTRAN_EXTENDED_TESTS
       "Build the exhaustive HIP Fortran symbol test (needs a complete ROCm)" OFF)

if(BUILD_FORTRAN_EXTENDED_TESTS)
  set(_symsrc "${CMAKE_CURRENT_SOURCE_DIR}/test_hip.F03")
  set_source_files_properties("${_symsrc}" PROPERTIES LANGUAGE Fortran)
  add_executable(hip_fortran_test_extended "${_symsrc}")
  set_target_properties(hip_fortran_test_extended PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(hip_fortran_test_extended PRIVATE hip_fortran hip::host)
  target_compile_options(hip_fortran_test_extended PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME hip_fortran_extended COMMAND hip_fortran_test_extended)
  # Nothing in it executes, so unlike the tests above it does not need a GPU:
  # the whole point is what it proves at compile and link time.
  set_tests_properties(hip_fortran_extended PROPERTIES LABELS "symbols")
endif()
