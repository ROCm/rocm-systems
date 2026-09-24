cmake_minimum_required(VERSION 3.16)

include("${CMAKE_CURRENT_LIST_DIR}/cmake/RcclAssertSourceLine.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/RcclTargetHelpers.cmake")

# Checked against the original sources because hipify output does not exist yet
# at configure time. Paths are relative to this file, so the same checks cover
# the in-build and standalone contexts.
set(_rccl_src_root "${CMAKE_CURRENT_LIST_DIR}/../../src")
rccl_assert_source_line("${_rccl_src_root}/graph/connect.cc"
  "NCCL_PARAM(MinNchannels, \"MIN_NCHANNELS\", -2)" "fakes/tuning_fakes.cc: ncclParamMinNchannels()")
rccl_assert_source_line("${_rccl_src_root}/graph/connect.cc"
  "NCCL_PARAM(MaxNchannels, \"MAX_NCHANNELS\", -2)" "fakes/tuning_fakes.cc: ncclParamMaxNchannels()")
rccl_assert_source_line("${_rccl_src_root}/enqueue/enqueue.cc"
  "RCCL_PARAM(ForceCe, \"FORCE_CE\", 1)" "fakes/wrap_fakes.cc: rcclParamForceCe()")
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/tuning_fakes.cc"
  "int64_t g_paramMinNchannels = -2;" "the MinNchannels fake default")
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/tuning_fakes.cc"
  "int64_t g_paramMaxNchannels = -2;" "the MaxNchannels fake default")
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/wrap_fakes.cc"
  "static int64_t DefaultParamForceCe() { return 1; }" "the ForceCe fake default")
unset(_rccl_src_root)

set(RCCL_MICRO_TEST_SOURCES
  p2p-test.cc
  rma-proxy-progress-test.cc
  gin-plugin-init-test.cc
  rma-test.cc
  rma-ce-test.cc
  ras-test.cc
  tuning-general-test.cc
  ras-client-test.cc
  group-test.cc
  devcomm-test.cc
  wrap-test.cc
  dev-runtime-test.cc
  # sym-kernels-index-test.cc: added conditionally below, only when GENERATE_SYM_KERNELS is ON.
  fakes/dev_runtime_micro_fakes.cc
  fakes/nccl_fakes.cc
  fakes/libc_fakes.cc
  fakes/ras_param_fakes.cc
  fakes/rma_fakes.cc
  fakes/comm_fakes.cc
  fakes/recorder_fakes.cc
  fakes/utils_fakes.cc
  fakes/collective_stubs.cc
  fakes/devcomm_fakes.cc
  fakes/hip_fakes.cc
  fakes/env_fakes.cc
  fakes/ce_fakes.cc
  fakes/collectives_fakes.cc
  fakes/dda_fakes.cc
  fakes/rccl_arch_thresholds_fakes.cc
  fakes/strongstream_stubs.cc
  fakes/sym_kernels_fakes.cc
  fakes/transport_stubs.cc
  fakes/tuning_fakes.cc
  fakes/wrap_fakes.cc
  ../common/main_altrsmi.cpp
  ../common/ProcessIsolatedTestRunner.cpp
)

# Define the per-target micro-test source lists shared by BOTH build paths.
function(rccl_define_micro_source_lists)
  # rccl-UnitTestsMicroWarpSpeed -- scheduler UUTs under ENABLE_WARP_SPEED.
  set(TEST_MICRO_SCHEDULER_SOURCE_FILES
    scheduler-test.cc
    fakes/enqueue_fakes.cc
    fakes/bootstrap_stubs.cc
    fakes/topo_stubs.cc
    fakes/transport_stubs.cc
    fakes/nccl_stubs.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    # Shared per-production-TU fakes
    fakes/ce_fakes.cc
    fakes/collectives_fakes.cc
    fakes/comm_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/env_fakes.cc
    fakes/os_fakes.cc
    fakes/proxy_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/recorder_fakes.cc
    fakes/register_stubs.cc
    fakes/strongstream_stubs.cc
    fakes/sym_kernels_fakes.cc
    fakes/sym_kernels_index_fakes.cc
    fakes/tuning_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroInit(+-uncached,-faultinj) -- src/init.cc.
  set(TEST_MICRO_INIT_SOURCE_FILES
    init-test.cc
    fakes/init_fakes.cc
    fakes/bootstrap_stubs.cc
    fakes/topo_stubs.cc
    fakes/transport_stubs.cc
    fakes/nccl_stubs.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    # Shared per-production-TU fakes (see MICROTEST_README.md for the map).
    fakes/amdsmi_fakes.cc
    fakes/api_trace_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/env_fakes.cc
    fakes/env_plugin_fakes.cc
    fakes/gin_fakes.cc
    fakes/group_fakes.cc
    fakes/init_nvtx_fakes.cc
    fakes/kernel_config_fakes.cc
    fakes/libc_interposers.cc
    fakes/mem_manager_fakes.cc
    fakes/os_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/recorder_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/strongstream_stubs.cc
    fakes/sym_kernels_fakes.cc
    fakes/tuning_fakes.cc
    fakes/rccl_arch_thresholds_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ../common/main_altrsmi.cpp
    ../common/ProcessIsolatedTestRunner.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroEnqueue(+-devlinker) -- src/enqueue.cc.
  set(TEST_MICRO_ENQUEUE_SOURCE_FILES
    enqueue-test.cc
    fakes/enqueue_test_deps.cc
    fakes/bootstrap_stubs.cc
    fakes/topo_stubs.cc
    fakes/transport_stubs.cc
    fakes/nccl_stubs.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    # Shared per-production-TU fakes (see MICROTEST_README.md for the map).
    fakes/ce_fakes.cc
    fakes/collconfig_fakes.cc
    fakes/collectives_fakes.cc
    fakes/comm_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/env_fakes.cc
    fakes/group_fakes.cc
    fakes/os_fakes.cc
    fakes/proxy_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/recorder_fakes.cc
    fakes/register_stubs.cc
    fakes/sched_stubs.cc
    fakes/strongstream_stubs.cc
    fakes/sym_kernels_fakes.cc
    fakes/tuning_fakes.cc
    fakes/rccl_arch_thresholds_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroSymKernels -- REAL src/sym_kernels.cc + generated host TU.
  set(TEST_MICRO_SYM_KERNELS_SOURCE_FILES
    sym-kernels-test.cc
    fakes/bootstrap_stubs.cc
    fakes/topo_stubs.cc
    fakes/transport_stubs.cc
    fakes/nccl_stubs.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    fakes/env_fakes.cc
    fakes/devcomm_fakes.cc
    fakes/os_fakes.cc
    fakes/strongstream_stubs.cc
    fakes/recorder_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)
endfunction()

# Create one host-only micro-test binary and apply all the boilerplate that
# differs only by build mode. The per-target specifics are passed in:
#   SOURCES         the target's source list
#   EXTRA_INCLUDES  include dirs appended after the mode's include-dir var
#   EXTRA_DEFS      compile definitions appended after the mode's define baseline
#   CATEGORIES_YAML CTest category yaml (in-build only; ignored standalone)
#   REMOVE_DEFS     defines to drop from the mode baseline before EXTRA_DEFS are
#                   added (init strips the three preprocessor-gated macros it sets
#                   per variant so they are never inherited from the common set)
#   NO_PIE          add -no-pie to the link (standalone only; see rccl-HostUnitTests)
function(rccl_add_micro_binary _name)
  cmake_parse_arguments(_ramb "NO_PIE" "CATEGORIES_YAML"
    "SOURCES;EXTRA_INCLUDES;EXTRA_DEFS;REMOVE_DEFS" ${ARGN})

  if(BUILD_TESTS)
    set(_ramb_base_defs ${RCCL_COMMON_COMPILE_DEFS})
    if(_ramb_REMOVE_DEFS)
      list(REMOVE_ITEM _ramb_base_defs ${_ramb_REMOVE_DEFS})
    endif()
    add_executable(${_name} ${_ramb_SOURCES})
    target_include_directories(${_name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR} ${RCCL_COMMON_INCLUDE_DIRS} ${_ramb_EXTRA_INCLUDES})
    target_compile_definitions(${_name} PRIVATE
      ${_ramb_base_defs} ${_ramb_EXTRA_DEFS})
    rccl_apply_rocshmem_gin_optout(${_name})
    rccl_apply_hip_compile_only(${_name})   # HIP compile requirements, no runtime link
    target_link_libraries(${_name} PRIVATE ${MICRO_TEST_LINK_LIBS})
    add_dependencies(${_name} hipify_all copy_nccl_device_headers)
    rccl_apply_host_only_flags(${_name})
    rocm_install(TARGETS ${_name} COMPONENT tests)
    if(_ramb_CATEGORIES_YAML)
      rccl_register_test_categories(${_name} "${_ramb_CATEGORIES_YAML}")
    endif()
  else()
    add_executable(${_name} ${_ramb_SOURCES})
    # Mirror add_test_executable(): record for the HOST_TEST_COVERAGE loop.
    list(APPEND HOST_TEST_BINARIES ${_name})
    set(HOST_TEST_BINARIES "${HOST_TEST_BINARIES}" PARENT_SCOPE)
    target_compile_options(${_name} PRIVATE
      -Wno-unused-parameter -Wno-unused-variable -ffunction-sections)
    target_include_directories(${_name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR} ${RCCL_INCLUDE_DIRS} ${_ramb_EXTRA_INCLUDES})
    target_compile_definitions(${_name} PRIVATE
      __HIP_PLATFORM_AMD__ NCCL_OS_LINUX ROCM_VERSION=70000 ${_ramb_EXTRA_DEFS})
    if(_ramb_NO_PIE)
      target_link_options(${_name} PRIVATE -no-hip-rt -no-pie -Wl,--gc-sections)
    else()
      target_link_options(${_name} PRIVATE -no-hip-rt -Wl,--gc-sections)
    endif()
    target_link_libraries(${_name} PRIVATE
      GTest::GTest Threads::Threads fmt::fmt-header-only)
  endif()
endfunction()

# =====================================================================
# Standalone project setup (only used when NOT in an in-RCCL-build).
#
# Everything a standalone `cmake -B build` on this directory needs that is
# NOT shared with the in-RCCL-build path: GTest/fmt resolution, the hipified
# snapshot location, the amdsmi/HIP fabric probes, the production-.cc wrapper
# library, and the rccl-HostUnitTests binary. It also exports the handful of
# per-mode variables the shared micro-binary block below consumes:
#   RCCL_HOST_HIPIFY_DIR      hipified-snapshot base dir
#   RCCL_INCLUDE_DIRS         include dirs for the standalone micro binaries
#   _MICRO_MODE_DEFS          defs the in-build path inherits from the rccl
#                             target (USE_AMDSMI + the fabric probes); the
#                             standalone project has no rccl target to read
#   _ENQ_BUILD_DEVLINKER      build the shipping enqueue -devlinker arm too
#   _MICRO_BUILD_SYM_KERNELS  sym-kernels sources are present
#   HOST_TEST_BINARIES        coverage instruments every binary in this list
#
# When invoked directly (cmake -B build on this directory) BUILD_TESTS is
# not defined, so we fall through to the standalone project that builds
# rccl-HostUnitTests + rccl-UnitTestsMicro from a hipified snapshot.
# =====================================================================
function(rccl_standalone_setup)
  find_package(Threads REQUIRED)
  include(FetchContent)

  # --- Resolve GTest: system -> RCCL-vendored -> FetchContent fallback ---
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    set(_rccl_gtest_root "${CMAKE_CURRENT_SOURCE_DIR}/../../build/gtest")
    if(EXISTS "${_rccl_gtest_root}/lib/libgtest.a" OR EXISTS "${_rccl_gtest_root}/lib64/libgtest.a")
      message(STATUS "Using RCCL-vendored GTest: ${_rccl_gtest_root}")
      find_package(GTest REQUIRED PATHS "${_rccl_gtest_root}" NO_DEFAULT_PATH)
    else()
      message(STATUS "Downloading GTest via FetchContent")
      FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE
      )
      FetchContent_MakeAvailable(googletest)
      add_library(GTest::GTest ALIAS gtest)
    endif()
  endif()

  # Both find_package(GTest) code paths above (system and RCCL-vendored) export
  # only the modern GTest::gtest target; GTest::GTest is a legacy alias CMake's
  # own FindGTest module stopped providing. Add it once here instead of after
  # every branch so target_link_libraries(... GTest::GTest) below works
  # regardless of which resolution path was taken.
  if(TARGET GTest::gtest AND NOT TARGET GTest::GTest)
    # Promote before aliasing: GTest's package config creates GTest::gtest as a
    # non-GLOBAL imported target, and ALIAS of one of those needs CMake 3.18,
    # above this project's declared 3.16 minimum. Promotion works from 3.11 and
    # only from the directory that created the target, which is this file.
    set_target_properties(GTest::gtest PROPERTIES IMPORTED_GLOBAL TRUE)
    add_library(GTest::GTest ALIAS GTest::gtest)
  endif()

  # --- fmt: the RCCL-vendored header-only copy (already defined for in-tree builds) ---
  if(NOT TARGET fmt::fmt-header-only)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../external/fmt"
                     "${CMAKE_BINARY_DIR}/external/fmt")
  endif()

  # --- Resolve RCCL build directory ---
  set(RCCL_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../build" CACHE PATH
      "RCCL build directory (auto-detects release/debug subdirs)")

  if(NOT EXISTS "${RCCL_BUILD_DIR}/hipify")
    foreach(_subdir release debug Release Debug RelWithDebInfo)
      if(EXISTS "${RCCL_BUILD_DIR}/${_subdir}/hipify")
        set(RCCL_BUILD_DIR "${RCCL_BUILD_DIR}/${_subdir}")
        message(STATUS "Detected build subdir layout: ${RCCL_BUILD_DIR}")
        break()
      endif()
    endforeach()
  endif()

  set(RCCL_HIPIFY_DIR "${RCCL_BUILD_DIR}/hipify")
  set(RCCL_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

  if(NOT EXISTS "${RCCL_HIPIFY_DIR}/src/graph/search.cc")
    message(FATAL_ERROR "Hipified sources not found at ${RCCL_HIPIFY_DIR}. Run the RCCL build first.")
  endif()
  message(STATUS "hipcc: ${CMAKE_CXX_COMPILER}")
  message(STATUS "Hipified sources: ${RCCL_HIPIFY_DIR}")

  # --- Detect whether the system amdsmi.h exposes the UALoE fabric struct ---
  # Compile-only (STATIC_LIBRARY) avoids link failures from amdsmi symbols.
  include(CheckCXXSourceCompiles)
  unset(AMDSMI_FABRIC_API CACHE)
  unset(AMDSMI_HEADER_HAS_FABRIC_TELEMETRY CACHE)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  check_cxx_source_compiles("
    #include <amd_smi/amdsmi.h>
    int main() {
      amdsmi_fabric_telemetry_category_t cat = AMDSMI_FABRIC_TELEMETRY_CATEGORY_UNKNOWN;
      (void)cat;
      return 0;
    }
  " AMDSMI_HEADER_HAS_FABRIC_TELEMETRY)
  check_cxx_source_compiles("
    #include <amd_smi/amdsmi.h>
    #if !defined(AMDSMI_FABRIC_MAX_LOCAL_GPUS)
    #error amdsmi.h has no UALoE fabric types
    #endif
    int main() {
      amdsmi_fabric_info_t info{};
      amdsmi_fabric_info_v1_t v1{};
      (void)info;
      (void)v1;
      return 0;
    }
  " AMDSMI_FABRIC_API)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  if(AMDSMI_HEADER_HAS_FABRIC_TELEMETRY AND NOT AMDSMI_FABRIC_API)
    set(AMDSMI_FABRIC_API TRUE)
    message(STATUS "amdsmi fabric telemetry types found - enabling AMDSMI_FABRIC_DIRECT")
  elseif(AMDSMI_FABRIC_API)
    message(STATUS "amdsmi fabric API found - enabling AMDSMI_FABRIC_DIRECT")
  else()
    message(STATUS "amdsmi fabric API not found (AMDSMI_FABRIC_DIRECT disabled)")
  endif()

  # --- Detect HIP fabric API support (mirrors top-level CMakeLists.txt) ---
  # mem_manager.h only defines its own hipMemFabricHandle_st fallback when
  # HIP_FABRIC_API is undefined; on HIP runtimes that already provide the type
  # (e.g. ROCm 7.14+), omitting this define causes a redefinition conflict
  # with hip_runtime_api.h.
  # Compile-only (STATIC_LIBRARY), like the amdsmi probe above: with the default
  # EXECUTABLE target type these checks link, which drags in the HIP runtime at
  # configure time even though the binary itself is linked with -no-hip-rt.
  include(CheckSymbolExists)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  check_symbol_exists("hipMemImportFromShareableHandle" "hip/hip_runtime_api.h" HIP_FABRIC_API_FUNC)
  check_cxx_source_compiles("
    #include <hip/hip_runtime_api.h>
    int main() {
      hipMemFabricHandle_t handle;
      (void)handle;
      return 0;
    }
  " HIP_FABRIC_HANDLE_TYPE)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  if(HIP_FABRIC_API_FUNC AND HIP_FABRIC_HANDLE_TYPE)
    set(HIP_FABRIC_API ON)
    message(STATUS "HIP Fabric API enabled (hipMemImportFromShareableHandle and hipMemFabricHandle_t found)")
  else()
    set(HIP_FABRIC_API OFF)
  endif()

  # --- Common include paths for hipified RCCL headers ---
  set(RCCL_INCLUDE_DIRS
    ${RCCL_HIPIFY_DIR}/src/include
    ${RCCL_HIPIFY_DIR}/src/include/plugin
    ${RCCL_HIPIFY_DIR}/src/include/plugin/tuner
    ${RCCL_BUILD_DIR}/include
    ${RCCL_HIPIFY_DIR}/src
    ${RCCL_HIPIFY_DIR}/src/graph
    ${RCCL_TEST_DIR}
    ${RCCL_TEST_DIR}/common
  )

  # --- accl-profiler plugin directory ---
  set(ACCL_PROFILER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../plugins/profiler/accl")
  set(RCCL_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

  # --- Wrapper library for production .cc files that need link stubs ---
  add_library(rccl-source-wrappers STATIC
    mem_manager_src_wrapper.cc
    mem_manager_wrapper_test.cc
    bootstrap_wrapper.cc
    init_stubs.cpp
    wrapper_link_stubs.cpp
    hip_prop_r0600_stub.cpp
    accl_profiler_wrapper.cc
  )
  target_include_directories(rccl-source-wrappers PRIVATE
    ${ACCL_PROFILER_DIR}
    ${RCCL_INCLUDE_DIRS}
  )
  # Standalone HostUnitTests never find_package(hip); in-tree already has hip::host.
  if(NOT TARGET hip::host)
    find_package(hip QUIET)
  endif()
  foreach(_hip_tgt hip::host hip::device)
    if(TARGET ${_hip_tgt})
      target_include_directories(rccl-source-wrappers PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_INCLUDE_DIRECTORIES>)
      target_compile_definitions(rccl-source-wrappers PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_COMPILE_DEFINITIONS>)
    endif()
  endforeach()
  # accl_profiler_wrapper.cc needs the non-hipified profiler headers;
  # scope the extra include to just that file to avoid os.h/rocmwrap.h
  # conflicts with the hipified sources used by other wrappers.
  set_source_files_properties(accl_profiler_wrapper.cc PROPERTIES
    COMPILE_FLAGS "-I${RCCL_SRC_ROOT}/src/include/plugin -I${RCCL_SRC_ROOT}/src/include"
  )
  target_compile_definitions(rccl-source-wrappers PRIVATE
    __HIP_PLATFORM_AMD__
    NCCL_OS_LINUX
    ROCM_VERSION=70000
    $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
    $<$<BOOL:${AMDSMI_HEADER_HAS_FABRIC_TELEMETRY}>:AMDSMI_HEADER_HAS_FABRIC_TELEMETRY>
    $<$<BOOL:${HIP_FABRIC_API}>:HIP_FABRIC_API>
  )
  target_link_libraries(rccl-source-wrappers PRIVATE
    GTest::GTest
    fmt::fmt-header-only
  )

  # --- Main test executable ---
  add_test_executable(rccl-HostUnitTests
    main.cpp
    ${RCCL_TEST_DIR}/BitOpsTests.cpp
    ${RCCL_TEST_DIR}/IommuPassthrough_test.cpp
    ${RCCL_TEST_DIR}/VersionInfoTests.cpp
    ${RCCL_TEST_DIR}/DdaCollCommonTests.cpp
    ${RCCL_TEST_DIR}/RomeTopoConsensusTests.cpp
    RomeTopoConsensusLogTests.cpp
    ${RCCL_TEST_DIR}/MiscTests.cpp
    ${RCCL_TEST_DIR}/VersionGateTests.cpp
    ${RCCL_TEST_DIR}/TimeoutTests.cpp
    ${RCCL_TEST_DIR}/EnqueueCountTests.cpp
    ${RCCL_TEST_DIR}/BootstrapBidirTests.cpp
    ${RCCL_TEST_DIR}/AltRsmiTests.cpp
    ${RCCL_TEST_DIR}/AcclProfilerTests.cpp
    ${RCCL_TEST_DIR}/common/ProcessIsolatedTestRunner.cpp
    ${RCCL_HIPIFY_DIR}/src/misc/kernel_config.cc
    ${RCCL_HIPIFY_DIR}/src/misc/alt_rsmi.cc
    ${RCCL_HIPIFY_DIR}/src/graph/rome_topo_consensus.cc
  )

  target_include_directories(rccl-HostUnitTests PRIVATE
    ${ACCL_PROFILER_DIR}
    ${RCCL_INCLUDE_DIRS}
  )
  set_source_files_properties(${RCCL_TEST_DIR}/AcclProfilerTests.cpp PROPERTIES
    COMPILE_FLAGS "-I${RCCL_SRC_ROOT}/src/include/plugin -I${RCCL_SRC_ROOT}/src/include"
  )

  target_compile_definitions(rccl-HostUnitTests PRIVATE
    __HIP_PLATFORM_AMD__
    NCCL_OS_LINUX
    ROCM_VERSION=70000
    RCCL_EXPOSE_STATIC
    ARSMI_TEST_BUILD
    $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
    $<$<BOOL:${AMDSMI_HEADER_HAS_FABRIC_TELEMETRY}>:AMDSMI_HEADER_HAS_FABRIC_TELEMETRY>
    $<$<BOOL:${HIP_FABRIC_API}>:HIP_FABRIC_API>
  )

  target_link_options(rccl-HostUnitTests PRIVATE -no-hip-rt -no-pie)

  target_link_libraries(rccl-HostUnitTests PRIVATE
    GTest::GTest
    Threads::Threads
    fmt::fmt-header-only
    # RCCL_TEST_CODE_COVERAGE compiles a dlsym() call in
    # ProcessIsolatedTestRunner.cpp; the in-tree test build links dl for the same
    # runner. Empty on platforms where libdl is folded into libc (glibc >= 2.34),
    # required on older ones.
    ${CMAKE_DL_LIBS}
    -Wl,--whole-archive
    rccl-source-wrappers
    -Wl,--no-whole-archive
  )

  # --- Per-mode variables consumed by the shared micro-binary block ---------
  set(RCCL_HOST_HIPIFY_DIR "${RCCL_HIPIFY_DIR}" PARENT_SCOPE)
  set(RCCL_INCLUDE_DIRS "${RCCL_INCLUDE_DIRS}" PARENT_SCOPE)
  # USE_AMDSMI + the fabric probes: the in-RCCL-build path inherits these from
  # the rccl target via RCCL_COMMON_COMPILE_DEFS, applied uniformly to every
  # micro binary. This standalone project has no rccl target to read, so add
  # them here (equally uniformly) as the per-mode baseline.
  set(_MICRO_MODE_DEFS
    USE_AMDSMI
    $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
    $<$<BOOL:${AMDSMI_HEADER_HAS_FABRIC_TELEMETRY}>:AMDSMI_HEADER_HAS_FABRIC_TELEMETRY>
    $<$<BOOL:${HIP_FABRIC_API}>:HIP_FABRIC_API>
    PARENT_SCOPE)
  # ENABLE_DEVICE_LINKER defaults ON, so the enqueue arm that SHIPS is the
  # -devlinker one. The in-RCCL-build path builds only the single inherited-arm
  # binary; run_host_tests.sh (the only in-repo workflow that runs these) uses
  # this standalone path, so build the shipping arm explicitly too.
  set(_ENQ_BUILD_DEVLINKER TRUE PARENT_SCOPE)
  # No GENERATE_SYM_KERNELS variable exists in this standalone project(); check
  # for the generated host TU on disk instead.
  if(EXISTS "${RCCL_HIPIFY_DIR}/gensrc/symmetric/sym_kernels_host.cc")
    set(_MICRO_BUILD_SYM_KERNELS TRUE PARENT_SCOPE)
  else()
    set(_MICRO_BUILD_SYM_KERNELS FALSE PARENT_SCOPE)
  endif()
  set(HOST_TEST_BINARIES "${HOST_TEST_BINARIES}" PARENT_SCOPE)
endfunction()

# --- Coverage (all host-only standalone test binaries) ------------------
# Single switch for the whole standalone build: llvm source-based coverage is
# either on for every host-only test binary or off for all of them. Host-only
# build, so no -Xarch_device split is needed -- --offload-host-only never runs
# the amdgcn device pass (see CMAKE_CXX_FLAGS_INIT below). On by default so the
# binaries always produce a profile; disable with -DHOST_TEST_COVERAGE=OFF.
#
# Applied uniformly, per target:
#  - the -fprofile-instr-generate/-fcoverage-mapping instrumentation flags;
#  - RCCL_TEST_CODE_COVERAGE=1, which is what makes process-isolated tests flush
#    their counters in the forked child (common/ProcessIsolatedTestRunner.cpp).
#    Without it those tests run but contribute nothing to the profile;
#  - libdl, which the RCCL_TEST_CODE_COVERAGE dlsym() path in that runner needs
#    (empty where libdl is folded into libc, glibc >= 2.34).
# rccl-source-wrappers is a static library reached by rccl-HostUnitTests, so it
# only takes the instrumentation flags -- mem_manager.cc reaches the binary
# through it and otherwise disappears from the report entirely.
function(rccl_standalone_coverage)
  option(HOST_TEST_COVERAGE "Build the host-only test binaries with llvm source-based coverage" ON)
  if(HOST_TEST_COVERAGE)
    # HOST_TEST_BINARIES is populated by add_test_executable() and by
    # rccl_add_micro_binary(), so new test binaries are instrumented automatically.
    foreach(_cov_tgt ${HOST_TEST_BINARIES})
      target_compile_options(${_cov_tgt} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
      target_link_options(${_cov_tgt} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
      target_compile_definitions(${_cov_tgt} PRIVATE RCCL_TEST_CODE_COVERAGE=1)
      target_link_libraries(${_cov_tgt} PRIVATE ${CMAKE_DL_LIBS})
    endforeach()
    target_compile_options(rccl-source-wrappers PRIVATE -fprofile-instr-generate -fcoverage-mapping)
  endif()
endfunction()

# =====================================================================
# Driver: pick the mode, then define the micro-test binaries ONCE.
#
# When this directory is pulled in via add_subdirectory(host) from
# test/CMakeLists.txt (i.e. as part of ./install.sh -t), BUILD_TESTS is set and
# the parent build has already established the toolchain, project(), and the
# shared RCCL_COMMON_INCLUDE_DIRS / RCCL_COMMON_COMPILE_DEFS / MICRO_TEST_LINK_LIBS
# variables; the rccl target supplies USE_AMDSMI, the fabric probes, and the
# ENABLE_* variant defines, so the per-mode variables below are trivial.
#
# When invoked directly (cmake -B build on this directory) BUILD_TESTS is not
# defined, so the standalone toolchain + project() are set up here and
# rccl_standalone_setup() builds rccl-HostUnitTests + the wrapper library and
# fills in the same per-mode variables the in-build branch sets inline.
# =====================================================================
if(BUILD_TESTS)
  set(RCCL_HOST_HIPIFY_DIR "${PROJECT_BINARY_DIR}/hipify")
  set(_MICRO_MODE_DEFS "")
  set(_ENQ_BUILD_DEVLINKER FALSE)
  set(_MICRO_BUILD_SYM_KERNELS ${GENERATE_SYM_KERNELS})
else()
  # --- hipcc is mandatory (must be resolved before project() at top level) ---
  # Auto-detect hipcc via ROCM_PATH, or pass -DCMAKE_CXX_COMPILER=hipcc.
  if(NOT DEFINED CMAKE_CXX_COMPILER OR CMAKE_CXX_COMPILER STREQUAL "")
    if(NOT ROCM_PATH)
      if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "$ENV{ROCM_PATH}")
      else()
        set(ROCM_PATH "/opt/rocm")
      endif()
    endif()
    find_program(_hipcc hipcc HINTS ${ROCM_PATH} PATH_SUFFIXES bin NO_CMAKE_FIND_ROOT_PATH)
    if(NOT _hipcc)
      find_program(_hipcc hipcc)
    endif()
    if(_hipcc)
      set(CMAKE_CXX_COMPILER "${_hipcc}")
    else()
      message(FATAL_ERROR
        "hipcc not found. Either:\n"
        "  cmake -B build -DCMAKE_CXX_COMPILER=/path/to/hipcc\n"
        "  cmake -B build -DROCM_PATH=/opt/rocm-6.x")
    endif()
  endif()

  # hipcc needs --gcc-install-dir to find C++ standard headers.
  find_program(_system_gxx g++ REQUIRED)
  execute_process(
    COMMAND ${_system_gxx} -print-file-name=libstdc++.a
    OUTPUT_VARIABLE _gcc_libstdcpp
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  get_filename_component(_gcc_install_dir "${_gcc_libstdcpp}" DIRECTORY)
  set(CMAKE_CXX_FLAGS_INIT "--gcc-install-dir=${_gcc_install_dir} --offload-host-only")

  project(rccl-HostUnitTests LANGUAGES CXX)

  # hipcc prepends -O3 and CMake's Debug flags carry no -O, so Debug built at -O3.
  # Appended, not prepended: the last -O on the command line wins.
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -O0")

  option(HOST_TEST_CCACHE "Route host-test compiles through ccache when available" ON)
  if(HOST_TEST_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
      message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
      set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
  endif()

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  add_compile_options(-Wno-tautological-constant-out-of-range-compare)

  # Wrapper around add_executable() that also records the target in
  # HOST_TEST_BINARIES, so the coverage function instruments every host-only test
  # binary without a separately-maintained list. A macro (not a function) so
  # list(APPEND) mutates the caller's variable. Use this instead of
  # add_executable() for every test binary in the standalone build.
  macro(add_test_executable _name)
    list(APPEND HOST_TEST_BINARIES ${_name})
    add_executable(${_name} ${ARGN})
  endmacro()

  rccl_standalone_setup()
endif()

# =====================================================================
# Shared micro-test binary definitions (identical for BOTH build paths).
#
# The only per-mode inputs are the variables set above:
#   RCCL_HOST_HIPIFY_DIR / _MICRO_MODE_DEFS / _ENQ_BUILD_DEVLINKER /
#   _MICRO_BUILD_SYM_KERNELS
# plus the mode-specific include dirs / base defs / link libs applied inside
# rccl_add_micro_binary() itself (BUILD_TESTS branch vs standalone branch).
# CATEGORIES_YAML is honoured in-build and ignored standalone; NO_PIE is honoured
# standalone and ignored in-build -- so both are passed unconditionally.
# =====================================================================
rccl_define_micro_source_lists()

# Mark every hipified misc/ oracle TU any target below compiles as GENERATED.
# The property is directory-scoped and idempotent, so one call up front covers
# all targets -- no need to repeat it per binary.
rccl_mark_oracle_tus_generated(${RCCL_HOST_HIPIFY_DIR} argcheck archinfo utils)

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicro
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicro NO_PIE
  SOURCES ${RCCL_MICRO_TEST_SOURCES}
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/devcomm              # devcomm_v22902.h, included unqualified
    ${RCCL_HOST_HIPIFY_DIR}/src/include/nccl_device  # device.h includes rccl_ptr.h unqualified
  EXTRA_DEFS ${_MICRO_MODE_DEFS}
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro.yaml")

# real IsArchMatch, for wrap-test.cc (path is build-directory-relative).
target_sources(rccl-UnitTestsMicro PRIVATE
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc)

target_compile_definitions(rccl-UnitTestsMicro PRIVATE
  P2P_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/p2p_tmp.cc"
  RAS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/ras.cc"
  TUNING_GENERAL_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/tuning_general.cc"
  RAS_CLIENT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/client.cc"
  RMA_PROXY_PROGRESS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_proxy_progress.cc"
  # rma_tmp.cc, not rma.cc: src/CMakeLists.txt lists plugin/rma.cc first, so
  # add_file_unique() renames this one. Re-check after any source-list reorder.
  RMA_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_tmp.cc"
  # rma_ce.cc has no basename collision, so it keeps its own name.
  RMA_CE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_ce.cc"
  GROUP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/group.cc"
  DEVCOMM_V22902_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/devcomm/devcomm_v22902.cc"
  DEVCOMM_V22907_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/devcomm/devcomm_v22907.cc"
  WRAP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rccl_wrap.cc"
  RCCL_TRANSPORT_STUBS_OMIT_COLLECTIVE_FLOOR
  DEV_RUNTIME_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/dev_runtime.cc"
  DEV_RUNTIME_SEGMENTS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/dev_runtime_segments.cc"
  # plugin/gin.cc is the first gin.cc in SRC_FILES, so hipify keeps the name.
  GIN_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/gin.cc"
)

# sym_kernels_host.cc only exists when sym kernels are generated (GENERATE_SYM_KERNELS
# in-build / present on disk standalone).
if(_MICRO_BUILD_SYM_KERNELS)
  target_sources(rccl-UnitTestsMicro PRIVATE sym-kernels-index-test.cc)
  target_compile_definitions(rccl-UnitTestsMicro PRIVATE
    SYM_KERNELS_HOST_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/gensrc/symmetric/sym_kernels_host.cc")
elseif(NOT BUILD_TESTS)
  message(WARNING "Skipping sym-kernels-index-test.cc for rccl-UnitTestsMicro: missing ${RCCL_HOST_HIPIFY_DIR}/gensrc/symmetric/sym_kernels_host.cc")
endif()

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroWarpSpeed -- tests for ENABLE_WARP_SPEED
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroWarpSpeed NO_PIE
  SOURCES ${TEST_MICRO_SCHEDULER_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ENABLE_WARP_SPEED
    # Not inherited via RCCL_COMMON_COMPILE_DEFS (unlike ENABLE_FAULT_INJECTION); the default-ON, shipping config.
    GENERATE_SYM_KERNELS
    # Hipify renames this to *_tmp.cc: src/enqueue/task_sched/allgatherv_sched.cc keeps the basename instead.
    ALLGATHERV_SCHED_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/scheduler/allgatherv_sched_tmp.cc"
    SYMMETRIC_SCHED_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/scheduler/symmetric_sched.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_scheduler.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroInit and variants
#
# Three binaries, not one: init.cc gates HIP_HOST_UNCACHED_MEMORY/
# HIP_UNCACHED_MEMORY and, separately, ENABLE_FAULT_INJECTION at the preprocessor,
# so each arm's code is absent from the object unless its macro is set. The two
# macro pairs are lexically disjoint in init.cc, so base + -uncached + -faultinj
# reaches every arm without a 2x2 cross product. In-build strips the three
# inherited macros via REMOVE_DEFS (standalone's base defs never had them, so it
# is a no-op there) and each variant sets its own below.
# ---------------------------------------------------------------------------
foreach(_variant "" "-uncached" "-faultinj")
  set(_init_tgt "rccl-UnitTestsMicroInit${_variant}")
  rccl_add_micro_binary(${_init_tgt} NO_PIE
    SOURCES ${TEST_MICRO_INIT_SOURCE_FILES}
    REMOVE_DEFS ENABLE_FAULT_INJECTION HIP_HOST_UNCACHED_MEMORY HIP_UNCACHED_MEMORY
    EXTRA_DEFS ${_MICRO_MODE_DEFS} INIT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/init.cc"
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_init.yaml")
  if(_variant STREQUAL "-uncached")
    target_compile_definitions(${_init_tgt} PRIVATE HIP_HOST_UNCACHED_MEMORY HIP_UNCACHED_MEMORY)
  elseif(_variant STREQUAL "-faultinj")
    target_compile_definitions(${_init_tgt} PRIVATE ENABLE_FAULT_INJECTION)
  endif()
endforeach()

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroEnqueue (+ shipping -devlinker arm standalone)
#
# enqueue.cc gates rcclShmemDynamicSize at the PREPROCESSOR on RCCL_DEVICE_LINKER,
# so a single compile only ever reaches one arm. In-build inherits RCCL_DEVICE_LINKER
# (ENABLE_DEVICE_LINKER default ON) into the base binary; standalone has no rccl
# target to read, so it also builds the -devlinker arm explicitly.
# ---------------------------------------------------------------------------
set(_enq_common_args
  SOURCES ${TEST_MICRO_ENQUEUE_SOURCE_FILES}
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/device               # device/common.h (neutered)
    ${RCCL_HOST_HIPIFY_DIR}/src/include/nccl_device
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    # Per-symbol omissions from the shared nccl_stubs.cc; enqueue.cc itself
    # supplies both, so neither needs a replacement definition.
    RCCL_STUBS_OMIT_ncclInitKernelsForDevice
    RCCL_STUBS_OMIT_ncclParamGraphStreamOrdering
    ENQUEUE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/enqueue.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_enqueue.yaml"
  NO_PIE)

rccl_add_micro_binary(rccl-UnitTestsMicroEnqueue ${_enq_common_args})
if(_ENQ_BUILD_DEVLINKER)
  rccl_add_micro_binary(rccl-UnitTestsMicroEnqueue-devlinker ${_enq_common_args})
  target_compile_definitions(rccl-UnitTestsMicroEnqueue-devlinker PRIVATE RCCL_DEVICE_LINKER)
endif()

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroSymKernels -- REAL src/sym_kernels.cc + generated host TU.
# Reuses rccl-UnitTestsMicro's yaml: its "Sym*" pattern already covers this
# binary's suites too.
# ---------------------------------------------------------------------------
if(_MICRO_BUILD_SYM_KERNELS)
  rccl_add_micro_binary(rccl-UnitTestsMicroSymKernels NO_PIE
    SOURCES ${TEST_MICRO_SYM_KERNELS_SOURCE_FILES}
    EXTRA_DEFS
      ${_MICRO_MODE_DEFS}
      GENERATE_SYM_KERNELS
      SYM_KERNELS_HOST_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/gensrc/symmetric/sym_kernels_host.cc"
      SYM_KERNELS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/sym_kernels.cc"
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro.yaml")
endif()

# =====================================================================
# Standalone-only tail: coverage + no CTest registration.
# =====================================================================
if(BUILD_TESTS)
  # In-RCCL-build path registers these binaries via CTest categories
  # (rccl_register_test_categories, called inside rccl_add_micro_binary).
  return()
endif()

# NOTE: the standalone project intentionally does not register these binaries
# with CTest (matching test/host from #9320); run them directly out of the build
# directory. The binaries are recorded via add_test_executable() /
# rccl_add_micro_binary() rather than a list repeated here, so the two cannot
# drift.
rccl_standalone_coverage()
