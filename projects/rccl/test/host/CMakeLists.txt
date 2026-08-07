cmake_minimum_required(VERSION 3.16)

# =====================================================================
# In-RCCL-build integration.
#
# When this directory is pulled in via add_subdirectory(host) from
# test/CMakeLists.txt (i.e. as part of ./install.sh -t), BUILD_TESTS is
# set and the parent build has already established the toolchain,
# project(), and the shared RCCL_COMMON_INCLUDE_DIRS /
# RCCL_COMMON_COMPILE_DEFS / MICRO_TEST_LINK_LIBS variables. In that mode
# we only build the host-only micro-test binary(ies) here and return
# before the standalone project() setup below.
#
# When invoked directly (cmake -B build on this directory) BUILD_TESTS is
# not defined, so we fall through to the standalone project that builds
# rccl-HostUnitTests + rccl-UnitTestsMicro from a hipified snapshot.
# =====================================================================
if(BUILD_TESTS)
  # rccl-UnitTestsMicro: standalone unit-test binary compiled directly into
  # the test binary. This lets us use preprocessor and link-time substitution
  # to provide test doubles for anything that would require a GPU or other
  # special hardware. Deliberately NOT linking librccl.so or the HIP runtime,
  # so any un-shimmed HIP call surfaces as a link error rather than binding
  # libamdhip64.so (HIP compile requirements are applied below without linking).
  set(TEST_MICRO_SOURCE_FILES
    p2p-test.cc
    fakes/nccl_fakes.cc                  # reusable stubs for nccl* symbols
    fakes/p2p_fakes.cc                   # p2p-specific stubs (arch/topo + alloc emulators)
    fakes/hip_fakes.cc                   # HIP runtime seams + link-satisfying stubs
    ../common/main_altrsmi.cpp           # minimal gtest main, no RCCL deps
    ../common/ProcessIsolatedTestRunner.cpp
  )

  add_executable(rccl-UnitTestsMicro ${TEST_MICRO_SOURCE_FILES})

  target_include_directories(rccl-UnitTestsMicro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RCCL_COMMON_INCLUDE_DIRS})
  target_compile_definitions(rccl-UnitTestsMicro PRIVATE ${RCCL_COMMON_COMPILE_DEFS})

  # Apply HIP's *compile* usage requirements (headers, device intrinsics/codegen,
  # compile defs) WITHOUT linking the HIP runtime. $<COMPILE_ONLY:hip::host> would
  # express this in one line but requires CMake >= 3.27; pulling the interface
  # properties directly keeps this valid at the project's declared minimum (3.16).
  foreach(_hip_tgt hip::host hip::device)
    if(TARGET ${_hip_tgt})
      target_include_directories(rccl-UnitTestsMicro PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_INCLUDE_DIRECTORIES>)
      target_compile_definitions(rccl-UnitTestsMicro PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_COMPILE_DEFINITIONS>)
      target_compile_options(rccl-UnitTestsMicro PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_COMPILE_OPTIONS>)
    endif()
  endforeach()

  target_link_libraries(rccl-UnitTestsMicro PRIVATE ${MICRO_TEST_LINK_LIBS})

  # Pass the absolute path of hipified source files through as string
  # literals so the #include in the test translation unit stays independent of
  # the build directory layout.
  target_compile_definitions(rccl-UnitTestsMicro PRIVATE
    P2P_CC_PATH="${PROJECT_BINARY_DIR}/hipify/src/transport/p2p_tmp.cc"
  )

  # Make sure the hipify step has run before trying to compile the test.
  add_dependencies(rccl-UnitTestsMicro hipify_all copy_nccl_device_headers)

  # Always build this binary with llvm source-based coverage.
  # Restrict the instrumentation to the host compilation pass: applying it to the
  # amdgcn device pass pulls in LLVM profiling runtime symbols
  # (__llvm_profile_raw_version, __start___llvm_prf_cnts, ...) that don't exist on
  # the device, which makes amdgcn-link fail.
  #
  # The "SHELL:" prefix keeps each "-Xarch_device <flag>" pair intact: without
  # it CMake de-duplicates the repeated -Xarch_device token, which would leave
  # -fno-coverage-mapping unqualified and silently drop __llvm_covmap from the
  # host pass too.
  target_compile_options(rccl-UnitTestsMicro PRIVATE
    -fprofile-instr-generate -fcoverage-mapping
    "SHELL:-Xarch_device -fno-profile-instr-generate"
    "SHELL:-Xarch_device -fno-coverage-mapping")
  target_link_options(rccl-UnitTestsMicro PRIVATE
    -fprofile-instr-generate -fcoverage-mapping)

  if(BUILD_ADDRESS_SANITIZER)
    target_compile_options(rccl-UnitTestsMicro PRIVATE -fsanitize=address)
    target_link_options(rccl-UnitTestsMicro PRIVATE -fsanitize=address -shared-libasan)
    if(DEFINED ASAN_RUNTIME_DIR)
      target_link_options(rccl-UnitTestsMicro PRIVATE "LINKER:-rpath,${ASAN_RUNTIME_DIR}")
    endif()
  endif()

  # Deliberately NOT linking librccl.so: this binary is for tests that compile
  # their unit-under-test source file(s) directly. Add files to
  # TEST_MICRO_SOURCE_FILES above as new units come under test.
  rocm_install(TARGETS rccl-UnitTestsMicro COMPONENT tests)

  # Register with CTest exactly as rccl-UnitTests / rccl-UnitTestsFixtures do:
  # via RCCL's shared category mechanism (no bare add_test, matching the rest of
  # test/CMakeLists.txt). apply_test_category_labels is defined by
  # shared/ctest/TestCategories.cmake, which the parent test/CMakeLists.txt
  # include()s (globally) when building inside rocm-systems; INSTALL_TEST_FILE is
  # likewise set by the parent. Guard on COMMAND so a non-monorepo configure
  # (shared/ctest absent) still succeeds -- there, as with the other binaries,
  # the target is built but not category-registered.
  if(COMMAND apply_test_category_labels)
    set(_micro_categories_yaml "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro.yaml")
    if(EXISTS "${_micro_categories_yaml}")
      message(STATUS "Applying test categories for rccl-UnitTestsMicro")
      apply_test_category_labels(
        rccl-UnitTestsMicro
        "${_micro_categories_yaml}"
        "${PROJECT_BINARY_DIR}"
        "${INSTALL_TEST_FILE}"
      )
    else()
      message(WARNING "Skipping test categories for rccl-UnitTestsMicro: missing ${_micro_categories_yaml}")
    endif()
  endif()

  return()
endif()

# --- hipcc is mandatory ---
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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wno-tautological-constant-out-of-range-compare)

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

# --- Resolve fmt: system -> FetchContent fallback ---
find_package(fmt QUIET)
if(NOT fmt_FOUND)
  message(STATUS "Downloading fmt via FetchContent")
  FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.2.1
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(fmt)
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
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
check_cxx_source_compiles("
  #include <amd_smi/amdsmi.h>
  int main() {
    amdsmi_fabric_info_t info{};
    (void)info;
    return 0;
  }
" AMDSMI_FABRIC_API)
set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
if(AMDSMI_FABRIC_API)
  message(STATUS "amdsmi fabric API found - enabling AMDSMI_FABRIC_DIRECT")
else()
  message(STATUS "amdsmi fabric API not found (AMDSMI_FABRIC_DIRECT disabled)")
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

# --- Wrapper library for production .cc files that need link stubs ---
add_library(rccl-source-wrappers STATIC
  mem_manager_src_wrapper.cc
  mem_manager_wrapper_test.cc
  bootstrap_wrapper.cc
  init_stubs.cpp
  wrapper_link_stubs.cpp
)
target_include_directories(rccl-source-wrappers PRIVATE ${RCCL_INCLUDE_DIRS})
target_compile_definitions(rccl-source-wrappers PRIVATE
  __HIP_PLATFORM_AMD__
  NCCL_OS_LINUX
  ROCM_VERSION=70000
  $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
)
target_link_libraries(rccl-source-wrappers PRIVATE
  GTest::GTest
  fmt::fmt-header-only
)

# --- Main test executable ---
add_executable(rccl-HostUnitTests
  main.cpp
  ${RCCL_TEST_DIR}/BitOpsTests.cpp
  ${RCCL_TEST_DIR}/IommuPassthrough_test.cpp
  ${RCCL_TEST_DIR}/VersionInfoTests.cpp
  ${RCCL_TEST_DIR}/DdaCollCommonTests.cpp
  ${RCCL_TEST_DIR}/RomeTopoConsensusTests.cpp
  ${RCCL_TEST_DIR}/MiscTests.cpp
  ${RCCL_TEST_DIR}/VersionGateTests.cpp
  ${RCCL_TEST_DIR}/TimeoutTests.cpp
  ${RCCL_TEST_DIR}/EnqueueCountTests.cpp
  ${RCCL_TEST_DIR}/BootstrapBidirTests.cpp
  ${RCCL_TEST_DIR}/AltRsmiTests.cpp
  ${RCCL_TEST_DIR}/common/ProcessIsolatedTestRunner.cpp
  ${RCCL_HIPIFY_DIR}/src/misc/kernel_config.cc
  ${RCCL_HIPIFY_DIR}/src/misc/alt_rsmi.cc
  ${RCCL_HIPIFY_DIR}/src/graph/rome_topo_consensus.cc
)

target_include_directories(rccl-HostUnitTests PRIVATE ${RCCL_INCLUDE_DIRS})

target_compile_definitions(rccl-HostUnitTests PRIVATE
  __HIP_PLATFORM_AMD__
  NCCL_OS_LINUX
  ROCM_VERSION=70000
  RCCL_EXPOSE_STATIC
  ARSMI_TEST_BUILD
  $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
)

target_link_options(rccl-HostUnitTests PRIVATE -no-hip-rt)

target_link_libraries(rccl-HostUnitTests PRIVATE
  GTest::GTest
  Threads::Threads
  fmt::fmt-header-only
  -Wl,--whole-archive
  rccl-source-wrappers
  -Wl,--no-whole-archive
)

# --- Standalone micro-test binary ---------------------------------------
# rccl-UnitTestsMicro: same host-only model as rccl-HostUnitTests (hipcc
# --offload-host-only, -no-hip-rt), built from the hipified snapshot. It
# #includes the unit-under-test production .cc directly (via P2P_CC_PATH) and
# satisfies every HIP/nccl symbol from fakes/ + micro_link_stubs.cc rather
# than librccl.so or the HIP runtime.
add_executable(rccl-UnitTestsMicro
  p2p-test.cc
  fakes/nccl_fakes.cc
  fakes/p2p_fakes.cc
  fakes/hip_fakes.cc
  micro_link_stubs.cc
  ${RCCL_TEST_DIR}/common/main_altrsmi.cpp
  ${RCCL_TEST_DIR}/common/ProcessIsolatedTestRunner.cpp
)

target_compile_options(rccl-UnitTestsMicro PRIVATE
  -Wno-unused-parameter -Wno-unused-variable)

target_include_directories(rccl-UnitTestsMicro PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${RCCL_INCLUDE_DIRS}
)

target_compile_definitions(rccl-UnitTestsMicro PRIVATE
  __HIP_PLATFORM_AMD__
  NCCL_OS_LINUX
  ROCM_VERSION=70000
  P2P_CC_PATH="${RCCL_HIPIFY_DIR}/src/transport/p2p_tmp.cc"
)

target_link_options(rccl-UnitTestsMicro PRIVATE -no-hip-rt)
target_link_libraries(rccl-UnitTestsMicro PRIVATE
  GTest::GTest
  Threads::Threads
  fmt::fmt-header-only
)

# llvm source-based coverage (host-only build: no -Xarch_device split needed
# because --offload-host-only never runs the amdgcn device pass).
# On by default; disable with -DMICRO_COVERAGE=OFF.
option(MICRO_COVERAGE "Build rccl-UnitTestsMicro with llvm source-based coverage" ON)
if(MICRO_COVERAGE)
  target_compile_options(rccl-UnitTestsMicro PRIVATE -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(rccl-UnitTestsMicro PRIVATE -fprofile-instr-generate -fcoverage-mapping)
endif()

# NOTE: the standalone project intentionally does not register these binaries
# with CTest (matching test/host from #9320); run them directly, e.g.
#   ./build/rccl-UnitTestsMicro   or   ./build/rccl-HostUnitTests
# The in-RCCL-build path registers rccl-UnitTestsMicro via CTest categories.
