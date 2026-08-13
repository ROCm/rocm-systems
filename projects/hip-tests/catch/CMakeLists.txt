# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)

if (HIP_PLATFORM MATCHES "nvidia")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  set(GPGPU_LANGUAGE CUDA)
  set(GPGPU_LINKER_LIBRARIES hip::host hip::device cuda)
elseif (HIP_PLATFORM MATCHES "amd")
  set(GPGPU_LANGUAGE HIP)
  set(GPGPU_LINKER_LIBRARIES hip::host hip::device)
endif()
project(hiptests LANGUAGES C CXX ${GPGPU_LANGUAGE})
enable_language(${GPGPU_LANGUAGE})

if (HIP_PLATFORM MATCHES "nvidia")
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_HIP_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
endif()

option(ENABLE_ADDRESS_SANITIZER "Option to enable ASAN build" OFF)
option(BUILD_SHARED_LIBS "Option for testing shared libraries" ON)
option(BUILD_UNIT_TESTS "Build unit tests" ON)
option(BUILD_PERF_TESTS "Build perf tests" OFF)
option(BUILD_STRESS_TESTS "Build stress tests" OFF)
option(BUILD_CONTRACT_TESTS "Build semantic contract tests" ON)
option(TEST_CLOCK_CYCLE "Option to use clock64" OFF)
option(RTC_TESTING "Run tests using HIP RTC to compile the kernels" OFF)
option(ENABLE_SPIRV "Build hip-tests for SPIRV" OFF)
option(ENABLE_YAML_TAGS "Enable YAML tags for test cases" ON)
option(STANDALONE_TESTS "Generate standalone executable per source file" OFF)
option(ENABLE_OOB_TESTS "Build OOB tests, deliberately wrong inputs" ON)

# Build tests with -Wall -Werror. Defaults to OFF
set(_hiptests_werror_default OFF)
option(HIP_TESTS_ENABLE_WERROR "Build tests with -Wall -Werror (Linux default ON)" ${_hiptests_werror_default})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../../lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(hip REQUIRED)
include_directories(${hip_INCLUDE_DIRS})
find_package(hiprtc)
find_package(Git)
find_package(Python 3 COMPONENTS Interpreter REQUIRED)
find_program(HIPCONFIG_EXEC hipconfig REQUIRED)

# hip root dir is 3 levels down the hip-config.cmake file path
set(HIP_PATH ${hip_DIR}/../../../)
if(NOT DEFINED ROCM_PATH)
  # Derive ROCM_PATH from the HIP location instead of hardcoding to /opt/rocm
  set(ROCM_PATH ${HIP_PATH})
endif()

if(UNIX)
  if(NOT ENABLE_SPIRV)
    find_program(ROCM_AGENT_ENUMERATOR_EXEC rocm_agent_enumerator HINTS ${ROCM_PATH}/bin ${HIP_PATH}/bin)
  endif()
  find_program(GCC_EXEC gcc)
  find_program(GXX_EXEC g++)
else() # Win32
  find_program(LLVM_RC_EXEC NAMES llvm-rc llvm-rc.exe
                 PATHS ${HIP_PATH} ${HIP_PATH}../lc/bin ${ROCM_PATH})
endif()

message(STATUS "HIP Lib path: ${HIP_LIB_INSTALL_DIR}")

# used by clang to find hip
set(HIP_PATH_OPT "--hip-path=${HIP_PATH}")

# Make hipcc path available to tests that need hipcc
if(HIP_HIPCC_EXECUTABLE)
  add_definitions(-DHIP_HIPCC_EXECUTABLE="${HIP_HIPCC_EXECUTABLE}")
endif()

if(TEST_CLOCK_CYCLE)
  add_definitions(-DTEST_CLOCK_CYCLE)
endif()

if(ENABLE_YAML_TAGS)
  add_definitions(-DENABLE_YAML_TAGS)
endif()

if(NOT WIN32)
  set(CPACK_SET_DESTDIR ON CACHE BOOL "Installer package will install hip catch to CMAKE_INSTALL_PREFIX instead of CPACK_PACKAGING_INSTALL_PREFIX")
endif()

# Check if platform is set
if(NOT HIP_PLATFORM STREQUAL "amd" AND NOT HIP_PLATFORM STREQUAL "nvidia")
  message(FATAL_ERROR "Unexpected HIP_PLATFORM: " ${HIP_PLATFORM})
endif()

execute_process(COMMAND ${HIPCONFIG_EXEC} --version
                OUTPUT_VARIABLE HIP_VERSION
                RESULT_VARIABLE result
                OUTPUT_STRIP_TRAILING_WHITESPACE)

if(result AND NOT result EQUAL 0)
  message(FATAL_ERROR "Failure trying to obtain HIP version via command: ${HIPCONFIG_EXEC} --version: ${result}")
endif()

if(NOT WIN32)
  if(NOT BUILD_SHARED_LIBS)
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/llvm)
    find_package(amd_comgr)
    find_package(hsa-runtime64)
    link_libraries(hsa-runtime64::hsa-runtime64)
    link_libraries(amd_comgr)
    link_libraries(hiprtc-builtins)
  endif() # end BUILD_SHARED_LIBS
endif() # end win32

# Address sanitizer options
if(NOT DEFINED THEROCK_SANITIZER)
  set(THEROCK_SANITIZER)
endif()

set(ASAN_CONFIG_DEF)
set(HIP_TESTS_ASAN_ENABLED OFF)
if(ENABLE_ADDRESS_SANITIZER OR THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  set(HIP_TESTS_ASAN_ENABLED ON)
  message(STATUS "Building catch tests with Address Sanitizer options")
  # If user is building hip-tests standalone (without TheRock) and wants to turn on sanitizer
  # We append the flags manually, otherwise we use TheRock's CXX flags which already should have
  # `-fsanitize=address` in it.
  if(ENABLE_ADDRESS_SANITIZER)
    # We support clang for this
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR NOT CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
      message(WARNING "CXX and HIP compiler are not Clang, sanitizer build might fail")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer -shared-libasan -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -shared-libasan -g")
  endif()
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fsanitize=address -fno-omit-frame-pointer -shared-libasan -g")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -shared-libasan")
  add_compile_definitions(ENABLE_ADDRESS_SANITIZER)
  set(ASAN_CONFIG_DEF "-DENABLE_ADDRESS_SANITIZER")
endif()

string(REPLACE "." ";" VERSION_LIST ${HIP_VERSION})
list(GET VERSION_LIST 0 HIP_VERSION_MAJOR)
list(GET VERSION_LIST 1 HIP_VERSION_MINOR)
list(GET VERSION_LIST 2 HIP_VERSION_PATCH_GITHASH)
string(REPLACE "-" ";" VERSION_LIST ${HIP_VERSION_PATCH_GITHASH})
list(GET VERSION_LIST 0 HIP_VERSION_PATCH)
list(GET VERSION_LIST 1 HIP_VERSION_GITHASH)

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(HIP_PACKAGING_VERSION_PATCH ${HIP_VERSION_PATCH}.$ENV{ROCM_LIBPATCH_VERSION})
else()
  set(HIP_PACKAGING_VERSION_PATCH ${HIP_VERSION_PATCH}-${HIP_VERSION_GITHASH})
endif()

find_package(Catch2 3.8.1 QUIET)
if(Catch2_FOUND)
  message(STATUS "Catch2 v3.8.1 found: ${Catch2_DIR}")
  # for packaging
  set(CATCH_ADD_TESTS_SCRIPT ${Catch2_DIR}/CatchAddTests.cmake)
  file(GLOB CATCH_LIBS "${Catch2_DIR}/../../*Catch2*")
else()
  # On Windows, ensure Catch2 uses the same MSVC runtime library and build configuration as our tests
  if(WIN32)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "MSVC runtime library" FORCE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(win_disable_warnings -fms-runtime-lib=static_dbg)
    endif()
  endif()
  # Setup catch2 manually
  include(FetchContent)
  FetchContent_Declare(
        Catch2
        DOWNLOAD_EXTRACT_TIMESTAMP OFF
        URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/Catch2-3.8.1.tar.gz"
        URL_HASH SHA256=18b3f70ac80fccc340d8c6ff0f339b2ae64944782f8d2fca2bd705cf47cadb79
    )
  FetchContent_MakeAvailable(Catch2)
  # Required for MSVC to avoid -Werror-ing Catch2 code
  foreach(_catch2_target Catch2 Catch2WithMain)
   if(TARGET ${_catch2_target})
      get_target_property(_incdirs ${_catch2_target} INTERFACE_INCLUDE_DIRECTORIES)
      if(_incdirs)
        set_target_properties(${_catch2_target} PROPERTIES
          INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_incdirs}")
      endif()
    endif()
  endforeach()
  # for packaging
  set(CATCH_ADD_TESTS_SCRIPT ${Catch2_SOURCE_DIR}/extras/CatchAddTests.cmake)
  set(CATCH_LIBS $<TARGET_FILE:Catch2>)
endif()

include(cmake/hip-tests.cmake)
include(CTest)

set(CATCH_BUILD_DIR catch_tests)

set(CATCH_INCLUDE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/include)
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CATCH_INCLUDE_BINARY_DIR})

include_directories(
    "./include"
    ${CATCH_INCLUDE_BINARY_DIR}
    "./kernels"
    "${CMAKE_CURRENT_LIST_DIR}/external/picojson"
)

if (RTC_TESTING)
  add_definitions(-DRTC_TESTING=ON)
endif()
add_definitions(-DKERNELS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")

if (WIN32)
  configure_file(catchProp_in_rc.in ${CMAKE_CURRENT_BINARY_DIR}/catchProp.rc @ONLY)

  # generates the .res files to be used by executables to populate the properties
  # expects LC folder with clang, llvm-rc to be present one level up of HIP
  execute_process(COMMAND ${LLVM_RC_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/catchProp.rc
                  OUTPUT_VARIABLE RC_OUTPUT)
  set(PROP_RC ${CMAKE_CURRENT_BINARY_DIR})
  # When args to linker exceeds max chars.
  # msbuild writes args to a rsp file.
  # This is used to reference the obj file correctly
  SET(CMAKE_C_RESPONSE_FILE_LINK_FLAG "")
  SET(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "")
endif()

# Add debug and optimization flags for Windows builds
if(WIN32)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options($<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:-g>)
        add_compile_options($<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:-gcodeview>)
        add_compile_options($<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Release>>:-O2>)
        add_compile_options($<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:RelWithDebInfo>>:-O2>)
        add_compile_options($<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Debug>>:-O0>)
    endif()
    if(CMAKE_HIP_COMPILER_ID STREQUAL "Clang" OR HIP_PLATFORM STREQUAL "amd")
        add_compile_options($<$<AND:$<COMPILE_LANGUAGE:HIP>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:-g>)
        add_compile_options($<$<AND:$<COMPILE_LANGUAGE:HIP>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:-gcodeview>)
        add_compile_options($<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:Release>>:-O2>)
        add_compile_options($<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:RelWithDebInfo>>:-O2>)
        add_compile_options($<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:Debug>>:-O0>)
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_HIP_COMPILER_ID STREQUAL "Clang" OR HIP_PLATFORM STREQUAL "amd")
        add_link_options($<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:-Xlinker>)
        add_link_options($<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:/DEBUG:FULL>)
    endif()
endif()

if(HIP_PLATFORM STREQUAL "amd")
  if(WIN32)
    # clang-cl emits -Wignored-attributes for __declspec on HIP types, -Wmicrosoft-cast for
    # HANDLE/void* implicit casts, and -Wunknown-argument for HIP-specific driver flags
    set(win_disable_warnings ${win_disable_warnings} -Wno-ignored-attributes -Wno-microsoft-cast -Wno-unknown-argument)
    # to bypass unsafe printf warning in HIP_CHECK macro
    set(win_disable_warnings ${win_disable_warnings} -Wno-unsafe-buffer-usage-in-libc-call)
    # C++98 compat flags prohibits using nullptr
    set(win_disable_warnings ${win_disable_warnings} -Wno-c++98-compat -Wno-c++98-compat-pedantic)
    # Padding is ABI-informational only; reordering test struct fields would hurt readability
    set(win_disable_warnings ${win_disable_warnings} -Wno-padded)
    # Build generates header copies into build/catch_tests/include; both copies are identical
    set(win_disable_warnings ${win_disable_warnings} -Wno-shadow-header)
    # Meyer's singletons in headers are intentional test-framework pattern
    set(win_disable_warnings ${win_disable_warnings} -Wno-exit-time-destructors -Wno-global-constructors)
    # Same singletons trigger unique-object-duplication due to hidden visibility
    set(win_disable_warnings ${win_disable_warnings} -Wno-unique-object-duplication)
    # Advisory-only suggestions from experimental clang lifetime analysis passes
    set(win_disable_warnings ${win_disable_warnings} -Wno-lifetime-safety-intra-tu-suggestions -Wno-lifetime-safety-cross-tu-suggestions)
    # GET_TAGS macro passes empty '...' — GNU extension, widely used in test frameworks
    set(win_disable_warnings ${win_disable_warnings} -Wno-variadic-macro-arguments-omitted)
    # Catch2 listener classes defined entirely in one .cc file; weak vtable is safe there
    set(win_disable_warnings ${win_disable_warnings} -Wno-weak-vtables)
    # Raw-array loops in test helpers have controlled bounds; refactoring to std::span is out of scope
    set(win_disable_warnings ${win_disable_warnings} -Wno-unsafe-buffer-usage)
    # { }-body macros (HIP_CHECK, HIPRTC_CHECK, etc.) are called with trailing ; throughout the codebase;
    # changing them all to do-while(0) is out of scope for this -Werror enablement pass
    set(win_disable_warnings ${win_disable_warnings} -Wno-extra-semi-stmt)
    # -Wswitch-default and -Wcovered-switch-default are mutually exclusive on complete enum switches:
    # the former requires a default, the latter forbids it when all values are covered
    set(win_disable_warnings ${win_disable_warnings} -Wno-switch-default)
    # assemblyFile_Verification is a template defined in a shared header; TUs that include it
    # but don't instantiate it trigger -Wunused-template, which we cannot avoid without
    # splitting the header
    set(win_disable_warnings ${win_disable_warnings} -Wno-unused-template)
    # hipGraphAddDependencies tests take vector references before push_back; the lifetime analysis
    # flags this as invalidation but the references are not used after reallocation
    set(win_disable_warnings ${win_disable_warnings} -Wno-lifetime-safety-invalidation)
    # Test code intentionally compares GPU kernel outputs for exact equality (pass/fail stored as
    # 0.0f/1.0f, or exact integer results converted to float). These are correct by construction.
    set(win_disable_warnings ${win_disable_warnings} -Wno-float-equal)
    # calculateExpected returns a named local; NRVO applicability is compiler-version-dependent
    # but the copy is always correct. This is a missed-optimization advisory only.
    set(win_disable_warnings ${win_disable_warnings} -Wno-nrvo)
    # memcpy3d_tests_common.hh and similar add default: for defensive coding in complete-enum
    # switches; -Wcovered-switch-default conflicts with -Wswitch-default for these cases
    set(win_disable_warnings ${win_disable_warnings} -Wno-covered-switch-default)
    # Test helpers redeclare 'params'/'extent' in nested scopes intentionally; renaming all
    # instances would be large refactoring out of scope for this -Werror pass
    set(win_disable_warnings ${win_disable_warnings} -Wno-shadow)
    # __device__ global arrays in tests don't need extern declarations; they are TU-local symbols
    set(win_disable_warnings ${win_disable_warnings} -Wno-missing-variable-declarations)
    # Catch2 GENERATE() deduces int from integer literals; assigning to size_t/uint32_t is pervasive
    # across many shared test headers and cannot be fixed without touching Catch2 internals
    set(win_disable_warnings ${win_disable_warnings} -Wno-sign-conversion)
    # size_t arithmetic results narrowed to uint32_t for dim3 ctor; systematic in shared headers
    set(win_disable_warnings ${win_disable_warnings} -Wno-shorten-64-to-32)
    # int→float conversions in HipTest::memsetReverse<float> calls via shared headers
    set(win_disable_warnings ${win_disable_warnings} -Wno-implicit-int-float-conversion)
    # hipGraphNodeType enum has values added after some switches were written; not all cases needed
    set(win_disable_warnings ${win_disable_warnings} -Wno-switch-enum)
    # Hundreds of test-local helper functions and __global__ kernels across all test subdirectories
    # lack forward declarations; adding static to each would touch hundreds of files out of scope
    set(win_disable_warnings ${win_disable_warnings} -Wno-missing-prototypes)
    # C-style casts (void**)&ptr, (hipDeviceptr_t)x are pervasive in HIP API call sites and
    # generated/copied CUDA-compat code; rewriting all to reinterpret_cast is out of scope
    set(win_disable_warnings ${win_disable_warnings} -Wno-old-style-cast)
    # std::unique_lock<std::mutex> lk(m) deduced via CTAD; supported in C++17, advisory-only
    set(win_disable_warnings ${win_disable_warnings} -Wno-ctad-maybe-unsupported)
    # Windows backslash paths in Doxygen comment blocks are parsed as unknown \command sequences
    set(win_disable_warnings ${win_disable_warnings} -Wno-documentation-unknown-command)
    # int-to-char narrowing in test helper fill loops; values are bounded by construction
    set(win_disable_warnings ${win_disable_warnings} -Wno-implicit-int-conversion)
    # hipKernelNodeParams.func is void*; comparing with typed function pointer triggers -Wpedantic
    set(win_disable_warnings ${win_disable_warnings} -Wno-pedantic)
    # #define N 1024 and similar test-local constants are unused in some test cases; suppressing
    # pervasive across all test files
    set(win_disable_warnings ${win_disable_warnings} -Wno-unused-macros)
    # Clang lifetime analysis incorrectly flags Catch2 SECTION-scoped delete as use-after-free
    # because it cannot track SECTION control flow; the test logic is correct
    set(win_disable_warnings ${win_disable_warnings} -Wno-lifetime-safety-use-after-free)
    # Static debug-helper functions (printDeviceMem, etc.) left in test files for debugging;
    # removing them is out of scope for this -Werror enablement pass
    set(win_disable_warnings ${win_disable_warnings} -Wno-unused-function)
    # __HIP_NO_IMAGE_SUPPORT may not be defined on all platforms; #ifdef would change semantics
    set(win_disable_warnings ${win_disable_warnings} -Wno-undef)
    # setDefaultData<double> and checkVectors<float> templates use float literals that get
    # promoted to double/long double when T=double/long double; systemic across template tests
    set(win_disable_warnings ${win_disable_warnings} -Wno-double-promotion)
    # __sync_fetch_and_add in hipMemCoherencyTst.cc tests atomic coherency deliberately;
    # seq-cst is intentional for the correctness test
    set(win_disable_warnings ${win_disable_warnings} -Wno-atomic-implicit-seq-cst)
    # %u/__LINE__ and %x/int signedness mismatches in fprintf/printf throughout test helpers;
    # values are bounded and correct, changing all format strings is out of scope
    set(win_disable_warnings ${win_disable_warnings} -Wno-format-signedness)
    # hipMemcpyWithStream.cc has a synchronous hipMemcpy call inside an always-false branch that
    # tests the signature; clang correctly sees it as unreachable, but the branch is intentional
    set(win_disable_warnings ${win_disable_warnings} -Wno-unreachable-code)
    # switch cases in hipMemcpySync.cc end with return + break; the break is unreachable but
    # intentional defensive style. -Wno-unreachable-code does not cover this sub-warning.
    set(win_disable_warnings ${win_disable_warnings} -Wno-unreachable-code-break)
    # launchKernel() doxygen @param entries lack descriptions (numBlocks, numThreads, etc.);
    # fixing the docs is out of scope for this -Werror enablement pass
    set(win_disable_warnings ${win_disable_warnings} -Wno-documentation)
    # float+i assigned to int arrays in test data initializers (e.g. 3.146f + i -> int);
    # fractional truncation is intentional test data, not a logic bug
    set(win_disable_warnings ${win_disable_warnings} -Wno-float-conversion)
    # sqrt(powf(...)) and similar: powf returns float but sqrt promotes to double and result
    # assigned back to float; pervasive in kernel math helpers, not a correctness issue
    set(win_disable_warnings ${win_disable_warnings} -Wno-implicit-float-conversion)
    # MemUtils.hh inline member functions have a trailing ; after the closing brace;
    # -Wno-extra-semi-stmt covers statement-level extra semicolons but not member-function ones
    set(win_disable_warnings ${win_disable_warnings} -Wno-extra-semi)
    # __hip_atomic_store/__hip_atomic_load builtins are deprecated in newer ROCm in favor of
    # __scoped_atomic_store_n/__scoped_atomic_load_n; the uses are in test code, not a correctness issue
    set(win_disable_warnings ${win_disable_warnings} -Wno-hip-deprecated-builtins)
    # Several constexpr variables (e.g. COMMAND_LEN, BUFFER_LEN) are used only in non-Windows
    # code paths guarded by #ifdef; suppressing rather than adding dummy uses
    set(win_disable_warnings ${win_disable_warnings} -Wno-unused-const-variable)
    # multipleUUID_Exe.cc has a 'return' after a complete if/else; the unreachable return is
    # removed at source but -Wno-unreachable-code does not cover the -return sub-warning
    set(win_disable_warnings ${win_disable_warnings} -Wno-unreachable-code-return)
    # hipDeviceSetGetLimit.cc compares size_t (unsigned) against 0; the check is defensive
    # guard code that is always false for unsigned but harmless
    set(win_disable_warnings ${win_disable_warnings} -Wno-tautological-unsigned-zero-compare)
    # Some headers (e.g. threaded_zig_zag_test.hh) lack a trailing newline; fixing all of them
    # is out of scope for this -Werror enablement pass
    set(win_disable_warnings ${win_disable_warnings} -Wno-newline-eof)
    # Windows for some reason does not honor the `CMAKE_CXX_STANDARD` we set initially in the file
    # Which results in errors about missing headers etc
    set(win_cxx_standard -std=c++17)
  endif()
  add_compile_options(${HIP_PATH_OPT} ${win_cxx_standard} -x hip -Wall -Wextra -Wvla -Wno-deprecated ${win_disable_warnings})
  if(HIP_TESTS_ENABLE_WERROR)
      add_compile_options(-Werror)
  endif()
endif()

# Turn off CMAKE_HIP_ARCHITECTURES Feature if cmake version is 3.21+
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21.0)
  set(CMAKE_HIP_ARCHITECTURES OFF)
endif()
message(STATUS "CMAKE HIP ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")

# Note to pass arch use format like -DOFFLOAD_ARCH_STR="--offload-arch=gfx900  --offload-arch=gfx906"
# having space at the start/end of OFFLOAD_ARCH_STR can cause build failures
# Identify the GPU Targets.
# This is done due to limitation of rocm_agent_enumerator
# While building test parallelly, rocm_agent_enumerator can fail and give out an empty target
# That results in hipcc building the test for gfx803 (the default target)
# preference to pass arch -
# OFFLOAD_ARCH_STR
# rocm_agent_enumerator
if(ENABLE_SPIRV)
  set(OFFLOAD_ARCH_STR "--offload-arch=amdgcnspirv")
elseif(DEFINED OFFLOAD_ARCH_STR)
  string(REPLACE "--offload-arch=" "" HIP_GPU_ARCH_LIST ${OFFLOAD_ARCH_STR})
elseif(DEFINED GPU_TARGETS)
  foreach(_hip_gpu_arch ${GPU_TARGETS})
    set(OFFLOAD_ARCH_STR "--offload-arch=${_hip_gpu_arch} ${OFFLOAD_ARCH_STR}")
  endforeach()
elseif(NOT DEFINED OFFLOAD_ARCH_STR
   AND EXISTS "${ROCM_AGENT_ENUMERATOR_EXEC}"
   AND HIP_PLATFORM STREQUAL "amd" AND UNIX)
  execute_process(COMMAND "${ROCM_AGENT_ENUMERATOR_EXEC}"
         OUTPUT_VARIABLE HIP_GPU_ARCH
         RESULT_VARIABLE ROCM_AGENT_ENUM_RESULT
         OUTPUT_STRIP_TRAILING_WHITESPACE)
  # Trim out gfx000
  string(REPLACE "gfx000\n" "" HIP_GPU_ARCH ${HIP_GPU_ARCH})
  if (NOT HIP_GPU_ARCH STREQUAL "")
    string(REGEX REPLACE "\n" ";" HIP_GPU_ARCH_LIST "${HIP_GPU_ARCH}")
    list(REMOVE_DUPLICATES  HIP_GPU_ARCH_LIST)
    list(LENGTH HIP_GPU_ARCH_LIST HIP_GPU_ARCH_LEN)
    set(OFFLOAD_ARCH_STR "")
    foreach(_hip_gpu_arch ${HIP_GPU_ARCH_LIST})
      set(OFFLOAD_ARCH_STR "--offload-arch=${_hip_gpu_arch} ${OFFLOAD_ARCH_STR}")
    endforeach()
  else()
    message(STATUS "ROCm Agent Enumerator found no valid architectures")
  endif()
endif()

# Trim any trailing spaces, this trips up the cxx function call
string(STRIP "${OFFLOAD_ARCH_STR}" OFFLOAD_ARCH_STR)

message(STATUS "Using offload arch string: ${OFFLOAD_ARCH_STR}")
if(DEFINED OFFLOAD_ARCH_STR)
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${OFFLOAD_ARCH_STR}")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" _HIP_BUILD_TYPE_UPPER)
separate_arguments(HIP_DEVICE_BUILD_FLAGS_NO_ARCH NATIVE_COMMAND
  "${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_${_HIP_BUILD_TYPE_UPPER}}")
list(FILTER HIP_DEVICE_BUILD_FLAGS_NO_ARCH EXCLUDE REGEX "--offload-arch=")

# -fsanitize=address does not work with SPIR-V targets so keep a SPIR-V-specific
# copy of the device build flags.
set(HIP_DEVICE_BUILD_FLAGS_NO_ARCH_SPIRV ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH})
if(ENABLE_ADDRESS_SANITIZER OR THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  list(FILTER HIP_DEVICE_BUILD_FLAGS_NO_ARCH_SPIRV EXCLUDE REGEX "-fsanitize=address")
  list(FILTER HIP_DEVICE_BUILD_FLAGS_NO_ARCH_SPIRV EXCLUDE REGEX "-fno-omit-frame-pointer")
  list(FILTER HIP_DEVICE_BUILD_FLAGS_NO_ARCH_SPIRV EXCLUDE REGEX "-shared-libasan")
endif()

# ASan is not supported by SPIR-V, so ASan-related device flags must be dropped in those cases.
if(ENABLE_SPIRV)
  set(HIP_DEVICE_BUILD_FLAGS_NO_ARCH ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH_SPIRV})
endif()

# get hip-tests commit short hash
execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE git_result
  OUTPUT_VARIABLE git_output
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(git_result EQUAL 0)
  set(HIP_TESTS_GITHASH ${git_output})
endif()

set(OFFLOAD_ARCH_LIST ${OFFLOAD_ARCH_STR})
separate_arguments(OFFLOAD_ARCH_LIST)
list(REMOVE_DUPLICATES  OFFLOAD_ARCH_LIST)
set(HIP_GPU_ARCH_LIST "")
foreach(__offload_arch ${OFFLOAD_ARCH_LIST})
    string(REPLACE "--offload-arch=" "" targetid ${__offload_arch})
    string(REPLACE ":" ";" target_id_list ${targetid})
    list(GET target_id_list 0 arch)
    string(APPEND HIP_GPU_ARCH_LIST "${arch};")
    list(REMOVE_DUPLICATES  HIP_GPU_ARCH_LIST)
endforeach()

# config generation
add_subdirectory(config)

# prints the catch info to a file
string(TIMESTAMP _timestamp UTC)
set(_autogen "# Auto-generated by cmake on ${_timestamp} UTC\n")
set(_catchInfo ${_autogen} "HIP_VERSION=${HIP_VERSION}\n")
set(_catchInfo ${_catchInfo} "HIP_PLATFORM=${HIP_PLATFORM}\n")
set(_catchInfo ${_catchInfo} "HIP_TESTS_GITHASH=${HIP_TESTS_GITHASH}\n")
set(_catchInfo ${_catchInfo} "ARCHS=${HIP_GPU_ARCH_LIST}\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/catchInfo.txt ${_catchInfo})

# Let `ctest` run from catch_tests/. CMake only writes CTestTestfile.cmake and
# DartConfiguration.tcl to the build root, so provide both under catch_tests/:
# write a CTestTestfile.cmake that redirects to the parent, and copy
# DartConfiguration.tcl so ctest's configuration (default timeout, etc.) applies.
set(_subdirs ${_autogen} "subdirs(..)\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/CTestTestfile.cmake ${_subdirs})

if(EXISTS "${CMAKE_BINARY_DIR}/DartConfiguration.tcl")
  configure_file(
    "${CMAKE_BINARY_DIR}/DartConfiguration.tcl"
    "${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/DartConfiguration.tcl"
    COPYONLY)
endif()

set(CATCH_SCRIPT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/script)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip-tests.cmake
     DESTINATION ${CATCH_SCRIPT_BINARY_DIR})

# Patch CatchAddTests.cmake in-place so that catch_discover_tests_impl writes
# an empty _tests.cmake when zero tests match the filter, instead of returning
# early without creating the file.  The generated _include.cmake hardcodes the
# path to CatchAddTests.cmake from Catch2's source tree, so we must patch it
# there (not a copy) for the fix to take effect at ctest startup.
# Without this, catch_discover_tests("[disabled]") on a binary with no
# [disabled] tests leaves a dangling _include.cmake whose include() fails:
#   "include could not find requested file: ..._tests.cmake"
# This is a known Catch2 bug fixed upstream in:
#   https://github.com/catchorg/Catch2/pull/3075
# Remove this patch once we update to a Catch2 version that includes that fix.
set(_catch_old_snippet "  if(num_tests STREQUAL \"0\")\n    return()\n  endif()")
set(_catch_new_snippet "  if(num_tests STREQUAL \"0\")\n    file(WRITE \"\${_CTEST_FILE}\" \"# no tests discovered for this spec\\n\")\n    return()\n  endif()")
file(READ "${CATCH_ADD_TESTS_SCRIPT}" _catch_content)
string(FIND "${_catch_content}" "${_catch_new_snippet}" _catch_already_patched)
if(_catch_already_patched EQUAL -1)
  string(REPLACE "${_catch_old_snippet}" "${_catch_new_snippet}" _catch_patched "${_catch_content}")
  if("${_catch_patched}" STREQUAL "${_catch_content}")
    message(FATAL_ERROR
      "Failed to patch CatchAddTests.cmake: expected snippet not found.\n"
      "The Catch2 version may have changed. Check upstream fix:\n"
      "  https://github.com/catchorg/Catch2/pull/3075")
  endif()
  file(WRITE "${CATCH_ADD_TESTS_SCRIPT}" "${_catch_patched}")
endif()
unset(_catch_old_snippet)
unset(_catch_new_snippet)
unset(_catch_content)
unset(_catch_patched)
unset(_catch_already_patched)

file(COPY ${CATCH_ADD_TESTS_SCRIPT}
         DESTINATION ${CATCH_SCRIPT_BINARY_DIR})
set_property(GLOBAL APPEND PROPERTY G_INSTALL_SCRIPT_TARGETS ${CATCH_ADD_TESTS_SCRIPT})

# copy python script and headers to catch test package
file(COPY ./unit/compileAndCaptureOutput.py
            DESTINATION  ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/unit )

file(COPY ./include/hip_test_common.hh DESTINATION ${CATCH_INCLUDE_BINARY_DIR})
file(COPY ./include/hip_test_context.hh DESTINATION ${CATCH_INCLUDE_BINARY_DIR})

# Enable device lambda on nvidia platforms
if(HIP_PLATFORM STREQUAL "nvidia")
  # SPIRV is not supported on it
  if(ENABLE_SPIRV)
    message(FATAL_ERROR "SPIRV is not supported on this platform")
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --extended-lambda")
  add_compile_options(-Xcompiler=-Wno-deprecated-declarations)
endif()

add_custom_target(build_tests ALL)
add_dependencies(build_tests hip_tests_config)

# Tests folder
# kernels and hipTestMain provide the KERNELS and Main_Object object libraries
# referenced by hip_add_exe_to_target. They are required by every test
# executable built through hip_add_exe_to_target (unit, contract, perf, and
# stress), so add them once when any of those test groups is enabled.
if(BUILD_UNIT_TESTS OR BUILD_CONTRACT_TESTS OR BUILD_PERF_TESTS OR BUILD_STRESS_TESTS)
  add_subdirectory(kernels ${CATCH_BUILD_DIR}/kernels)
  add_subdirectory(hipTestMain ${CATCH_BUILD_DIR}/hipTestMain)
endif()
if(BUILD_UNIT_TESTS)
  add_subdirectory(unit ${CATCH_BUILD_DIR}/unit)
  add_subdirectory(ABM ${CATCH_BUILD_DIR}/ABM)
  add_subdirectory(TypeQualifiers ${CATCH_BUILD_DIR}/TypeQualifiers)
  if(UNIX)
    add_subdirectory(multiproc ${CATCH_BUILD_DIR}/multiproc)
  endif()
endif()
if(BUILD_PERF_TESTS)
  add_subdirectory(performance ${CATCH_BUILD_DIR}/performance)
endif()
if(BUILD_STRESS_TESTS)
  add_subdirectory(stress ${CATCH_BUILD_DIR}/stress)
endif()
if(BUILD_CONTRACT_TESTS)
  add_subdirectory(contract ${CATCH_BUILD_DIR}/contract)
endif()

add_custom_target(gen_coverage
                  COMMAND ${CMAKE_COMMAND} -B build/
                  COMMAND ${CMAKE_COMMAND} --build build/
                  COMMAND ./build/generateHipAPICoverage ${HIP_PATH}/include
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../utils/coverage
                  COMMENT "Generating Test Coverage Report")

# packaging the tests
add_subdirectory(packaging)

# Doxygen documentation
# check if doxygen is installed
find_package(Doxygen)
if(DOXYGEN_FOUND)
  # Configure Doxyfile with proper paths
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/DoxyfileTests)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/DoxyfileTests.out)
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

  add_custom_target(build_docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        COMMENT "Generating test plan documentation with Doxygen")
else()
  message(STATUS "Doxygen was not found. Building test plan documentation will not be available")
endif()
