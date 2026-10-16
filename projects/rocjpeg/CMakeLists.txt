# ##############################################################################
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

cmake_minimum_required(VERSION 3.15)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# Check if lib/rocm_sysdeps/lib exists in the ROCm path which indicates ROCm installation via TheRock
set(USING_THE_ROCK OFF)
if(EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
  set(USING_THE_ROCK ON)
endif()

# RULE_LAUNCH_COMPILE is a Unix-shell 'VAR=value' prefix. On Windows the HIP
# kernels are compiled out of line and get --rocm-device-lib-path explicitly.
if(USING_THE_ROCK AND NOT WIN32)
  if(NOT DEFINED ENV{HIP_DEVICE_LIB_PATH})
    set(ENV{HIP_DEVICE_LIB_PATH} ${ROCM_PATH}/lib/llvm/amdgcn/bitcode)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "HIP_DEVICE_LIB_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode")
  endif()
endif()

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
# On Windows the host compiler is MSVC; only the HIP kernels are compiled with
# amdclang++ (see the rocjpeg_hip_kernels custom command below).
if(NOT WIN32)
  if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
  endif()
endif()

# rocjpeg Version
# NOTE: package version and rocjpeg_version.h is generated with this version
set(VERSION "1.13.0")

# Set Project Version and Language
project(rocjpeg VERSION ${VERSION} LANGUAGES CXX)

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
set(CMAKE_INSTALL_INCLUDEDIR "include" CACHE STRING "Include install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# avoid setting the default installation path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "rocjpeg default installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(DEFAULT_BUILD_TYPE "Release")
set(ENHANCED_MESSAGE ON)

if(ENHANCED_MESSAGE)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Red "${Esc}[31m")
  set(Green "${Esc}[32m")
  set(Yellow "${Esc}[33m")
  set(Blue "${Esc}[34m")
  set(BoldBlue "${Esc}[1;34m")
  set(Magenta "${Esc}[35m")
  set(Cyan "${Esc}[36m")
  set(White "${Esc}[37m")
endif()
message("-- ${BoldBlue}rocjpeg Version -- ${VERSION}${ColourReset}")
message("-- ${BoldBlue}rocjpeg Install Path -- ${CMAKE_INSTALL_PREFIX}${ColourReset}")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake ${ROCM_PATH}/share/rocmcmakebuildtools/cmake)

# rocjpeg Default Options
option(BUILD_WITH_AMD_ADVANCE "Build rocjpeg for advanced AMD GPU Architecture" OFF)

# rocjpeg Build Type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "rocjpeg Default Build Type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

# Set supported GPU Targets
if(NOT GPU_TARGETS AND NOT AMDGPU_TARGETS)
    find_package(ROCmCMakeBuildTools QUIET)
    if(NOT ROCmCMakeBuildTools_FOUND)
        find_package(ROCM QUIET)
    endif()
    include(ROCMCheckTargetIds OPTIONAL RESULT_VARIABLE HAS_ROCM_CHECK_TARGET_IDS)

    set(OPTIONAL_GPU_TARGETS "gfx950;gfx1150;gfx1151;gfx1152;gfx1153;gfx1170;gfx1171;gfx1200;gfx1201;gfx1250;gfx1250-strict")
    # rocm_check_target_ids probes with a HIP-language try_compile, which the
    # Visual Studio generator cannot run.
    if(HAS_ROCM_CHECK_TARGET_IDS AND NOT WIN32)
        rocm_check_target_ids(OPTIONAL_GPU_TARGETS_AVAILABLE TARGETS ${OPTIONAL_GPU_TARGETS})
    else() # if we don't have rocm_check_target_ids, just assume the targets are available
        set(OPTIONAL_GPU_TARGETS_AVAILABLE "${OPTIONAL_GPU_TARGETS}")
    endif()
    if(NOT WIN32)
        set(DEFAULT_GPU_TARGETS "gfx908;gfx90a;gfx942;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;${OPTIONAL_GPU_TARGETS_AVAILABLE}")
    else()
        # Windows also covers the integrated GPUs (gfx1035, gfx1103) that rocDecode's Windows samples target.
        set(DEFAULT_GPU_TARGETS "gfx908;gfx90a;gfx942;gfx1030;gfx1031;gfx1032;gfx1035;gfx1100;gfx1101;gfx1102;gfx1103;${OPTIONAL_GPU_TARGETS_AVAILABLE}")
    endif()
endif()

# Set AMD GPU_TARGETS
if((AMDGPU_TARGETS OR DEFINED ENV{AMDGPU_TARGETS}) AND (NOT GPU_TARGETS))
  message("-- ${Red}${PROJECT_NAME} DEPRECATION -- AMDGPU_TARGETS use is deprecated. Use GPU_TARGETS${ColourReset}")
  if(DEFINED ENV{AMDGPU_TARGETS})
    set(GPU_TARGETS $ENV{AMDGPU_TARGETS} CACHE STRING "List of specific machine types for library to target")
  else()
    set(GPU_TARGETS ${AMDGPU_TARGETS})
  endif()
endif()

if(DEFINED ENV{GPU_ARCHS})
  set(GPU_TARGETS $ENV{GPU_ARCHS} CACHE STRING "List of specific machine types for library to target")
elseif(GPU_TARGETS)
  message("-- ${White}${PROJECT_NAME} -- GPU_TARGETS set with -D option${ColourReset}")
else()
  set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
endif()
message("-- ${White}${PROJECT_NAME} -- AMD GPU_TARGETS: ${GPU_TARGETS}${ColourReset}")

# Add an option for enabling the rocprofiler-register
option(ROCJPEG_ENABLE_ROCPROFILER_REGISTER "Enable rocprofiler-register support" ON)

set (HIP_PLATFORM amd CACHE STRING "HIP platform")
find_package(HIP QUIET)
find_package(Libva QUIET)
if(NOT WIN32)
  find_package(Libdrm_amdgpu QUIET)
endif()
find_package(Threads QUIET)

if(NOT WIN32)
  if(ROCJPEG_ENABLE_ROCPROFILER_REGISTER)
    find_package(rocprofiler-register QUIET
      HINTS $ENV{rocprofiler_register_ROOT} $ENV{ROCPROFILER_REGISTER_ROOT} ${CMAKE_INSTALL_PREFIX}
      PATHS ${ROCM_PATH})
  endif()
else()
  # Not available on Windows. Clear any value inherited from an enclosing scope
  # -- a superbuild parent that found the package would otherwise leave
  # rocprofiler-register_FOUND true here.
  set(rocprofiler-register_FOUND FALSE)
  if(ROCJPEG_ENABLE_ROCPROFILER_REGISTER)
    message("-- ${Yellow}NOTE: rocprofiler-register is not available on Windows -- ROCJPEG_ENABLE_ROCPROFILER_REGISTER ignored${ColourReset}")
  endif()
endif()

# On Windows libdrm_amdgpu is not needed (vaon12 replaces DRM) and the CRT provides threads.
set(_deps_met FALSE)
if(NOT WIN32)
  if(HIP_FOUND AND Libva_FOUND AND Libdrm_amdgpu_FOUND AND Threads_FOUND)
    set(_deps_met TRUE)
  endif()
else()
  if(HIP_FOUND AND Libva_FOUND)
    set(_deps_met TRUE)
  endif()
endif()

if(_deps_met)
  # HIP
  if(NOT WIN32)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::device)
  else()
    # MSVC cannot take hip::device's clang-only flags; the kernels are compiled separately.
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)
  endif()
  # LibVA - headers always needed for VA-API type definitions.
  # On Linux TheRock builds, libva is loaded at runtime via dlopen with
  # RTLD_LOCAL | RTLD_DEEPBIND (see rocjpeg_vaapi_loader.cpp) to prevent symbol
  # collision with any system libva.so.2 in the same process; no direct
  # link-time dependency is needed in that case.
  # On Windows TheRock builds, rocm_sysdeps_va_win32.dll and rocm_sysdeps_va.dll
  # are likewise loaded at runtime via LoadLibraryExW (see rocjpeg_vaapi_loader.cpp).
  # On non-TheRock builds, link directly as before.
  include_directories(${LIBVA_INCLUDE_DIR})
  if(NOT WIN32)
    if(NOT (USING_THE_ROCK AND UNIX))
      set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LIBVA_LIBRARY})
      set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LIBVA_DRM_LIBRARY})
    endif()
    # DRM_AMDGPU
    include_directories(${LIBDRM_AMDGPU_INCLUDE_DIR})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LIBDRM_AMDGPU_LIBRARY})
  else()
    # vaon12: va_win32 instead of va-drm. FindLibva only looks for va-drm on
    # Linux, so LIBVA_DRM_LIBRARY is never referenced here (a stale cache entry
    # would otherwise reach the linker).
    include_directories(${LIBVA_WIN32_INCLUDE_DIR})
    if(NOT USING_THE_ROCK)
      set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LIBVA_LIBRARY} ${LIBVA_WIN32_LIBRARY})
    endif()
  endif()

  # rocprofiler
  if(rocprofiler-register_FOUND)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocprofiler-register::rocprofiler-register)
  endif()

  if(NOT WIN32)
    # Threads library (required for std::thread support)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} Threads::Threads)
  endif()

  # local include files
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api ${CMAKE_CURRENT_SOURCE_DIR}/src)
  # source files
  file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
  if(WIN32)
    # The Windows host compiler is MSVC, so the HIP device code cannot ride along on
    # hip::device the way it does on Linux. CMake's HIP language is not an option
    # either: it requires Ninja and forbids mixing MSVC-style and clang-style command
    # lines in one project. Compile the kernels out of line with amdclang++ -- the
    # same compiler the Linux path uses for this file -- and link the object into
    # the DLL. The per-config output path needs a generator expression in
    # add_custom_command(OUTPUT), which CMake only evaluates from 3.20 on.
    if(CMAKE_VERSION VERSION_LESS 3.20)
      message(FATAL_ERROR "Building rocjpeg on Windows requires CMake 3.20 or newer (found ${CMAKE_VERSION}).")
    endif()
    set(HIP_KERNEL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/rocjpeg_hip_kernels.cpp)
    list(REMOVE_ITEM SOURCES ${HIP_KERNEL_SOURCE})
    set(HIP_KERNEL_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++.exe)
    # amdclang++ needs one --offload-arch= per target; GPU_TARGETS is a list and
    # would otherwise expand into bare arguments that clang reads as inputs
    set(OFFLOAD_ARCH_FLAGS)
    foreach(GPU_TARGET ${GPU_TARGETS})
      list(APPEND OFFLOAD_ARCH_FLAGS --offload-arch=${GPU_TARGET})
    endforeach()
    # Visual Studio is a multi-config generator, and the CRT and optimization flags
    # below vary by configuration, so the object lives in a per-config directory:
    # with a shared path, an object left by a Debug build looks up to date to a later
    # Release build, which would then link a kernel built against the debug CRT.
    set(HIP_KERNEL_OBJ_DIR ${CMAKE_CURRENT_BINARY_DIR}/hip_kernels/$<CONFIG>)
    set(HIP_KERNEL_OBJ ${HIP_KERNEL_OBJ_DIR}/rocjpeg_hip_kernels.obj)
    add_custom_command(
      OUTPUT ${HIP_KERNEL_OBJ}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${HIP_KERNEL_OBJ_DIR}
      COMMAND ${HIP_KERNEL_COMPILER} -x hip -c ${HIP_KERNEL_SOURCE} -o ${HIP_KERNEL_OBJ} -D__HIP_PLATFORM_AMD__=1 -std=c++17
        ${OFFLOAD_ARCH_FLAGS}
        --rocm-path=${ROCM_PATH}
        --rocm-device-lib-path=${ROCM_PATH}/lib/llvm/amdgcn/bitcode
        -I${CMAKE_CURRENT_SOURCE_DIR}/src
        -I${ROCM_PATH}/include
        -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS
        # must track the MSVC config, or the kernel object links
        # against a different CRT than the rest of the library
        -fms-runtime-lib=$<IF:$<CONFIG:Debug>,dll_dbg,dll>
        $<IF:$<CONFIG:Debug>,-O0,-O3>
      # depend on this file too, so changing the flags above rebuilds
      DEPENDS ${HIP_KERNEL_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/src/rocjpeg_hip_kernels.h ${CMAKE_CURRENT_LIST_FILE}
      COMMENT "Compiling rocjpeg HIP kernels"
    )
    add_custom_target(rocjpeg_hip_kernels DEPENDS ${HIP_KERNEL_OBJ})
  endif()
  # rocjpeg.so / rocjpeg.dll
  add_library(${PROJECT_NAME} SHARED ${SOURCES})

  if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    # /-prefixed flags are MSVC syntax: a GNU-driver compiler on Windows reads
    # them as input filenames, so gate on the compiler rather than on WIN32.
    if(MSVC)
      # Suppress warnings from pre-existing code and external headers:
      # C4200: zero-sized array in va.h
      # C4018/C4267/C4244/C4146: signed/unsigned and narrowing conversions in
      #   pre-existing code (also present, unreported, on Linux)
      target_compile_options(${PROJECT_NAME} PRIVATE /W3 /wd4200 /wd4018 /wd4267 /wd4244 /wd4146)
    endif()
  elseif(CMAKE_BUILD_TYPE MATCHES Debug)
    # -O0 -- Don't Optimize output file
    # -g  -- generate debugging information, generate debugging information, dwarf-4 for making valgrind work
    # -Og -- Optimize for debugging experience rather than speed or size
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -gdwarf-4")
    target_compile_options(${PROJECT_NAME} PRIVATE -O0 -gdwarf-4 -Wall)
  else()
    # -O3       -- Optimize output file
    # -DNDEBUG  -- turn off asserts
    # -fPIC     -- Generate position-independent code if possible
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC")
    target_compile_options(${PROJECT_NAME} PRIVATE -O3 -DNDEBUG -fPIC -Wall)
  endif()
  message("-- ${BoldBlue}rocjpeg Build Type -- ${CMAKE_BUILD_TYPE}${ColourReset}")

  if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARY_LIST})
  else()
    # On Windows the implementation libraries are linked PRIVATE so they stay out
    # of the exported INTERFACE_LINK_LIBRARIES: raw .lib/.obj paths resolve to
    # absolute build-tree locations, and a consumer relinking against the installed
    # target (the ctest build-and-test samples) would fail on them.
    #
    # hip::host stays PUBLIC: the installed public header rocjpeg.h includes
    # hip/hip_runtime.h, so it is a genuine usage requirement for consumers.
    #
    # The kernel object is a link input rather than a target source, so
    # WINDOWS_EXPORT_ALL_SYMBOLS does not export the kernel launchers.
    set(_PRIVATE_LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST})
    list(REMOVE_ITEM _PRIVATE_LINK_LIBRARY_LIST hip::host)
    add_dependencies(${PROJECT_NAME} rocjpeg_hip_kernels)
    target_link_libraries(${PROJECT_NAME}
      PUBLIC hip::host
      PRIVATE ${_PRIVATE_LINK_LIBRARY_LIST} ${HIP_KERNEL_OBJ})
  endif()

  set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  if(NOT WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
  else()
    set_target_properties(${PROJECT_NAME} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()

  if(USING_THE_ROCK AND NOT WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN;$ORIGIN/rocm_sysdeps/lib" BUILD_WITH_INSTALL_RPATH TRUE)
    if(UNIX)
      # Enable dlopen-based VA isolation and link against libdl for dlopen/dlsym/dlclose.
      # dlopen is Linux-only.
      target_compile_definitions(${PROJECT_NAME} PRIVATE ROCJPEG_USE_DLOPEN_VA)
      target_link_libraries(${PROJECT_NAME} dl)
    endif()
  endif()
  if(USING_THE_ROCK AND WIN32)
    # Load rocm_sysdeps_va_win32.dll / rocm_sysdeps_va.dll at runtime via
    # LoadLibraryExW/GetProcAddress (kernel32, linked by default).
    target_compile_definitions(${PROJECT_NAME} PRIVATE ROCJPEG_USE_DLOPEN_VA)
  endif()

  # rocprofiler
  if(rocprofiler-register_FOUND)
    string(REPLACE "." ";" VERSION_LIST ${VERSION})
    list(GET VERSION_LIST 0 VERSION_MAJOR)
    list(GET VERSION_LIST 1 VERSION_MINOR)
    list(GET VERSION_LIST 2 VERSION_PATCH)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ROCJPEG_ROCPROFILER_REGISTER=1
      ROCJPEG_ROCP_REG_VERSION_MAJOR=${VERSION_MAJOR}
      ROCJPEG_ROCP_REG_VERSION_MINOR=${VERSION_MINOR}
      ROCJPEG_ROCP_REG_VERSION_PATCH=${VERSION_PATCH})
  endif()

  # Set a default compile option to reuse the interop buffer. Disable this option if the ROCm version is below 7.0.0.
  target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_INTEROP_BUFFER_REUSE=1)
  # rocm version
  set(ROCM_VERSION_FILE  ${ROCM_PATH}/.info/version-rocm)
  if(EXISTS "${ROCM_VERSION_FILE}")
      file(READ "${ROCM_VERSION_FILE}" ROCM_VERSION_STRING)
      string(STRIP "${ROCM_VERSION_STRING}" ROCM_VERSION_STRING)
      message(STATUS "Full ROCm version string: ${ROCM_VERSION_STRING}")
      # Match version string with regex
      string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)-([0-9]+)$" _ "${ROCM_VERSION_STRING}")
      if(CMAKE_MATCH_COUNT GREATER 0)
          set(rocm_major_version "${CMAKE_MATCH_1}")
          set(rocm_minor_version "${CMAKE_MATCH_2}")
          set(rocm_patch_version "${CMAKE_MATCH_3}")
          set(rocm_build_number "${CMAKE_MATCH_4}")
          message(STATUS "ROCm Major Version: ${rocm_major_version}")
          message(STATUS "ROCm Minor Version: ${rocm_minor_version}")
          message(STATUS "ROCm Patch Version: ${rocm_patch_version}")
          message(STATUS "ROCm Build Number: ${rocm_build_number}")
          # Convert strings to integers
          math(EXPR ROCM_VERSION_INT "${rocm_major_version} * 10000 + ${rocm_minor_version} * 100 + ${rocm_patch_version}")
          set(TARGET_VERSION_INT 70000)
          if(ROCM_VERSION_INT LESS TARGET_VERSION_INT)
              message(STATUS "ROCm version is below the required threshold (>= 7.0.0) for reusing the interop buffer, so the compile option: ENABLE_INTEROP_BUFFER_REUSE is disabled.")
              target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_INTEROP_BUFFER_REUSE=0)
          endif()
      endif()
  endif()

  #Generate BUILD_INFO
  configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/api/rocjpeg_version.h.in ${CMAKE_CURRENT_BINARY_DIR}/rocjpeg_version.h @ONLY )

  # install rocjpeg libs -- {ROCM_PATH}/lib
  if(NOT WIN32)
    install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} NAMELINK_SKIP)
    install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} NAMELINK_ONLY)
  else()
    install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
  # install rocjpeg include files -- {ROCM_PATH}/include/rocjpeg
  install(FILES api/rocjpeg/rocjpeg.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rocjpeg_version.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
  # install rocjpeg api trace include file -- {ROCM_PATH}/include/rocjpeg/amd_detail
  install(FILES api/amd_detail/rocjpeg_api_trace.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/amd_detail)

  # install rocjpeg samples -- {ROCM_PATH}/share/rocjpeg
  install(FILES samples/jpegDecode/CMakeLists.txt samples/jpegDecode/jpegdecode.cpp samples/jpegDecode/README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/jpegDecode)
  install(FILES samples/jpegDecodePerf/CMakeLists.txt samples/jpegDecodePerf/jpegdecodeperf.cpp samples/jpegDecodePerf/README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/jpegDecodePerf)
  install(FILES samples/jpegDecodeBatched/CMakeLists.txt samples/jpegDecodeBatched/jpegdecodebatched.cpp samples/jpegDecodeBatched/README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/jpegDecodeBatched)
  install(FILES samples/jpegDecodeAsync/CMakeLists.txt samples/jpegDecodeAsync/jpegdecodeasync.cpp samples/jpegDecodeAsync/README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/jpegDecodeAsync)
  install(FILES samples/jpegDecodeBatchedAsync/CMakeLists.txt samples/jpegDecodeBatchedAsync/jpegdecodebatchedasync.cpp samples/jpegDecodeBatchedAsync/README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/jpegDecodeBatchedAsync)
  install(FILES samples/jpegDecodeCaps/CMakeLists.txt samples/jpegDecodeCaps/jpegdecodecaps.cpp samples/jpegDecodeCaps/README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/jpegDecodeCaps)
  install(FILES samples/rocjpeg_samples_utils.h DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples)
  install(DIRECTORY data/images DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME})
  # install license information - {ROCM_PATH}/share/doc/rocjpeg
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR})
  # install test cmake
  install(FILES test/CMakeLists.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test)
  install(PROGRAMS test/run_rocjpeg.py DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test)
  install(FILES test/rocjpeg_negative_api_tests/CMakeLists.txt test/rocjpeg_negative_api_tests/README.md test/rocjpeg_negative_api_tests/rocjpeg_api_negative_tests.cpp test/rocjpeg_negative_api_tests/rocjpeg_api_negative_tests.h test/rocjpeg_negative_api_tests/rocjpegnegativetest.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test/rocjpeg_negative_api_tests)

  message("-- ${White}AMD ROCm rocjpeg -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
  message("-- ${White}AMD ROCm rocjpeg -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")

  # Cmake module config file configurations
  set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" CACHE INTERNAL "Default module path.")

  # Export the package for use from the build-tree
  # (this registers the build-tree with a global CMake-registry)
  export(PACKAGE ${PROJECT_NAME})
  set(CONF_LIB_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

  # Create the rocjpeg-config.cmake and rocjpeg-config-version files
  include(CMakePackageConfigHelpers)
  set(CONFIG_PACKAGE_INSTALL_DIR ${CONF_LIB_DIR}/cmake/${PROJECT_NAME})
  set(LIB_INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
  set(INCLUDE_INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
  set(BIN_INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")

  # Generate Config File for rocjpeg
  configure_package_config_file(
    ${CMAKE_MODULE_PATH}/${PROJECT_NAME}-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR}
    PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR BIN_INSTALL_DIR
  )

  # Generate config Version File for rocjpeg
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    COMPATIBILITY SameMajorVersion
  )

  # Install the rocjpeg-config.cmake and rocjpeg-config-version.cmake
  install(FILES
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
  )

  # Install the export set for use with the install-tree
  install(EXPORT ${PROJECT_NAME}-targets 
    FILE ${PROJECT_NAME}-targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
  )

  # Option to build test binaries during the main build (without running them)
  option(BUILD_TESTS "Build test binaries and install to share/${PROJECT_NAME}/bin" OFF)

  # Not supported on Windows: the nested ExternalProject builds assume a
  # single-config generator (CMAKE_BUILD_TYPE, an unsuffixed executable directly
  # in BINARY_DIR) and $ORIGIN install RPATHs, none of which hold for the
  # Visual Studio generator. Build the samples from share/rocjpeg/samples instead.
  if(BUILD_TESTS AND NOT WIN32)
    include(ExternalProject)
    set(ROCJPEG_TEST_BIN_DIR ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/bin)
    set(ROCJPEG_STAGE_DIR ${CMAKE_BINARY_DIR}/stage)

    # Stage-install the library and its cmake/include artifacts into the build
    # tree so ExternalProject find_package() works without requiring a prior
    # sudo make install. cmake --install --prefix rewrites all install paths,
    # and configure_package_config_file generates relocatable relative paths,
    # so the staged config correctly resolves headers and libs.
    add_custom_target(stage_${PROJECT_NAME}
      COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --prefix ${ROCJPEG_STAGE_DIR}
      DEPENDS ${PROJECT_NAME}
      COMMENT "Staging ${PROJECT_NAME} into build tree for test builds"
    )

    # Assemble the search prefix for the nested test configures.
    #
    # In a standalone build this is just the stage dir (so find_package(rocjpeg)
    # resolves the freshly staged config), and ROCM_PATH covers HIP, libva, etc.
    #
    # Under TheRock, ROCM_PATH is empty and every dependency lives in the
    # super-project build tree. TheRock injects a dependency provider into the
    # parent configure (via CMAKE_PROJECT_TOP_LEVEL_INCLUDES), but that provider
    # errors on any find_package() for a package it knows about yet that was not
    # declared as a dependency -- which includes rocjpeg itself -- so it cannot
    # be forwarded to the test configures. Instead forward the toolchain (for the
    # HIP-aware compiler and flags) and turn the provider's package map
    # (THEROCK_PACKAGE_DIR_<pkg>) into an ordinary CMAKE_PREFIX_PATH so the nested
    # configures find HIP and friends by normal config search. These variables are
    # undefined in a standalone build, leaving the prefix at just the stage dir.
    set(ROCJPEG_TEST_PREFIX_PATH "${ROCJPEG_STAGE_DIR}")
    foreach(_therock_pkg IN LISTS THEROCK_PROVIDED_PACKAGES)
      if(DEFINED THEROCK_PACKAGE_DIR_${_therock_pkg})
        list(APPEND ROCJPEG_TEST_PREFIX_PATH "${THEROCK_PACKAGE_DIR_${_therock_pkg}}")
      endif()
    endforeach()
    # ExternalProject_Add splits CMAKE_ARGS on ';', so pass the multi-entry prefix
    # through a custom LIST_SEPARATOR that the sub-build restores to ';'.
    string(REPLACE ";" "|" ROCJPEG_TEST_PREFIX_PATH_ALT "${ROCJPEG_TEST_PREFIX_PATH}")

    # Forward the toolchain only when the top-level build set one, and as an
    # absolute path: a relative CMAKE_TOOLCHAIN_FILE is resolved against the
    # top-level build dir, but each ExternalProject configures in its own build
    # dir where that relative path would not resolve. Empty expands to no argument.
    set(ROCJPEG_TEST_TOOLCHAIN_ARG "")
    if(CMAKE_TOOLCHAIN_FILE)
      get_filename_component(ROCJPEG_TEST_TOOLCHAIN_FILE
        "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE BASE_DIR "${CMAKE_BINARY_DIR}")
      set(ROCJPEG_TEST_TOOLCHAIN_ARG
        "-DCMAKE_TOOLCHAIN_FILE=${ROCJPEG_TEST_TOOLCHAIN_FILE}")
    endif()

    # RPATH for the installed test binaries. They install under
    # share/<project>/bin while librocjpeg, HIP and the bundled sysdeps install
    # to the lib dir -- so at runtime they must load the installed librocjpeg
    # (whose own RPATH resolves its sysdeps), not the build-tree stage copy the
    # nested configure links against. Derive the $ORIGIN-relative hop from the bin
    # dir to the lib dir so the RPATH tracks either path if it changes instead of
    # hard-coding '../../..'. Use GNUInstallDirs' absolute CMAKE_INSTALL_FULL_*
    # forms (not prefix + CMAKE_INSTALL_*, which is wrong if a *DIR is overridden
    # with an absolute path) so file(RELATIVE_PATH) gets two real absolute paths.
    # This keeps the tree relocatable as a whole -- the same $ORIGIN scheme
    # librocjpeg itself uses; relocating an individual binary out of the tree is
    # unsupported, as for any $ORIGIN-relative ROCm binary. BUILD_WITH_INSTALL_RPATH
    # bakes it into the built binary, which the macro copies verbatim
    # (install(PROGRAMS) does no RPATH rewriting). '|' is the LIST_SEPARATOR
    # restored to ';' in the sub-build.
    file(RELATIVE_PATH ROCJPEG_TEST_BIN_TO_LIB
      "${CMAKE_INSTALL_FULL_DATADIR}/${PROJECT_NAME}/bin"
      "${CMAKE_INSTALL_FULL_LIBDIR}")
    set(ROCJPEG_TEST_INSTALL_RPATH
      "$ORIGIN/${ROCJPEG_TEST_BIN_TO_LIB}|$ORIGIN/${ROCJPEG_TEST_BIN_TO_LIB}/rocm_sysdeps/lib")

    # Helper macro to add one ExternalProject test and register its install.
    macro(rocjpeg_add_test_ext TEST_EXE TEST_SOURCE_DIR)
      ExternalProject_Add(test_${TEST_EXE}
        SOURCE_DIR ${TEST_SOURCE_DIR}
        LIST_SEPARATOR "|"
        CMAKE_ARGS -DROCM_PATH=${ROCM_PATH}
                   ${ROCJPEG_TEST_TOOLCHAIN_ARG}
                   "-DCMAKE_PREFIX_PATH=${ROCJPEG_TEST_PREFIX_PATH_ALT}"
                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                   -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
                   "-DCMAKE_INSTALL_RPATH=${ROCJPEG_TEST_INSTALL_RPATH}"
        BUILD_ALWAYS ON
        INSTALL_COMMAND ""
        DEPENDS stage_${PROJECT_NAME}
      )
      ExternalProject_Get_Property(test_${TEST_EXE} BINARY_DIR)
      install(PROGRAMS ${BINARY_DIR}/${TEST_EXE} DESTINATION ${ROCJPEG_TEST_BIN_DIR} OPTIONAL)
    endmacro()

    # Sample tests
    rocjpeg_add_test_ext(jpegdecode             ${CMAKE_CURRENT_SOURCE_DIR}/samples/jpegDecode)
    rocjpeg_add_test_ext(jpegdecodebatched       ${CMAKE_CURRENT_SOURCE_DIR}/samples/jpegDecodeBatched)
    rocjpeg_add_test_ext(jpegdecodeperf          ${CMAKE_CURRENT_SOURCE_DIR}/samples/jpegDecodePerf)
    rocjpeg_add_test_ext(jpegdecodeasync         ${CMAKE_CURRENT_SOURCE_DIR}/samples/jpegDecodeAsync)
    rocjpeg_add_test_ext(jpegdecodebatchedasync  ${CMAKE_CURRENT_SOURCE_DIR}/samples/jpegDecodeBatchedAsync)
    rocjpeg_add_test_ext(jpegdecodecaps          ${CMAKE_CURRENT_SOURCE_DIR}/samples/jpegDecodeCaps)
  elseif(BUILD_TESTS)
    # Reached only when BUILD_TESTS=ON and WIN32, which the branch above excludes.
    message(WARNING "BUILD_TESTS=ON is ignored on Windows: no test binaries will be built "
                    "or installed to ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/bin. "
                    "Build the samples from ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples "
                    "after installing instead.")
  endif()

  # make test with CTest
  set(BUILD_FROM_SOURCE ON)
  enable_testing()
  include(CTest)
  add_subdirectory(test)

else()
  message("-- ${Red}AMD ROCm rocjpeg -- unmet dependencies${ColourReset}")
  if(NOT HIP_FOUND)
    message(FATAL_ERROR "-- ERROR!: HIP Not Found!")
  endif()
  if(NOT Libva_FOUND)
    message(FATAL_ERROR "-- ERROR!: libva Not Found!")
  endif()
  if(NOT WIN32 AND NOT Libdrm_amdgpu_FOUND)
    message(FATAL_ERROR "-- ERROR!: libdrm_amdgpu Not Found!")
  endif()
  if(NOT WIN32 AND NOT Threads_FOUND)
    message(FATAL_ERROR "-- ERROR!: Threads Not Found!")
  endif()
endif()
