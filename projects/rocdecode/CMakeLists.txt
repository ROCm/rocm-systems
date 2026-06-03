# ##############################################################################
# Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.10)

# CMP0091: MSVC_RUNTIME_LIBRARY target property is honoured only when NEW.
# Must be set before the first project() call.
cmake_policy(SET CMP0091 NEW)
# Force all targets to use the static Release CRT so they match the PAL
# pre-built libs (compiled with /MT). This must be set before any add_library
# or add_executable, which is why it is placed at the top of the file.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Platform detection
if(WIN32)
  set(IS_WINDOWS TRUE)
  set(IS_LINUX FALSE)
else()
  set(IS_WINDOWS FALSE)
  set(IS_LINUX TRUE)
endif()

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  # Default ROCM_PATH if not set
  if(IS_WINDOWS)
    set(ROCM_PATH c:/opt/rocm CACHE PATH "Default ROCm installation path")
  else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
  endif()
  message("-- INFO: Using default ROCM_PATH -- ${ROCM_PATH}")
endif()

# Check if lib/rocm_sysdeps/lib exists in the ROCm path which indicates ROCm installation via TheRock
set(USING_THE_ROCK OFF)
if(EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
  set(USING_THE_ROCK ON)
endif()

# Set AMD Clang as default compiler (MUST be before project() command)
# IMPORTANT: Compiler must be set before project() call for CMake to use it
if (NOT DEFINED CMAKE_CXX_COMPILER)
  if(IS_WINDOWS)
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang.exe)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++.exe)
  else()
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
  endif()
endif()

# rocdecode Version
set(VERSION "1.8.0")

# Set Project Version and Language
project(rocdecode VERSION ${VERSION} LANGUAGES CXX)

# C++ Standard (must be set after project())
# C++ Standard - C++20 required for PAL on Windows, C++17 for Linux
if(IS_WINDOWS)
  set(CMAKE_CXX_STANDARD 20)
else()
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)

# Set MSVC runtime library to static to match PAL (Windows only)
# amdclang++ has GNU frontend, so CMake adds runtime flags automatically via Platform files
# We need to replace them AFTER CMake adds them, which happens during target creation
if(IS_WINDOWS)
  message("-- ${Green}Setting static runtime library to match PAL${ColourReset}")
  # These string replacements won't work because /MD isn't in the flags yet
  # Instead, we need to set CMAKE_USER_MAKE_RULES_OVERRIDE or use target properties
  set(CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELWITHDEBINFO)
  foreach(CompilerFlag ${CompilerFlags})
    if(${CompilerFlag} MATCHES "/MD")
      string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
    endif()
  endforeach()
endif()

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
set(CMAKE_INSTALL_INCLUDEDIR "include" CACHE STRING "Include install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# avoid setting the default installation path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "${PROJECT_NAME} default installation path" FORCE)
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
message("-- ${BoldBlue}${PROJECT_NAME} Version -- ${VERSION}${ColourReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Install Path -- ${CMAKE_INSTALL_PREFIX}${ColourReset}")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)

# rocdecode Build Type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "rocdecode Default Build Type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
if(CMAKE_BUILD_TYPE MATCHES Debug)
  # -O0 -- Don't Optimize output file
  # -g  -- generate debugging information
  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od /Zi")
  elseif(IS_LINUX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -gdwarf-4")
  else()
    # Windows with AMD Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
  endif()
else()
  # Release build optimizations
  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /DNDEBUG")
  elseif(IS_LINUX)
    # -fPIC   -- Generate position-independent code (Linux only, not supported on Windows)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC")
  else()
    # Windows with AMD Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
  endif()
endif()
message("-- ${BoldBlue}${PROJECT_NAME} Build Type -- ${CMAKE_BUILD_TYPE}${ColourReset}")

# Developer options
# Add an option for enabling the rocprofiler-register
option(ROCDECODE_ENABLE_ROCPROFILER_REGISTER "Enable rocprofiler-register support" ON)
# Add an option for enabling FFMPEG avcodec host-based decoder (disabled on Windows)
if(IS_WINDOWS)
  option(ROCDECODE_ENABLE_HOST_DECODER "Enable rocdecode-host-based decoder support" OFF)
else()
  option(ROCDECODE_ENABLE_HOST_DECODER "Enable rocdecode-host-based decoder support" ON)
endif()

set (HIP_PLATFORM amd CACHE STRING "HIP platform")
find_package(HIP QUIET)

# VA-API and libdrm are Linux-only dependencies
if(IS_LINUX)
  find_package(Libva QUIET)
  find_package(Libdrm_amdgpu QUIET)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads QUIET)

# Check Libva version compatibility: requires libva >= 1.22 (Linux only)
if(IS_LINUX)
  if(Libva_FOUND)
    if("${Libva_VERSION}" VERSION_GREATER_EQUAL "1.22")
      message("-- ${White}\tLibva Version Supported${ColourReset}")
    else()
      set(Libva_FOUND FALSE)
      message("-- ${Yellow}\tLibva Version Not Supported (found ${Libva_VERSION}, need >= 1.22)${ColourReset}")
    endif()
  endif()
else()
  # On Windows, VA-API is not supported, set to not found
  set(Libva_FOUND FALSE)
  set(Libdrm_amdgpu_FOUND FALSE)
endif()

if(ROCDECODE_ENABLE_ROCPROFILER_REGISTER)
  find_package(rocprofiler-register QUIET
    HINTS $ENV{rocprofiler_register_ROOT} $ENV{ROCPROFILER_REGISTER_ROOT} ${CMAKE_INSTALL_PREFIX}
    PATHS ${ROCM_PATH})
endif()
if(ROCDECODE_ENABLE_HOST_DECODER)
  find_package(FFmpeg QUIET)
endif()

# Check required dependencies based on platform
if(IS_WINDOWS)
  # Windows only requires HIP and Threads
  set(DEPENDENCIES_MET FALSE)
  if(HIP_FOUND AND Threads_FOUND)
    set(DEPENDENCIES_MET TRUE)
  endif()
else()
  # Linux requires HIP, VA-API, libdrm, and Threads
  set(DEPENDENCIES_MET FALSE)
  if(HIP_FOUND AND Libva_FOUND AND Libdrm_amdgpu_FOUND AND Threads_FOUND)
    set(DEPENDENCIES_MET TRUE)
  endif()
endif()

if(DEPENDENCIES_MET)

  # HIP
  set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)

  # LibVA and DRM_AMDGPU (Linux only)
  if(IS_LINUX)
    # LibVA
    include_directories(${LIBVA_INCLUDE_DIR})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LIBVA_LIBRARY})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LIBVA_DRM_LIBRARY})

    # DRM_AMDGPU
    include_directories(${LIBDRM_AMDGPU_INCLUDE_DIR})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LIBDRM_AMDGPU_LIBRARY})
  endif()

  # PAL libraries (Windows only)
  if(IS_WINDOWS)
    if(NOT DEFINED PAL_ROOT)
      set(PAL_ROOT "" CACHE PATH "PAL root directory (required for Windows build)")
      message(FATAL_ERROR "PAL_ROOT must be set for Windows builds. Use: cmake -DPAL_ROOT=<path_to_pal> ..")
    endif()

    if(NOT EXISTS "${PAL_ROOT}/inc/core")
      message(FATAL_ERROR "PAL_ROOT (${PAL_ROOT}) does not contain inc/core directory")
    endif()

    if(NOT DEFINED UVD_INCLUDE_DIR)
      set(UVD_INCLUDE_DIR "" CACHE PATH "UVD firmware include directory containing drv_uvd_if.h (required for Windows build)")
      message(FATAL_ERROR "UVD_INCLUDE_DIR must be set for Windows builds. Use: cmake -DUVD_INCLUDE_DIR=<path> ..")
    endif()

    if(NOT EXISTS "${UVD_INCLUDE_DIR}/drv_uvd_if.h")
      message(FATAL_ERROR "UVD_INCLUDE_DIR (${UVD_INCLUDE_DIR}) does not contain drv_uvd_if.h")
    endif()

    include_directories(${UVD_INCLUDE_DIR})

    # Use the PAL package CMake target to extract its INTERFACE compile definitions,
    # include directories, and link libraries.  We store them in variables and apply
    # them to rocdecode after add_library, rather than linking the 'pal' target itself.
    # This avoids install(EXPORT) errors from 'pal' not being in the rocdecode export set.
    if(NOT TARGET pal)
      # Set the version before add_subdirectory so the PAL CMakeLists version check passes.
      set(PAL_CLIENT_INTERFACE_MAJOR_VERSION 954)
      add_subdirectory(${PAL_ROOT} ${CMAKE_BINARY_DIR}/pal_package EXCLUDE_FROM_ALL)
    endif()
    get_target_property(_pal_defs pal INTERFACE_COMPILE_DEFINITIONS)
    get_target_property(_pal_incs pal INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(_pal_libs pal INTERFACE_LINK_LIBRARIES)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST}
        ${_pal_libs}
        ws2_32.lib
    )
    message("-- ${Green}${PROJECT_NAME} -- PAL linked via CMake target from ${PAL_ROOT}${ColourReset}")
  endif()

  # rocprofiler
  if (rocprofiler-register_FOUND)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocprofiler-register::rocprofiler-register)
  endif()

  # Threads library (required for std::thread support)
  set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} Threads::Threads)

  # local include files
  if(IS_LINUX)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api ${CMAKE_CURRENT_SOURCE_DIR}/src/rocdecode ${CMAKE_CURRENT_SOURCE_DIR}/src/parser ${CMAKE_CURRENT_SOURCE_DIR}/src/rocdecode/vaapi)
  else()
    # Windows: include pal directory instead of vaapi
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api ${CMAKE_CURRENT_SOURCE_DIR}/src/rocdecode ${CMAKE_CURRENT_SOURCE_DIR}/src/parser ${CMAKE_CURRENT_SOURCE_DIR}/src/rocdecode/pal)
  endif()

  # source files
  file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
  # Exclude rocdecode-host
  list(FILTER SOURCES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/src/rocdecode-host/*.*")
  # Exclude VA-API files on Windows
  if(IS_WINDOWS)
    list(FILTER SOURCES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/src/rocdecode/vaapi/.*")
    message("-- ${Yellow}${PROJECT_NAME} -- VA-API files excluded for Windows build${ColourReset}")
  endif()

  # rocdecode.so (Linux) or rocdecode.dll (Windows)
  add_library(${PROJECT_NAME} SHARED ${SOURCES})

  # Windows-specific: export all symbols for DLL and define platform macro
  if(IS_WINDOWS)
    set_target_properties(${PROJECT_NAME} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        ROCDECODE_BUILD_WINDOWS=1
        WINVER=0x0A00
        _WIN32_WINNT=0x0A00
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
    # Apply PAL INTERFACE compile definitions and include directories extracted above.
    # These must match what the PAL library was compiled with to avoid ABI mismatches.
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${_pal_defs})
    target_include_directories(${PROJECT_NAME} PRIVATE ${_pal_incs})

    # Set static runtime library to match PAL (works for both MSVC and Clang)
    # PAL static libs are built with /MT (Release CRT only); always match that regardless of build config.
    set_property(TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE ROCDECODE_BUILD_LINUX=1)
  endif()

  # Warning flags
  if(MSVC)
    # MSVC: Use /W3 and disable specific noisy warnings
    target_compile_options(${PROJECT_NAME} PRIVATE /W3)
    target_compile_options(${PROJECT_NAME} PRIVATE
      /wd4018  # signed/unsigned mismatch
      /wd4146  # unary minus operator applied to unsigned type
      /wd4244  # conversion from type1 to type2, possible loss of data
      /wd4267  # conversion from size_t to smaller type
      /wd4365  # signed/unsigned mismatch
      /wd4996  # deprecated function (getenv)
    )
    # Suppress CRT security warnings
    target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
  else()
    # GCC/Clang: Use -Wall
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  endif()

  target_link_libraries(${PROJECT_NAME} PRIVATE ${LINK_LIBRARY_LIST})

  set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

  # RPATH settings (Linux only)
  if(IS_LINUX)
    if(USING_THE_ROCK)
      set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN;$ORIGIN/rocm_sysdeps/lib" BUILD_WITH_INSTALL_RPATH TRUE)
    endif()
  endif()

  # rocprofiler
  if (rocprofiler-register_FOUND)
    string(REPLACE "." ";" VERSION_LIST ${VERSION})
    list(GET VERSION_LIST 0 VERSION_MAJOR)
    list(GET VERSION_LIST 1 VERSION_MINOR)
    list(GET VERSION_LIST 2 VERSION_PATCH)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ROCDECODE_ROCPROFILER_REGISTER=1
      ROCDECODE_ROCP_REG_VERSION_MAJOR=${VERSION_MAJOR}
      ROCDECODE_ROCP_REG_VERSION_MINOR=${VERSION_MINOR}
      ROCDECODE_ROCP_REG_VERSION_PATCH=${VERSION_PATCH})
  endif()

  # make rocdecode-host for decoding on HOST only if FFMPEG is available (Linux only)
  if(IS_LINUX AND ROCDECODE_ENABLE_HOST_DECODER AND FFMPEG_FOUND)
    add_subdirectory(src/rocdecode-host)
  else()
    if(IS_WINDOWS)
      message("-- ${Yellow}${PROJECT_NAME} -- rocdecode-host not supported on Windows${ColourReset}")
    elseif(NOT FFMPEG_FOUND)
      message("-- ${Yellow}${PROJECT_NAME} -- FFMPEG NOT Found; rocdecode-host exculded${ColourReset}")
    endif()
  endif()

  # Generate BUILD_INFO
  configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/api/rocdecode_version.h.in ${CMAKE_CURRENT_BINARY_DIR}/rocdecode_version.h @ONLY )

  # install rocdecode libs -- {ROCM_PATH}/lib or {ROCM_PATH}/bin (Windows DLLs)
  if(IS_WINDOWS)
    # On Windows, DLLs go to bin, import libraries go to lib
    install(TARGETS ${PROJECT_NAME}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  else()
    # On Linux, shared libraries go to lib
    install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} NAMELINK_SKIP)
    install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} NAMELINK_ONLY)
  endif()
  # install rocdecode include files -- {ROCM_PATH}/include/rocdecode
  install(FILES api/rocdecode/rocdecode.h api/rocdecode/rocparser.h api/rocdecode/roc_bitstream_reader.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rocdecode_version.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
  # install rocdecode api trace include file -- {ROCM_PATH}/include/rocdecode/amd_detail
  install(FILES api/amd_detail/rocdecode_api_trace.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/amd_detail)

  # install rocdecode samples -- {ROCM_PATH}/share/rocdecode
  install(DIRECTORY cmake DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME})
  install(DIRECTORY utils/rocvideodecode DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils)
  install(FILES samples/videoDecode/CMakeLists.txt samples/videoDecode/README.md samples/videoDecode/videodecode.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/videoDecode)
  install(FILES samples/videoDecodeRaw/CMakeLists.txt samples/videoDecodeRaw/README.md samples/videoDecodeRaw/videodecoderaw.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/videoDecodeRaw)
  install(FILES samples/videoDecodeMem/CMakeLists.txt samples/videoDecodeMem/README.md samples/videoDecodeMem/videodecodemem.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/videoDecodeMem)
  install(FILES samples/videoDecodePerf/CMakeLists.txt samples/videoDecodePerf/README.md samples/videoDecodePerf/videodecodeperf.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/videoDecodePerf)
  install(FILES samples/videoDecodeRGB/CMakeLists.txt samples/videoDecodeRGB/README.md samples/videoDecodeRGB/videodecrgb.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/videoDecodeRGB)
  install(FILES samples/videoDecodeBatch/CMakeLists.txt samples/videoDecodeBatch/README.md samples/videoDecodeBatch/videodecodebatch.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/videoDecodeBatch)
  install(FILES samples/rocdecDecode/CMakeLists.txt samples/rocdecDecode/README.md samples/rocdecDecode/rocdecdecode.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/rocdecDecode)
  install(FILES samples/videoDecodePicFiles/CMakeLists.txt samples/videoDecodePicFiles/README.md samples/videoDecodePicFiles/videodecodepicfiles.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/videoDecodePicFiles)
  install(FILES samples/common.h samples/SampleCommon.cmake DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples)
  install(FILES utils/md5.h DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils)
  install(FILES utils/video_demuxer.h DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils)
  install(FILES utils/colorspace_kernels.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils)
  install(FILES utils/colorspace_kernels.h DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils)
  install(FILES utils/resize_kernels.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils)
  install(FILES utils/resize_kernels.h DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils)
  install(FILES utils/video_post_process.h DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils)
  install(FILES data/videos/AMD_driving_virtual_20-H265.mp4 data/videos/AMD_driving_virtual_20-H265.265 data/videos/AMD_driving_virtual_20-H264.mp4 data/videos/AMD_driving_virtual_20-H264.264 data/videos/AMD_driving_virtual_20-AV1.mp4 data/videos/AMD_driving_virtual_20-AV1.ivf data/videos/AMD_driving_virtual_20-VP9.ivf DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/video)
  install(DIRECTORY data/frames/ DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/frames)
  # install license information - {ROCM_PATH}/share/doc/rocdecode
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR})
  # install test cmake
  install(FILES test/CMakeLists.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test)
  install(DIRECTORY test/testScripts DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test)
  install(FILES test/rocDecodeNegativeApiTests/CMakeLists.txt test/rocDecodeNegativeApiTests/README.md test/rocDecodeNegativeApiTests/rocdecode_api_negative_tests.cpp test/rocDecodeNegativeApiTests/rocdecode_api_negative_tests.h test/rocDecodeNegativeApiTests/rocdecodenegativetest.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test/rocDecodeNegativeApiTests)

  message("-- ${White}AMD ROCm ${PROJECT_NAME} -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
  message("-- ${White}AMD ROCm ${PROJECT_NAME} -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")

  # Cmake module config file configurations
  set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" CACHE INTERNAL "Default module path.")

  # Export the package for use from the build-tree
  ## (this registers the build-tree with a global CMake-registry)
  export(PACKAGE ${PROJECT_NAME})
  set(CONF_LIB_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

  # Create the rocdecode-config.cmake and rocdecode-config-version files
  include(CMakePackageConfigHelpers)
  set(CONFIG_PACKAGE_INSTALL_DIR ${CONF_LIB_DIR}/cmake/${PROJECT_NAME})
  set(LIB_INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
  set(INCLUDE_INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
  set(BIN_INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")

  # Generate config Version File for rocdecode
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    COMPATIBILITY SameMajorVersion
  )

  # Generate Config File for rocdecode
  configure_package_config_file(
    ${CMAKE_MODULE_PATH}/${PROJECT_NAME}-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR}
    PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR BIN_INSTALL_DIR
  )

  # Install the rocdecode-config.cmake and rocdecode-config-version.cmake
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

  # make test with CTest (Linux only for now, as tests use FFmpeg)
  if(IS_LINUX)
    set(BUILD_FROM_SOURCE ON)
    enable_testing()
    include(CTest)
    add_subdirectory(test)
  else()
    message("-- ${Yellow}${PROJECT_NAME} -- Tests skipped on Windows (require FFmpeg integration)${ColourReset}")
  endif()

else()
  message("-- ${Red}AMD ROCm ${PROJECT_NAME} -- unmet dependencies${ColourReset}")
  if(NOT HIP_FOUND)
    message(FATAL_ERROR "-- ERROR!: HIP Not Found! On Windows, ensure HIP_PATH or ROCM_PATH is set.")
  endif()
  if(IS_LINUX)
    if(NOT Libva_FOUND)
      message(FATAL_ERROR "-- ERROR!: libva Not Found!")
    endif()
    if(NOT Libdrm_amdgpu_FOUND)
      message(FATAL_ERROR "-- ERROR!: libdrm_amdgpu Not Found!")
    endif()
  endif()
  if(NOT Threads_FOUND)
    message(FATAL_ERROR "-- ERROR!: Threads Not Found!")
  endif()
endif()
