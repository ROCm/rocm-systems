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

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if (NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
  set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

# rocdecode Version
set(VERSION "1.8.0")

# Set Project Version and Language
project(rocdecode VERSION ${VERSION} LANGUAGES CXX)

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
  # -g  -- generate debugging information, generate debugging information, dwarf-4 for making valgrind work
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -gdwarf-4")
else()
  # -O3       -- Optimize output file 
  # -DNDEBUG  -- turn off asserts 
  # -fPIC     -- Generate position-independent code if possible
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC")
endif()
message("-- ${BoldBlue}${PROJECT_NAME} Build Type -- ${CMAKE_BUILD_TYPE}${ColourReset}")

# Developer options
# Add an option for enabling the rocprofiler-register
option(ROCDECODE_ENABLE_ROCPROFILER_REGISTER "Enable rocprofiler-register support" ON)
# Add an option for enabling FFMPEG avcodec host-based decoder
option(ROCDECODE_ENABLE_HOST_DECODER "Enable rocdecode-host-based decoder support" ON)

set (HIP_PLATFORM amd CACHE STRING "HIP platform")
find_package(HIP QUIET)
find_package(Libva QUIET)
find_package(Libdrm_amdgpu QUIET)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads QUIET)

# Check Libva version compatibility: requires libva >= 1.22
if(Libva_FOUND)
  if("${Libva_VERSION}" VERSION_GREATER_EQUAL "1.22")
    message("-- ${White}\tLibva Version Supported${ColourReset}")
  else()
    set(Libva_FOUND FALSE)
    message("-- ${Yellow}\tLibva Version Not Supported (found ${Libva_VERSION}, need >= 1.22)${ColourReset}")
  endif()
endif()

if(ROCDECODE_ENABLE_ROCPROFILER_REGISTER)
  find_package(rocprofiler-register QUIET
    HINTS $ENV{rocprofiler_register_ROOT} $ENV{ROCPROFILER_REGISTER_ROOT} ${CMAKE_INSTALL_PREFIX}
    PATHS ${ROCM_PATH})
endif()
if(ROCDECODE_ENABLE_HOST_DECODER)
  find_package(FFmpeg QUIET)
endif()

if(HIP_FOUND AND Libva_FOUND AND Libdrm_amdgpu_FOUND AND Threads_FOUND)

  # HIP
  set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)
  # LibVA
  include_directories(${LIBVA_INCLUDE_DIR})
  set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LIBVA_LIBRARY})
  set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LIBVA_DRM_LIBRARY})

  # DRM_AMDGPU
  include_directories(${LIBDRM_AMDGPU_INCLUDE_DIR})
  set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LIBDRM_AMDGPU_LIBRARY})

  # rocprofiler
  if (rocprofiler-register_FOUND)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocprofiler-register::rocprofiler-register)
  endif()

  # Threads library (required for std::thread support)
  set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} Threads::Threads)

  # local include files
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api ${CMAKE_CURRENT_SOURCE_DIR}/src/rocdecode ${CMAKE_CURRENT_SOURCE_DIR}/src/parser ${CMAKE_CURRENT_SOURCE_DIR}/src/rocdecode/vaapi)
  # source files
  file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
  list(FILTER SOURCES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/src/rocdecode-host/*.*")
  # rocdecode.so
  add_library(${PROJECT_NAME} SHARED ${SOURCES})

  # --all-warnings/-Wall       -- Enable most warning messages
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARY_LIST})

  set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

  if(USING_THE_ROCK)
    set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN;$ORIGIN/rocm_sysdeps/lib" BUILD_WITH_INSTALL_RPATH TRUE)
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

  # make rocdecode-host for decoding on HOST only if FFMPEG is available
  if(ROCDECODE_ENABLE_HOST_DECODER AND FFMPEG_FOUND)
    add_subdirectory(src/rocdecode-host)
  else()
    if(NOT FFMPEG_FOUND)
      message("-- ${Yellow}${PROJECT_NAME} -- FFMPEG NOT Found; rocdecode-host exculded${ColourReset}")
    endif()
  endif()

  # Generate BUILD_INFO
  configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/api/rocdecode_version.h.in ${CMAKE_CURRENT_BINARY_DIR}/rocdecode_version.h @ONLY )

  # install rocdecode libs -- {ROCM_PATH}/lib
  install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} NAMELINK_SKIP)
  install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} NAMELINK_ONLY)
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
  install(FILES samples/common.h DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples)
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

  # make test with CTest
  set(BUILD_FROM_SOURCE ON)
  enable_testing()
  include(CTest)
  add_subdirectory(test)

else()
  message("-- ${Red}AMD ROCm ${PROJECT_NAME} -- unmet dependencies${ColourReset}")
  if(NOT HIP_FOUND)
    message(FATAL_ERROR "-- ERROR!: HIP Not Found!")
  endif()
  if(NOT Libva_FOUND)
    message(FATAL_ERROR "-- ERROR!: libva Not Found!")
  endif()
  if(NOT Libdrm_amdgpu_FOUND)
    message(FATAL_ERROR "-- ERROR!: libdrm_amdgpu Not Found!")
  endif()
  if(NOT Threads_FOUND)
    message(FATAL_ERROR "-- ERROR!: Threads Not Found!")
  endif()
endif()
