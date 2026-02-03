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
set(VERSION "1.7.0")

# Set Project Version and Language
project(rocdecode VERSION ${VERSION} LANGUAGES CXX)

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
set(CMAKE_INSTALL_INCLUDEDIR "include" CACHE STRING "Include install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

find_program(MAKE_NSIS_EXE makensis)
find_program(RPMBUILD_EXE rpmbuild)
find_program(DPKG_EXE dpkg)

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

# Find Ubuntu 22.04
file(READ "/etc/os-release" OS_RELEASE)
string(REGEX MATCH "22.04" UBUNTU_22_FOUND ${OS_RELEASE})

# Check Libva version compatibility based on installation method and OS:
# - Ubuntu 22.04 with standard ROCm installation: requires libva >= 1.16
# - TheRock installation (any OS): libva >= 1.8 is sufficient
# - Other OS versions with standard ROCm: libva >= 1.8 is sufficient
if(Libva_FOUND)
  if(UBUNTU_22_FOUND AND NOT USING_THE_ROCK)
    if((${Libva_VERSION_MAJOR} GREATER_EQUAL 1) AND (${Libva_VERSION_MINOR} GREATER_EQUAL 16))
      message("-- ${White}\tLibva Version Supported${ColourReset}")
    else()
      set(Libva_FOUND FALSE)
      message("-- ${Yellow}\tLibva Version Not Supported${ColourReset}")
    endif()
  else()
    if((${Libva_VERSION_MAJOR} GREATER_EQUAL 1) AND (${Libva_VERSION_MINOR} GREATER_EQUAL 8))
      message("-- ${White}\tLibva Version Supported${ColourReset}")
    else()
      set(Libva_FOUND FALSE)
      message("-- ${Yellow}\tLibva Version Not Supported${ColourReset}")
    endif()
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

if(HIP_FOUND AND Libva_FOUND AND Libdrm_amdgpu_FOUND)

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
  install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
  install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
  install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
  # install rocdecode include files -- {ROCM_PATH}/include/rocdecode
  install(FILES api/rocdecode/rocdecode.h api/rocdecode/rocparser.h api/rocdecode/roc_bitstream_reader.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME} COMPONENT dev)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rocdecode_version.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME} COMPONENT dev)
  # install rocdecode api trace include file -- {ROCM_PATH}/include/rocdecode/amd_detail
  install(FILES api/amd_detail/rocdecode_api_trace.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/amd_detail COMPONENT dev)

  # install rocdecode samples -- {ROCM_PATH}/share/rocdecode
  install(DIRECTORY cmake DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME} COMPONENT dev)
  install(DIRECTORY utils/rocvideodecode DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils COMPONENT dev)
  install(FILES samples/videoDecode/CMakeLists.txt samples/videoDecode/README.md samples/videoDecode/videodecode.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/videoDecode COMPONENT dev)
  install(FILES samples/videoDecodeRaw/CMakeLists.txt samples/videoDecodeRaw/README.md samples/videoDecodeRaw/videodecoderaw.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/videoDecodeRaw COMPONENT dev)
  install(FILES samples/videoDecodeMem/CMakeLists.txt samples/videoDecodeMem/README.md samples/videoDecodeMem/videodecodemem.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/videoDecodeMem COMPONENT dev)
  install(FILES samples/videoDecodePerf/CMakeLists.txt samples/videoDecodePerf/README.md samples/videoDecodePerf/videodecodeperf.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/videoDecodePerf COMPONENT dev)
  install(FILES samples/videoDecodeRGB/CMakeLists.txt samples/videoDecodeRGB/README.md samples/videoDecodeRGB/videodecrgb.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/videoDecodeRGB COMPONENT dev)
  install(FILES samples/videoDecodeBatch/CMakeLists.txt samples/videoDecodeBatch/README.md samples/videoDecodeBatch/videodecodebatch.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/videoDecodeBatch COMPONENT dev)
  install(FILES samples/rocdecDecode/CMakeLists.txt samples/rocdecDecode/README.md samples/rocdecDecode/rocdecdecode.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/rocdecDecode COMPONENT dev)
  install(FILES samples/common.h DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples COMPONENT dev)
  install(FILES utils/md5.h DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils COMPONENT dev)
  install(FILES utils/video_demuxer.h DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils COMPONENT dev)
  install(FILES utils/colorspace_kernels.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils COMPONENT dev)
  install(FILES utils/colorspace_kernels.h DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils COMPONENT dev)
  install(FILES utils/resize_kernels.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils COMPONENT dev)
  install(FILES utils/resize_kernels.h DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils COMPONENT dev)
  install(FILES utils/video_post_process.h DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/utils COMPONENT dev)
  install(FILES data/videos/AMD_driving_virtual_20-H265.mp4 data/videos/AMD_driving_virtual_20-H265.265 data/videos/AMD_driving_virtual_20-H264.mp4 data/videos/AMD_driving_virtual_20-H264.264 data/videos/AMD_driving_virtual_20-AV1.mp4 data/videos/AMD_driving_virtual_20-AV1.ivf data/videos/AMD_driving_virtual_20-VP9.ivf DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/video COMPONENT dev)
  install(DIRECTORY data/frames/ DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/frames COMPONENT dev)
  # install license information - {ROCM_PATH}/share/doc/rocdecode
  set(CPACK_RESOURCE_FILE_LICENSE  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
  install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT runtime)
  install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan COMPONENT asan)
  # install test cmake
  install(FILES test/CMakeLists.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test COMPONENT test)
  install(DIRECTORY test/testScripts DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test COMPONENT test)
  install(FILES test/rocDecodeNegativeApiTests/CMakeLists.txt test/rocDecodeNegativeApiTests/README.md test/rocDecodeNegativeApiTests/rocdecode_api_negative_tests.cpp test/rocDecodeNegativeApiTests/rocdecode_api_negative_tests.h test/rocDecodeNegativeApiTests/rocdecodenegativetest.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test/rocDecodeNegativeApiTests COMPONENT dev)

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
    COMPONENT dev
  )

  # Install the export set for use with the install-tree
  install(EXPORT ${PROJECT_NAME}-targets 
    FILE ${PROJECT_NAME}-targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" 
    COMPONENT dev
  )

  # make test with CTest
  set(BUILD_FROM_SOURCE ON)
  enable_testing()
  include(CTest)
  add_subdirectory(test)

  # set package information
  set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
  set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
  set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
  set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
  set(CPACK_PACKAGE_LICENSE "MIT")
  # TBD: Change to generic support email
  set(CPACK_PACKAGE_CONTACT "rocdecode Support <mivisionx.support@amd.com>")
  set(CPACK_PACKAGE_VENDOR "AMD ROCm")
  set(CPACK_PACKAGE_GROUP "Development/Tools")
  set(CPACK_PACKAGE_HOMEPAGE "https://github.com/ROCm/rocDecode")

  if(ENABLE_ASAN_PACKAGING)
    set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "AMD rocdecode address sanitizer libraries")
    set(CPACK_RPM_PACKAGE_SUMMARY "AMD rocdecode address sanitizer libraries")
  else()
    set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "AMD rocdecode is a high performance video decode SDK for AMD GPUs")
    set(CPACK_RPM_PACKAGE_SUMMARY "AMD rocdecode is a high performance video decode SDK for AMD GPUs")
  endif()

  if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
    set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
  endif()

  # Set the dependent packages
  # libva 2.16 (API 1.16) is needed to use mesa-amdgpu-va-drivers, libva-amdgpu
  # (AMD build of libva 2.16) should be used if distro's libva is too old
  set(ROCDECODE_DEBIAN_RUNTIME_PACKAGE_LIST  "hip-runtime-amd, libva-drm2 (>= 2.16.0) | libva-amdgpu-drm2, mesa-amdgpu-va-drivers")
  # Unfortunately RPM has a mix of naming schemes; RHEL has "libva", SLE has "libva2-drm2"
  set(ROCDECODE_RPM_RUNTIME_PACKAGE_LIST  "hip-runtime-amd, (libva or libva-drm2), mesa-amdgpu-va-drivers")
  # Add rocprofiler-register dependencies
  if(ROCDECODE_ENABLE_ROCPROFILER_REGISTER)
    set(ROCDECODE_DEBIAN_RUNTIME_PACKAGE_LIST "${ROCDECODE_DEBIAN_RUNTIME_PACKAGE_LIST}, rocprofiler-register")
    set(ROCDECODE_RPM_RUNTIME_PACKAGE_LIST "${ROCDECODE_RPM_RUNTIME_PACKAGE_LIST}, rocprofiler-register")
  endif()

  # Set the dev dependent packages
  set(ROCDECODE_DEBIAN_DEV_PACKAGE_LIST  "hip-dev, libva-dev (>= 2.16.0) | libva-amdgpu-dev, pkg-config, libavcodec-dev, libavformat-dev, libavutil-dev")
  if(UBUNTU_22_FOUND)
    set(ROCDECODE_DEBIAN_DEV_PACKAGE_LIST "${ROCDECODE_DEBIAN_DEV_PACKAGE_LIST}, libstdc++-12-dev")
  endif()
  # TBD - RPM packages need Fusion Packages - "libavcodec-devel, libavformat-devel, libavutil-devel"
  set(ROCDECODE_RPM_DEV_PACKAGE_LIST  "hip-devel, libva-devel, pkg-config")
  
  # '%{?dist}' breaks manual builds on debian systems due to empty Provides
  execute_process(
    COMMAND rpm --eval %{?dist}
    RESULT_VARIABLE PROC_RESULT
    OUTPUT_VARIABLE EVAL_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
    string(APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
  endif()

  # set package 
  set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  set(CPACK_DEBIAN_PACKAGE_HOMEPAGE ${CPACK_PACKAGE_HOMEPAGE})
  set(CPACK_RPM_PACKAGE_URL ${CPACK_PACKAGE_HOMEPAGE})
  set(CPACK_RPM_PACKAGE_AUTOREQPROV "no")
  # Debian package
  set(CPACK_DEB_COMPONENT_INSTALL ON)
  set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
  set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS "rocm-core, ${ROCDECODE_DEBIAN_RUNTIME_PACKAGE_LIST}")
  set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${PROJECT_NAME}-dev")
  set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS 
  "rocm-core, ${PROJECT_NAME}, ${ROCDECODE_DEBIAN_DEV_PACKAGE_LIST}")
  # Debian package - specific variable for ASAN
  set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
  set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "rocm-core-asan, ${ROCDECODE_DEBIAN_RUNTIME_PACKAGE_LIST}" )
  # Debian package - Test
  set(CPACK_DEBIAN_TEST_PACKAGE_NAME "${PROJECT_NAME}-test" )
  set(CPACK_DEBIAN_TEST_PACKAGE_DEPENDS "rocm-core, ${CPACK_DEBIAN_DEV_PACKAGE_NAME}" )
  # Debian package - Host
  set(CPACK_DEBIAN_HOST_PACKAGE_NAME "${PROJECT_NAME}-host" )
  set(CPACK_DEBIAN_HOST_PACKAGE_DEPENDS "rocm-core, ${CPACK_DEBIAN_DEV_PACKAGE_NAME}" )
  # RPM package
  set(CPACK_RPM_COMPONENT_INSTALL ON)
  set(CPACK_RPM_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
  set(CPACK_RPM_RUNTIME_PACKAGE_REQUIRES "rocm-core, ${ROCDECODE_RPM_RUNTIME_PACKAGE_LIST}")
  set(CPACK_RPM_DEV_PACKAGE_NAME "${PROJECT_NAME}-devel")
  set(CPACK_RPM_DEV_PACKAGE_REQUIRES "rocm-core, ${PROJECT_NAME}, ${ROCDECODE_RPM_DEV_PACKAGE_LIST}")
  set(CPACK_RPM_PACKAGE_LICENSE "MIT" )
  # RPM package specific variable for ASAN
  set(CPACK_RPM_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
  set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "rocm-core-asan, ${ROCDECODE_RPM_RUNTIME_PACKAGE_LIST}" )
  # RPM package specific variable for Test
  set(CPACK_RPM_TEST_PACKAGE_NAME "${PROJECT_NAME}-test" )
  set(CPACK_RPM_TEST_PACKAGE_REQUIRES "rocm-core, ${CPACK_RPM_DEV_PACKAGE_NAME}" )
  # RPM package specific variable for Host
  set(CPACK_RPM_HOST_PACKAGE_NAME "${PROJECT_NAME}-host" )
  set(CPACK_RPM_HOST_PACKAGE_REQUIRES "rocm-core, ${CPACK_RPM_DEV_PACKAGE_NAME}" )

  if(NOT ROCM_DEP_ROCMCORE)
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_RUNTIME_PACKAGE_REQUIRES ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core-asan," "" CPACK_RPM_ASAN_PACKAGE_REQUIRES ${CPACK_RPM_ASAN_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_DEV_PACKAGE_REQUIRES ${CPACK_RPM_DEV_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_TEST_PACKAGE_REQUIRES ${CPACK_RPM_TEST_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_HOST_PACKAGE_REQUIRES ${CPACK_RPM_TEST_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core-asan," "" CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_TEST_PACKAGE_DEPENDS ${CPACK_DEBIAN_TEST_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_HOST_PACKAGE_DEPENDS ${CPACK_DEBIAN_TEST_PACKAGE_DEPENDS})
  endif()

  if(ENABLE_ASAN_PACKAGING)
    # ASAN Package requires asan component with only libraries and license file
    set(CPACK_COMPONENTS_ALL asan)
  else()
    set(CPACK_COMPONENTS_ALL runtime dev test host)
  endif()

  set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
  set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
  set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
  set(CPACK_RPM_PACKAGE_RELEASE "local")

  if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  endif()
  if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
  endif()

  set(CPACK_GENERATOR "TGZ;ZIP")
  if(EXISTS ${MAKE_NSIS_EXE})
    list(APPEND CPACK_GENERATOR "NSIS")
  endif()
  if(EXISTS ${RPMBUILD_EXE})
    list(APPEND CPACK_GENERATOR "RPM")
    message("-- ${White}${PROJECT_NAME}: .rpm RunTime Package Requirements -- ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES}${ColourReset}")
    message("-- ${White}${PROJECT_NAME}: .rpm Devel Package Requirements -- ${CPACK_RPM_DEV_PACKAGE_REQUIRES}${ColourReset}")
  endif()
  if(EXISTS ${DPKG_EXE})
    list(APPEND CPACK_GENERATOR "DEB")
    message("-- ${White}${PROJECT_NAME}: .deb RunTime Package Requirements -- ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS}${ColourReset}")
    message("-- ${White}${PROJECT_NAME}: .deb Dev Package Requirements -- ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS}${ColourReset}")
  endif()

  include(CPack)

  cpack_add_component(runtime
                    DISPLAY_NAME "rocdecode Runtime Package"
                    DESCRIPTION "High perf video decode SDK for AMD GPUs. Rocdecode library and license.txt")

  cpack_add_component(dev
                    DISPLAY_NAME "rocdecode Develop Package"
                    DESCRIPTION "High perf video decode SDK for AMD GPUs. Rocdecode header files, and samples")

  cpack_add_component(asan
                    DISPLAY_NAME "rocdecode ASAN Package"
                    DESCRIPTION "High perf video decode SDK for AMD GPUs. rocdecode ASAN libraries")

  cpack_add_component(test
                    DISPLAY_NAME "rocdecode Test Package"
                    DESCRIPTION "High perf video decode SDK for AMD GPUs. rocdecode test")
  
    # make rocdecode-host for decoding on HOST only if FFMPEG is available
  if(ROCDECODE_ENABLE_HOST_DECODER AND FFMPEG_FOUND)
    cpack_add_component(host
                      DISPLAY_NAME "rocdecode host package"
                      DESCRIPTION "High perf video decode host SDK for AMD GPUs")
  endif()

else()
  message("-- ${Red}AMD ROCm ${PROJECT_NAME} -- unmet dependencies${ColourReset}")
  if(NOT HIP_FOUND)
    message(FATAL_ERROR "-- ERROR!: HIP Not Found! - please install rocm-hip-runtime-dev!")
  endif()
  if(NOT Libva_FOUND)
    message(FATAL_ERROR "-- ERROR!: libva Not Found - please install libva-dev/libva-devel!")
  endif()
  if(NOT Libdrm_amdgpu_FOUND)
    message(FATAL_ERROR "-- ERROR!: libdrm_amdgpu Not Found - please install libdrm-dev(DEBIAN)/libdrm-devel(RPM) package!")
  endif()
endif()
