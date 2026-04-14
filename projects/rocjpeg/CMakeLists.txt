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

if(USING_THE_ROCK)
  if(NOT DEFINED ENV{HIP_DEVICE_LIB_PATH})
    set(ENV{HIP_DEVICE_LIB_PATH} ${ROCM_PATH}/lib/llvm/amdgcn/bitcode)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "HIP_DEVICE_LIB_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode")
  endif()
endif()

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
  set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

# rocjpeg Version
# NOTE: package version and rocjpeg_version.h is generated with this version
set(VERSION "1.4.0")

# Set Project Version and Language
project(rocjpeg VERSION ${VERSION} LANGUAGES CXX)

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

    set(OPTIONAL_GPU_TARGETS "gfx950;gfx1151;gfx1200;gfx1201")
    if(HAS_ROCM_CHECK_TARGET_IDS)
        rocm_check_target_ids(OPTIONAL_GPU_TARGETS_AVAILABLE TARGETS ${OPTIONAL_GPU_TARGETS})
    else() # if we don't have rocm_check_target_ids, just assume the targets are available
        set(OPTIONAL_GPU_TARGETS_AVAILABLE "${OPTIONAL_GPU_TARGETS}")
    endif()
    set(DEFAULT_GPU_TARGETS "gfx908;gfx90a;gfx942;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;${OPTIONAL_GPU_TARGETS_AVAILABLE}")
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
find_package(Libdrm_amdgpu QUIET)

# Find Ubuntu 22.04
file(READ "/etc/os-release" OS_RELEASE)
string(REGEX MATCH "22.04" UBUNTU_22_FOUND ${OS_RELEASE})

# Check Libva version compatibility based on installation method and OS:
# - Ubuntu 22.04 with standard ROCm installation: requires libva >= 1.16
# - Other OS versions with standard ROCm: libva >= 1.8 is sufficient
# - TheRock installation (any OS): libva >= 1.8 is sufficient
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

if(ROCJPEG_ENABLE_ROCPROFILER_REGISTER)
  find_package(rocprofiler-register QUIET
    HINTS $ENV{rocprofiler_register_ROOT} $ENV{ROCPROFILER_REGISTER_ROOT} ${CMAKE_INSTALL_PREFIX}
    PATHS ${ROCM_PATH})
endif()

if(HIP_FOUND AND Libva_FOUND AND Libdrm_amdgpu_FOUND)
  # HIP
  set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::device)
  # LibVA
  include_directories(${LIBVA_INCLUDE_DIR})
  set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LIBVA_LIBRARY})
  set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LIBVA_DRM_LIBRARY})
  # DRM_AMDGPU
  include_directories(${LIBDRM_AMDGPU_INCLUDE_DIR})
  set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LIBDRM_AMDGPU_LIBRARY})

  # rocprofiler
  if(rocprofiler-register_FOUND)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocprofiler-register::rocprofiler-register)
  endif()

  # local include files
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api ${CMAKE_CURRENT_SOURCE_DIR}/src)
  # source files
  file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
  # rocjpeg.so
  add_library(${PROJECT_NAME} SHARED ${SOURCES})

  if(CMAKE_BUILD_TYPE MATCHES Debug)
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

  target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARY_LIST})

  set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

  if(USING_THE_ROCK)
    set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN;$ORIGIN/rocm_sysdeps/lib" BUILD_WITH_INSTALL_RPATH TRUE)
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
  install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
  install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
  install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
  # install rocjpeg include files -- {ROCM_PATH}/include/rocjpeg
  install(FILES api/rocjpeg/rocjpeg.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME} COMPONENT dev)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rocjpeg_version.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME} COMPONENT dev)
  # install rocjpeg api trace include file -- {ROCM_PATH}/include/rocjpeg/amd_detail
  install(FILES api/amd_detail/rocjpeg_api_trace.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/amd_detail COMPONENT dev)

  # install rocjpeg samples -- {ROCM_PATH}/share/rocjpeg
  install(FILES samples/jpegDecode/CMakeLists.txt samples/jpegDecode/jpegdecode.cpp samples/jpegDecode/README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/jpegDecode COMPONENT dev)
  install(FILES samples/jpegDecodePerf/CMakeLists.txt samples/jpegDecodePerf/jpegdecodeperf.cpp samples/jpegDecodePerf/README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/jpegDecodePerf COMPONENT dev)
  install(FILES samples/jpegDecodeBatched/CMakeLists.txt samples/jpegDecodeBatched/jpegdecodebatched.cpp samples/jpegDecodeBatched/README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples/jpegDecodeBatched COMPONENT dev)
  install(FILES samples/rocjpeg_samples_utils.h DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/samples COMPONENT dev)
  install(DIRECTORY data/images DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME} COMPONENT dev)
  # install license information - {ROCM_PATH}/share/doc/rocjpeg
  set(CPACK_RESOURCE_FILE_LICENSE  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
  install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT runtime)
  install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan COMPONENT asan)
  # install test cmake
  install(FILES test/CMakeLists.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test COMPONENT test)
  install(FILES test/rocjpeg_negative_api_tests/CMakeLists.txt test/rocjpeg_negative_api_tests/README.md test/rocjpeg_negative_api_tests/rocjpeg_api_negative_tests.cpp test/rocjpeg_negative_api_tests/rocjpeg_api_negative_tests.h test/rocjpeg_negative_api_tests/rocjpegnegativetest.cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test/rocjpeg_negative_api_tests COMPONENT test)

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
  set(CPACK_PACKAGE_CONTACT "rocjpeg Support <mivisionx.support@amd.com>")
  set(CPACK_PACKAGE_VENDOR "AMD ROCm")
  set(CPACK_PACKAGE_GROUP "Development/Tools")
  set(CPACK_PACKAGE_HOMEPAGE "https://github.com/ROCm/rocjpeg")

  if(ENABLE_ASAN_PACKAGING)
    set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "AMD rocjpeg address sanitizer libraries")
    set(CPACK_RPM_PACKAGE_SUMMARY "AMD rocjpeg address sanitizer libraries")
  else()
    set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "AMD rocjpeg is a high performance JPEG decode SDK for AMD GPUs")
    set(CPACK_RPM_PACKAGE_SUMMARY "AMD rocjpeg is a high performance JPEG decode SDK for AMD GPUs")
  endif()

  if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
    set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
  endif()

  # Set the dependent packages
  set(rocJPEG_DEBIAN_PACKAGE_LIST  "hip-runtime-amd, libva-drm2 (>= 2.16.0) | libva-amdgpu-drm2, mesa-amdgpu-va-drivers")
  set(rocJPEG_RPM_PACKAGE_LIST     "hip-runtime-amd, (libva or libva-drm2), mesa-amdgpu-va-drivers")
  # Add rocprofiler-register dependencies
  if(ROCJPEG_ENABLE_ROCPROFILER_REGISTER)
    set(rocJPEG_DEBIAN_PACKAGE_LIST "${rocJPEG_DEBIAN_PACKAGE_LIST}, rocprofiler-register")
    set(rocJPEG_RPM_PACKAGE_LIST "${rocJPEG_RPM_PACKAGE_LIST}, rocprofiler-register")
  endif()

  set(rocJPEG_DEBIAN_DEV_PACKAGE_LIST  "hip-dev, libva-dev (>= 2.16.0) | libva-amdgpu-dev")
  if(UBUNTU_22_FOUND)
    set(rocJPEG_DEBIAN_DEV_PACKAGE_LIST "${rocJPEG_DEBIAN_DEV_PACKAGE_LIST}, libstdc++-12-dev")
  endif()
  set(rocJPEG_RPM_DEV_PACKAGE_LIST  "hip-devel, libva-devel")

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
  set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS "rocm-core, ${rocJPEG_DEBIAN_PACKAGE_LIST}")
  set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${PROJECT_NAME}-dev")
  set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS 
  "rocm-core, ${PROJECT_NAME}, ${rocJPEG_DEBIAN_DEV_PACKAGE_LIST}")
  # Debian package - specific variable for ASAN
  set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
  set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "rocm-core-asan, ${rocJPEG_DEBIAN_PACKAGE_LIST}" )
  # Debian package - Test
  set(CPACK_DEBIAN_TEST_PACKAGE_NAME "${PROJECT_NAME}-test" )
  set(CPACK_DEBIAN_TEST_PACKAGE_DEPENDS "rocm-core, ${CPACK_DEBIAN_DEV_PACKAGE_NAME}" )
  # RPM package
  set(CPACK_RPM_COMPONENT_INSTALL ON)
  set(CPACK_RPM_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
  set(CPACK_RPM_RUNTIME_PACKAGE_REQUIRES "rocm-core, ${rocJPEG_RPM_PACKAGE_LIST}")
  set(CPACK_RPM_RUNTIME_PACKAGE_OBSOLETES "${PROJECT_NAME}")
  set(CPACK_RPM_DEV_PACKAGE_NAME "${PROJECT_NAME}-devel")
  set(CPACK_RPM_DEV_PACKAGE_REQUIRES "rocm-core, ${PROJECT_NAME}, ${rocJPEG_RPM_DEV_PACKAGE_LIST}")
  set(CPACK_RPM_DEV_PACKAGE_OBSOLETES "${PROJECT_NAME}-devel")
  set(CPACK_RPM_PACKAGE_LICENSE "MIT" )
  # RPM package specific variable for ASAN
  set(CPACK_RPM_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
  set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "rocm-core-asan, ${rocJPEG_RPM_PACKAGE_LIST}" )
  set(CPACK_RPM_ASAN_PACKAGE_OBSOLETES "${PROJECT_NAME}-asan")
  # RPM package specific variable for Test
  set(CPACK_RPM_TEST_PACKAGE_NAME "${PROJECT_NAME}-test" )
  set(CPACK_RPM_TEST_PACKAGE_REQUIRES "rocm-core, ${CPACK_RPM_DEV_PACKAGE_NAME}" )
  set(CPACK_RPM_TEST_PACKAGE_OBSOLETES "${PROJECT_NAME}-test")

  if(NOT ROCM_DEP_ROCMCORE)
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_RUNTIME_PACKAGE_REQUIRES ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core-asan," "" CPACK_RPM_ASAN_PACKAGE_REQUIRES ${CPACK_RPM_ASAN_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_DEV_PACKAGE_REQUIRES ${CPACK_RPM_DEV_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_TEST_PACKAGE_REQUIRES ${CPACK_RPM_TEST_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core-asan," "" CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_TEST_PACKAGE_DEPENDS ${CPACK_DEBIAN_TEST_PACKAGE_DEPENDS})
  endif()

  if(ENABLE_ASAN_PACKAGING)
    # ASAN Package requires asan component with only libraries and license file
    set(CPACK_COMPONENTS_ALL asan)
  else()
    set(CPACK_COMPONENTS_ALL runtime dev test)
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
    message("-- ${White}AMD ROCm rocjpeg RunTime Package -- ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES}${ColourReset}")
    message("-- ${White}AMD ROCm rocjpeg Dev Package -- ${CPACK_RPM_DEV_PACKAGE_REQUIRES}${ColourReset}")
  endif()
  if(EXISTS ${DPKG_EXE})
    list(APPEND CPACK_GENERATOR "DEB")
    message("-- ${White}AMD ROCm rocjpeg RunTime Package -- ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS}${ColourReset}")
    message("-- ${White}AMD ROCm rocjpeg Dev Package -- ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS}${ColourReset}")
  endif()

  include(CPack)

  cpack_add_component(runtime
                    DISPLAY_NAME "rocjpeg Runtime Package"
                    DESCRIPTION "High perf JPEG decode SDK for AMD GPUs. Rocjpeg library and license.txt")

  cpack_add_component(dev
                    DISPLAY_NAME "rocjpeg Develop Package"
                    DESCRIPTION "High perf JPEG decode SDK for AMD GPUs. Rocjpeg lib, header files and samples")

  cpack_add_component(asan
                    DISPLAY_NAME "rocjpeg ASAN Package"
                    DESCRIPTION "AMD rocjpeg is a high performance JPEG decode SDK for AMD GPUs. \
rocjpeg ASAN package provides rocjpeg ASAN libraries")

  cpack_add_component(test
                    DISPLAY_NAME "rocjpeg Test Package"
                    DESCRIPTION "AMD rocjpeg is a high performance JPEG decode SDK for AMD GPUs. \
rocjpeg Test package provides rocjpeg Test Components")

else()
  message("-- ${Red}AMD ROCm rocjpeg -- unmet dependencies${ColourReset}")
  if(NOT HIP_FOUND)
    message(FATAL_ERROR "-- ERROR!: HIP Not Found! - please install rocm-hip-runtime-dev!")
  endif()
  if(NOT Libva_FOUND)
    message(FATAL_ERROR "-- ERROR!: libva Not Found - please install (libva-dev >= 2.16 or libva-amdgpu-dev(DEBIAN))/(libva-devel >= 2.16 or libva-amdgpu-devel(RPM)) package!")
  endif()
  if(NOT Libdrm_amdgpu_FOUND)
    message(FATAL_ERROR "-- ERROR!: libdrm_amdgpu Not Found - please install libdrm-amdgpu-dev(DEBIAN)/libdrm-amdgpu-devel(RPM) package!")
  endif()
endif()
