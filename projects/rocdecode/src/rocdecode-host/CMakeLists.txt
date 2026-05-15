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

# rocdecode-host Version
set(VERSION "1.0.0")

# Set Project Version and Language
project(rocdecode-host VERSION ${VERSION} LANGUAGES CXX)

# add cmake find Config locations
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake)

set (HIP_PLATFORM amd CACHE STRING "HIP platform")
find_package(HIP QUIET)
find_package(Threads QUIET)
find_package(rocprofiler-register QUIET)
find_package(FFmpeg QUIET)

if(HIP_FOUND AND Threads_FOUND AND FFMPEG_FOUND)
  # HIP
  set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)
  # FFMPEG
  include_directories(${AVUTIL_INCLUDE_DIR} ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR})
  set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${FFMPEG_LIBRARIES})

  # rocprofiler
  if (rocprofiler-register_FOUND)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocprofiler-register::rocprofiler-register)
  endif()

  # local include files
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../api ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../rocdecode ${CMAKE_CURRENT_SOURCE_DIR}/../parser)
  # source files
  file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
  # rocdecode-host.so
  add_library(${PROJECT_NAME} SHARED ${SOURCES})

  target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARY_LIST})
  # FFMPEG multi-version support
  if(_FFMPEG_AVCODEC_VERSION VERSION_LESS_EQUAL 58.134.100) #FFmpeg version <= 4.4
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_AVCODEC_GREATER_THAN_58_134=0)
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_AVCODEC_GREATER_THAN_60_31=0)
  elseif(_FFMPEG_AVCODEC_VERSION VERSION_GREATER 58.134.100 AND _FFMPEG_AVCODEC_VERSION VERSION_LESS_EQUAL 60.31.100) #FFmpeg version >= 4.4 && version < 6.1
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_AVCODEC_GREATER_THAN_58_134=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_AVCODEC_GREATER_THAN_60_31=0)
  else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_AVCODEC_GREATER_THAN_58_134=0) #FFmpeg version >= 6.1
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_AVCODEC_GREATER_THAN_60_31=1)
  endif() #FFMpeg version >= 7.1
  set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  if(USING_THE_ROCK)
    set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN;$ORIGIN/rocm_sysdeps/lib" BUILD_WITH_INSTALL_RPATH TRUE)
  endif()

  message("-- ${White}AMD ROCm ${PROJECT_NAME} -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
  message("-- ${White}AMD ROCm ${PROJECT_NAME} -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")

  # Generate BUILD_INFO
  configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/../../api/rocdecode-host_version.h.in ${CMAKE_CURRENT_BINARY_DIR}/rocdecode-host_version.h @ONLY )

  # install rocdecode-host libs -- {ROCM_PATH}/lib
  set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
  install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT host NAMELINK_SKIP)
  install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT host NAMELINK_ONLY)
  install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT host)
  # install rocdecode include files -- {ROCM_PATH}/include/rocdecode
  install(FILES ${PROJECT_SOURCE_DIR}/../../api/rocdecode/rocdecode_host.h 
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocdecode COMPONENT host)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_version.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocdecode COMPONENT host)
  # TBD: NOTE the host utils are temp delivered with dev package, need to update downstream apps to move to host package
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../../utils/ffmpegvideodecode DESTINATION ${CMAKE_INSTALL_DATADIR}/rocdecode/utils COMPONENT dev)
  
  # Cmake module config file configurations
  set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake_modules/" CACHE INTERNAL "Default module path.")

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
    COMPONENT host
  )

  # Install the export set for use with the install-tree
  install(EXPORT ${PROJECT_NAME}-targets 
    FILE ${PROJECT_NAME}-targets.cmake
    NAMESPACE rocdecode::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" 
    COMPONENT host
  )

else()
  message("-- ERROR!: ${PROJECT_NAME} excluded! please install all the dependencies and try again!")
  if (NOT HIP_FOUND)
      message(FATAL_ERROR "-- ERROR!: HIP Not Found! - please install ROCm and HIP!")
  endif()
  if (NOT Threads_FOUND)
      message(FATAL_ERROR "-- ERROR!: Threads Not Found! - please install Threads!")
  endif()
  if (NOT FFMPEG_FOUND)
      message(FATAL_ERROR "-- ERROR!: FFMPEG Not Found! - please install FFMPEG!")
  endif()
endif()
