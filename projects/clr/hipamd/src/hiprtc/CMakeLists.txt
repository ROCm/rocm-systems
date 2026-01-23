# Copyright (c) 2020 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# This project builds hiprtc
# If ever this is to be a different lib living in different folder
# Please read this part
# Depends on: rocclr, so find_package(rocclr) will be required
# Building hip header requires hip include folders with hip_version.h

option(BUILD_SHARED_LIBS "Build the shared library" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(BUILD_SHARED_LIBS)
  add_library(hiprtc SHARED)
  target_compile_definitions(hiprtc PRIVATE BUILD_SHARED_LIBS)
  # Windows doesn't have a strip utility, so CMAKE_STRIP won't be set.
  if((CMAKE_BUILD_TYPE STREQUAL "Release") AND NOT ("${CMAKE_STRIP}" STREQUAL ""))
    add_custom_command(TARGET hiprtc POST_BUILD COMMAND ${CMAKE_STRIP} $<TARGET_FILE:hiprtc>)
  endif()
else()
  add_library(hiprtc STATIC $<TARGET_OBJECTS:rocclr>)
endif()

set_target_properties(hiprtc PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
  ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

if(WIN32)
  if(${HIP_LIB_VERSION_MAJOR} VERSION_GREATER 9)
    set(HIP_MAJOR_VERSION "${HIP_LIB_VERSION_MAJOR}")
  else()
    set(HIP_MAJOR_VERSION "0${HIP_LIB_VERSION_MAJOR}")
  endif()
  set(HIP_MINOR_VERSION "0${HIP_LIB_VERSION_MINOR}")
endif()

if(BUILD_SHARED_LIBS)
  if(WIN32)
    set_target_properties(hiprtc PROPERTIES
      OUTPUT_NAME "hiprtc${HIP_MAJOR_VERSION}${HIP_MINOR_VERSION}"
      ARCHIVE_OUTPUT_NAME "hiprtc")
  else()
    set_target_properties(hiprtc PROPERTIES
      VERSION ${HIP_LIB_VERSION_STRING}
      SOVERSION ${HIP_LIB_VERSION_MAJOR})
  endif()
endif()

target_sources(hiprtc PRIVATE hiprtc.cpp ../hip_comgr_helper.cpp hiprtcInternal.cpp)

set_target_properties(hiprtc PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON)

target_include_directories(hiprtc
  PRIVATE
  $<BUILD_INTERFACE:${HIP_COMMON_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  PUBLIC
  $<INSTALL_INTERFACE:include>)

if(BUILD_SHARED_LIBS)
  if(WIN32)
    target_sources(hiprtc PRIVATE hiprtc.def)
  else()
    target_link_libraries(hiprtc PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/hiprtc.map.in")
    set_target_properties(hiprtc PROPERTIES LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/hiprtc.map.in")
  endif()
endif()

if(WIN32)
  target_link_libraries(hiprtc PRIVATE Dbghelp.lib)
endif()

target_link_libraries(hiprtc PUBLIC ${CMAKE_DL_LIBS})
# Add link to comgr, hsa-runtime and other required libraries in target files
# This is required for static libraries
if(NOT BUILD_SHARED_LIBS)
  find_package(hsa-runtime64)
  find_package(amd_comgr)
  target_link_libraries(hiprtc PRIVATE
    pthread numa rt c amd_comgr hsa-runtime64::hsa-runtime64)
endif()

if(BUILD_SHARED_LIBS)
  target_link_libraries(hiprtc PRIVATE rocclr)
else()
  target_compile_definitions(hiprtc PRIVATE $<TARGET_PROPERTY:rocclr,COMPILE_DEFINITIONS>)
  target_include_directories(hiprtc PRIVATE $<TARGET_PROPERTY:rocclr,INCLUDE_DIRECTORIES>)
endif()

target_compile_definitions(hiprtc PUBLIC __HIP_PLATFORM_AMD__)

add_to_config(_versionInfo HIP_PACKAGING_VERSION_PATCH)
add_to_config(_versionInfo CPACK_DEBIAN_PACKAGE_RELEASE)
add_to_config(_versionInfo CPACK_RPM_PACKAGE_RELEASE)

add_to_config(_versionInfo HIP_VERSION_MAJOR)
add_to_config(_versionInfo HIP_VERSION_MINOR)
add_to_config(_versionInfo HIP_VERSION_PATCH)
add_to_config(_versionInfo HIP_VERSION_GITHASH)

# Enable preprocessed hiprtc-builtins library
include(HIPRTC RESULT_VARIABLE HIPRTC_CMAKE)
# Set binary roots to <PackageName>_ROOT cmake and environment variables similar to CMP0074
# to allow find_program to use LLVM_ROOT and Clang_ROOT
if(DEFINED ENV{LLVM_ROOT})
    set(ENV_LLVM_BIN $ENV{LLVM_ROOT}/bin)
endif()
if(DEFINED LLVM_ROOT)
    set(LLVM_BIN ${LLVM_ROOT}/bin)
endif()

if(DEFINED ENV{Clang_ROOT})
    set(ENV_Clang_BIN $ENV{Clang_ROOT}/bin)
endif()

if(DEFINED Clang_ROOT)
    set(Clang_BIN ${Clang_ROOT}/bin)
endif()
# Requires clang and llvm-mc to create this library.
find_program(clang clang HINTS ${Clang_BIN} ${ENV_Clang_BIN} ${ROCM_PATH}/llvm/bin ${ROCM_PATH})
find_program(llvm-mc llvm-mc HINTS ${LLVM_BIN} ${ENV_LLVM_BIN} ${ROCM_PATH}/llvm/bin ${ROCM_PATH})
set(HIPRTC_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/hip_rtc_gen")
set(HIPRTC_GEN_HEADER "${HIPRTC_GEN_DIR}/hipRTC_header.h")
set(HIPRTC_GEN_MCIN "${HIPRTC_GEN_DIR}/hipRTC_header.mcin")
set(HIPRTC_GEN_PREPROCESSED "${HIPRTC_GEN_DIR}/hipRTC")
set(HIPRTC_GEN_OBJ "${HIPRTC_GEN_DIR}/hipRTC_header${CMAKE_CXX_OUTPUT_EXTENSION}")
# list of headers which needs to be appended to the hiprtc preprocessed file
set(HIPRTC_HEADERS
${HIP_COMMON_INCLUDE_DIR}/hip/hip_common.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/device_library_decls.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/hip_assert.h
${HIP_COMMON_INCLUDE_DIR}/hip/library_types.h
${HIP_COMMON_INCLUDE_DIR}/hip/driver_types.h
${HIP_COMMON_INCLUDE_DIR}/hip/surface_types.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/amd_channel_descriptor.h
${HIP_COMMON_INCLUDE_DIR}/hip/texture_types.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/ockl_image.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/texture_fetch_functions.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/texture_indirect_functions.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/amd_surface_functions.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/amd_hip_complex.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/amd_hip_math_constants.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/math_fwd.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/amd_warp_functions.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/amd_device_functions.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/amd_warp_sync_functions.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/hip_cooperative_groups_helper.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/amd_hip_cooperative_groups.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/amd_hip_unsafe_atomics.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/amd_hip_atomic.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/math_fwd.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/hip_fp16_math_fwd.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/amd_hip_fp16.h
${PROJECT_SOURCE_DIR}/include/hip/amd_detail/amd_math_functions.h
)

# Generate required HIPRTC files.
FILE(MAKE_DIRECTORY ${HIPRTC_GEN_DIR})
generate_hiprtc_header("${HIPRTC_GEN_HEADER}")
generate_hiprtc_mcin("${HIPRTC_GEN_MCIN}" "${HIPRTC_GEN_PREPROCESSED}")

# Generate HIPRTC Builtins Preprocessed Object.
# Note: second command appends define macros at build time.
add_custom_command(
  OUTPUT ${HIPRTC_GEN_PREPROCESSED}
  COMMAND ${clang} -O3 --rocm-path=${PROJECT_SOURCE_DIR}/include/.. -std=c++17 -nogpulib --hip-version=${HIP_LIB_VERSION_MAJOR}.${HIP_LIB_VERSION_MINOR} -isystem ${HIP_COMMON_INCLUDE_DIR} -isystem ${PROJECT_SOURCE_DIR}/include -isystem ${PROJECT_BINARY_DIR}/include -isystem ${CMAKE_CURRENT_SOURCE_DIR}/include --cuda-device-only -D__HIPCC_RTC__ -DHIP_VERSION_MAJOR=${HIP_LIB_VERSION_MAJOR} -DHIP_VERSION_MINOR=${HIP_LIB_VERSION_MINOR} -x hip ${HIPRTC_GEN_HEADER} -E -P -o ${HIPRTC_GEN_PREPROCESSED}
  COMMAND ${CMAKE_COMMAND} -DHIPRTC_ADD_MACROS=1 -DHIPRTC_HEADERS="${HIPRTC_HEADERS}" -DHIPRTC_PREPROCESSED_FILE=${HIPRTC_GEN_PREPROCESSED} -P ${HIPRTC_CMAKE}
  DEPENDS ${clang} ${HIPRTC_GEN_HEADER})
add_custom_command(
  OUTPUT ${HIPRTC_GEN_OBJ}
  COMMAND ${llvm-mc} -o ${HIPRTC_GEN_OBJ} ${HIPRTC_GEN_MCIN} --filetype=obj
  DEPENDS ${llvm-mc} ${HIPRTC_GEN_PREPROCESSED} ${HIPRTC_GEN_MCIN})

# Create hiprtc-builtins library.
add_library(hiprtc-builtins ${HIPRTC_GEN_OBJ})
set_target_properties(hiprtc-builtins PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
  ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
  LINKER_LANGUAGE CXX
  VERSION ${HIP_LIB_VERSION_STRING})

# Windows and Linux have different naming conventions.
if(WIN32)
  # Windows uses DEF file to determine which symbols to expose.
  target_sources(hiprtc-builtins PRIVATE hiprtc-builtins.def)
  set_target_properties(hiprtc-builtins PROPERTIES
    OUTPUT_NAME "hiprtc-builtins${HIP_MAJOR_VERSION}${HIP_MINOR_VERSION}"
    ARCHIVE_OUTPUT_NAME "hiprtc-builtins")
  # Since ${HIPRTC_GEN_OBJ} was manually generated with llvm-mc, /MT did not embed
  # libcmt.lib inside of the obj. So we need to manually set it as defaultlib.
  target_link_options(hiprtc-builtins PRIVATE "LINKER:/DEFAULTLIB:libcmt")
else()
  # SOVERSION is only supported on Linux.
  set_target_properties(hiprtc-builtins PROPERTIES
    OUTPUT_NAME "hiprtc-builtins"
    SOVERSION ${HIP_LIB_VERSION_MAJOR})
endif()

# Test the header file works with simple compilation.
add_custom_command(
  OUTPUT ${HIPRTC_GEN_DIR}/tmp.bc
  COMMAND ${clang} -O3 --rocm-path=${PROJECT_SOURCE_DIR}/include/.. -std=c++14 -nogpulib -nogpuinc -emit-llvm -c -isystem ${HIP_COMMON_INCLUDE_DIR} -isystem ${PROJECT_BINARY_DIR}/include -isystem ${CMAKE_CURRENT_SOURCE_DIR}/include --cuda-device-only -D__HIPCC_RTC__ --offload-arch=gfx906 -x hip-cpp-output ${HIPRTC_GEN_PREPROCESSED} -o ${HIPRTC_GEN_DIR}/tmp.bc
  DEPENDS ${clang} ${HIPRTC_GEN_PREPROCESSED})

target_sources(hiprtc PRIVATE ${HIPRTC_GEN_OBJ})
target_compile_definitions(hiprtc PRIVATE __HIP_ENABLE_RTC)

add_dependencies(hiprtc hiprtc-builtins)
install(TARGETS hiprtc-builtins
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

INSTALL(TARGETS hiprtc
  EXPORT hiprtc-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

INSTALL(EXPORT hiprtc-targets NAMESPACE hiprtc:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hiprtc)

#############################
# hiprtc-config
#############################
include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/hiprtc-config.cmake.in
  ${PROJECT_BINARY_DIR}/hiprtc-config.cmake
  INSTALL_DESTINATION ${CONFIG_RTC_PACKAGE_INSTALL_DIR}
  PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR BIN_INSTALL_DIR)

write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/hiprtc-config-version.cmake
  VERSION "${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}.${HIP_VERSION_PATCH}"
  COMPATIBILITY SameMajorVersion)

INSTALL(FILES ${HIP_COMMON_INCLUDE_DIR}/hip/hiprtc.h DESTINATION "include/hip")

INSTALL(FILES ${HIP_COMMON_INCLUDE_DIR}/hip/hip_common.h DESTINATION "include/hip")

INSTALL(
  FILES ${PROJECT_BINARY_DIR}/hiprtc-config.cmake
        ${PROJECT_BINARY_DIR}/hiprtc-config-version.cmake
  DESTINATION ${CONFIG_RTC_PACKAGE_INSTALL_DIR})
