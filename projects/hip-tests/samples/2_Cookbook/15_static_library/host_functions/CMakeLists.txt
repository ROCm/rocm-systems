project(static_lib)

cmake_minimum_required(VERSION 3.10)

if(UNIX)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCM Path")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    endif()
  endif()
  # Search for rocm in common locations
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

# Find hip
find_package(hip REQUIRED)

# For windows, AR is MS Librarian and that is pickedby Visual Studio's command prompt.
if (WIN32)
  find_program(libpath NAMES lib.exe)
  set (CMAKE_AR ${libpath})
endif()

# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_BUILD_TYPE Release)

# Turn static library generation ON
option(BUILD_SHARED_LIBS "Build as a shared library" OFF)

set(CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/hipOptLibrary.cpp)

# For windows, We need to tell cmake how to create static library.
if (WIN32)
  set (CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> /out:<TARGET> <LINK_FLAGS> <OBJECTS>")
endif()

if(TARGET build_cookbook)
  set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
  set(EXCLUDE_OPTION )
endif()

set(TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/hipMain1.cpp)

# Generate static lib libHipOptLibrary.a.
add_library(HipOptLibrary ${EXCLUDE_OPTION} STATIC ${CPP_SOURCES})

# Set-up the correct flags to generate the static library.
target_link_options(HipOptLibrary PRIVATE --emit-static-lib)
target_include_directories(HipOptLibrary PRIVATE /opt/rocm/hsa/include)

# Create test executable that uses libHipOptLibrary.a
add_executable(test_opt_static ${EXCLUDE_OPTION} ${TEST_SOURCES})
add_dependencies(test_opt_static HipOptLibrary)
target_link_libraries(test_opt_static PRIVATE -lHipOptLibrary -L${CMAKE_CURRENT_BINARY_DIR})

if (WIN32)
  target_link_libraries(test_opt_static PRIVATE amdhip64 amd_comgr)
else()
  target_link_libraries(test_opt_static PRIVATE amdhip64 amd_comgr hsa-runtime64::hsa-runtime64)
endif()

if(TARGET build_cookbook)
add_dependencies(build_cookbook HipOptLibrary test_opt_static)
endif()