cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Set up ROCm paths before declaring HIP language
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()

# Set HIP compiler path before project declaration
if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

project(Aqlprofile_v2_tests LANGUAGES C CXX HIP)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED AQLPROFILE_TARGET)
    find_library(
        AQLPROFILE_TARGET
        REQUIRED
        NAMES hsa-amd-aqlprofile64 hsa-amd-aqlprofile
        HINTS /opt/rocm ${CMAKE_INSTALL_PREFIX}
        PATHS /opt/rocm
        PATH_SUFFIXES lib)
    enable_testing()
    include(CTest)
endif()

find_package(
    hsa-runtime64
    REQUIRED
    CONFIG
    HINTS /opt/rocm/
    PATHS /opt/rocm/)

get_property(
    HSA_RUNTIME_INCLUDE_DIRECTORIES
    TARGET hsa-runtime64::hsa-runtime64
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(
    HSA_H hsa.h
    PATHS ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
    PATH_SUFFIXES hsa
    NO_DEFAULT_PATH REQUIRED)
get_filename_component(HSA_RUNTIME_INC_PATH ${HSA_H} DIRECTORY)
include_directories(${HSA_RUNTIME_INC_PATH})

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O1" )

add_library(intercept SHARED)
target_include_directories(intercept PRIVATE ${HSA_RUNTIME_INC_PATH} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/core/include/  /opt/rocm/include)
target_sources(intercept PRIVATE intercept.cpp)
target_link_libraries(intercept PRIVATE hsa-runtime64::hsa-runtime64 ${AQLPROFILE_TARGET})
target_link_options(intercept PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exportmap -Wl,--no-undefined)
target_compile_definitions(intercept PUBLIC AMD_INTERNAL_BUILD)
add_executable(testv2)
target_sources(testv2 PRIVATE main.cpp workload.cpp counter.cpp agent.cpp)
target_include_directories(testv2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/core/include/ ${HSA_RUNTIME_INC_PATH} /opt/rocm/include)
target_link_libraries(testv2 PRIVATE hsa-runtime64::hsa-runtime64 ${AQLPROFILE_TARGET})
target_compile_definitions(testv2 PUBLIC AMD_INTERNAL_BUILD)

# Add a PRELOAD environment with libintercept
set(ENV{LD_PRELOAD} "$ENV{LD_PRELOAD}:${CMAKE_CURRENT_BINARY_DIR}/libintercept.so")

add_test(NAME testv2 COMMAND testv2)
set_tests_properties(testv2 PROPERTIES ENVIRONMENT "${LD_PRELOAD}" TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION "${AQLPROFILE_DEFAULT_FAIL_REGEX}")
