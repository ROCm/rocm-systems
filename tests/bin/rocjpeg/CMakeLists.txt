#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

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

project(rocprofiler-tool-test-app-rocjpeg LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_path(
    ROCJPEG_SHARE_DIR
    NAMES images
    PATHS ${ROCM_PATH}/share/rocjpeg/)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set_source_files_properties(rocjpeg.cpp PROPERTIES LANGUAGE HIP)
add_executable(rocjpeg-demo)
target_sources(rocjpeg-demo PRIVATE rocjpeg.cpp)

find_package(Threads REQUIRED)
find_package(rocJPEG REQUIRED)
target_link_libraries(
    rocjpeg-demo
    PRIVATE Threads::Threads hsa-runtime64 rocprofiler-sdk::tests-common-library
            rocprofiler-sdk::tests-build-flags rocJPEG::rocJPEG)
