#
# Multi-threaded code object loading test
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

project(rocprofiler-sdk-tests-bin-code-object-multi-threaded LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

find_package(rocprofiler-sdk REQUIRED)

# Find hiprtc for runtime compilation
find_library(
    HIPRTC_LIBRARY
    NAMES hiprtc
    HINTS ${ROCM_PATH}/lib ENV ROCM_PATH /opt/rocm/lib)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

add_executable(code-object-multithread-testapp)
target_sources(code-object-multithread-testapp PRIVATE main.cpp)
target_compile_options(code-object-multithread-testapp PRIVATE -W -Wall -Wextra)

find_package(Threads REQUIRED)
target_link_libraries(code-object-multithread-testapp PRIVATE Threads::Threads)

# Link hiprtc if found, otherwise test will fall back to static kernels
if(HIPRTC_LIBRARY)
    target_link_libraries(code-object-multithread-testapp PRIVATE ${HIPRTC_LIBRARY})
endif()
