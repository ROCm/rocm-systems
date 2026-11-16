#
# HIP application for the rocprofv3 range-replay integration tests.
#
# One range of three dispatches. RR_V3_MODE=packet calls begin/end. RR_V3_MODE=host uses
# ROCPROFILER_REPLAY_RANGE_BEGIN / _END. The executable does not link librocprofiler-sdk:
# that link makes the first kernel launch fail on gfx942. rocprofv3 brings the SDK in, and
# the link leaves the API references unresolved so that library satisfies them.
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

project(rocprofiler-sdk-tests-range-replay-rocprofv3-app LANGUAGES CXX HIP)

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

find_package(rocprofiler-sdk REQUIRED COMPONENTS headers)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(range-replay-rocprofv3-app)
target_sources(range-replay-rocprofv3-app PRIVATE main.cpp)
target_link_libraries(range-replay-rocprofv3-app
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers)
target_link_options(range-replay-rocprofv3-app PRIVATE
                    "LINKER:--unresolved-symbols=ignore-in-object-files")
