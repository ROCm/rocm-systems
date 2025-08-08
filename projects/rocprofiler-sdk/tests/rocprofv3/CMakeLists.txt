#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-rocprofv3-tests LANGUAGES C CXX)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "tests")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        "Release"
        CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

enable_testing()
include(CTest)

add_subdirectory(tracing)
add_subdirectory(tracing-plus-counter-collection)
add_subdirectory(tracing-hip-in-libraries)
add_subdirectory(counter-collection)
add_subdirectory(hsa-queue-dependency)
add_subdirectory(kernel-rename)
add_subdirectory(memory-allocation)
add_subdirectory(aborted-app)
add_subdirectory(summary)
add_subdirectory(roctracer-roctx)
add_subdirectory(scratch-memory)
add_subdirectory(pc-sampling)
add_subdirectory(collection-period)
add_subdirectory(internal-threading)
add_subdirectory(rocdecode-trace)
add_subdirectory(rocjpeg-trace)
add_subdirectory(advanced-thread-trace)
add_subdirectory(hip-stream-display)
add_subdirectory(agent-index)
add_subdirectory(negate-aggregate-tracing-options)
add_subdirectory(minimum-bytes)
add_subdirectory(conversion-script)
add_subdirectory(python-bindings)
add_subdirectory(rocpd)
