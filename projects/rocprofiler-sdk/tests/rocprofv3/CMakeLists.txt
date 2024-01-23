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

# needed for validation
find_package(Python3 REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pandas"
    RESULT_VARIABLE PYTHON_PANDAS_IMPORT
    OUTPUT_QUIET)

if(NOT ${PYTHON_PANDAS_IMPORT} EQUAL 0)
    message(
        FATAL_ERROR
            "The 'pandas' Python3 package is not installed. Please install it using the following command: '${Python3_EXECUTABLE} -m pip install pandas'"
        )
endif()

add_subdirectory(tracing)
add_subdirectory(counter-collection)
add_subdirectory(tracing-plus-cc)
