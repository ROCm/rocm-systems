#
# rocpd API test: exercises rocpd_get_version and rocpd_sql_load_schema
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-bin-rocpd-api-test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(rocprofiler-sdk-rocpd REQUIRED)

add_executable(rocpd-api-test)
target_sources(rocpd-api-test PRIVATE rocpd-api-test.cpp)
target_compile_options(rocpd-api-test PRIVATE -W -Wall -Wextra)
target_link_libraries(rocpd-api-test
                      PRIVATE rocprofiler-sdk-rocpd::rocprofiler-sdk-rocpd-shared-library)
