#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-bin-hip-in-libraries LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(hip-in-libraries)
target_sources(hip-in-libraries PRIVATE hip-in-libraries.cpp)
target_compile_options(hip-in-libraries PRIVATE -W -Wall -Wextra -Wpedantic -Wshadow
                                                -Werror)
target_link_libraries(hip-in-libraries PRIVATE transpose-shared-library
                                               vector-ops-shared-library)

find_package(hip REQUIRED)
target_link_libraries(hip-in-libraries PRIVATE hip::host)

find_package(Threads REQUIRED)
target_link_libraries(hip-in-libraries PRIVATE Threads::Threads)

if(TRANSPOSE_USE_MPI)
    find_package(MPI REQUIRED)
    target_compile_definitions(hip-in-libraries PRIVATE USE_MPI)
    target_link_libraries(hip-in-libraries PRIVATE MPI::MPI_C)
endif()
