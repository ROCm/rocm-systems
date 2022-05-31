cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(omnitrace-parallel-overhead LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release")
find_package(Threads REQUIRED)
add_executable(parallel-overhead parallel-overhead.cpp)
target_link_libraries(parallel-overhead PRIVATE Threads::Threads)

add_executable(parallel-overhead-locks parallel-overhead.cpp)
target_link_libraries(parallel-overhead-locks PRIVATE Threads::Threads)
target_compile_definitions(parallel-overhead-locks PRIVATE USE_LOCKS=1)

if(OMNITRACE_INSTALL_EXAMPLES)
    install(
        TARGETS parallel-overhead parallel-overhead-locks
        DESTINATION bin
        COMPONENT omnitrace-examples)
endif()
