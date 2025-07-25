cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

project(rocprofiler-systems-fork LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")

find_package(Threads REQUIRED)
find_package(rocprofiler-systems REQUIRED COMPONENTS user)
add_executable(fork-example fork.cpp)
target_link_libraries(
    fork-example
    PRIVATE Threads::Threads rocprofiler-systems::rocprofiler-systems
)
target_compile_options(fork-example PRIVATE ${_FLAGS})

if(ROCPROFSYS_INSTALL_EXAMPLES)
    install(TARGETS fork-example DESTINATION bin COMPONENT rocprofiler-systems-examples)
endif()
