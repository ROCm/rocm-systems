cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(rocprofiler-systems-fork LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
list(APPEND _FLAGS -fno-inline)

find_package(Threads REQUIRED)
find_package(rocprofiler-systems REQUIRED COMPONENTS user)
add_executable(fork-sample fork.cpp)
target_link_libraries(
    fork-sample
    PRIVATE Threads::Threads rocprofiler-systems::rocprofiler-systems
)
target_compile_options(fork-sample PRIVATE ${_FLAGS})

if(ROCPROFSYS_INSTALL_SAMPLES)
    install(TARGETS fork-sample DESTINATION bin COMPONENT rocprofiler-systems-samples)
endif()
