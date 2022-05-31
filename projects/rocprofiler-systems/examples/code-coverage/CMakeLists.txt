cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(omnitrace-code-coverage LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")

find_package(Threads REQUIRED)
add_executable(code-coverage code-coverage.cpp)
target_link_libraries(code-coverage PRIVATE Threads::Threads)
target_compile_options(code-coverage PRIVATE ${_FLAGS})

if(OMNITRACE_INSTALL_EXAMPLES)
    install(
        TARGETS code-coverage
        DESTINATION bin
        COMPONENT omnitrace-examples)
endif()
