cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(omnitrace-rewrite-caller-example LANGUAGES CXX)

if(OMNITRACE_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(${PROJECT_NAME} IN_LIST OMNITRACE_DISABLE_EXAMPLES OR ${_DIR} IN_LIST
                                                             OMNITRACE_DISABLE_EXAMPLES)
        return()
    endif()
endif()

set(CMAKE_BUILD_TYPE "Debug")

add_executable(rewrite-caller rewrite-caller.cpp)
target_compile_options(rewrite-caller PRIVATE ${_FLAGS})

if(OMNITRACE_INSTALL_EXAMPLES)
    install(
        TARGETS rewrite-caller
        DESTINATION bin
        COMPONENT omnitrace-examples)
endif()
