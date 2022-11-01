cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(omnitrace-user-api-example LANGUAGES CXX)

if(OMNITRACE_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(${PROJECT_NAME} IN_LIST OMNITRACE_DISABLE_EXAMPLES OR ${_DIR} IN_LIST
                                                             OMNITRACE_DISABLE_EXAMPLES)
        return()
    endif()
endif()

set(CMAKE_BUILD_TYPE "Debug")
find_package(Threads REQUIRED)
if(NOT TARGET omnitrace::omnitrace-user-library)
    find_package(omnitrace REQUIRED COMPONENTS user)
endif()
add_executable(user-api user-api.cpp)
target_link_libraries(user-api PRIVATE Threads::Threads omnitrace::omnitrace-user-library)

if(OMNITRACE_INSTALL_EXAMPLES)
    install(
        TARGETS user-api
        DESTINATION bin
        COMPONENT omnitrace-examples)
endif()
