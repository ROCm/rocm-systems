cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(omnitrace-user-api LANGUAGES CXX)

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
