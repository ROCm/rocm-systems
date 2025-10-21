cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(rocprofiler-systems-user-api-sample LANGUAGES CXX)

if(ROCPROFSYS_DISABLE_SAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(
        ${PROJECT_NAME} IN_LIST ROCPROFSYS_DISABLE_SAMPLES
        OR ${_DIR} IN_LIST ROCPROFSYS_DISABLE_SAMPLES
    )
        return()
    endif()
endif()

set(CMAKE_BUILD_TYPE "Debug")
find_package(Threads REQUIRED)
if(NOT TARGET rocprofiler-systems::rocprofiler-systems-user-library)
    find_package(rocprofiler-systems REQUIRED COMPONENTS user)
endif()
add_executable(user-api user-api.cpp)
target_link_libraries(
    user-api
    PRIVATE Threads::Threads rocprofiler-systems::rocprofiler-systems-user-library
)

if(ROCPROFSYS_INSTALL_SAMPLES)
    install(TARGETS user-api DESTINATION bin COMPONENT rocprofiler-systems-samples)
endif()
