cmake_minimum_required(VERSION 3.16.8)
project(SampleApp)

message(STATUS "ROCM_PATH: ${ROCM_PATH}")

if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm")
endif()

if(NOT DEFINED HIP_PATH)
    if(DEFINED ROCM_PATH)
        set(HIP_PATH ${ROCM_PATH})
    else()
        set(HIP_PATH "/opt/rocm")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --rocm-path=${ROCM_PATH}")

set(HIPCC_EXEC "hipcc")

set(CMAKE_C_COMPILER   "${HIP_PATH}/bin/${HIPCC_EXEC}")
set(CMAKE_CXX_COMPILER "${HIP_PATH}/bin/${HIPCC_EXEC}")

add_library(CommonLib Common.cpp)
target_include_directories(CommonLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(MultiplyKernel.code COMMAND ${CMAKE_CXX_COMPILER} --genco ${CMAKE_CURRENT_SOURCE_DIR}/MultiplyKernel.cpp -o MultiplyKernel.code)

add_library(LibMul Multiplication.cpp)
target_include_directories(LibMul PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(LibMul PUBLIC CommonLib)

add_dependencies(LibMul MultiplyKernel.code)

add_library(LibAdd Addition.cpp)
target_include_directories(LibAdd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(LibAdd PUBLIC CommonLib)

add_executable(app Application.cpp)
target_link_libraries(app PRIVATE LibMul LibAdd)



