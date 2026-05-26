#
#
#

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

find_package(Threads REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)

include(rocprofiler-sdk-custom-compilation)

function(rocprofiler_add_openmp_target_executable _NAME)
    add_executable(${_NAME})
    target_sources(${_NAME} PRIVATE ${_NAME}.cpp)
    target_link_libraries(${_NAME} PRIVATE Threads::Threads
                                            rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
    target_compile_options(${_NAME} PRIVATE -fopenmp)
    target_link_options(${_NAME} PRIVATE -fopenmp)

    foreach(_TARGET ${OPENMP_GPU_TARGETS})
        target_compile_options(${_NAME} PRIVATE --offload-arch=${_TARGET})
        target_link_options(${_NAME} PRIVATE --offload-arch=${_TARGET})
    endforeach()

    rocprofiler_sdk_custom_compilation(TARGET ${_NAME} COMPILER ${OMP_TARGET_COMPILER})
endfunction()

rocprofiler_add_openmp_target_executable(openmp-target)
rocprofiler_add_openmp_target_executable(openmp-attach)
