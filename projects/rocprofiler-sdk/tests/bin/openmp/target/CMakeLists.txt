#
#
#

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

find_package(Threads REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)

add_executable(openmp-target)
target_sources(openmp-target PRIVATE openmp-target.cpp)
target_link_libraries(openmp-target PRIVATE Threads::Threads
                                            rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
target_compile_options(openmp-target PRIVATE -fopenmp)
target_link_options(openmp-target PRIVATE -fopenmp)

foreach(_TARGET ${OPENMP_GPU_TARGETS})
    target_compile_options(openmp-target PRIVATE --offload-arch=${_TARGET})
    target_link_options(openmp-target PRIVATE --offload-arch=${_TARGET})
endforeach()

include(rocprofiler-sdk-custom-compilation)
rocprofiler_sdk_custom_compilation(TARGET openmp-target COMPILER ${OMP_TARGET_COMPILER})
