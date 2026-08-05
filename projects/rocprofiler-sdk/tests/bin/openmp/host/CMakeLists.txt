#
#
#

find_package(Threads REQUIRED)

add_executable(openmp-host)
target_sources(openmp-host PRIVATE openmp-host.cpp)
target_link_libraries(openmp-host PRIVATE Threads::Threads)
target_compile_options(openmp-host PRIVATE -fopenmp)
target_link_options(openmp-host PRIVATE -fopenmp)

include(rocprofiler-sdk-custom-compilation)
rocprofiler_sdk_custom_compilation(TARGET openmp-host COMPILER ${OMP_TARGET_COMPILER})
