#
# Integration test applications
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-tests-bin LANGUAGES C CXX)

set(CMAKE_BUILD_RPATH
    "\$ORIGIN:\$ORIGIN/../lib:$<TARGET_FILE_DIR:rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library>"
    )

# applications used by integration tests which DO link to rocprofiler-sdk-roctx
add_subdirectory(reproducible-runtime)
add_subdirectory(transpose)

set(CMAKE_BUILD_RPATH "\$ORIGIN:\$ORIGIN/../lib")

# applications used by integration tests which DO NOT link to rocprofiler-sdk-roctx
add_subdirectory(simple-transpose)
add_subdirectory(multistream)
add_subdirectory(vector-operations)
add_subdirectory(hip-in-libraries)
add_subdirectory(scratch-memory)
add_subdirectory(page-migration)
add_subdirectory(hsa-queue-dependency)
add_subdirectory(hip-graph)
add_subdirectory(hsa-memory-allocation)
