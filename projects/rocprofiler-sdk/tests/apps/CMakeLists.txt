#
# Integration test applications
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-test-apps LANGUAGES C CXX)

set(CMAKE_BUILD_RPATH "\$ORIGIN:\$ORIGIN/../lib")

# applications used by integration tests which DO NOT link to rocprofiler-sdk-roctx
add_subdirectory(simple-transpose)

set(CMAKE_BUILD_RPATH
    "\$ORIGIN:\$ORIGIN/../lib:$<TARGET_FILE_DIR:rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library>"
    )

# applications used by integration tests which DO link to rocprofiler-sdk-roctx
add_subdirectory(reproducible-runtime)
add_subdirectory(transpose)
