#
# Integration test application libraries
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-lib LANGUAGES C CXX)

set(CMAKE_BUILD_RPATH "\$ORIGIN:\$ORIGIN/../lib")

# libraries used by integration test apps which DO NOT link to rocprofiler-sdk-roctx
add_subdirectory(vector-operations)

# Only add the roctx shared-library directory to the build RPATH when the target exists
# (it may be absent when configuring tests against an installed SDK).
if(TARGET rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library)
    set(CMAKE_BUILD_RPATH
        "\$ORIGIN:\$ORIGIN/../lib:$<TARGET_FILE_DIR:rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library>"
        )
else()
    set(CMAKE_BUILD_RPATH "\$ORIGIN:\$ORIGIN/../lib")
endif()

# libraries used by integration test apps which DO link to rocprofiler-sdk-roctx
add_subdirectory(transpose)
