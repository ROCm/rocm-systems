#
# Python test libraries and packages
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-lib-python LANGUAGES C CXX)

set(ROCPROFILER_TESTS_PYTHON_SITE_PACKAGES_DIR
    "${CMAKE_BINARY_DIR}/lib/python3/site-packages"
    CACHE PATH "Destination for Python test packages")

function(rocprofiler_tests_configure_python_package_sources RELATIVE_DEST)
    foreach(_FILE ${ARGN})
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/${_FILE}
            ${ROCPROFILER_TESTS_PYTHON_SITE_PACKAGES_DIR}/${RELATIVE_DEST}/${_FILE}
            COPYONLY)
    endforeach()
endfunction()

add_subdirectory(python-hip-kernels)
add_subdirectory(torchesque)
add_subdirectory(tritonesque)
