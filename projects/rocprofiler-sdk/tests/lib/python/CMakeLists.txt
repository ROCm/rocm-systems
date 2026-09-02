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
add_subdirectory(python-hsa)

# python-rccl hard-depends on RCCL at build time (it includes <rccl/rccl.h> and links
# roc::rccl), so it can only be built when RCCL is installed. rccl_FOUND is set by the
# top-level find_package(rccl) in cmake/rocprofiler_config_interfaces.cmake and inherited
# here.
if(rccl_FOUND)
    add_subdirectory(python-rccl)
endif()

# python-rocdecode / python-rocjpeg hard-depend on the codec libs + headers at build time
# (they include the codec headers and link rocdecode::rocdecode / rocjpeg::rocjpeg), so
# build them only when the packages are discoverable AND new enough. rocdecode_FOUND /
# rocjpeg_FOUND and the versions are set by the top-level find_package() in
# cmake/rocprofiler_config_interfaces.cmake and inherited here. The version gates mirror
# tests/bin/CMakeLists.txt (which builds the rocdecode/rocjpeg demo binaries). These are
# optional, source-installable ROCm components.
if(rocdecode_FOUND AND rocdecode_VERSION VERSION_GREATER 0.8.0)
    add_subdirectory(python-rocdecode)
endif()

if(rocjpeg_FOUND AND rocjpeg_VERSION VERSION_GREATER 0.6.0)
    add_subdirectory(python-rocjpeg)
endif()

add_subdirectory(torchesque)
add_subdirectory(tritonesque)
