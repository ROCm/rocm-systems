#
# Python package
#

set(DEFAULT_PYTHON_RPATH "\$ORIGIN:\$ORIGIN/../../..:\$ORIGIN/../../../rocprofiler-sdk")

include("${CMAKE_CURRENT_LIST_DIR}/utilities.cmake")

if(NOT DEFINED ROCPROFILER_PYTHON_VERSIONS)
    get_default_python_version(DEFAULT_PYTHON_VERSION)
    set(ROCPROFILER_PYTHON_VERSIONS
        "${DEFAULT_PYTHON_VERSION}"
        CACHE STRING "")
endif()

add_subdirectory(roctx)
