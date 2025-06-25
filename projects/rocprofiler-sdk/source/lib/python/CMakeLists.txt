#
# Python package
#

set(CMAKE_MESSAGE_INDENT "[${PROJECT_NAME}][python] ")
set(DEFAULT_PYTHON_RPATH "\$ORIGIN:\$ORIGIN/../../..:\$ORIGIN/../../../rocprofiler-sdk")

include("${CMAKE_CURRENT_LIST_DIR}/utilities.cmake")

add_subdirectory(roctx)
add_subdirectory(rocpd)
