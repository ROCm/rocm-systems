#
# Python package - Pure Python Implementation
# Version-agnostic packages that work across all Python 3.6+ versions
#

set(CMAKE_MESSAGE_INDENT "[${PROJECT_NAME}][python] ")

message(STATUS "")
message(STATUS "===========================================================================")
message(STATUS "Pure Python Build Configuration")
message(STATUS "===========================================================================")
message(STATUS "Building version-agnostic Python packages using pure Python")
message(STATUS "No compilation required - works with all Python 3.6+ versions")
message(STATUS "===========================================================================")
message(STATUS "")

# Pure Python packages (no version-specific builds needed!)
add_subdirectory(roctx)      # roctx: Pure Python using ctypes
add_subdirectory(rocpd)      # rocpd: Pure Python implementation
add_subdirectory(rocprofv3)  # rocprofv3: Pure Python using ctypes
