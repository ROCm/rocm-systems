#
# Platform-abstraction layer for rocprofiler-register.
#
# loader.hpp and pe_parser.hpp define a thin platform-neutral interface.
# Each subdirectory contributes a self-contained implementation owned by
# that subdir's CMakeLists.txt. Layout mirrors rocprofiler-sdk PR #6399.
#

if(WIN32)
    add_subdirectory(windows)
else()
    add_subdirectory(gnulinux)
endif()
