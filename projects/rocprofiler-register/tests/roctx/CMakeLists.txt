#
#
#

if(NOT TARGET rocprofiler-register::rocprofiler-register)
    find_package(rocprofiler-register REQUIRED)
endif()

add_library(roctx SHARED)
add_library(roctx::roctx ALIAS roctx)
target_sources(roctx PRIVATE roctx.cpp roctx.hpp)
target_include_directories(roctx PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(roctx PRIVATE rocprofiler-register::rocprofiler-register)
set_target_properties(roctx PROPERTIES OUTPUT_NAME roctx64)
# WINDOWS-DIVERGENCE: see comments in tests/hsa-runtime and tests/amdhip
# CMakeLists.txt -- versioned PE filenames break base-name LoadLibrary.
if(NOT WIN32)
    set_target_properties(roctx PROPERTIES SOVERSION 4 VERSION 4.6.1)
endif()
rocp_register_strip_target(roctx)
