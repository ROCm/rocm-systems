# include guard
include_guard(DIRECTORY)

# ########################################################################################
#
# External Packages are found here
#
# ########################################################################################

target_include_directories(
    rocprofiler-register-headers
    INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source/include>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/include>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# ensure the env overrides the appending /opt/rocm later
string(REPLACE ":" ";" CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH};${CMAKE_PREFIX_PATH}")

# ----------------------------------------------------------------------------------------#
#
# Threading
#
# ----------------------------------------------------------------------------------------#

set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG OFF)

find_package(Threads ${rocprofiler_register_FIND_QUIETLY}
             ${rocprofiler_register_FIND_REQUIREMENT})
if(Threads_FOUND)
    target_link_libraries(rocprofiler-register-threading
                          INTERFACE $<BUILD_INTERFACE:Threads::Threads>)
endif()

# ----------------------------------------------------------------------------------------#
#
# dynamic linking (dl) and runtime (rt) libraries
#
# ----------------------------------------------------------------------------------------#

foreach(_LIB dl rt)
    find_library(${_LIB}_LIBRARY NAMES ${_LIB})
    find_package_handle_standard_args(${_LIB}-library REQUIRED_VARS ${_LIB}_LIBRARY)
    if(${_LIB}_LIBRARY)
        target_link_libraries(
            rocprofiler-register-${_LIB} INTERFACE $<BUILD_INTERFACE:${${_LIB}_LIBRARY}>
                                                   $<INSTALL_INTERFACE:${_LIB}>)
    endif()
endforeach()

# ----------------------------------------------------------------------------------------#
#
# stdc++fs (filesystem) library
#
# ----------------------------------------------------------------------------------------#

find_library(stdcxxfs_LIBRARY NAMES stdc++fs)
find_package_handle_standard_args(stdcxxfs-library REQUIRED_VARS stdcxxfs_LIBRARY)

if(stdcxxfs_LIBRARY)
    target_link_libraries(rocprofiler-register-stdcxxfs
                          INTERFACE $<BUILD_INTERFACE:${stdcxxfs_LIBRARY}>)
else()
    target_link_libraries(rocprofiler-register-stdcxxfs
                          INTERFACE $<BUILD_INTERFACE:stdc++fs>)
endif()
