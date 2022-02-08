# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying file
# Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

# ----------------------------------------------------------------------------------------#

# set(_ROCM_PATHS $ENV{ROCM_HOME} /opt/rocm)

# ----------------------------------------------------------------------------------------#

find_path(
    rocm-smi_ROOT_DIR
    NAMES include/rocm_smi/rocm_smi.h
    HINTS ${_ROCM_PATHS}
    PATHS ${_ROCM_PATHS}
    PATH_SUFFIXES rocm_smi)

mark_as_advanced(rocm-smi_ROOT_DIR)

# ----------------------------------------------------------------------------------------#

find_path(
    rocm-smi_INCLUDE_DIR
    NAMES rocm_smi/rocm_smi.h
    HINTS ${rocm-smi_ROOT_DIR} ${_ROCM_PATHS}
    PATHS ${rocm-smi_ROOT_DIR} ${_ROCM_PATHS}
    PATH_SUFFIXES rocm_smi/include rocm_smi)

mark_as_advanced(rocm-smi_INCLUDE_DIR)

# ----------------------------------------------------------------------------------------#

find_library(
    rocm-smi_LIBRARY
    NAMES rocm_smi64 rocm_smi
    HINTS ${rocm-smi_ROOT_DIR} ${_ROCM_PATHS}
    PATHS ${rocm-smi_ROOT_DIR} ${_ROCM_PATHS}
    PATH_SUFFIXES rocm_smi/lib rocm_smi/lib64 lib lib64)

if(rocm-smi_LIBRARY)
    get_filename_component(rocm-smi_LIBRARY_DIR "${rocm-smi_LIBRARY}" PATH CACHE)
endif()

mark_as_advanced(rocm-smi_LIBRARY)

# ----------------------------------------------------------------------------------------#

find_package_handle_standard_args(rocm-smi DEFAULT_MSG rocm-smi_ROOT_DIR
                                  rocm-smi_INCLUDE_DIR rocm-smi_LIBRARY)

# ------------------------------------------------------------------------------#

if(rocm-smi_FOUND)
    add_library(rocm-smi::rocm-smi INTERFACE IMPORTED)
    add_library(rocm-smi::roctx INTERFACE IMPORTED)
    set(rocm-smi_INCLUDE_DIRS ${rocm-smi_INCLUDE_DIR})
    set(rocm-smi_LIBRARIES ${rocm-smi_LIBRARY})
    set(rocm-smi_LIBRARY_DIRS ${rocm-smi_LIBRARY_DIR})

    target_include_directories(rocm-smi::rocm-smi INTERFACE ${rocm-smi_INCLUDE_DIR})
    target_link_libraries(rocm-smi::rocm-smi INTERFACE ${rocm-smi_LIBRARY})

endif()

# ------------------------------------------------------------------------------#

unset(_ROCM_PATHS)

# ------------------------------------------------------------------------------#
