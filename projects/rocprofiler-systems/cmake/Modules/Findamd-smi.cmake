# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying file
# Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

# ----------------------------------------------------------------------------------------#

if(NOT ROCM_PATH AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "$ENV{ROCM_PATH}")
endif()

foreach(_DIR ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm /opt/rocm/amd_smi)
    if(EXISTS ${_DIR})
        get_filename_component(_ABS_DIR "${_DIR}" REALPATH)
        list(APPEND _AMD_SMI_PATHS ${_ABS_DIR})
    endif()
endforeach()

# ----------------------------------------------------------------------------------------#

find_path(
    amd-smi_ROOT_DIR
    NAMES include/amd_smi/amdsmi.h
    HINTS ${_AMD_SMI_PATHS}
    PATHS ${_AMD_SMI_PATHS}
    PATH_SUFFIXES amd_smi
)

mark_as_advanced(amd-smi_ROOT_DIR)

# ----------------------------------------------------------------------------------------#

find_path(
    amd-smi_INCLUDE_DIR
    NAMES amd_smi/amdsmi.h
    HINTS ${amd-smi_ROOT_DIR} ${_AMD_SMI_PATHS}
    PATHS ${amd-smi_ROOT_DIR} ${_AMD_SMI_PATHS}
    PATH_SUFFIXES include amd_smi/include
)

mark_as_advanced(amd-smi_INCLUDE_DIR)

# ----------------------------------------------------------------------------------------#

find_library(
    amd-smi_LIBRARY
    NAMES amd_smi
    HINTS ${amd-smi_ROOT_DIR} ${_AMD_SMI_PATHS}
    PATHS ${amd-smi_ROOT_DIR} ${_AMD_SMI_PATHS}
    PATH_SUFFIXES amd-smi/lib lib
)

if(amd-smi_LIBRARY)
    get_filename_component(amd-smi_LIBRARY_DIR "${amd-smi_LIBRARY}" PATH CACHE)
endif()

mark_as_advanced(amd-smi_LIBRARY)

# ----------------------------------------------------------------------------------------#

find_package_handle_standard_args(
    amd-smi
    DEFAULT_MSG
    amd-smi_ROOT_DIR
    amd-smi_INCLUDE_DIR
    amd-smi_LIBRARY
)

# ------------------------------------------------------------------------------#

if(amd-smi_FOUND)
    add_library(amd-smi::amd-smi INTERFACE IMPORTED)
    add_library(amd-smi::roctx INTERFACE IMPORTED)
    set(amd-smi_INCLUDE_DIRS ${amd-smi_INCLUDE_DIR})
    set(amd-smi_LIBRARIES ${amd-smi_LIBRARY})
    set(amd-smi_LIBRARY_DIRS ${amd-smi_LIBRARY_DIR})

    target_include_directories(amd-smi::amd-smi INTERFACE ${amd-smi_INCLUDE_DIR})
    target_link_libraries(amd-smi::amd-smi INTERFACE ${amd-smi_LIBRARY})
endif()

# ------------------------------------------------------------------------------#

unset(_AMD_SMI_PATHS)

# ------------------------------------------------------------------------------#
