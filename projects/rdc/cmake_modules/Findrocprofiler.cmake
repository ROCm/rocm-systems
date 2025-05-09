# This module provides a rocprofiler::rocprofiler package
# You can specify the ROCM directory by setting ROCM_DIR

set(NAME rocprofiler)

if(NOT DEFINED ROCM_DIR)
    set(ROCM_DIR "/opt/rocm")
endif()
list(APPEND CMAKE_PREFIX_PATH ${ROCM_DIR})

find_library(
    ${NAME}_LIBRARY
    NAMES ${NAME} ${NAME}64 REQUIRED REGISTRY_VIEW BOTH
    PATH_SUFFIXES lib)

if(NOT DEFINED (${NAME}_INCLUDE_DIR))
    find_path(
        ${NAME}_INCLUDE_DIR
        NAMES ${NAME}.h
        HINTS "${ROCM_DIR}/include"
        PATH_SUFFIXES ${NAME} ${NAME}/inc)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    ${NAME}
    FOUND_VAR ${NAME}_FOUND
    REQUIRED_VARS ${NAME}_LIBRARY ${NAME}_INCLUDE_DIR)

if(${NAME}_FOUND AND NOT TARGET ${NAME}::${NAME})
    add_library(${NAME}::${NAME} UNKNOWN IMPORTED)
    set_target_properties(
        ${NAME}::${NAME}
        PROPERTIES IMPORTED_LOCATION "${${NAME}_LIBRARY}"
                   INTERFACE_COMPILE_OPTIONS "${PC_${NAME}_CFLAGS_OTHER}"
                   INTERFACE_INCLUDE_DIRECTORIES "${${NAME}_INCLUDE_DIR}")
endif()
