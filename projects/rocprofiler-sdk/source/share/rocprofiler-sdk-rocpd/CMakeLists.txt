#
#
#

set(rocpd_schemas data_views.sql rocpd_indexes.sql rocpd_metadata.sql rocpd_tables.sql
                  rocpd_views.sql summary_views.sql versions.yml)

function(rocpd_configure_schema_files)
    cmake_parse_arguments(_SCHEMA "" "VERSION" "FILES" ${ARGN})
    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME _SCHEMA_DIRNAME)
    if(_SCHEMA_VERSION)
        if(NOT _SCHEMA_DIRNAME STREQUAL _SCHEMA_VERSION)
            message(
                FATAL_ERROR
                    "rocpd_configure_schema_files for version ${_SCHEMA_VERSION} not located in subdirectory matching schema version (found: ${_SCHEMA_DIRNAME})"
                )
        endif()

        set(_BINARY_DEST
            "${PROJECT_BINARY_DIR}/share/rocprofiler-sdk-rocpd/versions/${_SCHEMA_VERSION}"
            )
        set(_INSTALL_DEST "share/rocprofiler-sdk-rocpd/versions/${_SCHEMA_VERSION}")
    else()
        if(NOT _SCHEMA_DIRNAME STREQUAL "rocprofiler-sdk-rocpd")
            message(
                FATAL_ERROR
                    "rocpd_configure_schema_files for latest version not located in 'rocprofiler-sdk-rocpd' subdirectory"
                )
        endif()

        set(_BINARY_DEST "${PROJECT_BINARY_DIR}/share/rocprofiler-sdk-rocpd")
        set(_INSTALL_DEST "share/rocprofiler-sdk-rocpd")
    endif()
    foreach(_FILE ${_SCHEMA_FILES})
        configure_file(${_FILE} ${_BINARY_DEST}/${_FILE} COPYONLY)
        install(
            FILES ${_BINARY_DEST}/${_FILE}
            DESTINATION ${_INSTALL_DEST}
            COMPONENT rocpd)
    endforeach()
endfunction()

rocpd_configure_schema_files(FILES ${rocpd_schemas})

# version compatibility
add_subdirectory(versions)
