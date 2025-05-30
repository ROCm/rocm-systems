#
#  functions/macros for python
#

include_guard(DIRECTORY)

macro(rocprofiler_reset_python3_cache)
    foreach(
        _VAR
        _Python3_Compiler_REASON_FAILURE
        _Python3_Development_REASON_FAILURE
        _Python3_EXECUTABLE
        _Python3_INCLUDE_DIR
        _Python3_INTERPRETER_PROPERTIES
        _Python3_INTERPRETER_SIGNATURE
        _Python3_LIBRARY_RELEASE
        _Python3_NumPy_REASON_FAILURE
        Python3_EXECUTABLE
        Python3_INCLUDE_DIR
        Python3_INTERPRETER_ID
        Python3_STDLIB
        Python3_STDARCH
        Python3_SITELIB
        Python3_SOABI
        ${ARGN})
        unset(${_VAR} CACHE)
        unset(${_VAR})
    endforeach()
endmacro()

macro(rocprofiler_find_python3 _VERSION)
    rocprofiler_reset_python3_cache()

    if("${_VERSION}" MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
        find_package(Python3 ${_VERSION} EXACT ${_ARGN} REQUIRED MODULE
                     COMPONENTS Interpreter Development)
    elseif("${_VERSION}" MATCHES "^([0-9]+)\\.([0-9]+)$")
        find_package(Python3 ${_VERSION}.0...${_VERSION}.999 ${_ARGN} REQUIRED MODULE
                     COMPONENTS Interpreter Development)
    else()
        message(
            FATAL_ERROR
                "Invalid Python3 version (${_VERSION}). Specify <MAJOR>.<MINOR> or <MAJOR>.<MINOR>.<PATCH>"
            )
    endif()
endmacro()

# make sure we have all python version candidates
set(ROCPROFILER_PYTHON_VERSION_CANDIDATES
    "3.20;3.19;3.18;3.17;3.16;3.15;3.14;3.13;3.12;3.11;3.10;3.9;3.8;3.7;3.6"
    CACHE STRING "Python versions to search for, newest first")

function(get_default_python_versions _VAR)
    rocprofiler_reset_python3_cache()

    set(_PYTHON_FOUND_VERSIONS)

    foreach(_VER IN LISTS ROCPROFILER_PYTHON_VERSION_CANDIDATES)
        find_package(Python3 ${_VER} EXACT COMPONENTS Interpreter Development)
        if(Python3_FOUND)
            list(APPEND _PYTHON_FOUND_VERSIONS
                 "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
        endif()
    endforeach()

    # If none found, do one last check for 3.6 (no EXACT)
    if(NOT _PYTHON_FOUND_VERSIONS)
        find_package(Python3 3.6 COMPONENTS Interpreter Development)
        if(Python3_FOUND)
            list(APPEND _PYTHON_FOUND_VERSIONS
                 "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
        endif()
    endif()

    # Set the output variable to the first found version, if any
    if(_PYTHON_FOUND_VERSIONS)
        set(${_VAR}
            "${_PYTHON_FOUND_VERSIONS}"
            PARENT_SCOPE)
    endif()

    rocprofiler_reset_python3_cache()
endfunction()

function(rocprofiler_roctx_python_bindings _VERSION)
    message(
        STATUS "Creating rocprofiler-sdk roctx python bindings for python ${_VERSION}")

    rocprofiler_find_python3(${_VERSION})

    set(roctx_PYTHON_INSTALL_DIRECTORY
        ${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/roctx
        )
    set(roctx_PYTHON_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/${roctx_PYTHON_INSTALL_DIRECTORY})
    set(roctx_PYTHON_SOURCES __init__.py context_decorators.py)

    foreach(_SOURCE ${roctx_PYTHON_SOURCES})
        configure_file(${CMAKE_CURRENT_LIST_DIR}/${_SOURCE}
                       ${roctx_PYTHON_OUTPUT_DIRECTORY}/${_SOURCE} COPYONLY)
        install(
            FILES ${roctx_PYTHON_OUTPUT_DIRECTORY}/${_SOURCE}
            DESTINATION ${roctx_PYTHON_INSTALL_DIRECTORY}
            COMPONENT roctx)
    endforeach()

    add_library(rocprofiler-sdk-roctx-python-bindings-${_VERSION} MODULE)
    target_sources(rocprofiler-sdk-roctx-python-bindings-${_VERSION}
                   PRIVATE libpyroctx.cpp)
    target_include_directories(rocprofiler-sdk-roctx-python-bindings-${_VERSION} SYSTEM
                               PRIVATE ${Python3_INCLUDE_DIRS})
    target_link_libraries(
        rocprofiler-sdk-roctx-python-bindings-${_VERSION}
        PRIVATE rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library
                rocprofiler-sdk::rocprofiler-sdk-pybind11 ${Python3_LIBRARIES})

    set_target_properties(
        rocprofiler-sdk-roctx-python-bindings-${_VERSION}
        PROPERTIES OUTPUT_NAME libpyroctx
                   RUNTIME_OUTPUT_DIRECTORY ${roctx_PYTHON_OUTPUT_DIRECTORY}
                   LIBRARY_OUTPUT_DIRECTORY ${roctx_PYTHON_OUTPUT_DIRECTORY}
                   ARCHIVE_OUTPUT_DIRECTORY ${roctx_PYTHON_OUTPUT_DIRECTORY}
                   PDB_OUTPUT_DIRECTORY ${roctx_PYTHON_OUTPUT_DIRECTORY}
                   PREFIX ""
                   SUFFIX ".${Python3_SOABI}${CMAKE_SHARED_LIBRARY_SUFFIX}"
                   BUILD_RPATH "${DEFAULT_PYTHON_RPATH}"
                   INSTALL_RPATH "${DEFAULT_PYTHON_RPATH}")

    install(
        TARGETS rocprofiler-sdk-roctx-python-bindings-${_VERSION}
        DESTINATION ${roctx_PYTHON_INSTALL_DIRECTORY}
        COMPONENT roctx)
endfunction()

function(rocprofiler_rocpd_python_bindings_target_sources _VERSION)
    target_sources(rocprofiler-sdk-rocpd-python-bindings-${_VERSION} ${ARGN})
endfunction()

function(rocprofiler_rocpd_python_bindings _VERSION)
    message(
        STATUS "Creating rocprofiler-sdk rocpd python bindings for python ${_VERSION}")
    rocprofiler_find_python3(${_VERSION})

    set(rocpd_PYTHON_INSTALL_DIRECTORY
        ${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/rocpd
        )
    set(rocpd_PYTHON_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/${rocpd_PYTHON_INSTALL_DIRECTORY})
    set(rocpd_PYTHON_SOURCES
        chrome_tracing.py
        csv.py
        importer.py
        __init__.py
        __main__.py
        output_config.py
        pftrace.py
        schema.py
        time_window.py)
    set(rocpd_SCHEMA_SOURCES
        schema_data/data_views.sql schema_data/marker_views.sql
        schema_data/rocpd_indexes.sql schema_data/rocpd_tables.sql
        schema_data/rocpd_views.sql schema_data/summary_views.sql)

    foreach(_SOURCE ${rocpd_PYTHON_SOURCES})
        configure_file(${CMAKE_CURRENT_LIST_DIR}/${_SOURCE}
                       ${rocpd_PYTHON_OUTPUT_DIRECTORY}/${_SOURCE} COPYONLY)
        install(
            FILES ${rocpd_PYTHON_OUTPUT_DIRECTORY}/${_SOURCE}
            DESTINATION ${rocpd_PYTHON_INSTALL_DIRECTORY}
            COMPONENT core)
    endforeach()

    foreach(_SOURCE ${rocpd_SCHEMA_SOURCES})
        configure_file(${CMAKE_CURRENT_LIST_DIR}/${_SOURCE}
                       ${rocpd_PYTHON_OUTPUT_DIRECTORY}/${_SOURCE} COPYONLY)
        install(
            FILES ${rocpd_PYTHON_OUTPUT_DIRECTORY}/${_SOURCE}
            DESTINATION ${rocpd_PYTHON_INSTALL_DIRECTORY}/schema_data
            COMPONENT core)
    endforeach()

    add_library(rocprofiler-sdk-rocpd-python-bindings-${_VERSION} MODULE)
    target_sources(
        rocprofiler-sdk-rocpd-python-bindings-${_VERSION}
        PRIVATE libpyrocpd.cpp libpyrocpd.hpp
                $<TARGET_OBJECTS:rocprofiler-sdk::rocprofiler-sdk-object-library>)
    target_include_directories(rocprofiler-sdk-rocpd-python-bindings-${_VERSION} SYSTEM
                               PRIVATE ${Python3_INCLUDE_DIRS})
    target_link_libraries(
        rocprofiler-sdk-rocpd-python-bindings-${_VERSION}
        PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
                rocprofiler-sdk::rocprofiler-sdk-build-flags
                rocprofiler-sdk::rocprofiler-sdk-memcheck
                rocprofiler-sdk::rocprofiler-sdk-common-library
                rocprofiler-sdk::rocprofiler-sdk-output-library
                rocprofiler-sdk::rocprofiler-sdk-cereal
                rocprofiler-sdk::rocprofiler-sdk-perfetto
                rocprofiler-sdk::rocprofiler-sdk-otf2
                rocprofiler-sdk::rocprofiler-sdk-sqlite3
                rocprofiler-sdk::rocprofiler-sdk-pybind11
                rocprofiler-sdk::rocprofiler-sdk-gotcha
                rocprofiler-sdk::rocprofiler-sdk-dw
                rocprofiler-sdk::rocprofiler-sdk-static-library
                ${Python3_LIBRARIES})

    set_target_properties(
        rocprofiler-sdk-rocpd-python-bindings-${_VERSION}
        PROPERTIES OUTPUT_NAME libpyrocpd
                   RUNTIME_OUTPUT_DIRECTORY ${rocpd_PYTHON_OUTPUT_DIRECTORY}
                   LIBRARY_OUTPUT_DIRECTORY ${rocpd_PYTHON_OUTPUT_DIRECTORY}
                   ARCHIVE_OUTPUT_DIRECTORY ${rocpd_PYTHON_OUTPUT_DIRECTORY}
                   PDB_OUTPUT_DIRECTORY ${rocpd_PYTHON_OUTPUT_DIRECTORY}
                   PREFIX ""
                   SUFFIX ".${Python3_SOABI}${CMAKE_SHARED_LIBRARY_SUFFIX}"
                   BUILD_RPATH "${DEFAULT_PYTHON_RPATH}"
                   INSTALL_RPATH "${DEFAULT_PYTHON_RPATH}")

    install(
        TARGETS rocprofiler-sdk-rocpd-python-bindings-${_VERSION}
        DESTINATION ${rocpd_PYTHON_INSTALL_DIRECTORY}
        COMPONENT core)
endfunction()

function(rocprofiler_rocpd_python_packaging _VERSION)
    message(
        STATUS "Creating rocprofiler-sdk rocpd python packaging for python ${_VERSION}")
    rocprofiler_find_python3(${_VERSION})

    add_custom_target(
        rocprofiler-sdk-rocpd-${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR} ALL
        ${Python3_EXECUTABLE}
        -m
        pip
        install
        -q
        -q
        --prefix
        ${PROJECT_BINARY_DIR}
        -I
        ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        COMMENT
            "Packaging rocpd for python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}..."
        )

    install(
        DIRECTORY
            ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        USE_SOURCE_PERMISSIONS
        COMPONENT core)
endfunction()
