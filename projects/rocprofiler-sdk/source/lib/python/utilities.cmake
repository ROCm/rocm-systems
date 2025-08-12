#
#  functions/macros for python
#

include_guard(DIRECTORY)

function(rocprofiler_roctx_python_bindings _VERSION)
    message(
        STATUS "Building rocprofiler-sdk roctx python bindings for python ${_VERSION}")

    rocprofiler_find_python3(${_VERSION} QUIET)

    set(roctx_PYTHON_INSTALL_DIRECTORY
        ${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/roctx
        )
    set(roctx_PYTHON_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/${roctx_PYTHON_INSTALL_DIRECTORY})
    set(roctx_PYTHON_SOURCES __init__.py context_decorators.py)

    foreach(_SOURCE ${roctx_PYTHON_SOURCES})
        configure_file(${CMAKE_CURRENT_LIST_DIR}/${_SOURCE}
                       ${roctx_PYTHON_OUTPUT_DIRECTORY}/${_SOURCE} @ONLY)
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

function(rocprofiler_rocpd_python_bindings_object_sources)
    if(TARGET rocprofiler-sdk-rocpd-python-bindings-object-library)
        target_sources(rocprofiler-sdk-rocpd-python-bindings-object-library ${ARGN})
    endif()
endfunction()

function(rocprofiler_rocpd_python_bindings_target_sources _VERSION)
    target_sources(rocprofiler-sdk-rocpd-python-bindings-${_VERSION} ${ARGN})
endfunction()

function(rocprofiler_rocpd_python_bindings _VERSION)
    message(
        STATUS "Building rocprofiler-sdk rocpd python bindings for python ${_VERSION}")
    rocprofiler_find_python3(${_VERSION} QUIET)

    set(rocpd_PYTHON_INSTALL_DIRECTORY
        ${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/rocpd
        )
    set(rocpd_PYTHON_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/${rocpd_PYTHON_INSTALL_DIRECTORY})
    set(rocpd_PYTHON_SOURCES
        csv.py
        filter.py
        importer.py
        __init__.py
        __main__.py
        output_config.py
        otf2.py
        pftrace.py
        query.py
        schema.py
        summary.py
        time_window.py)

    foreach(_SOURCE ${rocpd_PYTHON_SOURCES})
        configure_file(${CMAKE_CURRENT_LIST_DIR}/${_SOURCE}
                       ${rocpd_PYTHON_OUTPUT_DIRECTORY}/${_SOURCE} @ONLY)
        install(
            FILES ${rocpd_PYTHON_OUTPUT_DIRECTORY}/${_SOURCE}
            DESTINATION ${rocpd_PYTHON_INSTALL_DIRECTORY}
            COMPONENT rocpd)
    endforeach()

    if(NOT TARGET rocprofiler-sdk-rocpd-python-bindings-object-library)
        add_library(rocprofiler-sdk-rocpd-python-bindings-object-library OBJECT)
        add_library(rocprofiler-sdk::rocpd-python-bindings-object-library ALIAS
                    rocprofiler-sdk-rocpd-python-bindings-object-library)
        target_link_libraries(
            rocprofiler-sdk-rocpd-python-bindings-object-library
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
                    rocprofiler-sdk::rocprofiler-sdk-rocpd-library)
        set_target_properties(rocprofiler-sdk-rocpd-python-bindings-object-library
                              PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    add_library(rocprofiler-sdk-rocpd-python-bindings-${_VERSION} MODULE)
    target_sources(
        rocprofiler-sdk-rocpd-python-bindings-${_VERSION}
        PRIVATE libpyrocpd.cpp libpyrocpd.hpp
                $<TARGET_OBJECTS:rocprofiler-sdk::rocprofiler-sdk-object-library>
                $<TARGET_OBJECTS:rocprofiler-sdk::rocpd-python-bindings-object-library>)
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
                rocprofiler-sdk::rocprofiler-sdk-rocpd-library
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
        COMPONENT rocpd)
endfunction()
