# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# ########################################################################################
#
# rocprofsys (Python)
#
# ########################################################################################

if(ROCPROFSYS_BUILD_STATIC_LIBSTDCXX)
    # rocprofiler_systems_message(FATAL_ERROR "static libstdc++ is not compatible with
    # python bindings")
endif()

# if set, will screw up loading library
unset(CMAKE_DEBUG_POSTFIX)
set(CMAKE_CXX_CLANG_TIDY)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME python)

# ########################################################################################

function(ROCPROFILER_SYSTEMS_CONFIGURE_PYTARGET _TARGET _VERSION)
    add_library(rocprofiler-systems::${_TARGET} ALIAS ${_TARGET})
    target_link_libraries(${_TARGET} PRIVATE libpyrocprofiler-systems-interface)
    add_dependencies(libpyrocprofsys ${_TARGET})

    list(LENGTH ROCPROFSYS_PYTHON_VERSIONS _NUM_PYTHON_VERSIONS)
    if(ROCPROFSYS_PYTHON_VERSION_SPECIFIC_INSTALL AND _NUM_PYTHON_VERSIONS GREATER 1)
        set(_INSTALL_DEST
            "${CMAKE_INSTALL_LIBDIR}/python${_VERSION}/site-packages/rocprofsys"
        )
        set(_PYLIB "${CMAKE_INSTALL_LIBDIR}/python${_VERSION}/site-packages/rocprofsys")
        set(_OUTPUT_DIR
            "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/python${_VERSION}/site-packages/rocprofsys"
        )
    else()
        set(_INSTALL_DEST "${CMAKE_INSTALL_PYTHONDIR}/rocprofsys")
        set(_PYLIB "${CMAKE_INSTALL_PYTHONDIR}/rocprofsys")
        set(_OUTPUT_DIR "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}/rocprofsys")
    endif()

    set_target_properties(
        ${_TARGET}
        PROPERTIES
            PREFIX ""
            OUTPUT_NAME libpyrocprofsys
            LIBRARY_OUTPUT_DIRECTORY ${_OUTPUT_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${_OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${_OUTPUT_DIR}
            PDB_OUTPUT_DIRECTORY ${_OUTPUT_DIR}
            ${EXTRA_PROPERTIES}
    )

    if(NOT IS_ABSOLUTE "${_PYLIB}")
        set(_PYLIB ${CMAKE_INSTALL_PREFIX}/${_PYLIB})
    endif()

    if(SKBUILD)
        set(LIB_RELPATH ../../..)
    else()
        file(
            RELATIVE_PATH
            LIB_RELPATH
            "${_PYLIB}"
            "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        )
    endif()

    set_target_properties(
        ${_TARGET}
        PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/${LIB_RELPATH}"
    )

    install(TARGETS ${_TARGET} DESTINATION ${_INSTALL_DEST} OPTIONAL)
endfunction()

# ########################################################################################

add_library(rocprofiler-systems-python-compile-options INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-python-compile-options
    ALIAS rocprofiler-systems-python-compile-options
)
add_target_cxx_flag_if_avail(
    rocprofiler-systems-python-compile-options "-Wno-unused-value"
    "-Wno-range-loop-analysis" "-Wno-deprecated-declarations"
    "-Wno-unused-but-set-parameter" "-ftls-model=global-dynamic"
)

file(GLOB pyheaders ${CMAKE_CURRENT_LIST_DIR}/libpyrocprofsys*.hpp)
set(pysources ${CMAKE_CURRENT_LIST_DIR}/libpyrocprofsys.cpp)

set(pybind_libs pybind11::module)

add_library(libpyrocprofiler-systems-interface INTERFACE)
target_link_libraries(
    libpyrocprofiler-systems-interface
    INTERFACE
        pybind11::module
        timemory::timemory-headers
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-compile-options
        rocprofiler-systems::rocprofiler-systems-static-libgcc-optional
        rocprofiler-systems::rocprofiler-systems-lto
        rocprofiler-systems::rocprofiler-systems-dl-library
        rocprofiler-systems::rocprofiler-systems-user-library
        rocprofiler-systems::rocprofiler-systems-python
        rocprofiler-systems::rocprofiler-systems-python-compile-options
)

rocprofiler_systems_target_compile_definitions(libpyrocprofiler-systems-interface
                                               INTERFACE ROCPROFSYS_PYBIND11_SOURCE
)

add_custom_target(libpyrocprofsys)

list(LENGTH ROCPROFSYS_PYTHON_VERSIONS _NUM_PYTHON_VERSIONS)
file(GLOB_RECURSE PYTHON_FILES ${CMAKE_CURRENT_SOURCE_DIR}/rocprofsys/*.py)
foreach(_IN ${PYTHON_FILES})
    if(ROCPROFSYS_PYTHON_VERSION_SPECIFIC_INSTALL AND _NUM_PYTHON_VERSIONS GREATER 1)
        foreach(_VER ${ROCPROFSYS_PYTHON_VERSIONS})
            string(
                REPLACE
                "${CMAKE_CURRENT_SOURCE_DIR}/rocprofsys"
                "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/python${_VER}/site-packages/rocprofsys"
                _OUT
                "${_IN}"
            )
            configure_file(${_IN} ${_OUT} @ONLY)
            install(
                FILES ${_OUT}
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${_VER}/site-packages/rocprofsys
                OPTIONAL
            )
        endforeach()
    else()
        string(
            REPLACE
            "${CMAKE_CURRENT_SOURCE_DIR}/rocprofsys"
            "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}/rocprofsys"
            _OUT
            "${_IN}"
        )
        configure_file(${_IN} ${_OUT} @ONLY)
        install(FILES ${_OUT} DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/rocprofsys OPTIONAL)
    endif()
endforeach()

rocprofiler_systems_python_console_script("${BINARY_NAME_PREFIX}-python" "rocprofsys")

set(_INDEX 0)
foreach(_VERSION ${ROCPROFSYS_PYTHON_VERSIONS})
    # add_library(libpyrocprofsys MODULE ${pysources} ${pyheaders})
    list(GET ROCPROFSYS_PYTHON_ROOT_DIRS ${_INDEX} Python3_ROOT_DIR)
    rocprofiler_systems_pybind11_add_module(
        libpyrocprofiler-systems-${_VERSION} MODULE
        PYTHON_VERSION ${_VERSION}
        VISIBILITY "hidden" ${pysources} ${pyheaders}
    )
    rocprofiler_systems_configure_pytarget(libpyrocprofiler-systems-${_VERSION}
                                           ${_VERSION}
    )
    math(EXPR _INDEX "${_INDEX} + 1")
endforeach()

if(ROCPROFSYS_PYTHON_VERSION_SPECIFIC_INSTALL AND _NUM_PYTHON_VERSIONS GREATER 1)
    foreach(_VER ${ROCPROFSYS_PYTHON_VERSIONS})
        set(_PYTHON_OUT_DIR
            "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/python${_VER}/site-packages"
        )
        configure_file(
            ${CMAKE_CURRENT_LIST_DIR}/setup.py.in
            ${_PYTHON_OUT_DIR}/setup.py
            @ONLY
        )
        configure_file(
            ${CMAKE_CURRENT_LIST_DIR}/setup.cfg.in
            ${_PYTHON_OUT_DIR}/setup.cfg
            @ONLY
        )
        configure_file(
            ${CMAKE_CURRENT_LIST_DIR}/pyproject.toml
            ${_PYTHON_OUT_DIR}/pyproject.toml
            COPYONLY
        )
    endforeach()
else()
    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/setup.py.in
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}/setup.py
        @ONLY
    )
    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/setup.cfg.in
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}/setup.cfg
        @ONLY
    )
    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/pyproject.toml
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}/pyproject.toml
        COPYONLY
    )
endif()
