# ########################################################################################
#
# omnitrace (Python)
#
# ########################################################################################

if(OMNITRACE_BUILD_STATIC_LIBSTDCXX)
    # omnitrace_message(FATAL_ERROR "static libstdc++ is not compatible with python
    # bindings")
endif()

# if set, will screw up loading library
unset(CMAKE_DEBUG_POSTFIX)
set(CMAKE_CXX_CLANG_TIDY)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME python)

# ########################################################################################

function(OMNITRACE_CONFIGURE_PYTARGET _TARGET _VERSION)

    add_library(omnitrace::${_TARGET} ALIAS ${_TARGET})
    target_link_libraries(${_TARGET} PRIVATE libpyomnitrace-interface)
    add_dependencies(libpyomnitrace ${_TARGET})

    set_target_properties(
        ${_TARGET}
        PROPERTIES PREFIX ""
                   OUTPUT_NAME libpyomnitrace
                   LIBRARY_OUTPUT_DIRECTORY
                   ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}/omnitrace
                   ARCHIVE_OUTPUT_DIRECTORY
                   ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}/omnitrace
                   RUNTIME_OUTPUT_DIRECTORY
                   ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}/omnitrace
                   PDB_OUTPUT_DIRECTORY
                   ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}/omnitrace
                   ${EXTRA_PROPERTIES})

    set(_PYLIB ${CMAKE_INSTALL_PYTHONDIR}/omnitrace)
    if(NOT IS_ABSOLUTE "${_PYLIB}")
        set(_PYLIB ${CMAKE_INSTALL_PREFIX}/${_PYLIB})
    endif()

    if(SKBUILD)
        set(LIB_RELPATH ../../..)
    else()
        file(RELATIVE_PATH LIB_RELPATH "${_PYLIB}"
             "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    endif()

    set_target_properties(${_TARGET} PROPERTIES INSTALL_RPATH
                                                "\$ORIGIN:\$ORIGIN/${LIB_RELPATH}")

    install(
        TARGETS ${_TARGET}
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/omnitrace
        OPTIONAL)
endfunction()

# ########################################################################################

add_library(omnitrace-python-compile-options INTERFACE)
add_library(omnitrace::omnitrace-python-compile-options ALIAS
            omnitrace-python-compile-options)
add_target_cxx_flag_if_avail(
    omnitrace-python-compile-options "-Wno-unused-value" "-Wno-range-loop-analysis"
    "-Wno-deprecated-declarations" "-Wno-unused-but-set-parameter"
    "-ftls-model=global-dynamic")

file(GLOB pyheaders ${CMAKE_CURRENT_LIST_DIR}/libpyomnitrace*.hpp)
set(pysources ${CMAKE_CURRENT_LIST_DIR}/libpyomnitrace.cpp)

set(pybind_libs pybind11::module)

add_library(libpyomnitrace-interface INTERFACE)
target_link_libraries(
    libpyomnitrace-interface
    INTERFACE pybind11::module
              timemory::timemory-headers
              omnitrace::omnitrace-headers
              omnitrace::omnitrace-compile-options
              omnitrace::omnitrace-static-libgcc-optional
              omnitrace::omnitrace-lto
              omnitrace::omnitrace-dl-library
              omnitrace::omnitrace-user-library
              omnitrace::omnitrace-python
              omnitrace::omnitrace-python-compile-options)

omnitrace_target_compile_definitions(libpyomnitrace-interface
                                     INTERFACE OMNITRACE_PYBIND11_SOURCE)

add_custom_target(libpyomnitrace)

file(GLOB_RECURSE PYTHON_FILES ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace/*.py)
foreach(_IN ${PYTHON_FILES})
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/omnitrace"
                   "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}/omnitrace" _OUT
                   "${_IN}")
    configure_file(${_IN} ${_OUT} @ONLY)
    install(
        FILES ${_OUT}
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/omnitrace
        OPTIONAL)
endforeach()

omnitrace_python_console_script("omnitrace-python" "omnitrace")

set(_INDEX 0)
foreach(_VERSION ${OMNITRACE_PYTHON_VERSIONS})
    # add_library(libpyomnitrace MODULE ${pysources} ${pyheaders})
    list(GET OMNITRACE_PYTHON_ROOT_DIRS ${_INDEX} Python3_ROOT_DIR)
    omnitrace_pybind11_add_module(
        libpyomnitrace-${_VERSION} MODULE
        PYTHON_VERSION ${_VERSION}
        VISIBILITY "hidden" ${pysources} ${pyheaders})
    omnitrace_configure_pytarget(libpyomnitrace-${_VERSION} ${_VERSION})

    if(OMNITRACE_USE_PYTHON)
        omnitrace_python_console_script(
            "omnitrace-python" "omnitrace"
            VERSION ${_VERSION}
            ROOT_DIR "${Python3_ROOT_DIR}")
    endif()
    math(EXPR _INDEX "${_INDEX} + 1")
endforeach()

configure_file(${CMAKE_CURRENT_LIST_DIR}/setup.py.in
               ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}/setup.py @ONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/setup.cfg.in
               ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}/setup.cfg @ONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/pyproject.toml
               ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}/pyproject.toml COPYONLY)
