# ########################################################################################
#
# omnitrace (Python)
#
# ########################################################################################

# if set, will screw up loading library
unset(CMAKE_DEBUG_POSTFIX)
set(CMAKE_CXX_CLANG_TIDY)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME python)

# ########################################################################################

function(OMNITRACE_CONFIGURE_PYTARGET _TARGET _VERSION)

    add_library(omnitrace::${_TARGET} ALIAS ${_TARGET})
    target_link_libraries(${_TARGET} PRIVATE libpyomnitrace-interface)

    set_target_properties(
        ${_TARGET}
        PROPERTIES PREFIX ""
                   OUTPUT_NAME libpyomnitrace
                   LIBRARY_OUTPUT_DIRECTORY
                   ${PROJECT_BINARY_DIR}/lib/python/site-packages/omnitrace
                   ARCHIVE_OUTPUT_DIRECTORY
                   ${PROJECT_BINARY_DIR}/lib/python/site-packages/omnitrace
                   RUNTIME_OUTPUT_DIRECTORY
                   ${PROJECT_BINARY_DIR}/lib/python/site-packages/omnitrace
                   PDB_OUTPUT_DIRECTORY
                   ${PROJECT_BINARY_DIR}/lib/python/site-packages/omnitrace
                   INSTALL_RPATH_USE_LINK_PATH ON
                   ${EXTRA_PROPERTIES})

    set(_PYLIB ${CMAKE_INSTALL_PYTHONDIR}/omnitrace)
    if(NOT IS_ABSOLUTE "${_PYLIB}")
        set(_PYLIB ${CMAKE_INSTALL_PREFIX}/${_PYLIB})
    endif()

    if(SKBUILD)
        set(LIB_RELPATH ../../../..)
    else()
        file(RELATIVE_PATH LIB_RELPATH "${_PYLIB}"
             "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    endif()

    set_target_properties(
        ${_TARGET}
        PROPERTIES
            INSTALL_RPATH
            "\$ORIGIN:\$ORIGIN/${LIB_RELPATH}:\$ORIGIN/../../../..:${CMAKE_INSTALL_RPATH}"
        )

    install(
        TARGETS ${_TARGET}
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/omnitrace
        OPTIONAL)
endfunction()

# ########################################################################################

add_library(omnitrace-python-compile-options INTERFACE)
add_library(omnitrace::omnitrace-python-compile-options ALIAS
            omnitrace-python-compile-options)
add_cxx_flag_if_avail("-frtti" omnitrace-python-compile-options)
add_cxx_flag_if_avail("-Wno-unused-value" omnitrace-python-compile-options)
add_cxx_flag_if_avail("-Wno-range-loop-analysis" omnitrace-python-compile-options)
add_cxx_flag_if_avail("-ftls-model=global-dynamic" omnitrace-python-compile-options)
add_cxx_flag_if_avail("-Wno-deprecated-declarations" omnitrace-python-compile-options)
add_cxx_flag_if_avail("-Wno-unused-but-set-parameter" omnitrace-python-compile-options)

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
              omnitrace::omnitrace-lto
              omnitrace::omnitrace-dl-library
              omnitrace::omnitrace-python
              omnitrace::omnitrace-python-compile-options)

target_compile_definitions(libpyomnitrace-interface INTERFACE OMNITRACE_PYBIND11_SOURCE)

include(PyBind11Tools)

omnitrace_watch_for_change(OMNITRACE_PYTHON_ROOT_DIRS _PYTHON_DIRS_CHANGED)
if(_PYTHON_DIRS_CHANGED)
    unset(OMNITRACE_PYTHON_VERSION CACHE)
    unset(OMNITRACE_PYTHON_VERSIONS CACHE)
else()
    foreach(_VAR PREFIX ENVS)
        omnitrace_watch_for_change(OMNITRACE_PYTHON_${_VAR} _CHANGED)
        if(_CHANGED)
            unset(OMNITRACE_PYTHON_ROOT_DIRS CACHE)
            unset(OMNITRACE_PYTHON_VERSIONS CACHE)
            break()
        endif()
    endforeach()
endif()

if(OMNITRACE_PYTHON_PREFIX AND OMNITRACE_PYTHON_ENVS)
    omnitrace_directory(
        FAIL
        PREFIX ${OMNITRACE_PYTHON_PREFIX}
        PATHS ${OMNITRACE_PYTHON_ENVS}
        OUTPUT_VARIABLE _PYTHON_ROOT_DIRS)
    set(OMNITRACE_PYTHON_ROOT_DIRS
        "${_PYTHON_ROOT_DIRS}"
        CACHE INTERNAL "Root directories for python")
endif()

if(NOT OMNITRACE_PYTHON_VERSIONS AND OMNITRACE_PYTHON_VERSION)
    set(OMNITRACE_PYTHON_VERSIONS "${OMNITRACE_PYTHON_VERSION}")
    if(NOT OMNITRACE_PYTHON_ROOT_DIRS)
        omnitrace_find_python(_PY VERSION ${OMNITRACE_PYTHON_VERSION})
        set(OMNITRACE_PYTHON_ROOT_DIRS
            "${_PY_ROOT_DIR}"
            CACHE INTERNAL "" FORCE)
    endif()
    unset(OMNITRACE_PYTHON_VERSION CACHE)
elseif(
    NOT OMNITRACE_PYTHON_VERSIONS
    AND NOT OMNITRACE_PYTHON_VERSION
    AND OMNITRACE_PYTHON_ROOT_DIRS)
    set(_PY_VERSIONS)
    foreach(_DIR ${OMNITRACE_PYTHON_ROOT_DIRS})
        omnitrace_find_python(_PY ROOT_DIR ${_DIR})
        if(NOT _PY_FOUND)
            continue()
        endif()
        if(NOT "${_PY_VERSION}" IN_LIST _PY_VERSIONS)
            list(APPEND _PY_VERSIONS "${_PY_VERSION}")
        endif()
    endforeach()
    set(OMNITRACE_PYTHON_VERSIONS
        "${_PY_VERSIONS}"
        CACHE INTERNAL "" FORCE)
elseif(
    NOT OMNITRACE_PYTHON_VERSIONS
    AND NOT OMNITRACE_PYTHON_VERSION
    AND NOT OMNITRACE_PYTHON_ROOT_DIRS)
    omnitrace_find_python(_PY REQUIRED)
    set(OMNITRACE_PYTHON_ROOT_DIRS
        "${_PY_ROOT_DIR}"
        CACHE INTERNAL "" FORCE)
    set(OMNITRACE_PYTHON_VERSIONS
        "${_PY_VERSION}"
        CACHE INTERNAL "" FORCE)
endif()

omnitrace_watch_for_change(OMNITRACE_PYTHON_ROOT_DIRS)
omnitrace_watch_for_change(OMNITRACE_PYTHON_VERSIONS)

omnitrace_check_python_dirs_and_versions(FAIL)

file(GLOB_RECURSE PYTHON_FILES ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace/*.py)
foreach(_IN ${PYTHON_FILES})
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/omnitrace"
                   "${PROJECT_BINARY_DIR}/lib/python/site-packages/omnitrace" _OUT
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
               ${PROJECT_BINARY_DIR}/lib/python/site-packages/setup.py @ONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/setup.cfg.in
               ${PROJECT_BINARY_DIR}/lib/python/site-packages/setup.cfg @ONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/pyproject.toml
               ${PROJECT_BINARY_DIR}/lib/python/site-packages/pyproject.toml COPYONLY)
