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
                   LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/python/omnitrace
                   ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/python/omnitrace
                   RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/python/omnitrace
                   PDB_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/python/omnitrace
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

# ----------------------------------------------------------------------------
# Console scripts
#
function(OMNITRACE_PYTHON_CONSOLE_SCRIPT SCRIPT_NAME SCRIPT_SUBMODULE)
    set(options)
    set(args VERSION ROOT_DIR)
    set(kwargs)
    cmake_parse_arguments(ARG "${options}" "${args}" "${kwargs}" ${ARGN})

    if(ARG_VERSION AND ARG_ROOT_DIR)
        set(Python3_ROOT_DIR "${ARG_ROOT_DIR}")
        find_package(Python3 ${ARG_VERSION} EXACT QUIET MODULE COMPONENTS Interpreter)
        set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
        configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/console-script.in
                       ${PROJECT_BINARY_DIR}/bin/${SCRIPT_NAME}-${ARG_VERSION} @ONLY)

        if(CMAKE_INSTALL_PYTHONDIR)
            install(
                PROGRAMS ${PROJECT_BINARY_DIR}/bin/${SCRIPT_NAME}-${ARG_VERSION}
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                OPTIONAL)
        endif()

        if(OMNITRACE_BUILD_TESTING OR OMNITRACE_BUILD_PYTHON)
            add_test(
                NAME ${SCRIPT_NAME}-console-script-test-${ARG_VERSION}
                COMMAND ${PROJECT_BINARY_DIR}/bin/${SCRIPT_NAME}-${ARG_VERSION} --help
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
            set_tests_properties(
                ${SCRIPT_NAME}-console-script-test-${ARG_VERSION}
                PROPERTIES LABELS "python;python-${ARG_VERSION};console-script")
            add_test(
                NAME ${SCRIPT_NAME}-generic-console-script-test-${ARG_VERSION}
                COMMAND ${PROJECT_BINARY_DIR}/bin/${SCRIPT_NAME} --help
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
            set_tests_properties(
                ${SCRIPT_NAME}-generic-console-script-test-${ARG_VERSION}
                PROPERTIES ENVIRONMENT "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}" LABELS
                           "python;python-${ARG_VERSION};console-script")
        endif()
    else()
        set(PYTHON_EXECUTABLE "python3")

        configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/console-script.in
                       ${PROJECT_BINARY_DIR}/bin/${SCRIPT_NAME} @ONLY)

        if(CMAKE_INSTALL_PYTHONDIR)
            install(
                PROGRAMS ${PROJECT_BINARY_DIR}/bin/${SCRIPT_NAME}
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                OPTIONAL)
        endif()
    endif()
endfunction()

include(PyBind11Tools)

# OMNITRACE_PYTHON_ROOT_DIRS=/opt/conda/envs/py36;/opt/conda/envs/py37;/opt/conda/envs/py38;/opt/conda/envs/py39
# OMNITRACE_PYTHON_VERSIONS=3.6;3.7;3.8;3.9

if(NOT OMNITRACE_PYTHON_VERSIONS AND OMNITRACE_PYTHON_VERSION)
    set(OMNITRACE_PYTHON_VERSIONS "${OMNITRACE_PYTHON_VERSION}")
    if(NOT OMNITRACE_PYTHON_ROOT_DIRS)
        omnitrace_find_python(_PY VERSION ${OMNITRACE_PYTHON_VERSION})
        set(OMNITRACE_PYTHON_ROOT_DIRS
            "${_PY_ROOT_DIR}"
            CACHE INTERNAL "" FORCE)
    endif()
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

list(LENGTH OMNITRACE_PYTHON_VERSIONS _NUM_PYTHON_VERSIONS)
list(LENGTH OMNITRACE_PYTHON_ROOT_DIRS _NUM_PYTHON_ROOT_DIRS)

if(NOT _NUM_PYTHON_VERSIONS EQUAL _NUM_PYTHON_ROOT_DIRS)
    omnitrace_message(
        WARNING
        "Error! Number of python versions  : ${_NUM_PYTHON_VERSIONS}. VERSIONS :: ${OMNITRACE_PYTHON_VERSIONS}"
        )
    omnitrace_message(
        WARNING
        "Error! Number of python root directories : ${_NUM_PYTHON_ROOT_DIRS}. ROOT DIRS :: ${OMNITRACE_PYTHON_ROOT_DIRS}"
        )
    omnitrace_message(
        FATAL_ERROR
        "Error! Number of python versions != number of python root directories")
endif()

file(GLOB_RECURSE PYTHON_FILES ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace/*.py)
foreach(_IN ${PYTHON_FILES})
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/omnitrace"
                   "${PROJECT_BINARY_DIR}/python/omnitrace" _OUT "${_IN}")
    configure_file(${_IN} ${_OUT} @ONLY)
    install(
        FILES ${_OUT}
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/omnitrace
        OPTIONAL)
endforeach()

omnitrace_python_console_script("omnitrace-python" "omnitrace")

execute_process(
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_LIBDIR}/python
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../../python
            ${CMAKE_INSTALL_LIBDIR}/python/site-packages
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

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

if(PYTHON_EXECUTABLE AND OFF)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/setup.py.in
                   ${PROJECT_BINARY_DIR}/python/setup.py @ONLY)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/setup.cfg.in
                   ${PROJECT_BINARY_DIR}/python/setup.cfg @ONLY)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/pyproject.toml
                   ${PROJECT_BINARY_DIR}/python/pyproject.toml COPYONLY)
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} setup.py dist_info
        OUTPUT_VARIABLE _OUT
        RESULT_VARIABLE _RET
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/python)
    set(_DIST_DIR
        ${PROJECT_BINARY_DIR}/python/${PROJECT_NAME}-${PROJECT_VERSION}.dist-info)
    if(NOT EXISTS ${_DIST_DIR})
        set(_DIST_DIR ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.dist-info)
    endif()
    if(EXISTS ${_DIST_DIR} AND IS_DIRECTORY ${_DIST_DIR})
        configure_file(${PROJECT_SOURCE_DIR}/LICENSE ${_DIST_DIR}/LICENSE.txt COPYONLY)
        install(
            DIRECTORY ${_DIST_DIR}
            DESTINATION ${CMAKE_INSTALL_PYTHONDIR}
            OPTIONAL)
    endif()
endif()
