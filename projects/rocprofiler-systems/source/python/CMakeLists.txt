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

function(OMNITRACE_CONFIGURE_PYTARGET _TARGET)

    add_library(omnitrace::${_TARGET} ALIAS ${_TARGET})
    target_link_libraries(${_TARGET} PRIVATE libpyomnitrace-interface)

    set(_SUBDIR ${ARGN})
    if(_SUBDIR)
        set(_SUBDIR "/${_SUBDIR}")
    endif()

    set_target_properties(
        ${_TARGET}
        PROPERTIES PREFIX ""
                   SUFFIX "${PYTHON_MODULE_EXTENSION}"
                   LIBRARY_OUTPUT_DIRECTORY
                   ${PROJECT_BINARY_DIR}/python/omnitrace${_SUBDIR}
                   ARCHIVE_OUTPUT_DIRECTORY
                   ${PROJECT_BINARY_DIR}/python/omnitrace${_SUBDIR}
                   RUNTIME_OUTPUT_DIRECTORY
                   ${PROJECT_BINARY_DIR}/python/omnitrace${_SUBDIR}
                   PDB_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/python/omnitrace${_SUBDIR}
                   INSTALL_RPATH_USE_LINK_PATH ON
                   ${EXTRA_PROPERTIES})

    set(_PYLIB ${CMAKE_INSTALL_PYTHONDIR}/omnitrace${_SUBDIR})
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
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/omnitrace${_SUBDIR}
        OPTIONAL)

    if(NOT "${_TARGET}" STREQUAL "libpyomnitrace")
        add_dependencies(libpyomnitrace ${_TARGET})
    endif()
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

add_library(libpyomnitrace MODULE ${pysources} ${pyheaders})
omnitrace_configure_pytarget(libpyomnitrace)

add_subdirectory(omnitrace)

if(PYTHON_EXECUTABLE)
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
