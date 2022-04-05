# Python configuration
#

# include guard
include_guard(DIRECTORY)

# Stops lookup as soon as a version satisfying version constraints is found.
set(Python3_FIND_STRATEGY
    "LOCATION"
    CACHE STRING
          "Stops lookup as soon as a version satisfying version constraints is found")

# virtual environment is used before any other standard paths to look-up for the
# interpreter
set(Python3_FIND_VIRTUALENV
    "FIRST"
    CACHE STRING "Virtual environment is used before any other standard paths")
set_property(CACHE Python3_FIND_VIRTUALENV PROPERTY STRINGS "FIRST;LAST;NEVER")

if(APPLE)
    set(Python3_FIND_FRAMEWORK
        "LAST"
        CACHE STRING
              "Order of preference between Apple-style and unix-style package components")
    set_property(CACHE Python3_FIND_FRAMEWORK PROPERTY STRINGS "FIRST;LAST;NEVER")
endif()

# PyPy does not support embedding the interpreter
set(Python3_FIND_IMPLEMENTATIONS
    "CPython"
    CACHE STRING "Different implementations which will be searched.")
set_property(CACHE Python3_FIND_IMPLEMENTATIONS PROPERTY STRINGS
                                                         "CPython;IronPython;PyPy")

# variable is a 3-tuple specifying, in order, pydebug (d), pymalloc (m) and unicode (u)
# set(Python3_FIND_ABI "OFF" "OFF" "OFF" CACHE STRING "variable is a 3-tuple specifying
# pydebug (d), pymalloc (m) and unicode (u)")

# Create CMake cache entries for the above artifact specification variables so that users
# can edit them interactively. This disables support for multiple version/component
# requirements.
set(Python3_ARTIFACTS_INTERACTIVE
    ON
    CACHE BOOL "Create CMake cache entries so that users can edit them interactively")

# if("${Python3_USE_STATIC_LIBS}" STREQUAL "ANY") set(Python3_USE_STATIC_LIBS "OFF" CACHE
# STRING "If ON, only static libs; if OFF, only shared libs; if ANY, shared then static")
# set_property(CACHE Python3_USE_STATIC_LIBS PROPERTY STRINGS "ON;OFF;ANY") else()
# unset(Python3_USE_STATIC_LIBS) endif()

foreach(_VAR FIND_STRATEGY FIND_VIRTUALENV FIND_FRAMEWORK FIND_IMPLEMENTATIONS
             ARTIFACTS_INTERACTIVE)
    if(DEFINED Python3_${_VAR})
        set(Python_${_VAR}
            "${Python3_${_VAR}}"
            CACHE STRING "Set via Python3_${_VAR} setting (omnitrace)")
        mark_as_advanced(Python_${_VAR})
        mark_as_advanced(Python3_${_VAR})
    endif()
endforeach()

# display version
omnitrace_add_feature(OMNITRACE_PYTHON_VERSION "Python version for omnitrace" DOC)

# search hint
if(PYTHON_ROOT_DIR AND NOT Python3_ROOT_DIR)
    set(Python3_ROOT_DIR ${PYTHON_ROOT_DIR})
endif()

# legacy specification of interpreter
if(PYTHON_EXECUTABLE AND NOT Python3_EXECUTABLE)
    set(Python3_EXECUTABLE
        "${PYTHON_EXECUTABLE}"
        CACHE FILEPATH "Path to Python3 interpreter")
endif()

# default python types to search for
set(Python_ADDITIONAL_VERSIONS
    "3.9;3.8;3.7;3.6"
    CACHE STRING "Python versions supported by omnitrace")

# override types to search for
if(OMNITRACE_PYTHON_VERSION)
    set(Python_ADDITIONAL_VERSIONS
        ${OMNITRACE_PYTHON_VERSION}
        CACHE STRING "Python versions supported by omnitrace" FORCE)
elseif(PYBIND11_PYTHON_VERSION)
    set(Python_ADDITIONAL_VERSIONS
        ${PYBIND11_PYTHON_VERSION}
        CACHE STRING "Python versions supported by omnitrace")
endif()

# unset the version strings
if(_PYVERSION_LAST AND (OMNITRACE_PYTHON_VERSION VERSION_LESS _PYVERSION_LAST
                        OR OMNITRACE_PYTHON_VERSION VERSION_GREATER _PYVERSION_LAST))
    unset(OMNITRACE_PYTHON_VERSION CACHE)
    unset(PYBIND11_PYTHON_VERSION CACHE)
    unset(CMAKE_INSTALL_PYTHONDIR CACHE)
endif()

# if OMNITRACE_PYTHON_VERSION specified, set to desired python version
set(_PYVERSION ${OMNITRACE_PYTHON_VERSION})

# if OMNITRACE_PYTHON_VERSION is not set but PYBIND11_PYTHON_VERSION is
if("${_PYVERSION}" STREQUAL "" AND PYBIND11_PYTHON_VERSION)
    set(_PYVERSION ${PYBIND11_PYTHON_VERSION})
endif()

# basically just used to get Python3_SITEARCH for installation
find_package(Python3 ${_PYVERSION} MODULE ${OMNITRACE_FIND_REQUIREMENT}
             COMPONENTS Interpreter Development)

# executable
set(PYTHON_EXECUTABLE
    "${Python3_EXECUTABLE}"
    CACHE FILEPATH "Set via Python3_EXECUTABLE (omnitrace)" FORCE)
# includes
if(Python3_INCLUDE_DIR AND NOT Python3_INCLUDE_DIRS)
    set(Python3_INCLUDE_DIRS ${Python3_INCLUDE_DIR})
endif()
if(Python3_INCLUDE_DIRS)
    set(PYTHON_INCLUDE_DIR
        "${Python3_INCLUDE_DIRS}"
        CACHE PATH "Set via Python3_INCLUDE_DIR (omnitrace)" FORCE)
    set(PYTHON_INCLUDE_DIRS
        "${Python3_INCLUDE_DIRS}"
        CACHE PATH "Set via Python3_INCLUDE_DIRS (omnitrace)" FORCE)
endif()
# libraries
set(PYTHON_LIBRARY_DEBUG
    "${Python3_LIBRARY_DEBUG}"
    CACHE FILEPATH "Set via Python3_LIBRARY_DEBUG (omnitrace)" FORCE)
set(PYTHON_LIBRARY_RELEASE
    "${Python3_LIBRARY_RELEASE}"
    CACHE FILEPATH "Set via Python3_LIBRARY_DEBUG (omnitrace)" FORCE)
if(Python3_LIBRARY_RELEASE)
    set(PYTHON_LIBRARY
        "${Python3_LIBRARY_RELEASE}"
        CACHE FILEPATH "Set via Python3_LIBRARY (omnitrace)" FORCE)
    set(PYTHON_LIBRARIES
        "${Python3_LIBRARY_RELEASE}"
        CACHE FILEPATH "Set via Python3_LIBRARIES (omnitrace)" FORCE)
else(Python3_LIBRARY_DEBUG)
    set(PYTHON_LIBRARY
        "${Python3_LIBRARY_DEBUG}"
        CACHE FILEPATH "Set via Python3_LIBRARY (omnitrace)" FORCE)
    set(PYTHON_LIBRARIES
        "${Python3_LIBRARY_DEBUG}"
        CACHE FILEPATH "Set via Python3_LIBRARIES (omnitrace)" FORCE)
endif()
set(PYTHON_LIBRARY_DIRS
    "${Python3_LIBRARY_DIRS}"
    CACHE PATH "Set via Python3_LIBRARY_DIRS (omnitrace)" FORCE)
set(PYTHON_LINK_OPTIONS
    "${Python3_LINK_OPTIONS}"
    CACHE STRING "Set via Python3_LINK_OPTIONS (omnitrace)" FORCE)

# module
set(PYTHON_MODULE_EXTENSION
    "${Python3_MODULE_EXTENSION}"
    CACHE STRING "Set via Python3_MODULE_EXTENSION (omnitrace)" FORCE)
set(PYTHON_MODULE_PREFIX
    "${Python3_MODULE_PREFIX}"
    CACHE STRING "Set via Python3_MODULE_PREFIX (omnitrace)" FORCE)

# version
set(PYTHON_VERSION
    "${Python3_VERSION}"
    CACHE STRING "Set via Python3_VERSION (omnitrace)" FORCE)
set(PYTHON_VERSION_MAJOR
    "${Python3_VERSION_MAJOR}"
    CACHE STRING "Set via Python3_VERSION_MAJOR (omnitrace)" FORCE)
set(PYTHON_VERSION_MINOR
    "${Python3_VERSION_MINOR}"
    CACHE STRING "Set via Python3_VERSION_MINOR (omnitrace)" FORCE)

# find_package
set(PythonInterp_FOUND ${Python3_Interpreter_FOUND})
set(PythonLibs_FOUND ${Python3_Development_FOUND})

# set OMNITRACE_PYTHON_VERSION if we have the python version
if(PYTHON_VERSION_STRING)
    set(OMNITRACE_PYTHON_VERSION
        "${PYTHON_VERSION_STRING}"
        CACHE STRING "Python version for omnitrace")
endif()

# if either not found, disable
if(NOT Python3_FOUND)
    set(OMNITRACE_USE_PYTHON OFF)
    set(OMNITRACE_BUILD_PYTHON OFF)
    omnitrace_inform_empty_interface(omnitrace-python "Python embedded interpreter")
    omnitrace_inform_empty_interface(omnitrace-plotting "Python plotting from C++")
else()
    set(OMNITRACE_PYTHON_VERSION
        "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}"
        CACHE STRING "Python version for omnitrace")
    omnitrace_add_feature(PYTHON_EXECUTABLE "Python executable")
endif()

# C++ standard
if(NOT MSVC)
    if(NOT "${PYBIND11_CPP_STANDARD}" STREQUAL "-std=c++${CMAKE_CXX_STANDARD}")
        set(PYBIND11_CPP_STANDARD
            -std=c++${CMAKE_CXX_STANDARD}
            CACHE STRING "PyBind11 CXX standard" FORCE)
    endif()
else()
    if(NOT "${PYBIND11_CPP_STANDARD}" STREQUAL "/std:c++${CMAKE_CXX_STANDARD}")
        set(PYBIND11_CPP_STANDARD
            /std:c++${CMAKE_CXX_STANDARD}
            CACHE STRING "PyBind11 CXX standard" FORCE)
    endif()
endif()

option(PYBIND11_INSTALL "Enable Pybind11 installation" OFF)

if(OMNITRACE_BUILD_PYTHON AND NOT TARGET pybind11)
    # checkout PyBind11 if not checked out
    omnitrace_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/pybind11
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/timemory
        REPO_URL https://github.com/jrmadsen/pybind11.git
        REPO_BRANCH master)

    # add PyBind11 to project omnitrace_save_variables(IPO VARIABLES
    # CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    add_subdirectory(${PROJECT_SOURCE_DIR}/external/timemory/external/pybind11)
    # omnitrace_restore_variables(IPO VARIABLES CMAKE_INTERPROCEDURAL_OPTIMIZATION)
endif()

if(NOT PYBIND11_PYTHON_VERSION)
    unset(PYBIND11_PYTHON_VERSION CACHE)
    execute_process(
        COMMAND
            ${PYTHON_EXECUTABLE} -c
            "import sys; print('{}.{}'.format(sys.version_info[0], sys.version_info[1]))"
        OUTPUT_VARIABLE PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    set(PYBIND11_PYTHON_VERSION
        "${PYTHON_VERSION}"
        CACHE STRING "Python version")
endif()

omnitrace_add_feature(PYBIND11_CPP_STANDARD "PyBind11 C++ standard")
omnitrace_add_feature(PYBIND11_PYTHON_VERSION "PyBind11 Python version")

if(NOT "${OMNITRACE_PYTHON_VERSION}" MATCHES "${PYBIND11_PYTHON_VERSION}*")
    message(STATUS "OMNITRACE_PYTHON_VERSION is set to ${OMNITRACE_PYTHON_VERSION}")
    message(STATUS "PYBIND11_PYTHON_VERSION is set to ${PYBIND11_PYTHON_VERSION}")
    message(
        FATAL_ERROR "Mismatched 'OMNITRACE_PYTHON_VERSION' and 'PYBIND11_PYTHON_VERSION'")
endif()

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c
            "import time ; print('{} {}'.format(time.ctime(), time.tzname[0]))"
    OUTPUT_VARIABLE OMNITRACE_INSTALL_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

string(REPLACE "  " " " OMNITRACE_INSTALL_DATE "${OMNITRACE_INSTALL_DATE}")

if(SKBUILD)
    set(OMNITRACE_INSTALL_PYTHON
        "prefix"
        CACHE STRING "SKBUILD forced python install type" FORCE)
else()
    set(OMNITRACE_INSTALL_PYTHON
        "lib"
        CACHE STRING "Installation type for python (prefix, lib, or global)")
endif()

omnitrace_add_feature(OMNITRACE_INSTALL_PYTHON
                      "Installation type for python (prefix, lib, or global)")

if(SKBUILD OR "${OMNITRACE_INSTALL_PYTHON}" STREQUAL "prefix")
    set(CMAKE_INSTALL_PYTHONDIR
        ${CMAKE_INSTALL_PREFIX}
        CACHE PATH "Installation directory for python")
elseif(SPACK_BUILD OR "${OMNITRACE_INSTALL_PYTHON}" STREQUAL "lib")
    set(CMAKE_INSTALL_PYTHONDIR
        lib/python${PYBIND11_PYTHON_VERSION}/site-packages
        CACHE PATH "Installation directory for python")
else()
    string(REPLACE "\\" "/" Python3_SITEARCH "${Python3_SITEARCH}")
    set(CMAKE_INSTALL_PYTHONDIR ${Python3_SITEARCH})
    omnitrace_add_feature(Python3_SITEARCH
                          "site-packages directory of python installation")
    set(_REMOVE OFF)
    # make the directory if it doesn't exist
    if(NOT EXISTS ${Python3_SITEARCH}/omnitrace)
        set(_REMOVE ON)
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E make_directory ${Python3_SITEARCH}/omnitrace
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            ERROR_QUIET)
    endif()
    # figure out if we can install to Python3_SITEARCH
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E touch ${Python3_SITEARCH}/omnitrace/__init__.py
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        ERROR_VARIABLE ERR_MSG
        RESULT_VARIABLE ERR_CODE)
    # remove the directory if we created it
    if(_REMOVE)
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${Python3_SITEARCH}/omnitrace
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            ERROR_QUIET)
    endif()
    # check the error code of the touch command
    if(ERR_CODE)
        if("${OMNITRACE_INSTALL_PYTHON}" STREQUAL "global")
            message(
                FATAL_ERROR
                    "omnitrace could not install python files to ${Python3_SITEARCH} (not writable):\n${ERR_MSG}"
                )
        endif()
        # get the python directory name, e.g. 'python3.6' from
        # '/opt/local/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6'
        get_filename_component(PYDIR "${Python3_STDLIB}" NAME)
        omnitrace_add_feature(Python3_STDLIB
                              "standard-library directory of python installation")
        # Should not be CMAKE_INSTALL_LIBDIR! Python won't look in a lib64 folder
        set(CMAKE_INSTALL_PYTHONDIR lib/${PYDIR}/site-packages)
    endif()
endif()

if(OMNITRACE_BUILD_PYTHON OR pybind11_FOUND)
    set(_PYBIND11_INCLUDE_DIRS)
    foreach(_TARG pybind11 pybind11::pybind11 pybind11::module)
        if(TARGET ${_TARG})
            get_target_property(_INCLUDE_DIR ${_TARG} INTERFACE_INCLUDE_DIRECTORIES)
            if(_INCLUDE_DIR)
                list(APPEND _PYBIND11_INCLUDE_DIRS ${_INCLUDE_DIR})
            endif()
        endif()
    endforeach()
    if(_PYBIND11_INCLUDE_DIRS)
        list(REMOVE_DUPLICATES _PYBIND11_INCLUDE_DIRS)
    endif()
    omnitrace_target_compile_definitions(omnitrace-python INTERFACE OMNITRACE_USE_PYTHON)
    if(NOT APPLE)
        target_link_libraries(omnitrace-python INTERFACE ${PYTHON_LIBRARIES})
    endif()
    if(PYTHON_INCLUDE_DIRS)
        target_include_directories(omnitrace-python SYSTEM
                                   INTERFACE ${PYTHON_INCLUDE_DIRS})
    endif()
    if(PYBIND11_INCLUDE_DIRS)
        target_include_directories(omnitrace-python SYSTEM
                                   INTERFACE ${PYBIND11_INCLUDE_DIRS})
    endif()
    if(PYBIND11_INCLUDE_DIR)
        target_include_directories(omnitrace-python SYSTEM
                                   INTERFACE $<BUILD_INTERFACE:${PYBIND11_INCLUDE_DIR}>)
    endif()
    if(_PYBIND11_INCLUDE_DIRS)
        target_include_directories(omnitrace-python SYSTEM
                                   INTERFACE $<BUILD_INTERFACE:${_PYBIND11_INCLUDE_DIRS}>)
    endif()
endif()

if(APPLE)
    if(CMAKE_VERSION VERSION_LESS 3.18)
        target_link_libraries(omnitrace-python INTERFACE "-undefined dynamic_lookup")
    else()
        target_link_libraries(
            omnitrace-python
            INTERFACE "$<$<LINK_LANGUAGE:CXX>:-undefined dynamic_lookup>")
    endif()
endif()

if(WIN32)
    # Windows produces:
    #
    # CMake Warning (dev) at tests/test-python-install-import.cmake:3 (SET): Syntax error
    # in cmake code at
    # C:/projects/omnitrace/build-omnitrace/tests/test-python-install-import.cmake:3 when
    # parsing string C:\Python36-x64\Lib\site-packages Invalid escape sequence \P
    string(REPLACE "\\" "/" INSTALL_PYTHONDIR "${CMAKE_INSTALL_PYTHONDIR}")
else()
    set(INSTALL_PYTHONDIR "${CMAKE_INSTALL_PYTHONDIR}")
endif()

configure_file(
    ${PROJECT_SOURCE_DIR}/external/timemory/cmake/Templates/test-python-install-import.cmake.in
    ${PROJECT_BINARY_DIR}/tests/test-python-install-import.cmake
    @ONLY)
unset(INSTALL_PYTHONDIR)

omnitrace_add_feature(CMAKE_INSTALL_PYTHONDIR
                      "Installation prefix of the python bindings")

set(_PYVERSION_LAST
    "${OMNITRACE_PYTHON_VERSION}"
    CACHE INTERNAL "Last version" FORCE)

find_package(PythonInterp ${OMNITRACE_PYTHON_VERSION} EXACT REQUIRED)
find_package(PythonLibs ${OMNITRACE_PYTHON_VERSION} EXACT REQUIRED)
# find_package(PythonExtensions REQUIRED)

if("${PYTHON_MODULE_EXTENSION}" STREQUAL "")
    execute_process(
        COMMAND
            "${Python3_EXECUTABLE}" "-c" "
from distutils import sysconfig as s;import sys;import struct;
print('.'.join(str(v) for v in sys.version_info));
print(sys.prefix);
print(s.get_python_inc(plat_specific=True));
print(s.get_python_lib(plat_specific=True));
print(s.get_config_var('EXT_SUFFIX') or s.get_config_var('SO'));
print(hasattr(sys, 'gettotalrefcount')+0);
print(struct.calcsize('@P'));
print(s.get_config_var('LDVERSION') or s.get_config_var('VERSION'));
print(s.get_config_var('LIBDIR') or '');
print(s.get_config_var('MULTIARCH') or '');
"
        RESULT_VARIABLE _PYTHON_SUCCESS
        OUTPUT_VARIABLE _PYTHON_VALUES
        ERROR_VARIABLE _PYTHON_ERROR_VALUE)

    if(_PYTHON_SUCCESS MATCHES 0)
        # Convert the process output into a list
        if(WIN32)
            string(REGEX REPLACE "\\\\" "/" _PYTHON_VALUES ${_PYTHON_VALUES})
        endif()
        string(REGEX REPLACE ";" "\\\\;" _PYTHON_VALUES ${_PYTHON_VALUES})
        string(REGEX REPLACE "\n" ";" _PYTHON_VALUES ${_PYTHON_VALUES})
        list(GET _PYTHON_VALUES 0 _PYTHON_VERSION_LIST)
        list(GET _PYTHON_VALUES 1 PYTHON_PREFIX)
        list(GET _PYTHON_VALUES 2 PYTHON_INCLUDE_DIR)
        list(GET _PYTHON_VALUES 3 PYTHON_SITE_PACKAGES)
        list(GET _PYTHON_VALUES 4 PYTHON_MODULE_EXTENSION)
        list(GET _PYTHON_VALUES 5 PYTHON_IS_DEBUG)
        list(GET _PYTHON_VALUES 6 PYTHON_SIZEOF_VOID_P)
        list(GET _PYTHON_VALUES 7 PYTHON_LIBRARY_SUFFIX)
        list(GET _PYTHON_VALUES 8 PYTHON_LIBDIR)
        list(GET _PYTHON_VALUES 9 PYTHON_MULTIARCH)
    else()
        message(WARNING "${_PYTHON_ERROR_VALUE}")
    endif()

    if("${PYTHON_MODULE_EXTENSION}" STREQUAL "")
        message(WARNING "Python module extension is empty!")
    endif()
endif()
