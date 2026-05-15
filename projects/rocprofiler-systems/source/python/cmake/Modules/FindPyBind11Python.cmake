# * Find python libraries This module finds the libraries corresponding to the Python
#   interpreter FindPythonInterp provides. This code sets the following variables:
#
# PYTHONLIBS_FOUND           - have the Python libs been found PYTHON_PREFIX - path to the
# Python installation PYTHON_LIBRARIES           - path to the python library
# PYTHON_INCLUDE_DIRS        - path to where Python.h is found PYTHON_MODULE_EXTENSION -
# lib extension, e.g. '.so' or '.pyd' PYTHON_MODULE_PREFIX - lib name prefix: usually an
# empty string PYTHON_SITE_PACKAGES       - path to installation site-packages
# PYTHON_IS_DEBUG            - whether the Python interpreter is a debug build
#
# Thanks to talljimbo for the patch adding the 'LDVERSION' config variable usage.

# =============================================================================
# Copyright 2001-2009 Kitware, Inc. Copyright 2012 Continuum Analytics, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list of
#   conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this list of
#   conditions and the following disclaimer in the documentation and/or other materials
#   provided with the distribution.
#
# * Neither the names of Kitware, Inc., the Insight Software Consortium, nor the names of
#   their contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR # A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

# Checking for the extension makes sure that `LibsNew` was found and not just `Libs`.

set(_find_quiet)
set(_find_required)
set(_find_exact)

if(PyBind11Python_FIND_QUIETLY)
    set(_find_quiet QUIET)
endif()

if(PyBind11Python_FIND_REQUIRED)
    set(_find_required REQUIRED)
endif()

if(PyBind11Python_FIND_VERSION)
    set(_find_exact EXACT)
endif()

if(NOT DEFINED PyBind11Python_PYTHON)
    set(PyBind11Python_PYTHON Python3)
    if(PyBind11Python_FIND_VERSION AND "${PyBind11Python_FIND_VERSION}" VERSION_LESS 3.0)
        set(PyBind11Python_PYTHON Python2)
    endif()
endif()

if(NOT PyBind11Python_COMPONENTS)
    set(PyBind11Python_COMPONENTS Interpreter Development)
endif()

# Use the Python interpreter to find the libs.
find_package(
    ${PyBind11Python_PYTHON}
    ${PyBind11Python_FIND_VERSION}
    ${_find_exact}
    MODULE
    ${_find_required}
    ${_find_quiet}
    COMPONENTS ${PyBind11Python_COMPONENTS}
)

# According to
# https://stackoverflow.com/questions/646518/python-how-to-detect-debug-interpreter
# testing whether sys has the gettotalrefcount function is a reliable, cross-platform way
# to detect a CPython debug interpreter.
#
# The library suffix is from the config var LDVERSION sometimes, otherwise VERSION.
# VERSION will typically be like "2.7" on unix, and "27" on windows.
execute_process(
    COMMAND
        "${${PyBind11Python_PYTHON}_EXECUTABLE}" "-c"
        "
import sys;import struct;
import sysconfig as s
USE_SYSCONFIG = sys.version_info >= (3, 10)
if not USE_SYSCONFIG:
    from distutils import sysconfig as ds
print('.'.join(str(v) for v in sys.version_info));
print(sys.prefix);
if USE_SYSCONFIG:
    scheme = s.get_default_scheme()
    if scheme == 'posix_local':
        # Debian's default scheme installs to /usr/local/ but we want to find headers in /usr/
        scheme = 'posix_prefix'
    print(s.get_path('platinclude', scheme))
    print(s.get_path('platlib'))
else:
    print(ds.get_python_inc(plat_specific=True));
    print(ds.get_python_lib(plat_specific=True));
print(s.get_config_var('EXT_SUFFIX') or s.get_config_var('SO'));
print(hasattr(sys, 'gettotalrefcount')+0);
print(struct.calcsize('@P'));
print(s.get_config_var('LDVERSION') or s.get_config_var('VERSION'));
print(s.get_config_var('LIBDIR') or '');
print(s.get_config_var('MULTIARCH') or '');
"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE _PYTHON_VALUES
    ERROR_VARIABLE _PYTHON_ERROR_VALUE
)

if(NOT _PYTHON_SUCCESS MATCHES 0)
    if(PyBind11Python_FIND_REQUIRED)
        message(FATAL_ERROR "Python config failure:\n${_PYTHON_ERROR_VALUE}")
    endif()
    set(PyBind11Python_FOUND FALSE)
    return()
endif()

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

# Make sure the Python has the same pointer-size as the chosen compiler Skip if
# CMAKE_SIZEOF_VOID_P is not defined
if(
    CMAKE_SIZEOF_VOID_P
    AND (NOT "${PYTHON_SIZEOF_VOID_P}" STREQUAL "${CMAKE_SIZEOF_VOID_P}")
)
    if(PyBind11Python_FIND_REQUIRED)
        math(EXPR _PYTHON_BITS "${PYTHON_SIZEOF_VOID_P} * 8")
        math(EXPR _CMAKE_BITS "${CMAKE_SIZEOF_VOID_P} * 8")
        message(
            FATAL_ERROR
            "Python config failure: Python is ${_PYTHON_BITS}-bit, "
            "chosen compiler is  ${_CMAKE_BITS}-bit"
        )
    endif()
    set(PyBind11Python_FOUND FALSE)
    return()
endif()

# The built-in FindPython didn't always give the version numbers
string(REGEX REPLACE "\\." ";" _PYTHON_VERSION_LIST ${_PYTHON_VERSION_LIST})
list(GET _PYTHON_VERSION_LIST 0 PYTHON_VERSION_MAJOR)
list(GET _PYTHON_VERSION_LIST 1 PYTHON_VERSION_MINOR)
list(GET _PYTHON_VERSION_LIST 2 PYTHON_VERSION_PATCH)
set(PYTHON_VERSION
    "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.${PYTHON_VERSION_PATCH}"
)

# Make sure all directory separators are '/'
string(REGEX REPLACE "\\\\" "/" PYTHON_PREFIX "${PYTHON_PREFIX}")
string(REGEX REPLACE "\\\\" "/" PYTHON_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
string(REGEX REPLACE "\\\\" "/" PYTHON_SITE_PACKAGES "${PYTHON_SITE_PACKAGES}")

# We use PYTHON_INCLUDE_DIR, PYTHON_LIBRARY and PYTHON_DEBUG_LIBRARY for the cache entries
# because they are meant to specify the location of a single library. We now set the
# variables listed by the documentation for this module.
set(PYTHON_EXECUTABLE "${${PyBind11Python_PYTHON}_EXECUTABLE}")
set(PYTHON_INCLUDE_DIRS "${${PyBind11Python_PYTHON}_INCLUDE_DIRS}")
set(PYTHON_LIBRARIES "${${PyBind11Python_PYTHON}_LIBRARIES}")
if(NOT PYTHON_DEBUG_LIBRARY)
    set(PYTHON_DEBUG_LIBRARY "")
endif()
set(PYTHON_DEBUG_LIBRARIES "${PYTHON_DEBUG_LIBRARY}")

# find_package_message(PyBind11Python "Found PyBind11Python: ${PYTHON_LIBRARIES}"
# "${PYTHON_EXECUTABLE}${PYTHON_VERSION_STRING}")

if(NOT PYTHON_MODULE_PREFIX)
    set(PYTHON_MODULE_PREFIX "")
endif()
