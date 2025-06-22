# Python configuration
#

# include guard
include_guard(DIRECTORY)

# Stops lookup as soon as a version satisfying version constraints is found.
set(Python3_FIND_STRATEGY
    "LOCATION"
    CACHE STRING
    "Stops lookup as soon as a version satisfying version constraints is found"
)

# virtual environment is used before any other standard paths to look-up for the
# interpreter
set(Python3_FIND_VIRTUALENV
    "FIRST"
    CACHE STRING
    "Virtual environment is used before any other standard paths"
)
set_property(CACHE Python3_FIND_VIRTUALENV PROPERTY STRINGS "FIRST;LAST;NEVER")

if(APPLE)
    set(Python3_FIND_FRAMEWORK
        "LAST"
        CACHE STRING
        "Order of preference between Apple-style and unix-style package components"
    )
    set_property(CACHE Python3_FIND_FRAMEWORK PROPERTY STRINGS "FIRST;LAST;NEVER")
endif()

# PyPy does not support embedding the interpreter
set(Python3_FIND_IMPLEMENTATIONS
    "CPython"
    CACHE STRING
    "Different implementations which will be searched."
)
set_property(
    CACHE Python3_FIND_IMPLEMENTATIONS
    PROPERTY STRINGS "CPython;IronPython;PyPy"
)

# variable is a 3-tuple specifying, in order, pydebug (d), pymalloc (m) and unicode (u)
# set(Python3_FIND_ABI "OFF" "OFF" "OFF" CACHE STRING "variable is a 3-tuple specifying
# pydebug (d), pymalloc (m) and unicode (u)")

# Create CMake cache entries for the above artifact specification variables so that users
# can edit them interactively. This disables support for multiple version/component
# requirements.
set(Python3_ARTIFACTS_INTERACTIVE
    OFF
    CACHE BOOL
    "Create CMake cache entries so that users can edit them interactively"
    FORCE
)

# if("${Python3_USE_STATIC_LIBS}" STREQUAL "ANY") set(Python3_USE_STATIC_LIBS "OFF" CACHE
# STRING "If ON, only static libs; if OFF, only shared libs; if ANY, shared then static")
# set_property(CACHE Python3_USE_STATIC_LIBS PROPERTY STRINGS "ON;OFF;ANY") else()
# unset(Python3_USE_STATIC_LIBS) endif()

foreach(
    _VAR
    FIND_STRATEGY
    FIND_VIRTUALENV
    FIND_FRAMEWORK
    FIND_IMPLEMENTATIONS
    ARTIFACTS_INTERACTIVE
)
    if(DEFINED Python3_${_VAR})
        set(Python_${_VAR}
            "${Python3_${_VAR}}"
            CACHE STRING
            "Set via Python3_${_VAR} setting (rocprofsys)"
        )
        mark_as_advanced(Python_${_VAR})
        mark_as_advanced(Python3_${_VAR})
    endif()
endforeach()

# display version
rocprofiler_systems_add_feature(ROCPROFSYS_PYTHON_VERSIONS
                                "Python version for rocprofsys" DOC
)

option(PYBIND11_INSTALL "Enable Pybind11 installation" OFF)

if(ROCPROFSYS_BUILD_PYTHON AND NOT TARGET pybind11)
    # checkout PyBind11 if not checked out
    rocprofiler_systems_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/pybind11
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/pybind11.git
        REPO_BRANCH omnitrace
    )

    if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()
    set(PYBIND11_NOPYTHON ON)
    rocprofiler_systems_save_variables(IPO VARIABLES CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    add_subdirectory(${PROJECT_SOURCE_DIR}/external/pybind11)
    rocprofiler_systems_restore_variables(IPO
                                          VARIABLES CMAKE_INTERPROCEDURAL_OPTIMIZATION
    )
endif()

execute_process(
    COMMAND
        ${PYTHON_EXECUTABLE} -c
        "import time ; print('{} {}'.format(time.ctime(), time.tzname[0]))"
    OUTPUT_VARIABLE ROCPROFSYS_INSTALL_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

string(REPLACE "  " " " ROCPROFSYS_INSTALL_DATE "${ROCPROFSYS_INSTALL_DATE}")
