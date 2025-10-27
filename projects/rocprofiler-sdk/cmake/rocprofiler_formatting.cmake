# ------------------------------------------------------------------------------#
#
# creates following targets to format code:
# - format
# - format-source
# - format-cmake
# - format-python
# - format-rocprofiler-source
# - format-rocprofiler-cmake
# - format-rocprofiler-python
#
# ------------------------------------------------------------------------------#

include_guard(DIRECTORY)

include(rocprofiler_utilities)

if(ROCPROFILER_BUILD_DEVELOPER)
    set(_FMT_REQUIRED REQUIRED)
else()
    set(_FMT_REQUIRED)
endif()

# checks that clang-format is version 11.x.x
function(_rocprofiler_check_clang_format_version _OUT _EXE)
    execute_process(
        COMMAND ${_EXE} --version
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        OUTPUT_VARIABLE _CLANG_FMT_OUT
        RESULT_VARIABLE _CLANG_FMT_RET
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_CLANG_FMT_RET EQUAL 0 AND "${_CLANG_FMT_OUT}" MATCHES
                                  "version 11\\.([0-9]+)\\.([0-9]+)")
        set(${_OUT}
            ON
            PARENT_SCOPE)
    else()
        set(${_OUT}
            OFF
            PARENT_SCOPE)
    endif()
endfunction()

_rocprofiler_get_python_user_bin(_PYTHON_USER_BIN)
if(NOT ROCPROFILER_CLANG_FORMAT_EXE
   AND _PYTHON_USER_BIN
   AND EXISTS "${_PYTHON_USER_BIN}/clang-format")
    _rocprofiler_check_clang_format_version(_IS_VALID_CLANG_FMT
                                            "${_PYTHON_USER_BIN}/clang-format")
    if(_IS_VALID_CLANG_FMT)
        set(ROCPROFILER_CLANG_FORMAT_EXE
            "${_PYTHON_USER_BIN}/clang-format"
            CACHE FILEPATH "clang-format exe")
    endif()
endif()

if(NOT ROCPROFILER_CMAKE_FORMAT_EXE
   AND _PYTHON_USER_BIN
   AND EXISTS "${_PYTHON_USER_BIN}/cmake-format")
    set(ROCPROFILER_CMAKE_FORMAT_EXE
        "${_PYTHON_USER_BIN}/cmake-format"
        CACHE FILEPATH "cmake-format exe")
endif()

if(NOT ROCPROFILER_BLACK_FORMAT_EXE
   AND _PYTHON_USER_BIN
   AND EXISTS "${_PYTHON_USER_BIN}/black")
    set(ROCPROFILER_BLACK_FORMAT_EXE
        "${_PYTHON_USER_BIN}/black"
        CACHE FILEPATH "black exe")
endif()

find_program(
    ROCPROFILER_CLANG_FORMAT_EXE ${_FMT_REQUIRED}
    NAMES clang-format-11 clang-format-mp-11 clang-format
    PATHS ${_PYTHON_USER_BIN}
    HINTS ${_PYTHON_USER_BIN}
    PATH_SUFFIXES bin)
find_program(
    ROCPROFILER_CMAKE_FORMAT_EXE ${_FMT_REQUIRED}
    NAMES cmake-format
    PATHS ${_PYTHON_USER_BIN}
    HINTS ${_PYTHON_USER_BIN}
    PATH_SUFFIXES bin)
find_program(
    ROCPROFILER_BLACK_FORMAT_EXE ${_FMT_REQUIRED}
    NAMES black
    PATHS ${_PYTHON_USER_BIN}
    HINTS ${_PYTHON_USER_BIN}
    PATH_SUFFIXES bin)

_rocprofiler_check_clang_format_version(_IS_VALID_CLANG_FMT
                                        "${ROCPROFILER_CLANG_FORMAT_EXE}")
if(NOT _IS_VALID_CLANG_FMT)
    if(ROCPROFILER_BUILD_DEVELOPER)
        message(
            AUTHOR_WARNING
                "[rocprofiler] clang-format version 11 not found. Please see rocprofiler-sdk CONTRIBUTING.md for instructions on installing clang-format version 11."
            )
    endif()
    unset(ROCPROFILER_CLANG_FORMAT_EXE CACHE)
endif()

add_custom_target(format-rocprofiler)
if(NOT TARGET format)
    add_custom_target(format)
endif()
foreach(_TYPE source python cmake)
    if(NOT TARGET format-${_TYPE})
        add_custom_target(format-${_TYPE})
    endif()
endforeach()

if(ROCPROFILER_CLANG_FORMAT_EXE
   OR ROCPROFILER_BLACK_FORMAT_EXE
   OR ROCPROFILER_CMAKE_FORMAT_EXE)

    set(rocp_source_files)
    set(rocp_header_files)
    set(rocp_python_files)
    set(rocp_cmake_files ${PROJECT_SOURCE_DIR}/CMakeLists.txt
                         ${PROJECT_SOURCE_DIR}/external/CMakeLists.txt)

    foreach(_DIR cmake samples source tests benchmark)
        foreach(_TYPE header_files source_files cmake_files python_files)
            set(${_TYPE})
        endforeach()
        file(GLOB_RECURSE header_files ${PROJECT_SOURCE_DIR}/${_DIR}/*.h
             ${PROJECT_SOURCE_DIR}/${_DIR}/*.hpp ${PROJECT_SOURCE_DIR}/${_DIR}/*.h.in
             ${PROJECT_SOURCE_DIR}/${_DIR}/*.hpp.in)
        file(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/${_DIR}/*.c
             ${PROJECT_SOURCE_DIR}/${_DIR}/*.cpp)
        file(GLOB_RECURSE cmake_files ${PROJECT_SOURCE_DIR}/${_DIR}/*CMakeLists.txt
             ${PROJECT_SOURCE_DIR}/${_DIR}/*.cmake)
        file(GLOB_RECURSE python_files ${PROJECT_SOURCE_DIR}/${_DIR}/*.py
             ${PROJECT_SOURCE_DIR}/${_DIR}/*.py.in)
        foreach(_TYPE header_files source_files cmake_files python_files)
            list(APPEND rocp_${_TYPE} ${${_TYPE}})
        endforeach()
    endforeach()

    foreach(_TYPE header_files source_files cmake_files python_files)
        if(rocp_${_TYPE})
            list(REMOVE_DUPLICATES rocp_${_TYPE})
            list(SORT rocp_${_TYPE})
        endif()
    endforeach()

    if(ROCPROFILER_CLANG_FORMAT_EXE)
        add_custom_target(
            format-rocprofiler-source
            ${ROCPROFILER_CLANG_FORMAT_EXE} -i ${rocp_header_files} ${rocp_source_files}
            COMMENT
                "[rocprofiler] Running source formatter ${ROCPROFILER_CLANG_FORMAT_EXE}..."
            )
    endif()

    if(ROCPROFILER_BLACK_FORMAT_EXE AND rocp_python_files)
        add_custom_target(
            format-rocprofiler-python
            ${ROCPROFILER_BLACK_FORMAT_EXE} -q ${rocp_python_files}
            COMMENT
                "[rocprofiler] Running python formatter ${ROCPROFILER_BLACK_FORMAT_EXE}..."
            )
    endif()

    if(ROCPROFILER_CMAKE_FORMAT_EXE)
        add_custom_target(
            format-rocprofiler-cmake
            ${ROCPROFILER_CMAKE_FORMAT_EXE} -i ${rocp_cmake_files}
            COMMENT
                "[rocprofiler] Running cmake formatter ${ROCPROFILER_CMAKE_FORMAT_EXE}..."
            )
    endif()

    foreach(_TYPE source python cmake)
        if(TARGET format-rocprofiler-${_TYPE})
            add_dependencies(format-rocprofiler format-rocprofiler-${_TYPE})
            add_dependencies(format-${_TYPE} format-rocprofiler-${_TYPE})
        endif()
    endforeach()

    foreach(_TYPE source python cmake)
        if(TARGET format-rocprofiler-${_TYPE})
            add_dependencies(format format-rocprofiler-${_TYPE})
        endif()
    endforeach()
endif()
