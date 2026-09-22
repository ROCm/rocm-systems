# ------------------------------------------------------------------------------#
#
# creates following targets to format code:
# - format
# - format-source
# - format-cmake
# - format-python
# - format-rocprofiler-register-source
# - format-rocprofiler-register-cmake
# - format-rocprofiler-register-python
#
# ------------------------------------------------------------------------------#

include_guard(DIRECTORY)

if(NOT ROCPROFILER_REGISTER_CLANG_FORMAT_EXE AND EXISTS
                                                 $ENV{HOME}/.local/bin/clang-format)
    execute_process(
        COMMAND $ENV{HOME}/.local/bin/clang-format --version
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        OUTPUT_VARIABLE _CLANG_FMT_OUT
        RESULT_VARIABLE _CLANG_FMT_RET
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_CLANG_FMT_RET EQUAL 0)
        if("${_CLANG_FMT_OUT}" MATCHES "version 11\\.([0-9]+)\\.([0-9]+)")
            set(ROCPROFILER_REGISTER_CLANG_FORMAT_EXE
                "$ENV{HOME}/.local/bin/clang-format"
                CACHE FILEPATH "clang-format exe")
        endif()
    endif()
endif()

find_program(
    ROCPROFILER_REGISTER_CLANG_FORMAT_EXE
    NAMES clang-format-11 clang-format-mp-11 clang-format
    PATHS $ENV{HOME}/.local
    HINTS $ENV{HOME}/.local
    PATH_SUFFIXES bin)
find_program(
    ROCPROFILER_REGISTER_CMAKE_FORMAT_EXE
    NAMES cmake-format
    PATHS $ENV{HOME}/.local
    HINTS $ENV{HOME}/.local
    PATH_SUFFIXES bin)
find_program(
    ROCPROFILER_REGISTER_BLACK_FORMAT_EXE
    NAMES black
    PATHS $ENV{HOME}/.local
    HINTS $ENV{HOME}/.local
    PATH_SUFFIXES bin)

add_custom_target(format-rocprofiler-register)
if(NOT TARGET format)
    add_custom_target(format)
endif()
foreach(_TYPE source python cmake)
    if(NOT TARGET format-${_TYPE})
        add_custom_target(format-${_TYPE})
    endif()
endforeach()

if(ROCPROFILER_REGISTER_CLANG_FORMAT_EXE
   OR ROCPROFILER_REGISTER_BLACK_FORMAT_EXE
   OR ROCPROFILER_REGISTER_CMAKE_FORMAT_EXE)

    set(rocp_source_files)
    set(rocp_header_files)
    set(rocp_python_files)
    set(rocp_cmake_files ${PROJECT_SOURCE_DIR}/CMakeLists.txt
                         ${PROJECT_SOURCE_DIR}/external/CMakeLists.txt)

    foreach(_DIR cmake samples source tests)
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

    if(ROCPROFILER_REGISTER_CLANG_FORMAT_EXE AND (rocp_source_files OR rocp_header_files))
        add_custom_target(
            format-rocprofiler-register-source
            ${ROCPROFILER_REGISTER_CLANG_FORMAT_EXE} -i ${rocp_header_files}
            ${rocp_source_files}
            COMMENT
                "[rocprofiler-register] Running source formatter ${ROCPROFILER_REGISTER_CLANG_FORMAT_EXE}..."
            )
    endif()

    if(ROCPROFILER_REGISTER_BLACK_FORMAT_EXE AND rocp_python_files)
        add_custom_target(
            format-rocprofiler-register-python
            ${ROCPROFILER_REGISTER_BLACK_FORMAT_EXE} -q ${rocp_python_files}
            COMMENT
                "[rocprofiler-register] Running python formatter ${ROCPROFILER_REGISTER_BLACK_FORMAT_EXE}..."
            )
    endif()

    if(ROCPROFILER_REGISTER_CMAKE_FORMAT_EXE AND rocp_cmake_files)
        add_custom_target(
            format-rocprofiler-register-cmake
            ${ROCPROFILER_REGISTER_CMAKE_FORMAT_EXE} -i ${rocp_cmake_files}
            COMMENT
                "[rocprofiler-register] Running cmake formatter ${ROCPROFILER_REGISTER_CMAKE_FORMAT_EXE}..."
            )
    endif()

    foreach(_TYPE source python cmake)
        if(TARGET format-rocprofiler-register-${_TYPE})
            add_dependencies(format-rocprofiler-register
                             format-rocprofiler-register-${_TYPE})
            add_dependencies(format-${_TYPE} format-rocprofiler-register-${_TYPE})
        endif()
    endforeach()

    foreach(_TYPE source python cmake)
        if(TARGET format-rocprofiler-register-${_TYPE})
            add_dependencies(format format-rocprofiler-register-${_TYPE})
        endif()
    endforeach()
endif()
