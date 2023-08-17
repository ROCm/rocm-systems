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

find_program(ROCPROFILER_REGISTER_CLANG_FORMAT_EXE NAMES clang-format-11
                                                         clang-format-mp-11)
find_program(ROCPROFILER_REGISTER_CMAKE_FORMAT_EXE NAMES cmake-format)
find_program(ROCPROFILER_REGISTER_BLACK_FORMAT_EXE NAMES black)

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
    set(rocp_cmake_files ${PROJECT_SOURCE_DIR}/CMakeLists.txt)

    foreach(_DIR cmake samples source tests)
        foreach(_TYPE header_files source_files cmake_files python_files)
            set(${_TYPE})
        endforeach()
        file(GLOB_RECURSE header_files ${PROJECT_SOURCE_DIR}/${_DIR}/*.h
             ${PROJECT_SOURCE_DIR}/${_DIR}/*.hpp)
        file(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/${_DIR}/*.c
             ${PROJECT_SOURCE_DIR}/${_DIR}/*.cpp)
        file(GLOB_RECURSE cmake_files ${PROJECT_SOURCE_DIR}/${_DIR}/*CMakeLists.txt
             ${PROJECT_SOURCE_DIR}/${_DIR}/*.cmake)
        file(GLOB_RECURSE python_files ${PROJECT_SOURCE_DIR}/${_DIR}/*.py)
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
                "[rocprofiler-register] Running Python formatter ${ROCPROFILER_REGISTER_BLACK_FORMAT_EXE}..."
            )
    endif()

    if(ROCPROFILER_REGISTER_CMAKE_FORMAT_EXE AND rocp_cmake_files)
        add_custom_target(
            format-rocprofiler-register-cmake
            ${ROCPROFILER_REGISTER_CMAKE_FORMAT_EXE} -i ${rocp_cmake_files}
            COMMENT
                "[rocprofiler-register] Running CMake formatter ${ROCPROFILER_REGISTER_CMAKE_FORMAT_EXE}..."
            )
    endif()

    foreach(_TYPE source python cmake)
        if(TARGET format-rocprofiler-register-${_TYPE})
            add_dependencies(format-rocprofiler-register
                             format-rocprofiler-register-${_TYPE})
            add_dependencies(format-${_TYPE} format-rocprofiler-register-${_TYPE})
        endif()
    endforeach()

    foreach(_TYPE source python)
        if(TARGET format-rocprofiler-register-${_TYPE})
            add_dependencies(format format-rocprofiler-register-${_TYPE})
        endif()
    endforeach()
endif()
