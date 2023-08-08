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

find_program(ROCPROFILER_CLANG_FORMAT_EXE NAMES clang-format-11 clang-format-mp-11)
find_program(ROCPROFILER_CMAKE_FORMAT_EXE NAMES cmake-format)
find_program(ROCPROFILER_BLACK_FORMAT_EXE NAMES black)

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

    if(ROCPROFILER_CLANG_FORMAT_EXE)
        add_custom_target(
            format-rocprofiler-source
            ${ROCPROFILER_CLANG_FORMAT_EXE} -i ${rocp_header_files} ${rocp_source_files}
            COMMENT
                "[rocprofiler] Running source formatter ${ROCPROFILER_CLANG_FORMAT_EXE}..."
            )
    endif()

    if(ROCPROFILER_BLACK_FORMAT_EXE)
        add_custom_target(
            format-rocprofiler-python
            ${ROCPROFILER_BLACK_FORMAT_EXE} -q ${rocp_python_files}
            COMMENT
                "[rocprofiler] Running Python formatter ${ROCPROFILER_BLACK_FORMAT_EXE}..."
            )
    endif()

    if(ROCPROFILER_CMAKE_FORMAT_EXE)
        add_custom_target(
            format-rocprofiler-cmake
            ${ROCPROFILER_CMAKE_FORMAT_EXE} -i ${rocp_cmake_files}
            COMMENT
                "[rocprofiler] Running CMake formatter ${ROCPROFILER_CMAKE_FORMAT_EXE}..."
            )
    endif()

    foreach(_TYPE source python cmake)
        if(TARGET format-rocprofiler-${_TYPE})
            add_dependencies(format-rocprofiler format-rocprofiler-${_TYPE})
            add_dependencies(format-${_TYPE} format-rocprofiler-${_TYPE})
        endif()
    endforeach()

    foreach(_TYPE source python)
        if(TARGET format-rocprofiler-${_TYPE})
            add_dependencies(format format-rocprofiler-${_TYPE})
        endif()
    endforeach()
endif()
