include_guard(DIRECTORY)

# ----------------------------------------------------------------------------------------#
#
# Clang Tidy
#
# ----------------------------------------------------------------------------------------#

# clang-tidy
macro(ROCPROFILER_SYSTEMS_ACTIVATE_CLANG_TIDY)
    if(ROCPROFSYS_USE_CLANG_TIDY)
        find_program(CLANG_TIDY_COMMAND NAMES clang-tidy)
        rocprofiler_systems_add_feature(CLANG_TIDY_COMMAND "Path to clang-tidy command")
        if(NOT CLANG_TIDY_COMMAND)
            timemory_message(
                WARNING "ROCPROFSYS_USE_CLANG_TIDY is ON but clang-tidy is not found!")
            set(ROCPROFSYS_USE_CLANG_TIDY OFF)
        else()
            set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_COMMAND})

            # Create a preprocessor definition that depends on .clang-tidy content so the
            # compile command will change when .clang-tidy changes.  This ensures that a
            # subsequent build re-runs clang-tidy on all sources even if they do not
            # otherwise need to be recompiled.  Nothing actually uses this definition.  We
            # add it to targets on which we run clang-tidy just to get the build
            # dependency on the .clang-tidy file.
            file(SHA1 ${CMAKE_CURRENT_LIST_DIR}/.clang-tidy clang_tidy_sha1)
            set(CLANG_TIDY_DEFINITIONS "CLANG_TIDY_SHA1=${clang_tidy_sha1}")
            unset(clang_tidy_sha1)
        endif()
    endif()
endmacro()

# ------------------------------------------------------------------------------#
#
# clang-format target
#
# ------------------------------------------------------------------------------#

find_program(ROCPROFSYS_CLANG_FORMAT_EXE NAMES clang-format-18 clang-format)

find_program(ROCPROFSYS_CMAKE_FORMAT_EXE NAMES cmake-format)
find_program(ROCPROFSYS_BLACK_FORMAT_EXE NAMES black)

add_custom_target(format-rocprofiler-systems)
if(NOT TARGET format)
    add_custom_target(format)
endif()
foreach(_TYPE source python cmake)
    if(NOT TARGET format-${_TYPE})
        add_custom_target(format-${_TYPE})
    endif()
endforeach()

if(ROCPROFSYS_CLANG_FORMAT_EXE
   OR ROCPROFSYS_BLACK_FORMAT_EXE
   OR ROCPROFSYS_CMAKE_FORMAT_EXE)
    file(GLOB_RECURSE sources ${PROJECT_SOURCE_DIR}/source/*.cpp
         ${PROJECT_SOURCE_DIR}/source/*.c)
    file(GLOB_RECURSE headers ${PROJECT_SOURCE_DIR}/source/*.hpp
         ${PROJECT_SOURCE_DIR}/source/*.hpp.in ${PROJECT_SOURCE_DIR}/source/*.h
         ${PROJECT_SOURCE_DIR}/source/*.h.in)
    file(GLOB_RECURSE examples ${PROJECT_SOURCE_DIR}/examples/*.cpp
         ${PROJECT_SOURCE_DIR}/examples/*.c ${PROJECT_SOURCE_DIR}/examples/*.hpp
         ${PROJECT_SOURCE_DIR}/examples/*.h)
    file(GLOB_RECURSE tests_source ${PROJECT_SOURCE_DIR}/tests/source/*.cpp
         ${PROJECT_SOURCE_DIR}/tests/source/*.hpp)
    file(GLOB_RECURSE external ${PROJECT_SOURCE_DIR}/examples/lulesh/external/kokkos/*)
    file(
        GLOB_RECURSE
        cmake_files
        ${PROJECT_SOURCE_DIR}/source/*CMakeLists.txt
        ${PROJECT_SOURCE_DIR}/examples/*CMakeLists.txt
        ${PROJECT_SOURCE_DIR}/tests/*CMakeLists.txt
        ${PROJECT_SOURCE_DIR}/source/*.cmake
        ${PROJECT_SOURCE_DIR}/examples/*.cmake
        ${PROJECT_SOURCE_DIR}/tests/*.cmake
        ${PROJECT_SOURCE_DIR}/cmake/*.cmake
        ${PROJECT_SOURCE_DIR}/source/*.cmake)
    list(APPEND cmake_files ${PROJECT_SOURCE_DIR}/CMakeLists.txt)
    if(external)
        list(REMOVE_ITEM examples ${external})
        list(REMOVE_ITEM cmake_files ${external})
    endif()

    if(ROCPROFSYS_CLANG_FORMAT_EXE)
        add_custom_target(
            format-rocprofiler-systems-source
            ${ROCPROFSYS_CLANG_FORMAT_EXE} -i ${sources} ${headers} ${examples}
            ${tests_source}
            COMMENT
                "[rocprofiler-systems] Running C++ formatter ${ROCPROFSYS_CLANG_FORMAT_EXE}..."
            )
    endif()

    if(ROCPROFSYS_BLACK_FORMAT_EXE)
        add_custom_target(
            format-rocprofiler-systems-python
            ${ROCPROFSYS_BLACK_FORMAT_EXE} -q ${PROJECT_SOURCE_DIR}
            COMMENT
                "[rocprofiler-systems] Running Python formatter ${ROCPROFSYS_BLACK_FORMAT_EXE}..."
            )
        if(NOT TARGET format-python)
            add_custom_target(format-python)
        endif()
    endif()

    if(ROCPROFSYS_CMAKE_FORMAT_EXE)
        add_custom_target(
            format-rocprofiler-systems-cmake
            ${ROCPROFSYS_CMAKE_FORMAT_EXE} -i ${cmake_files}
            COMMENT
                "[rocprofiler-systems] Running CMake formatter ${ROCPROFSYS_CMAKE_FORMAT_EXE}..."
            )
        if(NOT TARGET format-cmake)
            add_custom_target(format-cmake)
        endif()
    endif()

    foreach(_TYPE source python cmake)
        if(TARGET format-rocprofiler-systems-${_TYPE})
            add_dependencies(format-rocprofiler-systems
                             format-rocprofiler-systems-${_TYPE})
            add_dependencies(format-${_TYPE} format-rocprofiler-systems-${_TYPE})
        endif()
    endforeach()

    foreach(_TYPE source python)
        if(TARGET format-rocprofiler-systems-${_TYPE})
            add_dependencies(format format-rocprofiler-systems-${_TYPE})
        endif()
    endforeach()
else()
    message(STATUS "clang-format could not be found. format build target not available.")
endif()
