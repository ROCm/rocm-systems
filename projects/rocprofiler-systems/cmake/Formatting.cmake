include_guard(DIRECTORY)

# ----------------------------------------------------------------------------------------#
#
# Clang Tidy
#
# ----------------------------------------------------------------------------------------#

# clang-tidy
macro(OMNITRACE_ACTIVATE_CLANG_TIDY)
    if(OMNITRACE_USE_CLANG_TIDY)
        find_program(CLANG_TIDY_COMMAND NAMES clang-tidy)
        omnitrace_add_feature(CLANG_TIDY_COMMAND "Path to clang-tidy command")
        if(NOT CLANG_TIDY_COMMAND)
            timemory_message(
                WARNING "OMNITRACE_USE_CLANG_TIDY is ON but clang-tidy is not found!")
            set(OMNITRACE_USE_CLANG_TIDY OFF)
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

find_program(OMNITRACE_CLANG_FORMAT_EXE NAMES clang-format-11 clang-format-mp-11
                                              clang-format)

find_program(OMNITRACE_CMAKE_FORMAT_EXE NAMES cmake-format)

if(OMNITRACE_CLANG_FORMAT_EXE)
    file(GLOB_RECURSE sources ${PROJECT_SOURCE_DIR}/source/*.cpp)
    file(GLOB_RECURSE headers ${PROJECT_SOURCE_DIR}/source/*.hpp
         ${PROJECT_SOURCE_DIR}/source/*.hpp.in ${PROJECT_SOURCE_DIR}/source/*.h
         ${PROJECT_SOURCE_DIR}/source/*.h.in)
    file(GLOB_RECURSE examples ${PROJECT_SOURCE_DIR}/examples/*.cpp
         ${PROJECT_SOURCE_DIR}/examples/*.hpp)
    file(GLOB_RECURSE external ${PROJECT_SOURCE_DIR}/examples/lulesh/external/kokkos/*)
    file(GLOB_RECURSE cmake_files ${PROJECT_SOURCE_DIR}/source/*CMakeLists.txt
         ${PROJECT_SOURCE_DIR}/examples/*CMakeLists.txt
         ${PROJECT_SOURCE_DIR}/tests/*CMakeLists.txt ${PROJECT_SOURCE_DIR}/cmake/*.cmake)
    list(APPEND cmake_files ${PROJECT_SOURCE_DIR}/CMakeLists.txt)
    if(external)
        list(REMOVE_ITEM examples ${external})
        list(REMOVE_ITEM cmake_files ${external})
    endif()
    add_custom_target(
        format-omnitrace-source
        ${OMNITRACE_CLANG_FORMAT_EXE} -i ${sources} ${headers} ${examples}
        COMMENT "[omnitrace] Running C++ formatter ${OMNITRACE_CLANG_FORMAT_EXE}...")
    add_custom_target(format-omnitrace)
    add_dependencies(format-omnitrace format-omnitrace-source)
    if(NOT TARGET format)
        add_custom_target(format)
    endif()
    add_dependencies(format format-omnitrace)
    if(OMNITRACE_CMAKE_FORMAT_EXE)
        add_custom_target(
            format-omnitrace-cmake
            ${OMNITRACE_CMAKE_FORMAT_EXE} -i ${cmake_files}
            COMMENT "[omnitrace] Running CMake formatter ${OMNITRACE_CMAKE_FORMAT_EXE}..."
            )
        if(NOT TARGET format-cmake)
            add_custom_target(format-cmake)
        endif()
        add_dependencies(format-cmake format-omnitrace-cmake)
    endif()
else()
    message(
        AUTHOR_WARNING
            "clang-format could not be found. format build target not available.")
endif()
