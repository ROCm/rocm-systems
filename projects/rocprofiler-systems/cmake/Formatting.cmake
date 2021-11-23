include_guard(DIRECTORY)

# ----------------------------------------------------------------------------------------#
#
# Clang Tidy
#
# ----------------------------------------------------------------------------------------#

# clang-tidy
macro(HOSTTRACE_ACTIVATE_CLANG_TIDY)
    if(HOSTTRACE_USE_CLANG_TIDY)
        find_program(CLANG_TIDY_COMMAND NAMES clang-tidy)
        hosttrace_add_feature(CLANG_TIDY_COMMAND "Path to clang-tidy command")
        if(NOT CLANG_TIDY_COMMAND)
            timemory_message(
                WARNING "HOSTTRACE_USE_CLANG_TIDY is ON but clang-tidy is not found!")
            set(HOSTTRACE_USE_CLANG_TIDY OFF)
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

find_program(HOSTTRACE_CLANG_FORMAT_EXE NAMES clang-format-11 clang-format-mp-11
                                              clang-format)

if(HOSTTRACE_CLANG_FORMAT_EXE)
    file(GLOB_RECURSE sources ${PROJECT_SOURCE_DIR}/src/*.cpp)
    file(GLOB_RECURSE headers ${PROJECT_SOURCE_DIR}/include/*.hpp)
    file(GLOB_RECURSE examples ${PROJECT_SOURCE_DIR}/examples/*.cpp
         ${PROJECT_SOURCE_DIR}/examples/*.hpp)
    add_custom_target(
        format-hosttrace
        ${HOSTTRACE_CLANG_FORMAT_EXE} -i ${sources} ${headers} ${examples}
        COMMENT "Running C++ formatter ${HOSTTRACE_CLANG_FORMAT_EXE}...")
    if(NOT TARGET format)
        add_custom_target(format)
    endif()
    add_dependencies(format format-hosttrace)
else()
    message(
        AUTHOR_WARNING
            "clang-format could not be found. format build target not available.")
endif()
