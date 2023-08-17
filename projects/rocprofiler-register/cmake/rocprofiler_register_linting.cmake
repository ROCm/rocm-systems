include_guard(GLOBAL)

# ----------------------------------------------------------------------------------------#
#
# Clang Tidy
#
# ----------------------------------------------------------------------------------------#

find_program(
    ROCPROFILER_REGISTER_CLANG_TIDY_COMMAND
    NAMES clang-tidy-18
          clang-tidy-17
          clang-tidy-16
          clang-tidy-15
          clang-tidy-14
          clang-tidy-13
          clang-tidy-12
          clang-tidy-11
          clang-tidy)

macro(ROCPROFILER_REGISTER_ACTIVATE_CLANG_TIDY)
    if(ROCPROFILER_REGISTER_ENABLE_CLANG_TIDY)
        if(NOT ROCPROFILER_REGISTER_CLANG_TIDY_COMMAND)
            message(
                FATAL_ERROR
                    "ROCPROFILER_REGISTER_ENABLE_CLANG_TIDY is ON but clang-tidy is not found!"
                )
        endif()

        set(CMAKE_CXX_CLANG_TIDY ${ROCPROFILER_REGISTER_CLANG_TIDY_COMMAND}
                                 -header-filter=${PROJECT_SOURCE_DIR}/.*)

        # Create a preprocessor definition that depends on .clang-tidy content so the
        # compile command will change when .clang-tidy changes.  This ensures that a
        # subsequent build re-runs clang-tidy on all sources even if they do not otherwise
        # need to be recompiled.  Nothing actually uses this definition.  We add it to
        # targets on which we run clang-tidy just to get the build dependency on the
        # .clang-tidy file.
        file(SHA1 ${PROJECT_SOURCE_DIR}/.clang-tidy clang_tidy_sha1)
        set(CLANG_TIDY_DEFINITIONS "CLANG_TIDY_SHA1=${clang_tidy_sha1}")
        unset(clang_tidy_sha1)
    endif()
endmacro()

macro(ROCPROFILER_REGISTER_DEACTIVATE_CLANG_TIDY)
    set(CMAKE_CXX_CLANG_TIDY)
endmacro()
