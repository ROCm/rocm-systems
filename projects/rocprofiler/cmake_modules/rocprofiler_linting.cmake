include_guard(DIRECTORY)

# ----------------------------------------------------------------------------------------#
#
# Clang Tidy
#
# ----------------------------------------------------------------------------------------#

if(ROCPROFILER_ENABLE_CLANG_TIDY)
    find_program(ROCPROFILER_CLANG_TIDY_COMMAND NAMES clang-tidy)

    if(NOT ROCPROFILER_CLANG_TIDY_COMMAND)
        message(
            WARNING "ROCPROFILER_ENABLE_CLANG_TIDY is ON but clang-tidy is not found!")
        set(ROCPROFILER_ENABLE_CLANG_TIDY OFF)
    else()
        set(CMAKE_CXX_CLANG_TIDY ${ROCPROFILER_CLANG_TIDY_COMMAND}
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
endif()
