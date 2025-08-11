include_guard(GLOBAL)

# ----------------------------------------------------------------------------------------#
#
# Clang Tidy
#
# ----------------------------------------------------------------------------------------#

if(ROCPROFILER_BUILD_DEVELOPER)
    set(_TIDY_REQUIRED REQUIRED)
else()
    set(_TIDY_REQUIRED)
endif()

if(NOT ROCPROFILER_CLANG_TIDY_EXE AND EXISTS $ENV{HOME}/.local/bin/clang-tidy)
    execute_process(
        COMMAND $ENV{HOME}/.local/bin/clang-tidy --version
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        OUTPUT_VARIABLE _CLANG_TIDY_OUT
        RESULT_VARIABLE _CLANG_TIDY_RET
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

    if(_CLANG_TIDY_RET EQUAL 0)
        if("${_CLANG_TIDY_OUT}" MATCHES "version 1[5-9]\\.([0-9]+)\\.([0-9]+)")
            set(ROCPROFILER_CLANG_TIDY_EXE
                "$ENV{HOME}/.local/bin/clang-tidy"
                CACHE FILEPATH "clang-tidy exe")
        endif()
    endif()
endif()

find_program(
    ROCPROFILER_CLANG_TIDY_EXE ${_TIDY_REQUIRED}
    NAMES clang-tidy-18
          clang-tidy-17
          clang-tidy-16
          clang-tidy-15
          clang-tidy-14
          clang-tidy-13
          clang-tidy-12
          clang-tidy-11
          clang-tidy
    PATHS $ENV{HOME}/.local
    HINTS $ENV{HOME}/.local
    PATH_SUFFIXES bin)

macro(ROCPROFILER_ACTIVATE_CLANG_TIDY)
    if(ROCPROFILER_ENABLE_CLANG_TIDY)
        if(NOT ROCPROFILER_CLANG_TIDY_EXE)
            message(
                FATAL_ERROR
                    "ROCPROFILER_ENABLE_CLANG_TIDY is ON but clang-tidy is not found!")
        endif()

        rocprofiler_add_feature(ROCPROFILER_CLANG_TIDY_EXE
                                "path to clang-tidy executable")

        set(CMAKE_CXX_CLANG_TIDY
            ${ROCPROFILER_CLANG_TIDY_EXE} -header-filter=${PROJECT_SOURCE_DIR}/source/.*
            --warnings-as-errors=*,-misc-header-include-cycle)

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

macro(ROCPROFILER_DEACTIVATE_CLANG_TIDY)
    set(CMAKE_CXX_CLANG_TIDY)
endmacro()
