#
#   Custom compilation of source code
#

include_guard(GLOBAL)

# ----------------------------------------------------------------------------------------#
# this provides a function to easily select which files use alternative compiler:
#
# * GLOBAL      --> all files
# * TARGET      --> all files in a target
# * SOURCE      --> specific source files
# * DIRECTORY   --> all files in directory
# * PROJECT     --> all files/targets in a project/subproject
#
function(rocprofiler_sdk_custom_compilation)
    cmake_parse_arguments(COMP "GLOBAL;PROJECT" "COMPILER" "DIRECTORY;TARGET;SOURCE"
                          ${ARGN})

    # find rocprofiler-sdk-launch-compiler
    find_program(
        ROCPROFILER_SDK_COMPILE_LAUNCHER
        NAMES rocprofiler-sdk-launch-compiler
        HINTS ${rocprofiler-sdk_ROOT_DIR} ${PROJECT_BINARY_DIR} ${CMAKE_BINARY_DIR}
        PATHS ${rocprofiler-sdk_ROOT_DIR} ${PROJECT_BINARY_DIR} ${CMAKE_BINARY_DIR}
        PATH_SUFFIXES libexec/rocprofiler-sdk)

    if(NOT COMP_COMPILER)
        message(
            FATAL_ERROR
                "rocprofiler_sdk_custom_compilation not provided COMPILER argument")
    elseif(NOT EXISTS "${COMP_COMPILER}")
        cmake_path(GET COMP_COMPILER FILENAME COMP_COMPILER_BASE)
        find_program(
            ROCPROFILER_SDK_COMPILER_${COMP_COMPILER_BASE}
            NAMES ${COMP_COMPILER_BASE}
            PATH_SUFFIXES bin REQUIRED)
        set(COMP_COMPILER ROCPROFILER_SDK_COMPILER_${COMP_COMPILER_BASE})
    endif()

    if(NOT ROCPROFILER_SDK_COMPILE_LAUNCHER)
        message(
            FATAL_ERROR
                "rocprofiler-sdk could not find 'rocprofiler-sdk-launch-compiler'. Please set '-DROCPROFILER_SDK_COMPILE_LAUNCHER=/path/to/launcher'"
            )
    endif()

    if(COMP_GLOBAL)
        # if global, don't bother setting others
        set_property(
            GLOBAL
            PROPERTY
                RULE_LAUNCH_COMPILE
                "${ROCPROFILER_SDK_COMPILE_LAUNCHER} ${COMP_COMPILER} ${CMAKE_CXX_COMPILER}"
            )
        set_property(
            GLOBAL
            PROPERTY
                RULE_LAUNCH_LINK
                "${ROCPROFILER_SDK_COMPILE_LAUNCHER} ${COMP_COMPILER} ${CMAKE_CXX_COMPILER}"
            )
    else()
        foreach(_TYPE PROJECT DIRECTORY TARGET SOURCE)
            # make project/subproject scoping easy, e.g.
            # rocprofiler_sdk_custom_compilation(PROJECT) after project(...)
            if("${_TYPE}" STREQUAL "PROJECT" AND COMP_${_TYPE})
                list(APPEND COMP_DIRECTORY ${PROJECT_SOURCE_DIR})
                unset(COMP_${_TYPE})
            endif()

            # set the properties if defined
            if(COMP_${_TYPE})
                foreach(_VAL ${COMP_${_TYPE}})
                    set_property(
                        ${_TYPE} ${_VAL}
                        PROPERTY
                            RULE_LAUNCH_COMPILE
                            "${ROCPROFILER_SDK_COMPILE_LAUNCHER} ${COMP_COMPILER} ${CMAKE_CXX_COMPILER}"
                        )
                    set_property(
                        ${_TYPE} ${_VAL}
                        PROPERTY
                            RULE_LAUNCH_LINK
                            "${ROCPROFILER_SDK_COMPILE_LAUNCHER} ${COMP_COMPILER} ${CMAKE_CXX_COMPILER}"
                        )
                endforeach()
            endif()
        endforeach()
    endif()
endfunction()
