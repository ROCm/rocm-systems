# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#        Compilers
#
##########################################################################################
#
#   sets (cached):
#
#       CMAKE_C_COMPILER_IS_<TYPE>
#       CMAKE_CXX_COMPILER_IS_<TYPE>
#
#   where TYPE is:
#       - GNU
#       - CLANG
#       - INTEL
#       - INTEL_ICC
#       - INTEL_ICPC
#       - PGI
#       - XLC
#       - HP_ACC
#       - MIPS
#       - MSVC
#

include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
include(CheckCSourceRuns)

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)

include(CMakeParseArguments)

include(MacroUtilities)

if("${LIBNAME}" STREQUAL "")
    string(TOLOWER "${PROJECT_NAME}" LIBNAME)
endif()

add_interface_library(${LIBNAME}-compile-options
    "Adds the standard set of compiler flags used by timemory")

#----------------------------------------------------------------------------------------#
# macro converting string to list
#----------------------------------------------------------------------------------------#
macro(to_list _VAR _STR)
    STRING(REPLACE "  " " " ${_VAR} "${_STR}")
    STRING(REPLACE " " ";" ${_VAR} "${_STR}")
endmacro(to_list _VAR _STR)


#----------------------------------------------------------------------------------------#
# macro converting string to list
#----------------------------------------------------------------------------------------#
macro(to_string _VAR _STR)
    STRING(REPLACE ";" " " ${_VAR} "${_STR}")
endmacro(to_string _VAR _STR)


#----------------------------------------------------------------------------------------#
#   Macro to add to string
#----------------------------------------------------------------------------------------#
macro(add _VAR _FLAG)
    if(NOT "${_FLAG}" STREQUAL "")
        if("${${_VAR}}" STREQUAL "")
            set(${_VAR} "${_FLAG}")
        else()
            set(${_VAR} "${${_VAR}} ${_FLAG}")
        endif()
    endif()
endmacro()


#----------------------------------------------------------------------------------------#
# macro to remove duplicates from string
#----------------------------------------------------------------------------------------#
macro(set_no_duplicates _VAR)
    if(NOT "${ARGN}" STREQUAL "")
        set(${_VAR} "${ARGN}")
    endif()
    # remove the duplicates
    if(NOT "${${_VAR}}" STREQUAL "")
        # create list of flags
        to_list(_VAR_LIST "${${_VAR}}")
        list(REMOVE_DUPLICATES _VAR_LIST)
        to_string(${_VAR} "${_VAR_LIST}")
    endif(NOT "${${_VAR}}" STREQUAL "")
endmacro(set_no_duplicates _VAR)

#----------------------------------------------------------------------------------------#
# call before running check_{c,cxx}_compiler_flag
#----------------------------------------------------------------------------------------#
macro(hosttrace_begin_flag_check)
    if(HOSTTRACE_QUIET_CONFIG)
        if(NOT DEFINED CMAKE_REQUIRED_QUIET)
            set(CMAKE_REQUIRED_QUIET OFF)
        endif()
        hosttrace_save_variables(FLAG_CHECK
            VARIABLES CMAKE_REQUIRED_QUIET)
        set(CMAKE_REQUIRED_QUIET ON)
    endif()
endmacro()

#----------------------------------------------------------------------------------------#
# call after running check_{c,cxx}_compiler_flag
#----------------------------------------------------------------------------------------#
macro(hosttrace_end_flag_check)
    if(HOSTTRACE_QUIET_CONFIG)
        hosttrace_restore_variables(FLAG_CHECK
            VARIABLES CMAKE_REQUIRED_QUIET)
    endif()
endmacro()

##########################################################################################
#
#                               C compiler flags
#
##########################################################################################


#----------------------------------------------------------------------------------------#
# add C flag to target
#----------------------------------------------------------------------------------------#
macro(ADD_TARGET_C_FLAG _TARG)
    string(REPLACE "-" "_" _MAKE_TARG "${_TARG}")
    list(APPEND HOSTTRACE_MAKE_TARGETS ${_MAKE_TARG})

    target_compile_options(${_TARG} INTERFACE $<$<COMPILE_LANGUAGE:C>:${ARGN}>)
    list(APPEND ${_MAKE_TARG}_C_FLAGS ${ARGN})
endmacro()


#----------------------------------------------------------------------------------------#
# add C flag w/o check
#----------------------------------------------------------------------------------------#
macro(ADD_C_FLAG FLAG)
    set(_TARG )
    set(_LTARG )
    if(NOT "${ARGN}" STREQUAL "")
        set(_TARG ${ARGN})
        string(TOLOWER "_${ARGN}" _LTARG)
    endif()
    if(NOT "${FLAG}" STREQUAL "")
        if("${_LTARG}" STREQUAL "")
            list(APPEND ${PROJECT_NAME}_C_FLAGS "${FLAG}")
            list(APPEND ${PROJECT_NAME}_C_COMPILE_OPTIONS "${FLAG}")
            add_target_c_flag(${LIBNAME}-compile-options ${FLAG})
        else()
            add_target_c_flag(${_TARG} ${FLAG})
        endif()
    endif()
    unset(_TARG)
    unset(_LTARG)
endmacro()


#----------------------------------------------------------------------------------------#
# check C flag
#----------------------------------------------------------------------------------------#
macro(ADD_C_FLAG_IF_AVAIL FLAG)
    set(_ENABLE ON)
    if(DEFINED HOSTTRACE_BUILD_C AND NOT HOSTTRACE_BUILD_C)
        set(_ENABLE OFF)
    endif()
    set(_TARG )
    set(_LTARG )
    if(NOT "${ARGN}" STREQUAL "")
        set(_TARG ${ARGN})
        string(TOLOWER "_${ARGN}" _LTARG)
    endif()
    if(NOT "${FLAG}" STREQUAL "")
        string(REGEX REPLACE "^/" "c${_LTARG}_" FLAG_NAME "${FLAG}")
        string(REGEX REPLACE "^-" "c${_LTARG}_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")
        if(NOT HOSTTRACE_BUILD_C)
            set(${FLAG_NAME} ON)
        else()
            hosttrace_begin_flag_check()
            check_c_compiler_flag("-Werror" c_werror)
            if(c_werror)
                check_c_compiler_flag("${FLAG} -Werror" ${FLAG_NAME})
            else()
                check_c_compiler_flag("${FLAG}" ${FLAG_NAME})
            endif()
            hosttrace_end_flag_check()
            if(${FLAG_NAME})
                if("${_LTARG}" STREQUAL "")
                    list(APPEND ${PROJECT_NAME}_C_FLAGS "${FLAG}")
                    list(APPEND ${PROJECT_NAME}_C_COMPILE_OPTIONS "${FLAG}")
                    add_target_c_flag(${LIBNAME}-compile-options ${FLAG})
                else()
                    add_target_c_flag(${_TARG} ${FLAG})
                endif()
            endif()
        endif()
    endif()
    unset(_TARG)
    unset(_LTARG)
endmacro()


#----------------------------------------------------------------------------------------#
# add C flag to target
#----------------------------------------------------------------------------------------#
macro(ADD_TARGET_C_FLAG_IF_AVAIL _TARG)
    foreach(_FLAG ${ARGN})
        add_c_flag_if_avail(${_FLAG} ${_TARG})
    endforeach()
endmacro()


##########################################################################################
#
#                                   CXX compiler flags
#
##########################################################################################



#----------------------------------------------------------------------------------------#
# add CXX flag to target
#----------------------------------------------------------------------------------------#
macro(ADD_TARGET_CXX_FLAG _TARG)
    string(REPLACE "-" "_" _MAKE_TARG "${_TARG}")
    list(APPEND HOSTTRACE_MAKE_TARGETS ${_MAKE_TARG})

    target_compile_options(${_TARG} INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${ARGN}>)
    list(APPEND ${_MAKE_TARG}_CXX_FLAGS ${ARGN})
    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
        target_compile_options(${_TARG} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${ARGN}>)
        list(APPEND ${_MAKE_TARG}_CUDA_FLAGS -Xcompiler=${ARGN})
    elseif(CMAKE_CUDA_COMPILER_IS_CLANG)
        target_compile_options(${_TARG} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${ARGN}>)
        list(APPEND ${_MAKE_TARG}_CUDA_FLAGS ${ARGN})
    endif()
endmacro()


#----------------------------------------------------------------------------------------#
# add CXX flag w/o check
#----------------------------------------------------------------------------------------#
macro(ADD_CXX_FLAG FLAG)
    set(_TARG )
    set(_LTARG )
    if(NOT "${ARGN}" STREQUAL "")
        set(_TARG ${ARGN})
        string(TOLOWER "_${ARGN}" _LTARG)
    endif()
    if(NOT "${FLAG}" STREQUAL "")
        if("${_LTARG}" STREQUAL "")
            list(APPEND ${PROJECT_NAME}_CXX_FLAGS "${FLAG}")
            list(APPEND ${PROJECT_NAME}_CXX_COMPILE_OPTIONS "${FLAG}")
            add_target_cxx_flag(${LIBNAME}-compile-options ${FLAG})
        else()
            add_target_cxx_flag(${_TARG} ${FLAG})
        endif()
    endif()
    unset(_TARG)
    unset(_LTARG)
endmacro()


#----------------------------------------------------------------------------------------#
# check CXX flag
#----------------------------------------------------------------------------------------#
macro(ADD_CXX_FLAG_IF_AVAIL FLAG)
    set(_TARG )
    set(_LTARG )
    if(NOT "${ARGN}" STREQUAL "")
        set(_TARG ${ARGN})
        string(TOLOWER "_${ARGN}" _LTARG)
    endif()
    if(NOT "${FLAG}" STREQUAL "")
        string(REGEX REPLACE "^/" "cxx${_LTARG}_" FLAG_NAME "${FLAG}")
        string(REGEX REPLACE "^-" "cxx${_LTARG}_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE "/" "_" FLAG_NAME "${FLAG_NAME}")
        hosttrace_begin_flag_check()
        check_cxx_compiler_flag("-Werror" cxx_werror)
        if(cxx_werror)
            check_cxx_compiler_flag("${FLAG} -Werror" ${FLAG_NAME})
        else()
            check_cxx_compiler_flag("${FLAG}" ${FLAG_NAME})
        endif()
        hosttrace_end_flag_check()
        if(${FLAG_NAME})
            if("${_LTARG}" STREQUAL "")
                list(APPEND ${PROJECT_NAME}_CXX_FLAGS "${FLAG}")
                list(APPEND ${PROJECT_NAME}_CXX_COMPILE_OPTIONS "${FLAG}")
                add_target_cxx_flag(${LIBNAME}-compile-options ${FLAG})
            else()
                add_target_cxx_flag(${_TARG} ${FLAG})
            endif()
        endif()
    endif()
    unset(_TARG)
    unset(_LTARG)
endmacro()


#----------------------------------------------------------------------------------------#
# add CXX flag to target
#----------------------------------------------------------------------------------------#
macro(ADD_TARGET_CXX_FLAG_IF_AVAIL _TARG)
    foreach(_FLAG ${ARGN})
        add_cxx_flag_if_avail(${_FLAG} ${_TARG})
    endforeach()
endmacro()


##########################################################################################
#
#                                       Common
#
##########################################################################################


#----------------------------------------------------------------------------------------#
# check C and CXX flag to compile-options w/o checking
#----------------------------------------------------------------------------------------#
macro(ADD_FLAG)
    foreach(_ARG ${ARGN})
        ADD_C_FLAG("${_ARG}")
        ADD_CXX_FLAG("${_ARG}")
    endforeach()
endmacro()


#----------------------------------------------------------------------------------------#
# add C and CXX flag w/o checking
#----------------------------------------------------------------------------------------#
macro(ADD_TARGET_FLAG _TARG)
    foreach(_ARG ${ARGN})
        ADD_TARGET_C_FLAG(${_TARG} ${_ARG})
        ADD_TARGET_CXX_FLAG(${_TARG} ${_ARG})
    endforeach()
endmacro()


#----------------------------------------------------------------------------------------#
# check C and CXX flag
#----------------------------------------------------------------------------------------#
macro(ADD_FLAG_IF_AVAIL)
    foreach(_ARG ${ARGN})
        ADD_C_FLAG_IF_AVAIL("${_ARG}")
        ADD_CXX_FLAG_IF_AVAIL("${_ARG}")
    endforeach()
endmacro()


#----------------------------------------------------------------------------------------#
# check C and CXX flag
#----------------------------------------------------------------------------------------#
macro(ADD_TARGET_FLAG_IF_AVAIL _TARG)
    foreach(_ARG ${ARGN})
        ADD_TARGET_C_FLAG_IF_AVAIL(${_TARG} ${_ARG})
        ADD_TARGET_CXX_FLAG_IF_AVAIL(${_TARG} ${_ARG})
    endforeach()
endmacro()


#----------------------------------------------------------------------------------------#
# check flag
#----------------------------------------------------------------------------------------#
function(HOSTTRACE_TARGET_FLAG _TARG_TARGET)
    cmake_parse_arguments(_TARG "IF_AVAIL" "MODE" "FLAGS;LANGUAGES" ${ARGN})

    if(NOT _TARG_MODE)
        set(_TARG_MODE INTERFACE)
    endif()

    get_property(ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

    if(NOT _TARG_LANGUAGES)
        get_property(_TARG_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    endif()

    string(TOLOWER "_${_TARG_TARGET}" _LTARG)

    foreach(_FLAG ${_TARG_FLAGS})
        foreach(_LANG ${_TARG_LANGUAGES})
            if(NOT _TARG_IF_AVAIL)
                target_compile_options(${_TARG_TARGET} ${_TARG_MODE} $<$<COMPILE_LANGUAGE:${_LANG}>:${_FLAG}>)
                continue()
            endif()

            if("${_LANG}" STREQUAL "C")
                string(REGEX REPLACE "^/" "c${_LTARG}_" FLAG_NAME "${_FLAG}")
                string(REGEX REPLACE "^-" "c${_LTARG}_" FLAG_NAME "${FLAG_NAME}")
                string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
                string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
                string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")
                hosttrace_begin_flag_check()
                check_c_compiler_flag("-Werror" c_werror)
                if(c_werror)
                    check_c_compiler_flag("${FLAG} -Werror" ${FLAG_NAME})
                else()
                    check_c_compiler_flag("${FLAG}" ${FLAG_NAME})
                endif()
                hosttrace_end_flag_check()
                if(${FLAG_NAME})
                    target_compile_options(${_TARG_TARGET} ${_TARG_MODE}
                        $<$<COMPILE_LANGUAGE:${_LANG}>:${_FLAG}>)
                endif()
            elseif("${_LANG}" STREQUAL "CXX")
                string(REGEX REPLACE "^/" "cxx${_LTARG}_" FLAG_NAME "${_FLAG}")
                string(REGEX REPLACE "^-" "cxx${_LTARG}_" FLAG_NAME "${FLAG_NAME}")
                string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
                string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
                string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")
                hosttrace_begin_flag_check()
                check_cxx_compiler_flag("-Werror" cxx_werror)
                if(cxx_werror)
                    check_cxx_compiler_flag("${FLAG} -Werror" ${FLAG_NAME})
                else()
                    check_cxx_compiler_flag("${FLAG}" ${FLAG_NAME})
                endif()
                hosttrace_end_flag_check()
                if(${FLAG_NAME})
                    target_compile_options(${_TARG_TARGET} ${_TARG_MODE}
                        $<$<COMPILE_LANGUAGE:${_LANG}>:${_FLAG}>)
                    if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
                        target_compile_options(${_TARG_TARGET} ${_TARG_MODE}
                            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_FLAG}>)
                    elseif(CMAKE_CUDA_COMPILER_IS_CLANG)
                        target_compile_options(${_TARG_TARGET} ${_TARG_MODE}
                            $<$<COMPILE_LANGUAGE:CUDA>:${_FLAG}>)
                    endif()
                endif()
            endif()
        endforeach()
    endforeach()
endfunction()


#----------------------------------------------------------------------------------------#
# add CUDA flag to target
#----------------------------------------------------------------------------------------#
macro(ADD_TARGET_CUDA_FLAG _TARG)
    string(REPLACE "-" "_" _MAKE_TARG "${_TARG}")
    list(APPEND HOSTTRACE_MAKE_TARGETS ${_MAKE_TARG})

    target_compile_options(${_TARG} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${ARGN}>)
    list(APPEND ${_MAKE_TARG}_CUDA_FLAGS ${ARGN})
endmacro()

#----------------------------------------------------------------------------------------#
# add to any language
#----------------------------------------------------------------------------------------#
function(ADD_USER_FLAGS _TARGET _LANGUAGE)

    set(_FLAGS ${${_LANGUAGE}FLAGS} $ENV{${_LANGUAGE}FLAGS}
        ${${_LANGUAGE}_FLAGS} $ENV{${_LANGUAGE}_FLAGS})

    string(REPLACE " " ";" _FLAGS "${_FLAGS}")

    set(${PROJECT_NAME}_${_LANGUAGE}_FLAGS
        ${${PROJECT_NAME}_${_LANGUAGE}_FLAGS} ${_FLAGS} PARENT_SCOPE)

    set(${PROJECT_NAME}_${_LANGUAGE}_COMPILE_OPTIONS
        ${${PROJECT_NAME}_${_LANGUAGE}_COMPILE_OPTIONS} ${_FLAGS} PARENT_SCOPE)

    target_compile_options(${_TARGET} INTERFACE
        $<$<COMPILE_LANGUAGE:${_LANGUAGE}>:${_FLAGS}>)
endfunction()

#----------------------------------------------------------------------------------------#
# add compiler definition
#----------------------------------------------------------------------------------------#
function(HOSTTRACE_TARGET_COMPILE_DEFINITIONS _TARG _VIS)
    foreach(_DEF ${ARGN})
        target_compile_definitions(${_TARG} ${_VIS}
            $<$<COMPILE_LANGUAGE:CXX>:${_DEF}>)
        if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
            target_compile_definitions(${_TARG} ${_VIS}
                $<$<COMPILE_LANGUAGE:CUDA>:${_DEF}>)
        elseif(CMAKE_CUDA_COMPILER_IS_CLANG)
            target_compile_definitions(${_TARG} ${_VIS}
                $<$<COMPILE_LANGUAGE:CUDA>:${_DEF}>)
        endif()
    endforeach()
endfunction()

#----------------------------------------------------------------------------------------#
# determine compiler types for each language
#----------------------------------------------------------------------------------------#
get_property(ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
foreach(LANG C CXX CUDA)

    if(NOT DEFINED CMAKE_${LANG}_COMPILER)
        set(CMAKE_${LANG}_COMPILER "")
    endif()

    if(NOT DEFINED CMAKE_${LANG}_COMPILER_ID)
        set(CMAKE_${LANG}_COMPILER_ID "")
    endif()

    function(SET_COMPILER_VAR VAR _BOOL)
        set(CMAKE_${LANG}_COMPILER_IS_${VAR} ${_BOOL} CACHE INTERNAL
            "CMake ${LANG} compiler identification (${VAR})" FORCE)
        mark_as_advanced(CMAKE_${LANG}_COMPILER_IS_${VAR})
    endfunction()

    if(("${LANG}" STREQUAL "C" AND CMAKE_COMPILER_IS_GNUCC)
        OR
       ("${LANG}" STREQUAL "CXX" AND CMAKE_COMPILER_IS_GNUCXX))

        # GNU compiler
        SET_COMPILER_VAR(       GNU                 1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "icc.*")

        # Intel icc compiler
        SET_COMPILER_VAR(       INTEL               1)
        SET_COMPILER_VAR(       INTEL_ICC           1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "icpc.*")

        # Intel icpc compiler
        SET_COMPILER_VAR(       INTEL               1)
        SET_COMPILER_VAR(       INTEL_ICPC          1)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "AppleClang")

        # Clang/LLVM compiler
        SET_COMPILER_VAR(       CLANG               1)
        SET_COMPILER_VAR( APPLE_CLANG               1)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "Clang")

        # Clang/LLVM compiler
        SET_COMPILER_VAR(       CLANG               1)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "PGI")

        # PGI compiler
        SET_COMPILER_VAR(       PGI                 1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "xlC" AND UNIX)

        # IBM xlC compiler
        SET_COMPILER_VAR(       XLC                 1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "aCC" AND UNIX)

        # HP aC++ compiler
        SET_COMPILER_VAR(       HP_ACC              1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "CC" AND
           CMAKE_SYSTEM_NAME MATCHES "IRIX" AND UNIX)

        # IRIX MIPSpro CC Compiler
        SET_COMPILER_VAR(       MIPS                1)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "Intel")

        SET_COMPILER_VAR(       INTEL               1)

        set(CTYPE ICC)
        if("${LANG}" STREQUAL "CXX")
            set(CTYPE ICPC)
        endif()

        SET_COMPILER_VAR(       INTEL_${CTYPE}      1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "MSVC")

        # Windows Visual Studio compiler
        SET_COMPILER_VAR(       MSVC                1)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "NVIDIA")

        # NVCC
        SET_COMPILER_VAR(       NVIDIA              1)

    endif()

    # set other to no
    foreach(TYPE GNU INTEL INTEL_ICC INTEL_ICPC APPLE_CLANG CLANG PGI XLC HP_ACC MIPS MSVC NVIDIA)
        if(NOT DEFINED CMAKE_${LANG}_COMPILER_IS_${TYPE})
            SET_COMPILER_VAR(${TYPE} 0)
        endif()
    endforeach()

endforeach()
