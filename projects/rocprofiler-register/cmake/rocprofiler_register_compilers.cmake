# include guard
# ########################################################################################
#
# Compilers
#
# ########################################################################################
#
# sets (cached):
#
# CMAKE_C_COMPILER_IS_<TYPE> CMAKE_CXX_COMPILER_IS_<TYPE>
#
# where TYPE is: - GNU - CLANG - INTEL - INTEL_ICC - INTEL_ICPC - PGI - XLC - HP_ACC -
# MIPS - MSVC
#

include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
include(CheckCSourceRuns)

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)

include(CMakeParseArguments)

include(rocprofiler_register_utilities)

# ----------------------------------------------------------------------------------------#
# macro converting string to list
# ----------------------------------------------------------------------------------------#
macro(ROCPROFILER_REGISTER_TO_LIST _VAR _STR)
    string(REPLACE "  " " " ${_VAR} "${_STR}")
    string(REPLACE " " ";" ${_VAR} "${_STR}")
endmacro()

# ----------------------------------------------------------------------------------------#
# macro converting string to list
# ----------------------------------------------------------------------------------------#
macro(ROCPROFILER_REGISTER_TO_STRING _VAR _STR)
    string(REPLACE ";" " " ${_VAR} "${_STR}")
endmacro()

# ----------------------------------------------------------------------------------------#
# Macro to add to string
# ----------------------------------------------------------------------------------------#
macro(add _VAR _FLAG)
    if(NOT "${_FLAG}" STREQUAL "")
        if("${${_VAR}}" STREQUAL "")
            set(${_VAR} "${_FLAG}")
        else()
            set(${_VAR} "${${_VAR}} ${_FLAG}")
        endif()
    endif()
endmacro()

# ----------------------------------------------------------------------------------------#
# call before running check_{c,cxx}_compiler_flag
# ----------------------------------------------------------------------------------------#
macro(ROCPROFILER_REGISTER_BEGIN_FLAG_CHECK)
    if(ROCPROFILER_REGISTER_QUIET_CONFIG)
        if(NOT DEFINED CMAKE_REQUIRED_QUIET)
            set(CMAKE_REQUIRED_QUIET OFF)
        endif()
        rocprofiler_register_save_variables(FLAG_CHECK VARIABLES CMAKE_REQUIRED_QUIET)
        set(CMAKE_REQUIRED_QUIET ON)
    endif()
endmacro()

# ----------------------------------------------------------------------------------------#
# call after running check_{c,cxx}_compiler_flag
# ----------------------------------------------------------------------------------------#
macro(ROCPROFILER_REGISTER_END_FLAG_CHECK)
    if(ROCPROFILER_REGISTER_QUIET_CONFIG)
        rocprofiler_register_restore_variables(FLAG_CHECK VARIABLES CMAKE_REQUIRED_QUIET)
    endif()
endmacro()

# ----------------------------------------------------------------------------------------#
# check flag
# ----------------------------------------------------------------------------------------#
function(ROCPROFILER_REGISTER_TARGET_COMPILE_OPTIONS _TARG_TARGET)
    cmake_parse_arguments(_TARG "BUILD_INTERFACE;FORCE" ""
                          "PUBLIC;INTERFACE;PRIVATE;LANGUAGES;LINK_LANGUAGES" ${ARGN})

    if(NOT _TARG_MODE)
        set(_TARG_MODE INTERFACE)
    endif()

    get_property(_ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    set(_SUPPORTED_LANGUAGES "C" "CXX")

    if(NOT _TARG_LANGUAGES)
        foreach(_LANG ${_ENABLED_LANGUAGES})
            if(_LANG IN_LIST _SUPPORTED_LANGUAGES)
                list(APPEND _TARG_LANGUAGES ${_LANG})
            endif()
        endforeach()
    endif()

    string(TOLOWER "_${_TARG_TARGET}" _TARG_TARGET_LC)

    function(rocprofiler_register_target_compile_option_impl _TARGET_IMPL
             _TARGET_MODE_IMPL _TARGET_LANG_IMPL _TARGET_FLAG_IMPL)
        if(_TARG_BUILD_INTERFACE)
            target_compile_options(
                ${_TARGET_IMPL}
                ${_TARGET_MODE_IMPL}
                $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:${_TARGET_LANG_IMPL}>:${_TARGET_FLAG_IMPL}>>
                )
        else()
            target_compile_options(
                ${_TARGET_IMPL} ${_TARGET_MODE_IMPL}
                $<$<COMPILE_LANGUAGE:${_TARGET_LANG_IMPL}>:${_TARGET_FLAG_IMPL}>)
        endif()

        if("${_TARGET_LANG_IMPL}" IN_LIST _TARG_LINK_LANGUAGES)
            if(_TARG_BUILD_INTERFACE)
                target_link_options(
                    ${_TARGET_IMPL}
                    ${_TARGET_MODE_IMPL}
                    $<BUILD_INTERFACE:$<$<LINK_LANGUAGE:${_TARGET_LANG_IMPL}>:${_TARGET_FLAG_IMPL}>>
                    )
            else()
                target_link_options(
                    ${_TARGET_IMPL} ${_TARGET_MODE_IMPL}
                    $<$<LINK_LANGUAGE:${_TARGET_LANG_IMPL}>:${_TARGET_FLAG_IMPL}>)
            endif()
        endif()
    endfunction()

    function(rocprofiler_register_target_compile_option_patch_name _P_LANG _P_IN _P_OUT)
        string(TOLOWER "${_P_LANG}" _P_LANG)
        string(REGEX REPLACE "^(/|-)" "${_P_LANG}${_TARG_TARGET_LC}_" _NAME "${_P_IN}")
        string(REPLACE "-" "_" _NAME "${_NAME}")
        string(REPLACE " " "_" _NAME "${_NAME}")
        string(REPLACE "=" "_" _NAME "${_NAME}")
        set(${_P_OUT}
            "${_NAME}"
            PARENT_SCOPE)
    endfunction()

    if(NOT DEFINED rocprofiler_register_c_error AND NOT DEFINED
                                                    rocprofiler_register_cxx_error)
        rocprofiler_register_begin_flag_check()
        check_c_compiler_flag("-Werror" c_rocprofiler_register_werror)
        check_cxx_compiler_flag("-Werror" cxx_rocprofiler_register_werror)
        rocprofiler_register_end_flag_check()
    endif()

    foreach(_TARG_MODE PUBLIC INTERFACE PRIVATE)
        foreach(_FLAG ${_TARG_${_TARG_MODE}})
            foreach(_LANG ${_TARG_LANGUAGES})
                unset(FLAG_NAME)
                rocprofiler_register_target_compile_option_patch_name(${_LANG} "${_FLAG}"
                                                                      FLAG_NAME)

                if(_TARG_FORCE)
                    set(${FLAG_NAME}
                        1
                        CACHE INTERNAL "${_LANG} flag: ${_FLAG}")
                else()
                    rocprofiler_register_begin_flag_check()

                    if("${_LANG}" STREQUAL "C")
                        if(c_rocprofiler_register_werror)
                            check_c_compiler_flag("${_FLAG} -Werror" ${FLAG_NAME})
                        else()
                            check_c_compiler_flag("${_FLAG}" ${FLAG_NAME})
                        endif()
                    elseif("${_LANG}" STREQUAL "CXX")
                        if(cxx_rocprofiler_register_werror)
                            check_cxx_compiler_flag("${_FLAG} -Werror" ${FLAG_NAME})
                        else()
                            check_cxx_compiler_flag("${_FLAG}" ${FLAG_NAME})
                        endif()
                    else()
                        message(
                            FATAL_ERROR
                                "rocprofiler_register_target_compile_option :: unknown language: ${_LANG}"
                            )
                    endif()

                    rocprofiler_register_end_flag_check()
                endif()

                if(${FLAG_NAME})
                    rocprofiler_register_target_compile_option_impl(
                        ${_TARG_TARGET} ${_TARG_MODE} ${_LANG} "${_FLAG}")
                endif()
            endforeach()
        endforeach()
    endforeach()
endfunction()

# ----------------------------------------------------------------------------------------#
# add to any language
# ----------------------------------------------------------------------------------------#
function(ROCPROFILER_REGISTER_TARGET_USER_FLAGS _TARGET _LANGUAGE)

    set(_FLAGS ${${_LANGUAGE}FLAGS} $ENV{${_LANGUAGE}FLAGS} ${${_LANGUAGE}_FLAGS}
               $ENV{${_LANGUAGE}_FLAGS})

    string(REPLACE " " ";" _FLAGS "${_FLAGS}")

    set(${PROJECT_NAME}_${_LANGUAGE}_FLAGS
        ${${PROJECT_NAME}_${_LANGUAGE}_FLAGS} ${_FLAGS}
        PARENT_SCOPE)

    set(${PROJECT_NAME}_${_LANGUAGE}_COMPILE_OPTIONS
        ${${PROJECT_NAME}_${_LANGUAGE}_COMPILE_OPTIONS} ${_FLAGS}
        PARENT_SCOPE)

    target_compile_options(${_TARGET}
                           INTERFACE $<$<COMPILE_LANGUAGE:${_LANGUAGE}>:${_FLAGS}>)
endfunction()

# ----------------------------------------------------------------------------------------#
# add compiler definition
# ----------------------------------------------------------------------------------------#
function(ROCPROFILER_REGISTER_TARGET_COMPILE_DEFINITIONS _TARG _VIS)
    foreach(_DEF ${ARGN})
        if(NOT "${_DEF}" MATCHES "[A-Za-z_]+=.*" AND "${_DEF}" MATCHES
                                                     "^ROCPROFILER_REGISTER_")
            set(_DEF "${_DEF}=1")
        endif()
        target_compile_definitions(${_TARG} ${_VIS} $<$<COMPILE_LANGUAGE:CXX>:${_DEF}>)
    endforeach()
endfunction()

# ----------------------------------------------------------------------------------------#
# determine compiler types for each language
# ----------------------------------------------------------------------------------------#
get_property(ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
foreach(LANG C CXX HIP CUDA)

    if(NOT DEFINED CMAKE_${LANG}_COMPILER)
        set(CMAKE_${LANG}_COMPILER "")
    endif()

    if(NOT DEFINED CMAKE_${LANG}_COMPILER_ID)
        set(CMAKE_${LANG}_COMPILER_ID "")
    endif()

    function(SET_COMPILER_VAR VAR _BOOL)
        set(CMAKE_${LANG}_COMPILER_IS_${VAR}
            ${_BOOL}
            CACHE INTERNAL "CMake ${LANG} compiler identification (${VAR})" FORCE)
        mark_as_advanced(CMAKE_${LANG}_COMPILER_IS_${VAR})
    endfunction()

    if(("${LANG}" STREQUAL "C" AND CMAKE_COMPILER_IS_GNUCC)
       OR ("${LANG}" STREQUAL "CXX" AND CMAKE_COMPILER_IS_GNUCXX))

        # GNU compiler
        set_compiler_var(GNU 1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "icc.*")

        # Intel icc compiler
        set_compiler_var(INTEL 1)
        set_compiler_var(INTEL_ICC 1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "icpc.*")

        # Intel icpc compiler
        set_compiler_var(INTEL 1)
        set_compiler_var(INTEL_ICPC 1)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "AppleClang")

        # Clang/LLVM compiler
        set_compiler_var(CLANG 1)
        set_compiler_var(APPLE_CLANG 1)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "Clang")

        # Clang/LLVM compiler
        set_compiler_var(CLANG 1)

        # HIP Clang compiler
        if(CMAKE_${LANG}_COMPILER MATCHES "hipcc")
            set_compiler_var(HIPCC 1)
        endif()

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "PGI")

        # PGI compiler
        set_compiler_var(PGI 1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "xlC" AND UNIX)

        # IBM xlC compiler
        set_compiler_var(XLC 1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "aCC" AND UNIX)

        # HP aC++ compiler
        set_compiler_var(HP_ACC 1)

    elseif(
        CMAKE_${LANG}_COMPILER MATCHES "CC"
        AND CMAKE_SYSTEM_NAME MATCHES "IRIX"
        AND UNIX)

        # IRIX MIPSpro CC Compiler
        set_compiler_var(MIPS 1)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "Intel")

        set_compiler_var(INTEL 1)

        set(CTYPE ICC)
        if("${LANG}" STREQUAL "CXX")
            set(CTYPE ICPC)
        endif()

        set_compiler_var(INTEL_${CTYPE} 1)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "MSVC")

        # Windows Visual Studio compiler
        set_compiler_var(MSVC 1)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "NVIDIA")

        # NVCC
        set_compiler_var(NVIDIA 1)

    endif()

    # set other to no
    foreach(
        TYPE
        GNU
        INTEL
        INTEL_ICC
        INTEL_ICPC
        APPLE_CLANG
        CLANG
        PGI
        XLC
        HP_ACC
        MIPS
        MSVC
        NVIDIA
        HIPCC)
        if(NOT DEFINED CMAKE_${LANG}_COMPILER_IS_${TYPE})
            set_compiler_var(${TYPE} 0)
        endif()
    endforeach()

endforeach()
