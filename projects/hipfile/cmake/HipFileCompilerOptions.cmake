# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Set compiler flags on target based on compilers being used on sources
#
# Usage:
#   hipfile_set_compiler_flags(<target> [NO_SAFE_BUFFERS])
#
#   NO_SAFE_BUFFERS     Don't apply the HIPFILE_WARN_UNSAFE_BUFFER_OPS
#                       flags (see HipFileClangSafeBuffers.cmake)

include(HipFileClangCompilerOptions)
include(HipFileGNUCompilerOptions)
include(HipFileSanitizers)

function(hipfile_set_compiler_flags target)
    cmake_parse_arguments(arg "NO_SAFE_BUFFERS" "" "" ${ARGN})

    get_target_property(sources ${target} SOURCES)
    foreach(source IN LISTS sources)
        get_source_file_property(language ${source} LANGUAGE)
        # CMake HIP language prior to 3.28 only supports AMD, so change LANGUAGE to CUDA for HIP files
        if(HIPFILE_BUILD_NVIDIA_DETAIL AND CMAKE_VERSION VERSION_LESS "3.28" AND language STREQUAL HIP)
            set_source_files_properties(${source} PROPERTIES LANGUAGE "CUDA")
            set(language CUDA)
            message(STATUS "Setting ${source} LANGUAGE to CUDA because CMake < 3.28")
        endif()
        set(compiler_id "${CMAKE_${language}_COMPILER_ID}")
        set(compiler_version "${CMAKE_${language}_COMPILER_VERSION}")

        # Only use default flags with include-what-you-use. Otherwise you'll
        # clutter the output with a lot of "unrecognized flags" warnings if
        # there's mismatch between IWYU's clang and the compiler you are using.
        if(NOT HIPFILE_USE_IWYU)
            if(compiler_id STREQUAL "GNU" OR compiler_id STREQUAL "NVIDIA")
                get_hipfile_gnu_warning_flags(compiler_flags ${language} ${compiler_version})
            elseif(compiler_id STREQUAL "Clang")
                get_hipfile_clang_warning_flags(compiler_flags ${language} ${compiler_version})
            endif()
        endif()
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANG_AND_ID:${language},${compiler_id}>:${compiler_flags}>)
        if(HIPFILE_USE_CODE_COVERAGE)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fprofile-instr-generate -fcoverage-mapping>)
            target_link_options(${target} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fprofile-instr-generate>)
            target_link_options(${target} PRIVATE $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-fprofile-instr-generate>)
        endif()
    endforeach()

    # Clang "safe buffer" warnings and standard library hardening (see
    # HipFileClangSafeBuffers.cmake)
    #
    # C++ only, since HIP kernels use raw pointers by necessity, and device
    # code can't call the standard library's assertion handlers. The
    # warnings are skipped with IWYU for the same reason as the warning
    # flags, above.
    if(HIPFILE_WARN_UNSAFE_BUFFER_OPS AND NOT arg_NO_SAFE_BUFFERS)
        target_compile_definitions(${target} PRIVATE
            $<$<COMPILE_LANG_AND_ID:CXX,Clang>:_GLIBCXX_ASSERTIONS>
            $<$<COMPILE_LANG_AND_ID:CXX,Clang>:_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST>
        )
        if(NOT HIPFILE_USE_IWYU)
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions>
            )
        endif()
    endif()

    if(HIPFILE_USE_SANITIZERS OR HIPFILE_USE_THREAD_SANITIZER)
        hipfile_add_sanitizers(${target})
    endif()

    if(NOT BUILD_TESTING)
        target_compile_options(${target} PRIVATE -fvisibility=hidden)
    endif()
endfunction()
