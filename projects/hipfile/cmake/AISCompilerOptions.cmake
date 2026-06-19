# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Set compiler flags on target based on compilers being used on sources

include(AISClangCompilerOptions)
include(AISGNUCompilerOptions)
include(AISSanitizers)

function(ais_set_compiler_flags target)
    get_target_property(sources ${target} SOURCES)
    foreach(source IN LISTS sources)
        get_source_file_property(language ${source} LANGUAGE)
        # CMake HIP language prior to 3.28 only supports AMD, so change LANGUAGE to CUDA for HIP files
        if(AIS_BUILD_NVIDIA_DETAIL AND CMAKE_VERSION VERSION_LESS "3.28" AND language STREQUAL HIP)
            set_source_files_properties(${source} PROPERTIES LANGUAGE "CUDA")
            set(language CUDA)
            message(STATUS "Setting ${source} LANGUAGE to CUDA because CMake < 3.28")
        endif()
        set(compiler_id "${CMAKE_${language}_COMPILER_ID}")
        set(compiler_version "${CMAKE_${language}_COMPILER_VERSION}")

        # Only use default flags with include-what-you-use. Otherwise you'll
        # clutter the output with a lot of "unrecognized flags" warnings if
        # there's mismatch between IWYU's clang and the compiler you are using.
        if(NOT AIS_USE_IWYU)
            if(compiler_id STREQUAL "GNU" OR compiler_id STREQUAL "NVIDIA")
                get_ais_gnu_warning_flags(compiler_flags ${compiler_version})
            elseif(compiler_id STREQUAL "Clang")
                get_ais_clang_warning_flags(compiler_flags ${compiler_version})
            endif()
        endif()
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANG_AND_ID:${language},${compiler_id}>:${compiler_flags}>)
        if(AIS_USE_CODE_COVERAGE)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fprofile-instr-generate -fcoverage-mapping>)
            target_link_options(${target} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fprofile-instr-generate>)
            target_link_options(${target} PRIVATE $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-fprofile-instr-generate>)
        endif()
    endforeach()

    if(AIS_USE_SANITIZERS OR AIS_USE_THREAD_SANITIZER)
        ais_add_sanitizers(${target})
    endif()

    if(NOT BUILD_TESTING)
        target_compile_options(${target} PRIVATE -fvisibility=hidden)
    endif()
endfunction()
