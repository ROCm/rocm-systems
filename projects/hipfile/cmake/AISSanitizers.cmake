# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

option(AIS_USE_SANITIZERS "Build with sanitizers (clang only - see docs for a list)" OFF)
option(AIS_USE_THREAD_SANITIZER "Build with -fsanitize=thread (clang only - not compatible with AIS_USE_SANITIZERS)" OFF)

if(AIS_USE_THREAD_SANITIZER)
    message(WARNING "TSAN has known problems with higher levels of entropy, try using `sudo sysctl vm.mmap_rnd_bits=28` if you encounter errors concerning unexpected memory mappings.")
endif()

function(ais_add_sanitizers target)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "The sanitizers are only available on clang")
    endif()

    if(AIS_USE_SANITIZERS AND AIS_USE_THREAD_SANITIZER)
        message(FATAL_ERROR "AIS_USE_SANITIZERS is not compatible with AIS_USE_THREAD_SANITIZER")
    endif()

    if(AIS_USE_SANITIZERS)
        set(SANITIZER_LIST address,float-divide-by-zero,integer,leak,local-bounds,nullability,undefined,vptr)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${SANITIZER_LIST}>
            $<$<COMPILE_LANGUAGE:CXX>:-fno-sanitize-recover=integer>
        )
        target_link_options(${target} PRIVATE
            -Xarch_host -fsanitize=${SANITIZER_LIST}
            -Xarch_host -fno-sanitize-recover=integer
        )
    endif()

    if(AIS_USE_THREAD_SANITIZER)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=thread>
        )
        target_link_options(${target} PRIVATE
            -Xarch_host -fsanitize=thread
        )
    endif()

    # Options for all sanitizers
    target_compile_options(${target} PRIVATE
        -fno-omit-frame-pointer
    )
    target_link_options(${target} PRIVATE
        -fuse-ld=lld
        --rtlib=compiler-rt
        --unwindlib=libgcc
    )
endfunction()
