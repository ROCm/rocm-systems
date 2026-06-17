# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Warning flags for GNU g++
#
# https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html

function(get_ais_gnu_warning_flags outvar compiler_version)

    # Warning flags for g++ 9 and earlier
    set(flags
        # Basic "high" warning levels
        -Wall
        -Wextra

        # Avoid non-standard C/C++ behavior
        # Can't use -pedantic with nvcc at this time, as nvcc
        # generates code with non-standard #line styles, leading
        # to a LOT of warnings
        #-pedantic

        # Check for ABI warnings
        # Most of this is noise, but probably useful to turn
        # on from time to time
        #-Wabi

        # Turn on stack protection options
        -fstack-clash-protection
        -fstack-protector-strong

        # Misc warnings
        #-Waggregate-return # We return structs, but might be useful
                            # to turn on to see where this can be
                            # minimized
        -Walloca
        -Walloc-zero
        -Warray-bounds=2 # TODO: Consider making =3 in newer g++
        -Wcast-align
        -Wcast-qual
        -Wconversion
        -Wdate-time
        -Wdouble-promotion
        -Wduplicated-branches
        -Wduplicated-cond
        -Wfloat-equal
        -Wformat=2
        -Wformat-nonliteral
        -Wformat-overflow=2
        -Wformat-security
        -Wformat-signedness
        -Wformat-truncation=2
        -Wformat-y2k
        -Winvalid-pch
        # This is a warning for when using <C++11
        #-Wlong-long
        -Wlogical-op
        -Wmissing-declarations
        -Wnormalized
        -Wnull-dereference
        -Wpacked
        #-Wpadded # Probably should be a developer warning
        -Wpointer-arith
        -Wredundant-decls
        -Wshadow
        -Wshadow-local
        -Wshift-overflow=2
        -Wstrict-overflow=4
        -Wswitch-default
        -Wswitch-enum
        -Wtrampolines
        -Wundef
        -Wuninitialized
        -Wunknown-pragmas
        -Wunsafe-loop-optimizations
        -Wunused
        -Wunused-macros
        -Wuseless-cast
        -Wvla
        -Wzero-as-null-pointer-constant

        # TODO: Add size warnings when we pick a limit
    )

    if(compiler_version VERSION_GREATER_EQUAL 12)
        set(flags
            # Misc warnings
            -Wbidi-chars=any
            -Winterference-size
            -Wtrivial-auto-var-init
            ${flags}
        )

        # Only use _FORTIFY_SOURCE if the optimization level is -O2, -O3, or -Os
        string(JOIN " " MYCXXFLAGS ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}})
        if(MYCXXFLAGS MATCHES "-O[2-3s]")
            set(flags
                -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3
                ${flags}
            )
        endif()
    endif()

    if(compiler_version VERSION_GREATER_EQUAL 13)
        set(flags
            # Turn on strict flex arrays (helps ASAN, _FORTIFY_SOURCE, etc.)
            -fstrict-flex-arrays=3
            # Misc warnings
            -Winvalid-utf8
            ${flags}
        )
    endif()

    if(compiler_version VERSION_GREATER_EQUAL 14)
        set(flags
            # Misc warnings
            -Walloc-size
            -Wcalloc-transposed-args
            -Wflex-array-member-not-at-end
            -Wnrvo
            ${flags}
        )
    endif()

    if(compiler_version VERSION_GREATER_EQUAL 15)
        set(flags
            # Misc warnings
            -Wtrailing-whitespace
            -Wleading-whitespace=tabs
            ${flags}
        )
    endif()

    set(${outvar} ${flags} PARENT_SCOPE)

endfunction()
