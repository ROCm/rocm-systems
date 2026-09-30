# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Warning flags for llvm/clang
#
# https://clang.llvm.org/docs/DiagnosticsReference.html

function(get_ais_clang_warning_flags outvar compiler_version)

    # Warning flags for clang 17 and earlier
    set(flags
        # Basic "high" warning levels
        -Wall
        -Wextra

        # Avoid non-standard C/C++ behavior
        -pedantic

        # Suggestions from the Effective C++ book
        -Weffc++

        # Build with this until it becomes a problem
        # -Wnvcc-compat requires clang 19 and is handled below
        -Wcuda-compat

        # Don't bake gcc-isms into the code. clang will usually deal with
        # this but it's not a good idea.
        -Wgnu

        # Don't use deprecated constructs
        -Wdeprecated

        # Aggressively warn about thread-safety issues
        # Requires markup: https://clang.llvm.org/docs/ThreadSafetyAnalysis.html
        -Wthread-safety
        -Wthread-safety-beta
        -Wthread-safety-negative
        -Wthread-safety-verbose

        # Turn this on later (generates a lot of noise now)
        -Wdocumentation

        # Avoid things that will cause problems on Windows
        -Wmicrosoft

        # Do not warn about incompatibility with earlier versions
        # of C++. The pedantic versions suppress a larger set of
        # warnings than the non-pedantic ones.
        -Wno-c++98-compat-pedantic
        -Wno-c++11-compat-pedantic
        -Wno-c++14-compat-pedantic
        -Wno-pre-c++17-compat-pedantic

        # Turn on stack protection options
        -fstack-clash-protection
        -fstack-protector-strong    # Consider -all instead of -strong

        # Turn on strict flex arrays (helps ASAN, _FORTIFY_SOURCE, etc.)
        -fstrict-flex-arrays=3

        # Initialize local variables with a pattern
        -ftrivial-auto-var-init=pattern

        # Misc warnings
        #
        # This includes most warnings that are not enabled by default
        # with the exception of C++ 11/14/etc. and clang extension
        # warnings since we only support one compiler with a defined
        # language level
        #
        # This list is not set in stone and we should consider removing
        # diagnostics that return a lot of false positives.
        #
        # Anything commented out with no comment isn't supported w/ clang
        # versions in common use. We'll want to update this scheme with
        # one that better respects versions in the future.
        -Warc-repeated-use-of-weak
        -Warray-bounds-pointer-arithmetic
        -Wassign-enum
        -Watomic-implicit-seq-cst
        -Watomic-properties
        -Wbad-function-cast
        -Wbinary-literal
        -Wcast-align
        -Wcast-function-type
        -Wcast-qual
        -Wcomma
        -Wconditional-uninitialized
        -Wconsumed
        -Wconversion
        #-Wcovered-switch-default (flags default labels where we handle all enum values)
        -Wcstring-format-directive
        -Wctad-maybe-unsupported
        -Wdate-time
        -Wdirect-ivar-access
        -Wdisabled-macro-expansion
        -Wdouble-promotion
        -Wdtor-name
        -Wduplicate-enum
        -Wduplicate-method-arg
        -Wduplicate-method-match
        -Wendif-labels
        -Wexit-time-destructors
        -Wexpansion-to-defined
        -Wexplicit-ownership-type
        -Wextra-semi
        -Wextra-semi-stmt
        -Wfloat-equal
        -Wformat=2
        -Wformat-non-iso
        -Wformat-pedantic
        -Wformat-security
        -Wformat-type-confusion
        -Wfour-char-constants
        -Wfuse-ld-path
        -Wglobal-constructors
        -Wheader-hygiene
        -Widiomatic-parentheses
        -Wimplicit-fallthrough
        -Wimplicit-retain-self
        -Wincompatible-function-pointer-types-strict
        -Wincomplete-module
        -Winconsistent-missing-destructor-override
        -Winvalid-or-nonexistent-directory
        -Wlocal-type-template-args
        -Wloop-analysis
        -Wmain
        -Wmain-return-type
        -Wmax-tokens
        -Wmethod-signatures
        -Wmissing-include-dirs
        -Wmissing-noreturn
        -Wmissing-prototypes
        -Wmissing-variable-declarations
        -Wnarrowing
        -Wnewline-eof
        -Wnoexcept-type
        -Wnon-gcc
        -Wnon-virtual-dtor
        -Wnonportable-system-include-path
        -Wnullable-to-nonnull-conversion
        -Wold-style-cast
        -Wopenmp
        -Wover-aligned
        -Woverriding-method-mismatch
        -Wpacked
        #-Wpadded # Generates a LOT of unfixable noise
        -Wpartial-availability
        -Wpointer-arith
        -Wpoison-system-directories
        -Wpragmas
        -Wquoted-include-in-framework-header
        -Wreceiver-forward-class
        -Wredundant-parens
        -Wreserved-identifier
        -Wsequence-point
        -Wshadow-all
        -Wshift-sign-overflow
        -Wsigned-enum-bitfield
        -Wsource-uses-openmp
        -Wspir-compat
        -Wstatic-in-inline
        -Wstrict-potentially-direct-selector
        -Wstrict-prototypes
        -Wstrict-selector-match
        -Wsuggest-destructor-override
        -Wsuggest-override
        -Wsuper-class-method-mismatch
        -Wswitch-default
        -Werror=switch-enum # To make sure we never miss a CUDA enum value
        -Wtautological-constant-in-range-compare
        -Wtype-limits
        -Wunaligned-access
        -Wundeclared-selector
        -Wundef
        -Wundef-prefix
        -Wundefined-func-template
        -Wundefined-reinterpret-cast
        -Wunguarded-availability
        -Wunnamed-type-template-args
        -Wunreachable-code-aggressive
        -Wused-but-marked-unused
        -Wvariadic-macros
        -Wvector-conversion
        -Wvla
        -Wno-weak-vtables # Lots of noise in gtest code, so turn off for now
        -Wzero-as-null-pointer-constant

        # -Wlocal-type-template-args enables
        # -Wc++98-compat-local-type-template-args.
        -Wno-c++98-compat-local-type-template-args
    )

    if(compiler_version VERSION_GREATER_EQUAL 18.1.6)
        set(flags
            # Misc warnings
            -Wnonportable-private-system-apinotes-path
            -Wopenacc
            -Wsource-uses-openacc
            ${flags}
        )
    endif()

    if(compiler_version VERSION_GREATER_EQUAL 19.1)
        set(flags
            # Build with this until it becomes a problem
            # -Wcuda-compat is handled in the general flags, above
            -Wnvcc-compat

            # Misc warnings
            -Wformat-signedness
            ${flags}
        )
    endif()

    if(compiler_version VERSION_GREATER_EQUAL 20.1)
        set(flags
            # Misc warnings
            -Wdecls-in-multiple-modules
            -Wvariadic-macro-arguments-omitted
            ${flags}
        )
    endif()

    if(compiler_version VERSION_GREATER_EQUAL 21.0)
        set(flags
            # Aggressively warn about thread-safety issues
            # Requires markup: https://clang.llvm.org/docs/ThreadSafetyAnalysis.html
            -Wthread-safety-pointer

            # Misc warnings
            -Wignored-base-class-qualifiers
            -Wshift-bool
            -Wunique-object-duplication
            ${flags}
        )
    endif()

    # Only use _FORTIFY_SOURCE if the optimization level is -O2, -O3, or -Os
    string(JOIN " " MYCXXFLAGS ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}})
    if(MYCXXFLAGS MATCHES "-O[2-3s]")
        set(flags
            -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3
            ${flags}
        )
    endif()

    set(${outvar} ${flags} PARENT_SCOPE)

endfunction()
