# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Check every externally linkable callable definition, including Debug COMDATs.
# ISA-local helpers use a namespace; exported backend entry points use a suffix.
# Local text symbols cannot be selected from another object and data is separate.
execute_process(
    COMMAND "${NM}" -A -C --defined-only "${ARCHIVE}"
    OUTPUT_VARIABLE symbols
    COMMAND_ERROR_IS_FATAL ANY
)
string(REPLACE "\n" ";" lines "${symbols}")
foreach(line IN LISTS lines)
    if(NOT line MATCHES "[ ]([TW])[ ](.+)$")
        continue()
    endif()
    set(symbol "${CMAKE_MATCH_2}")
    if(symbol MATCHES "std::")
        message(
            FATAL_ERROR
            "Runtime standard-library definition in arithmetic object: ${line}"
        )
    endif()
    set(domain baseline)
    set(entry_suffix "")
    if(line MATCHES "_(x86_64_v[34](_bf16)?)[.]cpp[.]o:")
        set(domain ${CMAKE_MATCH_1})
        set(entry_suffix ${domain})
    elseif(line MATCHES "_avx512vnni[.]cpp[.]o:")
        set(domain x86_64_v4_vnni)
        set(entry_suffix avx512vnni)
    endif()
    if(NOT domain STREQUAL "baseline")
        if(
            NOT symbol MATCHES "rocjitsu::arith::${domain}::"
            AND NOT symbol MATCHES "rocjitsu::arith::[A-Za-z0-9_]+_${entry_suffix}[(<]"
        )
            message(
                FATAL_ERROR
                "Callable definition lacks its ISA identity: ${line}"
            )
        endif()
    endif()
    string(SHA256 key "${symbol}")
    if(DEFINED owner_${key} AND NOT "${owner_${key}}" STREQUAL "${domain}")
        message(
            FATAL_ERROR
            "Callable defined in incompatible ISA domains: ${line}"
        )
    endif()
    set(owner_${key} "${domain}")
endforeach()
