# Checks the built library's dynamic symbol table for the Dxg* entry points.
#
# The version script is text, not the library. Nothing else in this directory
# loads librocdxg either - the lifecycle test links amd_kfd_lifecycle.cpp out of
# hsa-runtime and never touches this artifact. So the one thing that decides
# whether ROCr's dlsym() finds these symbols, the dynamic symbol table of the
# artifact we ship, had no test at all.
#
# The regression this exists to catch: an entry point's C linkage comes from a
# declaration, not from its definition. Lose the declaration and the definition
# still compiles, but as a local C++ symbol - the version script's unmangled
# name then matches nothing, the dynamic table loses the entry, and ROCr's
# dlsym() returns NULL at runtime. Every text-level check still passes.
#
# Names are given by the caller rather than parsed out of the header or the
# version script, so that deleting one from either is a failure here.
#
# Expects LIBRARY, NM, EXPECTED_VERSION and EXPECTED_SYMBOLS on the command
# line, the last comma-separated so it survives being passed as one -D argument.

cmake_minimum_required(VERSION 3.6.3)

string(REPLACE "," ";" EXPECTED_SYMBOLS "${EXPECTED_SYMBOLS}")

if (NOT EXISTS "${LIBRARY}")
  message(FATAL_ERROR "built library not found: ${LIBRARY}")
endif()

# -D reads the dynamic table specifically: a symbol only in .symtab is one a
# consumer cannot bind to, which is exactly the failure being tested for.
execute_process(COMMAND "${NM}" -D --defined-only "${LIBRARY}"
                OUTPUT_VARIABLE nm_out
                ERROR_VARIABLE nm_err
                RESULT_VARIABLE nm_res)

if (NOT nm_res EQUAL 0)
  message(FATAL_ERROR "${NM} -D --defined-only ${LIBRARY} failed (${nm_res}): ${nm_err}")
endif()

# Best-effort context for a failure: if the expected symbol is missing, the
# static table usually says why - a mangled or local definition shows up there.
execute_process(COMMAND "${NM}" --defined-only "${LIBRARY}"
                OUTPUT_VARIABLE nm_static_out
                ERROR_VARIABLE nm_static_err
                RESULT_VARIABLE nm_static_res)
if (NOT nm_static_res EQUAL 0)
  set(nm_static_out "")
endif()

set(failures "")

foreach (symbol IN LISTS EXPECTED_SYMBOLS)
  # 'T' and not 't': a lowercase t is a local definition, which is what a symbol
  # that lost its extern "C" declaration decays into.
  if (nm_out MATCHES "[ \t]T[ \t]+${symbol}@@${EXPECTED_VERSION}([ \t\r\n]|$)")
    continue()
  endif()

  if (nm_out MATCHES "[ \t][A-Za-z][ \t]+${symbol}([@ \t\r\n]|$)")
    list(APPEND failures
         "${symbol}: in the dynamic table, but not as a global text symbol "
         "tagged @@${EXPECTED_VERSION}")
  elseif (nm_static_out MATCHES "[ \t][a-z][ \t]+[A-Za-z0-9_]*${symbol}[A-Za-z0-9_]*")
    list(APPEND failures
         "${symbol}: defined only locally, so nothing can bind to it - a "
         "declaration carrying extern \"C\" and default visibility is "
         "probably missing")
  else()
    list(APPEND failures "${symbol}: absent from the dynamic symbol table")
  endif()
endforeach()

if (failures)
  string(REPLACE ";" "\n  " failures "${failures}")
  # Whatever the dynamic table does say about these names, which is usually
  # what explains the failure.
  set(context "")
  foreach (symbol IN LISTS EXPECTED_SYMBOLS)
    string(REGEX MATCHALL "[^\r\n]*${symbol}[^\r\n]*" symbol_lines "${nm_out}")
    list(APPEND context ${symbol_lines})
  endforeach()
  string(REPLACE ";" "\n  " context "${context}")
  message(FATAL_ERROR
          "${LIBRARY} does not export what it must:\n  ${failures}\n"
          "Matching lines in nm -D --defined-only:\n  ${context}")
endif()

list(LENGTH EXPECTED_SYMBOLS count)
message(STATUS
        "${count} entry point(s) exported as global text @@${EXPECTED_VERSION}: "
        "${EXPECTED_SYMBOLS}")
