# ===================================================================================
# Post-build sanity check: verify librccl.so has no unresolved (undefined) symbols.
#
# This reproduces, at build time, the failure mode that otherwise only shows up at
# runtime when the library is dlopen()'d (e.g. ctypes.CDLL() from a Python test)
# and aborts with an "undefined symbol" error.
#
# Invoked via:  cmake -DRCCL_LIB=<path-to-librccl.so> -P check_undefined_symbols.cmake
# ===================================================================================

if(NOT DEFINED RCCL_LIB)
  message(FATAL_ERROR "check_undefined_symbols.cmake requires -DRCCL_LIB=<path>")
endif()

if(NOT EXISTS "${RCCL_LIB}")
  message(FATAL_ERROR "Undefined-symbol check: library not found: ${RCCL_LIB}")
endif()

find_program(LDD_EXECUTABLE ldd)
if(NOT LDD_EXECUTABLE)
  message(WARNING "Undefined-symbol check skipped: 'ldd' not found in PATH.")
  return()
endif()

# `ldd -r` performs full (data + function) relocation resolution against the
# library's dependency graph and prints a line per unresolved symbol of the form:
#     undefined symbol: <name>    (<path>)
# It exits 0 even when symbols are missing, so we parse the output instead of
# relying on the return code.
execute_process(
  COMMAND ${LDD_EXECUTABLE} -r "${RCCL_LIB}"
  OUTPUT_VARIABLE ldd_stdout
  ERROR_VARIABLE  ldd_stderr
)

set(ldd_output "${ldd_stdout}\n${ldd_stderr}")
string(REPLACE "\n" ";" ldd_lines "${ldd_output}")

set(undefined_symbols "")
set(missing_deps "")
foreach(line ${ldd_lines})
  if(line MATCHES "undefined symbol: ([^ \t]+)")
    list(APPEND undefined_symbols "${CMAKE_MATCH_1}")
  elseif(line MATCHES "=> not found")
    list(APPEND missing_deps "${line}")
  endif()
endforeach()

# A missing dependency (=> not found) means ldd could not fully resolve the
# graph, so the absence of "undefined symbol" lines would be unreliable. Warn
# loudly but don't fail the build on what is really an environment issue.
if(missing_deps)
  list(REMOVE_DUPLICATES missing_deps)
  string(REPLACE ";" "\n  " missing_deps_pretty "${missing_deps}")
  message(WARNING "Undefined-symbol check incomplete - unresolved dependencies:\n  ${missing_deps_pretty}")
endif()

# Accumulate all failures so every problem is reported in a single error.
set(failures "")

if(undefined_symbols)
  list(REMOVE_DUPLICATES undefined_symbols)

  # Demangle for readability when c++filt is available.
  find_program(CXXFILT_EXECUTABLE c++filt)
  set(report "")
  foreach(sym ${undefined_symbols})
    set(pretty "${sym}")
    if(CXXFILT_EXECUTABLE)
      execute_process(
        COMMAND ${CXXFILT_EXECUTABLE} "${sym}"
        OUTPUT_VARIABLE demangled
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(demangled AND NOT demangled STREQUAL sym)
        set(pretty "${sym}  [${demangled}]")
      endif()
    endif()
    string(APPEND report "\n  - ${pretty}")
  endforeach()

  string(APPEND failures "Unresolved symbols:${report}\n")
endif()

# RCCL atomics must stay inline/lock-free; a libatomic dependency signals a misaligned/oversized atomic (see -Watomic-alignment).
# Step 1: detect whether librccl pulled in a libatomic dependency.
set(libatomic_path "")
foreach(line ${ldd_lines})
  if(line MATCHES "libatomic\\.so[^ ]* => ([^ \t]+)")
    set(libatomic_path "${CMAKE_MATCH_1}")
  endif()
endforeach()
if(libatomic_path)
  # Step 2: nm is required to name the symbols; without it, warn and skip (do not fail).
  find_program(NM_EXECUTABLE nm)
  if(NOT NM_EXECUTABLE)
    message(WARNING "libatomic check skipped: 'nm' not found.")
  else()
    # Step 2a: list the symbols libatomic defines and the symbols librccl leaves undefined.
    execute_process(COMMAND ${NM_EXECUTABLE} -D --defined-only "${libatomic_path}" OUTPUT_VARIABLE atomic_defined)
    execute_process(COMMAND ${NM_EXECUTABLE} -D --undefined-only "${RCCL_LIB}" OUTPUT_VARIABLE rccl_undefined)
    # Step 2b: collect libatomic's exported symbol names (strip version suffix after '@').
    string(REPLACE "\n" ";" atomic_defined_lines "${atomic_defined}")
    set(atomic_names "")
    foreach(l ${atomic_defined_lines})
      if(l MATCHES " [A-Za-z] ([^ \t@]+)")
        list(APPEND atomic_names "${CMAKE_MATCH_1}")
      endif()
    endforeach()
    # Step 2c: report each librccl undefined symbol that libatomic provides.
    set(atomic_report "")
    string(REPLACE "\n" ";" rccl_undefined_lines "${rccl_undefined}")
    foreach(l ${rccl_undefined_lines})
      if(l MATCHES " [UwvV] ([^ \t@]+)")
        set(sym "${CMAKE_MATCH_1}")
        list(FIND atomic_names "${sym}" idx)
        if(NOT idx EQUAL -1)
          string(APPEND atomic_report "\n  - ${sym}")
        endif()
      endif()
    endforeach()
    # Step 3: record the libatomic failure with the offending symbols.
    string(APPEND failures "libatomic dependency (misaligned/oversized atomic):${atomic_report}\n")
  endif()
endif()

# Report all accumulated failures once, or pass.
if(NOT failures STREQUAL "")
  get_filename_component(lib_name "${RCCL_LIB}" NAME)
  message(FATAL_ERROR
    "Undefined-symbol check failed for ${lib_name}\n"
    "${failures}"
    "Bypass: -DCHECK_UNDEFINED_SYMBOLS=OFF (install.sh: --no-undef-check)")
endif()

message(STATUS "Undefined-symbol check passed")
