# Cross-checks the Dxg* entry points the installed header declares against the
# ones the version script makes global.
#
# The two drift in opposite, equally bad directions. A name in the header that
# the version script does not export is a declaration a consumer can compile
# against and then fail to bind. A name the version script exports that the
# header does not declare is an entry point in the dev package's ABI with
# nothing for its definition to be checked against: DxgAbiCheck is defined
# extern "C" in src/dxg/abi.cpp and ROCr binds it by dlsym() through a typedef
# of its own, so with no declaration in between the two signatures can drift
# with no diagnostic anywhere and the mismatch surfaces as undefined behaviour.
#
# Deliberately reads the sources rather than the built library, so that it still
# says something in a configuration that cannot link librocdxg.so - the WIN_SDK
# link needs libwkmi.a. check_dynamic_exports.cmake covers the built artifact.
#
# Expects HEADER and VERSION_SCRIPT on the command line.

cmake_minimum_required(VERSION 3.6.3)

# A match that ends in ';' is two list elements, the second empty, so empties
# have to be meaningful rather than silently dropped.
cmake_policy(SET CMP0007 NEW)

if (NOT EXISTS "${HEADER}")
  message(FATAL_ERROR "installed header not found: ${HEADER}")
endif()

if (NOT EXISTS "${VERSION_SCRIPT}")
  message(FATAL_ERROR "version script not found: ${VERSION_SCRIPT}")
endif()

file(READ "${HEADER}" header_text)
file(READ "${VERSION_SCRIPT}" version_text)

# A declaration is "<return type> HSAKMTAPI <name>(", which is the only shape
# the entry points in this header take. hsakmt.h puts each of those three on a
# line of its own, so the separator has to admit newlines.
string(REGEX MATCHALL "HSAKMTAPI[ \t\r\n]+Dxg[A-Za-z0-9_]+[ \t\r\n]*\\(" declared_matches "${header_text}")
set(declared "")
foreach (match IN LISTS declared_matches)
  string(REGEX REPLACE "^HSAKMTAPI[ \t\r\n]+(Dxg[A-Za-z0-9_]+)[ \t\r\n]*\\($" "\\1" name "${match}")
  if (name)
    list(APPEND declared "${name}")
  endif()
endforeach()

# Only the global block counts; a name under "local:" is hidden, not exported.
# Each match ends in the version script's ';', which CMake reads as a list
# separator, so iterating a MATCHALL result yields an empty element after every
# real one.
string(REGEX REPLACE "[ \t\r\n]*local:.*$" "" global_text "${version_text}")
string(REGEX MATCHALL "Dxg[A-Za-z0-9_]+[ \t]*;" exported_matches "${global_text}")
set(exported "")
foreach (match IN LISTS exported_matches)
  string(REGEX REPLACE "[ \t;]+$" "" name "${match}")
  if (name)
    list(APPEND exported "${name}")
  endif()
endforeach()

# Only when neither side has anything is the reading at fault. One empty side
# is the regression itself, and there is currently one Dxg* name in total, so
# saying "the pattern is stale" here would send a reader to the regex instead
# of to the declaration that just went missing.
if (NOT declared AND NOT exported)
  message(FATAL_ERROR
          "no Dxg* names in either ${HEADER} or ${VERSION_SCRIPT}; the match patterns are stale")
endif()

list(SORT declared)
list(SORT exported)
list(REMOVE_DUPLICATES declared)
list(REMOVE_DUPLICATES exported)

set(declared_not_exported "${declared}")
if (exported)
  list(REMOVE_ITEM declared_not_exported ${exported})
endif()

set(exported_not_declared "${exported}")
if (declared)
  list(REMOVE_ITEM exported_not_declared ${declared})
endif()

if (declared_not_exported)
  message(FATAL_ERROR
          "declared in the installed header but not exported: ${declared_not_exported}")
endif()

if (exported_not_declared)
  message(FATAL_ERROR
          "exported but not declared in the installed header: ${exported_not_declared}")
endif()

list(LENGTH declared count)
message(STATUS "${count} Dxg entry point(s) declared and exported: ${declared}")
