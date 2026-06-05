# hotswap_copy_glob.cmake — stage glob-matched artifacts into a directory.
#
# Invoked via `cmake -P` from the HotSwap ExternalProject install steps in
# cmake/Hotswap.cmake. It copies the files a HotSwap build stage produces
# (the comgr/ROCR shared libraries and the LLVM tools) into the flat
# `${stage}/lib`, `${stage}/llvm-tools` layout mirage discovers.
#
# Inputs (all passed with -D):
#   HS_ROOT            Directory to resolve patterns against (a build tree).
#   HS_GLOB            ;-separated list of filename glob patterns, e.g.
#                      "libamd_comgr.so*" or "lld;ld.lld;llc;llvm-mc".
#   HS_DST             Destination directory (created if absent).
#   HS_RECURSE_ANCHOR  Optional. When set, the file with this name is located
#                      recursively under HS_ROOT and its parent directory is
#                      used as the glob base instead of HS_ROOT (ROCR buries
#                      libhsa-runtime64.so in a build subdirectory whose path
#                      is not known ahead of time).
#
# FOLLOW_SYMLINK_CHAIN is used so versioned SONAME links (e.g.
# libhsa-runtime64.so -> libhsa-runtime64.so) are reproduced verbatim.

if(NOT DEFINED HS_ROOT OR NOT DEFINED HS_GLOB OR NOT DEFINED HS_DST)
  message(FATAL_ERROR "hotswap_copy_glob: HS_ROOT, HS_GLOB and HS_DST are required")
endif()

set(_base "${HS_ROOT}")
if(DEFINED HS_RECURSE_ANCHOR AND NOT HS_RECURSE_ANCHOR STREQUAL "")
  file(GLOB_RECURSE _anchor "${HS_ROOT}/${HS_RECURSE_ANCHOR}")
  if(NOT _anchor)
    message(FATAL_ERROR "hotswap_copy_glob: ${HS_RECURSE_ANCHOR} not found under ${HS_ROOT}")
  endif()
  list(GET _anchor 0 _anchor0)
  get_filename_component(_base "${_anchor0}" DIRECTORY)
endif()

set(_patterns)
foreach(_g IN LISTS HS_GLOB)
  list(APPEND _patterns "${_base}/${_g}")
endforeach()

file(GLOB _files LIST_DIRECTORIES false ${_patterns})
if(NOT _files)
  message(FATAL_ERROR "hotswap_copy_glob: no files matched ${_patterns}")
endif()

file(MAKE_DIRECTORY "${HS_DST}")
foreach(_f IN LISTS _files)
  file(INSTALL "${_f}" DESTINATION "${HS_DST}"
       USE_SOURCE_PERMISSIONS FOLLOW_SYMLINK_CHAIN)
endforeach()
message(STATUS "hotswap: staged ${_files} -> ${HS_DST}")
