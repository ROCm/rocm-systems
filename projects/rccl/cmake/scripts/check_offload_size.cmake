# cmake -DOFFLOAD_BINARY=<file> -P check_offload_size.cmake
#
# -fembed-offload-object keeps only the low 32 bits of the file's size, and the link then fails
# with "The end of the file was unexpectedly encountered".
file(SIZE "${OFFLOAD_BINARY}" _size)
if(_size GREATER_EQUAL 4294967296)
  math(EXPR _mib "${_size} / 1048576")
  message(FATAL_ERROR
    "${OFFLOAD_BINARY} is ${_mib} MiB, and the compiler cannot embed 4 GiB or more. "
    "Build fewer GPU targets, or use a toolchain whose offload packager supports --compress.")
endif()
