# ---------------------------------------------------------------------------
# src/os/: one binary per test file. They share misc/host_tests.cmake's _MICRO_LINUX_TEST_SOURCES.
# ---------------------------------------------------------------------------
# misc/socket-test.cc and misc/shmutils-test.cc link the real linux.cc, behind the seams linux-test.cc puts around it.
# Its own headers come first, as in linux-test.cc, so the seam renames no declaration in them.
set(_MICRO_OS_LINUX_OPTIONS)
foreach(_header os.h comm.h crypt.h socket.h utils.h fakes/libc_seam.h fakes/dl_seam.h)
  list(APPEND _MICRO_OS_LINUX_OPTIONS -include ${_header})
endforeach()
set_source_files_properties(${RCCL_HOST_HIPIFY_DIR}/src/os/linux.cc PROPERTIES
  GENERATED TRUE
  COMPILE_OPTIONS "${_MICRO_OS_LINUX_OPTIONS}")
set(_MICRO_OS_DEFS
  LINUX_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/os/linux.cc"
  LINUX_IPCSOCKET_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/os/linux_ipcsocket.cc"
  LINUX_SOCKET_PAIR_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/os/linux_socket_pair.cc")

# One src/os/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_os_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} os/${_test} ${ARGN}
    OBJECT_PREFIX rccl-os-obj
    COMMON_DEFS ${_MICRO_OS_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_os.yaml")
endmacro()

rccl_add_os_micro_binary(rccl-UnitTestsMicroOsLinux linux-test.cc
  SOURCES
    ${_MICRO_LINUX_TEST_SOURCES}
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/socket.cc)  # ncclSocketProgress, ncclSocketToString

# linux_ipcsocket.cc defines ncclIpcFdClose, which fakes/nccl_fakes.cc otherwise supplies.
rccl_add_os_micro_binary(rccl-UnitTestsMicroOsLinuxIpcsocket linux-ipcsocket-test.cc
  SOURCES ${_MICRO_LINUX_TEST_SOURCES}
  EXTRA_DEFS RCCL_NCCL_FAKES_OMIT_IPC_FD_CLOSE)

rccl_add_os_micro_binary(rccl-UnitTestsMicroOsLinuxSocketPair linux-socket-pair-test.cc
  SOURCES ${_MICRO_LINUX_TEST_SOURCES})
