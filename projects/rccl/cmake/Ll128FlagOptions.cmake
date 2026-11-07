# Experimental: bytes of flag per LL128 line (see src/include/rccl_ll128_flag.h). Host (net proxy) and device code
# must agree, so it is a build setting.
set(RCCL_LL128_FLAG_BYTES "8" CACHE STRING "Bytes of flag per LL128 line: 8 (default), 4, 2 or 1")
set_property(CACHE RCCL_LL128_FLAG_BYTES PROPERTY STRINGS 8 4 2 1)
if(NOT RCCL_LL128_FLAG_BYTES MATCHES "^[8421]$")
  message(FATAL_ERROR "RCCL_LL128_FLAG_BYTES must be 8, 4, 2 or 1, got '${RCCL_LL128_FLAG_BYTES}'")
endif()
if(NOT RCCL_LL128_FLAG_BYTES STREQUAL "8")
  if(ENABLE_TDM_PRIM_LL128)
    message(FATAL_ERROR "ENABLE_TDM_PRIM_LL128 needs RCCL_LL128_FLAG_BYTES=8")
  endif()
  add_compile_definitions(RCCL_LL128_FLAG_BYTES=${RCCL_LL128_FLAG_BYTES})
endif()

# Testing only: wrap a narrow LL128 flag every 256 (1) or 32 (2) steps so its wraparound handling runs quickly.
set(RCCL_LL128_FLAG_TEST "0" CACHE STRING "Testing only: wrap a narrow LL128 flag every 256 (1) or 32 (2) steps")
set_property(CACHE RCCL_LL128_FLAG_TEST PROPERTY STRINGS 0 1 2)
if(NOT RCCL_LL128_FLAG_TEST MATCHES "^[012]$")
  message(FATAL_ERROR "RCCL_LL128_FLAG_TEST must be 0, 1 or 2, got '${RCCL_LL128_FLAG_TEST}'")
endif()
if(NOT RCCL_LL128_FLAG_TEST STREQUAL "0")
  add_compile_definitions(RCCL_LL128_FLAG_TEST=${RCCL_LL128_FLAG_TEST})
endif()
