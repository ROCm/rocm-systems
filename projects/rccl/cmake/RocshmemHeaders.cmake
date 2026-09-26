# Order targets after the rocSHMEM GDA header staging. Targets linking rccl
# inherit it from rccl; those that do not still reach the staged headers via
# nccl_device.h and would otherwise race the copy, failing by scheduling.
# No-op when GIN is off or before src/ creates the target, so callers need no
# guard of their own.
function(rccl_order_rocshmem_headers)
  if(NOT ENABLE_ROCSHMEM_GIN)
    return()
  endif()
  if(NOT TARGET copy_rocshmem_headers)
    return()
  endif()
  foreach(_target IN LISTS ARGN)
    add_dependencies(${_target} copy_rocshmem_headers)
  endforeach()
endfunction()
