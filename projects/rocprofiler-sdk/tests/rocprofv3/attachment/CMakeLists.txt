#
#
#

add_subdirectory(attach-once)
add_subdirectory(attach-twice)
add_subdirectory(attach-tree)
if(NOT ROCPROFILER_DISABLE_TRACE_DECODER)
    add_subdirectory(attach-once-att)
endif()
