set(ROCM_TIMESYNC_CORE_SOURCES
  ipc.cpp atomic.cpp ringbuffer.cpp
)

add_library(rocm-timesync-core 
    OBJECT
    ${ROCM_TIMESYNC_CORE_SOURCES}
)

set_target_properties(rocm-timesync-core
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(rocm-timesync-core
    PUBLIC rocm-timesync-core-headers
)
