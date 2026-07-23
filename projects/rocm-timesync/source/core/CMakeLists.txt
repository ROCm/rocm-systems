set(ROCM_TIMESYNC_CORE_SOURCES
  ipc.cpp atomic.cpp ringbuffer.cpp
)

add_library(rocm-timesync-core 
    OBJECT
    ${ROCM_TIMESYNC_CORE_SOURCES}
)

target_compile_options(rocm-timesync-core
    PRIVATE -fPIC
)

target_link_libraries(rocm-timesync-core
    PUBLIC rocm-timesync-core-headers
)
