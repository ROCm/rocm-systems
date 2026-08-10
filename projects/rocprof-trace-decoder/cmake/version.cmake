#
# Decoder version, and the compile definitions that carry it into the library and its tests.
#
# Split out from CMakeLists.txt because test/ and test/unit/ can each be configured as their
# own top-level project, in which case the decoder's top-level CMakeLists.txt never runs.
#

if(NOT DEFINED VERSION_MAJOR)
    set(VERSION_MAJOR 0)
endif()

if(NOT DEFINED VERSION_MINOR)
    set(VERSION_MINOR 2)
endif()

if(NOT DEFINED VERSION_PATCH)
    set(VERSION_PATCH 2)
endif()

set(TTD_VERSION_COMPILE_DEFS
    ROCPROF_TRACE_DECODER_VERSION_MAJOR=${VERSION_MAJOR} ROCPROF_TRACE_DECODER_VERSION_MINOR=${VERSION_MINOR}
    ROCPROF_TRACE_DECODER_VERSION_PATCH=${VERSION_PATCH})
