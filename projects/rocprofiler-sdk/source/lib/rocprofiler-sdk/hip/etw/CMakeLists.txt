#
# ETW consumer for the rocm_hip_tlg provider emitted by amdhip64 on Windows
#
# session.cpp is built everywhere so that the context start/stop call sites need no
# preprocessor guards; off Windows it supplies no-ops. The decoder is Windows-only because it
# is written directly against EVENT_RECORD and TDH.
#
target_sources(rocprofiler-sdk-object-library PRIVATE session.cpp session.hpp)

if(WIN32)
    target_sources(rocprofiler-sdk-object-library PRIVATE decoder.cpp decoder.hpp)
endif()
