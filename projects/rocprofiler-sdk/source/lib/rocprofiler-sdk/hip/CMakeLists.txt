# names.cpp re-expands hip.def.cpp keeping only the operation names, so it builds
# everywhere. The ETW consumer needs it to translate a traced function name into a
# ROCPROFILER_HIP_*_API_ID_* operation id.
target_sources(rocprofiler-sdk-object-library PRIVATE defines.hpp hip.hpp names.cpp
                                                      names.hpp)

add_subdirectory(etw)

# The HIP callback/buffer tracing sources pull in details/ostream.hpp, which references
# Linux glibc types (__locale_struct, etc.), so they are excluded on Windows until that is
# ported.
if(WIN32)
    set(ROCPROFILER_LIB_HIP_SOURCES windows_stubs.cpp)
    set(ROCPROFILER_LIB_HIP_HEADERS event.hpp graph.hpp stream.hpp)
else()
    set(ROCPROFILER_LIB_HIP_SOURCES abi.cpp event.cpp graph.cpp hip.cpp stream.cpp)
    set(ROCPROFILER_LIB_HIP_HEADERS event.hpp graph.hpp stream.hpp utils.hpp)
endif()

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_HIP_SOURCES}
                                                      ${ROCPROFILER_LIB_HIP_HEADERS})

if(NOT WIN32)
    add_subdirectory(details)
endif()
