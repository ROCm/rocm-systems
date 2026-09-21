if(WIN32)
    # The HIP callback/buffer tracing sources (hip.cpp, abi.cpp, stream.cpp, event.cpp,
    # graph.cpp) pull in details/ostream.hpp, which references Linux glibc types
    # (__locale_struct, etc.). Headers only until that is ported.
    target_sources(rocprofiler-sdk-object-library PRIVATE defines.hpp hip.hpp)
else()
    set(ROCPROFILER_LIB_HIP_SOURCES abi.cpp event.cpp graph.cpp hip.cpp stream.cpp)
    set(ROCPROFILER_LIB_HIP_HEADERS defines.hpp event.hpp graph.hpp hip.hpp stream.hpp
                                    utils.hpp)
    target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_HIP_SOURCES}
                                                          ${ROCPROFILER_LIB_HIP_HEADERS})
    add_subdirectory(details)
endif()
