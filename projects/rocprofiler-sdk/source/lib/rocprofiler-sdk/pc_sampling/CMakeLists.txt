#
# PC Sampling
# - HSA support officially added in HSA-Runtime v1.14.0
#
if(WIN32)
    # PC sampling delivers its samples through the KFD ioctl interface, which Windows
    # reaches via D3DKMT instead; windows_stubs.cpp supplies the stand-in symbols.
    target_sources(rocprofiler-sdk-object-library PRIVATE windows_stubs.cpp service.hpp)
    return()
endif()

add_subdirectory(parser)

if(hsa-runtime64_VERSION AND hsa-runtime64_VERSION VERSION_LESS 1.14.0)
    return()
endif()

set(ROCPROFILER_PC_SAMPLING_SOURCES hsa_adapter.cpp utils.cpp service.cpp cid_manager.cpp
                                    code_object.cpp)
set(ROCPROFILER_PC_SAMPLING_HEADERS hsa_adapter.hpp utils.hpp service.hpp types.hpp
                                    cid_manager.hpp code_object.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_PC_SAMPLING_SOURCES}
                                                      ${ROCPROFILER_PC_SAMPLING_HEADERS})

add_subdirectory(ioctl)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
