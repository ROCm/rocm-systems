set(ROCPROFILER_PC_SAMPLING_SOURCES hsa_adapter.cpp utils.cpp service.cpp cid_manager.cpp
                                    code_object.cpp)
set(ROCPROFILER_PC_SAMPLING_HEADERS hsa_adapter.hpp utils.hpp service.hpp types.hpp
                                    cid_manager.hpp code_object.hpp)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_PC_SAMPLING_SOURCES}
                                                  ${ROCPROFILER_PC_SAMPLING_HEADERS})

add_subdirectory(parser)
add_subdirectory(ioctl)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
