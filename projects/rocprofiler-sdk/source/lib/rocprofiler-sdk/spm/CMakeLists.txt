set(ROCPROFILER_LIB_SPM_SOURCES core.cpp service.cpp decode.cpp dlsym.cpp
                                dispatch_handlers.cpp asynchandler.cpp)
set(ROCPROFILER_LIB_SPM_HEADERS core.hpp dlsym.hpp dispatch_handlers.hpp asynchandler.hpp)
target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_SPM_SOURCES}
                                                      ${ROCPROFILER_LIB_SPM_HEADERS})
if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
