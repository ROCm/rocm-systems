set(ROCPROFILER_LIB_SPM_SOURCES spm_core.cpp spm_service.cpp spm_decode.cpp spm_dlsym.cpp
                                spm_dispatch_handlers.cpp)
set(ROCPROFILER_LIB_SPM_HEADERS spm_core.hpp spm_dlsym.hpp spm_dispatch_handlers.hpp)
target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_SPM_SOURCES}
                                                      ${ROCPROFILER_LIB_SPM_HEADERS})
add_subdirectory(tests)
