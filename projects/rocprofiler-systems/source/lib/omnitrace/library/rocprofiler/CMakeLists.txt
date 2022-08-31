#
if(OMNITRACE_USE_ROCPROFILER)
    target_sources(
        omnitrace-object-library PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hsa_rsrc_factory.hpp
                                         ${CMAKE_CURRENT_LIST_DIR}/hsa_rsrc_factory.cpp)
endif()
