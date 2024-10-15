#
if(ROCPROFSYS_USE_ROCPROFILER OR ROCPROFSYS_USE_ROCTRACER)
    target_sources(
        rocprofiler-systems-object-library
        PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hsa_rsrc_factory.hpp
                ${CMAKE_CURRENT_LIST_DIR}/hsa_rsrc_factory.cpp)
endif()
