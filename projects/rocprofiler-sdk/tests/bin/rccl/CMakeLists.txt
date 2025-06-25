find_package(rccl)
find_package(Threads REQUIRED)

if(rccl_FOUND)
    add_subdirectory(single-process)
else()
    message(STATUS "RCCL package was not found, not building RCCL tests")
endif()
