if(NOT RTC_TESTING)
    set(TEST_SRC
        Set.cpp
    )

    add_library(KERNELS EXCLUDE_FROM_ALL OBJECT ${TEST_SRC})
    set_source_files_properties(${TEST_SRC} PROPERTIES LANGUAGE HIP)
    target_link_libraries(KERNELS PRIVATE Catch2::Catch2)
endif()
