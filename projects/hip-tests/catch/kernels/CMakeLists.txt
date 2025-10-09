if(NOT RTC_TESTING)
    set(TEST_SRC
        Set.cpp
    )

    add_library(KERNELS EXCLUDE_FROM_ALL OBJECT ${TEST_SRC})
    target_compile_options(KERNELS PUBLIC -std=c++17)
    target_link_libraries(KERNELS PRIVATE Catch2::Catch2)
endif()
