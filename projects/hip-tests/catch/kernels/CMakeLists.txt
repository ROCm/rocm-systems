if(NOT RTC_TESTING)
    set(TEST_SRC
        Set.cpp
    )

    hip_add_library(NAME KERNELS
                    TYPE OBJECT
                    SOURCES ${TEST_SRC})
endif()
