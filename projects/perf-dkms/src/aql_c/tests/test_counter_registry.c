/**
 * @file test_counter_registry.c
 * @brief Userspace tests for counter registry functionality
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Include counter registry headers */
#include "../counter_registry.h"
#include "../arch_creator.h"

/* Test result tracking */
static int tests_run = 0;
static int tests_passed = 0;

#define TEST_ASSERT(condition, message) do { \
    tests_run++; \
    if (condition) { \
        tests_passed++; \
        printf("PASS: %s\n", message); \
    } else { \
        printf("FAIL: %s\n", message); \
    } \
} while(0)

/* Test lookup counter by name */
void test_lookup_counter_by_name(void) {
    printf("\n=== Testing lookup_counter_by_name ===\n");

    /* Test valid counter names */
    const counter_def_t* counter = lookup_counter_by_name("SQ_WAVES");
    TEST_ASSERT(counter != NULL, "lookup_counter_by_name: SQ_WAVES found");
    TEST_ASSERT(counter->id == COUNTER_SQ_WAVES, "SQ_WAVES has correct ID");
    TEST_ASSERT(counter->hw_block == HW_IP_BLOCK_SQ, "SQ_WAVES has correct hw_block");

    counter = lookup_counter_by_name("GL2C_HIT");
    TEST_ASSERT(counter != NULL, "lookup_counter_by_name: GL2C_HIT found");
    TEST_ASSERT(counter->id == COUNTER_GL2C_HIT, "GL2C_HIT has correct ID");
    TEST_ASSERT(counter->hw_block == HW_IP_BLOCK_GL2C, "GL2C_HIT has correct hw_block");

    counter = lookup_counter_by_name("TA_TA_BUSY");
    TEST_ASSERT(counter != NULL, "lookup_counter_by_name: TA_TA_BUSY found");
    TEST_ASSERT(counter->id == COUNTER_TA_TA_BUSY, "TA_TA_BUSY has correct ID");
    TEST_ASSERT(counter->hw_block == HW_IP_BLOCK_TA, "TA_TA_BUSY has correct hw_block");

    /* Test invalid counter names */
    counter = lookup_counter_by_name("INVALID_COUNTER");
    TEST_ASSERT(counter == NULL, "lookup_counter_by_name: Invalid counter returns NULL");

    counter = lookup_counter_by_name(NULL);
    TEST_ASSERT(counter == NULL, "lookup_counter_by_name: NULL input returns NULL");

    counter = lookup_counter_by_name("");
    TEST_ASSERT(counter == NULL, "lookup_counter_by_name: Empty string returns NULL");
}

/* Test lookup counter by ID */
void test_lookup_counter_by_id(void) {
    printf("\n=== Testing lookup_counter_by_id ===\n");

    /* Test valid counter IDs */
    const counter_def_t* counter = lookup_counter_by_id(COUNTER_SQ_WAVES);
    TEST_ASSERT(counter != NULL, "lookup_counter_by_id: SQ_WAVES found");
    TEST_ASSERT(strcmp(counter->name, "SQ_WAVES") == 0, "SQ_WAVES has correct name");

    counter = lookup_counter_by_id(COUNTER_GL2C_HIT);
    TEST_ASSERT(counter != NULL, "lookup_counter_by_id: GL2C_HIT found");
    TEST_ASSERT(strcmp(counter->name, "GL2C_HIT") == 0, "GL2C_HIT has correct name");

    /* Test invalid counter IDs */
    counter = lookup_counter_by_id(0);
    TEST_ASSERT(counter == NULL, "lookup_counter_by_id: ID 0 returns NULL");

    counter = lookup_counter_by_id(COUNTER_ID_LAST);
    TEST_ASSERT(counter == NULL, "lookup_counter_by_id: ID >= COUNTER_ID_LAST returns NULL");

    counter = lookup_counter_by_id(-1);
    TEST_ASSERT(counter == NULL, "lookup_counter_by_id: Negative ID returns NULL");
}

/* Test lookup event ID */
void test_lookup_event_id(void) {
    printf("\n=== Testing lookup_event_id ===\n");

    /* Create GFX12 architecture for testing */
    arch_t* arch = arch_create_by_name("gfx12");
    TEST_ASSERT(arch != NULL, "GFX12 architecture created successfully");

    if (arch == NULL) {
        printf("ERROR: Could not create GFX12 architecture, skipping lookup_event_id tests\n");
        return;
    }

    /* Test valid GFX12 event lookups */
    const counter_def_t* counter = lookup_counter_by_id(COUNTER_SQ_WAVES);
    TEST_ASSERT(counter != NULL, "SQ_WAVES counter found");
    if (counter) {
        uint32_t event_id = lookup_event_id(counter, arch);
        TEST_ASSERT(event_id == 4, "lookup_event_id: SQ_WAVES GFX12 event ID is 4");
    }

    counter = lookup_counter_by_id(COUNTER_GL2C_HIT);
    TEST_ASSERT(counter != NULL, "GL2C_HIT counter found");
    if (counter) {
        uint32_t event_id = lookup_event_id(counter, arch);
        TEST_ASSERT(event_id == 41, "lookup_event_id: GL2C_HIT GFX12 event ID is 41");
    }

    counter = lookup_counter_by_id(COUNTER_SQ_BUSY_CYCLES);
    TEST_ASSERT(counter != NULL, "SQ_BUSY_CYCLES counter found");
    if (counter) {
        uint32_t event_id = lookup_event_id(counter, arch);
        TEST_ASSERT(event_id == 3, "lookup_event_id: SQ_BUSY_CYCLES GFX12 event ID is 3");
    }

    counter = lookup_counter_by_id(COUNTER_GRBM_COUNT);
    TEST_ASSERT(counter != NULL, "GRBM_COUNT counter found");
    if (counter) {
        uint32_t event_id = lookup_event_id(counter, arch);
        TEST_ASSERT(event_id == 0, "lookup_event_id: GRBM_COUNT GFX12 event ID is 0");
    }

    /* Test with NULL parameters */
    uint32_t event_id = lookup_event_id(NULL, arch);
    TEST_ASSERT(event_id == 0, "lookup_event_id: NULL counter returns 0");

    counter = lookup_counter_by_id(COUNTER_SQ_WAVES);
    if (counter) {
        event_id = lookup_event_id(counter, NULL);
        TEST_ASSERT(event_id == 0, "lookup_event_id: NULL architecture returns 0");
    }

    /* Clean up */
    arch_destroy(arch);
}

/* Test get all counters */
void test_get_all_counters(void) {
    printf("\n=== Testing get_all_counters ===\n");

    const counter_def_t* counters = get_all_counters();
    TEST_ASSERT(counters != NULL, "get_all_counters: Returns non-NULL pointer");

    size_t count = get_counter_count();
    TEST_ASSERT(count > 0, "get_counter_count: Returns positive count");
    TEST_ASSERT(count == 36, "get_counter_count: Returns expected count of 36");

    /* Verify first and last counters */
    TEST_ASSERT(counters[0].id == COUNTER_GL2C_EA_RDREQ, "First counter has correct ID");
    TEST_ASSERT(strcmp(counters[0].name, "GL2C_EA_RDREQ") == 0, "First counter has correct name");

    /* Verify some counters in the middle */
    int found_sq_waves = 0;
    int found_gl2c_hit = 0;
    int found_ta_busy = 0;

    for (size_t i = 0; i < count; i++) {
        if (counters[i].id == COUNTER_SQ_WAVES) {
            found_sq_waves = 1;
            TEST_ASSERT(strcmp(counters[i].name, "SQ_WAVES") == 0, "SQ_WAVES counter has correct name in array");
        }
        if (counters[i].id == COUNTER_GL2C_HIT) {
            found_gl2c_hit = 1;
            TEST_ASSERT(strcmp(counters[i].name, "GL2C_HIT") == 0, "GL2C_HIT counter has correct name in array");
        }
        if (counters[i].id == COUNTER_TA_TA_BUSY) {
            found_ta_busy = 1;
            TEST_ASSERT(strcmp(counters[i].name, "TA_TA_BUSY") == 0, "TA_TA_BUSY counter has correct name in array");
        }
    }

    TEST_ASSERT(found_sq_waves, "SQ_WAVES counter found in get_all_counters array");
    TEST_ASSERT(found_gl2c_hit, "GL2C_HIT counter found in get_all_counters array");
    TEST_ASSERT(found_ta_busy, "TA_TA_BUSY counter found in get_all_counters array");
}

/* Test block type coverage */
void test_block_type_coverage(void) {
    printf("\n=== Testing block type coverage ===\n");

    const counter_def_t* counters = get_all_counters();
    size_t count = get_counter_count();

    int gl2c_count = 0;
    int sq_count = 0;
    int ta_count = 0;
    int grbm_count = 0;

    for (size_t i = 0; i < count; i++) {
        switch (counters[i].hw_block) {
            case HW_IP_BLOCK_GL2C:
                gl2c_count++;
                break;
            case HW_IP_BLOCK_SQ:
                sq_count++;
                break;
            case HW_IP_BLOCK_TA:
                ta_count++;
                break;
            case HW_IP_BLOCK_GRBM:
                grbm_count++;
                break;
            default:
                break;
        }
    }

    TEST_ASSERT(gl2c_count == 9, "GL2C block has 9 counters");
    TEST_ASSERT(sq_count == 22, "SQ block has 22 counters");
    TEST_ASSERT(ta_count == 3, "TA block has 3 counters");
    TEST_ASSERT(grbm_count == 2, "GRBM block has 2 counters");

    TEST_ASSERT(gl2c_count + sq_count + ta_count + grbm_count == (int)count,
                "All counters are accounted for across block types");
}

int main(void) {
    printf("Starting counter registry tests...\n");

    test_lookup_counter_by_name();
    test_lookup_counter_by_id();
    test_lookup_event_id();
    test_get_all_counters();
    test_block_type_coverage();

    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);

    if (tests_passed == tests_run) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}