/**
 * @file test_dimension_decode.c
 * @brief Test dimension encoding/decoding functions
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* Include the dimension functions */
#include "../arch_creator_common.h"

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

/* GFX12 parameters for testing */
#define TEST_SE_COUNT       4
#define TEST_SA_COUNT       2
#define TEST_WGP_PER_SA     4

/* Test encoding function */
static void test_encode_dimension_index(void) {
    printf("\n=== Testing Dimension Encoding ===\n");

    /* Test known cases */
    TEST_ASSERT(encode_dimension_index(0, 0, 0, TEST_SA_COUNT, TEST_WGP_PER_SA) == 0,
                "SE=0, SA=0, WGP=0 encodes to index 0");
    TEST_ASSERT(encode_dimension_index(0, 0, 3, TEST_SA_COUNT, TEST_WGP_PER_SA) == 3,
                "SE=0, SA=0, WGP=3 encodes to index 3");
    TEST_ASSERT(encode_dimension_index(0, 1, 0, TEST_SA_COUNT, TEST_WGP_PER_SA) == 4,
                "SE=0, SA=1, WGP=0 encodes to index 4");
    TEST_ASSERT(encode_dimension_index(0, 1, 3, TEST_SA_COUNT, TEST_WGP_PER_SA) == 7,
                "SE=0, SA=1, WGP=3 encodes to index 7");
    TEST_ASSERT(encode_dimension_index(1, 0, 0, TEST_SA_COUNT, TEST_WGP_PER_SA) == 8,
                "SE=1, SA=0, WGP=0 encodes to index 8");
    TEST_ASSERT(encode_dimension_index(3, 1, 3, TEST_SA_COUNT, TEST_WGP_PER_SA) == 31,
                "SE=3, SA=1, WGP=3 encodes to index 31");
}

/* Test decoding function */
static void test_decode_dimension_index(void) {
    printf("\n=== Testing Dimension Decoding ===\n");

    /* Test known cases */
    dimension_coords_t coords;

    coords = decode_dimension_index(0, TEST_SA_COUNT, TEST_WGP_PER_SA);
    TEST_ASSERT(coords.se == 0 && coords.sa == 0 && coords.wgp == 0 && coords.flat_index == 0,
                "Index 0 decodes to SE=0, SA=0, WGP=0");

    coords = decode_dimension_index(3, TEST_SA_COUNT, TEST_WGP_PER_SA);
    TEST_ASSERT(coords.se == 0 && coords.sa == 0 && coords.wgp == 3 && coords.flat_index == 3,
                "Index 3 decodes to SE=0, SA=0, WGP=3");

    coords = decode_dimension_index(4, TEST_SA_COUNT, TEST_WGP_PER_SA);
    TEST_ASSERT(coords.se == 0 && coords.sa == 1 && coords.wgp == 0 && coords.flat_index == 4,
                "Index 4 decodes to SE=0, SA=1, WGP=0");

    coords = decode_dimension_index(7, TEST_SA_COUNT, TEST_WGP_PER_SA);
    TEST_ASSERT(coords.se == 0 && coords.sa == 1 && coords.wgp == 3 && coords.flat_index == 7,
                "Index 7 decodes to SE=0, SA=1, WGP=3");

    coords = decode_dimension_index(8, TEST_SA_COUNT, TEST_WGP_PER_SA);
    TEST_ASSERT(coords.se == 1 && coords.sa == 0 && coords.wgp == 0 && coords.flat_index == 8,
                "Index 8 decodes to SE=1, SA=0, WGP=0");

    coords = decode_dimension_index(31, TEST_SA_COUNT, TEST_WGP_PER_SA);
    TEST_ASSERT(coords.se == 3 && coords.sa == 1 && coords.wgp == 3 && coords.flat_index == 31,
                "Index 31 decodes to SE=3, SA=1, WGP=3");
}

/* Test round-trip encode/decode */
static void test_round_trip(void) {
    printf("\n=== Testing Round-trip Encode/Decode ===\n");

    /* Test all valid combinations */
    int all_passed = 1;
    for (uint32_t se = 0; se < TEST_SE_COUNT; se++) {
        for (uint32_t sa = 0; sa < TEST_SA_COUNT; sa++) {
            for (uint32_t wgp = 0; wgp < TEST_WGP_PER_SA; wgp++) {
                uint32_t encoded = encode_dimension_index(se, sa, wgp, TEST_SA_COUNT, TEST_WGP_PER_SA);
                dimension_coords_t decoded = decode_dimension_index(encoded, TEST_SA_COUNT, TEST_WGP_PER_SA);

                if (decoded.se != se || decoded.sa != sa || decoded.wgp != wgp || decoded.flat_index != encoded) {
                    all_passed = 0;
                    printf("FAIL: Round-trip failed for SE=%u, SA=%u, WGP=%u (encoded=%u, decoded=SE:%u SA:%u WGP:%u)\n",
                           se, sa, wgp, encoded, decoded.se, decoded.sa, decoded.wgp);
                }
            }
        }
    }

    TEST_ASSERT(all_passed, "All 32 round-trip encode/decode operations");
}

/* Test validation function */
static void test_validation(void) {
    printf("\n=== Testing Coordinate Validation ===\n");

    dimension_coords_t valid_coords = {0, 0, 0, 0};
    TEST_ASSERT(validate_dimension_coords(&valid_coords, TEST_SE_COUNT, TEST_SA_COUNT, TEST_WGP_PER_SA) == 1,
                "Valid coordinates (0,0,0) pass validation");

    dimension_coords_t valid_max = {TEST_SE_COUNT-1, TEST_SA_COUNT-1, TEST_WGP_PER_SA-1, 31};
    TEST_ASSERT(validate_dimension_coords(&valid_max, TEST_SE_COUNT, TEST_SA_COUNT, TEST_WGP_PER_SA) == 1,
                "Valid max coordinates pass validation");

    dimension_coords_t invalid_se = {TEST_SE_COUNT, 0, 0, 0};
    TEST_ASSERT(validate_dimension_coords(&invalid_se, TEST_SE_COUNT, TEST_SA_COUNT, TEST_WGP_PER_SA) == 0,
                "Invalid SE coordinate fails validation");

    dimension_coords_t invalid_sa = {0, TEST_SA_COUNT, 0, 0};
    TEST_ASSERT(validate_dimension_coords(&invalid_sa, TEST_SE_COUNT, TEST_SA_COUNT, TEST_WGP_PER_SA) == 0,
                "Invalid SA coordinate fails validation");

    dimension_coords_t invalid_wgp = {0, 0, TEST_WGP_PER_SA, 0};
    TEST_ASSERT(validate_dimension_coords(&invalid_wgp, TEST_SE_COUNT, TEST_SA_COUNT, TEST_WGP_PER_SA) == 0,
                "Invalid WGP coordinate fails validation");

    TEST_ASSERT(validate_dimension_coords(NULL, TEST_SE_COUNT, TEST_SA_COUNT, TEST_WGP_PER_SA) == 0,
                "NULL coordinates fail validation");
}

/* Test edge cases */
static void test_edge_cases(void) {
    printf("\n=== Testing Edge Cases ===\n");

    /* Test boundary values */
    uint32_t max_index = (TEST_SE_COUNT * TEST_SA_COUNT * TEST_WGP_PER_SA) - 1;
    dimension_coords_t coords = decode_dimension_index(max_index, TEST_SA_COUNT, TEST_WGP_PER_SA);
    TEST_ASSERT(coords.se == TEST_SE_COUNT-1 && coords.sa == TEST_SA_COUNT-1 && coords.wgp == TEST_WGP_PER_SA-1,
                "Maximum index decodes to maximum coordinates");

    /* Test that we can handle the full range */
    int sequential_test_passed = 1;
    for (uint32_t i = 0; i < 32; i++) {
        dimension_coords_t decoded = decode_dimension_index(i, TEST_SA_COUNT, TEST_WGP_PER_SA);
        uint32_t re_encoded = encode_dimension_index(decoded.se, decoded.sa, decoded.wgp, TEST_SA_COUNT, TEST_WGP_PER_SA);
        if (re_encoded != i) {
            sequential_test_passed = 0;
            break;
        }
    }
    TEST_ASSERT(sequential_test_passed, "Sequential decode/encode for all 32 indices");
}

/* Print a reference table */
static void print_reference_table(void) {
    printf("\n=== Reference Table (First 16 entries) ===\n");
    printf("Index | SE | SA | WGP | Formula\n");
    printf("------|----|----|-----|--------\n");

    for (uint32_t i = 0; i < 16 && i < 32; i++) {
        dimension_coords_t coords = decode_dimension_index(i, TEST_SA_COUNT, TEST_WGP_PER_SA);
        printf("%5u | %2u | %2u | %3u | (%u×2×4) + (%u×4) + %u = %u\n",
               i, coords.se, coords.sa, coords.wgp,
               coords.se, coords.sa, coords.wgp, i);
    }
    printf("  ... | ...| ...| ... | ...\n");

    /* Show last entry */
    dimension_coords_t coords = decode_dimension_index(31, TEST_SA_COUNT, TEST_WGP_PER_SA);
    printf("%5u | %2u | %2u | %3u | (%u×2×4) + (%u×4) + %u = 31\n",
           31, coords.se, coords.sa, coords.wgp,
           coords.se, coords.sa, coords.wgp);
}

/* Main test runner */
int main(void) {
    printf("=== Dimension Encode/Decode Test Suite ===\n");
    printf("Testing with SE_COUNT=%d, SA_COUNT=%d, WGP_PER_SA=%d (Total: %d instances)\n",
           TEST_SE_COUNT, TEST_SA_COUNT, TEST_WGP_PER_SA,
           TEST_SE_COUNT * TEST_SA_COUNT * TEST_WGP_PER_SA);

    test_encode_dimension_index();
    test_decode_dimension_index();
    test_round_trip();
    test_validation();
    test_edge_cases();
    print_reference_table();

    printf("\n=== Test Results ===\n");
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