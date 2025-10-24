/**
 * @file test_dimension_helpers.c
 * @brief Unit tests for dimension helper functions
 *
 * Tests pmu_extract_dimensions(), pmu_validate_dimensions(), and related
 * dimension encoding/decoding functionality.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

/* Include the dimension header */
#include "../../../pmu_dimension.h"
#include "../arch_creator_common.h"

/* Test result tracking */
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    printf("Running test: %s\n", #name); \
    test_##name()

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            printf("  FAIL: %s (line %d)\n", message, __LINE__); \
            tests_failed++; \
            return; \
        } \
    } while(0)

#define PASS() \
    do { \
        printf("  PASS\n"); \
        tests_passed++; \
    } while(0)

/**
 * Test basic dimension extraction from config1
 */
static void test_extract_basic_dimensions(void)
{
    struct pmu_dimension_coords dims = {0};
    uint64_t config1;

    /* Test case 1: SE=2, SA=1, WGP=3 */
    config1 = (2ULL << 8) | (1ULL << 16) | (3ULL << 24);
    pmu_extract_dimensions(config1, &dims);

    ASSERT(dims.valid == true, "Dimensions should be valid when config1 != 0");
    ASSERT(dims.xcc == 0, "XCC should be 0");
    ASSERT(dims.se == 2, "SE should be 2");
    ASSERT(dims.sa == 1, "SA should be 1");
    ASSERT(dims.wgp == 3, "WGP should be 3");
    ASSERT(dims.cu == 0, "CU should be 0");
    ASSERT(dims.aggregate == false, "Aggregate should be false");

    PASS();
}

/**
 * Test all dimension fields
 */
static void test_extract_all_dimensions(void)
{
    struct pmu_dimension_coords dims = {0};
    uint64_t config1;

    /* Test case: XCC=1, SE=3, SA=1, WGP=2, CU=5 */
    config1 = (1ULL << 0) | (3ULL << 8) | (1ULL << 16) | (2ULL << 24) | (5ULL << 32);
    pmu_extract_dimensions(config1, &dims);

    ASSERT(dims.valid == true, "Dimensions should be valid");
    ASSERT(dims.xcc == 1, "XCC should be 1");
    ASSERT(dims.se == 3, "SE should be 3");
    ASSERT(dims.sa == 1, "SA should be 1");
    ASSERT(dims.wgp == 2, "WGP should be 2");
    ASSERT(dims.cu == 5, "CU should be 5");

    PASS();
}

/**
 * Test aggregate flag
 */
static void test_extract_flags(void)
{
    struct pmu_dimension_coords dims = {0};
    uint64_t config1;

    /* Test case: SE=1 with aggregate flag */
    config1 = (1ULL << 8) | (1ULL << 40);  /* SE=1, aggregate=1 */
    pmu_extract_dimensions(config1, &dims);

    ASSERT(dims.valid == true, "Dimensions should be valid");
    ASSERT(dims.se == 1, "SE should be 1");
    ASSERT(dims.aggregate == true, "Aggregate should be true");

    /* Test case: aggregate flag alone */
    config1 = (1ULL << 40);  /* aggregate=1 */
    pmu_extract_dimensions(config1, &dims);

    ASSERT(dims.valid == true, "Dimensions should be valid");
    ASSERT(dims.aggregate == true, "Aggregate should be true");

    PASS();
}

/**
 * Test config1=0 (no dimensions specified)
 */
static void test_extract_zero_config(void)
{
    struct pmu_dimension_coords dims = {0};
    uint64_t config1 = 0;

    pmu_extract_dimensions(config1, &dims);

    ASSERT(dims.valid == false, "Dimensions should not be valid when config1=0");
    ASSERT(dims.xcc == 0, "All fields should be zero");
    ASSERT(dims.se == 0, "All fields should be zero");
    ASSERT(dims.sa == 0, "All fields should be zero");
    ASSERT(dims.wgp == 0, "All fields should be zero");
    ASSERT(dims.cu == 0, "All fields should be zero");

    PASS();
}

/**
 * Test dimension validation with valid coordinates
 */
static void test_validate_valid_dimensions(void)
{
    struct pmu_dimension_coords dims = {0};
    struct pmu_dimension_limits limits = {
        .max_xcc = 0,  /* 1 XCC */
        .max_se = 3,   /* 4 SEs */
        .max_sa = 1,   /* 2 SAs */
        .max_wgp = 3,  /* 4 WGPs */
        .max_cu = 63   /* 64 CUs */
    };

    /* Test valid coordinates at minimum */
    dims.xcc = 0;
    dims.se = 0;
    dims.sa = 0;
    dims.wgp = 0;
    dims.cu = 0;
    dims.valid = true;

    ASSERT(pmu_validate_dimensions(&dims, &limits) == true,
           "Min coordinates should be valid");

    /* Test valid coordinates at maximum */
    dims.xcc = 0;
    dims.se = 3;
    dims.sa = 1;
    dims.wgp = 3;
    dims.cu = 63;

    ASSERT(pmu_validate_dimensions(&dims, &limits) == true,
           "Max coordinates should be valid");

    /* Test mid-range coordinates */
    dims.xcc = 0;
    dims.se = 2;
    dims.sa = 1;
    dims.wgp = 2;
    dims.cu = 30;

    ASSERT(pmu_validate_dimensions(&dims, &limits) == true,
           "Mid-range coordinates should be valid");

    PASS();
}

/**
 * Test dimension validation with invalid coordinates
 */
static void test_validate_invalid_dimensions(void)
{
    struct pmu_dimension_coords dims = {0};
    struct pmu_dimension_limits limits = {
        .max_xcc = 0,  /* 1 XCC */
        .max_se = 3,   /* 4 SEs */
        .max_sa = 1,   /* 2 SAs */
        .max_wgp = 3,  /* 4 WGPs */
        .max_cu = 63   /* 64 CUs */
    };

    /* Test SE out of range */
    dims.xcc = 0;
    dims.se = 4;  /* Too large */
    dims.sa = 0;
    dims.wgp = 0;
    dims.cu = 0;
    dims.valid = true;

    ASSERT(pmu_validate_dimensions(&dims, &limits) == false,
           "SE out of range should be invalid");

    /* Test SA out of range */
    dims.se = 0;
    dims.sa = 2;  /* Too large */

    ASSERT(pmu_validate_dimensions(&dims, &limits) == false,
           "SA out of range should be invalid");

    /* Test WGP out of range */
    dims.sa = 0;
    dims.wgp = 4;  /* Too large */

    ASSERT(pmu_validate_dimensions(&dims, &limits) == false,
           "WGP out of range should be invalid");

    /* Test CU out of range */
    dims.wgp = 0;
    dims.cu = 64;  /* Too large */

    ASSERT(pmu_validate_dimensions(&dims, &limits) == false,
           "CU out of range should be invalid");

    PASS();
}

/**
 * Test dimension validation with invalid flag
 */
static void test_validate_invalid_flag(void)
{
    struct pmu_dimension_coords dims = {0};
    struct pmu_dimension_limits limits = {
        .max_xcc = 0,
        .max_se = 3,
        .max_sa = 1,
        .max_wgp = 3,
        .max_cu = 63
    };

    /* Valid coordinates but invalid flag */
    dims.xcc = 0;
    dims.se = 2;
    dims.sa = 1;
    dims.wgp = 2;
    dims.cu = 0;
    dims.valid = false;  /* Not valid */

    ASSERT(pmu_validate_dimensions(&dims, &limits) == true,
           "Validation should still check ranges even if valid=false");

    PASS();
}

/**
 * Test encode_dimension_index helper
 */
static void test_encode_dimension_index(void)
{
    uint32_t flat_index;

    /* GFX12 has: 4 SEs, 2 SAs, 4 WGPs per SA */
    uint32_t sa_count = 2;
    uint32_t wgp_per_sa = 4;

    /* Test case 1: SE=0, SA=0, WGP=0 */
    flat_index = encode_dimension_index(0, 0, 0, sa_count, wgp_per_sa);
    ASSERT(flat_index == 0, "First index should be 0");

    /* Test case 2: SE=0, SA=0, WGP=1 */
    flat_index = encode_dimension_index(0, 0, 1, sa_count, wgp_per_sa);
    ASSERT(flat_index == 1, "SE=0,SA=0,WGP=1 should be index 1");

    /* Test case 3: SE=0, SA=1, WGP=0 */
    flat_index = encode_dimension_index(0, 1, 0, sa_count, wgp_per_sa);
    ASSERT(flat_index == 4, "SE=0,SA=1,WGP=0 should be index 4");

    /* Test case 4: SE=1, SA=0, WGP=0 */
    flat_index = encode_dimension_index(1, 0, 0, sa_count, wgp_per_sa);
    ASSERT(flat_index == 8, "SE=1,SA=0,WGP=0 should be index 8");

    /* Test case 5: SE=2, SA=1, WGP=3 */
    flat_index = encode_dimension_index(2, 1, 3, sa_count, wgp_per_sa);
    ASSERT(flat_index == 23, "SE=2,SA=1,WGP=3 should be index 23");

    PASS();
}

/**
 * Test decode_dimension_index helper
 */
static void test_decode_dimension_index(void)
{
    dimension_coords_t coords;
    uint32_t sa_count = 2;
    uint32_t wgp_per_sa = 4;

    /* Test case 1: Index 0 */
    coords = decode_dimension_index(0, sa_count, wgp_per_sa);
    ASSERT(coords.se == 0, "Index 0: SE should be 0");
    ASSERT(coords.sa == 0, "Index 0: SA should be 0");
    ASSERT(coords.wgp == 0, "Index 0: WGP should be 0");

    /* Test case 2: Index 5 */
    coords = decode_dimension_index(5, sa_count, wgp_per_sa);
    ASSERT(coords.se == 0, "Index 5: SE should be 0");
    ASSERT(coords.sa == 1, "Index 5: SA should be 1");
    ASSERT(coords.wgp == 1, "Index 5: WGP should be 1");

    /* Test case 3: Index 8 */
    coords = decode_dimension_index(8, sa_count, wgp_per_sa);
    ASSERT(coords.se == 1, "Index 8: SE should be 1");
    ASSERT(coords.sa == 0, "Index 8: SA should be 0");
    ASSERT(coords.wgp == 0, "Index 8: WGP should be 0");

    /* Test case 4: Index 23 */
    coords = decode_dimension_index(23, sa_count, wgp_per_sa);
    ASSERT(coords.se == 2, "Index 23: SE should be 2");
    ASSERT(coords.sa == 1, "Index 23: SA should be 1");
    ASSERT(coords.wgp == 3, "Index 23: WGP should be 3");

    PASS();
}

/**
 * Test encode/decode round-trip
 */
static void test_encode_decode_roundtrip(void)
{
    uint32_t sa_count = 2;
    uint32_t wgp_per_sa = 4;

    /* Test all valid SE/SA/WGP combinations for GFX12 */
    for (uint32_t se = 0; se < 4; se++) {
        for (uint32_t sa = 0; sa < 2; sa++) {
            for (uint32_t wgp = 0; wgp < 4; wgp++) {
                uint32_t flat = encode_dimension_index(se, sa, wgp, sa_count, wgp_per_sa);
                dimension_coords_t coords = decode_dimension_index(flat, sa_count, wgp_per_sa);

                ASSERT(coords.se == se, "Round-trip: SE mismatch");
                ASSERT(coords.sa == sa, "Round-trip: SA mismatch");
                ASSERT(coords.wgp == wgp, "Round-trip: WGP mismatch");
                ASSERT(coords.flat_index == flat, "Round-trip: flat index mismatch");
            }
        }
    }

    PASS();
}

/**
 * Test validate_dimension_coords helper
 */
static void test_validate_dimension_coords(void)
{
    dimension_coords_t coords;
    uint32_t se_count = 4;
    uint32_t sa_count = 2;
    uint32_t wgp_per_sa = 4;

    /* Valid coordinates */
    coords.se = 2;
    coords.sa = 1;
    coords.wgp = 3;

    ASSERT(validate_dimension_coords(&coords, se_count, sa_count, wgp_per_sa) == 1,
           "Valid coordinates should pass");

    /* SE out of range */
    coords.se = 4;
    coords.sa = 1;
    coords.wgp = 0;

    ASSERT(validate_dimension_coords(&coords, se_count, sa_count, wgp_per_sa) == 0,
           "SE out of range should fail");

    /* SA out of range */
    coords.se = 0;
    coords.sa = 2;
    coords.wgp = 0;

    ASSERT(validate_dimension_coords(&coords, se_count, sa_count, wgp_per_sa) == 0,
           "SA out of range should fail");

    /* WGP out of range */
    coords.se = 0;
    coords.sa = 0;
    coords.wgp = 4;

    ASSERT(validate_dimension_coords(&coords, se_count, sa_count, wgp_per_sa) == 0,
           "WGP out of range should fail");

    /* NULL pointer */
    ASSERT(validate_dimension_coords(NULL, se_count, sa_count, wgp_per_sa) == 0,
           "NULL pointer should fail");

    PASS();
}

/**
 * Test max value extraction (255 for each field)
 */
static void test_extract_max_values(void)
{
    struct pmu_dimension_coords dims = {0};
    uint64_t config1;

    /* All fields at maximum (8 bits = 255) */
    config1 = (255ULL << 0) |   /* XCC */
              (255ULL << 8) |   /* SE */
              (255ULL << 16) |  /* SA */
              (255ULL << 24) |  /* WGP */
              (255ULL << 32);   /* CU */

    pmu_extract_dimensions(config1, &dims);

    ASSERT(dims.valid == true, "Dimensions should be valid");
    ASSERT(dims.xcc == 255, "XCC should be 255");
    ASSERT(dims.se == 255, "SE should be 255");
    ASSERT(dims.sa == 255, "SA should be 255");
    ASSERT(dims.wgp == 255, "WGP should be 255");
    ASSERT(dims.cu == 255, "CU should be 255");

    PASS();
}

/**
 * Main test runner
 */
int main(void)
{
    printf("=== Dimension Helper Unit Tests ===\n\n");

    /* Extraction tests */
    TEST(extract_basic_dimensions);
    TEST(extract_all_dimensions);
    TEST(extract_flags);
    TEST(extract_zero_config);
    TEST(extract_max_values);

    /* Validation tests */
    TEST(validate_valid_dimensions);
    TEST(validate_invalid_dimensions);
    TEST(validate_invalid_flag);

    /* Encode/decode tests */
    TEST(encode_dimension_index);
    TEST(decode_dimension_index);
    TEST(encode_decode_roundtrip);
    TEST(validate_dimension_coords);

    /* Summary */
    printf("\n=== Test Summary ===\n");
    printf("PASSED: %d\n", tests_passed);
    printf("FAILED: %d\n", tests_failed);
    printf("TOTAL:  %d\n", tests_passed + tests_failed);

    return (tests_failed == 0) ? 0 : 1;
}
