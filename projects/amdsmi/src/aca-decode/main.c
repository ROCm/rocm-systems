/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * @file main.c
 * @brief Demo program showing how to use the ACA decoder
 *
 * This is a demonstration program that shows how to use the ACA decoder
 * with sample raw data to decode ACA error information.
 */

#include <stdio.h>
#include <aca_api.h>
#include <aca_version.h>
#include <aca_constants.h>
#include <stdint.h>
#include <inttypes.h>

// Function prototype
void print_error_info(const aca_error_info_t *info);
void print_version_info(void);

// Function to print error info in JSON format
void print_error_info(const aca_error_info_t *info)
{
    printf("{\n");
    printf("   \"bank\": \"%s\",\n", info->bank_ref);
    printf("   \"error_location\": {\n");
    printf("      \"oam\": \"%d\",\n", info->oam);
    printf("      \"aid\": \"%d\",\n", info->aid);
    printf("      \"instance\": \"%s\"\n", info->instance_ref);
    printf("   },\n");
    printf("   \"severity\": \"%s\",\n", info->severity_ref);
    printf("   \"afid\": \"%d\",\n", info->afid);
    printf("   \"scrub\": \"%u\",\n", info->scrub);
    printf("   \"err_ext\": \"%u\",\n", info->error_code_ext);
    printf("   \"error_category\": \"%s\",\n", info->category_ref);
    printf("   \"error_type\": \"%s\",\n", info->error_type_ref);
    printf("   \"address\": \"0x%" PRIx64 "\",\n", info->raw_addr);
    printf("   \"syndrome\": \"0x%" PRIx64 "\"\n", info->raw_synd);
    printf("}\n");
}

// Function to print version information
void print_version_info(void)
{
    printf("=== ACA Decoder Library Version Information ===\n");
    printf("Version: %s\n", aca_get_version_string());
    printf("Major: %d\n", aca_get_version_major());
    printf("Minor: %d\n", aca_get_version_minor());
    printf("Patch: %d\n", aca_get_version_patch());

    aca_version_info_t version_info = aca_get_version_info();
    printf("Complete version info:\n");
    printf("  Major: %d\n", version_info.major);
    printf("  Minor: %d\n", version_info.minor);
    printf("  Patch: %d\n", version_info.patch);
    printf("  String: %s\n", version_info.string);
    printf("===============================================\n\n");
}

int main()
{
    // Display version information
    print_version_info();

    // Sample usage of decode_afid with 32-byte register array (HBM FATAL ERROR, expected output is 4)
    uint64_t register_array_32[ACA_REGISTER_ARRAY_SIZE_32_BYTES] = {0xbaa000000004081b, 0x0, 0x209600090f00, 0x5d000000};
    int afid_32 = decode_afid(register_array_32, ACA_REGISTER_ARRAY_SIZE_32_BYTES, 0, 1);
    printf("Decoded AFID (32-byte array): %d\n", afid_32);

    // Sample usage of decode_afid with 32-byte register array (GC FATAL ERROR, expected output is 3)
    uint64_t register_array_test[ACA_REGISTER_ARRAY_SIZE_32_BYTES] = {0xbea00000003b0000, 0x100000029, 0x1200136430400, 0x20b};
    int afid_test = decode_afid(register_array_test, ACA_REGISTER_ARRAY_SIZE_32_BYTES, 0, 1);
    printf("Decoded AFID (test array): %d\n", afid_test);

    // Sample usage of decode_afid with 128-byte register array (HBM CORRECTED ERROR, expected output is 1)
    uint64_t register_array_128[ACA_REGISTER_ARRAY_SIZE_128_BYTES] = {
        0xffff,
        0xdc2040000000011b,
        0x0,
        0xd008000801000000,
        0x25000001ff,
        0x209600191f00,
        0xa000000,
        0x0,
        0x0,
        0x0,
        0xd008000801000000,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0};
    int afid_128 = decode_afid(register_array_128, ACA_REGISTER_ARRAY_SIZE_128_BYTES, 0, 1);
    printf("Decoded AFID (128-byte array): %d\n", afid_128);

    // sample for bad page
    uint64_t register_array_bad_page[ACA_REGISTER_ARRAY_SIZE_128_BYTES] = {
        0x1,
        0xb000000000000137,
        0x0,
        0x0,
        0x1ff00000002,
        0x9600000000,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0};

    // when flag is 0b1000, it indicates that the error threshold has been exceeded
    // and is always a HBM error. The expected output is 19.
    int afid_bad_page = decode_afid(register_array_bad_page, ACA_REGISTER_ARRAY_SIZE_128_BYTES, ACA_FLAG_THRESHOLD_EXCEEDED, 1);
    printf("Decoded AFID (bad page): %d\n", afid_bad_page);

    const aca_error_info_t error_info_32 = decode_error_info(register_array_32, ACA_REGISTER_ARRAY_SIZE_32_BYTES, 0, 1);
    print_error_info(&error_info_32);

    const aca_error_info_t error_info_128 = decode_error_info(register_array_128, ACA_REGISTER_ARRAY_SIZE_128_BYTES, 0, 1);
    print_error_info(&error_info_128);

    return 0;
}
