
/*
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rocjpeg_api_negative_tests.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>

namespace {

/**
 * @brief A minimal 16x16 4:2:0 baseline JPEG used as a fuzzing seed.
 *
 * Mutating a well formed stream reaches far more of the parser than random
 * bytes do, because most of the marker structure survives each mutation.
 */
const std::vector<uint8_t> &ColorSeedJpeg() {
    static const std::vector<uint8_t> kSeed = {
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x28, 0x1C, 0x1E, 0x23, 0x1E, 0x19, 0x28, 0x23, 0x21, 0x23, 0x2D,
        0x2B, 0x28, 0x30, 0x3C, 0x64, 0x41, 0x3C, 0x37, 0x37, 0x3C, 0x7B, 0x58,
        0x5D, 0x49, 0x64, 0x91, 0x80, 0x99, 0x96, 0x8F, 0x80, 0x8C, 0x8A, 0xA0,
        0xB4, 0xE6, 0xC3, 0xA0, 0xAA, 0xDA, 0xAD, 0x8A, 0x8C, 0xC8, 0xFF, 0xCB,
        0xDA, 0xEE, 0xF5, 0xFF, 0xFF, 0xFF, 0x9B, 0xC1, 0xFF, 0xFF, 0xFF, 0xFA,
        0xFF, 0xE6, 0xFD, 0xFF, 0xF8, 0xFF, 0xDB, 0x00, 0x43, 0x01, 0x2B, 0x2D,
        0x2D, 0x3C, 0x35, 0x3C, 0x76, 0x41, 0x41, 0x76, 0xF8, 0xA5, 0x8C, 0xA5,
        0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8,
        0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8,
        0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8,
        0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8,
        0xF8, 0xF8, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x10, 0x00, 0x10, 0x03,
        0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xFF, 0xC4, 0x00,
        0x15, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0xFF, 0xC4, 0x00, 0x14,
        0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xC4, 0x00, 0x14, 0x01, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x03, 0xFF, 0xC4, 0x00, 0x14, 0x11, 0x01, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03,
        0x11, 0x00, 0x3F, 0x00, 0x94, 0x01, 0x1D, 0xFF, 0xD9
    };
    return kSeed;
}

/**
 * @brief A minimal 8x8 grayscale (4:0:0) baseline JPEG used as a fuzzing seed.
 *
 * The single-component path takes different branches than the color path in
 * both the SOF component loop and the chroma subsampling classifier.
 */
const std::vector<uint8_t> &GraySeedJpeg() {
    static const std::vector<uint8_t> kSeed = {
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x28, 0x1C, 0x1E, 0x23, 0x1E, 0x19, 0x28, 0x23, 0x21, 0x23, 0x2D,
        0x2B, 0x28, 0x30, 0x3C, 0x64, 0x41, 0x3C, 0x37, 0x37, 0x3C, 0x7B, 0x58,
        0x5D, 0x49, 0x64, 0x91, 0x80, 0x99, 0x96, 0x8F, 0x80, 0x8C, 0x8A, 0xA0,
        0xB4, 0xE6, 0xC3, 0xA0, 0xAA, 0xDA, 0xAD, 0x8A, 0x8C, 0xC8, 0xFF, 0xCB,
        0xDA, 0xEE, 0xF5, 0xFF, 0xFF, 0xFF, 0x9B, 0xC1, 0xFF, 0xFF, 0xFF, 0xFA,
        0xFF, 0xE6, 0xFD, 0xFF, 0xF8, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x08,
        0x00, 0x08, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x03, 0xFF, 0xC4, 0x00, 0x14, 0x10, 0x01, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,
        0x17, 0xFF, 0xD9
    };
    return kSeed;
}

/**
 * @brief A reproducible pseudo random number generator (xorshift32).
 *
 * A fuzz failure is only actionable if it can be replayed, so the generator is
 * seeded deterministically and the seed is printed with every run.
 */
class FuzzRandom {
    public:
        explicit FuzzRandom(uint32_t seed) : state_(seed == 0 ? 1u : seed) {}
        uint32_t Next() {
            state_ ^= state_ << 13;
            state_ ^= state_ >> 17;
            state_ ^= state_ << 5;
            return state_;
        }
        /** @brief Returns a value in [0, bound), or 0 when bound is 0. */
        uint32_t Below(uint32_t bound) { return bound == 0 ? 0 : Next() % bound; }
    private:
        uint32_t state_;
};

/**
 * @brief Renders a stream as hex so a failing fuzz case can be pasted back in.
 *
 * Long streams are elided in the middle; the head holds the marker structure
 * and the tail holds the truncation point, which is what a triage usually needs.
 */
std::string HexDump(const std::vector<uint8_t> &data) {
    static constexpr size_t kHead = 48;
    static constexpr size_t kTail = 16;
    std::ostringstream oss;
    oss << std::uppercase << std::hex << std::setfill('0');
    oss << "size=" << std::dec << data.size() << " bytes:";
    const bool elide = data.size() > kHead + kTail;
    const size_t head_count = elide ? kHead : data.size();
    for (size_t i = 0; i < head_count; i++) {
        oss << (i % 16 == 0 ? "\n  " : " ") << std::hex << std::setw(2) << static_cast<int>(data[i]);
    }
    if (elide) {
        oss << "\n  ... " << std::dec << (data.size() - kHead - kTail) << " bytes omitted ...";
        for (size_t i = data.size() - kTail; i < data.size(); i++) {
            oss << " " << std::hex << std::setw(2) << static_cast<int>(data[i]);
        }
    }
    return oss.str();
}

/**
 * @brief Reports whether a status is one of the documented RocJpegStatus values.
 *
 * A status outside the enum means the parser returned uninitialized or
 * corrupted state rather than a decision about the stream.
 */
bool IsKnownStatus(RocJpegStatus status) {
    return status <= ROCJPEG_STATUS_SUCCESS && status > ROCJPEG_STATUS_MAX_VALUE;
}

/**
 * @brief Finds the offset of the first occurrence of a marker in a stream.
 *
 * @param data The stream to search.
 * @param marker The marker code that follows the 0xFF prefix.
 * @return The offset of the 0xFF byte, or data.size() when the marker is absent.
 */
size_t FindMarker(const std::vector<uint8_t> &data, uint8_t marker) {
    for (size_t i = 0; i + 1 < data.size(); i++) {
        if (data[i] == 0xFF && data[i + 1] == marker) {
            return i;
        }
    }
    return data.size();
}

/**
 * @brief Reads an unsigned environment variable, falling back to a default.
 *
 * The whole string has to be a positive number that fits in a uint32_t. A
 * value that is out of range, negative or only partly numeric is rejected
 * rather than narrowed, because narrowing would quietly change how much
 * fuzzing runs: 4294967296 would wrap to zero and skip the fuzz loops
 * altogether, and "100junk" would be read as 100. A rejected value is
 * reported so that a typo in the override does not look like a clean run.
 */
uint32_t EnvOrDefault(const char *name, uint32_t fallback) {
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    // strtoull negates a leading minus into the unsigned range, so "-18446744073709551615"
    // would come back as 1. Reject the sign before parsing instead.
    const char *digits = value;
    while (*digits == ' ' || *digits == '\t') {
        digits++;
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(digits, &end, 0);
    if (*digits != '-' && end != digits && *end == '\0' && parsed > 0 && parsed <= UINT32_MAX) {
        return static_cast<uint32_t>(parsed);
    }
    std::cerr << "warning: ignoring " << name << "=" << value
              << ", expected a number between 1 and " << UINT32_MAX
              << "; using " << fallback << std::endl;
    return fallback;
}

}  // namespace

RocJpegApiNegativeTests:: RocJpegApiNegativeTests() {};

RocJpegApiNegativeTests::~RocJpegApiNegativeTests() {
    RocJpegStatus rocjpeg_status = rocJpegDestroy(rocjpeg_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        std::cerr << "Failed to destroy rocjpeg handle: " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
    }
    rocjpeg_status = rocJpegStreamDestroy(rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        std::cerr << "Failed to destroy rocjpeg stream handle " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
    }
}

int RocJpegApiNegativeTests::TestInvalidStreamCreate() {
    std::cout << "info: Executing negative test cases for the rocJpegStreamCreate API" << std::endl;
    //Scenario 1: Pass nullptr for jpeg_stream_handle
    RocJpegStatus rocjpeg_status = rocJpegStreamCreate(nullptr);
    if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
        std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }
    // Create a valid rocJPEG stream handle - This step ensures a valid rocjpeg_stream_handle_ is available for subsequent negative testing of other rocJPEG parser APIs.
    rocjpeg_status = rocJpegStreamCreate(&rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        std::cerr << "Expected ROCJPEG_STATUS_SUCCESS but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int RocJpegApiNegativeTests::TestInvalidStreamParse() {
    std::cout << "info: Executing negative test cases for the rocJpegStreamParse API" << std::endl;

    // Scenario 1: Pass nullptr for data and jpeg_stream_handle
    RocJpegStatus rocjpeg_status = rocJpegStreamParse(nullptr, 0, nullptr);
    if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
        std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 2: Pass a valid jpeg_stream_handle but nullptr for data
    rocjpeg_status = rocJpegStreamParse(nullptr, 0, rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
        std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 3: Invalid SOI marker
    std::vector<uint8_t> invalid_soi_data = {0xFF, 0x00}; // Invalid SOI marker
    rocjpeg_status = rocJpegStreamParse(invalid_soi_data.data(), invalid_soi_data.size(), rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_BAD_JPEG) {
        std::cerr << "Expected ROCJPEG_STATUS_BAD_JPEG but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 4: Invalid DRI marker
    std::vector<uint8_t> invalid_dri_data = {0xFF, 0xD8, 0xFF, 0xDD, 0x00, 0x03}; // Invalid DRI marker length
    rocjpeg_status = rocJpegStreamParse(invalid_dri_data.data(), invalid_dri_data.size(), rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_BAD_JPEG) {
        std::cerr << "Expected ROCJPEG_STATUS_BAD_JPEG but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 5: Invalid SOS marker - provide an invalid number of components (e.g., the number of components cannot exceed 3, but 4 is provided)
    std::vector<uint8_t> invalid_sos_data = {0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x01, 0x04}; // Invalid number of component
    rocjpeg_status = rocJpegStreamParse(invalid_sos_data.data(), invalid_sos_data.size(), rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_BAD_JPEG) {
        std::cerr << "Expected ROCJPEG_STATUS_BAD_JPEG but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 6: Invalid number of quantization tables in the DQT marker
    std::vector<uint8_t> invalid_quantization_data = {0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x03, 0x1F}; // Invalid quantization table
    rocjpeg_status = rocJpegStreamParse(invalid_quantization_data.data(), invalid_quantization_data.size(), rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_BAD_JPEG) {
        std::cerr << "Expected ROCJPEG_STATUS_BAD_JPEG but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 7: Invalid number of Huffman tables in the DHT marker
    std::vector<uint8_t> invalid_huffman_table_data = {0xFF, 0xD8, 0xFF, 0xC4, 0x00, 0x03, 0x02}; // Too many Huffman tables
    rocjpeg_status = rocJpegStreamParse(invalid_huffman_table_data.data(), invalid_huffman_table_data.size(), rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_BAD_JPEG) {
        std::cerr << "Expected ROCJPEG_STATUS_BAD_JPEG for invalid number of Huffman tables but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 8: Invalid AC Huffman table in the DHT marker
    std::vector<uint8_t> invalid_ac_huffman_table_data = {
        0xFF, 0xD8, //SOI
        0xFF, 0xC4, 0x00, 0x03, 0x10, // DHT with AC Hufman table
        0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0xA3 // Array of the invalid number of AC codes - the count of values cannot exceed 0xA2, but 0xA3 is provided 
    };
    rocjpeg_status = rocJpegStreamParse(invalid_ac_huffman_table_data.data(), invalid_ac_huffman_table_data.size(), rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_BAD_JPEG) {
        std::cerr << "Expected ROCJPEG_STATUS_BAD_JPEG but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 9: Invalid DC Huffman table in the DHT marker
    std::vector<uint8_t> invalid_dc_huffman_table_data = {
        0xFF, 0xD8, // SOI
        0xFF, 0xC4, 0x00, 0x03, 0x01, // DHT with DC Hufman table
        0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0x0D // Array of the invalid number of DC codes - the count of values cannot exceed 0x0C, but 0x0D is provided 
    }; // Invalid DC Huffman table
    rocjpeg_status = rocJpegStreamParse(invalid_dc_huffman_table_data.data(), invalid_dc_huffman_table_data.size(), rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_BAD_JPEG) {
        std::cerr << "Expected ROCJPEG_STATUS_BAD_JPEG but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 10: invalid number of JPEG component in the SOF marker
    std::vector<uint8_t> invalid_num_component_data = {
        0xFF, 0xD8, // SOI
        0xFF, 0xC0, 0x00, 0x08, // Invalid SOF with the number of component is set to 4
        0x08, 0x00, 0x10, 0x00, 0x10, 0x04
    };
    rocjpeg_status = rocJpegStreamParse(invalid_num_component_data.data(), invalid_num_component_data.size(), rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_BAD_JPEG) {
        std::cerr << "Expected ROCJPEG_STATUS_BAD_JPEG but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 11: Invalid quantization table selector specified in the SOF marker
    std::vector<uint8_t> Invalid_quantization_table_selector_data = {
        0xFF, 0xD8, // SOI
        0xFF, 0xC0, 0x00, 0x0B, // SOF with 3 components with invalid quantization table selector is set to 4
        0x08, 0x00, 0x10, 0x00, 0x10, 0x03, 0x00, 0x00, 0x04
    };
    rocjpeg_status = rocJpegStreamParse(Invalid_quantization_table_selector_data.data(), Invalid_quantization_table_selector_data.size(), rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_BAD_JPEG) {
        std::cerr << "Expected ROCJPEG_STATUS_BAD_JPEG but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 11: Mismatch in the number of components between the SOS and SOF markers
    std::vector<uint8_t> component_mismatch_data = {
        0xFF, 0xD8, // SOI
        0xFF, 0xC0, 0x00, 0x11, // SOF with 3 components
        0x08, 0x00, 0x10, 0x00, 0x10, 0x03, 0x01, 0xFF, 0x00, 0x02, 0xFF, 0x01, 0x03, 0xFF, 0x02,
        0xFF, 0xDA, 0x00, 0x07, // SOS with 2 components (mismatch)
        0x01, 0x00, 0x02, 0x11, 0x00
    };
    rocjpeg_status = rocJpegStreamParse(component_mismatch_data.data(), component_mismatch_data.size(), rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_BAD_JPEG) {
        std::cerr << "Expected ROCJPEG_STATUS_BAD_JPEG but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 12: Invalid AC Huffman table selector in the SOS marker
    std::vector<uint8_t> invalid_ac_huffman_sos_data = {
        0xFF, 0xD8, // SOI
        0xFF, 0xDA, 0x00, 0x07, // SOS with invalid number of AC Huffman table
        0x01, 0x00, 0x04, 0x11, 0x00
    };
    rocjpeg_status = rocJpegStreamParse(invalid_ac_huffman_sos_data.data(), invalid_ac_huffman_sos_data.size(), rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_BAD_JPEG) {
        std::cerr << "Expected ROCJPEG_STATUS_BAD_JPEG but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 13: Invalid DC Huffman table selector in the SOS marker
    std::vector<uint8_t> invalid_dc_huffman_sos_data = {
        0xFF, 0xD8, // SOI
        0xFF, 0xDA, 0x00, 0x07, // SOS with invalid number of DC Huffman table
        0x01, 0x00, 0x44, 0x11, 0x00
    };
    rocjpeg_status = rocJpegStreamParse(invalid_dc_huffman_sos_data.data(), invalid_dc_huffman_sos_data.size(), rocjpeg_stream_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_BAD_JPEG) {
        std::cerr << "Expected ROCJPEG_STATUS_BAD_JPEG but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int RocJpegApiNegativeTests::TestInvalidStreamDestroy() {
    std::cout << "info: Executing negative test cases for the rocJpegStreamDestroy API" << std::endl;
    //Scenario 1: Pass nullptr for jpeg_stream_handle
    RocJpegStatus rocjpeg_status = rocJpegStreamDestroy(nullptr);
    if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
        std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int RocJpegApiNegativeTests::TestInvalidCreate() {
    std::cout << "info: Executing negative test cases for the rocJpegCreate API" << std::endl;
    // Scenario 1: Pass nullptr for decoder_handle and decoder_create_info
    RocJpegStatus rocjpeg_status = rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, 0, nullptr);
    if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
        std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 2: Pass valid pointer for handle but invalid negative device_id
    int device_id = -1; // Invalid device ID
    rocjpeg_status = rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, device_id, &rocjpeg_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_EXECUTION_FAILED) {
        std::cerr << "Expected ROCJPEG_STATUS_EXECUTION_FAILED but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }
    rocjpeg_status = rocJpegDestroy(rocjpeg_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        std::cerr << "Expected ROCJPEG_STATUS_SUCCESS but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 3: Pass valid pointer for handle but invalid device_id
    device_id = 255; // Invalid device ID
    rocjpeg_status = rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, device_id, &rocjpeg_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
        std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }
    rocjpeg_status = rocJpegDestroy(rocjpeg_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        std::cerr << "Expected ROCJPEG_STATUS_SUCCESS but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 4: Pass valid pointer for handle but unsupported backend
    device_id = 0;
    rocjpeg_status = rocJpegCreate(ROCJPEG_BACKEND_HYBRID, device_id, &rocjpeg_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_NOT_IMPLEMENTED) {
        std::cerr << "Expected ROCJPEG_STATUS_NOT_IMPLEMENTED but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }
    rocjpeg_status = rocJpegDestroy(rocjpeg_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        std::cerr << "Expected ROCJPEG_STATUS_SUCCESS but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 5: Use an unsupported backend
    RocJpegBackend backend = static_cast<RocJpegBackend>(-1);
    rocjpeg_status = rocJpegCreate(backend, device_id, &rocjpeg_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
        std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }
    rocjpeg_status = rocJpegDestroy(rocjpeg_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        std::cerr << "Expected ROCJPEG_STATUS_SUCCESS but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Create a valid rocJPEG handle - This step ensures a valid rocjpeg_handle_ is available for subsequent negative testing of other rocJPEG APIs.
    rocjpeg_status = rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, device_id, &rocjpeg_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        std::cerr << "Expected ROCJPEG_STATUS_SUCCESS but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int RocJpegApiNegativeTests::TestInvalidDestroy() {
    std::cout << "info: Executing negative test cases for the rocJpegDestroy API" << std::endl;
    //Scenario 1: Pass nullptr for decoder_handle
    RocJpegStatus rocjpeg_status = rocJpegDestroy(nullptr);
    if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
        std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int RocJpegApiNegativeTests::TestInvalidGetImageInfo() {
    std::cout << "info: Executing negative test cases for the rocJpegGetImageInfo API" << std::endl;
    // Scenario 1: Pass nullptr for all parameters
    RocJpegStatus rocjpeg_status = rocJpegGetImageInfo(rocjpeg_handle_, nullptr, nullptr, nullptr, nullptr, nullptr);
    if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
        std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int RocJpegApiNegativeTests::TestInvalidDecode() {
    std::cout << "info: Executing negative test cases for the rocJpegDecode API" << std::endl;
   // Scenario 1: Pass nullptr for all parameters
   RocJpegStatus rocjpeg_status = rocJpegDecode(nullptr, nullptr, nullptr, nullptr);
   if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
       std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
       return EXIT_FAILURE;
   }

   // Scenario 2: Pass valid handle but nullptr for other parameters
   rocjpeg_status = rocJpegDecode(rocjpeg_handle_, nullptr, nullptr, nullptr);
   if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
       std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
       return EXIT_FAILURE;
   }

   // Scenario 3: Pass valid handle and stream but nullptr for decode_params and destination
   rocjpeg_status = rocJpegDecode(rocjpeg_handle_, rocjpeg_stream_handle_, nullptr, nullptr);
   if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
       std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
       return EXIT_FAILURE;
   }

   // Scenario 4: Pass valid handle, stream, and decode_params but nullptr for destination
   RocJpegDecodeParams decode_params = {}; // Assume this is initialized with valid data
   rocjpeg_status = rocJpegDecode(rocjpeg_handle_, rocjpeg_stream_handle_, &decode_params, nullptr);
   if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
       std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
       return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

int RocJpegApiNegativeTests::TestInvalidDecodeBatched() {
    std::cout << "info: Executing negative test cases for the rocJpegDecodeBatched API" << std::endl;
    return EXIT_SUCCESS;
}

int RocJpegApiNegativeTests::TestInvalidGetErrorName() {
    std::cout << "info: Executing negative test cases for the rocJpegGetErrorName API" << std::endl;
    // Scenario 1: Pass an invalid error code
    RocJpegStatus invalid_status = static_cast<RocJpegStatus>(-999); // Invalid error code
    const char *error_name = rocJpegGetErrorName(invalid_status);
    if (error_name == nullptr) {
        std::cerr << "Expected a valid error but got nullptr" << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 2: Pass a valid error code and ensure it returns a non-null name
    for (int i = 0; i >= ROCJPEG_STATUS_MAX_VALUE; i--) {
        RocJpegStatus valid_status = static_cast<RocJpegStatus>(i);;
        error_name = rocJpegGetErrorName(valid_status);
        if (error_name == nullptr) {
            std::cerr << "Expected a valid error but got nullptr" << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Scenario 3: Pass a boundary value (e.g., maximum enum value + 1)
    RocJpegStatus boundary_status = static_cast<RocJpegStatus>(ROCJPEG_STATUS_SUCCESS + 1);
    error_name = rocJpegGetErrorName(boundary_status);
    if (error_name == nullptr) {
        std::cerr << "Expected a valid error but got nullptr" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

RocJpegStatus RocJpegApiNegativeTests::ParseExactBuffer(const std::vector<uint8_t> &data) {
    // std::vector may over-allocate, which would hide a one-byte overread behind
    // its own spare capacity. An exactly sized allocation puts the sanitizer
    // redzone directly after the last byte of the stream.
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[data.empty() ? 1 : data.size()]);
    if (!data.empty()) {
        std::memcpy(buffer.get(), data.data(), data.size());
    }
    return rocJpegStreamParse(buffer.get(), data.size(), rocjpeg_stream_handle_);
}

int RocJpegApiNegativeTests::CheckParseInvariants(const std::vector<uint8_t> &data, const std::string &case_name) {
    RocJpegStatus rocjpeg_status = ParseExactBuffer(data);

    if (!IsKnownStatus(rocjpeg_status)) {
        std::cerr << "[" << case_name << "] rocJpegStreamParse returned an undocumented status ("
                  << static_cast<int>(rocjpeg_status) << ")\n  " << HexDump(data) << std::endl;
        return EXIT_FAILURE;
    }
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        // Rejecting a malformed stream is always an acceptable outcome.
        return EXIT_SUCCESS;
    }

    // A baseline JPEG the parser reports as decodable must carry both a frame
    // header and a scan, so the corresponding marker bytes have to be somewhere
    // in the stream. The converse does not hold - a marker byte pair can occur
    // inside payload data - so this only ever fails on a stream that truly has
    // no such marker, which is what makes it safe to apply to every fuzz case.
    // rocJpegGetImageInfo cannot report an empty scan, so without this check an
    // accepted stream with no entropy-coded data would look perfectly healthy.
    if (FindMarker(data, 0xDA) >= data.size()) {
        std::cerr << "[" << case_name << "] a stream with no SOS marker was accepted as decodable\n  "
                  << HexDump(data) << std::endl;
        return EXIT_FAILURE;
    }
    if (FindMarker(data, 0xC0) >= data.size()) {
        std::cerr << "[" << case_name << "] a stream with no SOF marker was accepted as decodable\n  "
                  << HexDump(data) << std::endl;
        return EXIT_FAILURE;
    }

    if (!check_image_info_) {
        // Without a decoder handle the parsed description cannot be read back,
        // so the status is all there is to check.
        return EXIT_SUCCESS;
    }

    // The stream was accepted, so the parsed description has to be one the
    // decoder could actually act on. This is the check that catches
    // over-acceptance: a stream with no scan, no frame header or nonsense
    // dimensions reaches this point looking successful, and only the extracted
    // image info shows that nothing decodable was found.
    uint8_t num_components = 0;
    RocJpegChromaSubsampling subsampling = ROCJPEG_CSS_UNKNOWN;
    uint32_t widths[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t heights[ROCJPEG_MAX_COMPONENT] = {};
    rocjpeg_status = rocJpegGetImageInfo(rocjpeg_handle_, rocjpeg_stream_handle_, &num_components,
                                         &subsampling, widths, heights);
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        std::cerr << "[" << case_name << "] the stream parsed successfully but rocJpegGetImageInfo failed with "
                  << rocJpegGetErrorName(rocjpeg_status) << "\n  " << HexDump(data) << std::endl;
        return EXIT_FAILURE;
    }
    if (num_components < 1 || num_components > 3) {
        std::cerr << "[" << case_name << "] an accepted stream reports " << static_cast<int>(num_components)
                  << " components; only 1 to 3 are supported\n  " << HexDump(data) << std::endl;
        return EXIT_FAILURE;
    }
    if (widths[0] == 0 || heights[0] == 0) {
        std::cerr << "[" << case_name << "] an accepted stream reports zero-sized dimensions ("
                  << widths[0] << "x" << heights[0] << ")\n  " << HexDump(data) << std::endl;
        return EXIT_FAILURE;
    }
    if (subsampling == ROCJPEG_CSS_UNKNOWN || subsampling == ROCJPEG_CSS_411) {
        std::cerr << "[" << case_name << "] an accepted stream reports an unsupported chroma subsampling ("
                  << static_cast<int>(subsampling) << ")\n  " << HexDump(data) << std::endl;
        return EXIT_FAILURE;
    }
    // 4:0:0 means there is no chroma at all, so it describes a single-component
    // image and nothing else. Checking only one direction of that equivalence
    // leaves the more interesting failure uncovered: a multi-component frame
    // header whose chroma sampling factors are zero is classified as 4:0:0, and
    // a one-way check would accept it because the component count looks fine on
    // its own and the subsampling looks fine on its own.
    if ((num_components == 1) != (subsampling == ROCJPEG_CSS_400)) {
        std::cerr << "[" << case_name << "] an accepted stream reports " << static_cast<int>(num_components)
                  << " components with subsampling " << static_cast<int>(subsampling)
                  << "; 4:0:0 (" << static_cast<int>(ROCJPEG_CSS_400)
                  << ") and a single component have to imply each other\n  " << HexDump(data) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int RocJpegApiNegativeTests::TestStreamParseFuzz() {
    std::cout << "info: Executing fuzz test cases for the rocJpegStreamParse API" << std::endl;

    const uint32_t seed = EnvOrDefault("ROCJPEG_FUZZ_SEED", 0x5EEDBEEF);
    const uint32_t iterations = EnvOrDefault("ROCJPEG_FUZZ_ITERATIONS", 20000);
    std::cout << "info: fuzz seed = 0x" << std::hex << seed << std::dec
              << ", iterations = " << iterations
              << " (override with ROCJPEG_FUZZ_SEED / ROCJPEG_FUZZ_ITERATIONS)" << std::endl;

    const std::vector<uint8_t> &color_seed = ColorSeedJpeg();
    const std::vector<uint8_t> &gray_seed = GraySeedJpeg();

    // The seeds have to be accepted, otherwise every mutation derived from them
    // would be rejected for the wrong reason and the fuzzing would prove nothing.
    if (ParseExactBuffer(color_seed) != ROCJPEG_STATUS_SUCCESS ||
        ParseExactBuffer(gray_seed) != ROCJPEG_STATUS_SUCCESS) {
        std::cerr << "The embedded fuzzing seeds are no longer accepted by the parser; "
                     "the fuzz corpus needs to be regenerated." << std::endl;
        return EXIT_FAILURE;
    }

    // Reading back a parsed image description needs a decoder handle, which the
    // preceding tests have already created. Probe it on a seed that is known to
    // parse: if the probe works, every accepted stream from here on is held to
    // the full set of postconditions.
    {
        uint8_t num_components = 0;
        RocJpegChromaSubsampling subsampling = ROCJPEG_CSS_UNKNOWN;
        uint32_t widths[ROCJPEG_MAX_COMPONENT] = {};
        uint32_t heights[ROCJPEG_MAX_COMPONENT] = {};
        check_image_info_ = rocJpegGetImageInfo(rocjpeg_handle_, rocjpeg_stream_handle_, &num_components,
                                                &subsampling, widths, heights) == ROCJPEG_STATUS_SUCCESS;
        if (!check_image_info_) {
            std::cout << "info: no usable decoder handle, checking reported statuses only "
                         "(parsed image descriptions will not be verified)" << std::endl;
        }
    }

    if (CheckParseInvariants(color_seed, "seed/color") || CheckParseInvariants(gray_seed, "seed/gray")) {
        return EXIT_FAILURE;
    }

    // Part 1: hand-written streams that reproduce previously fixed defects. Each
    // one is expected to be rejected; an acceptance here is a regression.
    struct RegressionCase {
        const char *name;
        std::vector<uint8_t> data;
    };
    const std::vector<RegressionCase> regressions = {
        // The stream ends immediately after a marker code, so the two-byte
        // segment length field is not present at all.
        {"marker at end of stream", {0xFF, 0xD8, 0xFF, 0xC0}},
        // Only one of the two segment length bytes is present.
        {"one length byte remaining", {0xFF, 0xD8, 0xFF, 0xC0, 0x00}},
        // A segment length below 2 cannot even cover the length field itself and
        // used to leave the marker loop without making progress.
        {"zero segment length", {0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x00, 0x08, 0x00}},
        {"segment length of one", {0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x01, 0x08, 0x00}},
        // A segment length that reaches past the buffer.
        {"segment length past end", {0xFF, 0xD8, 0xFF, 0xC0, 0xFF, 0xFF, 0x08, 0x00}},
        // A SOF segment whose declared length is too small for the fixed
        // 8-byte frame header the parser reads from it.
        {"SOF length 4 at end of stream", {0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x04}},
        // A SOF segment large enough for the header but not for the component
        // descriptors it declares.
        {"SOF too small for its components",
            {0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x08, 0x08, 0x00, 0x10, 0x00, 0x10, 0x03}},
        // A scan that covers no component at all.
        {"SOS with zero components",
            {0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x06, 0x00, 0x00, 0x3F, 0x00}},
        // A SOS segment that declares more components than it carries.
        {"SOS too small for its components",
            {0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x06, 0x03, 0x00, 0x3F, 0x00}},
        // An unbroken run of 0xFF fill bytes with no marker code after it.
        {"unterminated fill byte run", {0xFF, 0xD8, 0xFF, 0xFF, 0xFF, 0xFF}},
        // A marker code that is not preceded by the mandatory 0xFF prefix, so
        // the byte is payload data being read as if it started a segment.
        {"marker without its 0xFF prefix",
            {0xFF, 0xD8, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, 0x01, 0x11, 0x00}},
        // The 0xFF00 byte-stuffing sequence has no meaning outside of
        // entropy-coded data and must not be read as a marker.
        {"byte stuffing in the header sequence", {0xFF, 0xD8, 0xFF, 0x00, 0x00, 0x0B}},
        // A DRI segment that declares four bytes but carries two.
        {"truncated DRI segment", {0xFF, 0xD8, 0xFF, 0xDD, 0x00, 0x04}},
        // Streams too short to hold even the SOI and EOI markers.
        {"two byte stream", {0xFF, 0xD8}},
        {"single byte stream", {0xFF}},
        {"empty stream", {}},
    };
    for (const RegressionCase &regression : regressions) {
        RocJpegStatus rocjpeg_status = ParseExactBuffer(regression.data);
        if (rocjpeg_status == ROCJPEG_STATUS_SUCCESS) {
            std::cerr << "[" << regression.name << "] Expected the malformed stream to be rejected but it was accepted\n  "
                      << HexDump(regression.data) << std::endl;
            return EXIT_FAILURE;
        }
        if (!IsKnownStatus(rocjpeg_status)) {
            std::cerr << "[" << regression.name << "] rocJpegStreamParse returned an undocumented status ("
                      << static_cast<int>(rocjpeg_status) << ")\n  " << HexDump(regression.data) << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Part 2: streams that carry tables and an end-of-image marker but never
    // start a scan. They have to be rejected rather than reported as parsed with
    // no entropy-coded data behind them.
    {
        const size_t sos_offset = FindMarker(color_seed, 0xDA);
        if (sos_offset >= color_seed.size()) {
            std::cerr << "The embedded color seed no longer contains an SOS marker." << std::endl;
            return EXIT_FAILURE;
        }
        std::vector<uint8_t> headers_only(color_seed.begin(), color_seed.begin() + sos_offset);

        std::vector<uint8_t> eoi_before_sos = headers_only;
        eoi_before_sos.push_back(0xFF);
        eoi_before_sos.push_back(0xD9);

        const std::vector<RegressionCase> scanless = {
            {"headers followed by EOI, no scan", eoi_before_sos},
            {"headers with no scan and no EOI", headers_only},
        };
        for (const RegressionCase &scanless_case : scanless) {
            if (ParseExactBuffer(scanless_case.data) == ROCJPEG_STATUS_SUCCESS) {
                std::cerr << "[" << scanless_case.name << "] Expected a stream without a scan to be rejected but it was accepted\n  "
                          << HexDump(scanless_case.data) << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    // Part 3: a complete stream whose trailing EOI marker was cut off. The
    // entropy-coded data simply runs to the end of the buffer, so this one is
    // expected to be accepted - it exercises the end-of-image search on a stream
    // that never matches.
    {
        std::vector<uint8_t> no_eoi(color_seed.begin(), color_seed.end() - 2);
        if (ParseExactBuffer(no_eoi) != ROCJPEG_STATUS_SUCCESS) {
            std::cerr << "[scan without a trailing EOI] Expected the stream to be accepted but it was rejected\n  "
                      << HexDump(no_eoi) << std::endl;
            return EXIT_FAILURE;
        }
        if (CheckParseInvariants(no_eoi, "scan without a trailing EOI")) {
            return EXIT_FAILURE;
        }
    }

    // Part 4: every prefix of both seeds. This walks the truncation boundary
    // through each marker segment in turn, which is where fixed-offset reads
    // past the end of the buffer show up.
    for (const std::vector<uint8_t> *stream_seed : {&color_seed, &gray_seed}) {
        for (size_t length = 0; length <= stream_seed->size(); length++) {
            std::vector<uint8_t> truncated(stream_seed->begin(), stream_seed->begin() + length);
            if (CheckParseInvariants(truncated, "truncation at " + std::to_string(length) + " bytes")) {
                return EXIT_FAILURE;
            }
        }
    }

    // Part 5: randomized mutation of the seeds. The mutations deliberately
    // favour the fields the parser trusts - segment lengths, component counts
    // and marker codes - rather than spreading uniformly over the payload bytes.
    FuzzRandom random(seed);
    uint32_t accepted = 0;
    for (uint32_t iteration = 0; iteration < iterations; iteration++) {
        const std::vector<uint8_t> &base = (random.Next() & 1u) ? color_seed : gray_seed;
        std::vector<uint8_t> mutated = base;

        switch (random.Below(6)) {
            case 0: {
                // Flip a handful of individual bytes anywhere in the stream.
                const uint32_t flips = 1 + random.Below(4);
                for (uint32_t i = 0; i < flips; i++) {
                    mutated[random.Below(static_cast<uint32_t>(mutated.size()))] =
                        static_cast<uint8_t>(random.Below(256));
                }
                break;
            }
            case 1: {
                // Truncate at a random point.
                mutated.resize(random.Below(static_cast<uint32_t>(mutated.size()) + 1));
                break;
            }
            case 2: {
                // Rewrite the length field of a random marker segment to an
                // extreme value, which is what drives the parser off the end.
                static const uint16_t kLengths[] = {0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
                                                    0x0007, 0x00FF, 0x7FFF, 0xFFFE, 0xFFFF};
                std::vector<size_t> segment_offsets;
                for (size_t i = 0; i + 3 < mutated.size(); i++) {
                    // Length-bearing markers are 0xFF followed by a code that is
                    // neither a standalone marker nor a stuffed zero byte.
                    const uint8_t code = mutated[i + 1];
                    if (mutated[i] == 0xFF && code != 0x00 && code != 0xFF &&
                        code != 0x01 && !(code >= 0xD0 && code <= 0xD9)) {
                        segment_offsets.push_back(i);
                    }
                }
                if (!segment_offsets.empty()) {
                    const size_t offset = segment_offsets[random.Below(static_cast<uint32_t>(segment_offsets.size()))];
                    const uint16_t length = kLengths[random.Below(sizeof(kLengths) / sizeof(kLengths[0]))];
                    mutated[offset + 2] = static_cast<uint8_t>(length >> 8);
                    mutated[offset + 3] = static_cast<uint8_t>(length & 0xFF);
                }
                break;
            }
            case 3: {
                // Replace a marker code with another one, so segments are parsed
                // by a handler that expects a different layout.
                static const uint8_t kMarkers[] = {0xC0, 0xC2, 0xC4, 0xD8, 0xD9, 0xDA, 0xDB, 0xDD,
                                                   0xD0, 0xD7, 0x01, 0xE0, 0xFE, 0x00};
                std::vector<size_t> marker_offsets;
                for (size_t i = 0; i + 1 < mutated.size(); i++) {
                    if (mutated[i] == 0xFF) {
                        marker_offsets.push_back(i + 1);
                    }
                }
                if (!marker_offsets.empty()) {
                    mutated[marker_offsets[random.Below(static_cast<uint32_t>(marker_offsets.size()))]] =
                        kMarkers[random.Below(sizeof(kMarkers) / sizeof(kMarkers[0]))];
                }
                break;
            }
            case 4: {
                // Splice a run of 0xFF fill bytes in, optionally leaving the
                // stream ending inside that run.
                const size_t offset = random.Below(static_cast<uint32_t>(mutated.size()) + 1);
                const uint32_t run = 1 + random.Below(8);
                mutated.insert(mutated.begin() + offset, run, 0xFF);
                if (random.Below(4) == 0) {
                    mutated.resize(offset + run);
                }
                break;
            }
            default: {
                // Erase a random span, shifting every later field out of place.
                if (mutated.size() > 4) {
                    const size_t offset = random.Below(static_cast<uint32_t>(mutated.size()) - 1);
                    const size_t count = 1 + random.Below(static_cast<uint32_t>(mutated.size() - offset));
                    mutated.erase(mutated.begin() + offset, mutated.begin() + offset + count);
                }
                break;
            }
        }

        if (CheckParseInvariants(mutated, "mutation " + std::to_string(iteration))) {
            std::cerr << "info: replay with ROCJPEG_FUZZ_SEED=0x" << std::hex << seed << std::dec
                      << " ROCJPEG_FUZZ_ITERATIONS=" << iterations << std::endl;
            return EXIT_FAILURE;
        }
        if (ParseExactBuffer(mutated) == ROCJPEG_STATUS_SUCCESS) {
            accepted++;
        }
    }

    // Part 6: buffers of pure noise, half of them carrying a valid SOI so the
    // parser gets past the initial signature check.
    for (uint32_t iteration = 0; iteration < iterations; iteration++) {
        std::vector<uint8_t> noise(random.Below(64));
        for (uint8_t &byte : noise) {
            byte = static_cast<uint8_t>(random.Below(256));
        }
        if (noise.size() >= 2 && (random.Next() & 1u)) {
            noise[0] = 0xFF;
            noise[1] = 0xD8;
        }
        if (CheckParseInvariants(noise, "random buffer " + std::to_string(iteration))) {
            std::cerr << "info: replay with ROCJPEG_FUZZ_SEED=0x" << std::hex << seed << std::dec
                      << " ROCJPEG_FUZZ_ITERATIONS=" << iterations << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "info: fuzzing completed, " << accepted << " of " << iterations
              << " mutated streams were accepted and satisfied every postcondition" << std::endl;

    return EXIT_SUCCESS;
}

int RocJpegApiNegativeTests::RunTests() {
    if (TestInvalidStreamCreate() || TestInvalidStreamParse () || TestInvalidStreamDestroy() || TestInvalidCreate() || TestInvalidDestroy() ||
        TestInvalidGetImageInfo() || TestInvalidDecode() || TestInvalidDecodeBatched() || TestInvalidGetErrorName() ||
        TestStreamParseFuzz()) {
        std::cerr << "One or more negative tests failed." << std::endl;
        return EXIT_FAILURE;
    } else {
        return EXIT_SUCCESS;
    }
}
