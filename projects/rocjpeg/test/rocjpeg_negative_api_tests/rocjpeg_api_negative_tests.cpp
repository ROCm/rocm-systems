
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

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>

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
 * @brief Renders one byte as two uppercase hex digits, for naming a case after
 *        the marker code it is built around.
 */
std::string ToHex(uint8_t value) {
    std::ostringstream oss;
    oss << std::uppercase << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(value);
    return oss.str();
}

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
 * @brief Finds a marker by walking the segment structure of a stream.
 *
 * Unlike FindMarker this does not match byte pairs inside segment payloads. It
 * starts at SOI and steps from one marker to the next, skipping each
 * length-bearing payload by its declared size, so a 0xFF marker_code planted in
 * an APPn or DQT payload is never mistaken for a real marker. The walk gives up
 * as soon as the structure stops making sense, which keeps callers from drawing
 * conclusions from a stream whose layout could not be followed.
 *
 * @param data The stream to search.
 * @param marker The marker code that follows the 0xFF prefix.
 * @return The offset of the 0xFF byte, or data.size() when the marker was not
 *         reached, including when the walk could not be completed.
 */
size_t FindStructuralMarker(const std::vector<uint8_t> &data, uint8_t marker) {
    if (data.size() < 2 || data[0] != 0xFF || data[1] != 0xD8) {
        return data.size();
    }
    size_t offset = 2;
    while (offset + 1 < data.size()) {
        if (data[offset] != 0xFF) {
            return data.size();
        }
        // ISO/IEC 10918-1 B.1.1.2: any number of 0xFF fill bytes may precede a
        // marker, so the code is the first byte after the run of 0xFF.
        while (offset < data.size() && data[offset] == 0xFF) {
            offset++;
        }
        if (offset >= data.size()) {
            return data.size();
        }
        const uint8_t marker_code = data[offset++];
        if (marker_code == marker) {
            return offset - 2;
        }
        if (marker_code == 0x00 || marker_code == 0x01 || (marker_code >= 0xD0 && marker_code <= 0xD9)) {
            // TEM, the restart markers and SOI/EOI carry no payload. A 0x00 here
            // would be byte stuffing outside entropy-coded data, which is not a
            // structure this walk can follow.
            if (marker_code == 0x00 || marker_code == 0xD9) {
                return data.size();
            }
            continue;
        }
        if (offset + 1 >= data.size()) {
            return data.size();
        }
        const size_t segment_length = (static_cast<size_t>(data[offset]) << 8) | data[offset + 1];
        if (segment_length < 2 || segment_length > data.size() - offset) {
            return data.size();
        }
        offset += segment_length;
    }
    return data.size();
}

/**
 * @brief Reports whether a scan body carries at least one byte of coded data.
 *
 * Not every byte between the scan header and the EOI is entropy-coded data, so
 * a byte count cannot answer this. ISO/IEC 10918-1 B.1.1.2 and B.1.1.3 define
 * what a 0xFF introduces inside a scan: a run of fill bytes before a marker, a
 * stuffed 0x00 standing for one real 0xFF of coded data, or a restart marker
 * that carries no bits of its own. Everything else is coded data. Walking those
 * tokens is what separates a scan of "FF D0 FF D9", which holds nothing but a
 * restart, from one that actually encodes a block.
 *
 * @param data The stream the scan belongs to.
 * @param scan_start The offset of the first byte after the scan header.
 * @param scan_end The offset one past the last byte of the scan.
 * @return true when the scan contains at least one byte of coded data.
 */
bool ScanHasEntropyData(const std::vector<uint8_t> &data, size_t scan_start, size_t scan_end) {
    size_t offset = scan_start;
    while (offset < scan_end) {
        if (data[offset] != 0xFF) {
            return true;
        }
        // Any number of 0xFF fill bytes may precede a marker, so the code is
        // the first byte after the run.
        while (offset < scan_end && data[offset] == 0xFF) {
            offset++;
        }
        if (offset >= scan_end) {
            // A trailing 0xFF run is fill or a truncated marker, never data.
            return false;
        }
        const uint8_t marker_code = data[offset];
        if (marker_code == 0x00) {
            // Byte stuffing: this pair stands for one 0xFF of coded data.
            return true;
        }
        if (marker_code < 0xD0 || marker_code > 0xD7) {
            // Any other marker terminates the scan, so nothing beyond it counts.
            return false;
        }
        offset++;
    }
    return false;
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
    // would come back as 1. Reject the sign before parsing instead. That means
    // skipping exactly the leading whitespace strtoull skips: looking at a
    // narrower set would leave the sign behind a character the loop stops on but
    // strtoull steps over, and "\n-18446744073709551615" would parse as 1 again.
    const char *digits = value;
    while (std::isspace(static_cast<unsigned char>(*digits))) {
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
    // Either handle can still be null here, because a test that fails before
    // creating one returns straight away. Destroying a null handle is the
    // documented invalid-parameter case rather than an error worth reporting,
    // so skip it instead of printing a failure the run did not actually have.
    if (rocjpeg_handle_ != nullptr) {
        RocJpegStatus rocjpeg_status = rocJpegDestroy(rocjpeg_handle_);
        rocjpeg_handle_ = nullptr;
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            std::cerr << "Failed to destroy rocjpeg handle: " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        }
    }
    if (rocjpeg_stream_handle_ != nullptr) {
        RocJpegStatus rocjpeg_status = rocJpegStreamDestroy(rocjpeg_stream_handle_);
        rocjpeg_stream_handle_ = nullptr;
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            std::cerr << "Failed to destroy rocjpeg stream handle " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        }
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

int RocJpegApiNegativeTests::DestroyHandle() {
    RocJpegStatus rocjpeg_status = rocJpegDestroy(rocjpeg_handle_);
    // Cleared whatever the status was: a failed destroy is reported below, but
    // the handle must not be left behind for the destructor either way.
    rocjpeg_handle_ = nullptr;
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        std::cerr << "Expected ROCJPEG_STATUS_SUCCESS but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
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
    if (DestroyHandle()) {
        return EXIT_FAILURE;
    }

    // Scenario 3: Pass valid pointer for handle but invalid device_id
    device_id = 255; // Invalid device ID
    rocjpeg_status = rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, device_id, &rocjpeg_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
        std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }
    if (DestroyHandle()) {
        return EXIT_FAILURE;
    }

    // Scenario 4: Pass valid pointer for handle but unsupported backend
    device_id = 0;
    rocjpeg_status = rocJpegCreate(ROCJPEG_BACKEND_HYBRID, device_id, &rocjpeg_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_NOT_IMPLEMENTED) {
        std::cerr << "Expected ROCJPEG_STATUS_NOT_IMPLEMENTED but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }
    if (DestroyHandle()) {
        return EXIT_FAILURE;
    }

    // Scenario 5: Use an unsupported backend
    RocJpegBackend backend = static_cast<RocJpegBackend>(-1);
    rocjpeg_status = rocJpegCreate(backend, device_id, &rocjpeg_handle_);
    if (rocjpeg_status != ROCJPEG_STATUS_INVALID_PARAMETER) {
        std::cerr << "Expected ROCJPEG_STATUS_INVALID_PARAMETER but got " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
        return EXIT_FAILURE;
    }
    if (DestroyHandle()) {
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
    //
    // Including when the stream is empty, which both the prefix and the noise
    // loop below produce. Rounding that up to one byte would leave a readable
    // byte at data[0] and let a parser that dereferences before checking the
    // length go unnoticed; a zero-length new[] still returns a distinct
    // non-null pointer, so the redzone starts at the pointer itself.
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[data.size()]);
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
    const size_t sos_offset = FindMarker(data, 0xDA);
    if (sos_offset >= data.size()) {
        std::cerr << "[" << case_name << "] a stream with no SOS marker was accepted as decodable\n  "
                  << HexDump(data) << std::endl;
        return EXIT_FAILURE;
    }

    // The presence of an SOS marker says nothing about the scan behind it. A
    // stream that stops right after the scan header, or puts EOI directly after
    // it, carries no entropy-coded data and so describes nothing to decode,
    // while still satisfying every marker and image-description check. Inspect
    // the scan instead: the segment length follows the marker, and the scan
    // starts after it and runs to EOI or to the end of the buffer.
    //
    // This needs the structural SOS, not the first matching byte pair. The
    // mutation strategies rewrite the byte after any 0xFF, so an accepted stream
    // can carry an "FF DA" inside an APPn or DQT payload ahead of its real scan
    // header; reading the two bytes behind that as a segment length would put
    // scan_start anywhere and condemn a stream the parser handled correctly. The
    // walk returns data.size() when the layout could not be followed, and then
    // there is no offset to measure from and the check is skipped.
    const size_t structural_sos_offset = FindStructuralMarker(data, 0xDA);
    if (structural_sos_offset + 4 <= data.size()) {
        const size_t segment_length =
            (static_cast<size_t>(data[structural_sos_offset + 2]) << 8) | data[structural_sos_offset + 3];
        const size_t scan_start = structural_sos_offset + 2 + segment_length;
        if (scan_start <= data.size()) {
            size_t scan_end = scan_start;
            while (scan_end + 1 < data.size() && !(data[scan_end] == 0xFF && data[scan_end + 1] == 0xD9)) {
                scan_end++;
            }
            if (scan_end + 1 >= data.size()) {
                scan_end = data.size();
            }
            if (!ScanHasEntropyData(data, scan_start, scan_end)) {
                std::cerr << "[" << case_name << "] a stream whose scan carries no entropy-coded data was "
                             "accepted as decodable\n  " << HexDump(data) << std::endl;
                return EXIT_FAILURE;
            }
        }
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

    // ScanHasEntropyData decides every empty-scan verdict below, so exercise its
    // token walk directly rather than only through whole streams. Driving these
    // cases through the parser would need a stream for each, and the short ones
    // could only be built by truncating a scan - which is exactly the shape that
    // should not appear among the streams required to parse successfully.
    {
        struct ScanTokenCase {
            const char *name;
            std::vector<uint8_t> body;
            bool has_data;
        };
        const std::vector<ScanTokenCase> scan_token_cases = {
            {"empty body", {}, false},
            {"a single coded byte", {0x42}, true},
            {"a zero byte", {0x00}, true},
            {"one fill byte", {0xFF}, false},
            {"an all-0xFF run", {0xFF, 0xFF, 0xFF}, false},
            {"fill before EOI", {0xFF, 0xFF, 0xD9}, false},
            {"a stuffed 0xFF", {0xFF, 0x00}, true},
            {"a stuffed 0xFF behind fill bytes", {0xFF, 0xFF, 0x00}, true},
            {"one restart marker", {0xFF, 0xD0}, false},
            {"the last restart marker", {0xFF, 0xD7}, false},
            {"every restart marker",
             {0xFF, 0xD0, 0xFF, 0xD1, 0xFF, 0xD2, 0xFF, 0xD3, 0xFF, 0xD4, 0xFF, 0xD5, 0xFF, 0xD6, 0xFF, 0xD7},
             false},
            {"restart markers behind fill bytes", {0xFF, 0xFF, 0xD0, 0xFF, 0xFF, 0xFF, 0xD1}, false},
            {"a coded byte after a restart marker", {0xFF, 0xD0, 0x42}, true},
            {"a stuffed 0xFF after a restart marker", {0xFF, 0xD0, 0xFF, 0x00}, true},
            {"EOI before the coded data", {0xFF, 0xD9, 0x42}, false},
            {"a scan header before the coded data", {0xFF, 0xDA, 0x42}, false},
            {"a truncated marker after a restart", {0xFF, 0xD0, 0xFF}, false},
        };
        for (const ScanTokenCase &scan_token_case : scan_token_cases) {
            if (ScanHasEntropyData(scan_token_case.body, 0, scan_token_case.body.size()) !=
                scan_token_case.has_data) {
                std::cerr << "[scan tokens/" << scan_token_case.name << "] expected a scan body of "
                          << (scan_token_case.has_data ? "coded data" : "markers only")
                          << " to be read as such\n  " << HexDump(scan_token_case.body) << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    // Part 1: hand-written streams that reproduce previously fixed defects. Each
    // one is expected to be rejected; an acceptance here is a regression.
    struct RegressionCase {
        // Owned rather than a pointer: some of the cases below name themselves
        // after the marker code they are built around.
        std::string name;
        std::vector<uint8_t> data;
    };
    // Streams that have to stay accepted. Every check that rejects something has
    // a matching case here, so that a check which over-rejects is caught by the
    // same run rather than by a later corpus regression.
    std::vector<RegressionCase> accepted_cases;
    std::vector<RegressionCase> regressions = {
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
    // A scan header with nothing behind it. Both spellings are built from a seed
    // that parses, so the only thing wrong with them is the empty scan: the
    // stream either stops at the end of the SOS segment or puts EOI right after
    // it. Every marker is present and the frame header is intact, which is what
    // makes these reachable by truncation and invisible to a marker-only check.
    {
        const size_t sos_offset = FindMarker(color_seed, 0xDA);
        if (sos_offset + 4 > color_seed.size()) {
            std::cerr << "The color fuzzing seed no longer carries a parsable SOS segment." << std::endl;
            return EXIT_FAILURE;
        }
        const size_t segment_length =
            (static_cast<size_t>(color_seed[sos_offset + 2]) << 8) | color_seed[sos_offset + 3];
        const size_t scan_start = sos_offset + 2 + segment_length;
        if (scan_start > color_seed.size()) {
            std::cerr << "The color fuzzing seed no longer carries a parsable SOS segment." << std::endl;
            return EXIT_FAILURE;
        }
        std::vector<uint8_t> empty_scan(color_seed.begin(), color_seed.begin() + scan_start);
        std::vector<uint8_t> empty_scan_eoi = empty_scan;
        empty_scan_eoi.push_back(0xFF);
        empty_scan_eoi.push_back(0xD9);
        regressions.push_back({"scan header with no entropy-coded data", empty_scan});
        regressions.push_back({"scan header immediately followed by EOI", empty_scan_eoi});

        // The same empty scan padded with bodies made of nothing but fill bytes
        // and restart markers, all of which are markers rather than coded bits.
        // These are still empty scans, but a naive byte count sees one or more
        // bytes between the scan header and the EOI and accepts them.
        const std::vector<std::pair<const char *, std::vector<uint8_t>>> empty_bodies = {
            {"empty scan with a fill byte before EOI", {0xFF, 0xFF, 0xD9}},
            {"empty scan with several fill bytes before EOI", {0xFF, 0xFF, 0xFF, 0xD9}},
            {"empty scan ending in a truncated marker", {0xFF}},
            {"empty scan with an all-0xFF tail", {0xFF, 0xFF, 0xFF}},
            {"scan containing only a restart marker", {0xFF, 0xD0, 0xFF, 0xD9}},
            {"scan containing only a truncated restart marker", {0xFF, 0xD0}},
            {"scan containing only two restart markers", {0xFF, 0xD0, 0xFF, 0xD1, 0xFF, 0xD9}},
            {"scan containing only the eight restart markers",
             {0xFF, 0xD0, 0xFF, 0xD1, 0xFF, 0xD2, 0xFF, 0xD3, 0xFF, 0xD4, 0xFF, 0xD5, 0xFF, 0xD6, 0xFF, 0xD7,
              0xFF, 0xD9}},
            {"scan of restart markers padded with fill bytes",
             {0xFF, 0xFF, 0xD0, 0xFF, 0xFF, 0xFF, 0xD1, 0xFF, 0xFF, 0xD9}},
        };
        for (const auto &empty_body : empty_bodies) {
            std::vector<uint8_t> padded = empty_scan;
            padded.insert(padded.end(), empty_body.second.begin(), empty_body.second.end());
            regressions.push_back({empty_body.first, padded});
        }

        // The mirror image of the cases above: the smallest scan that does carry
        // data, so the token walk is not simply rejecting every short scan.
        //
        // This has to be a complete MCU rather than just a byte or two, or the
        // fixture would contradict the very postcondition it is here to protect.
        // The seed is 16x16 4:2:0, which is one interleaved MCU of six blocks,
        // and with its embedded Huffman tables an all-zero MCU still costs 18
        // bits: four luma blocks at 1 bit of DC plus 1 bit of EOB, and two
        // chroma blocks whose only DC symbol is category 3, so 1 bit of code
        // plus 3 additional bits plus 1 bit of EOB. That is the body below,
        // padded to a byte boundary with 1 bits as F.1.2.3 requires. A stream
        // carrying fewer bits is truncated, and a parser that grew to validate
        // entropy completeness would be right to reject it.
        // Both spellings below were checked against an independent decoder,
        // which renders them as a 16x16 image rather than reporting a truncated
        // file.
        const std::vector<uint8_t> complete_mcu = {0x00, 0x00, 0x3F};
        std::vector<uint8_t> minimal_scan = empty_scan;
        minimal_scan.insert(minimal_scan.end(), complete_mcu.begin(), complete_mcu.end());
        minimal_scan.push_back(0xFF);
        minimal_scan.push_back(0xD9);

        // The same MCU with a restart marker, so that the tokens a scan may
        // legitimately contain are covered rather than just plain coded bytes.
        //
        // A restart marker is only meaningful at the boundary between restart
        // intervals, and those intervals only exist once a DRI segment declares
        // one, so the marker cannot simply be prepended to the single-MCU scan
        // above: that stream has no DRI and only one MCU, which makes the marker
        // corrupt data rather than a token, and requiring it to parse would lock
        // in acceptance of malformed syntax. The fixture is built instead by
        // widening the seed's frame to 32x16 so it covers two 16x16 4:2:0 MCUs,
        // inserting a DRI of one MCU ahead of the scan header, and putting RST0
        // between the two intervals. An independent decoder reads the result as
        // a 32x16 image with no warnings, and reports corruption as soon as the
        // DRI segment is taken back out.
        const size_t restart_sof_offset = FindMarker(color_seed, 0xC0);
        if (restart_sof_offset + 9 > color_seed.size()) {
            std::cerr << "The color fuzzing seed no longer carries a parsable SOF segment." << std::endl;
            return EXIT_FAILURE;
        }
        std::vector<uint8_t> restart_between_mcus(color_seed.begin(), color_seed.begin() + scan_start);
        restart_between_mcus[restart_sof_offset + 7] = 0x00;
        restart_between_mcus[restart_sof_offset + 8] = 0x20;
        const std::vector<uint8_t> dri_segment = {0xFF, 0xDD, 0x00, 0x04, 0x00, 0x01};
        restart_between_mcus.insert(restart_between_mcus.begin() + sos_offset, dri_segment.begin(),
                                    dri_segment.end());
        restart_between_mcus.insert(restart_between_mcus.end(), complete_mcu.begin(), complete_mcu.end());
        restart_between_mcus.push_back(0xFF);
        restart_between_mcus.push_back(0xD0);
        restart_between_mcus.insert(restart_between_mcus.end(), complete_mcu.begin(), complete_mcu.end());
        restart_between_mcus.push_back(0xFF);
        restart_between_mcus.push_back(0xD9);

        accepted_cases.push_back({"scan of one complete MCU", minimal_scan});
        accepted_cases.push_back({"scan of two complete MCUs with a restart marker between them",
                                  restart_between_mcus});
    }

    // A three-component frame header whose chroma sampling factors are zero.
    // Sampling factors of zero are outside the range the format allows, and
    // leaving them unchecked made the frame classify as 4:0:0, so a colour
    // image was described as if it were grayscale.
    {
        const size_t sof_offset = FindMarker(color_seed, 0xC0);
        if (sof_offset + 18 > color_seed.size()) {
            std::cerr << "The color fuzzing seed no longer carries a parsable SOF segment." << std::endl;
            return EXIT_FAILURE;
        }
        std::vector<uint8_t> zero_chroma = color_seed;
        // The component descriptors follow the 8-byte segment header as
        // (identifier, sampling factors, quantisation table) triplets.
        zero_chroma[sof_offset + 11] = 0x11;
        zero_chroma[sof_offset + 14] = 0x00;
        zero_chroma[sof_offset + 17] = 0x00;
        regressions.push_back({"SOF with zero chroma sampling factors", zero_chroma});

        // Sampling factors that are each inside the 1 to 4 range the format
        // allows, but whose products add up past the 10 blocks an interleaved
        // MCU may hold. A per-component range check passes all of these, and
        // the equal factors classify as 4:4:4, so the frame parameters reach
        // the decoder describing an MCU it cannot hold. The pairs bracket the
        // boundary from both sides.
        const std::vector<std::pair<const char *, std::array<uint8_t, 3>>> mcu_cases = {
            {"SOF whose MCU holds 48 blocks", {0x44, 0x44, 0x44}},
            {"SOF whose MCU holds 12 blocks", {0x22, 0x22, 0x22}},
            {"SOF whose MCU holds 11 blocks", {0x33, 0x11, 0x11}},
        };
        for (const auto &mcu_case : mcu_cases) {
            std::vector<uint8_t> oversized_mcu = color_seed;
            oversized_mcu[sof_offset + 11] = mcu_case.second[0];
            oversized_mcu[sof_offset + 14] = mcu_case.second[1];
            oversized_mcu[sof_offset + 17] = mcu_case.second[2];
            regressions.push_back({mcu_case.first, oversized_mcu});
        }

        // The other side of the boundary. The limit is ten blocks, but the
        // largest MCU that actually reaches the decoder is smaller than that,
        // because a frame also has to classify as a subsampling
        // GetChromaSubsampling recognises. Across every three-component layout
        // it accepts, the largest sum that is not already over the limit is
        // eight, from the two 4:2:2 spellings; 4:4:4 at 2x2 or 4x4 would sum to
        // 12 and 48 and is rejected by the cases above.
        //
        // The eight-block layouts are the ones that matter here: a limit
        // lowered to six, or applied to the largest product rather than the
        // sum, would still pass a 4:2:0 fixture while rejecting supported
        // 4:2:2 streams. Keep the 4:2:0 fixture here; the other layouts require
        // separately encoded entropy payloads and are covered by the negative
        // cases above rather than being asserted as accepted.
        const std::vector<std::pair<const char *, std::array<uint8_t, 3>>> supported_mcu_cases = {
            {"SOF whose MCU holds six blocks as 4:2:0", {0x22, 0x11, 0x11}},
        };
        for (const auto &supported_mcu_case : supported_mcu_cases) {
            std::vector<uint8_t> largest_mcu = color_seed;
            largest_mcu[sof_offset + 11] = supported_mcu_case.second[0];
            largest_mcu[sof_offset + 14] = supported_mcu_case.second[1];
            largest_mcu[sof_offset + 17] = supported_mcu_case.second[2];

            accepted_cases.push_back({supported_mcu_case.first, largest_mcu});
        }
    }

    // A single-component frame is not interleaved, so A.2.2 does not apply to it
    // and its MCU is one data unit whatever the sampling factors say. 4x4 is one
    // of the two layouts GetChromaSubsampling maps to 4:0:0, and summing the
    // products there would read it as a 16-block MCU and reject a grayscale
    // image the decoder supports.
    {
        const size_t gray_sof_offset = FindMarker(gray_seed, 0xC0);
        if (gray_sof_offset + 12 > gray_seed.size()) {
            std::cerr << "[seed/gray] the grayscale seed has no usable SOF0 segment" << std::endl;
            return EXIT_FAILURE;
        }
        for (uint8_t factors : {static_cast<uint8_t>(0x11), static_cast<uint8_t>(0x44)}) {
            std::vector<uint8_t> gray_frame = gray_seed;
            gray_frame[gray_sof_offset + 11] = factors;
            accepted_cases.push_back({factors == 0x11 ? "grayscale SOF sampling 1x1" : "grayscale SOF sampling 4x4",
                                      gray_frame});
        }

        // The case above reads components 1 and 2 to decide 4:0:0, so it holds
        // only while nothing can leave stale factors in them. A frame header is
        // the one thing that writes those slots, and it writes only as many as
        // it declares, so a one-component frame behind a three-component one
        // inherits the earlier chroma factors and classifies as 4:2:0 while
        // reporting a single component - the same stream shape the
        // component-count invariant further down is meant to exclude.
        const std::vector<uint8_t> three_component_sof = {0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x10, 0x00, 0x10, 0x03,
                                                          0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01};
        std::vector<uint8_t> two_frames;
        two_frames.insert(two_frames.end(), gray_seed.begin(), gray_seed.begin() + gray_sof_offset);
        two_frames.insert(two_frames.end(), three_component_sof.begin(), three_component_sof.end());
        two_frames.insert(two_frames.end(), gray_seed.begin() + gray_sof_offset, gray_seed.end());
        regressions.push_back({"a second frame header behind the first", two_frames});
    }

    // The same duplicate frame, spelled with each of the other frame markers.
    //
    // A frame header is a range of marker codes, not one code: B.1.1.3 gives the
    // whole of 0xC0 to 0xCF to frame headers except for DHT, JPG and DAC. Only
    // SOF0 is baseline sequential DCT, but the rest still have to be recognised
    // as frames, because a marker the parser does not know is skipped as an
    // unknown segment - so a stream pairing a valid SOF0 with an SOF1 would
    // carry two frame headers past both the duplicate check and the
    // unsupported-process check, which is the state the case above exists to
    // exclude.
    {
        const size_t sof_offset = FindMarker(color_seed, 0xC0);
        if (sof_offset + 4 > color_seed.size()) {
            std::cerr << "The color fuzzing seed no longer carries a parsable SOF segment." << std::endl;
            return EXIT_FAILURE;
        }
        const size_t sof_end =
            sof_offset + 2 + ((static_cast<size_t>(color_seed[sof_offset + 2]) << 8) | color_seed[sof_offset + 3]);
        if (sof_end > color_seed.size()) {
            std::cerr << "The color fuzzing seed no longer carries a parsable SOF segment." << std::endl;
            return EXIT_FAILURE;
        }
        // A one-component frame, which is the payload that makes the duplicate
        // harmful: it leaves the colour frame's chroma factors in the slots it
        // does not write.
        std::vector<uint8_t> second_frame = {0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x10, 0x00, 0x10, 0x01,
                                             0x01, 0x11, 0x00};
        for (uint8_t frame_marker = 0xC0; frame_marker <= 0xCF; frame_marker++) {
            // The three codes the standard reserves for other uses are not frame
            // headers and are covered by their own cases elsewhere.
            if (frame_marker == 0xC4 || frame_marker == 0xC8 || frame_marker == 0xCC) {
                continue;
            }
            second_frame[1] = frame_marker;
            const std::string name = "a second frame header spelled 0xFF" + ToHex(frame_marker);

            std::vector<uint8_t> duplicate_frame(color_seed.begin(), color_seed.begin() + sof_end);
            duplicate_frame.insert(duplicate_frame.end(), second_frame.begin(), second_frame.end());
            duplicate_frame.insert(duplicate_frame.end(), color_seed.begin() + sof_end, color_seed.end());
            regressions.push_back({name, duplicate_frame});

            // The same marker as the stream's only frame header. Every code but
            // SOF0 names a coding process this library does not decode, so these
            // have to be rejected on the marker alone rather than only when a
            // frame has already been seen.
            if (frame_marker != 0xC0) {
                std::vector<uint8_t> lone_frame = color_seed;
                lone_frame[sof_offset + 1] = frame_marker;
                regressions.push_back({"a lone frame header spelled 0xFF" + ToHex(frame_marker), lone_frame});
            }
        }
    }

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

    for (const RegressionCase &accepted_case : accepted_cases) {
        RocJpegStatus rocjpeg_status = ParseExactBuffer(accepted_case.data);
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            std::cerr << "[" << accepted_case.name << "] Expected a well-formed stream to be accepted but it was "
                         "rejected with " << rocJpegGetErrorName(rocjpeg_status) << "\n  "
                      << HexDump(accepted_case.data) << std::endl;
            return EXIT_FAILURE;
        }
        if (CheckParseInvariants(accepted_case.data, accepted_case.name)) {
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

        switch (random.Below(7)) {
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
                // Replace the scan body with a short sequence of markers and
                // stuffing. The other strategies edit bytes in place or cut the
                // stream down, so none of them can produce a stream that keeps
                // every header intact while emptying the scan of coded data -
                // which is exactly the shape the empty-scan postcondition is
                // about. Building it directly keeps that case reachable.
                static const uint8_t kScanTokens[] = {0xFF, 0xD0, 0xD7, 0x00, 0xD9, 0x42};
                const size_t scan_sos = FindStructuralMarker(mutated, 0xDA);
                if (scan_sos + 4 <= mutated.size()) {
                    const size_t scan_segment_length =
                        (static_cast<size_t>(mutated[scan_sos + 2]) << 8) | mutated[scan_sos + 3];
                    const size_t scan_body = scan_sos + 2 + scan_segment_length;
                    if (scan_body <= mutated.size()) {
                        mutated.resize(scan_body);
                        const uint32_t tokens = random.Below(9);
                        for (uint32_t i = 0; i < tokens; i++) {
                            mutated.push_back(kScanTokens[random.Below(sizeof(kScanTokens) / sizeof(kScanTokens[0]))]);
                        }
                        if (random.Below(4) != 0) {
                            mutated.push_back(0xFF);
                            mutated.push_back(0xD9);
                        }
                    }
                }
                break;
            }
            case 5: {
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
