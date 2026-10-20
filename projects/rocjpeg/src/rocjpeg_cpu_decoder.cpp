/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "rocjpeg_cpu_decoder.h"
#include "rocjpeg_commons.h"

#include <algorithm>
#include <new>

namespace {

/**
 * @brief Maps the zigzag position of a coefficient to its row-major position in the 8x8 block (ISO/IEC 10918-1
 *        Figure A.6).
 *
 * The 16 extra entries absorb run lengths that step past the last coefficient in corrupt data: such a coefficient
 * lands on position 63 instead of outside the block, which is also what libjpeg-turbo does.
 */
constexpr uint8_t kZigzagToNatural[64 + 16] = {
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
};

constexpr int kMaxCodeLength = 16;

/**
 * @brief A Huffman decoding table built as described in ISO/IEC 10918-1 F.2.2.3 (MAXCODE, VALPTR, HUFFVAL).
 */
struct HuffmanTable {
    int32_t max_code[kMaxCodeLength + 1];   // Largest code of each length, or -1 if there is none.
    int32_t value_offset[kMaxCodeLength + 1]; // Added to a code of that length to index values.
    uint8_t values[256];
};

/**
 * @brief Builds a Huffman decoding table from the code counts per length and the symbol values.
 * @param counts The number of codes of each length 1 to 16.
 * @param values The symbol values in code order.
 * @param max_values The capacity of values.
 * @param is_dc Whether the table codes DC differences, whose magnitude categories cannot exceed 15.
 * @param table Receives the decoding table.
 * @return false if the counts describe more codes than the values hold, or more codes of a length than fit in it.
 */
bool BuildHuffmanTable(const uint8_t *counts, const uint8_t *values, int max_values, bool is_dc, HuffmanTable &table) {
    int total = 0;
    for (int length = 1; length <= kMaxCodeLength; length++) {
        total += counts[length - 1];
    }
    if (total > max_values) {
        return false;
    }
    int32_t code = 0;
    int index = 0;
    for (int length = 1; length <= kMaxCodeLength; length++) {
        int count = counts[length - 1];
        if (count == 0) {
            table.max_code[length] = -1;
            table.value_offset[length] = 0;
        } else {
            table.value_offset[length] = index - code;
            index += count;
            code += count;
            // The all-ones code of each length is reserved (ISO/IEC 10918-1 C.2), so the codes must stay below it.
            if (code > (1 << length) - 1) {
                return false;
            }
            table.max_code[length] = code - 1;
        }
        code <<= 1;
    }
    for (int i = 0; i < total; i++) {
        if (is_dc && values[i] > 15) {
            return false;
        }
        table.values[i] = values[i];
    }
    return true;
}

constexpr uint8_t kMarkerRst0 = 0xD0;
constexpr uint8_t kMarkerRst7 = 0xD7;
constexpr uint8_t kMarkerSof0 = 0xC0;
constexpr uint8_t kMarkerEoi = 0xD9;

/**
 * @brief Reads bits from entropy-coded data, removing byte stuffing and stopping at markers.
 *
 * It follows libjpeg-turbo's handling of markers and missing data:
 * - A marker ends the entropy-coded data. It stays unread until a restart, and the end of the data counts as an EOI
 *   marker there, as libjpeg-turbo's memory source inserts one.
 * - Bits needed beyond the end of the data read as 0, and set the insufficient-data state. In that state the decoder
 *   skips the remaining MCUs until a restart marker is read, which leaves those blocks at the mid-level value.
 * - A restart reads the next marker and resynchronizes the way jpeg_resync_to_restart does when it is not the
 *   expected restart marker.
 */
class BitReader {
    public:
        BitReader(const uint8_t *data, size_t size) : data_(data), size_(size) {}

        uint32_t GetBits(int count) {
            while (bit_count_ < count) {
                buffer_ = (buffer_ << 8) | NextByte();
                bit_count_ += 8;
            }
            bit_count_ -= count;
            uint32_t bits = static_cast<uint32_t>(buffer_ >> bit_count_) & ((1u << count) - 1);
            buffer_ &= (static_cast<uint64_t>(1) << bit_count_) - 1;
            return bits;
        }

        bool InsufficientData() const { return insufficient_data_; }

        /**
         * @brief Discards the remaining bits and reads the restart marker that ends the current interval.
         */
        void Restart() {
            buffer_ = 0;
            bit_count_ = 0;
            if (unread_marker_ == 0) {
                FindNextMarker(pos_);
            }
            const int desired = next_restart_number_;
            next_restart_number_ = (next_restart_number_ + 1) & 7;
            if (unread_marker_ == kMarkerRst0 + desired) {
                ConsumeMarker();
            } else {
                Resync(desired);
            }
            // If the marker is still unread, the next interval is empty, and staying in the insufficient-data state
            // avoids decoding it from padding bits.
            if (unread_marker_ == 0) {
                insufficient_data_ = false;
            }
        }

    private:
        uint32_t NextByte() {
            if (unread_marker_ != 0) {
                insufficient_data_ = true;
                return 0;
            }
            if (pos_ >= size_) {
                SetMarkerAtEnd();
                insufficient_data_ = true;
                return 0;
            }
            uint8_t byte = data_[pos_];
            if (byte != 0xFF) {
                pos_++;
                return byte;
            }
            // Skip fill bytes; the byte after them is either stuffing or a marker code.
            size_t next = pos_ + 1;
            while (next < size_ && data_[next] == 0xFF) {
                next++;
            }
            if (next >= size_) {
                SetMarkerAtEnd();
                insufficient_data_ = true;
                return 0;
            }
            if (data_[next] == 0x00) {
                pos_ = next + 1;
                return 0xFF;
            }
            unread_marker_ = data_[next];
            marker_code_pos_ = next;
            insufficient_data_ = true;
            return 0;
        }

        void SetMarkerAtEnd() {
            pos_ = size_;
            unread_marker_ = kMarkerEoi;
            marker_code_pos_ = size_;
        }

        /**
         * @brief Finds the next marker at or after from, skipping other bytes, fill bytes, and stuffed zeros.
         */
        void FindNextMarker(size_t from) {
            size_t pos = from;
            while (true) {
                while (pos < size_ && data_[pos] != 0xFF) {
                    pos++;
                }
                while (pos < size_ && data_[pos] == 0xFF) {
                    pos++;
                }
                if (pos >= size_) {
                    SetMarkerAtEnd();
                    return;
                }
                if (data_[pos] != 0x00) {
                    unread_marker_ = data_[pos];
                    marker_code_pos_ = pos;
                    return;
                }
                pos++;
            }
        }

        void ConsumeMarker() {
            pos_ = marker_code_pos_ + 1;
            unread_marker_ = 0;
        }

        /**
         * @brief Resynchronizes after a marker other than the expected restart marker.
         *
         * Invalid markers and the two restart markers before the expected one are discarded and the search goes on.
         * A valid non-restart marker, or one of the two restart markers after the expected one, stays unread so that
         * the next interval is empty. Any other restart marker is taken as the expected one.
         */
        void Resync(int desired) {
            while (true) {
                const int marker = unread_marker_;
                int action;
                if (marker < kMarkerSof0) {
                    action = 2;
                } else if (marker < kMarkerRst0 || marker > kMarkerRst7) {
                    action = 3;
                } else if (marker == kMarkerRst0 + ((desired + 1) & 7) || marker == kMarkerRst0 + ((desired + 2) & 7)) {
                    action = 3;
                } else if (marker == kMarkerRst0 + ((desired - 1) & 7) || marker == kMarkerRst0 + ((desired - 2) & 7)) {
                    action = 2;
                } else {
                    action = 1;
                }
                if (action == 1) {
                    ConsumeMarker();
                    return;
                }
                if (action == 3) {
                    return;
                }
                ConsumeMarker();
                FindNextMarker(pos_);
            }
        }

        const uint8_t *data_;
        size_t size_;
        size_t pos_ = 0;
        uint64_t buffer_ = 0;
        int bit_count_ = 0;
        int unread_marker_ = 0;
        size_t marker_code_pos_ = 0;
        int next_restart_number_ = 0;
        bool insufficient_data_ = false;
};

/**
 * @brief Decodes one Huffman-coded symbol (ISO/IEC 10918-1 F.2.2.3).
 *
 * An invalid code decodes as 0 after consuming 17 bits, which keeps the bit position in step with libjpeg-turbo
 * on corrupt data.
 */
uint8_t DecodeSymbol(BitReader &reader, const HuffmanTable &table) {
    int32_t code = static_cast<int32_t>(reader.GetBits(1));
    int length = 1;
    while (code > table.max_code[length]) {
        code = (code << 1) | static_cast<int32_t>(reader.GetBits(1));
        if (++length > kMaxCodeLength) {
            return 0;
        }
    }
    return table.values[code + table.value_offset[length]];
}

/**
 * @brief Converts the additional bits of a coefficient to its signed value (ISO/IEC 10918-1 F.2.2.1, EXTEND).
 */
int32_t Extend(uint32_t bits, int category) {
    return bits < (1u << (category - 1)) ? static_cast<int32_t>(bits) - (1 << category) + 1 : static_cast<int32_t>(bits);
}

// Accurate integer inverse DCT: the separable algorithm of Loeffler, Ligtenberg and Moschytz with 13-bit fixed-point
// constants, keeping 2 extra bits of precision between the column and the row pass. These are the parameters of
// libjpeg-turbo's default ("islow") method, so the results match it exactly.
constexpr int kConstBits = 13;
constexpr int kPass1Bits = 2;
constexpr int64_t kFix_0_298631336 = 2446;
constexpr int64_t kFix_0_390180644 = 3196;
constexpr int64_t kFix_0_541196100 = 4433;
constexpr int64_t kFix_0_765366865 = 6270;
constexpr int64_t kFix_0_899976223 = 7373;
constexpr int64_t kFix_1_175875602 = 9633;
constexpr int64_t kFix_1_501321110 = 12299;
constexpr int64_t kFix_1_847759065 = 15137;
constexpr int64_t kFix_1_961570560 = 16069;
constexpr int64_t kFix_2_053119869 = 16819;
constexpr int64_t kFix_2_562915447 = 20995;
constexpr int64_t kFix_3_072711026 = 25172;

inline int64_t Descale(int64_t value, int shift) {
    return (value + (static_cast<int64_t>(1) << (shift - 1))) >> shift;
}

/**
 * @brief Computes the 8-point 1-D inverse DCT of in[0], in[stride], ..., in[7 * stride].
 * @param out Receives the 8 outputs, before descaling.
 */
inline void Idct1D(const int64_t *in, int stride, int64_t out[8]) {
    // Even part.
    int64_t z2 = in[2 * stride];
    int64_t z3 = in[6 * stride];
    int64_t z1 = (z2 + z3) * kFix_0_541196100;
    int64_t tmp2 = z1 - z3 * kFix_1_847759065;
    int64_t tmp3 = z1 + z2 * kFix_0_765366865;
    int64_t tmp0 = (in[0] + in[4 * stride]) * (static_cast<int64_t>(1) << kConstBits);
    int64_t tmp1 = (in[0] - in[4 * stride]) * (static_cast<int64_t>(1) << kConstBits);
    int64_t tmp10 = tmp0 + tmp3;
    int64_t tmp13 = tmp0 - tmp3;
    int64_t tmp11 = tmp1 + tmp2;
    int64_t tmp12 = tmp1 - tmp2;

    // Odd part.
    tmp0 = in[7 * stride];
    tmp1 = in[5 * stride];
    tmp2 = in[3 * stride];
    tmp3 = in[1 * stride];
    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    int64_t z4 = tmp1 + tmp3;
    int64_t z5 = (z3 + z4) * kFix_1_175875602;
    tmp0 *= kFix_0_298631336;
    tmp1 *= kFix_2_053119869;
    tmp2 *= kFix_3_072711026;
    tmp3 *= kFix_1_501321110;
    z1 *= -kFix_0_899976223;
    z2 *= -kFix_2_562915447;
    z3 = z3 * -kFix_1_961570560 + z5;
    z4 = z4 * -kFix_0_390180644 + z5;
    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;

    out[0] = tmp10 + tmp3;
    out[7] = tmp10 - tmp3;
    out[1] = tmp11 + tmp2;
    out[6] = tmp11 - tmp2;
    out[2] = tmp12 + tmp1;
    out[5] = tmp12 - tmp1;
    out[3] = tmp13 + tmp0;
    out[4] = tmp13 - tmp0;
}

/**
 * @brief Clamps a level-shifted sample to 0..255 the way libjpeg-turbo's range-limit table does.
 *
 * The table is indexed with the low 10 bits of the value, so a value far outside the sample range (only possible in
 * corrupt data) wraps around before it is clamped. For in-range data this is a plain clamp.
 */
inline uint8_t RangeLimit(int64_t value) {
    int32_t wrapped = static_cast<int32_t>(value & 1023);
    if (wrapped >= 512) {
        wrapped -= 1024;
    }
    wrapped += 128;
    return static_cast<uint8_t>(wrapped < 0 ? 0 : (wrapped > 255 ? 255 : wrapped));
}

/**
 * @brief Dequantizes a block of coefficients and writes its inverse DCT to the plane.
 * @param coefficients The quantized coefficients in row-major order.
 * @param quantization The quantization table in row-major order.
 */
void InverseDct(const int16_t coefficients[64], const uint16_t quantization[64], uint8_t *output, size_t stride) {
    int64_t dequantized[64];
    for (int i = 0; i < 64; i++) {
        dequantized[i] = static_cast<int64_t>(coefficients[i]) * quantization[i];
    }
    int64_t workspace[64];
    int64_t column[8];
    for (int x = 0; x < 8; x++) {
        Idct1D(&dequantized[x], 8, column);
        for (int y = 0; y < 8; y++) {
            workspace[y * 8 + x] = Descale(column[y], kConstBits - kPass1Bits);
        }
    }
    int64_t row[8];
    for (int y = 0; y < 8; y++) {
        Idct1D(&workspace[y * 8], 1, row);
        uint8_t *out = output + y * stride;
        for (int x = 0; x < 8; x++) {
            // The extra factor of 8 (3 bits) is the normalization of the 2-D inverse DCT.
            out[x] = RangeLimit(Descale(row[x], kConstBits + kPass1Bits + 3));
        }
    }
}

struct ScanComponent {
    int frame_index;
    int h_sampling;
    int v_sampling;
    const HuffmanTable *dc_table;
    const HuffmanTable *ac_table;
    uint16_t quantization[64];
    int32_t dc_predictor;
};

/**
 * @brief Decodes one 8x8 block (ISO/IEC 10918-1 F.2.2) and writes its samples.
 */
void DecodeBlock(BitReader &reader, ScanComponent &component, uint8_t *output, size_t stride) {
    int16_t coefficients[64] = {};
    int category = DecodeSymbol(reader, *component.dc_table);
    int32_t diff = category ? Extend(reader.GetBits(category), category) : 0;
    // Corrupt data can push the predictor past the int32_t range; wrap it rather than overflow.
    component.dc_predictor = static_cast<int32_t>(static_cast<uint32_t>(component.dc_predictor) + static_cast<uint32_t>(diff));
    coefficients[0] = static_cast<int16_t>(component.dc_predictor);
    for (int k = 1; k < 64; k++) {
        int run_size = DecodeSymbol(reader, *component.ac_table);
        int run = run_size >> 4;
        int size = run_size & 15;
        if (size) {
            k += run;
            coefficients[kZigzagToNatural[k]] = static_cast<int16_t>(Extend(reader.GetBits(size), size));
        } else {
            if (run != 15) {
                break;
            }
            k += 15;
        }
    }
    InverseDct(coefficients, component.quantization, output, stride);
}

uint32_t CeilDiv(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

}  // namespace

RocJpegStatus RocJpegCpuDecoder::Decode(const JpegStreamParameters &jpeg_stream_params, CpuDecodedImage &image) {
    const PictureParameterBuffer &picture = jpeg_stream_params.picture_parameter_buffer;
    const SliceParameterBuffer &slice = jpeg_stream_params.slice_parameter_buffer;
    const QuantizationMatrixBuffer &quantization = jpeg_stream_params.quantization_matrix_buffer;
    const HuffmanTableBuffer &huffman = jpeg_stream_params.huffman_table_buffer;

    if (picture.num_components == 0 || picture.num_components > NUM_COMPONENTS - 1 ||
        slice.num_components != picture.num_components || picture.picture_width == 0 || picture.picture_height == 0 ||
        jpeg_stream_params.slice_data_buffer == nullptr) {
        ErrorLog(g_rocjpeg_logger, "The CPU decoder needs one scan that covers every component of a non-empty image");
        return ROCJPEG_STATUS_BAD_JPEG;
    }

    HuffmanTable dc_tables[HUFFMAN_TABLES];
    HuffmanTable ac_tables[HUFFMAN_TABLES];
    bool dc_valid[HUFFMAN_TABLES] = {};
    bool ac_valid[HUFFMAN_TABLES] = {};
    for (int t = 0; t < HUFFMAN_TABLES; t++) {
        if (!huffman.load_huffman_table[t]) {
            continue;
        }
        const auto &table = huffman.huffman_table[t];
        dc_valid[t] = BuildHuffmanTable(table.num_dc_codes, table.dc_values, sizeof(table.dc_values), true, dc_tables[t]);
        ac_valid[t] = BuildHuffmanTable(table.num_ac_codes, table.ac_values, sizeof(table.ac_values), false, ac_tables[t]);
    }

    int h_max = 1;
    int v_max = 1;
    for (int c = 0; c < picture.num_components; c++) {
        h_max = std::max<int>(h_max, picture.components[c].h_sampling_factor);
        v_max = std::max<int>(v_max, picture.components[c].v_sampling_factor);
    }

    ScanComponent components[NUM_COMPONENTS - 1];
    for (int s = 0; s < slice.num_components; s++) {
        ScanComponent &component = components[s];
        // The scan codes the components in frame order, as RocJpegStreamParser requires. Matching by position rather
        // than searching by ID gives every component its own plane even if a stream repeats a component ID.
        component.frame_index = s;
        if (picture.components[s].component_id != slice.components[s].component_selector) {
            ErrorLog(g_rocjpeg_logger, "The scan does not code the components in frame order");
            return ROCJPEG_STATUS_BAD_JPEG;
        }
        const auto &frame_component = picture.components[component.frame_index];
        int dc_selector = slice.components[s].dc_table_selector;
        int ac_selector = slice.components[s].ac_table_selector;
        int q_selector = frame_component.quantiser_table_selector;
        if (frame_component.h_sampling_factor < 1 || frame_component.h_sampling_factor > 4 ||
            frame_component.v_sampling_factor < 1 || frame_component.v_sampling_factor > 4 ||
            dc_selector >= HUFFMAN_TABLES || ac_selector >= HUFFMAN_TABLES || !dc_valid[dc_selector] ||
            !ac_valid[ac_selector] || q_selector >= 4 || !quantization.load_quantiser_table[q_selector]) {
            ErrorLog(g_rocjpeg_logger, "A scan component has an invalid sampling factor, or uses a missing or invalid table");
            return ROCJPEG_STATUS_BAD_JPEG;
        }
        component.h_sampling = frame_component.h_sampling_factor;
        component.v_sampling = frame_component.v_sampling_factor;
        component.dc_table = &dc_tables[dc_selector];
        component.ac_table = &ac_tables[ac_selector];
        // The tables are stored in zigzag order, as they appear in the DQT segment.
        for (int k = 0; k < 64; k++) {
            component.quantization[kZigzagToNatural[k]] = quantization.quantiser_table[q_selector][k];
        }
        component.dc_predictor = 0;
    }

    const uint32_t width = picture.picture_width;
    const uint32_t height = picture.picture_height;
    const bool interleaved = slice.num_components > 1;
    // An interleaved scan is made of MCUs that each hold h x v blocks of every component (ISO/IEC 10918-1 A.2.3).
    // A single-component scan codes the component's blocks one by one, whatever its sampling factors (A.2.2).
    const uint32_t mcus_x = interleaved ? CeilDiv(width, 8 * h_max) : 0;
    const uint32_t mcus_y = interleaved ? CeilDiv(height, 8 * v_max) : 0;

    image = CpuDecodedImage();
    image.num_components = picture.num_components;
    try {
        for (int s = 0; s < slice.num_components; s++) {
            const ScanComponent &component = components[s];
            int c = component.frame_index;
            image.widths[c] = CeilDiv(width * component.h_sampling, h_max);
            image.heights[c] = CeilDiv(height * component.v_sampling, v_max);
            uint32_t blocks_x = interleaved ? mcus_x * component.h_sampling : CeilDiv(image.widths[c], 8);
            uint32_t blocks_y = interleaved ? mcus_y * component.v_sampling : CeilDiv(image.heights[c], 8);
            image.strides[c] = blocks_x * 8;
            image.planes[c].resize(static_cast<size_t>(image.strides[c]) * blocks_y * 8);
        }
    } catch (const std::bad_alloc &) {
        image = CpuDecodedImage();
        ErrorLog(g_rocjpeg_logger, "Failed to allocate the decoded planes");
        return ROCJPEG_STATUS_OUTOF_MEMORY;
    }

    BitReader reader(jpeg_stream_params.slice_data_buffer, slice.slice_data_size);
    const uint32_t restart_interval = slice.restart_interval;
    uint32_t mcu_count = 0;
    // Processes a restart if one is due, and returns whether the MCU is skipped because the data ran out. A skipped
    // MCU leaves its blocks at the mid-level value and the DC predictors unchanged, as in libjpeg-turbo.
    auto start_mcu = [&]() {
        if (restart_interval != 0 && mcu_count != 0 && mcu_count % restart_interval == 0) {
            reader.Restart();
            for (int s = 0; s < slice.num_components; s++) {
                components[s].dc_predictor = 0;
            }
        }
        mcu_count++;
        return reader.InsufficientData();
    };
    auto process_block = [&](bool skip, ScanComponent &component, uint8_t *output, size_t stride) {
        if (skip) {
            for (int y = 0; y < 8; y++) {
                std::fill_n(output + y * stride, 8, static_cast<uint8_t>(128));
            }
        } else {
            DecodeBlock(reader, component, output, stride);
        }
    };

    if (interleaved) {
        for (uint32_t mcu_y = 0; mcu_y < mcus_y; mcu_y++) {
            for (uint32_t mcu_x = 0; mcu_x < mcus_x; mcu_x++) {
                const bool skip = start_mcu();
                for (int s = 0; s < slice.num_components; s++) {
                    ScanComponent &component = components[s];
                    int c = component.frame_index;
                    size_t stride = image.strides[c];
                    for (int v = 0; v < component.v_sampling; v++) {
                        for (int h = 0; h < component.h_sampling; h++) {
                            size_t block_x = static_cast<size_t>(mcu_x) * component.h_sampling + h;
                            size_t block_y = static_cast<size_t>(mcu_y) * component.v_sampling + v;
                            process_block(skip, component, image.planes[c].data() + block_y * 8 * stride + block_x * 8, stride);
                        }
                    }
                }
            }
        }
    } else {
        ScanComponent &component = components[0];
        int c = component.frame_index;
        size_t stride = image.strides[c];
        uint32_t blocks_x = image.strides[c] / 8;
        uint32_t blocks_y = static_cast<uint32_t>(image.planes[c].size() / stride / 8);
        for (uint32_t block_y = 0; block_y < blocks_y; block_y++) {
            for (uint32_t block_x = 0; block_x < blocks_x; block_x++) {
                const bool skip = start_mcu();
                process_block(skip, component, image.planes[c].data() + static_cast<size_t>(block_y) * 8 * stride + static_cast<size_t>(block_x) * 8, stride);
            }
        }
    }
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus RocJpegCpuDecoder::CheckOutput(RocJpegOutputFormat output_format, ChromaSubsampling subsampling) {
    if (output_format != ROCJPEG_OUTPUT_NATIVE && output_format != ROCJPEG_OUTPUT_YUV_PLANAR && output_format != ROCJPEG_OUTPUT_Y) {
        ErrorLog(g_rocjpeg_logger, "The CPU backend only implements the native, YUV planar, and Y output formats!");
        return ROCJPEG_STATUS_NOT_IMPLEMENTED;
    }
    if (subsampling != CSS_444 && subsampling != CSS_440 && subsampling != CSS_422 && subsampling != CSS_420 && subsampling != CSS_400) {
        ErrorLog(g_rocjpeg_logger, "The chroma subsampling is not supported by the CPU backend!");
        return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
    }
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus RocJpegCpuDecoder::LayOutOutput(const JpegStreamParameters &jpeg_stream_params, const CpuDecodedImage &image,
                                              RocJpegOutputFormat output_format, CpuOutput &output) {
    const ChromaSubsampling subsampling = jpeg_stream_params.chroma_subsampling;
    RocJpegStatus rocjpeg_status = CheckOutput(output_format, subsampling);
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        return rocjpeg_status;
    }
    const uint32_t width = jpeg_stream_params.picture_parameter_buffer.picture_width;
    const uint32_t height = jpeg_stream_params.picture_parameter_buffer.picture_height;
    const bool half_width = subsampling == CSS_422 || subsampling == CSS_420;
    const bool half_height = subsampling == CSS_440 || subsampling == CSS_420;
    const uint32_t chroma_width = half_width ? width >> 1 : width;
    const uint32_t chroma_height = half_height ? height >> 1 : height;

    // The subsampling is classified from the sampling factors, and some factor combinations get a label whose
    // geometry they do not have (for example Y 2x2 with chroma 2x1 is labelled 4:2:2, but its chroma is full width
    // and half height). The output layout follows the label, so it is only produced when the planes have that
    // geometry; the reads below then stay within the planes.
    const int components = subsampling == CSS_400 ? 1 : 3;
    bool geometry_matches = image.num_components == components && image.widths[0] == width && image.heights[0] == height;
    for (int c = 1; c < components; c++) {
        const uint32_t expected_width = half_width ? (width + 1) >> 1 : width;
        const uint32_t expected_height = half_height ? (height + 1) >> 1 : height;
        geometry_matches = geometry_matches && image.widths[c] == expected_width && image.heights[c] == expected_height;
    }
    if (!geometry_matches) {
        ErrorLog(g_rocjpeg_logger, "The sampling factors do not have the geometry of their chroma subsampling, which the CPU backend does not support!");
        return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
    }

    output = CpuOutput();
    CpuOutputPlane *planes = output.planes;
    int &num_planes = output.num_planes;
    std::vector<uint8_t> &packed = output.packed;
    const uint8_t *y_plane = image.planes[0].data();
    const uint8_t *u_plane = image.planes[1].data();
    const uint8_t *v_plane = image.planes[2].data();
    try {
        if (output_format == ROCJPEG_OUTPUT_Y || subsampling == CSS_400) {
            planes[num_planes++] = {y_plane, image.strides[0], width, height};
        } else if (output_format == ROCJPEG_OUTPUT_YUV_PLANAR || subsampling == CSS_444 || subsampling == CSS_440) {
            planes[num_planes++] = {y_plane, image.strides[0], width, height};
            planes[num_planes++] = {u_plane, image.strides[1], chroma_width, chroma_height};
            planes[num_planes++] = {v_plane, image.strides[2], chroma_width, chroma_height};
        } else if (subsampling == CSS_420) {
            // NV12: luma, then rows of width bytes alternating U and V.
            packed.resize(static_cast<size_t>(width) * chroma_height);
            for (uint32_t y = 0; y < chroma_height; y++) {
                const uint8_t *u_row = u_plane + static_cast<size_t>(y) * image.strides[1];
                const uint8_t *v_row = v_plane + static_cast<size_t>(y) * image.strides[2];
                uint8_t *out = packed.data() + static_cast<size_t>(y) * width;
                for (uint32_t x = 0; x < width; x++) {
                    out[x] = (x & 1) ? v_row[x >> 1] : u_row[x >> 1];
                }
            }
            planes[num_planes++] = {y_plane, image.strides[0], width, height};
            planes[num_planes++] = {packed.data(), width, width, chroma_height};
        } else {
            // YUYV: one plane of 2 x width bytes per row, Y0 U0 Y1 V0.
            const size_t row_bytes = static_cast<size_t>(width) * 2;
            packed.resize(row_bytes * height);
            for (uint32_t y = 0; y < height; y++) {
                const uint8_t *y_row = y_plane + static_cast<size_t>(y) * image.strides[0];
                const uint8_t *u_row = u_plane + static_cast<size_t>(y) * image.strides[1];
                const uint8_t *v_row = v_plane + static_cast<size_t>(y) * image.strides[2];
                uint8_t *out = packed.data() + y * row_bytes;
                for (uint32_t x = 0; x < width; x++) {
                    out[2 * x] = y_row[x];
                    out[2 * x + 1] = (x & 1) ? v_row[x >> 1] : u_row[x >> 1];
                }
            }
            planes[num_planes++] = {packed.data(), row_bytes, static_cast<uint32_t>(row_bytes), height};
        }
    } catch (const std::bad_alloc &) {
        output = CpuOutput();
        ErrorLog(g_rocjpeg_logger, "Failed to allocate the packed output");
        return ROCJPEG_STATUS_OUTOF_MEMORY;
    }
    return ROCJPEG_STATUS_SUCCESS;
}
