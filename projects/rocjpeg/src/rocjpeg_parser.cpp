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
#include "rocjpeg_parser.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

RocJpegStreamParser::RocJpegStreamParser() : stream_{nullptr}, stream_start_{nullptr}, stream_end_{nullptr}, stream_length_{0},
    jpeg_stream_parameters_{} {
}

RocJpegStreamParser::~RocJpegStreamParser() {
    stream_ = nullptr;
    stream_end_ = nullptr;
    stream_length_ = 0;
}

/**
 * @brief Parses a JPEG stream.
 *
 * This function parses a JPEG stream and extracts various markers and parameters from it.
 *
 * @param jpeg_stream A pointer to the JPEG stream.
 * @param jpeg_stream_size The size of the JPEG stream in bytes.
 * @return True if the JPEG stream was successfully parsed, false otherwise.
 */
bool RocJpegStreamParser::ParseJpegStream(const uint8_t *jpeg_stream, uint32_t jpeg_stream_size) {
    FunctionEntryLogWithArgs(g_rocjpeg_logger, RocJpegFmtPtr(jpeg_stream) + ", " + ROCJPEG_TOSTR(jpeg_stream_size));
    std::lock_guard<std::mutex> lock(mutex_);
    if (jpeg_stream == nullptr) {
        CriticalLog(g_rocjpeg_logger, "Invalid argument!");
        FunctionExitLog(g_rocjpeg_logger);
        return false;
    }

    stream_ = jpeg_stream;
    stream_start_ = jpeg_stream;
    stream_length_ = jpeg_stream_size;
    stream_end_ = stream_ + stream_length_;

    jpeg_stream_parameters_ = {};
    bool soi_marker_found = false;
    bool sos_marker_found = false;
    bool dht_marker_found = false;
    bool dqt_marker_found = false;
    uint8_t marker;
    const uint8_t *next_chunck;
    int32_t chuck_len;

    // The first two bytes of a JPEG must be 0XFFD8
    if (*stream_ != 0xFF || *(stream_ + 1) != SOI) {
        ErrorLog(g_rocjpeg_logger, "Invalid JPEG!");
        FunctionExitLog(g_rocjpeg_logger);
        return false;
    }

    soi_marker_found = ParseSOI();
    if (!soi_marker_found) {
        ErrorLog(g_rocjpeg_logger, "Failed to find the SOI marker!");
    }

    while (!sos_marker_found  && stream_ <= stream_end_) {
        while ((*stream_ == 0xFF))
            stream_++;
        marker = *stream_++;
        chuck_len = swap_bytes(stream_);
        next_chunck = stream_ + chuck_len;

        switch (marker) {
            case SOF:
                if (!ParseSOF()) {
                    FunctionExitLog(g_rocjpeg_logger);
                    return false;
                }
                break;
            case SOF2:
                ErrorLog(g_rocjpeg_logger, "Progressive JPEG is not supported!");
                FunctionExitLog(g_rocjpeg_logger);
                return false;
            case DHT:
                if (!ParseDHT()) {
                    FunctionExitLog(g_rocjpeg_logger);
                    return false;
                }
                dht_marker_found = true;
                break;
            case DQT:
                if (!ParseDQT()) {
                    FunctionExitLog(g_rocjpeg_logger);
                    return false;
                }
                dqt_marker_found = true;
                break;
            case DRI:
                if (!ParseDRI()) {
                    FunctionExitLog(g_rocjpeg_logger);
                    return false;
                }
                break;
            case SOS:
                if (!ParseSOS()) {
                    FunctionExitLog(g_rocjpeg_logger);
                    return false;
                }
                sos_marker_found = true;
                break;
            default:
                break;
        }
        stream_ = next_chunck;
    }

    if (!dht_marker_found) {
        ErrorLog(g_rocjpeg_logger, "Didn't find any Huffman table!");
        FunctionExitLog(g_rocjpeg_logger);
        return false;
    }
    if (!dqt_marker_found) {
        ErrorLog(g_rocjpeg_logger, "Didn't find any quantization table!");
        FunctionExitLog(g_rocjpeg_logger);
        return false;
    }

    if (!ParseEOI()) {
        FunctionExitLog(g_rocjpeg_logger);
        return false;
    }

    FunctionExitLog(g_rocjpeg_logger);
    return true;
}

/**
 * @brief Parses the Start of Image (SOI) marker in the JPEG stream.
 *
 * This function searches for the SOI marker (0xFFD8) in the JPEG stream and updates the stream pointer accordingly.
 *
 * @return true if the SOI marker is found and the stream pointer is updated, false otherwise.
 */
bool RocJpegStreamParser::ParseSOI() {
    if (stream_ == nullptr) {
        return false;
    }
    while (!(*stream_ == 0xFF  && *(stream_ + 1) == SOI)) {
        if (stream_ <= stream_end_) {
            stream_++;
            continue;
        } else
            return false;
    }
    stream_ += 2;

    return true;
}

/**
 * @brief Parses the Start of Frame (SOF) marker in the JPEG stream.
 *
 * This function reads and processes the SOF marker in the JPEG stream. It extracts
 * information such as picture height, picture width, number of components, component
 * IDs, sampling factors, and quantization table selectors. It also calculates the
 * number of MCU (Minimum Coded Unit) blocks in the image and determines the chroma
 * subsampling scheme.
 *
 * @return true if the SOF marker is successfully parsed, false otherwise.
 */
bool RocJpegStreamParser::ParseSOF() {
    uint32_t component_id, sampling_factor;
    uint8_t quantiser_table_selector;

    if (stream_ == nullptr) {
        return false;
    }

    const uint8_t *sof_header = stream_;

    jpeg_stream_parameters_.picture_parameter_buffer.picture_height = swap_bytes(stream_ + 3);
    jpeg_stream_parameters_.picture_parameter_buffer.picture_width = swap_bytes(stream_ + 5);
    jpeg_stream_parameters_.picture_parameter_buffer.num_components = stream_[7];

    if (jpeg_stream_parameters_.picture_parameter_buffer.num_components > NUM_COMPONENTS - 1) {
        ErrorLog(g_rocjpeg_logger, "Unsupported JPEG: " +
            ROCJPEG_TOSTR(static_cast<int>(jpeg_stream_parameters_.picture_parameter_buffer.num_components)) +
            " components found (CMYK/YCCK not supported, only YCbCr/grayscale with up to 3 components)!");
        return false;
    }

    stream_ += 8;

    for (int32_t i = 0; i < jpeg_stream_parameters_.picture_parameter_buffer.num_components; i++) {
        component_id = *stream_++;
        sampling_factor = *stream_++;
        quantiser_table_selector = *stream_++;

        jpeg_stream_parameters_.picture_parameter_buffer.components[i].component_id = component_id;
        if (quantiser_table_selector >= NUM_COMPONENTS) {
            ErrorLog(g_rocjpeg_logger,"invalid number of the quantization table!");
            return false;
        }
        jpeg_stream_parameters_.picture_parameter_buffer.components[i].v_sampling_factor = sampling_factor & 0xF;
        jpeg_stream_parameters_.picture_parameter_buffer.components[i].h_sampling_factor = sampling_factor >> 4;
        jpeg_stream_parameters_.picture_parameter_buffer.components[i].quantiser_table_selector = quantiser_table_selector;
    }

    uint8_t max_h_factor = jpeg_stream_parameters_.picture_parameter_buffer.components[0].h_sampling_factor;
    uint8_t max_v_factor = jpeg_stream_parameters_.picture_parameter_buffer.components[0].v_sampling_factor;

    jpeg_stream_parameters_.slice_parameter_buffer.num_mcus = ((jpeg_stream_parameters_.picture_parameter_buffer.picture_width + max_h_factor * 8 - 1) / (max_h_factor * 8)) *
                                       ((jpeg_stream_parameters_.picture_parameter_buffer.picture_height + max_v_factor * 8 - 1) / (max_v_factor * 8));

    jpeg_stream_parameters_.chroma_subsampling = GetChromaSubsampling(jpeg_stream_parameters_.picture_parameter_buffer.components[0].h_sampling_factor,
                                                                      jpeg_stream_parameters_.picture_parameter_buffer.components[1].h_sampling_factor,
                                                                      jpeg_stream_parameters_.picture_parameter_buffer.components[2].h_sampling_factor,
                                                                      jpeg_stream_parameters_.picture_parameter_buffer.components[0].v_sampling_factor,
                                                                      jpeg_stream_parameters_.picture_parameter_buffer.components[1].v_sampling_factor,
                                                                      jpeg_stream_parameters_.picture_parameter_buffer.components[2].v_sampling_factor);

    if (jpeg_stream_parameters_.chroma_subsampling == CSS_411) {
        ErrorLog(g_rocjpeg_logger, "Unsupported chroma subsampling: 4:1:1 is not supported!");
        return false;
    }
    if (jpeg_stream_parameters_.chroma_subsampling == CSS_UNKNOWN) {
        uint8_t num_comp = jpeg_stream_parameters_.picture_parameter_buffer.num_components;
        std::string msg = "Unsupported chroma subsampling: unrecognized sampling factors "
            "(Y=" + ROCJPEG_TOSTR(static_cast<int>(jpeg_stream_parameters_.picture_parameter_buffer.components[0].h_sampling_factor)) +
            "x"  + ROCJPEG_TOSTR(static_cast<int>(jpeg_stream_parameters_.picture_parameter_buffer.components[0].v_sampling_factor));
        if (num_comp > 1) {
            msg += ", Cb=" + ROCJPEG_TOSTR(static_cast<int>(jpeg_stream_parameters_.picture_parameter_buffer.components[1].h_sampling_factor)) +
                   "x"     + ROCJPEG_TOSTR(static_cast<int>(jpeg_stream_parameters_.picture_parameter_buffer.components[1].v_sampling_factor)) +
                   ", Cr=" + ROCJPEG_TOSTR(static_cast<int>(jpeg_stream_parameters_.picture_parameter_buffer.components[2].h_sampling_factor)) +
                   "x"     + ROCJPEG_TOSTR(static_cast<int>(jpeg_stream_parameters_.picture_parameter_buffer.components[2].v_sampling_factor));
        }
        msg += ")!";
        ErrorLog(g_rocjpeg_logger, msg);
        return false;
    }

    if (g_rocjpeg_logger.GetLogLevel() >= kRocJpegLogDebug) {
        // CSS_411 and CSS_UNKNOWN already returned false above, so only indices 0-3,5 are reachable.
        static const char* css_names[] = { "4:4:4", "4:4:0", "4:2:2", "4:2:0", "4:1:1", "4:0:0" };
        int css_idx = static_cast<int>(jpeg_stream_parameters_.chroma_subsampling);
        uint32_t sof_offset = static_cast<uint32_t>(sof_header - stream_start_) - 2;
        uint16_t frame_length = swap_bytes(sof_header);
        uint8_t precision = sof_header[2];
        uint16_t width = jpeg_stream_parameters_.picture_parameter_buffer.picture_width;
        uint16_t height = jpeg_stream_parameters_.picture_parameter_buffer.picture_height;
        uint8_t num_comp = jpeg_stream_parameters_.picture_parameter_buffer.num_components;
        std::ostringstream oss;
        oss << std::uppercase << std::hex << std::setfill('0');
        oss << "\n*** Marker: SOF0 (Baseline DCT) (xFFC0) ***"
            << "\n  OFFSET: 0x" << std::setw(8) << sof_offset
            << "\n  Frame header length = " << std::dec << frame_length
            << "\n  Precision = " << static_cast<int>(precision)
            << "\n  Number of Lines = " << height
            << "\n  Samples per Line = " << width
            << "\n  Image Size = " << width << " x " << height
            << "\n  Chroma subsampling: " << css_names[css_idx]
            << "\n  Raw Image Orientation = " << (width >= height ? "Landscape" : "Portrait")
            << "\n  Number of Img components = " << static_cast<int>(num_comp);
        static const char* comp_labels[] = { "Lum: Y", "Chrom: Cb", "Chrom: Cr" };
        uint8_t max_h = jpeg_stream_parameters_.picture_parameter_buffer.components[0].h_sampling_factor;
        uint8_t max_v = jpeg_stream_parameters_.picture_parameter_buffer.components[0].v_sampling_factor;
        for (int i = 0; i < num_comp; i++) {
            auto& c = jpeg_stream_parameters_.picture_parameter_buffer.components[i];
            uint8_t samp_fac_byte = (c.h_sampling_factor << 4) | c.v_sampling_factor;
            uint8_t subsamp_h = (c.h_sampling_factor > 0) ? max_h / c.h_sampling_factor : 0;
            uint8_t subsamp_v = (c.v_sampling_factor > 0) ? max_v / c.v_sampling_factor : 0;
            const char* comp_label = (i < 3) ? comp_labels[i] : "Unknown";
            oss << "\n    Component[" << std::dec << (i + 1) << "]: "
                << "ID=0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c.component_id)
                << ", Samp Fac=0x" << std::setw(2) << static_cast<int>(samp_fac_byte)
                << " (Subsamp " << std::dec << static_cast<int>(subsamp_h)
                << " x " << static_cast<int>(subsamp_v) << ")"
                << ", Quant Tbl Sel=0x" << std::hex << std::setw(2) << static_cast<int>(c.quantiser_table_selector)
                << " (" << comp_label << ")";
        }
        DebugLog(g_rocjpeg_logger, oss.str());
    }

    return true;
}

/**
 * @brief Parses the DQT (Define Quantization Table) segment of a JPEG stream.
 *
 * This function reads the quantization tables from the JPEG stream and stores them in the
 * `jpeg_stream_parameters_.quantization_matrix_buffer` data structure.
 *
 * @return `true` if the DQT segment is successfully parsed, `false` otherwise.
 */
bool RocJpegStreamParser::ParseDQT() {
    int32_t quantization_table_index = 0;
    const uint8_t *dqt_block_end;

    if (stream_ == nullptr) {
        return false;
    }

    uint16_t table_length = swap_bytes(stream_);
    if (table_length < 2 || stream_ + table_length > stream_end_) {
        ErrorLog(g_rocjpeg_logger, "Invalid DQT marker length!");
        return false;
    }
    dqt_block_end = stream_ + table_length;
    uint32_t dqt_offset = static_cast<uint32_t>(stream_ - stream_start_) - 2;
    stream_ += 2;

    while (stream_ < dqt_block_end) {
        if (stream_ + 1 + 64 > dqt_block_end) {
            ErrorLog(g_rocjpeg_logger, "Truncated JPEG: DQT segment too small to contain a complete quantization table!");
            return false;
        }
        quantization_table_index = *stream_++;
        if (quantization_table_index >> 4) {
            ErrorLog(g_rocjpeg_logger,"16 bits quantization table is not supported!");
            return false;
        }
        if (quantization_table_index >= 4) {
            ErrorLog(g_rocjpeg_logger,"invalid number of quantization table!");
            return false;
        }

        std::memcpy(jpeg_stream_parameters_.quantization_matrix_buffer.quantiser_table[quantization_table_index & 0x0F], stream_, 64);
        jpeg_stream_parameters_.quantization_matrix_buffer.load_quantiser_table[quantization_table_index & 0x0F] = 1;

        if (g_rocjpeg_logger.GetLogLevel() >= kRocJpegLogDebug) {
            int tbl_id = quantization_table_index & 0x0F;
            const uint8_t *tbl = jpeg_stream_parameters_.quantization_matrix_buffer.quantiser_table[tbl_id];
            // Standard IJG reference quantization tables (quality=50 baseline).
            // scaling = mean(100 * actual[i] / ref[i])
            static const uint8_t k_std_luma[64] = {
                16, 11, 10, 16, 24, 40, 51, 61,
                12, 12, 14, 19, 26, 58, 60, 55,
                14, 13, 16, 24, 40, 57, 69, 56,
                14, 17, 22, 29, 51, 87, 80, 62,
                18, 22, 37, 56, 68,109,103, 77,
                24, 35, 55, 64, 81,104,113, 92,
                49, 64, 78, 87,103,121,120,101,
                72, 92, 95, 98,112,100,103, 99
            };
            static const uint8_t k_std_chroma[64] = {
                17, 18, 24, 47, 99, 99, 99, 99,
                18, 21, 26, 66, 99, 99, 99, 99,
                24, 26, 56, 99, 99, 99, 99, 99,
                47, 66, 99, 99, 99, 99, 99, 99,
                99, 99, 99, 99, 99, 99, 99, 99,
                99, 99, 99, 99, 99, 99, 99, 99,
                99, 99, 99, 99, 99, 99, 99, 99,
                99, 99, 99, 99, 99, 99, 99, 99
            };
            const uint8_t *ref = (tbl_id == 0) ? k_std_luma : k_std_chroma;
            double sum = 0.0;
            for (int k = 0; k < 64; k++) sum += 100.0 * tbl[k] / ref[k];
            double scaling = sum / 64.0;
            double var_sum = 0.0;
            for (int k = 0; k < 64; k++) {
                double d = 100.0 * tbl[k] / ref[k] - scaling;
                var_sum += d * d;
            }
            double variance = var_sum / 64.0;
            // Quality: all-1s table is the minimum possible (Q=100); otherwise use IJG inverse formula.
            bool all_min = true;
            for (int k = 0; k < 64; k++) { if (tbl[k] != 1) { all_min = false; break; } }
            double quality;
            if (all_min) quality = 100.0;
            else if (scaling < 100.0) quality = std::max(0.0, (200.0 - scaling) / 2.0);
            else quality = std::max(0.0, 5000.0 / scaling);

            std::string dest_label = (tbl_id == 0) ? "Luminance" : "Chrominance";
            std::ostringstream oss;
            oss << std::uppercase << std::hex << std::setfill('0');
            oss << "\n*** Marker: DQT (xFFDB) ***"
                << "\n  Define a Quantization Table."
                << "\n  OFFSET: 0x" << std::setw(8) << dqt_offset
                << "\n  Table length = " << std::dec << table_length
                << "\n  ----"
                << "\n  Precision=8 bits"
                << "\n  Destination ID=" << tbl_id << " (" << dest_label << ")";
            for (int row = 0; row < 8; row++) {
                oss << "\n    DQT, Row #" << row << ":";
                for (int col = 0; col < 8; col++) {
                    oss << " " << std::setw(3) << std::setfill(' ') << static_cast<int>(tbl[row * 8 + col]);
                }
            }
            oss << std::fixed << std::setprecision(2)
                << "\n    Approx quality factor = " << quality
                << " (scaling=" << scaling << " variance=" << variance << ")";
            DebugLog(g_rocjpeg_logger, oss.str());
        }

        stream_ += 64;
    }

    return true;
}

/**
 * @brief Parses the Define Huffman Table (DHT) segment in the JPEG stream.
 *
 * This function reads and processes the DHT segment in the JPEG stream. It extracts the Huffman table
 * information and stores it in the `jpeg_stream_parameters_.huffman_table_buffer` data structure.
 *
 * @return `true` if the DHT segment is successfully parsed, `false` otherwise.
 */
bool RocJpegStreamParser::ParseDHT() {
    uint32_t count, i;
    int32_t length, index;
    uint8_t ac_huffman_table, huffman_table_id;

    if (stream_ == nullptr) {
        return false;
    }

    const uint8_t *dht_header = stream_;
    uint16_t segment_length = swap_bytes(stream_);
    if (segment_length < 2 || stream_ + segment_length > stream_end_) {
        ErrorLog(g_rocjpeg_logger, "Invalid DHT marker length!");
        return false;
    }
    length = static_cast<int32_t>(segment_length) - 2;
    stream_ += 2;

    while (length > 0) {
        index = *stream_++;

        ac_huffman_table = index & 0xF0;
        huffman_table_id = index & 0x0F;

        if (huffman_table_id >= HUFFMAN_TABLES) {
            ErrorLog(g_rocjpeg_logger,"invalid number of Huffman table!");
            return false;
        }

        if (ac_huffman_table) {
            std::memcpy(jpeg_stream_parameters_.huffman_table_buffer.huffman_table[huffman_table_id].num_ac_codes, stream_, 16);
        } else {
            std::memcpy(jpeg_stream_parameters_.huffman_table_buffer.huffman_table[huffman_table_id].num_dc_codes, stream_, 16);
        }

        count = 0;
        for (i = 0; i < 16; i++) {
            count += *stream_++;
        }

        if (ac_huffman_table) {
            if (count > AC_HUFFMAN_TABLE_VALUES_SIZE) {
                ErrorLog(g_rocjpeg_logger,"invalid AC Huffman table!");
                return false;
            }
            std::memcpy(jpeg_stream_parameters_.huffman_table_buffer.huffman_table[huffman_table_id].ac_values, stream_, count);
            jpeg_stream_parameters_.huffman_table_buffer.load_huffman_table[huffman_table_id] = 1;
        } else {
            if (count > DC_HUFFMAN_TABLE_VALUES_SIZE) {
                ErrorLog(g_rocjpeg_logger, "Invalid DC Huffman table!");
                return false;
            }
            std::memcpy(jpeg_stream_parameters_.huffman_table_buffer.huffman_table[huffman_table_id].dc_values, stream_, count);
            jpeg_stream_parameters_.huffman_table_buffer.load_huffman_table[huffman_table_id] = 1;
        }

        if (g_rocjpeg_logger.GetLogLevel() >= kRocJpegLogDebug) {
            uint32_t dht_offset = static_cast<uint32_t>(dht_header - stream_start_) - 2;
            uint16_t dht_length = swap_bytes(dht_header);
            // num_codes is the 16-byte array we copied just before the values
            const uint8_t *num_codes = ac_huffman_table
                ? jpeg_stream_parameters_.huffman_table_buffer.huffman_table[huffman_table_id].num_ac_codes
                : jpeg_stream_parameters_.huffman_table_buffer.huffman_table[huffman_table_id].num_dc_codes;
            const uint8_t *values = stream_;  // stream_ already advanced past 16 num_codes bytes, now at value bytes

            std::ostringstream oss;
            oss << std::uppercase << std::hex << std::setfill('0');
            oss << "\n*** Marker: DHT (Define Huffman Table) (xFFC4) ***"
                << "\n  OFFSET: 0x" << std::setw(8) << dht_offset
                << "\n  Huffman table length = " << std::dec << dht_length
                << "\n  ----"
                << "\n  Destination ID = " << static_cast<int>(huffman_table_id)
                << "\n  Class = " << (ac_huffman_table ? "1 (AC Table)" : "0 (DC / Lossless Table)");

            uint32_t val_idx = 0;
            uint32_t total_codes = 0;
            for (int bit = 1; bit <= 16; bit++) {
                uint8_t n = num_codes[bit - 1];
                total_codes += n;
                oss << "\n    Codes of length " << std::setw(2) << std::setfill('0') << bit
                    << " bits (" << std::setw(3) << std::setfill('0') << std::dec << static_cast<int>(n) << " total): ";
                // Print up to 16 values per row, wrap with indentation
                for (uint8_t v = 0; v < n; v++) {
                    if (v > 0 && v % 16 == 0) oss << "\n                                         ";
                    oss << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
                        << static_cast<int>(values[val_idx]) << " ";
                    val_idx++;
                }
            }
            oss << "\n    Total number of codes: " << std::setw(3) << std::setfill('0') << std::dec << total_codes;
            DebugLog(g_rocjpeg_logger, oss.str());
        }

        length -= 1;
        length -= 16;
        length -= count;
        stream_ += count;
    }

    return true;
}

/**
 * @brief Parses the Start of Scan (SOS) marker in the JPEG stream.
 *
 * This function reads and processes the SOS marker in the JPEG stream.
 * It extracts the component IDs and Huffman table selectors for each component,
 * and performs various checks for validity.
 *
 * @return true if the SOS marker is successfully parsed, false otherwise.
 */
bool RocJpegStreamParser::ParseSOS() {
    uint32_t component_id, table;

    if (stream_ == nullptr) {
        return false;
    }

    const uint8_t *sos_header = stream_;
    uint32_t num_components = stream_[2];

    if (num_components > NUM_COMPONENTS - 1) {
        ErrorLog(g_rocjpeg_logger, "SOS has " + ROCJPEG_TOSTR(num_components) +
            " components, exceeds maximum supported (" + ROCJPEG_TOSTR(NUM_COMPONENTS - 1) + ")!");
        return false;
    }
    if (num_components != jpeg_stream_parameters_.picture_parameter_buffer.num_components) {
        ErrorLog(g_rocjpeg_logger, "SOS component count (" + ROCJPEG_TOSTR(num_components) +
            ") does not match SOF component count (" +
            ROCJPEG_TOSTR(static_cast<int>(jpeg_stream_parameters_.picture_parameter_buffer.num_components)) +
            ") - this may be a multi-scan progressive JPEG, which is not supported!");
        return false;
    }
    jpeg_stream_parameters_.slice_parameter_buffer.num_components = num_components;

    stream_ += 3;
    for (uint32_t i = 0; i < num_components; i++) {
        component_id = *stream_++;
        table = *stream_++;
        jpeg_stream_parameters_.slice_parameter_buffer.components[i].component_selector = component_id;
        jpeg_stream_parameters_.slice_parameter_buffer.components[i].dc_table_selector = ((table >> 4) & 0x0F);
        jpeg_stream_parameters_.slice_parameter_buffer.components[i].ac_table_selector = (table & 0x0F);

        if ((table & 0xF) >= 4) {
            ErrorLog(g_rocjpeg_logger,"invalid number of AC Huffman table!");
            return false;
        }
        if ((table >> 4) >= 4) {
            ErrorLog(g_rocjpeg_logger,"invalid number of DC Huffman table!");
            return false;
        }
        if (component_id != jpeg_stream_parameters_.picture_parameter_buffer.components[i].component_id) {
            ErrorLog(g_rocjpeg_logger,"component id mismatch between SOS and SOF marker!");
            return false;
        }
    }

    if (g_rocjpeg_logger.GetLogLevel() >= kRocJpegLogDebug) {
        uint32_t sos_offset = static_cast<uint32_t>(sos_header - stream_start_) - 2;
        uint16_t sos_length = swap_bytes(sos_header);
        // stream_ now points at Ss, Se, Ah/Al bytes
        uint8_t spectral_start = stream_[0];
        uint8_t spectral_end   = stream_[1];
        uint8_t succ_approx    = stream_[2];
        std::ostringstream oss;
        oss << std::uppercase << std::hex << std::setfill('0');
        oss << "\n*** Marker: SOS (Start of Scan) (xFFDA) ***"
            << "\n  OFFSET: 0x" << std::setw(8) << sos_offset
            << "\n  Scan header length = " << std::dec << sos_length
            << "\n  Number of img components = " << num_components;
        for (uint32_t i = 0; i < num_components; i++) {
            auto& c = jpeg_stream_parameters_.slice_parameter_buffer.components[i];
            oss << "\n    Component[" << std::dec << (i + 1) << "]: "
                << "selector=0x" << std::hex << std::setw(2) << static_cast<int>(c.component_selector)
                << ", table=" << std::dec << static_cast<int>(c.dc_table_selector) << "(DC)"
                << "," << static_cast<int>(c.ac_table_selector) << "(AC)";
        }
        oss << "\n  Spectral selection = " << std::dec << static_cast<int>(spectral_start)
            << " .. " << static_cast<int>(spectral_end)
            << "\n  Successive approximation = 0x" << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(succ_approx);
        DebugLog(g_rocjpeg_logger, oss.str());
    }

    stream_ += 3;

    return true;
}


/**
 * @brief Parses the Define Restart Interval (DRI) marker in the JPEG stream.
 *
 * This function reads the length field of the DRI marker and checks if it is valid.
 * If the length is valid, it updates the restart interval value in the slice parameter buffer.
 *
 * @return true if the DRI marker is successfully parsed, false otherwise.
 */
bool RocJpegStreamParser::ParseDRI() {
    uint32_t length;

    if (stream_ == nullptr) {
        return false;
    }

    length = swap_bytes(stream_);
    if (length != 4) {
        ErrorLog(g_rocjpeg_logger,"invalid size for DRI marker");
        return false;
    }

    jpeg_stream_parameters_.slice_parameter_buffer.restart_interval = swap_bytes(stream_ + 2);

    return true;
}

/**
 * @brief Parses the End of Image (EOI) marker in the JPEG stream.
 *
 * This function searches for the EOI marker in the JPEG stream and updates the slice data buffer
 * and slice data size in the jpeg_stream_parameters_ structure.
 *
 * @return true if the EOI marker is found and the slice data buffer is updated successfully, false otherwise.
 */
bool RocJpegStreamParser::ParseEOI() {

    if (stream_ == nullptr) {
        return false;
    }

    const uint8_t *stream_temp = stream_;
    while (stream_temp <= stream_end_ && !(*stream_temp == 0xFF  && *(stream_temp + 1) == EOI)) {
        stream_temp++;
        continue;
    }

    jpeg_stream_parameters_.slice_parameter_buffer.slice_data_size = stream_temp - stream_;
    jpeg_stream_parameters_.slice_data_buffer = stream_;

    if (g_rocjpeg_logger.GetLogLevel() >= kRocJpegLogDebug) {
        uint32_t eoi_offset = static_cast<uint32_t>(stream_temp - stream_start_);
        std::ostringstream oss;
        oss << "\n*** Marker: EOI (End of Image) (xFFD9) ***"
            << "\n  OFFSET: 0x" << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << eoi_offset;
        DebugLog(g_rocjpeg_logger, oss.str());
    }

    return true;
}

/**
 * @brief Determines the chroma subsampling format based on the given sampling factors.
 *
 * This function takes the horizontal and vertical sampling factors for each color component and determines
 * the chroma subsampling format. It returns the corresponding `ChromaSubsampling` enum value.
 *
 * @param c1_h_sampling_factor The horizontal sampling factor for color component 1.
 * @param c2_h_sampling_factor The horizontal sampling factor for color component 2.
 * @param c3_h_sampling_factor The horizontal sampling factor for color component 3.
 * @param c1_v_sampling_factor The vertical sampling factor for color component 1.
 * @param c2_v_sampling_factor The vertical sampling factor for color component 2.
 * @param c3_v_sampling_factor The vertical sampling factor for color component 3.
 * @return The chroma subsampling format determined based on the given sampling factors.
 */
ChromaSubsampling RocJpegStreamParser::GetChromaSubsampling(uint8_t c1_h_sampling_factor, uint8_t c2_h_sampling_factor, uint8_t c3_h_sampling_factor,
                                                   uint8_t c1_v_sampling_factor, uint8_t c2_v_sampling_factor, uint8_t c3_v_sampling_factor) {

    ChromaSubsampling subsampling;

    if ((c1_h_sampling_factor == 1 && c2_h_sampling_factor == 1 && c3_h_sampling_factor == 1 &&
         c1_v_sampling_factor == 1 && c2_v_sampling_factor == 1 && c3_v_sampling_factor == 1) ||
        (c1_h_sampling_factor == 2 && c2_h_sampling_factor == 2 && c3_h_sampling_factor == 2 &&
         c1_v_sampling_factor == 2 && c2_v_sampling_factor == 2 && c3_v_sampling_factor == 2) ||
        (c1_h_sampling_factor == 4 && c2_h_sampling_factor == 4 && c3_h_sampling_factor == 4 &&
         c1_v_sampling_factor == 4 && c2_v_sampling_factor == 4 && c3_v_sampling_factor == 4)) {
            subsampling = CSS_444;
    } else if (c1_h_sampling_factor == 1 && c2_h_sampling_factor == 1 && c3_h_sampling_factor == 1 &&
               c1_v_sampling_factor == 2 && c2_v_sampling_factor == 1 && c3_v_sampling_factor == 1) {
                    subsampling = CSS_440;
    } else if ((c1_h_sampling_factor == 2 && c2_h_sampling_factor == 1 && c3_h_sampling_factor == 1 &&
                c1_v_sampling_factor == 1 && c2_v_sampling_factor == 1 && c3_v_sampling_factor == 1) ||
               (c1_h_sampling_factor == 2 && c2_h_sampling_factor == 1 && c3_h_sampling_factor == 1 &&
                c1_v_sampling_factor == 2 && c2_v_sampling_factor == 2 && c3_v_sampling_factor == 2) ||
               (c1_h_sampling_factor == 2 && c2_h_sampling_factor == 2 && c3_h_sampling_factor == 2 &&
                c1_v_sampling_factor == 2 && c2_v_sampling_factor == 1 && c3_v_sampling_factor == 1)) {
                    subsampling = CSS_422;
    } else if (c1_h_sampling_factor == 2 && c2_h_sampling_factor == 1 && c3_h_sampling_factor == 1 &&
               c1_v_sampling_factor == 2 && c2_v_sampling_factor == 1 && c3_v_sampling_factor == 1) {
                    subsampling = CSS_420;
    } else if (c1_h_sampling_factor == 4 && c2_h_sampling_factor == 1 && c3_h_sampling_factor == 1 &&
               c1_v_sampling_factor == 1 && c2_v_sampling_factor == 1 && c3_v_sampling_factor == 1) {
                    subsampling = CSS_411;
    } else if ((c1_h_sampling_factor == 1 && c2_h_sampling_factor == 0 && c3_h_sampling_factor == 0 &&
                c1_v_sampling_factor == 1 && c2_v_sampling_factor == 0 && c3_v_sampling_factor == 0) ||
               (c1_h_sampling_factor == 4 && c2_h_sampling_factor == 0 && c3_h_sampling_factor == 0 &&
                c1_v_sampling_factor == 4 && c2_v_sampling_factor == 0 && c3_v_sampling_factor == 0)) {
                    subsampling = CSS_400;
    } else {
        subsampling = CSS_UNKNOWN;
    }

    return subsampling;
}
