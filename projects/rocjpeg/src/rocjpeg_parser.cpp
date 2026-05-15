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

RocJpegStreamParser::RocJpegStreamParser() : stream_{nullptr}, stream_end_{nullptr}, stream_length_{0},
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
    std::lock_guard<std::mutex> lock(mutex_);
    if (jpeg_stream == nullptr) {
        ERR("invalid argument!");
        return false;
    }

    stream_ = jpeg_stream;
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
        ERR("Invalid JPEG!");
        return false;
    }

    soi_marker_found = ParseSOI();
    if (!soi_marker_found) {
        ERR("failed to find the SOI marker!");
    }

    while (!sos_marker_found  && stream_ <= stream_end_) {
        while ((*stream_ == 0xFF))
            stream_++;
        marker = *stream_++;
        chuck_len = swap_bytes(stream_);
        next_chunck = stream_ + chuck_len;

        switch (marker) {
            case SOF:
                if (!ParseSOF())
                    return false;
                break;
            case DHT:
                if (!ParseDHT())
                    return false;
                dht_marker_found = true;
                break;
            case DQT:
                if (!ParseDQT())
                    return false;
                dqt_marker_found = true;
                break;
            case DRI:
                if (!ParseDRI())
                    return false;
                break;
            case SOS:
                if (!ParseSOS())
                    return false;
                sos_marker_found = true;
                break;
            default:
                break;
        }
        stream_ = next_chunck;
    }

    if (!dht_marker_found) {
        ERR("didn't find any Huffman table!");
        return false;
    }
    if (!dqt_marker_found) {
        ERR("didn't find any quantization table!");
        return false;
    }

    if (!ParseEOI())
        return false;

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

    jpeg_stream_parameters_.picture_parameter_buffer.picture_height = swap_bytes(stream_ + 3);
    jpeg_stream_parameters_.picture_parameter_buffer.picture_width = swap_bytes(stream_ + 5);
    jpeg_stream_parameters_.picture_parameter_buffer.num_components = stream_[7];

    if (jpeg_stream_parameters_.picture_parameter_buffer.num_components > NUM_COMPONENTS - 1) {
        ERR("invalid number of JPEG components!");
        return false;
    }

    stream_ += 8;

    for (int32_t i = 0; i < jpeg_stream_parameters_.picture_parameter_buffer.num_components; i++) {
        component_id = *stream_++;
        sampling_factor = *stream_++;
        quantiser_table_selector = *stream_++;

        jpeg_stream_parameters_.picture_parameter_buffer.components[i].component_id = component_id;
        if (quantiser_table_selector >= NUM_COMPONENTS) {
            ERR("invalid number of the quantization table!");
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

    dqt_block_end = stream_ + swap_bytes(stream_);
    stream_ += 2;

    while (stream_ < dqt_block_end) {
        quantization_table_index = *stream_++;
        if (quantization_table_index >> 4) {
            ERR("16 bits quantization table is not supported!");
            return false;
        }
        if (quantization_table_index >= 4) {
            ERR("invalid number of quantization table!");
            return false;
        }

        std::memcpy(jpeg_stream_parameters_.quantization_matrix_buffer.quantiser_table[quantization_table_index & 0x0F], stream_, 64);
        jpeg_stream_parameters_.quantization_matrix_buffer.load_quantiser_table[quantization_table_index & 0x0F] = 1;

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

    length = swap_bytes(stream_) - 2;
    stream_ += 2;

    while (length > 0) {
        index = *stream_++;

        ac_huffman_table = index & 0xF0;
        huffman_table_id = index & 0x0F;

        if (huffman_table_id >= HUFFMAN_TABLES) {
            ERR("invlaid number of Huffman table!");
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
                ERR("invalid AC Huffman table!");
                return false;
            }
            std::memcpy(jpeg_stream_parameters_.huffman_table_buffer.huffman_table[huffman_table_id].ac_values, stream_, count);
            jpeg_stream_parameters_.huffman_table_buffer.load_huffman_table[huffman_table_id] = 1;
        } else {
            if (count > DC_HUFFMAN_TABLE_VALUES_SIZE) {
                ERR("invlaid DC Huffman table!")
                return false;
            }
            std::memcpy(jpeg_stream_parameters_.huffman_table_buffer.huffman_table[huffman_table_id].dc_values, stream_, count);
            jpeg_stream_parameters_.huffman_table_buffer.load_huffman_table[huffman_table_id] = 1;
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

    uint32_t num_components = stream_[2];

    if (num_components > NUM_COMPONENTS - 1 || num_components != jpeg_stream_parameters_.picture_parameter_buffer.num_components) {
        ERR("invalid number of component!")
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
            ERR("invalid number of AC Huffman table!");
            return false;
        }
        if ((table >> 4) >= 4) {
            ERR("invalid number of DC Huffman table!");
            return false;
        }
        if (component_id != jpeg_stream_parameters_.picture_parameter_buffer.components[i].component_id) {
            ERR("component id mismatch between SOS and SOF marker!");
            return false;
        }
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
        ERR("invalid size for DRI marker");
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
