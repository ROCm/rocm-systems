/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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


#ifndef ROC_JPEG_PARSER_H_
#define ROC_JPEG_PARSER_H_

#include <stdint.h>
#include <iostream>
#include <cstring>
#include <mutex>
#include "rocjpeg_commons.h"

#pragma once

#define NUM_COMPONENTS 4
#define HUFFMAN_TABLES 2
#define AC_HUFFMAN_TABLE_VALUES_SIZE 162
#define DC_HUFFMAN_TABLE_VALUES_SIZE 12
#define swap_bytes(x) (((x)[0] << 8) | (x)[1])

/**
 * @brief Enumeration representing the common JPEG markers.
 *
 * The `JpegMarkers` enum defines the common JPEG markers used in a JPEG stream.
 */
enum JpegMarkers {
    SOI = 0xD8, /**< Start Of Image */
    SOF = 0xC0, /**< Start Of Frame for a baseline DCT-based JPEG. */
    DHT = 0xC4, /**< Define Huffman Table */
    DQT = 0xDB, /**< Define Quantization Table */
    DRI = 0xDD, /**< Define Restart Interval */
    SOS = 0xDA, /**< Start of Scan */
    EOI = 0xD9, /**< End Of Image */
};

/**
 * @brief Structure representing the picture parameter buffer.
 *
 * This structure contains information about the picture width, picture height,
 * color components, color space, rotation, and reserved fields.
 */
typedef struct PictureParameterBufferType {
    uint16_t picture_width; /**< The width of the picture. */
    uint16_t picture_height; /**< The height of the picture. */
    struct {
        uint8_t component_id; /**< The ID of the color component. */
        uint8_t h_sampling_factor; /**< The horizontal sampling factor. */
        uint8_t v_sampling_factor; /**< The vertical sampling factor. */
        uint8_t quantiser_table_selector; /**< The quantiser table selector. */
    } components[255]; /**< Array of color components. */
    uint8_t num_components; /**< The number of color components. */
    uint8_t color_space; /**< The color space of the picture. */
    uint32_t rotation; /**< The rotation of the picture. */
    uint32_t reserved[7]; /**< Reserved fields. */
} PictureParameterBuffer;

/**
 * @brief Structure representing the quantization matrix buffer.
 *
 * This structure holds the quantization tables used in the JPEG decoding process.
 * It consists of an array to indicate whether a quantization table is loaded or not,
 * a 2D array to store the quantization tables, and a reserved field.
 */
typedef struct QuantizationMatrixBufferType {
    uint8_t load_quantiser_table[4];    /**< Array indicating whether a quantization table is loaded or not. */
    uint8_t quantiser_table[4][64];     /**< 2D array to store the quantization tables. */
    uint32_t reserved[4];               /**< Reserved field. */
} QuantizationMatrixBuffer;

/**
 * @brief Struct representing a buffer for Huffman tables.
 *
 * This struct is used to store Huffman tables for JPEG decoding.
 * It contains two sets of Huffman tables, each consisting of
 * arrays for the number of DC codes, DC values, number of AC codes,
 * AC values, and padding.
 *
 * The `load_huffman_table` array is used to indicate whether a
 * particular Huffman table should be loaded or not.
 *
 * The `reserved` array is used for future expansion and is currently
 * unused.
 */
typedef struct HuffmanTableBufferType {
    uint8_t load_huffman_table[2]; /**< Array indicating which Huffman tables to load. */
    struct {
        uint8_t num_dc_codes[16]; /**< Array of the number of DC codes for each bit length. */
        uint8_t dc_values[12]; /**< Array of the DC values. */
        uint8_t num_ac_codes[16]; /**< Array of the number of AC codes for each bit length. */
        uint8_t ac_values[162]; /**< Array of the AC values. */
        uint8_t pad[2]; /**< Padding to align the structure. */
    } huffman_table[2]; /**< Array of two sets of Huffman tables. */
    uint32_t reserved[4]; /**< Reserved field for future use. */
} HuffmanTableBuffer;

/**
 * @brief Structure representing the slice parameter buffer.
 *
 * This structure contains information about the slice data, such as its size, offset, and flag.
 * It also includes the horizontal and vertical position of the slice, as well as the component,
 * DC table, and AC table selectors for each component. The number of components, restart interval,
 * number of MCUs, and reserved fields are also included.
 */
typedef struct SliceParameterBufferType {
    uint32_t slice_data_size;                /**< Size of the slice data. */
    uint32_t slice_data_offset;              /**< Offset of the slice data. */
    uint32_t slice_data_flag;                /**< Flag indicating the slice data. */
    uint32_t slice_horizontal_position;      /**< Horizontal position of the slice. */
    uint32_t slice_vertical_position;        /**< Vertical position of the slice. */
    struct {
        uint8_t component_selector;          /**< Component selector. */
        uint8_t dc_table_selector;           /**< DC table selector. */
        uint8_t ac_table_selector;           /**< AC table selector. */
    } components[4];                         /**< Array of component selectors. */
    uint8_t num_components;                  /**< Number of components. */
    uint16_t restart_interval;               /**< Restart interval. */
    uint32_t num_mcus;                       /**< Number of MCUs. */
    uint32_t reserved[4];                    /**< Reserved fields. */
} SliceParameterBuffer;

/**
 * @brief Enumeration representing the chroma subsampling formats.
 *
 * The `ChromaSubsampling` enum defines the possible chroma subsampling formats in a JPEG stream.
 * Each value corresponds to a specific chroma subsampling format, such as 4:4:4, 4:2:0, etc.
 * The `CSS_UNKNOWN` value is used to indicate an unknown or unsupported format.
 */
typedef enum {
    CSS_444 = 0,
    CSS_440 = 1,
    CSS_422 = 2,
    CSS_420 = 3,
    CSS_411 = 4,
    CSS_400 = 5,
    CSS_UNKNOWN = -1
} ChromaSubsampling;

/**
 * @brief Structure representing the parameters for a JPEG stream.
 *
 * This structure contains various buffers and data required for processing a JPEG stream.
 * It includes the picture parameter buffer, quantization matrix buffer, Huffman table buffer,
 * slice parameter buffer, chroma subsampling information, and the slice data buffer.
 */
typedef struct JpegParameterBuffersType {
    PictureParameterBuffer picture_parameter_buffer;
    QuantizationMatrixBuffer quantization_matrix_buffer;
    HuffmanTableBuffer huffman_table_buffer;
    SliceParameterBuffer slice_parameter_buffer;
    ChromaSubsampling chroma_subsampling;
    const uint8_t* slice_data_buffer;
} JpegStreamParameters;

/**
 * @class RocJpegStreamParser
 * @brief A class for parsing JPEG streams and extracting stream parameters.
 *
 * The RocJpegStreamParser class provides functionality to parse a JPEG stream and extract various parameters
 * such as Start of Image (SOI), Start of Frame (SOF), Quantization Tables (DQT), Start of Scan (SOS),
 * Huffman Tables (DHT), Define Restart Interval (DRI), and End of Image (EOI). It also provides a method to
 * retrieve the parsed JPEG stream parameters.
 */
class RocJpegStreamParser {
    public:
        /**
         * @brief Default constructor for RocJpegStreamParser.
         */
        RocJpegStreamParser();

        /**
         * @brief Destructor for RocJpegStreamParser.
         */
        ~RocJpegStreamParser();

        /**
         * @brief Parses a JPEG stream and extracts stream parameters.
         * @param jpeg_stream The pointer to the JPEG stream.
         * @param jpeg_stream_size The size of the JPEG stream.
         * @return True if the parsing is successful, false otherwise.
         */
        bool ParseJpegStream(const uint8_t* jpeg_stream, uint32_t jpeg_stream_size);

        /**
         * @brief Retrieves the JPEG stream parameters.
         * @return A pointer to the JpegStreamParameters object.
         */
        const JpegStreamParameters* GetJpegStreamParameters() const { return &jpeg_stream_parameters_; };

    private:
        /**
         * @brief Parses the Start of Image (SOI) marker.
         * @return True if the SOI marker is successfully parsed, false otherwise.
         */
        bool ParseSOI();

        /**
         * @brief Parses the Start of Frame (SOF) marker.
         * @return True if the SOF marker is successfully parsed, false otherwise.
         */
        bool ParseSOF();

        /**
         * @brief Parses the Define Quantization Table (DQT) marker.
         * @return True if the DQT marker is successfully parsed, false otherwise.
         */
        bool ParseDQT();

        /**
         * @brief Parses the Start of Scan (SOS) marker.
         * @return True if the SOS marker is successfully parsed, false otherwise.
         */
        bool ParseSOS();

        /**
         * @brief Parses the Define Huffman Table (DHT) marker.
         * @return True if the DHT marker is successfully parsed, false otherwise.
         */
        bool ParseDHT();

        /**
         * @brief Parses the Define Restart Interval (DRI) marker.
         * @return True if the DRI marker is successfully parsed, false otherwise.
         */
        bool ParseDRI();

        /**
         * @brief Parses the End of Image (EOI) marker.
         * @return True if the EOI marker is successfully parsed, false otherwise.
         */
        bool ParseEOI();

        /**
         * @brief Retrieves the chroma subsampling information.
         * @param c1_h_sampling_factor The horizontal sampling factor for component 1.
         * @param c2_h_sampling_factor The horizontal sampling factor for component 2.
         * @param c3_h_sampling_factor The horizontal sampling factor for component 3.
         * @param c1_v_sampling_factor The vertical sampling factor for component 1.
         * @param c2_v_sampling_factor The vertical sampling factor for component 2.
         * @param c3_v_sampling_factor The vertical sampling factor for component 3.
         * @return The chroma subsampling information.
         */
        ChromaSubsampling GetChromaSubsampling(uint8_t c1_h_sampling_factor, uint8_t c2_h_sampling_factor, uint8_t c3_h_sampling_factor,
                                               uint8_t c1_v_sampling_factor, uint8_t c2_v_sampling_factor, uint8_t c3_v_sampling_factor);

        const uint8_t *stream_; ///< Pointer to the JPEG stream.
        const uint8_t *stream_end_; ///< Pointer to the end of the JPEG stream.
        uint32_t stream_length_; ///< Length of the JPEG stream.
        JpegStreamParameters jpeg_stream_parameters_; ///< JPEG stream parameters.
        std::mutex mutex_; ///< Mutex for thread safety.
};

#endif  // ROC_JPEG_PARSER_H_