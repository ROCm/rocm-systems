
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

int RocJpegApiNegativeTests::RunTests() {
    if (TestInvalidStreamCreate() || TestInvalidStreamParse () || TestInvalidStreamDestroy() || TestInvalidCreate() || TestInvalidDestroy() ||
        TestInvalidGetImageInfo() || TestInvalidDecode() || TestInvalidDecodeBatched() || TestInvalidGetErrorName()) {
        std::cerr << "One or more negative tests failed." << std::endl;
        return EXIT_FAILURE;
    } else {
        return EXIT_SUCCESS;
    }
}
