
/*
Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "rocdecode_api_negative_tests.h"

RocDecodeApiNegativeTests:: RocDecodeApiNegativeTests() : decoder_create_info_{} {};

RocDecodeApiNegativeTests::~RocDecodeApiNegativeTests() {
    rocDecDestroyDecoder(decoder_handle_);
    rocDecDestroyVideoParser(parser_handle_);
}

int RocDecodeApiNegativeTests::TestInvalidCreateDecoder() {
    std::cout << "info: Executing negative test cases for the rocDecCreateDecoder API" << std::endl;
    // Scenario 1: Pass nullptr for decoder_handle and decoder_create_info
    rocDecStatus rocdecode_status = rocDecCreateDecoder(nullptr, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 2: Pass an empty decoder_create_info_
    rocdecode_status = rocDecCreateDecoder(&decoder_handle_, &decoder_create_info_);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    rocdecode_status = rocDecDestroyDecoder(decoder_handle_);
    if (rocdecode_status != ROCDEC_SUCCESS) {
        std::cerr << "Expected ROCDEC_SUCCESS but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 3: Pass invalid device_id in decoder_create_info
    decoder_create_info_.device_id = 255; // Assuming 255 is an invalid device ID
    decoder_create_info_.num_decode_surfaces = 1;
    rocdecode_status = rocDecCreateDecoder(&decoder_handle_, &decoder_create_info_);
    if (rocdecode_status != ROCDEC_NOT_SUPPORTED) {
        std::cerr << "Expected ROCDEC_NOT_SUPPORTED but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    rocdecode_status = rocDecDestroyDecoder(decoder_handle_);
    if (rocdecode_status != ROCDEC_SUCCESS) {
        std::cerr << "Expected ROCDEC_SUCCESS but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 4: Pass zero width and height in decoder_create_info
    decoder_create_info_.device_id = 0; // Reset to valid device ID
    decoder_create_info_.width = 0;
    decoder_create_info_.height = 0;
    rocdecode_status = rocDecCreateDecoder(&decoder_handle_, &decoder_create_info_);
    if (rocdecode_status != ROCDEC_NOT_SUPPORTED) {
        std::cerr << "Expected ROCDEC_NOT_SUPPORTED but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    rocdecode_status = rocDecDestroyDecoder(decoder_handle_);
    if (rocdecode_status != ROCDEC_SUCCESS) {
        std::cerr << "Expected ROCDEC_SUCCESS but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 5: Pass unsupported codec type
    decoder_create_info_.width = 1920;
    decoder_create_info_.height = 1080;
    decoder_create_info_.codec_type = static_cast<rocDecVideoCodec>(999); // Invalid codec type
    rocdecode_status = rocDecCreateDecoder(&decoder_handle_, &decoder_create_info_);
    if (rocdecode_status != ROCDEC_NOT_SUPPORTED) {
        std::cerr << "Expected ROCDEC_NOT_SUPPORTED but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    rocdecode_status = rocDecDestroyDecoder(decoder_handle_);
    if (rocdecode_status != ROCDEC_SUCCESS) {
        std::cerr << "Expected ROCDEC_SUCCESS but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 6: Pass unsupported chroma_format
    decoder_create_info_.codec_type = rocDecVideoCodec_HEVC;
    decoder_create_info_.chroma_format = static_cast<rocDecVideoChromaFormat>(999);
    rocdecode_status = rocDecCreateDecoder(&decoder_handle_, &decoder_create_info_);
    if (rocdecode_status != ROCDEC_NOT_SUPPORTED) {
        std::cerr << "Expected ROCDEC_NOT_SUPPORTED but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    rocdecode_status = rocDecDestroyDecoder(decoder_handle_);
    if (rocdecode_status != ROCDEC_SUCCESS) {
        std::cerr << "Expected ROCDEC_SUCCESS but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 7: Pass unsupported bit_depth
    decoder_create_info_.chroma_format = rocDecVideoChromaFormat_420;
    decoder_create_info_.bit_depth_minus_8 = 6;
    rocdecode_status = rocDecCreateDecoder(&decoder_handle_, &decoder_create_info_);
    if (rocdecode_status != ROCDEC_NOT_SUPPORTED) {
        std::cerr << "Expected ROCDEC_NOT_SUPPORTED but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    rocdecode_status = rocDecDestroyDecoder(decoder_handle_);
    if (rocdecode_status != ROCDEC_SUCCESS) {
        std::cerr << "Expected ROCDEC_SUCCESS but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Create a valid decoder_handle - This step ensures a valid decoder_handle_ is available for subsequent negative testing of other rocDecode APIs.
    decoder_create_info_.bit_depth_minus_8 = 2;
    rocdecode_status = rocDecCreateDecoder(&decoder_handle_, &decoder_create_info_);
    if (rocdecode_status != ROCDEC_SUCCESS) {
        std::cerr << "Expected ROCDEC_SUCCESS but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::TestInvalidDestroyDecoder() {
    std::cout << "info: Executing negative test cases for the rocDecDestroyDecoder API" << std::endl;
    //Scenario 1: Pass nullptr for decoder_handle
    rocDecStatus rocdecode_status = rocDecDestroyDecoder(nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::TestInvalidGetDecoderCaps() {
    std::cout << "info: Executing negative test cases for the rocDecGetDecoderCaps API" << std::endl;
    // Scenario 1: Pass nullptr for decode_caps
    rocDecStatus rocdecode_status = rocDecGetDecoderCaps(nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    // Scenario 2: Pass a decode_caps structure with unsupported codec type
    RocdecDecodeCaps decode_caps = {};
    decode_caps.codec_type = static_cast<rocDecVideoCodec>(-1); // Invalid codec type
    rocdecode_status = rocDecGetDecoderCaps(&decode_caps);
    if (rocdecode_status != ROCDEC_SUCCESS || decode_caps.is_supported != 0) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::TestInvalidDecodeFrame() {
    std::cout << "info: Executing negative test cases for the rocDecDecodeFrame API" << std::endl;
    // Scenario 1: Pass nullptr for pic_params
    rocDecStatus rocdecode_status = rocDecDecodeFrame(decoder_handle_, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 2: Pass invalid curr_pic_idx
    RocdecPicParams pic_params = {};
    pic_params.curr_pic_idx = -1;
    rocdecode_status = rocDecDecodeFrame(decoder_handle_, &pic_params);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 3: Pass invalid reference frame index
    pic_params.curr_pic_idx = 0;
    pic_params.pic_params.hevc.ref_frames[0].pic_idx = -1;
    rocdecode_status = rocDecDecodeFrame(decoder_handle_, &pic_params);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::TestInvalidGetDecodeStatus() {
    std::cout << "info: Executing negative test cases for the rocDecGetDecodeStatus API" << std::endl;
    // Scenario 1: Test with invalid picture parameters
    rocDecStatus rocdecode_status = rocDecGetDecodeStatus(decoder_handle_, 0, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 2: Pass with invalid parameter for pic_idx
    int pic_idx = 100;
    RocdecDecodeStatus decode_status;
    rocdecode_status = rocDecGetDecodeStatus(decoder_handle_, pic_idx, &decode_status);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::TestInvalidReconfigureDecoder() {
    std::cout << "info: Executing negative test cases for the rocDecReconfigureDecoder API" << std::endl;
    // Scenario 1: Pass nullptr for reconfig_params
    rocDecStatus rocdecode_status = rocDecReconfigureDecoder(decoder_handle_, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 2: Pass an invalid decoder handle
    RocdecReconfigureDecoderInfo reconfig_params = {};
    rocdecode_status = rocDecReconfigureDecoder(nullptr, &reconfig_params);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 3: Pass uninitialized reconfig_params
    rocdecode_status = rocDecReconfigureDecoder(decoder_handle_, &reconfig_params);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 4: Pass invalid width and height in reconfig_params
    reconfig_params.width = 0;
    reconfig_params.height = 0;
    rocdecode_status = rocDecReconfigureDecoder(decoder_handle_, &reconfig_params);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_NOT_SUPPORTED but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 5: Pass unsupported bit_depth in reconfig_params
    reconfig_params.width = 1920;
    reconfig_params.height = 1080;
    reconfig_params.bit_depth_minus_8 = 4; // Unsupported bit depth
    rocdecode_status = rocDecReconfigureDecoder(decoder_handle_, &reconfig_params);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_NOT_SUPPORTED but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int RocDecodeApiNegativeTests::TestinvalidGetVideoFrame() {
    std::cout << "info: Executing negative test cases for the rocDecGetVideoFrame API" << std::endl;
    // Scenario 1: Pass nullptr for dev_mem_ptr, horizontal_pitch, and vid_postproc_params
    rocDecStatus rocdecode_status = rocDecGetVideoFrame(decoder_handle_, 0, nullptr, nullptr, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 2: Pass an invalid picture index
    void *dev_mem_ptr[3] = {};
    uint32_t horizontal_pitch[3] = {};
    RocdecProcParams vid_postproc_params = {};
    rocdecode_status = rocDecGetVideoFrame(decoder_handle_, -1, dev_mem_ptr, horizontal_pitch, &vid_postproc_params);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::TestinvalidGetErrorName(){
    std::cout << "info: Executing negative test cases for the rocDecGetErrorName API" << std::endl;
    // Scenario 1: Pass an invalid error code
    rocDecStatus invalid_status = static_cast<rocDecStatus>(-999); // Invalid error code
    const char *error_name = rocDecGetErrorName(invalid_status);
    if (error_name == nullptr) {
        std::cerr << "Expected a valid error but got nullptr" << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 2: Pass a valid error code and ensure it returns a non-null name
    for (int i = 0; i >= -8; i--) {
        rocDecStatus valid_status = static_cast<rocDecStatus>(i);;
        error_name = rocDecGetErrorName(valid_status);
        if (error_name == nullptr) {
            std::cerr << "Expected a valid error but got nullptr" << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Scenario 3: Pass a boundary value (e.g., maximum enum value + 1)
    rocDecStatus boundary_status = static_cast<rocDecStatus>(ROCDEC_SUCCESS + 1);
    error_name = rocDecGetErrorName(boundary_status);
    if (error_name == nullptr) {
        std::cerr << "Expected a valid error but got nullptr" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::TestinvalidCreateVideoParser() {
    std::cout << "info: Executing negative test cases for the rocDecCreateVideoParser API" << std::endl;

    // Scenario 1: Pass nullptr for parser_handle and parser parameter
    rocDecStatus rocdecode_status = rocDecCreateVideoParser(nullptr, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 2: Pass a valid parser_handle but nullptr for parser parameter
    rocdecode_status = rocDecCreateVideoParser(&parser_handle_, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 3: Pass invalid code_type
    RocdecParserParams params = {};
    params.codec_type = static_cast<rocDecVideoCodec>(999); // Invalid codec type
    rocdecode_status = rocDecCreateVideoParser(&parser_handle_, &params);
    if (rocdecode_status != ROCDEC_NOT_IMPLEMENTED) {
        std::cerr << "Expected ROCDEC_NOT_IMPLEMENTED but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 4: Create a dummy parser_handle for testing other parser APIs
    params.codec_type = rocDecVideoCodec_HEVC; // Invalid codec type
    rocdecode_status = rocDecCreateVideoParser(&parser_handle_, &params);
    if (rocdecode_status != ROCDEC_SUCCESS) {
        std::cerr << "Expected ROCDEC_SUCCESS but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::TestinvalidParseVideoData() {
    std::cout << "info: Executing negative test cases for the rocDecParseVideoData API" << std::endl;
    // Scenario 1: Pass nullptr for parser_handle and parser parameter
    rocDecStatus rocdecode_status = rocDecParseVideoData(nullptr, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    // Scenario 2: Pass a parser_handle with a nullptr packet
    rocdecode_status = rocDecParseVideoData(parser_handle_, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 3: Pass a packet initialized to 0
    RocdecSourceDataPacket packet = {};
    rocdecode_status = rocDecParseVideoData(parser_handle_, &packet);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::TestinvalidDestroyVideoParser() {
    std::cout << "info: Executing negative test cases for the rocDecDestroyVideoParser API" << std::endl;
    // Scenario 1: Pass nullptr for parser_handle
    rocDecStatus rocdecode_status = rocDecDestroyVideoParser(nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::TestinvalidCreateBitstreamReader() {
    std::cout << "info: Executing negative test cases for the rocDecCreateBitstreamReader API" << std::endl;

    // Scenario 1: Pass nullptr for bs_reader_handle and input_file_path
    rocDecStatus rocdecode_status = rocDecCreateBitstreamReader(nullptr, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 2: Pass a valid bs_reader_handle but nullptr for input_file_path
    RocdecBitstreamReader bs_reader_handle;
    rocdecode_status = rocDecCreateBitstreamReader(&bs_reader_handle, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::TestinvalidGetBitstreamCodecType() {
    std::cout << "info: Executing negative test cases for the rocDecGetBitstreamCodecType API" << std::endl;

    // Scenario 1: Pass nullptr for bs_reader_handle and codec_type
    rocDecStatus rocdecode_status = rocDecGetBitstreamCodecType(nullptr, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 2: Pass a valid bs_reader_handle but nullptr for codec_type
    RocdecBitstreamReader bs_reader_handle;
    rocdecode_status = rocDecGetBitstreamCodecType(&bs_reader_handle, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::TestinvalidGetBitstreamBitDepth() {
    std::cout << "info: Executing negative test cases for the rocDecGetBitstreamBitDepth API" << std::endl;
    // Scenario 1: Pass nullptr for bs_reader_handle and bit_depth
    rocDecStatus rocdecode_status = rocDecGetBitstreamBitDepth(nullptr, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 2: Pass a valid bs_reader_handle but nullptr for bit_depth
    RocdecBitstreamReader bs_reader_handle;
    rocdecode_status = rocDecGetBitstreamBitDepth(&bs_reader_handle, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::TestinvalidGetBitstreamPicData() {
    std::cout << "info: Executing negative test cases for the rocDecGetBitstreamPicData API" << std::endl;
    // Scenario 1: Pass nullptr for all parameters
    rocDecStatus rocdecode_status = rocDecGetBitstreamPicData(nullptr, nullptr, nullptr, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    // Scenario 2: Pass a valid bs_reader_handle but nullptr for pic_data, pic_size, and pts
    RocdecBitstreamReader bs_reader_handle;
    rocdecode_status = rocDecGetBitstreamPicData(&bs_reader_handle, nullptr, nullptr, nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::TestinvalidDestroyBitstreamReader() {
    std::cout << "info: Executing negative test cases for the rocDecDestroyBitstreamReader API" << std::endl;
    // Scenario 1: Pass nullptr for bs_reader_handle
    rocDecStatus rocdecode_status = rocDecDestroyBitstreamReader(nullptr);
    if (rocdecode_status != ROCDEC_INVALID_PARAMETER) {
        std::cerr << "Expected ROCDEC_INVALID_PARAMETER but got " << rocDecGetErrorName(rocdecode_status) << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int RocDecodeApiNegativeTests::RunTests() {
    if (TestInvalidCreateDecoder() || TestInvalidDestroyDecoder() || TestInvalidGetDecoderCaps() || TestInvalidDecodeFrame() ||
        TestInvalidGetDecodeStatus() || TestInvalidReconfigureDecoder() || TestinvalidGetVideoFrame() || TestinvalidGetErrorName() ||
        TestinvalidCreateVideoParser() || TestinvalidParseVideoData() || TestinvalidDestroyVideoParser() ||
        TestinvalidCreateBitstreamReader() || TestinvalidGetBitstreamCodecType() || TestinvalidGetBitstreamBitDepth() ||
        TestinvalidGetBitstreamPicData() || TestinvalidDestroyBitstreamReader()) {
            std::cerr << "One or more negative tests failed." << std::endl;
            return EXIT_FAILURE;
        } else {
            return EXIT_SUCCESS;
        }
}