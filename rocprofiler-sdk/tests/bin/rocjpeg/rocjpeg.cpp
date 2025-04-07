/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include <rocjpeg/rocjpeg.h>
#include "rocjpeg_samples_utils.h"

int
main(int argc, char** argv)
{
    int                      device_id   = 0;
    bool                     save_images = false;
    uint8_t                  num_components;
    uint32_t                 widths[ROCJPEG_MAX_COMPONENT]              = {};
    uint32_t                 heights[ROCJPEG_MAX_COMPONENT]             = {};
    uint32_t                 channel_sizes[ROCJPEG_MAX_COMPONENT]       = {};
    uint32_t                 prior_channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t                 num_channels                               = 0;
    int                      total_images                               = 0;
    double                   time_per_image_all                         = 0;
    std::string              chroma_sub_sampling                        = "";
    std::string              input_path, output_file_path;
    std::vector<std::string> file_paths = {};
    bool                     is_dir     = false;
    bool                     is_file    = false;
    RocJpegChromaSubsampling subsampling;
    RocJpegBackend           rocjpeg_backend       = ROCJPEG_BACKEND_HARDWARE;
    RocJpegHandle            rocjpeg_handle        = nullptr;
    RocJpegStreamHandle      rocjpeg_stream_handle = nullptr;
    RocJpegImage             output_image          = {};
    RocJpegDecodeParams      decode_params         = {};
    RocJpegUtils             rocjpeg_utils;
    uint64_t                 num_bad_jpegs                         = 0;
    uint64_t                 num_jpegs_with_411_subsampling        = 0;
    uint64_t                 num_jpegs_with_unknown_subsampling    = 0;
    uint64_t                 num_jpegs_with_unsupported_resolution = 0;

    RocJpegUtils::ParseCommandLine(input_path,
                                   output_file_path,
                                   save_images,
                                   device_id,
                                   rocjpeg_backend,
                                   decode_params,
                                   nullptr,
                                   nullptr,
                                   argc,
                                   argv);

    bool     is_roi_valid = false;
    uint32_t roi_width;
    uint32_t roi_height;
    roi_width  = decode_params.crop_rectangle.right - decode_params.crop_rectangle.left;
    roi_height = decode_params.crop_rectangle.bottom - decode_params.crop_rectangle.top;

    if(!RocJpegUtils::GetFilePaths(input_path, file_paths, is_dir, is_file))
    {
        std::cerr << "ERROR: Failed to get input file paths!" << std::endl;
        return EXIT_FAILURE;
    }
    if(!RocJpegUtils::InitHipDevice(device_id))
    {
        std::cerr << "ERROR: Failed to initialize HIP!" << std::endl;
        return EXIT_FAILURE;
    }

    // CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &rocjpeg_handle));
    if(rocJpegCreate(rocjpeg_backend, device_id, &rocjpeg_handle) != ROCJPEG_STATUS_SUCCESS)
    {
        std::cerr << "rocJPEG tests not supported" << std::endl;
        return 0;
    }
    CHECK_ROCJPEG(rocJpegStreamCreate(&rocjpeg_stream_handle));
    std::vector<char> file_data;
    for(auto file_path : file_paths)
    {
        std::string base_file_name = file_path.substr(file_path.find_last_of("/\\") + 1);
        int         image_count    = 0;

        // Read an image from disk.
        std::ifstream input(file_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        if(!(input.is_open()))
        {
            std::cerr << "ERROR: Cannot open image: " << file_path << std::endl;
            return EXIT_FAILURE;
        }
        // Get the size
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if(file_data.size() < static_cast<size_t>(file_size))
        {
            file_data.resize(file_size);
        }
        if(!input.read(file_data.data(), file_size))
        {
            std::cerr << "ERROR: Cannot read from file: " << file_path << std::endl;
            return EXIT_FAILURE;
        }

        RocJpegStatus rocjpeg_status = rocJpegStreamParse(
            reinterpret_cast<uint8_t*>(file_data.data()), file_size, rocjpeg_stream_handle);
        if(rocjpeg_status != ROCJPEG_STATUS_SUCCESS)
        {
            if(is_dir)
            {
                num_bad_jpegs++;
                continue;
            }
            else
            {
                std::cerr << "ERROR: Failed to parse the input jpeg stream with "
                          << rocJpegGetErrorName(rocjpeg_status) << std::endl;
                return EXIT_FAILURE;
            }
        }

        CHECK_ROCJPEG(rocJpegGetImageInfo(
            rocjpeg_handle, rocjpeg_stream_handle, &num_components, &subsampling, widths, heights));

        if(roi_width > 0 && roi_height > 0 && roi_width <= widths[0] && roi_height <= heights[0])
        {
            is_roi_valid = true;
        }
        rocjpeg_utils.GetChromaSubsamplingStr(subsampling, chroma_sub_sampling);
        if(widths[0] < 64 || heights[0] < 64)
        {
            std::cerr << "The image resolution is not supported by VCN Hardware" << std::endl;
            if(is_dir)
            {
                num_jpegs_with_unsupported_resolution++;
                continue;
            }
            else
                return EXIT_FAILURE;
        }
        if(subsampling == ROCJPEG_CSS_411 || subsampling == ROCJPEG_CSS_UNKNOWN)
        {
            std::cerr << "The chroma sub-sampling is not supported by VCN Hardware" << std::endl;
            if(is_dir)
            {
                if(subsampling == ROCJPEG_CSS_411) num_jpegs_with_411_subsampling++;
                if(subsampling == ROCJPEG_CSS_UNKNOWN) num_jpegs_with_unknown_subsampling++;
                continue;
            }
            else
                return EXIT_FAILURE;
        }

        if(rocjpeg_utils.GetChannelPitchAndSizes(decode_params,
                                                 subsampling,
                                                 widths,
                                                 heights,
                                                 num_channels,
                                                 output_image,
                                                 channel_sizes))
        {
            std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
            return EXIT_FAILURE;
        }

        // allocate memory for each channel and reuse them if the sizes remain unchanged for a new
        // image.
        for(uint32_t i = 0; i < num_channels; i++)
        {
            if(prior_channel_sizes[i] != channel_sizes[i])
            {
                if(output_image.channel[i] != nullptr)
                {
                    CHECK_HIP(hipFree((void*) output_image.channel[i]));
                    output_image.channel[i] = nullptr;
                }
                CHECK_HIP(hipMalloc(&output_image.channel[i], channel_sizes[i]));
            }
        }

        if(is_roi_valid)
        {}
        auto start_time = std::chrono::high_resolution_clock::now();
        CHECK_ROCJPEG(
            rocJpegDecode(rocjpeg_handle, rocjpeg_stream_handle, &decode_params, &output_image));
        auto   end_time = std::chrono::high_resolution_clock::now();
        double time_per_image_in_milli_sec =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();
        image_count++;

        if(save_images)
        {
            std::string image_save_path = output_file_path;
            // if ROI is present, need to pass roi_width and roi_height
            uint32_t width  = is_roi_valid ? roi_width : widths[0];
            uint32_t height = is_roi_valid ? roi_height : heights[0];
            if(is_dir)
            {
                rocjpeg_utils.GetOutputFileExt(decode_params.output_format,
                                               base_file_name,
                                               width,
                                               height,
                                               subsampling,
                                               image_save_path);
            }
            rocjpeg_utils.SaveImage(image_save_path,
                                    &output_image,
                                    width,
                                    height,
                                    subsampling,
                                    decode_params.output_format);
        }

        if(is_dir)
        {
            total_images += image_count;
            time_per_image_all += time_per_image_in_milli_sec;
        }
        for(int i = 0; i < ROCJPEG_MAX_COMPONENT; i++)
        {
            prior_channel_sizes[i] = channel_sizes[i];
        }
    }

    for(uint32_t i = 0; i < num_channels; i++)
    {
        if(output_image.channel[i] != nullptr)
        {
            CHECK_HIP(hipFree((void*) output_image.channel[i]));
            output_image.channel[i] = nullptr;
        }
    }

    if(is_dir)
    {
        time_per_image_all = time_per_image_all / total_images;
        if(num_bad_jpegs || num_jpegs_with_411_subsampling || num_jpegs_with_unknown_subsampling ||
           num_jpegs_with_unsupported_resolution)
        {
            if(num_bad_jpegs)
            {
                std::cout << " ,total images that cannot be parsed: " << num_bad_jpegs;
            }
            if(num_jpegs_with_411_subsampling)
            {
                std::cout << " ,total images with YUV 4:1:1 chroam subsampling: "
                          << num_jpegs_with_411_subsampling;
            }
            if(num_jpegs_with_unknown_subsampling)
            {
                std::cout << " ,total images with unknwon chroam subsampling: "
                          << num_jpegs_with_unknown_subsampling;
            }
            if(num_jpegs_with_unsupported_resolution)
            {
                std::cout << " ,total images with unsupported_resolution: "
                          << num_jpegs_with_unsupported_resolution;
            }
            std::cout << std::endl;
        }
    }

    CHECK_ROCJPEG(rocJpegDestroy(rocjpeg_handle));
    CHECK_ROCJPEG(rocJpegStreamDestroy(rocjpeg_stream_handle));
    return EXIT_SUCCESS;
}
