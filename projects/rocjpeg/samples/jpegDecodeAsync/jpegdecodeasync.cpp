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

#include "../rocjpeg_samples_utils.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

int main(int argc, char **argv) {
    int device_id = 0;
    bool save_images = false;
    uint8_t num_components;
    uint32_t widths[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t heights[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t num_channels = 0;
    std::string chroma_sub_sampling = "";
    std::string input_path, output_file_path;
    std::vector<std::string> file_paths = {};
    bool is_dir = false;
    bool is_file = false;
    RocJpegChromaSubsampling subsampling;
    RocJpegBackend rocjpeg_backend = ROCJPEG_BACKEND_HARDWARE;
    RocJpegHandle rocjpeg_handle = nullptr;
    RocJpegStreamHandle rocjpeg_stream_handle = nullptr;
    RocJpegImage output_image = {};
    RocJpegDecodeParams decode_params = {};
    RocJpegUtils rocjpeg_utils;
    int num_iterations = 1;
    const int buf_num = 5;

    RocJpegUtils::ParseCommandLine(input_path, output_file_path, save_images, device_id, rocjpeg_backend, decode_params, nullptr, nullptr, argc, argv, &num_iterations);

    bool is_roi_valid = false;
    uint32_t roi_width;
    uint32_t roi_height;
    roi_width = decode_params.crop_rectangle.right - decode_params.crop_rectangle.left;
    roi_height = decode_params.crop_rectangle.bottom - decode_params.crop_rectangle.top;

    if (!RocJpegUtils::GetFilePaths(input_path, file_paths, is_dir, is_file)) {
        std::cerr << "ERROR: Failed to get input file paths!" << std::endl;
        return EXIT_FAILURE;
    }
    if (file_paths.empty()) {
        std::cerr << "ERROR: No input file found!" << std::endl;
        return EXIT_FAILURE;
    }
    if (!RocJpegUtils::InitHipDevice(device_id)) {
        std::cerr << "ERROR: Failed to initialize HIP!" << std::endl;
        return EXIT_FAILURE;
    }

    CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &rocjpeg_handle));
    CHECK_ROCJPEG(rocJpegStreamCreate(&rocjpeg_stream_handle));

    // Use the first file only
    std::string file_path = file_paths[0];

    // Read the image from disk.
    std::ifstream input(file_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if (!(input.is_open())) {
        std::cerr << "ERROR: Cannot open image: " << file_path << std::endl;
        return EXIT_FAILURE;
    }
    std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<char> file_data(file_size);
    if (!input.read(file_data.data(), file_size)) {
        std::cerr << "ERROR: Cannot read from file: " << file_path << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Input file name: " << file_path << std::endl;
    CHECK_ROCJPEG(rocJpegStreamParse(reinterpret_cast<uint8_t*>(file_data.data()), file_size, rocjpeg_stream_handle));
    CHECK_ROCJPEG(rocJpegGetImageInfo(rocjpeg_handle, rocjpeg_stream_handle, &num_components, &subsampling, widths, heights));

    if (roi_width > 0 && roi_height > 0 && roi_width <= widths[0] && roi_height <= heights[0]) {
        is_roi_valid = true;
    }

    rocjpeg_utils.GetChromaSubsamplingStr(subsampling, chroma_sub_sampling);
    std::cout << "Input image resolution: " << widths[0] << "x" << heights[0] << std::endl;
    std::cout << "Chroma subsampling: " + chroma_sub_sampling << std::endl;
    if (widths[0] < 64 || heights[0] < 64) {
        std::cerr << "The image resolution is not supported by VCN Hardware" << std::endl;
        return EXIT_FAILURE;
    }
    if (subsampling == ROCJPEG_CSS_411 || subsampling == ROCJPEG_CSS_UNKNOWN) {
        std::cerr << "The chroma sub-sampling is not supported by VCN Hardware" << std::endl;
        return EXIT_FAILURE;
    }

    if (rocjpeg_utils.GetChannelPitchAndSizes(decode_params, subsampling, widths, heights, num_channels, output_image, channel_sizes)) {
        std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate pipeline buffers
    std::vector<RocJpegImage> pipeline_images(buf_num);
    for (int p = 0; p < buf_num; p++) {
        memset(&pipeline_images[p], 0, sizeof(RocJpegImage));
        for (uint32_t c = 0; c < num_channels; c++) {
            pipeline_images[p].pitch[c] = output_image.pitch[c];
            CHECK_HIP(hipMalloc(&pipeline_images[p].channel[c], channel_sizes[c]));
        }
    }

    // Allocate output_image for saving
    for (uint32_t i = 0; i < num_channels; i++) {
        CHECK_HIP(hipMalloc(&output_image.channel[i], channel_sizes[i]));
    }

    std::queue<RocJpegImage*> pending_queue;
    std::queue<RocJpegImage*> available_queue;
    std::mutex mtx;
    std::condition_variable cv_pending, cv_available;
    RocJpegStatus sync_status = ROCJPEG_STATUS_SUCCESS;
    bool sync_error = false;
    RocJpegImage* last_sync_image = nullptr;

    for (int p = 0; p < buf_num; p++)
        available_queue.push(&pipeline_images[p]);

    // Sync thread: waits for submitted decodes and syncs them
    std::thread sync_thread([&]() {
        CHECK_HIP(hipSetDevice(device_id));
        for (int iter = 0; iter < num_iterations; iter++) {
            RocJpegImage* img;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_pending.wait(lock, [&]{ return !pending_queue.empty() || sync_error; });
                if (sync_error) return;
                img = pending_queue.front();
                pending_queue.pop();
            }
            RocJpegStatus status = rocJpegDecodeSync(rocjpeg_handle, img);
            if (status != ROCJPEG_STATUS_SUCCESS) {
                sync_status = status;
                sync_error = true;
                cv_available.notify_one();
                return;
            }
            last_sync_image = img;
            {
                std::lock_guard<std::mutex> lock(mtx);
                available_queue.push(img);
            }
            cv_available.notify_one();
        }
    });

    if (is_roi_valid) {
        std::cout << "Cropped image resolution: " << roi_width << "x" << roi_height << std::endl;
    }
    std::cout << "Async decoding started (" << num_iterations << " iterations), please wait! ... " << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Main thread: submits decodes
    for (int iter = 0; iter < num_iterations; iter++) {
        RocJpegImage* img;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_available.wait(lock, [&]{ return !available_queue.empty() || sync_error; });
            if (sync_error) break;
            img = available_queue.front();
            available_queue.pop();
        }
        RocJpegStatus status = rocJpegDecodeAsync(rocjpeg_handle, rocjpeg_stream_handle, &decode_params, img);
        if (status != ROCJPEG_STATUS_SUCCESS) {
            std::cerr << "ERROR: rocJpegDecodeAsync failed with " << rocJpegGetErrorName(status) << std::endl;
            sync_error = true;
            cv_pending.notify_one();
            break;
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            pending_queue.push(img);
        }
        cv_pending.notify_one();
    }

    // Wait for sync thread to finish
    sync_thread.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double time_per_image_ms = total_time_ms / num_iterations;

    if (sync_error) {
        std::cerr << "ERROR: Sync thread failed with " << rocJpegGetErrorName(sync_status) << std::endl;
        for (int p = 0; p < buf_num; p++) {
            for (int c = 0; c < ROCJPEG_MAX_COMPONENT; c++) {
                if (pipeline_images[p].channel[c] != nullptr)
                    hipFree(pipeline_images[p].channel[c]);
            }
        }
        for (int i = 0; i < ROCJPEG_MAX_COMPONENT; i++) {
            if (output_image.channel[i] != nullptr)
                hipFree((void *)output_image.channel[i]);
        }
        rocJpegDestroy(rocjpeg_handle);
        rocJpegStreamDestroy(rocjpeg_stream_handle);
        return EXIT_FAILURE;
    }

    // Copy last result to output_image for saving
    if (save_images && last_sync_image != nullptr) {
        for (uint32_t c = 0; c < num_channels; c++) {
            if (last_sync_image->channel[c] != nullptr && output_image.channel[c] != nullptr) {
                CHECK_HIP(hipMemcpy(output_image.channel[c], last_sync_image->channel[c], channel_sizes[c], hipMemcpyDeviceToDevice));
            }
        }
        std::string image_save_path = output_file_path;
        uint32_t width = is_roi_valid ? roi_width : widths[0];
        uint32_t height = is_roi_valid ? roi_height : heights[0];
        rocjpeg_utils.SaveImage(image_save_path, &output_image, width, height, subsampling, decode_params.output_format);
    }

    std::cout << "Total time (ms): " << total_time_ms << std::endl;
    std::cout << "Average processing time per image (ms): " << time_per_image_ms << std::endl;
    std::cout << "Average images per sec: " << 1000.0 / time_per_image_ms << std::endl;

    // Free pipeline buffers
    for (int p = 0; p < buf_num; p++) {
        for (int c = 0; c < ROCJPEG_MAX_COMPONENT; c++) {
            if (pipeline_images[p].channel[c] != nullptr) {
                CHECK_HIP(hipFree(pipeline_images[p].channel[c]));
            }
        }
    }
    for (int i = 0; i < ROCJPEG_MAX_COMPONENT; i++) {
        if (output_image.channel[i] != nullptr) {
            CHECK_HIP(hipFree((void *)output_image.channel[i]));
            output_image.channel[i] = nullptr;
        }
    }

    CHECK_ROCJPEG(rocJpegDestroy(rocjpeg_handle));
    CHECK_ROCJPEG(rocJpegStreamDestroy(rocjpeg_stream_handle));
    std::cout << "Decoding completed!" << std::endl;
    return EXIT_SUCCESS;
}
