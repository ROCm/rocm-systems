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

int main(int argc, char **argv) {
    int device_id = 0;
    bool save_images = false;
    uint8_t num_components;
    uint32_t widths[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t heights[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t num_channels = 0;
    int total_images = 0;
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
    uint64_t num_bad_jpegs = 0;
    uint64_t num_jpegs_with_411_subsampling = 0;
    uint64_t num_jpegs_with_unknown_subsampling = 0;
    uint64_t num_jpegs_with_unsupported_resolution = 0;
    int num_iterations = 1;

    RocJpegUtils::ParseCommandLine(input_path, output_file_path, save_images, device_id, rocjpeg_backend, decode_params, nullptr, nullptr, argc, argv, &num_iterations);

    bool is_roi_valid = false;
    uint32_t roi_width  = decode_params.crop_rectangle.right  - decode_params.crop_rectangle.left;
    uint32_t roi_height = decode_params.crop_rectangle.bottom - decode_params.crop_rectangle.top;

    if (!RocJpegUtils::GetFilePaths(input_path, file_paths, is_dir, is_file)) {
        std::cerr << "ERROR: Failed to get input file paths!" << std::endl;
        return EXIT_FAILURE;
    }
    if (!RocJpegUtils::InitHipDevice(device_id)) {
        std::cerr << "ERROR: Failed to initialize HIP!" << std::endl;
        return EXIT_FAILURE;
    }

    CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &rocjpeg_handle));
    CHECK_ROCJPEG(rocJpegStreamCreate(&rocjpeg_stream_handle));

    std::vector<char> file_data;

    // Each pipeline slot is self-contained: it owns its GPU buffers and carries
    // the image metadata needed for saving, so the main thread can overwrite its
    // local variables for the next image without touching in-flight slots.
    struct PipelineSlot {
        RocJpegImage image = {};
        uint32_t channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
        uint32_t num_channels = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        RocJpegChromaSubsampling subsampling = {};
        std::string file_path;
    };

    // Pipeline uses buf_num slots shared between main (submit) and sync (wait) threads.
    // The main thread only blocks when all slots are in-flight (available_queue empty).
    const int buf_num = 5;
    std::vector<PipelineSlot> pipeline_slots(buf_num);
    std::queue<PipelineSlot*> pending_queue;   // submitted, awaiting rocJpegDecodeSync
    std::queue<PipelineSlot*> available_queue; // idle, ready for next submit
    std::mutex mtx;
    std::condition_variable cv_pending, cv_available;
    RocJpegStatus async_status = ROCJPEG_STATUS_SUCCESS;
    bool sync_done  = false; // set by main thread when all images are submitted
    bool sync_error = false; // set by sync thread on failure
    int  in_flight  = 0;     // decodes submitted but not yet synced

    for (int p = 0; p < buf_num; p++)
        available_queue.push(&pipeline_slots[p]);

    // Sync thread: continuously drains pending_queue by calling rocJpegDecodeSync,
    // optionally saves the result, then returns the slot to available_queue.
    std::thread sync_thread([&]() {
        if (hipSetDevice(device_id) != hipSuccess) {
            std::lock_guard<std::mutex> lock(mtx);
            async_status = ROCJPEG_STATUS_RUNTIME_ERROR;
            sync_error   = true;
            cv_available.notify_all();
            return;
        }
        while (true) {
            PipelineSlot* slot;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_pending.wait(lock, [&]{ return !pending_queue.empty() || sync_done || sync_error; });
                if (sync_error) return;
                if (pending_queue.empty()) return; // sync_done with nothing left
                slot = pending_queue.front();
                pending_queue.pop();
            }

            RocJpegStatus s = rocJpegDecodeSync(rocjpeg_handle, &slot->image);

            if (s == ROCJPEG_STATUS_SUCCESS && save_images) {
                std::string image_save_path = output_file_path;
                uint32_t width  = is_roi_valid ? roi_width  : slot->width;
                uint32_t height = is_roi_valid ? roi_height : slot->height;
                std::string base = slot->file_path.substr(slot->file_path.find_last_of("/\\") + 1);
                if (is_dir)
                    rocjpeg_utils.GetOutputFileExt(decode_params.output_format, base, width, height, slot->subsampling, image_save_path);
                rocjpeg_utils.SaveImage(image_save_path, &slot->image, width, height, slot->subsampling, decode_params.output_format);
            }

            {
                std::lock_guard<std::mutex> lock(mtx);
                in_flight--;
                if (s != ROCJPEG_STATUS_SUCCESS) {
                    async_status = s;
                    sync_error   = true;
                    cv_available.notify_all();
                    return;
                }
                available_queue.push(slot);
            }
            cv_available.notify_one();
        }
    });

    auto shutdown = [&]() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            sync_done = true;
        }
        cv_pending.notify_one();
        if (sync_thread.joinable()) sync_thread.join();
    };

    auto total_start_time = std::chrono::high_resolution_clock::now();

    for (auto& file_path : file_paths) {
        std::string base_file_name = file_path.substr(file_path.find_last_of("/\\") + 1);

        // Read image from disk.
        std::ifstream input(file_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        if (!input.is_open()) {
            std::cerr << "ERROR: Cannot open image: " << file_path << std::endl;
            shutdown();
            return EXIT_FAILURE;
        }
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        if (file_data.size() < (size_t)file_size)
            file_data.resize(file_size);
        if (!input.read(file_data.data(), file_size)) {
            std::cerr << "ERROR: Cannot read from file: " << file_path << std::endl;
            shutdown();
            return EXIT_FAILURE;
        }

        std::cout << "Input file name: " << file_path << std::endl;
        RocJpegStatus rocjpeg_status = rocJpegStreamParse(reinterpret_cast<uint8_t*>(file_data.data()), file_size, rocjpeg_stream_handle);
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            if (is_dir) { num_bad_jpegs++; std::cout << std::endl; continue; }
            std::cerr << "ERROR: Failed to parse the input jpeg stream with " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
            shutdown();
            return EXIT_FAILURE;
        }

        CHECK_ROCJPEG(rocJpegGetImageInfo(rocjpeg_handle, rocjpeg_stream_handle, &num_components, &subsampling, widths, heights));

        if (roi_width > 0 && roi_height > 0 && roi_width <= widths[0] && roi_height <= heights[0])
            is_roi_valid = true;

        rocjpeg_utils.GetChromaSubsamplingStr(subsampling, chroma_sub_sampling);
        std::cout << "Input image resolution: " << widths[0] << "x" << heights[0] << std::endl;
        std::cout << "Chroma subsampling: " << chroma_sub_sampling << std::endl;

        if (widths[0] < 64 || heights[0] < 64) {
            std::cerr << "The image resolution is not supported by VCN Hardware" << std::endl;
            if (is_dir) { num_jpegs_with_unsupported_resolution++; std::cout << std::endl; continue; }
            shutdown();
            return EXIT_FAILURE;
        }
        if (subsampling == ROCJPEG_CSS_411 || subsampling == ROCJPEG_CSS_UNKNOWN) {
            std::cerr << "The chroma sub-sampling is not supported by VCN Hardware" << std::endl;
            if (is_dir) {
                if (subsampling == ROCJPEG_CSS_411)    num_jpegs_with_411_subsampling++;
                if (subsampling == ROCJPEG_CSS_UNKNOWN) num_jpegs_with_unknown_subsampling++;
                std::cout << std::endl;
                continue;
            }
            shutdown();
            return EXIT_FAILURE;
        }

        if (rocjpeg_utils.GetChannelPitchAndSizes(decode_params, subsampling, widths, heights, num_channels, output_image, channel_sizes)) {
            std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
            shutdown();
            return EXIT_FAILURE;
        }

        // Submit num_iterations async decodes for this image without waiting for
        // any of them to complete — the sync thread drains concurrently.
        for (int iter = 0; iter < num_iterations; iter++) {
            // Check for async failure before blocking on a slot.
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (sync_error) {
                    std::cerr << "ERROR: Async decode failed with " << rocJpegGetErrorName(async_status) << std::endl;
                    sync_done = true;
                    cv_pending.notify_one();
                    if (sync_thread.joinable()) sync_thread.join();
                    return EXIT_FAILURE;
                }
            }

            PipelineSlot* slot;
            {
                std::unique_lock<std::mutex> lock(mtx);
                // Block only when all slots are in-flight (natural backpressure).
                cv_available.wait(lock, [&]{ return !available_queue.empty() || sync_error; });
                if (sync_error) {
                    std::cerr << "ERROR: Async decode failed with " << rocJpegGetErrorName(async_status) << std::endl;
                    sync_done = true;
                    cv_pending.notify_one();
                    lock.unlock();
                    if (sync_thread.joinable()) sync_thread.join();
                    return EXIT_FAILURE;
                }
                slot = available_queue.front();
                available_queue.pop();
            }

            // Grow slot GPU buffers if this image's channels exceed current allocation.
            for (uint32_t c = 0; c < num_channels; c++) {
                if (channel_sizes[c] > slot->channel_sizes[c]) {
                    if (slot->image.channel[c] != nullptr) {
                        CHECK_HIP(hipFree(slot->image.channel[c]));
                        slot->image.channel[c] = nullptr;
                    }
                    CHECK_HIP(hipMalloc(&slot->image.channel[c], channel_sizes[c]));
                    slot->channel_sizes[c] = channel_sizes[c];
                }
                slot->image.pitch[c] = output_image.pitch[c];
            }
            slot->num_channels = num_channels;
            slot->width         = widths[0];
            slot->height        = heights[0];
            slot->subsampling   = subsampling;
            slot->file_path     = file_path;

            rocjpeg_status = rocJpegDecodeAsync(rocjpeg_handle, rocjpeg_stream_handle, &decode_params, &slot->image);
            if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
                std::cerr << "ERROR: rocJpegDecodeAsync returned " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    async_status = rocjpeg_status;
                    sync_error   = true;
                }
                cv_pending.notify_one();
                shutdown();
                return EXIT_FAILURE;
            }

            {
                std::lock_guard<std::mutex> lock(mtx);
                in_flight++;
                pending_queue.push(slot);
            }
            cv_pending.notify_one();
        }

        total_images++;
    }

    // Drain: wait for all in-flight decodes to complete before reporting stats.
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv_available.wait(lock, [&]{ return in_flight == 0 || sync_error; });
    }
    auto total_end_time = std::chrono::high_resolution_clock::now();

    shutdown();

    if (sync_error) {
        std::cerr << "ERROR: Async decode failed with " << rocJpegGetErrorName(async_status) << std::endl;
    } else if (total_images > 0) {
        double total_time_ms    = std::chrono::duration<double, std::milli>(total_end_time - total_start_time).count();
        double time_per_image_ms = total_time_ms / (total_images * num_iterations);
        std::cout << "Total decoded images: "                << total_images * num_iterations << std::endl;
        std::cout << "Average processing time per image (ms): " << time_per_image_ms << std::endl;
        std::cout << "Average images per sec (Images/Sec): " << 1000.0 / time_per_image_ms << std::endl;
    }

    if (num_bad_jpegs || num_jpegs_with_411_subsampling || num_jpegs_with_unknown_subsampling || num_jpegs_with_unsupported_resolution) {
        std::cout << "Total skipped images: "
                  << num_bad_jpegs + num_jpegs_with_411_subsampling + num_jpegs_with_unknown_subsampling + num_jpegs_with_unsupported_resolution;
        if (num_bad_jpegs)                        std::cout << " ,total images that cannot be parsed: "               << num_bad_jpegs;
        if (num_jpegs_with_411_subsampling)        std::cout << " ,total images with YUV 4:1:1 chroma subsampling: "  << num_jpegs_with_411_subsampling;
        if (num_jpegs_with_unknown_subsampling)    std::cout << " ,total images with unknown chroma subsampling: "    << num_jpegs_with_unknown_subsampling;
        if (num_jpegs_with_unsupported_resolution) std::cout << " ,total images with unsupported resolution: "        << num_jpegs_with_unsupported_resolution;
        std::cout << std::endl;
    }

    // Cleanup pipeline slot GPU buffers.
    for (int p = 0; p < buf_num; p++) {
        for (int c = 0; c < ROCJPEG_MAX_COMPONENT; c++) {
            if (pipeline_slots[p].image.channel[c] != nullptr) {
                CHECK_HIP(hipFree(pipeline_slots[p].image.channel[c]));
                pipeline_slots[p].image.channel[c] = nullptr;
            }
        }
    }

    // output_image holds only pitch/size metadata (no channel buffers allocated in async mode).
    CHECK_ROCJPEG(rocJpegDestroy(rocjpeg_handle));
    CHECK_ROCJPEG(rocJpegStreamDestroy(rocjpeg_stream_handle));
    std::cout << "Decoding completed!" << std::endl;
    return EXIT_SUCCESS;
}
