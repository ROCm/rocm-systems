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
    uint32_t channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t num_channels = 0;
    int total_images = 0;
    int batch_size = 2;
    std::string chroma_sub_sampling = "";
    std::string input_path, output_file_path;
    std::vector<std::string> file_paths = {};
    bool is_dir = false;
    bool is_file = false;
    RocJpegBackend rocjpeg_backend = ROCJPEG_BACKEND_HARDWARE;
    RocJpegHandle rocjpeg_handle = nullptr;
    RocJpegDecodeParams decode_params = {};
    RocJpegUtils rocjpeg_utils;
    uint64_t num_bad_jpegs = 0;
    uint64_t num_jpegs_with_411_subsampling = 0;
    uint64_t num_jpegs_with_unknown_subsampling = 0;
    uint64_t num_jpegs_with_unsupported_resolution = 0;

    RocJpegUtils::ParseCommandLine(input_path, output_file_path, save_images, device_id, rocjpeg_backend, decode_params, nullptr, &batch_size, argc, argv);

    if (!RocJpegUtils::GetFilePaths(input_path, file_paths, is_dir, is_file)) {
        std::cerr << "ERROR: Failed to get input file paths!" << std::endl;
        return EXIT_FAILURE;
    }
    if (!RocJpegUtils::InitHipDevice(device_id)) {
        std::cerr << "ERROR: Failed to initialize HIP!" << std::endl;
        return EXIT_FAILURE;
    }

    CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &rocjpeg_handle));

    batch_size = std::min(batch_size, static_cast<int>(file_paths.size()));

    // Create a pool of stream handles equal to the batch size.
    std::vector<RocJpegStreamHandle> rocjpeg_stream_handles(batch_size);
    for (int i = 0; i < batch_size; i++) {
        CHECK_ROCJPEG(rocJpegStreamCreate(&rocjpeg_stream_handles[i]));
    }

    // Each pipeline slot holds a full batch of GPU buffers.
    // The main thread only blocks when all slots are in-flight (available_queue empty).
    struct BatchPipelineSlot {
        std::vector<RocJpegImage> images;
        std::vector<std::vector<uint32_t>> channel_sizes_per_image;
        std::vector<uint32_t> num_channels_per_image;
        std::vector<std::vector<uint32_t>> widths;
        std::vector<std::vector<uint32_t>> heights;
        std::vector<RocJpegChromaSubsampling> subsamplings;
        std::vector<std::string> file_names;
        std::vector<RocJpegDecodeParams> decode_params_batch;
        int current_batch_size = 0;
    };

    const int buf_num = 5;
    std::vector<BatchPipelineSlot> pipeline_slots(buf_num);
    for (int p = 0; p < buf_num; p++) {
        pipeline_slots[p].images.resize(batch_size);
        pipeline_slots[p].channel_sizes_per_image.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
        pipeline_slots[p].num_channels_per_image.resize(batch_size, 0);
        pipeline_slots[p].widths.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
        pipeline_slots[p].heights.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
        pipeline_slots[p].subsamplings.resize(batch_size);
        pipeline_slots[p].file_names.resize(batch_size);
        pipeline_slots[p].decode_params_batch.resize(batch_size, decode_params);
    }

    std::queue<BatchPipelineSlot*> pending_queue;   // submitted, awaiting rocJpegDecodeBatchedSync
    std::queue<BatchPipelineSlot*> available_queue; // idle, ready for next submit
    std::mutex mtx;
    std::condition_variable cv_pending, cv_available;
    RocJpegStatus async_status = ROCJPEG_STATUS_SUCCESS;
    bool sync_done  = false;
    bool sync_error = false;
    int  in_flight  = 0;

    for (int p = 0; p < buf_num; p++)
        available_queue.push(&pipeline_slots[p]);

    // Sync thread: continuously drains pending_queue by calling rocJpegDecodeBatchedSync,
    // optionally saves the results, then returns the slot to available_queue.
    std::thread sync_thread([&]() {
        if (hipSetDevice(device_id) != hipSuccess) {
            std::lock_guard<std::mutex> lock(mtx);
            async_status = ROCJPEG_STATUS_RUNTIME_ERROR;
            sync_error   = true;
            cv_available.notify_all();
            return;
        }
        while (true) {
            BatchPipelineSlot* slot;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_pending.wait(lock, [&]{ return !pending_queue.empty() || sync_done || sync_error; });
                if (sync_error) return;
                if (pending_queue.empty()) return; // sync_done with nothing left
                slot = pending_queue.front();
                pending_queue.pop();
            }

            RocJpegStatus s = rocJpegDecodeBatchedSync(rocjpeg_handle, slot->images.data(), slot->current_batch_size);

            if (s == ROCJPEG_STATUS_SUCCESS && save_images) {
                for (int b = 0; b < slot->current_batch_size; b++) {
                    std::string image_save_path = output_file_path;
                    uint32_t img_roi_width  = slot->decode_params_batch[b].crop_rectangle.right  - slot->decode_params_batch[b].crop_rectangle.left;
                    uint32_t img_roi_height = slot->decode_params_batch[b].crop_rectangle.bottom - slot->decode_params_batch[b].crop_rectangle.top;
                    bool img_roi_valid = (img_roi_width > 0 && img_roi_height > 0 &&
                                         img_roi_width  <= slot->widths[b][0] &&
                                         img_roi_height <= slot->heights[b][0]);
                    uint32_t width  = img_roi_valid ? img_roi_width  : slot->widths[b][0];
                    uint32_t height = img_roi_valid ? img_roi_height : slot->heights[b][0];
                    if (is_dir) {
                        rocjpeg_utils.GetOutputFileExt(slot->decode_params_batch[b].output_format, slot->file_names[b], width, height, slot->subsamplings[b], image_save_path);
                    }
                    rocjpeg_utils.SaveImage(image_save_path, &slot->images[b], width, height, slot->subsamplings[b], slot->decode_params_batch[b].output_format);
                }
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

    std::vector<std::vector<char>> batch_file_data(batch_size);
    std::vector<uint32_t> temp_widths(ROCJPEG_MAX_COMPONENT, 0);
    std::vector<uint32_t> temp_heights(ROCJPEG_MAX_COMPONENT, 0);
    RocJpegChromaSubsampling temp_subsampling;
    std::vector<RocJpegStreamHandle> stream_handles_for_batch(batch_size);

    std::cout << "Decoding started, please wait! ... " << std::endl;
    auto total_start_time = std::chrono::high_resolution_clock::now();
    bool timing_started = false;

    for (int i = 0; i < static_cast<int>(file_paths.size()); i += batch_size) {
        int batch_end = std::min(i + batch_size, static_cast<int>(file_paths.size()));

        // Check for async failure before processing the next batch.
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

        // Wait for an available slot.
        BatchPipelineSlot* slot;
        {
            std::unique_lock<std::mutex> lock(mtx);
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

        int current_batch_size = 0;
        for (int j = i; j < batch_end; j++) {
            int index = j - i;
            std::string base_file_name = file_paths[j].substr(file_paths[j].find_last_of("/\\") + 1);

            std::ifstream input(file_paths[j].c_str(), std::ios::in | std::ios::binary | std::ios::ate);
            if (!input.is_open()) {
                std::cerr << "ERROR: Cannot open image: " << file_paths[j] << std::endl;
                shutdown();
                return EXIT_FAILURE;
            }
            std::streamsize file_size = input.tellg();
            input.seekg(0, std::ios::beg);
            if (batch_file_data[index].size() < (size_t)file_size)
                batch_file_data[index].resize(file_size);
            if (!input.read(batch_file_data[index].data(), file_size)) {
                std::cerr << "ERROR: Cannot read from file: " << file_paths[j] << std::endl;
                shutdown();
                return EXIT_FAILURE;
            }

            RocJpegStatus rocjpeg_status = rocJpegStreamParse(reinterpret_cast<uint8_t*>(batch_file_data[index].data()), file_size, rocjpeg_stream_handles[index]);
            if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
                if (is_dir) { num_bad_jpegs++; continue; }
                std::cerr << "ERROR: Failed to parse the input jpeg stream with " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
                shutdown();
                return EXIT_FAILURE;
            }

            CHECK_ROCJPEG(rocJpegGetImageInfo(rocjpeg_handle, rocjpeg_stream_handles[index], &num_components, &temp_subsampling, temp_widths.data(), temp_heights.data()));

            if (temp_widths[0] < 64 || temp_heights[0] < 64) {
                if (is_dir) { num_jpegs_with_unsupported_resolution++; continue; }
                std::cerr << "The image resolution is not supported by VCN Hardware" << std::endl;
                shutdown();
                return EXIT_FAILURE;
            }
            if (temp_subsampling == ROCJPEG_CSS_411 || temp_subsampling == ROCJPEG_CSS_UNKNOWN) {
                if (is_dir) {
                    if (temp_subsampling == ROCJPEG_CSS_411)     num_jpegs_with_411_subsampling++;
                    if (temp_subsampling == ROCJPEG_CSS_UNKNOWN) num_jpegs_with_unknown_subsampling++;
                    continue;
                }
                std::cerr << "The chroma sub-sampling is not supported by VCN Hardware" << std::endl;
                shutdown();
                return EXIT_FAILURE;
            }

            uint32_t local_num_channels = 0;
            if (rocjpeg_utils.GetChannelPitchAndSizes(slot->decode_params_batch[current_batch_size], temp_subsampling, temp_widths.data(), temp_heights.data(), local_num_channels, slot->images[current_batch_size], channel_sizes)) {
                std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
                shutdown();
                return EXIT_FAILURE;
            }

            // Grow slot GPU buffers if this image's channels exceed current allocation.
            for (uint32_t c = 0; c < local_num_channels; c++) {
                if (channel_sizes[c] > slot->channel_sizes_per_image[current_batch_size][c]) {
                    if (slot->images[current_batch_size].channel[c] != nullptr) {
                        CHECK_HIP(hipFree(slot->images[current_batch_size].channel[c]));
                        slot->images[current_batch_size].channel[c] = nullptr;
                    }
                    CHECK_HIP(hipMalloc(&slot->images[current_batch_size].channel[c], channel_sizes[c]));
                    slot->channel_sizes_per_image[current_batch_size][c] = channel_sizes[c];
                }
            }
            slot->num_channels_per_image[current_batch_size] = local_num_channels;
            slot->widths[current_batch_size]      = temp_widths;
            slot->heights[current_batch_size]     = temp_heights;
            slot->subsamplings[current_batch_size] = temp_subsampling;
            slot->file_names[current_batch_size]   = base_file_name;
            stream_handles_for_batch[current_batch_size] = rocjpeg_stream_handles[index];
            current_batch_size++;
        }

        slot->current_batch_size = current_batch_size;

        if (current_batch_size > 0) {
            if (!timing_started) {
                total_start_time = std::chrono::high_resolution_clock::now();
                timing_started = true;
            }
            RocJpegStatus rocjpeg_status = rocJpegDecodeBatchedAsync(rocjpeg_handle, stream_handles_for_batch.data(), current_batch_size, slot->decode_params_batch.data(), slot->images.data());
            if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
                std::cerr << "ERROR: rocJpegDecodeBatchedAsync returned " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
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

            total_images += current_batch_size;
        } else {
            // No valid images in this batch window; return slot immediately.
            std::lock_guard<std::mutex> lock(mtx);
            available_queue.push(slot);
        }
    }

    // Drain: wait for all in-flight batches to complete before reporting stats.
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv_available.wait(lock, [&]{ return in_flight == 0 || sync_error; });
    }
    auto total_end_time = std::chrono::high_resolution_clock::now();

    shutdown();

    if (sync_error) {
        std::cerr << "ERROR: Async decode failed with " << rocJpegGetErrorName(async_status) << std::endl;
    } else if (total_images > 0) {
        double total_time_ms     = std::chrono::duration<double, std::milli>(total_end_time - total_start_time).count();
        double time_per_image_ms = total_time_ms / total_images;
        std::cout << "Total decoded images: "                        << total_images << std::endl;
        std::cout << "Average processing time per image (ms): "      << time_per_image_ms << std::endl;
        std::cout << "Average images per sec (Images/Sec): "         << 1000.0 / time_per_image_ms << std::endl;
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
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < ROCJPEG_MAX_COMPONENT; c++) {
                if (pipeline_slots[p].images[b].channel[c] != nullptr) {
                    CHECK_HIP(hipFree(pipeline_slots[p].images[b].channel[c]));
                    pipeline_slots[p].images[b].channel[c] = nullptr;
                }
            }
        }
    }

    CHECK_ROCJPEG(rocJpegDestroy(rocjpeg_handle));
    for (auto& it : rocjpeg_stream_handles) {
        CHECK_ROCJPEG(rocJpegStreamDestroy(it));
    }
    std::cout << "Decoding completed!" << std::endl;
    return EXIT_SUCCESS;
}
