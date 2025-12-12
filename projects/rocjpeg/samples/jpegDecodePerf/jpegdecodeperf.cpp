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

struct DecodeInfo {
    std::vector<std::string> file_paths;
    RocJpegHandle rocjpeg_handle;
    std::vector<RocJpegStreamHandle> rocjpeg_stream_handles;
    uint64_t num_decoded_images;
    double images_per_sec;
    double image_size_in_mpixels_per_sec;
    uint64_t num_bad_jpegs;
    uint64_t num_jpegs_with_411_subsampling;
    uint64_t num_jpegs_with_unknown_subsampling;
    uint64_t num_jpegs_with_unsupported_resolution;
};

/**
 * @brief Decodes a batch of JPEG images and optionally saves the decoded images.
 *
 * @param decode_info parameters info for decoding a batch of jpeg images.
 * @param rocjpeg_utils Utility functions for RocJpeg operations.
 * @param decode_params Parameters for decoding the JPEG images (output_format, crop_rectangle)
 * @param save_images A boolean flag indicating whether to save the decoded images.
 * @param output_file_path The file path where the decoded images will be saved.
 * @param batch_size The number of images to be processed in each batch.
 */
void DecodeImages(DecodeInfo &decode_info, RocJpegUtils rocjpeg_utils, RocJpegDecodeParams &decode_params, bool save_images, std::string &output_file_path, int batch_size, int device_id) {

    bool is_roi_valid = false;
    uint32_t roi_width;
    uint32_t roi_height;
    uint8_t num_components;
    uint32_t channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    std::string chroma_sub_sampling = "";
    uint32_t num_channels = 0;
    double image_size_in_mpixels_all = 0;
    double total_decode_time_in_milli_sec = 0;
    int current_batch_size = 0;
    std::vector<std::vector<char>> batch_images(batch_size);
    std::vector<std::vector<uint32_t>> widths(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    std::vector<std::vector<uint32_t>> heights(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    std::vector<std::vector<uint32_t>> allocated_channel_sizes(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    std::vector<RocJpegChromaSubsampling> subsamplings(batch_size);
    std::vector<RocJpegImage> output_images(batch_size);
    std::vector<RocJpegDecodeParams> decode_params_batch(batch_size, decode_params);
    std::vector<std::string> base_file_names(batch_size);
    std::vector<RocJpegStreamHandle> rocjpeg_stream_handles(batch_size);
    std::vector<uint32_t> temp_widths(ROCJPEG_MAX_COMPONENT, 0);
    std::vector<uint32_t> temp_heights(ROCJPEG_MAX_COMPONENT, 0);
    RocJpegChromaSubsampling temp_subsampling;
    std::string temp_base_file_name;

    CHECK_HIP(hipSetDevice(device_id));

    // Pre-allocate GPU memory at 2560x1440 to minimize hipMalloc/hipFree during decoding.
    // Sizes mirror GetChannelPitchAndSizes logic: only channels required by the selected
    // output_format are allocated. For NATIVE/YUV_PLANAR the subsampling is unknown upfront,
    // so worst-case sizes are used per channel (ch0: packed 4:2:2 = 2w; ch1/2: 4:4:4 full height).
    static constexpr uint32_t kMemAlignment = 16;
    static constexpr uint32_t kPreAllocWidth = 2560;
    static constexpr uint32_t kPreAllocHeight = 1440;
    auto align_up = [](uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); };
    const uint32_t aligned_w  = align_up(kPreAllocWidth, kMemAlignment);
    const uint32_t aligned_w2 = align_up(kPreAllocWidth, kMemAlignment) * 2; // packed YUV 4:2:2
    const uint32_t aligned_w3 = align_up(kPreAllocWidth, kMemAlignment) * 3; // packed RGB
    const uint32_t aligned_h  = align_up(kPreAllocHeight, kMemAlignment);
    uint32_t kPreAllocSizes[ROCJPEG_MAX_COMPONENT] = {};
    switch (decode_params.output_format) {
        case ROCJPEG_OUTPUT_Y:
            // 1 channel: luma plane only
            kPreAllocSizes[0] = aligned_w * aligned_h;
            break;
        case ROCJPEG_OUTPUT_RGB:
            // 1 channel: packed RGB (pitch = width * 3)
            kPreAllocSizes[0] = aligned_w3 * aligned_h;
            break;
        case ROCJPEG_OUTPUT_RGB_PLANAR:
            // 3 channels: planar R, G, B — all full resolution
            kPreAllocSizes[0] = kPreAllocSizes[1] = kPreAllocSizes[2] = aligned_w * aligned_h;
            break;
        case ROCJPEG_OUTPUT_YUV_PLANAR:
            // up to 3 channels; worst case is 4:4:4 where all planes are full resolution
            kPreAllocSizes[0] = kPreAllocSizes[1] = kPreAllocSizes[2] = aligned_w * aligned_h;
            break;
        case ROCJPEG_OUTPUT_NATIVE:
        default:
            // Subsampling unknown at pre-alloc time; cover worst case per channel:
            //   ch0: packed 4:2:2 uses pitch=2w (largest single-channel layout)
            //   ch1/ch2: 4:4:4 uses full height (largest multi-channel layout)
            kPreAllocSizes[0] = aligned_w2 * aligned_h;
            kPreAllocSizes[1] = kPreAllocSizes[2] = aligned_w * aligned_h;
            break;
    }
    for (int b = 0; b < batch_size; b++) {
        for (int n = 0; n < ROCJPEG_MAX_COMPONENT; n++) {
            if (kPreAllocSizes[n] > 0) {
                CHECK_HIP(hipMalloc(&output_images[b].channel[n], kPreAllocSizes[n]));
                allocated_channel_sizes[b][n] = kPreAllocSizes[n];
            }
        }
    }

    // RAII guard: ensures all pre-allocated device buffers are freed on any exit path,
    // including early returns from I/O or API errors.
    struct OutputImagesGuard {
        std::vector<RocJpegImage>& images;
        ~OutputImagesGuard() {
            for (auto& it : images) {
                for (int i = 0; i < ROCJPEG_MAX_COMPONENT; i++) {
                    if (it.channel[i] != nullptr) {
                        (void)hipFree((void*)it.channel[i]);
                        it.channel[i] = nullptr;
                    }
                }
            }
        }
    } output_images_guard{output_images};

    for (int i = 0; i < decode_info.file_paths.size(); i += batch_size) {
        int batch_end = std::min(i + batch_size, static_cast<int>(decode_info.file_paths.size()));
        for (int j = i; j < batch_end; j++) {
            int index = j - i;

            temp_base_file_name = decode_info.file_paths[j].substr(decode_info.file_paths[j].find_last_of("/\\") + 1);
            // Read an image from disk.
            std::ifstream input(decode_info.file_paths[j].c_str(), std::ios::in | std::ios::binary | std::ios::ate);
            if (!(input.is_open())) {
                std::cerr << "ERROR: Cannot open image: " << decode_info.file_paths[j] << std::endl;
                return;
            }
            // Get the size
            std::streamsize file_size = input.tellg();
            input.seekg(0, std::ios::beg);
            // resize if buffer is too small
            if (batch_images[index].size() < file_size) {
                batch_images[index].resize(file_size);
            }
            if (!input.read(batch_images[index].data(), file_size)) {
                std::cerr << "ERROR: Cannot read from file: " << decode_info.file_paths[j] << std::endl;
                return;
            }

            RocJpegStatus rocjpeg_status = rocJpegStreamParse(reinterpret_cast<uint8_t*>(batch_images[index].data()), file_size, decode_info.rocjpeg_stream_handles[index]);
            if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
                decode_info.num_bad_jpegs++;
                continue;
            }

            CHECK_ROCJPEG(rocJpegGetImageInfo(decode_info.rocjpeg_handle, decode_info.rocjpeg_stream_handles[index], &num_components, &temp_subsampling, temp_widths.data(), temp_heights.data()));

            rocjpeg_utils.GetChromaSubsamplingStr(temp_subsampling, chroma_sub_sampling);
            if (temp_widths[0] < 64 || temp_heights[0] < 64) {
                decode_info.num_jpegs_with_unsupported_resolution++;
                continue;
            }

            if (temp_subsampling == ROCJPEG_CSS_411 || temp_subsampling == ROCJPEG_CSS_UNKNOWN) {
                if (temp_subsampling == ROCJPEG_CSS_411) {
                    decode_info.num_jpegs_with_411_subsampling++;
                }
                if (temp_subsampling == ROCJPEG_CSS_UNKNOWN) {
                    decode_info.num_jpegs_with_unknown_subsampling++;
                }
                continue;
            }

            if (rocjpeg_utils.GetChannelPitchAndSizes(decode_params_batch[index], temp_subsampling, temp_widths.data(), temp_heights.data(), num_channels, output_images[current_batch_size], channel_sizes)) {
                std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
                return;
            }

            // Reuse pre-allocated GPU memory if it is large enough; only grow when the current
            // image's channel size exceeds the true allocated buffer size.
            for (int n = 0; n < num_channels; n++) {
                if (channel_sizes[n] > allocated_channel_sizes[current_batch_size][n]) {
                    if (output_images[current_batch_size].channel[n] != nullptr) {
                        CHECK_HIP(hipFree((void *)output_images[current_batch_size].channel[n]));
                        output_images[current_batch_size].channel[n] = nullptr;
                    }
                    CHECK_HIP(hipMalloc(&output_images[current_batch_size].channel[n], channel_sizes[n]));
                    allocated_channel_sizes[current_batch_size][n] = channel_sizes[n];
                }
            }

            rocjpeg_stream_handles[current_batch_size] = decode_info.rocjpeg_stream_handles[index];
            subsamplings[current_batch_size] = temp_subsampling;
            widths[current_batch_size] = temp_widths;
            heights[current_batch_size] = temp_heights;
            base_file_names[current_batch_size] = temp_base_file_name;
            current_batch_size++;
        }

        double time_per_batch_in_milli_sec = 0;
        if (current_batch_size > 0) {
            auto start_time = std::chrono::high_resolution_clock::now();
            CHECK_ROCJPEG(rocJpegDecodeBatched(decode_info.rocjpeg_handle, rocjpeg_stream_handles.data(), current_batch_size, decode_params_batch.data(), output_images.data()));
            auto end_time = std::chrono::high_resolution_clock::now();
            time_per_batch_in_milli_sec = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        }

        double image_size_in_mpixels = 0;
        for (int b = 0; b < current_batch_size; b++) {
            image_size_in_mpixels += (static_cast<double>(widths[b][0]) * static_cast<double>(heights[b][0]) / 1000000);
        }

        decode_info.num_decoded_images += current_batch_size;

        if (save_images) {
            for (int b = 0; b < current_batch_size; b++) {
                std::string image_save_path = output_file_path;
                //if ROI is present, need to pass roi_width and roi_height
                roi_width = decode_params_batch[b].crop_rectangle.right - decode_params_batch[b].crop_rectangle.left;
                roi_height = decode_params_batch[b].crop_rectangle.bottom - decode_params_batch[b].crop_rectangle.top;
                is_roi_valid = (roi_width > 0 && roi_height > 0 && roi_width <= widths[b][0] && roi_height <= heights[b][0]) ? true : false;
                uint32_t width = is_roi_valid ? roi_width : widths[b][0];
                uint32_t height = is_roi_valid ? roi_height : heights[b][0];
                rocjpeg_utils.GetOutputFileExt(decode_params.output_format, base_file_names[b], width, height, subsamplings[b], image_save_path);
                rocjpeg_utils.SaveImage(image_save_path, &output_images[b], width, height, subsamplings[b], decode_params.output_format);
            }
        }

        total_decode_time_in_milli_sec += time_per_batch_in_milli_sec;
        image_size_in_mpixels_all += image_size_in_mpixels;

        current_batch_size = 0;
    }

    double avg_time_per_image = decode_info.num_decoded_images > 0 ? total_decode_time_in_milli_sec / decode_info.num_decoded_images : 0;
    decode_info.images_per_sec = avg_time_per_image > 0 ? 1000 / avg_time_per_image : 0;
    decode_info.image_size_in_mpixels_per_sec = decode_info.num_decoded_images > 0 ? decode_info.images_per_sec * image_size_in_mpixels_all / decode_info.num_decoded_images : 0;

}

int main(int argc, char **argv) {
    int device_id = 0;
    bool save_images = false;
    int num_threads = 1;
    int batch_size = 1;
    bool is_dir = false;
    bool is_file = false;
    RocJpegBackend rocjpeg_backend = ROCJPEG_BACKEND_HARDWARE;
    RocJpegDecodeParams decode_params = {};
    RocJpegUtils rocjpeg_utils;
    std::string input_path, output_file_path;
    std::vector<std::string> file_paths = {};
    std::vector<DecodeInfo> decode_info_per_thread;

    RocJpegUtils::ParseCommandLine(input_path, output_file_path, save_images, device_id, rocjpeg_backend, decode_params, &num_threads, &batch_size, argc, argv);
    if (!RocJpegUtils::GetFilePaths(input_path, file_paths, is_dir, is_file)) {
        std::cerr << "ERROR: Failed to get input file paths!" << std::endl;
        return EXIT_FAILURE;
    }
    if (!RocJpegUtils::InitHipDevice(device_id)) {
        std::cerr << "ERROR: Failed to initialize HIP!" << std::endl;
        return EXIT_FAILURE;
    }

    if (num_threads > file_paths.size()) {
        num_threads = file_paths.size();
    }

    decode_info_per_thread.resize(num_threads);

    for (int i = 0; i < num_threads; i++) {
        CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &decode_info_per_thread[i].rocjpeg_handle));
        decode_info_per_thread[i].rocjpeg_stream_handles.resize(batch_size);
        for (auto j = 0; j < batch_size; j++) {
            CHECK_ROCJPEG(rocJpegStreamCreate(&decode_info_per_thread[i].rocjpeg_stream_handles[j]));
        }
        decode_info_per_thread[i].num_decoded_images = 0;
        decode_info_per_thread[i].images_per_sec = 0;
        decode_info_per_thread[i].image_size_in_mpixels_per_sec = 0;
        decode_info_per_thread[i].num_bad_jpegs = 0;
        decode_info_per_thread[i].num_jpegs_with_411_subsampling = 0;
        decode_info_per_thread[i].num_jpegs_with_unknown_subsampling = 0;
        decode_info_per_thread[i].num_jpegs_with_unsupported_resolution = 0;
    }

    ThreadPool thread_pool(num_threads);

    size_t files_per_thread = file_paths.size() / num_threads;
    size_t remaining_files = file_paths.size() % num_threads;
    size_t start_index = 0;
    for (int i = 0; i < num_threads; i++) {
        size_t end_index = start_index + files_per_thread + (i < remaining_files ? 1 : 0);
        decode_info_per_thread[i].file_paths.assign(file_paths.begin() + start_index, file_paths.begin() + end_index);
        start_index = end_index;
    }

    std::cout << "Decoding started with " << num_threads << " threads, please wait!" << std::endl;
    auto overall_start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        thread_pool.ExecuteJob(std::bind(DecodeImages, std::ref(decode_info_per_thread[i]), rocjpeg_utils, std::ref(decode_params), save_images, std::ref(output_file_path), batch_size, device_id));
    }
    thread_pool.JoinThreads();
    auto overall_end_time = std::chrono::high_resolution_clock::now();
    double total_wall_time_in_sec = std::chrono::duration<double>(overall_end_time - overall_start_time).count();

    uint64_t total_decoded_images = 0;
    double total_images_per_sec = 0;
    double total_image_size_in_mpixels_per_sec = 0;
    uint64_t total_num_bad_jpegs = 0;
    uint64_t total_num_jpegs_with_411_subsampling = 0;
    uint64_t total_num_jpegs_with_unknown_subsampling = 0;
    uint64_t total_num_jpegs_with_unsupported_resolution = 0;

    for (auto i = 0; i < num_threads; i++) {
        total_decoded_images += decode_info_per_thread[i].num_decoded_images;
        total_image_size_in_mpixels_per_sec += decode_info_per_thread[i].image_size_in_mpixels_per_sec;
        total_images_per_sec += decode_info_per_thread[i].images_per_sec;
        total_num_bad_jpegs += decode_info_per_thread[i].num_bad_jpegs;
        total_num_jpegs_with_411_subsampling += decode_info_per_thread[i].num_jpegs_with_411_subsampling;
        total_num_jpegs_with_unknown_subsampling += decode_info_per_thread[i].num_jpegs_with_unknown_subsampling;
        total_num_jpegs_with_unsupported_resolution += decode_info_per_thread[i].num_jpegs_with_unsupported_resolution;
    }

    std::cout << "Total decoded images: " << total_decoded_images << std::endl;
    if (total_num_bad_jpegs || total_num_jpegs_with_411_subsampling || total_num_jpegs_with_unknown_subsampling || total_num_jpegs_with_unsupported_resolution) {
        std::cout << "Total skipped images: " << total_num_bad_jpegs + total_num_jpegs_with_411_subsampling + total_num_jpegs_with_unknown_subsampling + total_num_jpegs_with_unsupported_resolution;
        if (total_num_bad_jpegs) {
            std::cout << " ,total images that cannot be parsed: " << total_num_bad_jpegs;
        }
        if (total_num_jpegs_with_411_subsampling) {
            std::cout << " ,total images with YUV 4:1:1 chroma subsampling: " << total_num_jpegs_with_411_subsampling;
        }
        if (total_num_jpegs_with_unknown_subsampling) {
            std::cout << " ,total images with unknown chroma subsampling: " << total_num_jpegs_with_unknown_subsampling;
        }
        if (total_num_jpegs_with_unsupported_resolution) {
            std::cout << " ,total images with unsupported_resolution: " << total_num_jpegs_with_unsupported_resolution;
        }
        std::cout << std::endl;
    }

    if (total_decoded_images > 0) {
        std::cout << "Average processing time per image (ms): " << 1000 / total_images_per_sec << std::endl;
        std::cout << "Average decoded images per sec (Images/Sec): " << total_images_per_sec << std::endl;
        std::cout << "Average decoded images size (Mpixels/Sec): " << total_image_size_in_mpixels_per_sec << std::endl;
    }

    if (total_wall_time_in_sec >= 3600) {
        std::cout << "Total wall time (hours): " << total_wall_time_in_sec / 3600 << std::endl;
    } else if (total_wall_time_in_sec >= 60) {
        std::cout << "Total wall time (min): " << total_wall_time_in_sec / 60 << std::endl;
    } else {
        std::cout << "Total wall time (sec): " << total_wall_time_in_sec << std::endl;
    }

    for (int i = 0; i < num_threads; i++) {
        CHECK_ROCJPEG(rocJpegDestroy(decode_info_per_thread[i].rocjpeg_handle));
        for (auto j = 0; j < batch_size; j++) {
            CHECK_ROCJPEG(rocJpegStreamDestroy(decode_info_per_thread[i].rocjpeg_stream_handles[j]));
        }
    }

    std::cout << "Decoding completed!" << std::endl;
    return EXIT_SUCCESS;
}
