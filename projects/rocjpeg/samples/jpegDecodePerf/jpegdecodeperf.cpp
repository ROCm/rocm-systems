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

// Number of batches each worker can have in flight, submitted with rocJpegDecodeBatchedAsync() and not yet
// synchronized with rocJpegDecodeBatchedSync(). Each in-flight batch needs its own GPU output buffers. More batches in
// flight keep the decoder busy across longer read, parse, and synchronize times, as with large batches on GPUs with
// many JPEG cores, at the cost of GPU memory.
static constexpr int kNumBatchSlots = 4;

struct DecodeInfo {
    std::vector<std::string> file_paths;
    RocJpegHandle rocjpeg_handle;
    std::vector<RocJpegStreamHandle> rocjpeg_stream_handles;
    std::vector<RocJpegImage> output_images; // kNumBatchSlots * batch_size, batch_size per slot
    std::vector<std::vector<uint32_t>> allocated_channel_sizes;
    uint64_t num_decoded_images;
    double images_per_sec;
    double image_size_in_mpixels_per_sec;
    uint64_t num_bad_jpegs;
    uint64_t num_jpegs_with_411_subsampling;
    uint64_t num_jpegs_with_unknown_subsampling;
    uint64_t num_jpegs_with_unsupported_resolution;
};

/**
 * @brief A batch submitted with rocJpegDecodeBatchedAsync() and waiting for rocJpegDecodeBatchedSync().
 *
 * The stream handles and file data are not part of the slot: rocJpegDecodeBatchedAsync() no longer uses them
 * once it returns, so the next batch can be parsed into them right away.
 */
struct BatchSlot {
    std::vector<RocJpegDecodeParams> decode_params;
    std::vector<RocJpegImage> images; // must keep the same addresses from the submit to the sync
    std::vector<std::vector<uint32_t>> widths;
    std::vector<std::vector<uint32_t>> heights;
    std::vector<RocJpegChromaSubsampling> subsamplings;
    std::vector<std::string> base_file_names;
    int current_batch_size = 0;
    std::chrono::high_resolution_clock::time_point submit_time; // just before rocJpegDecodeBatchedAsync()
};

/**
 * @brief A ring of batch slots shared by the submit loop and the sync thread of one worker.
 *
 * The submit loop takes a free slot, fills and submits it, and marks it ready; the sync thread takes ready slots in
 * order and hands them back once synchronized. The submit loop waits only when no slot is free, that is, when every
 * slot is in flight.
 */
class BatchRing {
    public:
        explicit BatchRing(int num_slots) {
            for (int i = 0; i < num_slots; i++) {
                free_slots_.push(i);
            }
        }

        // Called by the submit loop; waits until a slot is free.
        int AcquireFreeSlot() {
            std::unique_lock<std::mutex> lock(mutex_);
            free_cv_.wait(lock, [&] { return !free_slots_.empty(); });
            int slot = free_slots_.front();
            free_slots_.pop();
            return slot;
        }

        void PushReadySlot(int slot) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ready_slots_.push(slot);
            }
            ready_cv_.notify_one();
        }

        // Called by the submit loop once it has no more batches.
        void Finish() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                finished_ = true;
            }
            ready_cv_.notify_one();
        }

        // Called by the sync thread; returns -1 once the submit loop has finished and every ready slot is taken.
        int PopReadySlot() {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_cv_.wait(lock, [&] { return !ready_slots_.empty() || finished_; });
            if (ready_slots_.empty()) {
                return -1;
            }
            int slot = ready_slots_.front();
            ready_slots_.pop();
            return slot;
        }

        void ReleaseSlot(int slot) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                free_slots_.push(slot);
            }
            free_cv_.notify_one();
        }

    private:
        std::mutex mutex_;
        std::condition_variable free_cv_;
        std::condition_variable ready_cv_;
        std::queue<int> free_slots_;
        std::queue<int> ready_slots_;
        bool finished_ = false;
};

/**
 * @brief Synchronizes the submitted batches of one worker until its submit loop finishes, and optionally saves the decoded images.
 *
 * Runs on its own thread so that the submit loop can read, parse, and submit the next batch while this thread waits
 * for the previous one with rocJpegDecodeBatchedSync().
 */
void SyncBatches(DecodeInfo &decode_info, std::vector<BatchSlot> &slots, BatchRing &ring, RocJpegUtils &rocjpeg_utils,
                 RocJpegOutputFormat output_format, bool save_images, const std::string &output_file_path, int device_id) {
    double image_size_in_mpixels_all = 0;
    double total_decode_time_in_milli_sec = 0;
    std::chrono::high_resolution_clock::time_point last_sync_end_time;

    CHECK_HIP(hipSetDevice(device_id));

    for (int slot_index = ring.PopReadySlot(); slot_index >= 0; slot_index = ring.PopReadySlot()) {
        BatchSlot &slot = slots[slot_index];

        CHECK_ROCJPEG(rocJpegDecodeBatchedSync(decode_info.rocjpeg_handle, slot.images.data(), slot.current_batch_size));
        auto sync_end_time = std::chrono::high_resolution_clock::now();
        // Decode calls overlap, so count only the time during which at least one batch is in flight: from this
        // batch's submit, or from the previous sync if that ended later, to this sync. Gaps in which the decoder
        // waits for the next batch to be read and parsed are not counted.
        auto busy_start_time = std::max(slot.submit_time, last_sync_end_time);
        total_decode_time_in_milli_sec += std::chrono::duration<double, std::milli>(sync_end_time - busy_start_time).count();
        last_sync_end_time = sync_end_time;

        for (int b = 0; b < slot.current_batch_size; b++) {
            image_size_in_mpixels_all += (static_cast<double>(slot.widths[b][0]) * static_cast<double>(slot.heights[b][0]) / 1000000);
        }
        decode_info.num_decoded_images += slot.current_batch_size;

        if (save_images) {
            for (int b = 0; b < slot.current_batch_size; b++) {
                std::string image_save_path = output_file_path;
                //if ROI is present, need to pass roi_width and roi_height
                uint32_t roi_width = slot.decode_params[b].crop_rectangle.right - slot.decode_params[b].crop_rectangle.left;
                uint32_t roi_height = slot.decode_params[b].crop_rectangle.bottom - slot.decode_params[b].crop_rectangle.top;
                bool is_roi_valid = (roi_width > 0 && roi_height > 0 && roi_width <= slot.widths[b][0] && roi_height <= slot.heights[b][0]) ? true : false;
                uint32_t width = is_roi_valid ? roi_width : slot.widths[b][0];
                uint32_t height = is_roi_valid ? roi_height : slot.heights[b][0];
                rocjpeg_utils.GetOutputFileExt(output_format, slot.base_file_names[b], width, height, slot.subsamplings[b], image_save_path);
                rocjpeg_utils.SaveImage(image_save_path, &slot.images[b], width, height, slot.subsamplings[b], output_format);
            }
        }

        ring.ReleaseSlot(slot_index);
    }

    double avg_time_per_image = decode_info.num_decoded_images > 0 ? total_decode_time_in_milli_sec / decode_info.num_decoded_images : 0;
    decode_info.images_per_sec = avg_time_per_image > 0 ? 1000 / avg_time_per_image : 0;
    decode_info.image_size_in_mpixels_per_sec = decode_info.num_decoded_images > 0 ? decode_info.images_per_sec * image_size_in_mpixels_all / decode_info.num_decoded_images : 0;
}

/**
 * @brief Reads, parses, and decodes the JPEG images of one worker, and optionally saves the decoded images.
 *
 * This thread reads and parses each batch and submits it with rocJpegDecodeBatchedAsync(), while a sync thread it
 * starts waits for the submitted batches with rocJpegDecodeBatchedSync(). The file I/O and parsing of a batch therefore
 * overlap with the hardware decode of the previous batch, and the output copy or color conversion of a batch overlaps
 * with the hardware decode of the next.
 *
 * @param decode_info parameters info for decoding a batch of jpeg images.
 * @param rocjpeg_utils Utility functions for RocJpeg operations.
 * @param decode_params Parameters for decoding the JPEG images (output_format, crop_rectangle)
 * @param save_images A boolean flag indicating whether to save the decoded images.
 * @param output_file_path The file path where the decoded images will be saved.
 * @param batch_size The number of images to be processed in each batch.
 */
void DecodeImages(DecodeInfo &decode_info, RocJpegUtils rocjpeg_utils, RocJpegDecodeParams &decode_params, bool save_images, std::string &output_file_path, int batch_size, int device_id) {

    uint8_t num_components;
    uint32_t channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    std::string chroma_sub_sampling = "";
    uint32_t num_channels = 0;
    std::vector<std::vector<char>> batch_images(batch_size);
    std::vector<uint32_t> temp_widths(ROCJPEG_MAX_COMPONENT, 0);
    std::vector<uint32_t> temp_heights(ROCJPEG_MAX_COMPONENT, 0);
    RocJpegChromaSubsampling temp_subsampling;

    CHECK_HIP(hipSetDevice(device_id));

    auto& output_images = decode_info.output_images;
    auto& allocated_channel_sizes = decode_info.allocated_channel_sizes;

    std::vector<BatchSlot> slots(kNumBatchSlots);
    for (auto &slot : slots) {
        slot.decode_params.resize(batch_size, decode_params);
        slot.images.resize(batch_size);
        slot.widths.assign(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
        slot.heights.assign(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
        slot.subsamplings.resize(batch_size);
        slot.base_file_names.resize(batch_size);
    }
    BatchRing ring(kNumBatchSlots);
    std::thread sync_thread(SyncBatches, std::ref(decode_info), std::ref(slots), std::ref(ring), std::ref(rocjpeg_utils),
                            decode_params.output_format, save_images, std::cref(output_file_path), device_id);

    auto submit_batch = [&](int slot_index) {
        BatchSlot &slot = slots[slot_index];
        slot.submit_time = std::chrono::high_resolution_clock::now();
        CHECK_ROCJPEG(rocJpegDecodeBatchedAsync(decode_info.rocjpeg_handle, decode_info.rocjpeg_stream_handles.data(), slot.current_batch_size, slot.decode_params.data(), slot.images.data()));
        ring.PushReadySlot(slot_index);
    };

    int slot_index = -1;
    for (size_t i = 0; i < decode_info.file_paths.size(); i++) {
        if (slot_index < 0) {
            slot_index = ring.AcquireFreeSlot();
            slots[slot_index].current_batch_size = 0;
        }
        BatchSlot &slot = slots[slot_index];
        int index = slot.current_batch_size;
        const std::string &file_path = decode_info.file_paths[i];

        // Read an image from disk.
        std::ifstream input(file_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        if (!(input.is_open())) {
            std::cerr << "ERROR: Cannot open image: " << file_path << std::endl;
            break;
        }
        // Get the size
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if (batch_images[index].size() < file_size) {
            batch_images[index].resize(file_size);
        }
        if (!input.read(batch_images[index].data(), file_size)) {
            std::cerr << "ERROR: Cannot read from file: " << file_path << std::endl;
            break;
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

        slot.decode_params[index] = RocJpegUtils::FitCropToImage(decode_params, temp_widths[0], temp_heights[0], false);
        slot.images[index] = {};
        if (rocjpeg_utils.GetChannelPitchAndSizes(slot.decode_params[index], temp_subsampling, temp_widths.data(), temp_heights.data(), num_channels, slot.images[index], channel_sizes)) {
            std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
            break;
        }

        // Point the image at this slot's output buffers. The slot is not in flight, so a buffer can be grown
        // when the image needs more than its current allocation.
        int output_index = slot_index * batch_size + index;
        for (uint32_t n = 0; n < num_channels; n++) {
            if (channel_sizes[n] > allocated_channel_sizes[output_index][n]) {
                if (output_images[output_index].channel[n] != nullptr) {
                    CHECK_HIP(hipFree((void *)output_images[output_index].channel[n]));
                    output_images[output_index].channel[n] = nullptr;
                }
                CHECK_HIP(hipMalloc(&output_images[output_index].channel[n], channel_sizes[n]));
                allocated_channel_sizes[output_index][n] = channel_sizes[n];
            }
            slot.images[index].channel[n] = output_images[output_index].channel[n];
        }

        slot.subsamplings[index] = temp_subsampling;
        slot.widths[index] = temp_widths;
        slot.heights[index] = temp_heights;
        slot.base_file_names[index] = file_path.substr(file_path.find_last_of("/\\") + 1);
        slot.current_batch_size++;

        if (slot.current_batch_size == batch_size) {
            submit_batch(slot_index);
            slot_index = -1;
        }
    }

    // Submit the last partial batch, if any.
    if (slot_index >= 0) {
        if (slots[slot_index].current_batch_size > 0) {
            submit_batch(slot_index);
        } else {
            ring.ReleaseSlot(slot_index);
        }
    }
    ring.Finish();
    sync_thread.join();
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
    if (batch_size <= 0) {
        std::cerr << "ERROR: The batch size must be greater than 0!" << std::endl;
        return EXIT_FAILURE;
    }
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

    // Pre-allocate GPU output buffers at 2560x1440 per image of each batch slot to avoid hipMalloc/hipFree during
    // decoding. Sizes mirror GetChannelPitchAndSizes logic: only channels required by the selected
    // output_format are allocated. For NATIVE/YUV_PLANAR the subsampling is unknown upfront,
    // so worst-case sizes are used per channel (ch0: packed 4:2:2 = 2w; ch1/2: 4:4:4 full height).
    static constexpr uint32_t kMemAlignment = 16;
    static constexpr uint32_t kPreAllocWidth = 2560;
    static constexpr uint32_t kPreAllocHeight = 1440;
    auto align_up = [](uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); };
    const uint32_t aligned_w  = align_up(kPreAllocWidth, kMemAlignment);
    const uint32_t aligned_w2 = align_up(kPreAllocWidth, kMemAlignment) * 2;
    const uint32_t aligned_w3 = align_up(kPreAllocWidth, kMemAlignment) * 3;
    const uint32_t aligned_h  = align_up(kPreAllocHeight, kMemAlignment);
    uint32_t kPreAllocSizes[ROCJPEG_MAX_COMPONENT] = {};
    switch (decode_params.output_format) {
        case ROCJPEG_OUTPUT_Y:
            kPreAllocSizes[0] = aligned_w * aligned_h;
            break;
        case ROCJPEG_OUTPUT_RGB:
            kPreAllocSizes[0] = aligned_w3 * aligned_h;
            break;
        case ROCJPEG_OUTPUT_RGB_PLANAR:
            kPreAllocSizes[0] = kPreAllocSizes[1] = kPreAllocSizes[2] = aligned_w * aligned_h;
            break;
        case ROCJPEG_OUTPUT_YUV_PLANAR:
            kPreAllocSizes[0] = kPreAllocSizes[1] = kPreAllocSizes[2] = aligned_w * aligned_h;
            break;
        case ROCJPEG_OUTPUT_NATIVE:
        default:
            kPreAllocSizes[0] = aligned_w2 * aligned_h;
            kPreAllocSizes[1] = kPreAllocSizes[2] = aligned_w * aligned_h;
            break;
    }

    for (int i = 0; i < num_threads; i++) {
        CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &decode_info_per_thread[i].rocjpeg_handle));
        decode_info_per_thread[i].rocjpeg_stream_handles.resize(batch_size);
        for (auto &stream_handle : decode_info_per_thread[i].rocjpeg_stream_handles) {
            CHECK_ROCJPEG(rocJpegStreamCreate(&stream_handle));
        }
        // Each in-flight batch slot of the worker gets its own set of output buffers.
        decode_info_per_thread[i].output_images.resize(kNumBatchSlots * batch_size);
        decode_info_per_thread[i].allocated_channel_sizes.assign(kNumBatchSlots * batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
        for (int b = 0; b < kNumBatchSlots * batch_size; b++) {
            for (int n = 0; n < ROCJPEG_MAX_COMPONENT; n++) {
                if (kPreAllocSizes[n] > 0) {
                    CHECK_HIP(hipMalloc(&decode_info_per_thread[i].output_images[b].channel[n], kPreAllocSizes[n]));
                    decode_info_per_thread[i].allocated_channel_sizes[b][n] = kPreAllocSizes[n];
                }
            }
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

    for (int i = 0; i < num_threads; i++) {
        for (auto& img : decode_info_per_thread[i].output_images) {
            for (int n = 0; n < ROCJPEG_MAX_COMPONENT; n++) {
                if (img.channel[n] != nullptr) {
                    CHECK_HIP(hipFree((void*)img.channel[n]));
                    img.channel[n] = nullptr;
                }
            }
        }
    }
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
        // Unlike the averages above, which count only the time during which a batch is being decoded, this includes
        // the time the decoder waits for file I/O and parsing.
        std::cout << "Overall throughput including file I/O and parsing (Images/Sec): " << total_decoded_images / total_wall_time_in_sec << std::endl;
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
        for (auto &stream_handle : decode_info_per_thread[i].rocjpeg_stream_handles) {
            CHECK_ROCJPEG(rocJpegStreamDestroy(stream_handle));
        }
    }

    std::cout << "Decoding completed!" << std::endl;
    return EXIT_SUCCESS;
}
