.. meta::
  :description: Using the rocDecode core API
  :keywords: rocDecode, AMD, ROCm, core API

********************************************************************
Using the rocdecdecode example
********************************************************************

rocDecode provides four core APIs exposed in the header files in the |apifolder|_ directory:

| The rocDecode parser API, exposed in ``rocparser.h``. 
| The hardware decoder API, exposed in ``rocdecode.h``.
| The software decoder API, exposed in ``rocdecode_host.h``.
| The bitstream reader API, exposed in ``roc_bitstream_reader.h``.

The |rocdecdecode|_ sample demonstrates how to use the rocDecode core APIs in an application. It shows how to use the parser and both the hardware and software decoders. For information on how to use the bitstream reader API, see :doc:`Using the rocDecode bitstream reader API <./using-rocDecode-bitstream>`.

The sample decodes raw elementary video frame files as input and produces individually decoded frames in YUV format as output. The input can be one individual frame file or multiple frames from one or more video files. The individual frame files must be numbered in ascending order of frames.

``rocdecdecode.cpp`` takes the following arguments:

.. list-table:: 
    :widths: 10 60 30 
    :header-rows: 1

    * - Argument      
      - Description 
      - Note

    * - ``-i``
      - Path to the input video frame file or to frame folder.
      - Required. 
    
    * - ``-o`` 
      - Output path. Saves the decoded YUV frames to this folder. 
      - Optional. Decoded frames aren't saved by default.

    * - ``-d`` 
      - GPU device ID. Set it to 0 for the first device, 1 for the second device, 2 for the third device, and so on for each subsequent device. 
      - Optional. Set to 0 by default.
  
    * - ``-b`` 
      - Backend. Set it to 0 to use the hardware decoder on the GPU or to 1 to use the software decoder on the CPU.
      - Optional. Set to 0 by default.


    * - ``-c``
      - Codec. Set to 0 for HEVC, 1 for H264, 2 for AV1, 4 for VP9, 5 for VP8, or 6 for MJPEG.
      - Optional. Set to 0 by default.
      
    * - ``-n`` 
      - Number of iterations for performance evaluation.
      - Optional. Set to 1 by default.

    * - ``-m`` 
      - The output surface memory type. The memory type where the surface data, such as the decoded frames, resides. Set this to 0 for intermediate GPU memory, to 1 for GPU memory, and to 2 for CPU memory. See :doc:`Surface data memory locations <../conceptual/rocDecode-memory-types>` for more information. 
      - Optional. Set to 0 by default. 

The ``DecoderInfo`` struct defined in the sample is used to store user-supplied parameters as well as the decoder and parser handles. 

The memory type and the type of decoder is set by the specified backend. If the GPU (device) backend is selected, both a parser and a hardware decoder are created. If the CPU (host) backend is selected, only a software decoder is created:

.. code:: cpp
  
  DecoderInfo dec_info;
  [...]
  int main(int argc, char** argv) {
    [...]
    dec_info.rocdec_codec_id = CodecTypeToRocDecVideoCodec(codec_type);
    dec_info.dec_device_id = device_id;
    dec_info.mem_type = (!backend) ? OUT_SURFACE_MEM_DEV_INTERNAL : OUT_SURFACE_MEM_HOST;
    init();
    if (backend == DECODER_BACKEND_DEVICE) {
      create_parser(dec_info);
      create_decoder(dec_info);
    } else {
      create_decoder_host(dec_info);
    }
  [...]
  }

All applications need to register the ``pfn_sequence_callback`` and ``pfn_display_picture`` callbacks. Applications that use the parser must also register the ``pfn_decode_picture`` callback.

When the GPU backend is selected, these callbacks are registered in the ``create_parser()`` function. ``create_parser`` also creates the parser using ``rocDecCreateVideoParser()``:

.. code:: cpp

  void create_parser(DecoderInfo& dec_info) {
    RocdecParserParams params = {};
    params.codec_type = dec_info.rocdec_codec_id;
    params.max_num_decode_surfaces = 6;
    params.max_display_delay = 1;      
    params.user_data = &dec_info;
    params.pfn_sequence_callback = handle_video_sequence;
    params.pfn_decode_picture = handle_picture_decode;
    params.pfn_display_picture = handle_picture_display;
    CHECK(rocDecCreateVideoParser(&dec_info.parser, &params));
  }

The ``create_decoder()`` function sets the decoder parameters and passes them to ``rocDecCreateDecoder()`` to create the hardware decoder:

.. code:: cpp
  
  void create_decoder(DecoderInfo& dec_info) {
    RocDecoderCreateInfo create_info = {};
    create_info.codec_type = dec_info.rocdec_codec_id;     // user specified codec_type for raw files
    [...]
    CHECK(rocDecCreateDecoder(&dec_info.decoder, &create_info));
  }

The ``create_decoder_host()`` function performs the same actions as ``create_decoder()``, but uses ``rocDecCreateDecoderHost()`` to create a software decoder. Because the parser isn't used with the software decoder, and because the software decoder uses different function calls, the callbacks for the software decoder are registered in ``create_decoder_host()``:

.. code:: cpp

  void create_decoder_host(DecoderInfo& dec_info) {
    RocDecoderHostCreateInfo create_info = {};
    create_info.codec_type = dec_info.rocdec_codec_id;
    [...]
    create_info.pfn_sequence_callback = handle_video_sequence_host;
    create_info.pfn_display_picture = handle_picture_display_host;
    CHECK(rocDecCreateDecoderHost(&dec_info.decoder, &create_info));
    dec_info.backend = DECODER_BACKEND_HOST;
  }

After the decoder and parser have been created, ``decode_frames`` is called. 

.. code:: cpp

  int main(int argc, char** argv) {
    [...]
    dec_info.dump_decoded_frames = dump_output_frames;
    auto input_frames = read_frames(input_file_names);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
      decode_frames(dec_info, input_frames);
    }
    [...]
  }

``decode_frames`` calls ``rocDecParseVideoData()`` or ``rocDecDecodeFrameHost()``, depending on the backend, to parse and decode the frames:

.. code:: cpp

  void decode_frames(DecoderInfo& dec_info, const std::vector<std::vector<uint8_t>>& frames) {
    // gpu backend using VCN
    if (dec_info.backend == DECODER_BACKEND_DEVICE) {
        for (int i=0; i < static_cast<int>(frames.size()); ++i) {
            RocdecSourceDataPacket packet = {};
            packet.payload_size = frames[i].size();
            packet.payload = frames[i].data();
            if (i == static_cast<int>(frames.size() - 1)) {
                packet.flags = ROCDEC_PKT_ENDOFPICTURE;     // mark end_of_picture flag for last frame
            }
            CHECK(rocDecParseVideoData(dec_info.parser, &packet));
        }
    } else if (dec_info.backend == DECODER_BACKEND_HOST) {
        for (int i=0; i < static_cast<int>(frames.size()); ++i) {
            RocdecPicParamsHost pic_params = {};
            pic_params.bitstream_data_len = frames[i].size();
            pic_params.bitstream_data = frames[i].data();
            if (i == static_cast<int>(frames.size() - 1)) {
                pic_params.flags = ROCDEC_PKT_ENDOFPICTURE;     // mark end_of_picture flag for last frame
            }
            CHECK(rocDecDecodeFrameHost(dec_info.decoder, &pic_params));
        }   
    }
  }

The registered callbacks are triggered during the calls to ``rocDecParseVideoData()`` and ``rocDecDecodeFrameHost()``. 

``pfn_decode_picture`` is triggered when a new frame is ready to be decoded, ``pfn_sequence_callback`` is triggered when a new sequence header is encountered, and ``pfn_display_picture`` is triggered when a frame has finished being decoded. 

``pfn_decode_picture`` needs to call ``rocDecDecodeFrame()`` or ``rocDecodeFrameHost()``, depending on the specified backend, to decode a frame. 

In ``rocdecdecode.cpp``, ``pfn_decode_picture`` calls ``handle_picture_decode()`` or ``handle_picture_decode_host()``, depending on the specified backend:

.. code:: cpp

  int ROCDECAPI handle_picture_decode(void* user_data, RocdecPicParams* params) {
    DecoderInfo *p_dec_info = static_cast<DecoderInfo *>(user_data);
    CHECK(rocDecDecodeFrame(p_dec_info->decoder, params));
    return 1;
  }

``pfn_sequence_callback`` is triggered when a format change occurs or when a new sequence header is encountered. When this happens, the decoder is reconfigured to handle the new sequence or format. 

``pfn_sequence_callback`` needs to call ``rocDecReconfigureDecoder()`` or ``rocDecReconfigureDecoderHost()`` depending on the backend, to reconfigure the decoder. 

In the ``rocdecdecode.cpp`` sample, ``pfn_sequence_callback`` calls ``handle_video_sequence()`` or ``handle_video_sequence_host()``, depending on the specified backend:

.. code:: cpp

  int ROCDECAPI handle_video_sequence(void* user_data, RocdecVideoFormat* format) {
    DecoderInfo *p_dec_info = static_cast<DecoderInfo *>(user_data);
    [...]
    RocdecReconfigureDecoderInfo reconfig_params = {};
    reconfig_params.width = format->coded_width;
    reconfig_params.height = format->coded_height;
    reconfig_params.bit_depth_minus_8 = bitdepth_minus_8;
    reconfig_params.num_decode_surfaces = format->min_num_decode_surfaces;
    reconfig_params.target_width = target_width;
    reconfig_params.target_height = target_height;
    reconfig_params.display_rect.left = format->display_area.left;
    reconfig_params.display_rect.right = format->display_area.right;
    reconfig_params.display_rect.top = format->display_area.top;
    reconfig_params.display_rect.bottom = format->display_area.bottom;        
    CHECK(rocDecReconfigureDecoder(p_dec_info->decoder, &reconfig_params));
    [...]
    return 1;
  }

``pfn_display_picture`` is triggered when a frame has been decoded. It needs to call ``rocDecGetVideoFrame()`` or ``rocDecGetVideoFrameHost()``, depending on the specified backend. 

``rocDecGetVideoFrame()`` and ``rocDecGetVideoFrameHost()`` map the video ID of the decoded frame to HIP. Calls to both these functions block until the frame is decoded and the memory mapping is complete. They return the HIP device pointer or the host memory pointer, depending on the backend specified, as well as information about the :doc:`output surface <../conceptual/rocDecode-memory-types>`. 

``pfn_display_picture`` calls ``handle_picture_display()`` or ``handle_handle_picture_display_host()``, depending on the specified backend, and saves the frames to file if the ``rocdecdecode`` was run with the ``-o`` option:

From the ``rocdecdecode.cpp`` sample:

.. code:: cpp

  int ROCDECAPI handle_picture_display(void* user_data, RocdecParserDispInfo* disp_info) {
    DecoderInfo *p_dec_info = static_cast<DecoderInfo *>(user_data);
    RocdecProcParams params = {};
    params.progressive_frame = disp_info->progressive_frame;
    params.top_field_first = disp_info->top_field_first;
    void* dev_mem_ptr[3] = { 0 };
    uint32_t pitch[3] = { 0 };
    CHECK(rocDecGetVideoFrame(p_dec_info->decoder, disp_info->picture_index, dev_mem_ptr, pitch, &params));
      if (p_dec_info->dump_decoded_frames) {
        save_frame_to_file(p_dec_info, dev_mem_ptr, pitch);
      }
    return 1;
  }

Once decoding is complete, ``rocDecDestroyVideoParser()`` needs to be called to destroy the parser, and either ``rocDecDestroyDecoderHost()`` or ``rocDecDestroyDecoder()`` needs to be called to destroy the decoder.

.. |rocdecdecode| replace:: ``rocdecdecode``
.. _rocdecdecode: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/samples/rocdecDecode/README.md

.. |apifolder| replace:: ``api/rocdecode/``
.. _apifolder: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/api/rocdecode

.. |utilsfolder| replace:: ``utils`` folder
.. _utilsfolder: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/utils
