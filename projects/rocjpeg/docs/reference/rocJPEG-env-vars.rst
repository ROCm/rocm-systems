.. meta::
    :description: rocJPEG environment variables
    :keywords: rocJPEG, environment variables, ROCm, AMD

********************************************************************
rocJPEG environment variables
********************************************************************

The following environment variables affect the rocJPEG runtime.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Environment variable
     - Values 
   * - ``HIP_VISIBLE_DEVICES``
     - | Comma-separated list of GPU indices. Parsed when ``ROCR_VISIBLE_DEVICES`` isn't set.
   * - ``LIBVA_DRIVERS_PATH``
     - | Optional. Path to the directory containing the VA-API driver.
       | Only required if the driver you want to use is neither the one that ships with ROCm nor the distribution-provided one. ROCm's libva finds its driver under ``ROCM_PATH`` on its own: ``%ROCM_PATH%\lib\rocm_sysdeps\bin`` on Windows, ``$ROCM_PATH/lib/rocm_sysdeps/lib`` on Linux.
       | This replaces libva's default search directory rather than adding to it, so an incorrect value makes VA-API initialization fail.
   * - ``ROCJPEG_ENABLE_VCN_HW_CSC``
     -  | ``1``: Enables hardware color-space conversion when there are eight or more JPEG cores and RGB or RGB planar output is in use.
        | Hardware color-space conversion is off by default regardless of the number of cores. 
        | Not used for ``CSS_440`` (4:4:0) chroma subsampling even when enabled.
        | Use the ``hw_rgb_subsampling_mask`` and ``hw_rgb_output_format_mask`` fields reported by ``rocJpegGetDecoderCaps()`` to check whether it is in effect.
   * - ``ROCJPEG_LOG_LEVEL``
     - | Sets the maximum severity of log messages. Each level defines the maximum severity of log messages to output.
       | ``0``: Output critical messages only (default) 
       | ``1``: Output critical and error messages
       | ``2``: Output critical, error, and warning messages
       | ``3``: Output critical, error, warning, and info messages
       | ``4``: Output critical, error, warning, info, and debug messages 
   * - ``ROCM_PATH``
     - | Path to the ROCm installation.
       | rocJPEG loads ROCm's VA-API libraries from ``lib/rocm_sysdeps`` under the installation prefix of the loaded rocJPEG library (``lib\rocm_sysdeps\bin`` on Windows, ``lib/rocm_sysdeps/lib`` on Linux), and falls back to ``ROCM_PATH`` when they aren't found there. On Windows, absolute ``PATH`` entries are searched last.
       | On Windows, keep ``ROCM_PATH`` set at run time. libva uses it to locate the VA-API driver.
   * - ``ROCR_VISIBLE_DEVICES``
     - | Takes a comma-separated list of GPU indices. 
       | When set, the VAAPI decoder path parses this list before it falls back to ``HIP_VISIBLE_DEVICES``. 

