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
   * - ``ROCJPEG_ENABLE_VCN_HW_CSC``
     -  | ``1``: Enables hardware color-space conversion when there are eight or more JPEG cores and RGB or RGB planar output is in use.
        | Hardware color-space conversion is off by default regardless of the number of cores. 
        | Not used for ``CSS_440`` (4:4:0) chroma subsampling even when enabled.
   * - ``ROCR_VISIBLE_DEVICES``
     - | Takes a comma-separated list of GPU indices. 
       | When set, the VAAPI decoder path parses this list before it falls back to ``HIP_VISIBLE_DEVICES``. 
   * - ``HIP_VISIBLE_DEVICES``
     - | Comma-separated list of GPU indices. Parsed when ``ROCR_VISIBLE_DEVICES`` isn't set.
