.. meta::
  :description: rocJPEG chroma subsampling and hardware capabilities
  :keywords: install, rocJPEG, AMD, ROCm, GPU, chroma, subsampling, VCN

********************************************************************
rocJPEG chroma subsampling and hardware capabilities
********************************************************************

rocJPEG supports the following chroma subsamplings:

* YUV 4:4:4
* YUV 4:4:0
* YUV 4:2:2
* YUV 4:2:0
* YUV 4:0:0

The following table shows the capabilities of the VCN and total number of JPEG cores for each supported GPU
architecture:

.. csv-table::
  :header: "GPU Architecture", "VCN Generation", "Total number of JPEG cores", "Max width, Max height"

  "gfx908 - MI1xx", "VCN 2.5.0", "2", "4096, 4096"
  "gfx90a - MI2xx", "VCN 2.6.0", "4", "4096, 4096"
  "gfx942 - MI300A", "VCN 3.0", "24", "16384, 16384"
  "gfx942 - MI300X", "VCN 3.0", "32", "16384, 16384"
  "gfx1030, gfx1031, gfx1032 - Navi2x", "VCN 3.x", "1", "16384, 16384"
  "gfx1100, gfx1101, gfx1102 - Navi3x", "VCN 4.0", "1", "16384, 16384"