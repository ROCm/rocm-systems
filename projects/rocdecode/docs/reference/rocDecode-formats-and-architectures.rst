.. meta::
  :description: rocDecode supported codex and hardware capabilities
  :keywords: install, rocDecode, AMD, ROCm, GPU, codec, VCN

********************************************************************
rocDecode supported codecs and hardware capabilities
********************************************************************

rocDecode supports the following codecs:

* H.265 (HEVC): 8 bit and 10 bit
* H.264 (AVC): 8 bit
* AV1: 8 bit and 10 bit
* VP9: 8 bit and 10 bit

The following table shows the codec support and capabilities of the VCN for each supported GPU
architecture:

.. csv-table::
  :header: "GPU Architecture", "VCN Generation", "Number of VCNs", "H.265/HEVC", "Max width, Max height - H.265", "H.264/AVC", "Max width, Max height - H.264", "AV1", "Max width, Max height - AV1", "VP9", "Max width, Max height - VP9"

  "gfx908 - MI1xx", "VCN 2.5.0", "2", "Yes", "7680, 4320", "Yes", "4096, 2160", "No", "N/A, N/A", "Yes", "7680, 4320"
  "gfx90a - MI2xx", "VCN 2.6.0", "2", "Yes", "7680, 4320", "Yes", "4096, 2160", "No", "N/A, N/A", "Yes", "7680, 4320"
  "gfx942 - MI3xx", "VCN 4.0", "3/4", "Yes", "7680, 4320", "Yes", "4096, 2176", "Yes", "8192, 4352", "Yes", "7680, 4320"
  "gfx1030, gfx1031, gfx1032 - Navi2x", "VCN 3.x", "2", "Yes", "7680, 4320", "Yes", "4096, 2176", "Yes", "8192, 4352", "Yes", "7680, 4320"
  "gfx1100, gfx1102 - Navi3x", "VCN 4.0", "2", "Yes", "7680, 4320", "Yes", "4096, 2176", "Yes", "8192, 4352", "Yes", "7680, 4320"
  "gfx1101 - Navi3x", "VCN 4.0", "1", "Yes", "7680, 4320", "Yes", "4096, 2176", "Yes", "8192, 4352", "Yes", "7680, 4320"