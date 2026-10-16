.. meta::
  :description: What is rocJPEG?
  :keywords: video decoding, rocJPEG, AMD, ROCm

********************************************************************
What is rocJPEG?
********************************************************************

AMD GPUs contain one or more media engines (VCNs). Each VCN has one or more JPEG engines 
that provide fully accelerated, hardware-based JPEG decoding. Hardware decoders consume lower power
than CPU-based decoders. Dedicated hardware decoders offload decoding tasks from the CPU, boosting
overall decoding throughput. With proper power management, decoding on hardware decoders can lower the
overall system power consumption and improve decoding performance.

Using the rocJPEG API, you can decode compressed JPEG streams while keeping the resulting YUV
images in video memory. With decoded images in video memory, you can run image post-processing
using ROCm HIP, thereby avoiding unnecessary data copies via PCIe bus. You can post-process images
using scaling or color space conversion and augmentation kernels (on a GPU or host) in a format for
GPU/CPU-accelerated inferencing and training.

In addition, you can use the rocJPEG API to create multiple instances of JPEG decoder based on the
number of available VCNs/JPEG engines in a GPU device. By configuring the decoder for a device, all available
VCNs can be used seamlessly for decoding a batch of JPEG streams in parallel.

rocJPEG accesses the JPEG engines through the Video Acceleration API (VA-API). rocJPEG is supported on
Linux and Windows 11. On Windows, VA-API is provided through the vaon12 backend (Mesa's VA-API on D3D12
translation layer), which translates VA-API calls to the D3D12 video API. The vaon12 backend and libva
are provided by TheRock for Windows under ``%ROCM_PATH%\lib\rocm_sysdeps``.

.. note::

  Hardware JPEG decoding is not yet available on Windows. The vaon12 driver doesn't currently expose
  a JPEG decode profile, so ``rocJpegCreate()`` with ``ROCJPEG_BACKEND_HARDWARE`` returns
  ``ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED``. JPEG stream parsing with ``rocJpegStreamCreate()``
  and ``rocJpegStreamParse()`` is available.