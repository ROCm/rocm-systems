.. meta::
   :description: rocDecode environment variables
   :keywords: rocDecode, environment variables, ROCm, AMD

********************************************************************
rocDecode environment variables
********************************************************************

The following environment variables affect the rocDecode runtime.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Environment variable
     - Values
   * - ``ROCDEC_LOG_LEVEL``
     - | Sets the maximum severity of log messages during decoding. The log level defines the maximum severity of log messages to output.
       | ``0``: Output critical messages only (default)
       | ``1``: Output critical and error messages
       | ``2``: Output critical, error and warning messages
       | ``3``: Output critical, error, warning and info messages
       | ``4``: Output critical, error, warning, info and debug messages
   * - ``ROCR_VISIBLE_DEVICES``
     - | Takes a comma-separated list of GPU indices.
       | When set, the decoder parses this list before it falls back to ``HIP_VISIBLE_DEVICES``.
   * - ``HIP_VISIBLE_DEVICES``
     - | Comma-separated list of GPU indices. Parsed when ``ROCR_VISIBLE_DEVICES`` isn't set.
   * - ``LIBVA_DRIVERS_PATH``
     - | Path to the directory containing the VA-API driver.
       | On Windows, set this to the directory containing ``vaon12_drv_video.dll``, which TheRock installs at ``%ROCM_PATH%\lib\rocm_sysdeps\bin``. When this variable isn't set, libva falls back to the driver directory that was configured when libva itself was built, which usually doesn't exist on the target machine.
       | On Linux, you don't normally need to set this. libva locates the driver on its own, so set it only to override that default.

