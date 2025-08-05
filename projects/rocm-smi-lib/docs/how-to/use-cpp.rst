.. meta::
   :description: Learn about using the ROCm SMI library with C++.
   :keywords: install, SMI, library, api, cpp, system management interface

***********************
Using ROCm SMI with C++
***********************

Installation
============

Follow the installation procedure for rocm_smi_lib. Refer to the :doc:`installation section <../install/install>`.

.. note::

   ``hipcc`` and other compilers will not automatically link in the ``librocm_smi64`` dynamic library. To ensure the
   ``librocm_smi64.so`` can be located, you must either set the ``LD_LIBRARY_PATH`` environment variable to the
   directory containing ``librocm_smi64.so`` (usually ``/opt/rocm/lib``) or pass the ``-lrocm_smi64`` flag to the compiler.

Device indices
==============

Many of the functions in the library take a "device index". The device index is a number greater than or equal to 0, and less than the number of devices detected, as determined by `rsmi_num_monitor_devices()`. The index is used to distinguish the detected devices from one another. It is important to note that a device may end up with a different index after a reboot, so an index should not be relied upon to be constant over reboots.

Hello ROCm SMI
================

The only required ROCm-SMI call for any program that wants to use ROCm-SMI is the ``rsmi_init()`` call. This call initializes some internal data structures that will be used by subsequent ROCm-SMI calls. 

When ROCm-SMI is no longer being used, ``rsmi_shut_down()`` should be called. This provides a way to do any releasing of resources that ROCm-SMI may have held. In many cases, this may have no effect, but may be necessary in future versions of the library.

A simple "Hello World" type program that displays the device ID of detected devices would look like this:

.. code-block:: c
  
    #include <stdint.h>
    #include "rocm_smi/rocm_smi.h"
    int main() {
      rsmi_status_t ret;
      uint32_t num_devices;
      uint16_t dev_id;
    
      // We will skip return code checks for this example, but it
      // is recommended to always check this as some calls may not
      // apply for some devices or ROCm releases
    
      ret = rsmi_init(0);
      ret = rsmi_num_monitor_devices(&num_devices);
    
      for (int i=0; i < num_devices; ++i) {
        ret = rsmi_dev_id_get(i, &dev_id);
        // dev_id holds the device ID of device i, upon a
        // successful call
      }
      ret = rsmi_shut_down();
      return 0;
    }
