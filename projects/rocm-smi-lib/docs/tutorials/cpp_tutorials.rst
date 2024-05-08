.. meta::
  :description: ROCm SMI tutorial
  :keywords: install, SMI, library, api, AMD, ROCm

ROCm SMI C++ API tutorial
----------------------------

.. code-block:: c++

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

For more examples please check the `C++ example <https://github.com/ROCm/rocm_smi_lib/blob/develop/rocm_smi/example/rocm_smi_example.cc>`_
or `tests. <https://github.com/ROCm/rocm_smi_lib/tree/develop/tests/rocm_smi_test/functional>`_
