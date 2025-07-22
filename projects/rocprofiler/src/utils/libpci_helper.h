/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */
#pragma once

#include <pciaccess.h>

namespace rocprofiler {
    typedef struct pci_device * (*pci_device_find_by_slot_handler_t)(uint32_t, 
        uint32_t, uint32_t, uint32_t);
    typedef int (*pci_device_probe_handler_t)(struct pci_device *);
    typedef int (*pci_device_map_range_handler_t)(struct pci_device *, pciaddr_t,
        pciaddr_t, unsigned, void **);
    typedef int (*pci_device_unmap_range_handler_t) (struct pci_device *, void *,
        pciaddr_t);
    typedef int (*pci_system_init_handler_t)();
    typedef void (*pci_system_cleanup_handler_t)();

    struct PcieAccessApi{
        pci_device_find_by_slot_handler_t pci_device_find_by_slot;
        pci_device_probe_handler_t pci_device_probe;
        pci_device_map_range_handler_t pci_device_map_range;
        pci_device_unmap_range_handler_t pci_device_unmap_range;
        pci_system_init_handler_t pci_system_init;
        pci_system_cleanup_handler_t pci_system_cleanup;
    };

    
    // A function just to load all the libpciaccess symbols
    PcieAccessApi* LoadPcieAccessLibAPI();
    // A simple getter function for easy access to the API table
    const PcieAccessApi* GetPciAccessLibApi();
    // Sets all API table functions to null and calls dlclose
    void UnLoadPcieAccessLibAPI();
}  // namespace rocprofiler
