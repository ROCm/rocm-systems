// Copyright (C) 2024 Advanced Micro Devices. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

use amdsmi::*;

fn main() {
    // Initialize the AMD SMI library
    if let Err(e) = amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus) {
        eprintln!("Failed to initialize AMD SMI: {}", e);
        return;
    }

    // Get socket handles
    let socket_handles = match amdsmi_get_socket_handles() {
        Ok(handles) => handles,
        Err(e) => {
            eprintln!("Failed to get socket handles: {}", e);
            amdsmi_shut_down().expect("Failed to shutdown AMD SMI");
            return;
        }
    };

    for socket_handle in socket_handles {
        // Get processor handles for each socket handle
        let processor_handles = match amdsmi_get_processor_handles(socket_handle) {
            Ok(handles) => handles,
            Err(e) => {
                eprintln!(
                    "Failed to get processor handles for socket {:?}: {}",
                    socket_handle, e
                );
                continue;
            }
        };

        for processor_handle in processor_handles {
            // Get GPU ID using the processor handle
            match amdsmi_get_gpu_id(processor_handle) {
                Ok(gpu_id) => println!("GPU ID: {}", gpu_id),
                Err(e) => eprintln!("Failed to get GPU ID: {}", e),
            }

            // Get GPU revision using the processor handle
            match amdsmi_get_gpu_revision(processor_handle) {
                Ok(gpu_revision) => println!("GPU Revision: {}", gpu_revision),
                Err(e) => eprintln!("Failed to get GPU revision: {}", e),
            }

            // Get GPU vendor name using the processor handle
            match amdsmi_get_gpu_vendor_name(processor_handle) {
                Ok(gpu_vendor_name) => println!("GPU Vendor Name: {}", gpu_vendor_name),
                Err(e) => eprintln!("Failed to get GPU vendor name: {}", e),
            }

            // Get GPU VRAM vendor using the processor handle
            match amdsmi_get_gpu_vram_vendor(processor_handle) {
                Ok(gpu_vram_vendor) => println!("GPU VRAM Vendor: {}", gpu_vram_vendor),
                Err(e) => eprintln!("Failed to get GPU VRAM vendor: {}", e),
            }

            // Get GPU subsystem ID using the processor handle
            match amdsmi_get_gpu_subsystem_id(processor_handle) {
                Ok(gpu_subsystem_id) => println!("GPU Subsystem ID: {}", gpu_subsystem_id),
                Err(e) => eprintln!("Failed to get GPU subsystem ID: {}", e),
            }

            // Get GPU subsystem name using the processor handle
            match amdsmi_get_gpu_subsystem_name(processor_handle) {
                Ok(gpu_subsystem_name) => println!("GPU Subsystem Name: {}", gpu_subsystem_name),
                Err(e) => eprintln!("Failed to get GPU subsystem name: {}", e),
            }

            // Get GPU BDF using the processor handle
            match amdsmi_get_gpu_device_bdf(processor_handle) {
                Ok(gpu_bdf) => println!("GPU BDF: {}", gpu_bdf),
                Err(e) => eprintln!("Failed to get GPU BDF: {}", e),
            }

            // Get GPU BDF ID using the processor handle
            match amdsmi_get_gpu_bdf_id(processor_handle) {
                Ok(gpu_bdf_id) => println!("GPU BDF ID: {}", gpu_bdf_id),
                Err(e) => eprintln!("Failed to get GPU BDF ID: {}", e),
            }

            println!();
        }
    }

    // Shutdown the AMD SMI library
    if let Err(e) = amdsmi_shut_down() {
        eprintln!("Failed to shutdown AMD SMI: {}", e);
    }
}
