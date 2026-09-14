// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
