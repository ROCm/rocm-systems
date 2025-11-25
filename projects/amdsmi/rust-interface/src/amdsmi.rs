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

use crate::amdsmi_wrapper;
use crate::utils::*;
use libc::free;
use std::mem::MaybeUninit;
use std::os::raw::c_void;
use std::ptr::null_mut;

/// Initializes the AMD SMI library.
///
/// This function must be called before any other AMD SMI functions are used.
/// It initializes the library with the specified flags.
///
/// # Arguments
///
/// * `init_flags` - Use a [`ProcessorTypeT`] for initialization flags.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if the initialization is successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
///     match amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus) {
///         Ok(_) => println!("AMD SMI initialized successfully"),
///         Err(e) => panic!("Failed to initialize AMD SMI: {}", e),
///     }
///     //Perform various tasks by invoking additional AMD SMI functions
///     //...
///
///     // Shut down the AMD SMI library
///     amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_init` call fails.
pub fn amdsmi_init(init_flags: AmdsmiInitFlagsT) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_init(init_flags as u64));
    Ok(())
}

/// Shuts down the AMD SMI library.
///
/// This function should be called when the AMD SMI library is no longer needed.
/// It performs any necessary cleanup and releases resources allocated by the library.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if the shutdown is successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
///     // Initialize the AMD SMI library
///     amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
///
///     // Perform operations with the AMD SMI library
///     // ...
/// #
/// #   // Shut down the AMD SMI library
///     match amdsmi_shut_down() {
///         Ok(_) => println!("AMD SMI shut down successfully"),
///         Err(e) => panic!("Failed to shut down AMD SMI: {}", e),
///     }
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_shut_down` call fails.
pub fn amdsmi_shut_down() -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_shut_down());
    Ok(())
}

/// Retrieves the socket handles for the AMD SMI library.
///
/// This function returns a list of socket handles available in the system.
/// Each handle can be used to query further information about the corresponding socket.
///
/// # Returns
///
/// * `AmdsmiResult<Vec<AmdsmiSocketHandle>>` - Returns `Ok(Vec<AmdsmiSocketHandle>)` containing the list of [`AmdsmiSocketHandle`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Retrieve the socket handles
///     match amdsmi_get_socket_handles() {
///         Ok(handles) => {
///             for handle in handles {
///                 // Perform operations with the AMD SMI library
///                 // ...
///             }
///         }
///         Err(e) => panic!("Failed to get socket handles: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_socket_handles` call fails.
pub fn amdsmi_get_socket_handles() -> AmdsmiResult<Vec<AmdsmiSocketHandle>> {
    let mut socket_count: u32 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_socket_handles(
        &mut socket_count,
        std::ptr::null_mut()
    ));

    let mut socket_handles = Vec::with_capacity(socket_count as usize);
    call_unsafe!(amdsmi_wrapper::amdsmi_get_socket_handles(
        &mut socket_count,
        socket_handles.as_mut_ptr()
    ));

    unsafe { socket_handles.set_len(socket_count as usize) };
    Ok(socket_handles)
}

/// Retrieves information about a specific socket.
///
/// This function returns a string containing information about the specified socket handle.
///
/// # Arguments
///
/// * `socket_handle` - A handle to the socket for which information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<String>` - Returns `Ok(String)` containing the socket information if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example socket handle, Get socket handle from amdsmi_get_socket_handles() function
///     let socket_handle = amdsmi_get_socket_handles().unwrap()[0];
///
///     // Retrieve the socket information
///     match amdsmi_get_socket_info(socket_handle) {
///         Ok(info) => println!("Socket info: {}", info),
///         Err(e) => panic!("Failed to get socket info: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_socket_info` call fails.
pub fn amdsmi_get_socket_info(socket_handle: AmdsmiSocketHandle) -> AmdsmiResult<String> {
    let (mut info, len) = define_cstr!(amdsmi_wrapper::AMDSMI_MAX_STRING_LENGTH);
    call_unsafe!(amdsmi_wrapper::amdsmi_get_socket_info(
        socket_handle,
        len,
        info.as_mut_ptr(),
    ));
    Ok(cstr_to_string!(info))
}

/// Retrieves the processor handles for a given socket.
///
/// This function returns a list of processor handles associated with the specified socket handle.
/// Each handle can be used to query further information about the corresponding processor.
///
/// # Arguments
///
/// * `socket_handle` - A handle to the socket for which processor handles are being queried.
///
/// # Returns
///
/// * `AmdsmiResult<Vec<AmdsmiProcessorHandle>>` - Returns `Ok(Vec<AmdsmiProcessorHandle>)` containing the list of [`AmdsmiProcessorHandle`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example socket handle, can get from amdsmi_get_socket_handles() function
///     let socket_handle = amdsmi_get_socket_handles().unwrap()[0];
///
///     // Retrieve the processor handles
///     match amdsmi_get_processor_handles(socket_handle) {
///         Ok(handles) => {
///             for handle in handles {
///                 // Perform operations with the AMD SMI library
///                 // ...
///             }
///         }
///         Err(e) => panic!("Failed to get processor handles: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_processor_handles` call fails.
pub fn amdsmi_get_processor_handles(
    socket_handle: AmdsmiSocketHandle,
) -> AmdsmiResult<Vec<AmdsmiProcessorHandle>> {
    let mut processor_count: u32 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_processor_handles(
        socket_handle,
        &mut processor_count,
        std::ptr::null_mut()
    ));

    let mut processor_handles = Vec::with_capacity(processor_count as usize);
    call_unsafe!(amdsmi_wrapper::amdsmi_get_processor_handles(
        socket_handle,
        &mut processor_count,
        processor_handles.as_mut_ptr()
    ));

    unsafe {
        processor_handles.set_len(processor_count as usize);
    }
    Ok(processor_handles)
}

/// Retrieves the type of a specific processor.
///
/// This function returns the processor type of the specified processor handle.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the type is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<ProcessorTypeT>` - Returns `Ok(ProcessorTypeT)` containing the [`ProcessorTypeT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the processor type
///     match amdsmi_get_processor_type(processor_handle) {
///         Ok(processor_type) => println!("Processor type: {:?}", processor_type),
///         Err(e) => panic!("Failed to get processor type: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_processor_type` call fails.
pub fn amdsmi_get_processor_type(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<ProcessorTypeT> {
    let mut processor_type = ProcessorTypeT::AmdsmiProcessorTypeUnknown;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_processor_type(
        processor_handle,
        &mut processor_type
    ));
    Ok(processor_type)
}

///Retrieves the processor handle for a given Bus-Device-Function (BDF) address.
///
/// This function returns a handle to the processor associated with the specified BDF address.
/// The BDF address is a unique identifier for a device on the PCI bus.
///
/// # Arguments
///
/// * `bdf` - A [`AmdsmiBdfT`] representing the Bus-Device-Function (BDF) address.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiProcessorHandle>` - Returns `Ok(AmdsmiProcessorHandle)` containing the [`AmdsmiProcessorHandle`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Get a processor_handle from the amdsmi_get_processor_handles() function
///     let processor_handle_example = amdsmi_get_processor_handles!()[0];
///     // Example BDF
///     let bdf = amdsmi_get_gpu_device_bdf(processor_handle_example).expect("Failed to get BDF");
///
///     // Retrieve the processor handle
///     let processor_handle = amdsmi_get_processor_handle_from_bdf(bdf).expect("Failed to get processor handle");
///
///     assert_eq!(processor_handle, processor_handle_example);
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_processor_handle_from_bdf` call fails.
pub fn amdsmi_get_processor_handle_from_bdf(
    bdf: AmdsmiBdfT,
) -> AmdsmiResult<AmdsmiProcessorHandle> {
    let mut processor_handle = MaybeUninit::<AmdsmiProcessorHandle>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_processor_handle_from_bdf(
        bdf,
        processor_handle.as_mut_ptr()
    ));
    Ok(unsafe { processor_handle.assume_init() })
}

/// Retrieves the GPU ID for a given processor handle.
///
/// This function returns the GPU ID associated with the specified processor handle.
/// The GPU ID is a unique identifier for the GPU.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU ID is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u32>` - Returns `Ok(u32)` containing the GPU ID if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU ID
///     match amdsmi_get_gpu_id(processor_handle) {
///         Ok(gpu_id) => println!("GPU ID: {}", gpu_id),
///         Err(e) => panic!("Failed to get GPU ID: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_id` call fails.
pub fn amdsmi_get_gpu_id(processor_handle: AmdsmiProcessorHandle) -> AmdsmiResult<u16> {
    let mut id: u16 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_id(processor_handle, &mut id));
    Ok(id)
}

/// Retrieves the GPU revision for a given processor handle.
///
/// This function returns the GPU revision associated with the specified processor handle.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU revision is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u32>` - Returns `Ok(u32)` containing the GPU revision if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU revision
///     match amdsmi_get_gpu_revision(processor_handle) {
///         Ok(gpu_revision) => println!("GPU Revision: {}", gpu_revision),
///         Err(e) => panic!("Failed to get GPU revision: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_revision` call fails.
pub fn amdsmi_get_gpu_revision(processor_handle: AmdsmiProcessorHandle) -> AmdsmiResult<u16> {
    let mut revision: u16 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_revision(
        processor_handle,
        &mut revision
    ));
    Ok(revision)
}

/// Retrieves the GPU vendor name for a given processor handle.
///
/// This function returns the GPU vendor name associated with the specified processor handle.
/// The vendor name provides information about the manufacturer of the GPU.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU vendor name is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<String>` - Returns `Ok(String)` containing the GPU vendor name if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU vendor name
///     match amdsmi_get_gpu_vendor_name(processor_handle) {
///         Ok(vendor_name) => println!("GPU Vendor Name: {}", vendor_name),
///         Err(e) => panic!("Failed to get GPU vendor name: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_vendor_name` call fails.
pub fn amdsmi_get_gpu_vendor_name(processor_handle: AmdsmiProcessorHandle) -> AmdsmiResult<String> {
    let (mut name, len) = define_cstr!(amdsmi_wrapper::AMDSMI_MAX_STRING_LENGTH);
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_vendor_name(
        processor_handle,
        name.as_mut_ptr(),
        len
    ));
    Ok(cstr_to_string!(name))
}

/// Retrieves the GPU VRAM vendor name for a given processor handle.
///
/// This function returns the GPU VRAM vendor name associated with the specified processor handle.
/// The VRAM vendor name provides information about the manufacturer of the GPU's VRAM.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU VRAM vendor name is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<String>` - Returns `Ok(String)` containing the GPU VRAM vendor name if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU VRAM vendor name
///     match amdsmi_get_gpu_vram_vendor(processor_handle) {
///         Ok(vram_vendor_name) => println!("GPU VRAM Vendor Name: {}", vram_vendor_name),
///         Err(e) => panic!("Failed to get GPU VRAM vendor name: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_vram_vendor` call fails.
pub fn amdsmi_get_gpu_vram_vendor(processor_handle: AmdsmiProcessorHandle) -> AmdsmiResult<String> {
    let (mut brand, len) = define_cstr!(amdsmi_wrapper::AMDSMI_MAX_STRING_LENGTH);
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_vram_vendor(
        processor_handle,
        brand.as_mut_ptr(),
        len as u32,
    ));
    Ok(cstr_to_string!(brand))
}

/// Retrieves the GPU subsystem ID for a given processor handle.
///
/// This function returns the GPU subsystem ID associated with the specified processor handle.
/// The subsystem ID provides information about the specific subsystem of the GPU.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU subsystem ID is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u32>` - Returns `Ok(u32)` containing the GPU subsystem ID if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU subsystem ID
///     match amdsmi_get_gpu_subsystem_id(processor_handle) {
///         Ok(subsystem_id) => println!("GPU Subsystem ID: {}", subsystem_id),
///         Err(e) => panic!("Failed to get GPU subsystem ID: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_subsystem_id` call fails.
pub fn amdsmi_get_gpu_subsystem_id(processor_handle: AmdsmiProcessorHandle) -> AmdsmiResult<u16> {
    let mut id: u16 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_subsystem_id(
        processor_handle,
        &mut id
    ));
    Ok(id)
}

/// Retrieves the GPU subsystem name for a given processor handle.
///
/// This function returns the GPU subsystem name associated with the specified processor handle.
/// The subsystem name provides information about the specific subsystem of the GPU.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU subsystem name is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<String>` - Returns `Ok(String)` containing the GPU subsystem name if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU subsystem name
///     match amdsmi_get_gpu_subsystem_name(processor_handle) {
///         Ok(subsystem_name) => println!("GPU Subsystem Name: {}", subsystem_name),
///         Err(e) => panic!("Failed to get GPU subsystem name: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_subsystem_name` call fails.
pub fn amdsmi_get_gpu_subsystem_name(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<String> {
    let (mut name, len) = define_cstr!(amdsmi_wrapper::AMDSMI_MAX_STRING_LENGTH);
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_subsystem_name(
        processor_handle,
        name.as_mut_ptr(),
        len
    ));
    Ok(cstr_to_string!(name))
}

/// Retrieves the PCI bandwidth for a given GPU processor handle.
///
/// This function returns the PCI bandwidth associated with the specified processor handle.
/// The PCI bandwidth provides information about the data transfer rate of the GPU over the PCI bus.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the PCI bandwidth is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u32>` - Returns `Ok(u32)` containing the PCI bandwidth if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the PCI bandwidth
///     match amdsmi_get_gpu_pci_bandwidth(processor_handle) {
///         Ok(pci_bandwidth) => println!("PCI Bandwidth: {:?}", pci_bandwidth),
///         Err(e) => panic!("Failed to get PCI bandwidth: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_pci_bandwidth` call fails.
pub fn amdsmi_get_gpu_pci_bandwidth(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiPcieBandwidthT> {
    let mut bandwidth = MaybeUninit::<AmdsmiPcieBandwidthT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_pci_bandwidth(
        processor_handle,
        bandwidth.as_mut_ptr()
    ));
    let bandwidth = unsafe { bandwidth.assume_init() };
    Ok(bandwidth)
}

/// Retrieves the Bus-Device-Function (BDF) ID for a given GPU processor handle.
///
/// This function returns the BDF ID associated with the specified processor handle.
/// The BDF ID is a unique identifier for a device on the PCI bus, which includes the bus number, device number, and function number.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the BDF ID is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u32>` - Returns `Ok(u32)` containing the BDF ID if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the BDF ID
///     match amdsmi_get_gpu_bdf_id(processor_handle) {
///         Ok(bdf_id) => println!("BDF ID: {:#010x}", bdf_id),
///         Err(e) => panic!("Failed to get BDF ID: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_bdf_id` call fails.
pub fn amdsmi_get_gpu_bdf_id(processor_handle: AmdsmiProcessorHandle) -> AmdsmiResult<u64> {
    let mut bdfid: u64 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_bdf_id(
        processor_handle,
        &mut bdfid
    ));
    Ok(bdfid)
}

/// Retrieves the NUMA (Non-Uniform Memory Access) node for a given GPU processor handle.
///
/// This function returns the NUMA node associated with the specified processor handle.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the NUMA node is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u32>` - Returns `Ok(u32)` containing the NUMA node value if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the NUMA node
///     match amdsmi_get_gpu_topo_numa_affinity(processor_handle) {
///         Ok(numa_node) => println!("NUMA Node ID: {}", numa_node),
///         Err(e) => panic!("Failed to get NUMA affinity: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_topo_numa_affinity` call fails.
pub fn amdsmi_get_gpu_topo_numa_affinity(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<i32> {
    let mut numa_node: i32 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_topo_numa_affinity(
        processor_handle,
        &mut numa_node
    ));
    Ok(numa_node)
}

/// Retrieves the PCI throughput for a given GPU processor handle.
///
/// This function returns the PCI throughput associated with the specified processor handle.
/// The PCI throughput provides information about the data transfer rate of the GPU over the PCI bus.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the PCI throughput is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<(u64, u64, u64)>` - Returns `Ok((u64, u64, u64))` containing the PCI throughput as below:
///   - The first value is the number of bytes sent will be written in 1 second.
///   - The second value is the number of bytes received will be written.
///   - The third value is the maximum packet size will be written.
///   If successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the PCI throughput
///     match amdsmi_get_gpu_pci_throughput(processor_handle) {
///         Ok((sent, received, max_pkt_sz)) => {
///             println!("PCI Throughput - Sent: {} B/s, Received: {} B/s, Max package size: {} B/s", sent, received, max_pkt_sz);
///         }
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_pci_throughput() not supported on this device"),
///         Err(e) => panic!("Failed to get PCI throughput: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_pci_throughput` call fails.
pub fn amdsmi_get_gpu_pci_throughput(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<(u64, u64, u64)> {
    let mut sent: u64 = 0;
    let mut received: u64 = 0;
    let mut max_pkt_sz: u64 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_pci_throughput(
        processor_handle,
        &mut sent,
        &mut received,
        &mut max_pkt_sz
    ));
    Ok((sent, received, max_pkt_sz))
}

/// Retrieves the PCI replay counter for a given GPU processor handle.
///
/// This function returns the PCI replay counter associated with the specified processor handle.
/// The PCI replay counter provides information about the sum of the NAK's received and generated by the GPU.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the PCI replay counter is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u64>` - Returns `Ok(u64)` containing the PCI replay counter if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the PCI replay counter
///     match amdsmi_get_gpu_pci_replay_counter(processor_handle) {
///         Ok(counter) => println!("PCI Replay Counter: {}", counter),
///         Err(e) => panic!("Failed to get PCI replay counter: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_pci_replay_counter` call fails.
pub fn amdsmi_get_gpu_pci_replay_counter(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<u64> {
    let mut counter: u64 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_pci_replay_counter(
        processor_handle,
        &mut counter
    ));
    Ok(counter)
}

/// Control the set of allowed PCIe bandwidths that can be used for a given GPU processor handle.
///
/// This function Control the set of allowed PCIe bandwidths that can be used with the specified processor handle.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the PCI bandwidth is being set.
/// * `bw_bitmask` - The allowed PCI bandwidth bitmask. A bitmask indicating the indices of the
/// bandwidths that are to be enabled (1) and disabled (0). Only the lowest AmdsmiFrequenciesT.num_supported bits of this mask are relevant.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if the PCI bandwidth is successfully set, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Set the PCI bandwidth
///     match amdsmi_set_gpu_pci_bandwidth(processor_handle, 0) {
///         Ok(_) => println!("PCI Bandwidth set successfully"),
///         Err(e) => panic!("Failed to set PCI bandwidth: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_gpu_pci_bandwidth` call fails.
pub fn amdsmi_set_gpu_pci_bandwidth(
    processor_handle: AmdsmiProcessorHandle,
    bw_bitmask: u64,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_gpu_pci_bandwidth(
        processor_handle,
        bw_bitmask
    ));
    Ok(())
}

/// Retrieves the energy accumulator counter information for the specified processor handle.
///
/// This function retrieves energy accumulator counter information for the specified processor handle,
/// returning the energy accumulator, counter resolution, and timestamp.
/// energy_accumulator * counter_resolution = total_energy_consumption in micro-Joules.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the energy count is being retrieved.
///
/// # Returns
///
/// * `AmdsmiResult<(u64, f32, u64)>` - Returns a tuple containing the energy accumulator, counter resolution, and timestamp if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the energy count
///     match amdsmi_get_energy_count(processor_handle) {
///         Ok((energy_accumulator, counter_resolution, timestamp)) => {
///             println!("Energy Accumulator: {}", energy_accumulator);
///             println!("Counter Resolution: {}", counter_resolution);
///             println!("Timestamp: {}", timestamp);
///         },
///         Err(e) => panic!("Failed to get energy count: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_energy_count` call fails.
pub fn amdsmi_get_energy_count(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<(u64, f32, u64)> {
    let mut energy_accumulator: u64 = 0;
    let mut counter_resolution: f32 = 0.0;
    let mut timestamp: u64 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_energy_count(
        processor_handle,
        &mut energy_accumulator,
        &mut counter_resolution,
        &mut timestamp
    ));
    Ok((energy_accumulator, counter_resolution, timestamp))
}

/// Sets the power cap for a given GPU processor handle.
///
/// This function sets the power cap associated with the specified processor handle.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the power cap is being set.
/// * `power_cap` - The desired power cap in milliwatts (mW).
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if the power cap is successfully set, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///     let sensor_ind : u32 = 0;
///
///     let cap_info = match amdsmi_get_power_cap_info(processor_handle, sensor_ind) {
///         Ok(info) => info,
///         Err(e) => panic!("Failed to get power cap information: {}", e),
///     };
///
///     let power_cap = (cap_info.min_power_cap + cap_info.max_power_cap) / 2;
///
///     // Set the power cap
///     match amdsmi_set_power_cap(processor_handle, sensor_ind, power_cap) {
///         Ok(_) => println!("Power cap {} set successfully", power_cap),
///         Err(e) => panic!("Failed to set power cap: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_power_cap` call fails.
pub fn amdsmi_set_power_cap(
    processor_handle: AmdsmiProcessorHandle,
    sensor_ind: u32,
    cap: u64,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_power_cap(
        processor_handle,
        sensor_ind,
        cap
    ));
    Ok(())
}

/// Sets the power profile for a given GPU processor handle.
///
/// This function sets the power profile associated with the specified processor handle.
/// The power profile determines the power consumption and performance characteristics of the GPU.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the power profile is being set.
/// * `profile` - A [`AmdsmiPowerProfilePresetMasksT`] type that hold the mask of the desired new power profile.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if the power profile is successfully set, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Set the power profile
///     match amdsmi_set_gpu_power_profile(processor_handle, AmdsmiPowerProfilePresetMasksT::AmdsmiPwrProfPrstCustomMask) {
///         Ok(_) => println!("Power profile set successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_set_gpu_power_profile() not supported on this device"),
///         Err(e) => panic!("Failed to set power profile: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_gpu_power_profile` call fails.
pub fn amdsmi_set_gpu_power_profile(
    processor_handle: AmdsmiProcessorHandle,
    profile: AmdsmiPowerProfilePresetMasksT,
) -> AmdsmiResult<()> {
    let reserved: u32 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_set_gpu_power_profile(
        processor_handle,
        reserved,
        profile
    ));
    Ok(())
}

/// Retrieves the total memory for a given GPU processor handle.
///
/// This function returns the total memory associated with the specified processor handle.
/// The total memory provides information about the total amount of memory that exist.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the total memory is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u64>` - Returns `Ok(u64)` containing the total memory in bytes if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the total memory
///     match amdsmi_get_gpu_memory_total(processor_handle, AmdsmiMemoryTypeT::AmdsmiMemTypeVram) {
///         Ok(total_memory) => println!("Total GPU Memory: {} bytes", total_memory),
///         Err(e) => panic!("Failed to get total GPU memory: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_memory_total` call fails.
pub fn amdsmi_get_gpu_memory_total(
    processor_handle: AmdsmiProcessorHandle,
    memory_type: AmdsmiMemoryTypeT,
) -> AmdsmiResult<u64> {
    let mut memory_total: u64 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_memory_total(
        processor_handle,
        memory_type,
        &mut memory_total
    ));
    Ok(memory_total)
}

/// Retrieves the memory usage for a given GPU processor handle.
///
/// This function returns the memory usage associated with the specified processor handle.
/// The memory usage provides information about the amount of memory currently being used on the GPU.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the memory usage is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u64>` - Returns `Ok(u64)` containing the memory usage in bytes if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the memory usage
///     match amdsmi_get_gpu_memory_usage(processor_handle, AmdsmiMemoryTypeT::AmdsmiMemTypeVram) {
///         Ok(memory_usage) => println!("GPU Memory Usage: {} bytes", memory_usage),
///         Err(e) => panic!("Failed to get GPU memory usage: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_memory_usage` call fails.
pub fn amdsmi_get_gpu_memory_usage(
    processor_handle: AmdsmiProcessorHandle,
    memory_type: AmdsmiMemoryTypeT,
) -> AmdsmiResult<u64> {
    let mut memory_used: u64 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_memory_usage(
        processor_handle,
        memory_type,
        &mut memory_used
    ));
    Ok(memory_used)
}

/// Retrieves the bad page information for a given GPU processor handle.
///
/// This function returns the bad page information associated with the specified processor handle.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the bad page information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<Vec<AmdsmiRetiredPageRecordT>>` - Returns `Ok(Vec<AmdsmiRetiredPageRecordT>)` containing a list of [`AmdsmiRetiredPageRecordT`] structs if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the bad page information
///     match amdsmi_get_gpu_bad_page_info(processor_handle) {
///         Ok(bad_page_info) => {
///             for page in bad_page_info {
///                 println!("Bad Page: {:?}", page);
///             }
///         }
///         Err(e) => panic!("Failed to get bad page information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_bad_page_info` call fails.
pub fn amdsmi_get_gpu_bad_page_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<Vec<AmdsmiRetiredPageRecordT>> {
    // First call to get the number of bad pages
    let mut num_pages: u32 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_bad_page_info(
        processor_handle,
        &mut num_pages,
        std::ptr::null_mut()
    ));

    // Allocate a vector with the capacity of num_pages
    let mut bad_pages: Vec<AmdsmiRetiredPageRecordT> = Vec::with_capacity(num_pages as usize);

    // Second call to get the bad page information
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_bad_page_info(
        processor_handle,
        &mut num_pages,
        bad_pages.as_mut_ptr()
    ));

    // Set the length of the vector to num_pages
    unsafe { bad_pages.set_len(num_pages as usize) };

    Ok(bad_pages)
}

/// Retrieves the RAS (Reliability, Availability, and Serviceability) feature information for a given GPU handle.
///
/// This function returns the RAS feature information for the specified GPU handle. The RAS feature
/// information includes the RAS version and schema information.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the RAS feature information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiRasFeatureT>` - Returns `Ok(AmdsmiRasFeatureT)` containing the [`AmdsmiRasFeatureT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the RAS feature information
///     match amdsmi_get_gpu_ras_feature_info(processor_handle) {
///         Ok(ras_info) => println!("GPU RAS Feature Info: {:?}", ras_info),
///         Err(e) => panic!("Failed to get GPU RAS Feature Info: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_ras_feature_info` call fails.
pub fn amdsmi_get_gpu_ras_feature_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiRasFeatureT> {
    let mut ras_info = MaybeUninit::<AmdsmiRasFeatureT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_ras_feature_info(
        processor_handle,
        ras_info.as_mut_ptr()
    ));
    let ras_info = unsafe { ras_info.assume_init() };
    Ok(ras_info)
}

/// Retrieves the RAS (Reliability, Availability, and Serviceability) block status for a given GPU handle.
///
/// This function returns if RAS features are enabled or disabled for given block.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the RAS block features are being queried.
/// * `block` - The specific GPU block for which the RAS features are being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiRasErrStateT>` - Returns `Ok(AmdsmiRasErrStateT)` containing the [`AmdsmiRasErrStateT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example GPU block
///     let block = AmdsmiGpuBlockT::AmdsmiGpuBlockGfx;
///
///     // Retrieve the enabled RAS block features
///     match amdsmi_get_gpu_ras_block_features_enabled(processor_handle, block) {
///         Ok(ras_features) => println!("Enabled RAS Block Features: {:?}", ras_features),
///         Err(e) => panic!("Failed to get enabled RAS block features: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_ras_block_features_enabled` call fails.
pub fn amdsmi_get_gpu_ras_block_features_enabled(
    processor_handle: AmdsmiProcessorHandle,
    block: AmdsmiGpuBlockT,
) -> AmdsmiResult<AmdsmiRasErrStateT> {
    let mut ras_features = MaybeUninit::<AmdsmiRasErrStateT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_ras_block_features_enabled(
        processor_handle,
        block,
        ras_features.as_mut_ptr()
    ));
    let ras_features = unsafe { ras_features.assume_init() };
    Ok(ras_features)
}

/// Retrieves the reserved memory pages information for a given GPU handle.
///
/// This function returns the reserved memory pages information for the specified GPU handle. The reserved memory pages
/// include details about the memory pages that are reserved by the GPU.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the reserved memory pages are being queried.
///
/// # Returns
///
/// * `AmdsmiResult<Vec<AmdsmiRetiredPageRecordT>>` - Returns `Ok(Vec<AmdsmiRetiredPageRecordT>)` containing the reserved memory pages if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the reserved memory pages
///     match amdsmi_get_gpu_memory_reserved_pages(processor_handle) {
///         Ok(pages) => println!("Reserved Memory Pages: {:?}", pages),
///         Err(e) => panic!("Failed to get reserved memory pages: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_memory_reserved_pages` call fails.
pub fn amdsmi_get_gpu_memory_reserved_pages(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<Vec<AmdsmiRetiredPageRecordT>> {
    let mut num_pages = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_memory_reserved_pages(
        processor_handle,
        &mut num_pages,
        std::ptr::null_mut()
    ));

    let mut pages = Vec::with_capacity(num_pages as usize);
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_memory_reserved_pages(
        processor_handle,
        &mut num_pages,
        pages.as_mut_ptr()
    ));

    unsafe {
        pages.set_len(num_pages as usize);
    }

    Ok(pages)
}

/// Retrieves the fan speed for a given GPU handle and sensor index.
///
/// This function returns the fan speed for the specified GPU handle and sensor index. The fan speed
/// provides information as a value relative to AMDSMI_MAX_FAN_SPEED.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the fan speed is being queried.
/// * `sensor_index` - The index of the fan sensor to query.
///
/// # Returns
///
/// * `AmdsmiResult<i64>` - Returns `Ok(i64)` containing the fan speed relative to MAX if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example sensor index
///     let sensor_index = 0;
///
///     // Retrieve the fan speed
///     match amdsmi_get_gpu_fan_speed(processor_handle, sensor_index) {
///         Ok(speed) => println!("GPU Fan Speed: {}", speed),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_fan_speed() not supported on this device"),
///         Err(e) => panic!("Failed to get GPU fan speed: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_fan_speed` call fails.
pub fn amdsmi_get_gpu_fan_speed(
    processor_handle: AmdsmiProcessorHandle,
    sensor_index: u32,
) -> AmdsmiResult<i64> {
    let mut speed = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_fan_speed(
        processor_handle,
        sensor_index,
        &mut speed
    ));
    Ok(speed)
}

/// Get the maximum fan speed of the device with the specified processor handle and sensor index.
///
/// Given a processor handle `processor_handle` and a sensor index `sensor_ind`, this function returns
/// the maximum fan speed for the specified GPU.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the maximum fan speed is being queried.
/// * `sensor_ind` - The index of the fan sensor to query.
///
/// # Returns
///
/// * `AmdsmiResult<u64>` - Returns `Ok(u64)` containing the maximum fan speed if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example sensor index
///     let sensor_ind = 0;
///
///     // Retrieve the maximum fan speed
///     match amdsmi_get_gpu_fan_speed_max(processor_handle, sensor_ind) {
///         Ok(speed) => println!("GPU Maximum Fan Speed: {}", speed),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_fan_speed_max() not supported on this device"),
///         Err(e) => panic!("Failed to get GPU maximum fan speed: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_fan_speed_max` call fails.
pub fn amdsmi_get_gpu_fan_speed_max(
    processor_handle: AmdsmiProcessorHandle,
    sensor_ind: u32,
) -> AmdsmiResult<u64> {
    let mut speed = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_fan_speed_max(
        processor_handle,
        sensor_ind,
        &mut speed
    ));
    Ok(speed)
}

/// Get the temperature metric of the device with the specified processor handle, sensor type, and metric.
///
/// Given a processor handle `processor_handle`, a sensor type `sensor_type`, and a temperature metric `metric`,
/// this function returns the temperature metric for the specified GPU.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the temperature metric is being queried.
/// * `sensor_type` - The type of the temperature sensor [`AmdsmiTemperatureTypeT`] to query.
/// * `metric` - The temperature metric [`AmdsmiTemperatureMetricT`] to query.
///
/// # Returns
///
/// * `AmdsmiResult<i64>` - Returns `Ok(i64)` containing the temperature metric value if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example sensor type and metric
///     let sensor_type = AmdsmiTemperatureTypeT::AmdsmiTemperatureTypeEdge;
///     let metric = AmdsmiTemperatureMetricT::AmdsmiTempCurrent;
///
///     // Retrieve the temperature metric
///     match amdsmi_get_temp_metric(processor_handle, sensor_type, metric) {
///         Ok(temp_metric) => println!("GPU Temperature Metric: {}", temp_metric),
///         Err(e) => panic!("Failed to get GPU temperature metric: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_temp_metric` call fails.
pub fn amdsmi_get_temp_metric(
    processor_handle: AmdsmiProcessorHandle,
    sensor_type: AmdsmiTemperatureTypeT,
    metric: AmdsmiTemperatureMetricT,
) -> AmdsmiResult<i64> {
    let mut temperature = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_temp_metric(
        processor_handle,
        sensor_type,
        metric,
        &mut temperature
    ));
    Ok(temperature)
}

/// Get the GPU cache information of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function returns the GPU cache information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the cache information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiGpuCacheInfoT>` - Returns `Ok(AmdsmiGpuCacheInfoT)` containing the [`AmdsmiGpuCacheInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU cache information
///     match amdsmi_get_gpu_cache_info(processor_handle) {
///         Ok(cache_info) => println!("GPU Cache Info: {:?}", cache_info),
///         Err(e) => panic!("Failed to get GPU cache info: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_cache_info` call fails.
pub fn amdsmi_get_gpu_cache_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiGpuCacheInfoT> {
    let mut info = MaybeUninit::<AmdsmiGpuCacheInfoT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_cache_info(
        processor_handle,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Get the voltage metric of the device with the specified processor handle, sensor type, and metric.
///
/// Given a processor handle `processor_handle`, a sensor type `sensor_type`, and a voltage metric type `metric`,
/// this function returns the voltage metric for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the voltage metric is being queried.
/// * `sensor_type` - The type of the voltage sensor [`AmdsmiVoltageTypeT`] to query.
/// * `metric` - The voltage metric [`amdsmi_get_gpu_volt_metric`] to query.
///
/// # Returns
///
/// * `AmdsmiResult<i64>` - Returns `Ok(i64)` containing the voltage metric if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example sensor type and metric
///     let sensor_type = AmdsmiVoltageTypeT::AmdsmiVoltTypeVddgfx;
///     let metric = AmdsmiVoltageMetricT::AmdsmiVoltCurrent;
///
///     // Retrieve the voltage metric
///     match amdsmi_get_gpu_volt_metric(processor_handle, sensor_type, metric) {
///         Ok(voltage_metric) => println!("GPU Voltage Metric: {}", voltage_metric),
///         Err(e) => panic!("Failed to get GPU voltage metric: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_volt_metric` call fails.
pub fn amdsmi_get_gpu_volt_metric(
    processor_handle: AmdsmiProcessorHandle,
    sensor_type: AmdsmiVoltageTypeT,
    metric: AmdsmiVoltageMetricT,
) -> AmdsmiResult<i64> {
    let mut voltage = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_volt_metric(
        processor_handle,
        sensor_type,
        metric,
        &mut voltage
    ));
    Ok(voltage)
}

/// Reset the GPU fan of the device with the specified processor handle and sensor index.
///
/// Given a processor handle `processor_handle` and a sensor index `sensor_ind`, this function resets the GPU fan
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU fan is being reset.
/// * `sensor_ind` - The 0-based sensor index of the fan sensor to reset.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example sensor index
///     let sensor_ind = 0;
///
///     // Reset the GPU fan
///     match amdsmi_reset_gpu_fan(processor_handle, sensor_ind) {
///         Ok(()) => println!("GPU fan reset successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_reset_gpu_fan() not supported on this device"),
///         Err(e) => panic!("Failed to reset GPU fan: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_reset_gpu_fan` call fails.
pub fn amdsmi_reset_gpu_fan(
    processor_handle: AmdsmiProcessorHandle,
    sensor_ind: u32,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_reset_gpu_fan(
        processor_handle,
        sensor_ind
    ));
    Ok(())
}

/// Get the fan speed in RPM for the GPU device with the specified processor handle and sensor index.
///
/// Given a processor handle `processor_handle` and a sensor index `sensor_ind`, this function retrieves the fan speed in RPM
/// for the specified GPU device.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the fan speed is being queried.
/// * `sensor_ind` - The index of the sensor for which the fan speed is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<i64>` - Returns `Ok(i64)` containing the fan speed in RPM if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example sensor index
///     let sensor_ind: u32 = 0;
///
///     // Retrieve the fan speed in RPM
///     match amdsmi_get_gpu_fan_rpms(processor_handle, sensor_ind) {
///         Ok(speed) => println!("Fan speed: {} RPM", speed),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_fan_rpms() not supported on this device"),
///         Err(e) => panic!("Failed to get fan speed: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_fan_rpms` call fails.
pub fn amdsmi_get_gpu_fan_rpms(
    processor_handle: AmdsmiProcessorHandle,
    sensor_ind: u32,
) -> AmdsmiResult<i64> {
    let mut speed: i64 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_fan_rpms(
        processor_handle,
        sensor_ind,
        &mut speed as *mut i64
    ));
    Ok(speed)
}

/// Set the GPU fan speed of the device with the specified processor handle, sensor index, and speed.
///
/// Given a processor handle `processor_handle`, a sensor index `sensor_ind`, and a fan speed `speed`,
/// this function sets the GPU fan speed for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU fan speed is being set.
/// * `sensor_ind` - The index of the fan sensor to set the speed for.
/// * `speed` - The speed to set the fan to, in RPM.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example sensor index and speed
///     let sensor_ind = 0;
///     let speed = 1333;
///
///     // Set the GPU fan speed
///     match amdsmi_set_gpu_fan_speed(processor_handle, sensor_ind, speed) {
///         Ok(()) => println!("GPU fan speed set successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_set_gpu_fan_speed() not supported on this device"),
///         Err(e) => panic!("Failed to set GPU fan speed: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_gpu_fan_speed` call fails.
pub fn amdsmi_set_gpu_fan_speed(
    processor_handle: AmdsmiProcessorHandle,
    sensor_ind: u32,
    speed: u64,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_gpu_fan_speed(
        processor_handle,
        sensor_ind,
        speed
    ));
    Ok(())
}

/// Get the GPU busy percent of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function returns the GPU busy percent
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor which is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u32>` - Returns `Ok(u32)` containing the GPU busy percent if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU busy percent
///     match amdsmi_get_gpu_busy_percent(processor_handle) {
///         Ok(gpu_busy_percent) => println!("GPU Busy Percent: {}", gpu_busy_percent),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_busy_percent() not supported on this device"),
///         Err(e) => panic!("Failed to get GPU busy_percent level: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_busy_percent` call fails.
pub fn amdsmi_get_gpu_busy_percent(
    processor_handle: AmdsmiProcessorHandle
) -> AmdsmiResult<u32> {
    let mut gpu_busy_percent = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_busy_percent(
        processor_handle,
        &mut gpu_busy_percent
    ));
    Ok(gpu_busy_percent)
}

///
/// Retrieves the utilization count for the specified processor handle and counter types.
///
/// This function retrieves the utilization count for the specified processor handle and counter types,
/// returning a list of utilization counters and a timestamp.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the utilization count is being retrieved.
/// * `counter_types` - A list of counter types [`AmdsmiUtilizationCounterTypeT`] for which the utilization count is being retrieved.
///
/// # Returns
///
/// * `AmdsmiResult<(Vec<AmdsmiUtilizationCounterT>, u64)>` - Returns a tuple containing a list of [`AmdsmiUtilizationCounterT`] and a timestamp if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Define the counter types
///     let counter_types = vec![
///         AmdsmiUtilizationCounterTypeT::AmdsmiFineGrainGfxActivity,
///         AmdsmiUtilizationCounterTypeT::AmdsmiFineGrainMemActivity,
///     ];
///
///     // Get the utilization count
///     match amdsmi_get_utilization_count(processor_handle, counter_types) {
///         Ok((utilization_counters, timestamp)) => {
///             println!("Timestamp: {}", timestamp);
///             for counter in utilization_counters {
///                 println!("Counter Type: {:?}, Value: {}", counter.type_, counter.value);
///             }
///         },
///         Err(e) => panic!("Failed to get utilization count: {}", e),
///     }
///
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_utilization_count` call fails.
pub fn amdsmi_get_utilization_count(
    processor_handle: AmdsmiProcessorHandle,
    counter_types: Vec<AmdsmiUtilizationCounterTypeT>,
) -> AmdsmiResult<(Vec<AmdsmiUtilizationCounterT>, u64)> {
    let count = counter_types.len() as u32;
    let mut utilization_counters: Vec<AmdsmiUtilizationCounterT> =
        Vec::with_capacity(count as usize);
    let mut timestamp = 0;

    // Initialize the utilization_counters with the appropriate types
    for counter_type in counter_types {
        let utilization_counter = AmdsmiUtilizationCounterT {
            type_: counter_type,
            value: 0,
            fine_value: [0; 4],
            fine_value_count: 0,
        };
        utilization_counters.push(utilization_counter);
    }

    call_unsafe!(amdsmi_wrapper::amdsmi_get_utilization_count(
        processor_handle,
        utilization_counters.as_mut_ptr(),
        count,
        &mut timestamp
    ));
    Ok((utilization_counters, timestamp))
}

/// Get the GPU performance level of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function returns the GPU performance level
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU performance level is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiDevPerfLevelT>` - Returns `Ok(AmdsmiDevPerfLevelT)` containing the [`AmdsmiDevPerfLevelT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU performance level
///     match amdsmi_get_gpu_perf_level(processor_handle) {
///         Ok(perf_level) => println!("GPU Performance Level: {:?}", perf_level),
///         Err(e) => panic!("Failed to get GPU performance level: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_perf_level` call fails.
pub fn amdsmi_get_gpu_perf_level(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiDevPerfLevelT> {
    let mut perf = MaybeUninit::<AmdsmiDevPerfLevelT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_perf_level(
        processor_handle,
        perf.as_mut_ptr()
    ));
    let perf = unsafe { perf.assume_init() };
    Ok(perf)
}

/// Enter performance determinism mode with provided processor handle.
///
/// Given a processor handle `processor_handle` and a clock value `clkvalue`, this function will enable the GPU performance
/// determinism mode which enforces a GFXCLK frequency SoftMax limit per GPU for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU performance determinism mode is being set.
/// * `clkvalue` - The clock value to set for the performance determinism mode.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example clock value
///     let clkvalue = 1000;
///
///     // Set the GPU performance determinism mode
///     match amdsmi_set_gpu_perf_determinism_mode(processor_handle, clkvalue) {
///         Ok(()) => println!("GPU performance determinism mode set successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_set_gpu_perf_determinism_mode() not supported on this device"),
///         Err(e) => panic!("Failed to set GPU performance determinism mode: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_gpu_perf_determinism_mode` call fails.
pub fn amdsmi_set_gpu_perf_determinism_mode(
    processor_handle: AmdsmiProcessorHandle,
    clkvalue: u64,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_gpu_perf_determinism_mode(
        processor_handle,
        clkvalue
    ));
    Ok(())
}

/// Get the GPU overdrive level of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function returns the GPU overdrive level
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU overdrive level is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u32>` - Returns `Ok(u32)` containing the GPU overdrive level if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU overdrive level
///     match amdsmi_get_gpu_overdrive_level(processor_handle) {
///         Ok(overdrive_level) => println!("GPU Overdrive Level: {}", overdrive_level),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_overdrive_level() not supported on this device"),
///         Err(e) => panic!("Failed to get GPU overdrive level: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_overdrive_level` call fails.
pub fn amdsmi_get_gpu_overdrive_level(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<u32> {
    let mut od = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_overdrive_level(
        processor_handle,
        &mut od
    ));
    Ok(od)
}

/// Get the memory overdrive percent for the GPU device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the memory overdrive level
/// for the specified GPU device.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the memory overdrive level is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u32>` - Returns `Ok(u32)` containing the memory overdrive level if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the memory overdrive level
///     match amdsmi_get_gpu_mem_overdrive_level(processor_handle) {
///         Ok(overdrive_level) => println!("Memory overdrive level: {}", overdrive_level),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_mem_overdrive_level() not supported on this device"),
///         Err(e) => panic!("Failed to get memory overdrive level: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_mem_overdrive_level` call fails.
pub fn amdsmi_get_gpu_mem_overdrive_level(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<u32> {
    let mut od: u32 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_mem_overdrive_level(
        processor_handle,
        &mut od as *mut u32
    ));
    Ok(od)
}

/// Get the clock frequencies of the device with the specified processor handle and clock type.
///
/// Given a processor handle `processor_handle` and a clock type `clk_type`, this function returns the clock frequencies
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the clock frequencies are being queried.
/// * `clk_type` - The type of the clock to query.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiFrequenciesT>` - Returns `Ok(AmdsmiFrequenciesT)` containing the [`AmdsmiFrequenciesT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example clock type
///     let clk_type = AmdsmiClkTypeT::AmdsmiClkTypeGfx;
///
///     // Retrieve the clock frequencies
///     match amdsmi_get_clk_freq(processor_handle, clk_type) {
///         Ok(frequencies) => println!("Clock Frequencies: {:?}", frequencies),
///         Err(e) => panic!("Failed to get clock frequencies: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_clk_freq` call fails.
pub fn amdsmi_get_clk_freq(
    processor_handle: AmdsmiProcessorHandle,
    clk_type: AmdsmiClkTypeT,
) -> AmdsmiResult<AmdsmiFrequenciesT> {
    let mut frequencies = MaybeUninit::<AmdsmiFrequenciesT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_clk_freq(
        processor_handle,
        clk_type,
        frequencies.as_mut_ptr()
    ));
    let frequencies = unsafe { frequencies.assume_init() };
    Ok(frequencies)
}

/// Reset the GPU of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function resets the GPU
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU is being reset.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Reset the GPU
///     match amdsmi_reset_gpu(processor_handle) {
///         Ok(()) => println!("GPU reset successfully"),
///         Err(e) => panic!("Failed to reset GPU: {}", e),
///     }
///     // Wait for some time
///     std::thread::sleep(std::time::Duration::from_secs(5));
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_reset_gpu` call fails.
pub fn amdsmi_reset_gpu(processor_handle: AmdsmiProcessorHandle) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_reset_gpu(processor_handle));
    Ok(())
}

/// Get the GPU overdrive voltage and frequency information of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function returns the GPU overdrive voltage and frequency information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU overdrive voltage and frequency information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiOdVoltFreqDataT>` - Returns `Ok(AmdsmiOdVoltFreqDataT)` containing the [`AmdsmiOdVoltFreqDataT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU overdrive voltage and frequency information
///     match amdsmi_get_gpu_od_volt_info(processor_handle) {
///         Ok(od_volt_info) => println!("GPU Overdrive Voltage and Frequency Info: {:?}", od_volt_info),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_od_volt_info() not supported on this device"),
///         Err(e) => panic!("Failed to get GPU overdrive voltage and frequency info: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_od_volt_info` call fails.
pub fn amdsmi_get_gpu_od_volt_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiOdVoltFreqDataT> {
    let mut odv = MaybeUninit::<AmdsmiOdVoltFreqDataT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_od_volt_info(
        processor_handle,
        odv.as_mut_ptr()
    ));
    let odv = unsafe { odv.assume_init() };
    Ok(odv)
}

/// Get the GPU metrics header information of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function returns the GPU metrics header information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU metrics header information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdMetricsTableHeaderT>` - Returns `Ok(AmdMetricsTableHeaderT)` containing the [`AmdMetricsTableHeaderT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU metrics header information
///     match amdsmi_get_gpu_metrics_header_info(processor_handle) {
///         Ok(header_info) => println!("GPU Metrics Header Info: {:?}", header_info),
///         Err(e) => panic!("Failed to get GPU metrics header info: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_metrics_header_info` call fails.
pub fn amdsmi_get_gpu_metrics_header_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdMetricsTableHeaderT> {
    let mut header_value = MaybeUninit::<AmdMetricsTableHeaderT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_metrics_header_info(
        processor_handle,
        header_value.as_mut_ptr()
    ));
    let header_value = unsafe { header_value.assume_init() };
    Ok(header_value)
}

/// Get the GPU metrics information of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function returns the GPU metrics information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU metrics information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiGpuMetricsT>` - Returns `Ok(AmdsmiGpuMetricsT)` containing the [`AmdsmiGpuMetricsT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU metrics information
///     match amdsmi_get_gpu_metrics_info(processor_handle) {
///         Ok(gpu_metrics) => println!("GPU Metrics Info: {:?}", gpu_metrics),
///         Err(e) => panic!("Failed to get GPU metrics info: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_metrics_info` call fails.
pub fn amdsmi_get_gpu_metrics_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiGpuMetricsT> {
    let mut pgpu_metrics = MaybeUninit::<AmdsmiGpuMetricsT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_metrics_info(
        processor_handle,
        pgpu_metrics.as_mut_ptr()
    ));
    let pgpu_metrics = unsafe { pgpu_metrics.assume_init() };
    Ok(pgpu_metrics)
}

/// Get the GPU power management metrics information of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function returns the GPU power management metrics information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU power management metrics information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<Vec<AmdsmiNameValueT>>` - Returns `Ok(Vec<AmdsmiNameValueT>)` containing a list of [`AmdsmiNameValueT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU power management metrics information
///     match amdsmi_get_gpu_pm_metrics_info(processor_handle) {
///         Ok(pm_metrics) => {
///             println!("Number of Metrics: {}", pm_metrics.len());
///             for metric in pm_metrics {
///                 println!("Metric: {:?}", metric);
///             }
///         },
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_pm_metrics_info() not supported on this device"),
///         Err(e) => panic!("Failed to get GPU power management metrics info: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_pm_metrics_info` call fails.
pub fn amdsmi_get_gpu_pm_metrics_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<Vec<AmdsmiNameValueT>> {
    let mut pm_metrics_ptr: *mut AmdsmiNameValueT = null_mut();
    let mut num_of_metrics = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_pm_metrics_info(
        processor_handle,
        &mut pm_metrics_ptr,
        &mut num_of_metrics
    ));

    let pm_metrics_slice =
        unsafe { std::slice::from_raw_parts(pm_metrics_ptr, num_of_metrics as usize) };
    let pm_metrics = pm_metrics_slice.to_vec();

    if !pm_metrics_ptr.is_null() {
        unsafe { free(pm_metrics_ptr as *mut c_void) };
    }

    Ok(pm_metrics)
}

/// Get the GPU register table information of the device with the specified processor handle and register type.
///
/// Given a processor handle `processor_handle` and a register type `reg_type`, this function returns the GPU register table information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU register table information is being queried.
/// * `reg_type` - The [`AmdsmiRegTypeT`] type of the register to query.
///
/// # Returns
///
/// * `AmdsmiResult<Vec<AmdsmiNameValueT>>` - Returns `Ok(Vec<AmdsmiNameValueT>)` containing a list of [`AmdsmiNameValueT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example register type
///     let reg_type = AmdsmiRegTypeT::AmdsmiRegPcie;
///
///     // Retrieve the GPU register table information
///     match amdsmi_get_gpu_reg_table_info(processor_handle, reg_type) {
///         Ok(reg_metrics) => {
///             println!("Number of Metrics: {}", reg_metrics.len());
///             for metric in reg_metrics {
///                 println!("Metric: {:?}", metric);
///             }
///         },
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_reg_table_info() not supported on this device"),
///         Err(e) => panic!("Failed to get GPU register table info: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_reg_table_info` call fails.
pub fn amdsmi_get_gpu_reg_table_info(
    processor_handle: AmdsmiProcessorHandle,
    reg_type: AmdsmiRegTypeT,
) -> AmdsmiResult<Vec<AmdsmiNameValueT>> {
    let mut reg_metrics_ptr: *mut AmdsmiNameValueT = null_mut();
    let mut num_of_metrics = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_reg_table_info(
        processor_handle,
        reg_type,
        &mut reg_metrics_ptr,
        &mut num_of_metrics
    ));

    let reg_metrics_slice =
        unsafe { std::slice::from_raw_parts(reg_metrics_ptr, num_of_metrics as usize) };
    let reg_metrics = reg_metrics_slice.to_vec();

    if !reg_metrics_ptr.is_null() {
        unsafe { free(reg_metrics_ptr as *mut c_void) };
    }

    Ok(reg_metrics)
}

/// Set the GPU clock range of the device with the specified processor handle, minimum clock value, maximum clock value, and clock type.
///
/// Given a processor handle `processor_handle`, a minimum clock value `minclkvalue`, a maximum clock value `maxclkvalue`, and a clock type `clk_type`,
/// this function sets the GPU clock range for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU clock range is being set.
/// * `minclkvalue` - The minimum clock value to set.
/// * `maxclkvalue` - The maximum clock value to set.
/// * `clk_type` - The type of the clock to set.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example clock values and clock type
///     let minclkvalue = 500; // Set minimum clock value to 500 MHz
///     let maxclkvalue = 1500; // Set maximum clock value to 1500 MHz
///     let clk_type = AmdsmiClkTypeT::AmdsmiClkTypeGfx;
///
///     // Set the GPU clock range
///     match amdsmi_set_gpu_clk_range(processor_handle, minclkvalue, maxclkvalue, clk_type) {
///         Ok(()) => println!("GPU clock range set successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_set_gpu_clk_range() not supported on this device"),
///         Err(e) => panic!("Failed to set GPU clock range: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_gpu_clk_range` call fails.
pub fn amdsmi_set_gpu_clk_range(
    processor_handle: AmdsmiProcessorHandle,
    minclkvalue: u64,
    maxclkvalue: u64,
    clk_type: AmdsmiClkTypeT,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_gpu_clk_range(
        processor_handle,
        minclkvalue,
        maxclkvalue,
        clk_type
    ));
    Ok(())
}

/// Set the GPU clock limit of the device with the specified processor handle, clock type, limit type, and clock value.
///
/// Given a processor handle `processor_handle`, a clock type `clk_type`, a limit type `limit_type`, and a clock value `clk_value`,
/// this function sets the GPU clock limit for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU clock limit is being set.
/// * `clk_type` - The type of the clock to set.
/// * `limit_type` - The type of the clock limit to set.
/// * `clk_value` - The clock value to set. Frequency values are in MHz.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example clock type, limit type, and clock value
///     let clk_type = AmdsmiClkTypeT::AmdsmiClkTypeGfx;
///     let limit_type = AmdsmiClkLimitTypeT::ClkLimitMin;
///     let clk_value = 1500; // Set clock value to 1500 MHz
///
///     // Set the GPU clock limit
///     match amdsmi_set_gpu_clk_limit(processor_handle, clk_type, limit_type, clk_value) {
///         Ok(()) => println!("GPU clock limit set successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_set_gpu_clk_limit() not supported on this device"),
///         Err(e) => panic!("Failed to set GPU clock limit: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_gpu_clk_limit` call fails.
pub fn amdsmi_set_gpu_clk_limit(
    processor_handle: AmdsmiProcessorHandle,
    clk_type: AmdsmiClkTypeT,
    limit_type: AmdsmiClkLimitTypeT,
    clk_value: u64,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_gpu_clk_limit(
        processor_handle,
        clk_type,
        limit_type,
        clk_value
    ));
    Ok(())
}

/// Set the GPU overdrive clock information of the device with the specified processor handle, level, clock value, and clock type.
///
/// Given a processor handle `processor_handle`, a frequency index `level`, a clock value `clkvalue`, and a clock type `clk_type`,
/// this function sets the GPU overdrive clock information for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU overdrive clock information is being set.
/// * `level` - The frequency index level [`AmdsmiFreqIndT`] to set.
/// * `clkvalue` - The clock value to set.
/// * `clk_type` - The type of the clock [`AmdsmiClkTypeT`] to set.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example frequency index level, clock value, and clock type
///     let level = AmdsmiFreqIndT::AmdsmiFreqIndMin;
///     let clkvalue = 1500; // Set clock value to 1500 MHz
///     let clk_type = AmdsmiClkTypeT::AmdsmiClkTypeGfx;
///
///     // Set the GPU overdrive clock information
///     match amdsmi_set_gpu_od_clk_info(processor_handle, level, clkvalue, clk_type) {
///         Ok(()) => println!("GPU overdrive clock information set successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_set_gpu_od_clk_info() not supported on this device"),
///         Err(e) => panic!("Failed to set GPU overdrive clock information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_gpu_od_clk_info` call fails.
pub fn amdsmi_set_gpu_od_clk_info(
    processor_handle: AmdsmiProcessorHandle,
    level: AmdsmiFreqIndT,
    clkvalue: u64,
    clk_type: AmdsmiClkTypeT,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_gpu_od_clk_info(
        processor_handle,
        level,
        clkvalue,
        clk_type
    ));
    Ok(())
}

/// Set the GPU overdrive voltage information of the device with the specified processor handle, voltage point, clock value, and voltage value.
///
/// Given a processor handle `processor_handle`, a voltage point `vpoint`, a clock value `clkvalue`, and a voltage value `voltvalue`,
/// this function sets the GPU overdrive voltage information for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU overdrive voltage information is being set.
/// * `vpoint` - The voltage point [0|1|2] on the voltage curve to set.
/// * `clkvalue` - The clock value component of voltage curve point to set.
/// * `voltvalue` - The voltage value component of voltage curve point to set.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example voltage point, clock value, and voltage value
///     let vpoint = 1;
///     let clkvalue = 1000;
///     let voltvalue = 980;
///
///     // Set the GPU overdrive voltage information
///     match amdsmi_set_gpu_od_volt_info(processor_handle, vpoint, clkvalue, voltvalue) {
///         Ok(()) => println!("GPU overdrive voltage information set successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_set_gpu_od_volt_info() not supported on this device"),
///         Err(e) => panic!("Failed to set GPU overdrive voltage information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_gpu_od_volt_info` call fails.
pub fn amdsmi_set_gpu_od_volt_info(
    processor_handle: AmdsmiProcessorHandle,
    vpoint: u32,
    clkvalue: u64,
    voltvalue: u64,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_gpu_od_volt_info(
        processor_handle,
        vpoint,
        clkvalue,
        voltvalue
    ));
    Ok(())
}

/// Get the GPU overdrive voltage curve regions of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function returns the GPU current valid overdrive voltage
/// curve regions for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU overdrive voltage curve regions are being queried.
///
/// # Returns
///
/// * `AmdsmiResult<Vec<AmdsmiFreqVoltRegionT>>` - Returns `Ok(Vec<AmdsmiFreqVoltRegionT>)` containing a list of [`AmdsmiFreqVoltRegionT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU overdrive voltage curve regions
///     match amdsmi_get_gpu_od_volt_curve_regions(processor_handle) {
///         Ok(regions) => {
///             println!("Number of Regions: {}", regions.len());
///             for region in regions {
///                 println!("Region: {:?}", region);
///             }
///         },
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_od_volt_curve_regions() not supported on this device"),
///         Err(e) => panic!("Failed to get GPU overdrive voltage curve regions: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_od_volt_curve_regions` call fails.
pub fn amdsmi_get_gpu_od_volt_curve_regions(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<Vec<AmdsmiFreqVoltRegionT>> {
    // First call to get the number of regions
    let mut num_regions = MaybeUninit::<u32>::uninit();

    // First call to get the number of regions
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_od_volt_curve_regions(
        processor_handle,
        num_regions.as_mut_ptr(),
        std::ptr::null_mut()
    ));

    let num_regions = unsafe { num_regions.assume_init() };

    // Allocate a vector with the capacity of num_regions
    let mut buffer: Vec<AmdsmiFreqVoltRegionT> = Vec::with_capacity(num_regions as usize);

    // Second call to get the actual data
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_od_volt_curve_regions(
        processor_handle,
        &mut (num_regions as u32),
        buffer.as_mut_ptr()
    ));

    // Set the length of the vector to num_regions
    unsafe { buffer.set_len(num_regions as usize) };

    Ok(buffer)
}

/// Get the GPU power profile presets of the device with the specified processor handle and sensor index.
///
/// Given a processor handle `processor_handle` and a sensor index `sensor_ind`, this function returns the GPU power profile presets
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU power profile presets are being queried.
/// * `sensor_ind` - The sensor index to query.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiPowerProfileStatusT>` - Returns `Ok(AmdsmiPowerProfileStatusT)` containing the [`AmdsmiPowerProfileStatusT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example sensor index
///     let sensor_ind = 0; // Sensor index 0
///
///     // Retrieve the GPU power profile presets
///     match amdsmi_get_gpu_power_profile_presets(processor_handle, sensor_ind) {
///         Ok(status) => println!("GPU Power Profile Status: {:?}", status),
///         Err(e) => panic!("Failed to get GPU power profile presets: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_power_profile_presets` call fails.
pub fn amdsmi_get_gpu_power_profile_presets(
    processor_handle: AmdsmiProcessorHandle,
    sensor_ind: u32,
) -> AmdsmiResult<AmdsmiPowerProfileStatusT> {
    let mut status = MaybeUninit::<AmdsmiPowerProfileStatusT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_power_profile_presets(
        processor_handle,
        sensor_ind,
        status.as_mut_ptr()
    ));
    let status = unsafe { status.assume_init() };
    Ok(status)
}

/// Set the GPU performance level of the device with the specified processor handle and performance level.
///
/// Given a processor handle `processor_handle` and a performance level `perf_lvl`, this function sets the GPU performance level
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU performance level is being set.
/// * `perf_lvl` - The performance level [`AmdsmiDevPerfLevelT`] to set.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example performance level
///     let perf_lvl = AmdsmiDevPerfLevelT::AmdsmiDevPerfLevelHigh;
///
///     // Set the GPU performance level
///     match amdsmi_set_gpu_perf_level(processor_handle, perf_lvl) {
///         Ok(()) => println!("GPU performance level set successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_set_gpu_perf_level() not supported on this device"),
///         Err(e) => panic!("Failed to set GPU performance level: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_gpu_perf_level` call fails.
pub fn amdsmi_set_gpu_perf_level(
    processor_handle: AmdsmiProcessorHandle,
    perf_lvl: AmdsmiDevPerfLevelT,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_gpu_perf_level(
        processor_handle,
        perf_lvl
    ));
    Ok(())
}

/// Set the GPU overdrive percent of the device with the specified processor handle and overdrive level.
///
/// Given a processor handle `processor_handle` and an overdrive level `od`, this function sets the GPU overdrive level
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU overdrive level is being set.
/// * `od` - The overdrive level to set.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example overdrive level
///     let od = 0; // Set overdrive level to 0
///
///     // Set the GPU overdrive level
///     match amdsmi_set_gpu_overdrive_level(processor_handle, od) {
///         Ok(()) => println!("GPU overdrive level set successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_set_gpu_overdrive_level() not supported on this device"),
///         Err(e) => panic!("Failed to set GPU overdrive level: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_gpu_overdrive_level` call fails.
pub fn amdsmi_set_gpu_overdrive_level(
    processor_handle: AmdsmiProcessorHandle,
    od: u32,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_gpu_overdrive_level(
        processor_handle,
        od
    ));
    Ok(())
}

/// Set the clock frequency of the device with the specified processor handle, clock type, and frequency bitmask.
///
/// Given a processor handle `processor_handle`, a clock type `clk_type`, and a frequency bitmask `freq_bitmask`,
/// this function sets the clock frequency for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the clock frequency is being set.
/// * `clk_type` - The type of the clock to set.
/// * `freq_bitmask` - The frequency bitmask to set. A bitmask indicating the indices of the frequencies
/// that are to be enabled (1) and disabled (0). Only the lowest `AmdsmiFrequenciesT.num_supported`
/// bits of this mask are relevant.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example clock type and frequency bitmask
///     let clk_type = AmdsmiClkTypeT::AmdsmiClkTypeGfx;
///     let freq_bitmask = 0; // Example frequency bitmask
///
///     // Set the clock frequency
///     match amdsmi_set_clk_freq(processor_handle, clk_type, freq_bitmask) {
///         Ok(()) => println!("Clock frequency set successfully"),
///         Err(e) => panic!("Failed to set clock frequency: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_clk_freq` call fails.
pub fn amdsmi_set_clk_freq(
    processor_handle: AmdsmiProcessorHandle,
    clk_type: AmdsmiClkTypeT,
    freq_bitmask: u64,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_clk_freq(
        processor_handle,
        clk_type,
        freq_bitmask
    ));
    Ok(())
}

/// Get the SoC P-state of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function returns the SoC P-state
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the SoC P-state is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiDpmPolicyT>` - Returns `Ok(AmdsmiDpmPolicyT)` containing the [`AmdsmiDpmPolicyT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the SoC P-state
///     match amdsmi_get_soc_pstate(processor_handle) {
///         Ok(policy) => println!("SoC P-state: {:?}", policy),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_soc_pstate() not supported on this device"),
///         Err(e) => panic!("Failed to get SoC P-state: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_soc_pstate` call fails.
pub fn amdsmi_get_soc_pstate(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiDpmPolicyT> {
    let mut policy = MaybeUninit::<AmdsmiDpmPolicyT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_soc_pstate(
        processor_handle,
        policy.as_mut_ptr()
    ));
    let policy = unsafe { policy.assume_init() };
    Ok(policy)
}

/// Set the SoC P-state of the device with the specified processor handle and policy ID.
///
/// Given a processor handle `processor_handle` and a policy ID `policy_id`, this function sets the SoC P-state
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the SoC P-state is being set.
/// * `policy_id` - The policy ID to set. The id is the id in [`AmdsmiDpmPolicyEntryT`], which can be obtained by calling amdsmi_get_soc_pstate().
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example policy ID
///     let policy_id = 0;
///
///     // Set the SoC P-state
///     match amdsmi_set_soc_pstate(processor_handle, policy_id) {
///         Ok(()) => println!("SoC P-state set successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_set_soc_pstate() not supported on this device"),
///         Err(e) => panic!("Failed to set SoC P-state: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_soc_pstate` call fails.
pub fn amdsmi_set_soc_pstate(
    processor_handle: AmdsmiProcessorHandle,
    policy_id: u32,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_soc_pstate(
        processor_handle,
        policy_id
    ));
    Ok(())
}

/// Get the XGMI per-link power down (PLPD) policy of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function returns the XGMI PLPD policy
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the XGMI PLPD policy is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiDpmPolicyT>` - Returns `Ok(AmdsmiDpmPolicyT)` containing the [`AmdsmiDpmPolicyT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the XGMI PLPD policy
///     match amdsmi_get_xgmi_plpd(processor_handle) {
///         Ok(xgmi_plpd) => println!("XGMI PLPD policy: {:?}", xgmi_plpd),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_xgmi_plpd() not supported on this device"),
///         Err(e) => panic!("Failed to get XGMI PLPD policy: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_xgmi_plpd` call fails.
pub fn amdsmi_get_xgmi_plpd(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiDpmPolicyT> {
    let mut xgmi_plpd = MaybeUninit::<AmdsmiDpmPolicyT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_xgmi_plpd(
        processor_handle,
        xgmi_plpd.as_mut_ptr()
    ));
    let xgmi_plpd = unsafe { xgmi_plpd.assume_init() };
    Ok(xgmi_plpd)
}

/// Set the XGMI per-link power down (PLPD) policy of the device with the specified processor handle and PLPD ID.
///
/// Given a processor handle `processor_handle` and a PLPD ID `plpd_id`, this function sets the XGMI PLPD policy
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the XGMI PLPD policy is being set.
/// * `plpd_id` - The PLPD ID to set. The id is the id in [`AmdsmiDpmPolicyEntryT`], which can be obtained by calling amdsmi_get_xgmi_plpd().
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example PLPD ID
///     let plpd_id = 0;
///
///     // Set the XGMI PLPD policy
///     match amdsmi_set_xgmi_plpd(processor_handle, plpd_id) {
///         Ok(()) => println!("XGMI PLPD policy set successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_set_xgmi_plpd() not supported on this device"),
///         Err(e) => panic!("Failed to set XGMI PLPD policy: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_xgmi_plpd` call fails.
pub fn amdsmi_set_xgmi_plpd(
    processor_handle: AmdsmiProcessorHandle,
    plpd_id: u32,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_xgmi_plpd(
        processor_handle,
        plpd_id
    ));
    Ok(())
}

/// Get the GPU process isolation status of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function returns the integer GPU process isolation status:
/// 0 - disabled, 1 - enabled.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU process isolation status is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u32>` - Returns `Ok(u32)` containing the GPU process isolation status if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU process isolation status
///     match amdsmi_get_gpu_process_isolation(processor_handle) {
///         Ok(pisolate) => println!("GPU process isolation status: {}", pisolate),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_process_isolation() not supported on this device"),
///         Err(e) => panic!("Failed to get GPU process isolation status: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_process_isolation` call fails.
pub fn amdsmi_get_gpu_process_isolation(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<u32> {
    let mut pisolate = MaybeUninit::<u32>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_process_isolation(
        processor_handle,
        pisolate.as_mut_ptr()
    ));
    let pisolate = unsafe { pisolate.assume_init() };
    Ok(pisolate)
}

/// Enable/disable the system Process Isolation for the given device handle.
///
/// Given a processor handle `processor_handle` and an isolation value `pisolate`, this function sets the GPU process isolation status
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU process isolation status is being set.
/// * `pisolate` - The isolation value to set. 0 is the process isolation disabled, and 1 is the process isolation enabled.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example isolation value
///     let pisolate = 1; // Set isolation to enabled
///
///     // Set the GPU process isolation status
///     match amdsmi_set_gpu_process_isolation(processor_handle, pisolate) {
///         Ok(()) => println!("GPU process isolation status set successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_set_gpu_process_isolation() not supported on this device"),
///         Err(e) => panic!("Failed to set GPU process isolation status: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_gpu_process_isolation` call fails.
pub fn amdsmi_set_gpu_process_isolation(
    processor_handle: AmdsmiProcessorHandle,
    pisolate: u32,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_gpu_process_isolation(
        processor_handle,
        pisolate
    ));
    Ok(())
}

/// Clean the GPU local data of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function cleans the GPU local data
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU local data is being cleaned.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Clean the GPU local data
///     match amdsmi_clean_gpu_local_data(processor_handle) {
///         Ok(()) => println!("GPU local data cleaned successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_clean_gpu_local_data() not supported on this device"),
///         Err(e) => panic!("Failed to clean GPU local data: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_clean_gpu_local_data` call fails.
pub fn amdsmi_clean_gpu_local_data(processor_handle: AmdsmiProcessorHandle) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_clean_gpu_local_data(
        processor_handle
    ));
    Ok(())
}

/// Get the GPU ECC error count for a specific block of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle` and a GPU block `block`, this function returns the ECC error count
/// for the specified block of the processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the ECC error count is being queried.
/// * `block` - The GPU block for which the ECC error count is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiErrorCountT>` - Returns `Ok(AmdsmiErrorCountT)` containing the [`AmdsmiErrorCountT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example GPU block
///     let block = AmdsmiGpuBlockT::AmdsmiGpuBlockGfx;
///
///     // Retrieve the GPU ECC error count
///     match amdsmi_get_gpu_ecc_count(processor_handle, block) {
///         Ok(ec) => println!("GPU ECC error count: {:?}", ec),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_ecc_count() not supported on this device"),
///         Err(e) => panic!("Failed to get GPU ECC error count: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_ecc_count` call fails.
pub fn amdsmi_get_gpu_ecc_count(
    processor_handle: AmdsmiProcessorHandle,
    block: AmdsmiGpuBlockT,
) -> AmdsmiResult<AmdsmiErrorCountT> {
    let mut ec = MaybeUninit::<AmdsmiErrorCountT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_ecc_count(
        processor_handle,
        block,
        ec.as_mut_ptr()
    ));
    let ec = unsafe { ec.assume_init() };
    Ok(ec)
}

/// Retrieve the enabled ECC bit-mask.
///
/// Given a processor handle `processor_handle`, this function returns the ECC enabled
/// ECC bit-mask for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the ECC enabled bit-mask are being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u64>` - Returns `Ok(u64)` containing the ECC enabled bit-mask if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU ECC enabled bit-mask
///     match amdsmi_get_gpu_ecc_enabled(processor_handle) {
///         Ok(enabled_mask) => println!("GPU ECC enabled: {}", enabled_mask),
///         Err(e) => panic!("Failed to get GPU ECC enabled: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_ecc_enabled` call fails.
pub fn amdsmi_get_gpu_ecc_enabled(processor_handle: AmdsmiProcessorHandle) -> AmdsmiResult<u64> {
    let mut enabled_mask = MaybeUninit::<u64>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_ecc_enabled(
        processor_handle,
        enabled_mask.as_mut_ptr()
    ));
    let enabled_mask = unsafe { enabled_mask.assume_init() };
    Ok(enabled_mask)
}

/// Get the GPU ECC status for a specific block of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle` and a GPU block `block`, this function returns the ECC status
/// for the specified block of the processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the ECC status is being queried.
/// * `block` - The GPU block for which the ECC status is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiRasErrStateT>` - Returns `Ok(AmdsmiRasErrStateT)` containing the [`AmdsmiRasErrStateT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example GPU block
///     let block = AmdsmiGpuBlockT::AmdsmiGpuBlockGfx;
///
///     // Retrieve the GPU ECC status
///     match amdsmi_get_gpu_ecc_status(processor_handle, block) {
///         Ok(state) => println!("GPU ECC status: {:?}", state),
///         Err(e) => panic!("Failed to get GPU ECC status: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_ecc_status` call fails.
pub fn amdsmi_get_gpu_ecc_status(
    processor_handle: AmdsmiProcessorHandle,
    block: AmdsmiGpuBlockT,
) -> AmdsmiResult<AmdsmiRasErrStateT> {
    let mut state = MaybeUninit::<AmdsmiRasErrStateT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_ecc_status(
        processor_handle,
        block,
        state.as_mut_ptr()
    ));
    let state = unsafe { state.assume_init() };
    Ok(state)
}

/// Convert an AMD SMI status code to a string representation.
///
/// Given a status code `status`, this function returns the string representation of the status code.
///
/// # Arguments
///
/// * `status` - The AMD SMI status code to convert.
///
/// # Returns
///
/// * `AmdsmiResult<String>` - Returns `Ok(String)` containing the string representation of the status code if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
///     // Example status code
///     let status = AmdsmiStatusT::AmdsmiStatusNotSupported;
///
///     // Convert the status code to a string
///     match amdsmi_status_code_to_string(status) {
///         Ok(status_str) => println!("Status string: {}", status_str),
///         Err(e) => panic!("Failed to convert status code to string: {}", e),
///     }
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_status_code_to_string` call fails.
pub fn amdsmi_status_code_to_string(status: AmdsmiStatusT) -> AmdsmiResult<String> {
    let mut status_string: *const ::std::os::raw::c_char = null_mut();
    call_unsafe!(amdsmi_wrapper::amdsmi_status_code_to_string(
        status,
        &mut status_string
    ));

    let c_str = unsafe { std::ffi::CStr::from_ptr(status_string) };
    let string = c_str.to_string_lossy().into_owned();

    Ok(string)
}

/// Check if a GPU counter group is supported for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle` and an event group `group`, this function checks if the specified GPU counter group
/// is supported for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU counter group support is being queried.
/// * `group` - The event group to check for support.
///
/// # Returns
///
/// * `AmdsmiResult<bool>` - Returns `Ok(bool)` indicating whether the GPU counter group is supported if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example event group
///     let group = AmdsmiEventGroupT::AmdsmiEvntGrpXgmi;
///
///     // Check if the GPU counter group is supported
///     match amdsmi_gpu_counter_group_supported(processor_handle, group) {
///         Ok(supported) => println!("GPU counter group supported: {}", supported),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_gpu_counter_group_supported() not supported on this device"),
///         Err(e) => panic!("Failed to check GPU counter group support: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_gpu_counter_group_supported` call fails.
pub fn amdsmi_gpu_counter_group_supported(
    processor_handle: AmdsmiProcessorHandle,
    group: AmdsmiEventGroupT,
) -> AmdsmiResult<bool> {
    call_unsafe!(amdsmi_wrapper::amdsmi_gpu_counter_group_supported(
        processor_handle,
        group
    ));
    // Here amdsmi_wrapper::amdsmi_gpu_counter_group_supported return successfull means supported.
    Ok(true)
}

/// Creates a GPU counter for the specified event type on the device with the specified processor handle.
///
/// This function creates a GPU counter for the specified event type on the specified processor handle,
/// allowing the user to monitor specific GPU events.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU counter is being created.
/// * `event_type` - The type of event for which the GPU counter is being created.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiEventHandleT>` - Returns `Ok(AmdsmiEventHandleT)` containing the [`AmdsmiEventHandleT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Check if the event group is supported
///     let event_group = AmdsmiEventGroupT::AmdsmiEvntGrpXgmi;
///     match amdsmi_gpu_counter_group_supported(processor_handle, event_group) {
///         Ok(_) => println!("Event group supported."),
///         Err(e) => {
///             assert!( e == AmdsmiStatusT::AmdsmiStatusNotSupported, "Event group not supported: {}", e);
///         }
///     }
///
///     // Check available GPU counters
///     let available_counters = match amdsmi_get_gpu_available_counters(processor_handle, event_group) {
///         Ok(available_counters) => available_counters,
///         Err(e) => {
///             assert!( e == AmdsmiStatusT::AmdsmiStatusNotSupported, "Failed to get available GPU counters: {}", e);
///             0
///         }
///     };
///
///     if available_counters == 0 {
///         println!("No available GPU counters or event group not supported.");
///         amdsmi_shut_down().expect("Failed to shut down AMD SMI");
///         return;
///     }
///
///     // Example event type
///     let event_type = AmdsmiEventTypeT::AmdsmiEvntXgmi0RequestTx;
///
///     // Create a GPU counter
///     match amdsmi_gpu_create_counter(processor_handle, event_type) {
///         Ok(event_handle) => {
///             println!("GPU counter created with handle: {:?}", event_handle);
///
///             // Start the GPU counter
///             let cmd_start = AmdsmiCounterCommandT::AmdsmiCntrCmdStart;
///             amdsmi_gpu_control_counter(event_handle, cmd_start).expect("Failed to start GPU counter");
///
///             // Wait for some time
///             std::thread::sleep(std::time::Duration::from_secs(1));
///
///             // Read the GPU counter value
///             let value = amdsmi_gpu_read_counter(event_handle).expect("Failed to read GPU counter");
///             println!("GPU counter value: {:?}", value);
///
///             // Stop the GPU counter
///             let cmd_stop = AmdsmiCounterCommandT::AmdsmiCntrCmdStop;
///             amdsmi_gpu_control_counter(event_handle, cmd_stop).expect("Failed to stop GPU counter");
///
///             // Destroy the GPU counter
///             amdsmi_gpu_destroy_counter(event_handle).expect("Failed to destroy GPU counter");
///         },
///         Err(e) => eprintln!("Failed to create GPU counter: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_gpu_create_counter` call fails.
pub fn amdsmi_gpu_create_counter(
    processor_handle: AmdsmiProcessorHandle,
    event_type: AmdsmiEventTypeT,
) -> AmdsmiResult<AmdsmiEventHandleT> {
    let mut evnt_handle = MaybeUninit::<AmdsmiEventHandleT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_gpu_create_counter(
        processor_handle,
        event_type,
        evnt_handle.as_mut_ptr()
    ));
    let evnt_handle = unsafe { evnt_handle.assume_init() };
    Ok(evnt_handle)
}

/// Destroy a GPU counter with the specified event handle.
///
/// Given an event handle `evnt_handle`, this function destroys the GPU counter.
///
/// # Arguments
///
/// * `evnt_handle` - The event handle of the GPU counter to be destroyed.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// Refer to the example in the documentation for [`amdsmi_gpu_create_counter`].
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_gpu_destroy_counter` call fails.
pub fn amdsmi_gpu_destroy_counter(evnt_handle: AmdsmiEventHandleT) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_gpu_destroy_counter(evnt_handle));
    Ok(())
}

/// Control a GPU counter with the specified event handle and command.
///
/// Given an event handle `evt_handle`, a command `cmd`,  this function controls the GPU counter.
///
/// # Arguments
///
/// * `evt_handle` - The event handle of the GPU counter to be controlled.
/// * `cmd` - The command to be executed on the GPU counter.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// Refer to the example in the documentation for [`amdsmi_gpu_create_counter`].
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_gpu_control_counter` call fails.
pub fn amdsmi_gpu_control_counter(
    evt_handle: AmdsmiEventHandleT,
    cmd: AmdsmiCounterCommandT,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_gpu_control_counter(
        evt_handle,
        cmd,
        std::ptr::null_mut()
    ));
    Ok(())
}

/// Read the value of a GPU counter with the specified event handle.
///
/// Given an event handle `evt_handle`, this function reads the value of the GPU counter.
///
/// # Arguments
///
/// * `evt_handle` - The event handle of the GPU counter to be read.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiCounterValueT>` - Returns `Ok(AmdsmiCounterValueT)` containing the [`AmdsmiCounterValueT`] if successful, or an error if it fails.
///
/// # Example
///
/// Refer to the example in the documentation for [`amdsmi_gpu_create_counter`].
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_gpu_read_counter` call fails.
pub fn amdsmi_gpu_read_counter(
    evt_handle: AmdsmiEventHandleT,
) -> AmdsmiResult<AmdsmiCounterValueT> {
    let mut value = MaybeUninit::<AmdsmiCounterValueT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_gpu_read_counter(
        evt_handle,
        value.as_mut_ptr()
    ));
    let value = unsafe { value.assume_init() };
    Ok(value)
}

/// Get the number of available GPU counters for the device with the specified processor handle and event group.
///
/// Given a processor handle `processor_handle` and an event group `grp`, this function retrieves the number of available GPU counters
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the available GPU counters are being queried.
/// * `grp` - The event group for which the available GPU counters are being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u32>` - Returns `Ok(u32)` containing the number of available GPU counters if successful, or an error if it fails.
///
/// # Example
///
/// Refer to the example in the documentation for [`amdsmi_gpu_create_counter`].
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_available_counters` call fails.
pub fn amdsmi_get_gpu_available_counters(
    processor_handle: AmdsmiProcessorHandle,
    grp: AmdsmiEventGroupT,
) -> AmdsmiResult<u32> {
    let mut available: u32 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_available_counters(
        processor_handle,
        grp,
        &mut available as *mut u32
    ));
    Ok(available)
}

/// Get the GPU compute process information.
///
/// This function retrieves information about GPU compute processes.
///
/// # Arguments
///
/// * `procs` - A mutable pointer to an array of `AmdsmiProcessInfoT` where the process information will be stored.
/// * `num_items` - A mutable pointer to a `u32` where the number of items will be stored.
///
/// # Returns
///
/// * `AmdsmiResult<(Vec<AmdsmiProcessInfoT>, u32)>` - Returns `Ok((Vec<AmdsmiProcessInfoT>, u32))` containing a list of [`AmdsmiProcessInfoT`] and the number of items if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Get the GPU compute process information
///     match amdsmi_get_gpu_compute_process_info() {
///         Ok((procs, num_items)) => {
///             println!("Number of GPU compute processes: {}", num_items);
///             for proc in procs {
///                 println!("{:?}", proc);
///             }
///         },
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_compute_process_info() not supported on this device"),
///         Err(e) => panic!("Failed to get GPU compute process information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_compute_process_info` call fails.
pub fn amdsmi_get_gpu_compute_process_info() -> AmdsmiResult<(Vec<AmdsmiProcessInfoT>, u32)> {
    let mut num_items = MaybeUninit::<u32>::uninit();

    // First call to get the number of items
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_compute_process_info(
        std::ptr::null_mut(),
        num_items.as_mut_ptr()
    ));

    let num_items = unsafe { num_items.assume_init() };
    let mut procs: Vec<AmdsmiProcessInfoT> = Vec::with_capacity(num_items as usize);

    // Second call to get the actual process information
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_compute_process_info(
        procs.as_mut_ptr(),
        &mut (num_items as u32)
    ));

    unsafe { procs.set_len(num_items as usize) };

    Ok((procs, num_items))
}

/// Get the GPU compute process information by PID.
///
/// This function retrieves information about a specific GPU compute process identified by its PID.
///
/// # Arguments
///
/// * `pid` - The process ID of the GPU compute process.
/// * `proc_` - A mutable pointer to an `AmdsmiProcessInfoT` where the process information will be stored.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiProcessInfoT>` - Returns `Ok(AmdsmiProcessInfoT)` containing the [`AmdsmiProcessInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust, should_panic
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example PID,
///     let pid = 12345;
///
///     // Get the process information by PID
///     match amdsmi_get_gpu_compute_process_info_by_pid(pid) {
///         Ok(proc_info) => println!("{:?}", proc_info),
///         Err(e) => panic!("Failed to get GPU compute process information by PID: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_compute_process_info_by_pid` call fails.
pub fn amdsmi_get_gpu_compute_process_info_by_pid(pid: u32) -> AmdsmiResult<AmdsmiProcessInfoT> {
    let mut proc_info = MaybeUninit::<AmdsmiProcessInfoT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_compute_process_info_by_pid(
        pid,
        proc_info.as_mut_ptr()
    ));
    let proc_info = unsafe { proc_info.assume_init() };
    Ok(proc_info)
}

/// Get the device indices currently being used by a process.
///
/// This function retrieves the GPU indices for a specific GPU compute process identified by its PID.
///
/// # Arguments
///
/// * `pid` - The process ID of the GPU compute process.
///
/// # Returns
///
/// * `AmdsmiResult<Vec<u32>>` - Returns a vector containing the GPU indices if successful, or an error if it fails.
///
/// # Example
///
/// ```rust,should_panic
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example PID, assuming a valid PID for a GPU compute process
///     let pid = 12345;
///
///     // Retrieve the GPU indices for the process
///     match amdsmi_get_gpu_compute_process_gpus(pid) {
///         Ok(gpu_indices) => {
///             for index in gpu_indices {
///                 println!("GPU Index: {}", index);
///             }
///         },
///         Err(e) => panic!("Failed to get GPU compute process devices: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_compute_process_gpus` call fails.
pub fn amdsmi_get_gpu_compute_process_gpus(pid: u32) -> AmdsmiResult<Vec<u32>> {
    let mut num_devices: u32 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_compute_process_gpus(
        pid,
        std::ptr::null_mut(),
        &mut num_devices
    ));

    let mut devices: Vec<u32> = Vec::with_capacity(num_devices as usize);
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_compute_process_gpus(
        pid,
        devices.as_mut_ptr(),
        &mut num_devices
    ));
    unsafe { devices.set_len(num_devices as usize) };

    Ok(devices)
}

/// Get the XGMI error status of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the XGMI error status
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the XGMI error status is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiXgmiStatusT>` - Returns `Ok(AmdsmiXgmiStatusT)` containing the [`AmdsmiXgmiStatusT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the XGMI error status
///     match amdsmi_gpu_xgmi_error_status(processor_handle) {
///         Ok(status) => println!("XGMI error status: {:?}", status),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_gpu_xgmi_error_status() not supported on this device"),
///         Err(e) => panic!("Failed to get XGMI error status: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_gpu_xgmi_error_status` call fails.
pub fn amdsmi_gpu_xgmi_error_status(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiXgmiStatusT> {
    let mut status = MaybeUninit::<AmdsmiXgmiStatusT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_gpu_xgmi_error_status(
        processor_handle,
        status.as_mut_ptr()
    ));
    let status = unsafe { status.assume_init() };
    Ok(status)
}

/// Reset the XGMI error status of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function resets the XGMI error status
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the XGMI error status is being reset.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Reset the XGMI error status
///     match amdsmi_reset_gpu_xgmi_error(processor_handle) {
///         Ok(()) => println!("XGMI error status reset successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_reset_gpu_xgmi_error() not supported on this device"),
///         Err(e) => panic!("Failed to reset XGMI error status: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_reset_gpu_xgmi_error` call fails.
pub fn amdsmi_reset_gpu_xgmi_error(processor_handle: AmdsmiProcessorHandle) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_reset_gpu_xgmi_error(
        processor_handle
    ));
    Ok(())
}

/// Get the link metrics of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the link metrics
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the link metrics are being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiLinkMetricsT>` - Returns `Ok(AmdsmiLinkMetricsT)` containing the [`AmdsmiLinkMetricsT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the link metrics
///     match amdsmi_get_link_metrics(processor_handle) {
///         Ok(link_metrics) => println!("Link metrics number: {}", link_metrics.num_links),
///         Err(e) => panic!("Failed to get link metrics: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_link_metrics` call fails.
pub fn amdsmi_get_link_metrics(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiLinkMetricsT> {
    let mut link_metrics = MaybeUninit::<AmdsmiLinkMetricsT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_link_metrics(
        processor_handle,
        link_metrics.as_mut_ptr()
    ));
    let link_metrics = unsafe { link_metrics.assume_init() };
    Ok(link_metrics)
}

/// Get the NUMA node number of the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the NUMA node number
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the NUMA node number is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<u32>` - Returns `Ok(u32)` containing the NUMA node number if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the NUMA node number
///     match amdsmi_topo_get_numa_node_number(processor_handle) {
///         Ok(numa_node) => println!("NUMA node number: {}", numa_node),
///         Err(e) => panic!("Failed to get NUMA node number: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_topo_get_numa_node_number` call fails.
pub fn amdsmi_topo_get_numa_node_number(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<u32> {
    let mut numa_node: u32 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_topo_get_numa_node_number(
        processor_handle,
        &mut numa_node
    ));
    Ok(numa_node)
}

/// Get the link weight between two processors with the specified processor handles.
///
/// Given source and destination processor handles `processor_handle_src` and `processor_handle_dst`,
/// this function retrieves the link weight between the two processors.
///
/// # Arguments
///
/// * `processor_handle_src` - A handle to the source processor.
/// * `processor_handle_dst` - A handle to the destination processor.
///
/// # Returns
///
/// * `AmdsmiResult<u64>` - Returns `Ok(u64)` containing the link weight if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor handles, assuming the number of processors is greater than one
///     let processor_handle_src = amdsmi_get_processor_handles!()[0];
///     let processor_handle_dst = amdsmi_get_processor_handles!()[1];
///
///     // Retrieve the link weight
///     match amdsmi_topo_get_link_weight(processor_handle_src, processor_handle_dst) {
///         Ok(weight) => println!("Link weight: {}", weight),
///         Err(e) => panic!("Failed to get link weight: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_topo_get_link_weight` call fails.
pub fn amdsmi_topo_get_link_weight(
    processor_handle_src: AmdsmiProcessorHandle,
    processor_handle_dst: AmdsmiProcessorHandle,
) -> AmdsmiResult<u64> {
    let mut weight: u64 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_topo_get_link_weight(
        processor_handle_src,
        processor_handle_dst,
        &mut weight
    ));
    Ok(weight)
}

/// Get the minimum and maximum bandwidth between two processors with the specified processor handles.
///
/// Given source and destination processor handles `processor_handle_src` and `processor_handle_dst`,
/// this function retrieves the minimum and maximum bandwidth between the two processors.
///
/// # Arguments
///
/// * `processor_handle_src` - A handle to the source processor.
/// * `processor_handle_dst` - A handle to the destination processor.
///
/// # Returns
///
/// * `AmdsmiResult<(u64, u64)>` - Returns `Ok((u64, u64))` containing the minimum and maximum bandwidth if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor handles, assuming the number of processors is greater than one
///     let processor_handle_src = amdsmi_get_processor_handles!()[0];
///     let processor_handle_dst = amdsmi_get_processor_handles!()[1];
///
///     // Retrieve the minimum and maximum bandwidth
///     match amdsmi_get_minmax_bandwidth_between_processors(processor_handle_src, processor_handle_dst) {
///         Ok((min_bandwidth, max_bandwidth)) => {
///             println!("Minimum bandwidth: {}", min_bandwidth);
///             println!("Maximum bandwidth: {}", max_bandwidth);
///         },
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_minmax_bandwidth_between_processors() not supported on this device"),
///         Err(e) => panic!("Failed to get bandwidth: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_minmax_bandwidth_between_processors` call fails.
pub fn amdsmi_get_minmax_bandwidth_between_processors(
    processor_handle_src: AmdsmiProcessorHandle,
    processor_handle_dst: AmdsmiProcessorHandle,
) -> AmdsmiResult<(u64, u64)> {
    let mut min_bandwidth: u64 = 0;
    let mut max_bandwidth: u64 = 0;
    call_unsafe!(
        amdsmi_wrapper::amdsmi_get_minmax_bandwidth_between_processors(
            processor_handle_src,
            processor_handle_dst,
            &mut min_bandwidth,
            &mut max_bandwidth
        )
    );
    Ok((min_bandwidth, max_bandwidth))
}

/// Check if P2P access is possible between two processors with the specified processor handles.
///
/// Given source and destination processor handles `processor_handle_src` and `processor_handle_dst`,
/// this function checks if P2P (peer-to-peer) access is possible between the two processors.
///
/// # Arguments
///
/// * `processor_handle_src` - A handle to the source processor.
/// * `processor_handle_dst` - A handle to the destination processor.
///
/// # Returns
///
/// * `AmdsmiResult<bool>` - Returns `Ok(bool)` indicating whether P2P access is possible if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor handles, assuming the number of processors is greater than one
///     let processor_handle_src = amdsmi_get_processor_handles!()[0];
///     let processor_handle_dst = amdsmi_get_processor_handles!()[1];
///
///     // Check if P2P access is possible
///     match amdsmi_is_p2p_accessible(processor_handle_src, processor_handle_dst) {
///         Ok(accessible) => println!("P2P access is possible: {}", accessible),
///         Err(e) => panic!("Failed to check P2P access: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_is_P2P_accessible` call fails.
pub fn amdsmi_is_p2p_accessible(
    processor_handle_src: AmdsmiProcessorHandle,
    processor_handle_dst: AmdsmiProcessorHandle,
) -> AmdsmiResult<bool> {
    let mut accessible: bool = false;
    call_unsafe!(amdsmi_wrapper::amdsmi_is_P2P_accessible(
        processor_handle_src,
        processor_handle_dst,
        &mut accessible
    ));
    Ok(accessible)
}

/// Retrieves the link type and number of hops between two processors.
///
/// This function retrieves the link type and number of hops between two processors,
/// returning the results in a tuple.
///
/// # Arguments
///
/// * `processor_handle_src` - A handle to the source processor.
/// * `processor_handle_dst` - A handle to the destination processor.
///
/// # Returns
///
/// * `AmdsmiResult<(u64, AmdsmiLinkTypeT)>` - Returns a tuple containing the number of hops and the link type if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor handles, assuming the number of processors is greater than one
///     let processor_handle_src = amdsmi_get_processor_handles!()[0];
///     let processor_handle_dst = amdsmi_get_processor_handles!()[1];
///
///     // Get the link type and number of hops
///     match amdsmi_topo_get_link_type(processor_handle_src, processor_handle_dst) {
///         Ok((hops, link_type)) => println!("Hops: {}, Link Type: {:?}", hops, link_type),
///         Err(e) => panic!("Failed to get link type: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_topo_get_link_type` call fails.
pub fn amdsmi_topo_get_link_type(
    processor_handle_src: AmdsmiProcessorHandle,
    processor_handle_dst: AmdsmiProcessorHandle,
) -> AmdsmiResult<(u64, AmdsmiLinkTypeT)> {
    let mut hops: u64 = 0;
    let mut link_type: AmdsmiLinkTypeT = AmdsmiLinkTypeT::AmdsmiLinkTypeUnknown;

    call_unsafe!(amdsmi_wrapper::amdsmi_topo_get_link_type(
        processor_handle_src,
        processor_handle_dst,
        &mut hops,
        &mut link_type
    ));

    Ok((hops, link_type))
}

/// Retrieves the P2P status between two processors.
///
/// This function retrieves the P2P status between two processors, returning the link type and P2P capability.
///
/// # Arguments
///
/// * `processor_handle_src` - A handle to the source processor.
/// * `processor_handle_dst` - A handle to the destination processor.
///
/// # Returns
///
/// * `AmdsmiResult<(AmdsmiLinkTypeT, AmdsmiP2pCapabilityT)>` - Returns a tuple containing the link type and P2P capability if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor handles, assuming the number of processors is greater than one
///     let processor_handle_src = amdsmi_get_processor_handles!()[0];
///     let processor_handle_dst = amdsmi_get_processor_handles!()[1];
///
///     // Get the P2P status
///     match amdsmi_topo_get_p2p_status(processor_handle_src, processor_handle_dst) {
///         Ok((link_type, p2p_capability)) => println!("Link Type: {:?}, P2P Capability: {:?}", link_type, p2p_capability),
///         Err(e) => panic!("Failed to get P2P status: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_topo_get_p2p_status` call fails.
pub fn amdsmi_topo_get_p2p_status(
    processor_handle_src: AmdsmiProcessorHandle,
    processor_handle_dst: AmdsmiProcessorHandle,
) -> AmdsmiResult<(AmdsmiLinkTypeT, AmdsmiP2pCapabilityT)> {
    let mut link_type: AmdsmiLinkTypeT = AmdsmiLinkTypeT::AmdsmiLinkTypeUnknown;
    let mut p2p_capability = MaybeUninit::<AmdsmiP2pCapabilityT>::uninit();

    call_unsafe!(amdsmi_wrapper::amdsmi_topo_get_p2p_status(
        processor_handle_src,
        processor_handle_dst,
        &mut link_type,
        p2p_capability.as_mut_ptr()
    ));

    let p2p_capability = unsafe { p2p_capability.assume_init() };

    Ok((link_type, p2p_capability))
}

/// Retrieves the GPU compute partition for the device with the specified processor handle.
///
/// This function retrieves the GPU compute partition for the specified processor handle,
/// returning the compute partition as a string.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU compute partition is being retrieved.
///
/// # Returns
///
/// * `AmdsmiResult<String>` - Returns the compute partition as a string if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU compute partition
///     match amdsmi_get_gpu_compute_partition(processor_handle) {
///         Ok(compute_partition) => println!("Compute Partition: {}", compute_partition),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_compute_partition() not supported on this device"),
///         Err(e) => panic!("Failed to get GPU compute partition: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_compute_partition` call fails.
pub fn amdsmi_get_gpu_compute_partition(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<String> {
    let (mut compute_partition, len) = define_cstr!(amdsmi_wrapper::AMDSMI_MAX_STRING_LENGTH);
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_compute_partition(
        processor_handle,
        compute_partition.as_mut_ptr(),
        len as u32
    ));

    Ok(cstr_to_string!(compute_partition))
}

/// Set the GPU compute partition for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle` and a compute partition type `compute_partition`,
/// this function sets the GPU compute partition for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU compute partition is being set.
/// * `compute_partition` - The compute partition type [`AmdsmiComputePartitionTypeT`] to set.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example compute partition type
///     let compute_partition = AmdsmiComputePartitionTypeT::AmdsmiComputePartitionSpx;
///
///     // Set the GPU compute partition
///     match amdsmi_set_gpu_compute_partition(processor_handle, compute_partition) {
///         Ok(()) => println!("GPU compute partition set successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_set_gpu_compute_partition() not supported on this device"),
///         Err(e) => panic!("Failed to set GPU compute partition: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_gpu_compute_partition` call fails.
pub fn amdsmi_set_gpu_compute_partition(
    processor_handle: AmdsmiProcessorHandle,
    compute_partition: AmdsmiComputePartitionTypeT,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_gpu_compute_partition(
        processor_handle,
        compute_partition
    ));
    Ok(())
}

/// Retrieves the GPU memory partition for the device with the specified processor handle.
///
/// This function retrieves the GPU memory partition for the specified processor handle,
/// returning the memory partition as a string.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU memory partition is being retrieved.
///
/// # Returns
///
/// * `AmdsmiResult<String>` - Returns the memory partition as a string if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU memory partition
///     match amdsmi_get_gpu_memory_partition(processor_handle) {
///         Ok(memory_partition) => println!("Memory Partition: {}", memory_partition),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_gpu_memory_partition() not supported on this device"),
///         Err(e) => panic!("Failed to get GPU memory partition: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_memory_partition` call fails.
pub fn amdsmi_get_gpu_memory_partition(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<String> {
    let (mut memory_partition, len) = define_cstr!(amdsmi_wrapper::AMDSMI_MAX_STRING_LENGTH);
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_memory_partition(
        processor_handle,
        memory_partition.as_mut_ptr(),
        len as u32
    ));

    Ok(cstr_to_string!(memory_partition))
}

/// Set the GPU memory partition for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle` and a memory partition type `memory_partition`,
/// this function sets the GPU memory partition for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU memory partition is being set.
/// * `memory_partition` - The memory partition type [`AmdsmiMemoryPartitionTypeT`] to set.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example memory partition type
///     let memory_partition = AmdsmiMemoryPartitionTypeT::AmdsmiMemoryPartitionNps1;
///
///     // Set the GPU memory partition
///     match amdsmi_set_gpu_memory_partition(processor_handle, memory_partition) {
///         Ok(()) => println!("GPU memory partition set successfully"),
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_set_gpu_memory_partition() not supported on this device"),
///         Err(e) => panic!("Failed to set GPU memory partition: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_gpu_memory_partition` call fails.
pub fn amdsmi_set_gpu_memory_partition(
    processor_handle: AmdsmiProcessorHandle,
    memory_partition: AmdsmiMemoryPartitionTypeT,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_gpu_memory_partition(
        processor_handle,
        memory_partition
    ));
    Ok(())
}

/// Initialize GPU event notification for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function initializes GPU event notification
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU event notification is being initialized.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Initialize GPU event notification
///     match amdsmi_init_gpu_event_notification(processor_handle) {
///         Ok(_) => println!("GPU event notification initialized successfully."),
///         Err(e) => {
///             panic!("Failed to initialize GPU event notification: {}", e);
///         }
///     }
///
///     // Set GPU event notification mask
///     let event_mask: u64 = 0x0000000000000003; // AmdsmiEvtNotifVmfault | AmdsmiEvtNotifThermalThrottle
///     match amdsmi_set_gpu_event_notification_mask(processor_handle, event_mask) {
///         Ok(_) => println!("GPU event notification mask set successfully."),
///         Err(e) => {
///             panic!("Failed to set GPU event notification mask: {}", e);
///         }
///     }
///
///     // Example timeout in milliseconds
///     let timeout_ms: i32 = 10000;
///     let mut num_elem : u32 = 10;
///
///     // Get the GPU event notification data
///     match amdsmi_get_gpu_event_notification(timeout_ms, num_elem) {
///         Ok(data) => {
///             for event in data {
///                 println!("Event: {:?}", event);
///             }
///         },
///         Err(e) => assert!(e == AmdsmiStatusT::AmdsmiStatusNoData, "Failed to get GPU event notification data: {}", e),
///     }
///
///     // Stop GPU event notification
///     match amdsmi_stop_gpu_event_notification(processor_handle) {
///         Ok(_) => println!("GPU event notification stopped successfully."),
///         Err(e) => {
///             panic!("Failed to stop GPU event notification: {}", e);
///         }
///     }
///
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_socket_handles` call fails.
pub fn amdsmi_init_gpu_event_notification(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_init_gpu_event_notification(
        processor_handle
    ));
    Ok(())
}

/// Set the GPU event notification mask for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle` and a mask `mask`, this function sets the GPU event notification mask
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU event notification mask is being set.
/// * `mask` - The event notification mask to set. Bitmask generated by OR'ing 1 or more elements of
/// [`AmdsmiEvtNotificationTypeT`] indicating which event types to listen for.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// Refer to the example in the documentation for [`amdsmi_init_gpu_event_notification`].
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_set_gpu_event_notification_mask` call fails.
pub fn amdsmi_set_gpu_event_notification_mask(
    processor_handle: AmdsmiProcessorHandle,
    mask: u64,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_set_gpu_event_notification_mask(
        processor_handle,
        mask
    ));
    Ok(())
}

/// Get the GPU event notification data.
///
/// Given a timeout in milliseconds `timeout_ms`, this function retrieves the GPU event notification data.
///
/// # Arguments
///
/// * `timeout_ms` - The timeout in milliseconds to wait for the event notification.
/// * `num_elem` - The number of elements to retrieve.
///
/// # Returns
///
/// * `AmdsmiResult<Vec<AmdsmiEvtNotificationDataT>>` - Returns `Ok(Vec<AmdsmiEvtNotificationDataT>)` containing the [`Vec<AmdsmiEvtNotificationDataT>`] if successful, or an error if it fails.
///
/// # Example
///
/// Refer to the example in the documentation for [`amdsmi_init_gpu_event_notification`].
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_event_notification` call fails.
pub fn amdsmi_get_gpu_event_notification(
    timeout_ms: i32,
    mut num_elem: u32,
) -> AmdsmiResult<Vec<AmdsmiEvtNotificationDataT>> {
    let mut data: Vec<AmdsmiEvtNotificationDataT> = Vec::with_capacity(num_elem as usize);

    // Call to get the actual event notification data
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_event_notification(
        timeout_ms,
        &mut num_elem,
        data.as_mut_ptr()
    ));

    unsafe { data.set_len(num_elem as usize) };

    Ok(data)
}

/// Stop GPU event notification for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function stops GPU event notification
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU event notification is being stopped.
///
/// # Returns
///
/// * `AmdsmiResult<()>` - Returns `Ok(())` if successful, or an error if it fails.
///
/// # Example
///
/// Refer to the example in the documentation for [`amdsmi_init_gpu_event_notification`].
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_stop_gpu_event_notification` call fails.
pub fn amdsmi_stop_gpu_event_notification(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<()> {
    call_unsafe!(amdsmi_wrapper::amdsmi_stop_gpu_event_notification(
        processor_handle
    ));
    Ok(())
}

/// Get the BDF (Bus-Device-Function) information for the GPU device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the BDF information
/// for the specified GPU device.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the BDF information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiBdfT>` - Returns `Ok(AmdsmiBdfT)` containing the [`AmdsmiBdfT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the BDF information
///     match amdsmi_get_gpu_device_bdf(processor_handle) {
///         Ok(bdf) => println!("BDF information: {}", bdf),
///         Err(e) => panic!("Failed to get BDF information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_device_bdf` call fails.
pub fn amdsmi_get_gpu_device_bdf(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiBdfT> {
    let mut bdf = MaybeUninit::<AmdsmiBdfT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_device_bdf(
        processor_handle,
        bdf.as_mut_ptr()
    ));
    let bdf = unsafe { bdf.assume_init() };
    Ok(bdf)
}

/// Get the UUID of the GPU device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the UUID of the specified GPU device.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the UUID is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<String>` - Returns `Ok(String)` containing the UUID if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the UUID
///     match amdsmi_get_gpu_device_uuid(processor_handle) {
///         Ok(uuid) => println!("GPU device UUID: {}", uuid),
///         Err(e) => panic!("Failed to get GPU device UUID: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_device_uuid` call fails.
pub fn amdsmi_get_gpu_device_uuid(processor_handle: AmdsmiProcessorHandle) -> AmdsmiResult<String> {
    let (mut uuid, uuid_len) = define_cstr!(amdsmi_wrapper::AMDSMI_GPU_UUID_SIZE);
    let mut uuid_len = uuid_len as std::os::raw::c_uint;

    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_device_uuid(
        processor_handle,
        &mut uuid_len,
        uuid.as_mut_ptr()
    ));

    Ok(cstr_to_string!(uuid))
}

/// Get the GPU driver information for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the GPU driver information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU driver information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiDriverInfoT>` - Returns `Ok(AmdsmiDriverInfoT)` containing the [`AmdsmiDriverInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU driver information
///     match amdsmi_get_gpu_driver_info(processor_handle) {
///         Ok(info) => println!("GPU driver information: {:?}", info),
///         Err(e) => panic!("Failed to get GPU driver information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_driver_info` call fails.
pub fn amdsmi_get_gpu_driver_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiDriverInfoT> {
    let mut info = MaybeUninit::<AmdsmiDriverInfoT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_driver_info(
        processor_handle,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Get the GPU ASIC information for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the GPU ASIC information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU ASIC information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiAsicInfoT>` - Returns `Ok(AmdsmiAsicInfoT)` containing the [`AmdsmiAsicInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU ASIC information
///     match amdsmi_get_gpu_asic_info(processor_handle) {
///         Ok(info) => println!("GPU ASIC information: {:?}", info),
///         Err(e) => panic!("Failed to get GPU ASIC information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_asic_info` call fails.
pub fn amdsmi_get_gpu_asic_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiAsicInfoT> {
    let mut info = MaybeUninit::<AmdsmiAsicInfoT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_asic_info(
        processor_handle,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Get the GPU VRAM information for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the GPU VRAM information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU VRAM information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiVramInfoT>` - Returns `Ok(AmdsmiVramInfoT)` containing the [`AmdsmiVramInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU VRAM information
///     match amdsmi_get_gpu_vram_info(processor_handle) {
///         Ok(info) => println!("GPU VRAM information: {:?}", info),
///         Err(e) => panic!("Failed to get GPU VRAM information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_vram_info` call fails.
pub fn amdsmi_get_gpu_vram_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiVramInfoT> {
    let mut info = MaybeUninit::<AmdsmiVramInfoT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_vram_info(
        processor_handle,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Get the GPU board information for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the GPU board information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU board information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiBoardInfoT>` - Returns `Ok(AmdsmiBoardInfoT)` containing the [`AmdsmiBoardInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the GPU board information
///     match amdsmi_get_gpu_board_info(processor_handle) {
///         Ok(info) => println!("GPU board information: {:?}", info),
///         Err(e) => panic!("Failed to get GPU board information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_board_info` call fails.
pub fn amdsmi_get_gpu_board_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiBoardInfoT> {
    let mut info = MaybeUninit::<AmdsmiBoardInfoT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_board_info(
        processor_handle,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Get the power cap information for the device with the specified processor handle and sensor index.
///
/// Given a processor handle `processor_handle` and a sensor index `sensor_ind`, this function retrieves the power cap information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the power cap information is being queried.
/// * `sensor_ind` - The index of the sensor for which the power cap information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiPowerCapInfoT>` - Returns `Ok(AmdsmiPowerCapInfoT)` containing the [`AmdsmiPowerCapInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example sensor index
///     let sensor_ind: u32 = 0;
///
///     // Retrieve the power cap information
///     match amdsmi_get_power_cap_info(processor_handle, sensor_ind) {
///         Ok(info) => println!("Power cap information: {:?}", info),
///         Err(e) => panic!("Failed to get power cap information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_power_cap_info` call fails.
pub fn amdsmi_get_power_cap_info(
    processor_handle: AmdsmiProcessorHandle,
    sensor_ind: u32,
) -> AmdsmiResult<AmdsmiPowerCapInfoT> {
    let mut info = MaybeUninit::<AmdsmiPowerCapInfoT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_power_cap_info(
        processor_handle,
        sensor_ind,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Get the PCIe information for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the PCIe information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the PCIe information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiPcieInfoT>` - Returns `Ok(AmdsmiPcieInfoT)` containing the [`AmdsmiPcieInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the PCIe information
///     match amdsmi_get_pcie_info(processor_handle) {
///         Ok(info) => println!("PCIe information: {:?}", info),
///         Err(e) => panic!("Failed to get PCIe information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_pcie_info` call fails.
pub fn amdsmi_get_pcie_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiPcieInfoT> {
    let mut info = MaybeUninit::<AmdsmiPcieInfoT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_pcie_info(
        processor_handle,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Get the XGMI information for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the XGMI information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the XGMI information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiXgmiInfoT>` - Returns `Ok(AmdsmiXgmiInfoT)` containing the [`AmdsmiXgmiInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the XGMI information
///     match amdsmi_get_xgmi_info(processor_handle) {
///         Ok(info) => println!("XGMI information: {:?}", info),
///         Err(e) => panic!("Failed to get XGMI information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_xgmi_info` call fails.
pub fn amdsmi_get_xgmi_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiXgmiInfoT> {
    let mut info = MaybeUninit::<AmdsmiXgmiInfoT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_xgmi_info(
        processor_handle,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Get the firmware information for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the firmware information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the firmware information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiFwInfoT>` - Returns `Ok(AmdsmiFwInfoT)` containing the [`AmdsmiFwInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the firmware information
///     let info = amdsmi_get_fw_info(processor_handle).expect("Failed to get firmware info");
///
///     assert!(0 < info.num_fw_info &&
///         info.num_fw_info < AmdsmiFwBlockT::AmdsmiFwIdMax as u8);
///
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_fw_info` call fails.
pub fn amdsmi_get_fw_info(processor_handle: AmdsmiProcessorHandle) -> AmdsmiResult<AmdsmiFwInfoT> {
    let mut info = MaybeUninit::<AmdsmiFwInfoT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_fw_info(
        processor_handle,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Get the VBIOS information for the GPU device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the VBIOS information
/// for the specified GPU device.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the VBIOS information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiVbiosInfoT>` - Returns `Ok(AmdsmiVbiosInfoT)` containing the [`AmdsmiVbiosInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the VBIOS information
///     match amdsmi_get_gpu_vbios_info(processor_handle) {
///         Ok(info) => println!("VBIOS information: {:?}", info),
///         Err(e) => panic!("Failed to get VBIOS information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_vbios_info` call fails.
pub fn amdsmi_get_gpu_vbios_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiVbiosInfoT> {
    let mut info = MaybeUninit::<AmdsmiVbiosInfoT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_vbios_info(
        processor_handle,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Get the GPU activity information for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the GPU activity information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU activity information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiEngineUsageT>` - Returns `Ok(AmdsmiEngineUsageT)` containing the [`AmdsmiEngineUsageT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!().first().expect("No processor handles found").clone();
///
///     // Retrieve the GPU activity information
///     match amdsmi_get_gpu_activity(processor_handle) {
///         Ok(info) => println!("GPU activity information: {:?}", info),
///         Err(e) => panic!("Failed to get GPU activity information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_activity` call fails.
pub fn amdsmi_get_gpu_activity(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiEngineUsageT> {
    let mut info = MaybeUninit::<AmdsmiEngineUsageT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_activity(
        processor_handle,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Get the power information for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the power information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the power information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiPowerInfoT>` - Returns `Ok(AmdsmiPowerInfoT)` containing the [`AmdsmiPowerInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the power information
///     match amdsmi_get_power_info(processor_handle) {
///         Ok(info) => println!("Power information: {:?}", info),
///         Err(e) => panic!("Failed to get power information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_power_info` call fails.
pub fn amdsmi_get_power_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiPowerInfoT> {
    let mut info = MaybeUninit::<AmdsmiPowerInfoT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_power_info(
        processor_handle,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Check if GPU power management is enabled for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function checks if GPU power management is enabled
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the power management status is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<bool>` - Returns `Ok(bool)` indicating whether power management is enabled if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Check if GPU power management is enabled
///     match amdsmi_is_gpu_power_management_enabled(processor_handle) {
///         Ok(enabled) => println!("GPU power management enabled: {}", enabled),
///         Err(e) => panic!("Failed to check GPU power management status: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_is_gpu_power_management_enabled` call fails.
pub fn amdsmi_is_gpu_power_management_enabled(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<bool> {
    let mut enabled: bool = false;
    call_unsafe!(amdsmi_wrapper::amdsmi_is_gpu_power_management_enabled(
        processor_handle,
        &mut enabled as *mut bool
    ));
    Ok(enabled)
}

/// Get the clock information for the device with the specified processor handle and clock type.
///
/// Given a processor handle `processor_handle` and a clock type `clk_type`, this function retrieves the clock information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the clock information is being queried.
/// * `clk_type` - The type of clock for which the information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiClkInfoT>` - Returns `Ok(AmdsmiClkInfoT)` containing the [`AmdsmiClkInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Example clock type
///     let clk_type = AmdsmiClkTypeT::AmdsmiClkTypeSys;
///
///     // Retrieve the clock information
///     match amdsmi_get_clock_info(processor_handle, clk_type) {
///         Ok(info) => println!("Clock information: {:?}", info),
///         Err(e) => panic!("Failed to get clock information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_clock_info` call fails.
pub fn amdsmi_get_clock_info(
    processor_handle: AmdsmiProcessorHandle,
    clk_type: AmdsmiClkTypeT,
) -> AmdsmiResult<AmdsmiClkInfoT> {
    let mut info = MaybeUninit::<AmdsmiClkInfoT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_clock_info(
        processor_handle,
        clk_type,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Get the VRAM usage information for the device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the VRAM usage information
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the VRAM usage information is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiVramUsageT>` - Returns `Ok(AmdsmiVramUsageT)` containing the [`AmdsmiVramUsageT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the VRAM usage information
///     match amdsmi_get_gpu_vram_usage(processor_handle) {
///         Ok(info) => println!("VRAM usage information: {:?}", info),
///         Err(e) => panic!("Failed to get VRAM usage information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_vram_usage` call fails.
pub fn amdsmi_get_gpu_vram_usage(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiVramUsageT> {
    let mut info = MaybeUninit::<AmdsmiVramUsageT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_vram_usage(
        processor_handle,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Get the total ECC error count for the GPU device with the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the total ECC error count
/// for the specified GPU device.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the ECC error count is being queried.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiErrorCountT>` - Returns `Ok(AmdsmiErrorCountT)` containing the [`AmdsmiErrorCountT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the total ECC error count
///     match amdsmi_get_gpu_total_ecc_count(processor_handle) {
///         Ok(error_count) => println!("Total ECC error count: {:?}", error_count),
///         Err(e) => panic!("Failed to get total ECC error count: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_total_ecc_count` call fails.
pub fn amdsmi_get_gpu_total_ecc_count(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiErrorCountT> {
    let mut ec = MaybeUninit::<AmdsmiErrorCountT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_total_ecc_count(
        processor_handle,
        ec.as_mut_ptr()
    ));
    let ec = unsafe { ec.assume_init() };
    Ok(ec)
}

/// Retrieves the list of GPU processes for the specified processor handle.
///
/// Given a processor handle `processor_handle`, this function retrieves the list of GPU processes
/// for the specified processor.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU process list is being retrieved.
///
/// # Returns
///
/// * `AmdsmiResult<Vec<AmdsmiProcInfoT>>` - Returns a vector containing the [`AmdsmiProcInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     // Retrieve the list of GPU processes
///     match amdsmi_get_gpu_process_list(processor_handle) {
///         Ok(process_list) => {
///             for process in process_list {
///                 println!("Process Info: {:?}", process);
///             }
///         },
///         Err(e) => panic!("Failed to get GPU process list: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_process_list` call fails.
pub fn amdsmi_get_gpu_process_list(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<Vec<AmdsmiProcInfoT>> {
    let mut num_processes: u32 = 0;
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_process_list(
        processor_handle,
        &mut num_processes,
        std::ptr::null_mut()
    ));

    let mut processes: Vec<AmdsmiProcInfoT> = Vec::with_capacity(num_processes as usize);
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_process_list(
        processor_handle,
        &mut num_processes,
        processes.as_mut_ptr()
    ));
    unsafe { processes.set_len(num_processes as usize) };

    Ok(processes)
}

/// Get the library version information.
///
/// This function retrieves the version information of the AMD SMI library.
///
/// # Arguments
///
/// * `version` - A mutable pointer to an `AmdsmiVersionT` structure where the version information will be stored.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiVersionT>` - Returns `Ok(AmdsmiVersionT)` containing the [`AmdsmiVersionT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Retrieve the library version information
///     match amdsmi_get_lib_version() {
///         Ok(version) => println!("Library version: {:?}", version),
///         Err(e) => panic!("Failed to get library version: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_lib_version` call fails.
pub fn amdsmi_get_lib_version() -> AmdsmiResult<AmdsmiVersionT> {
    let mut version = MaybeUninit::<AmdsmiVersionT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_lib_version(version.as_mut_ptr()));
    let version = unsafe { version.assume_init() };
    Ok(version)
}

/// Retrieve the GPU accelerator partition profile.
///
/// This function retrieves the GPU accelerator partition profile for a given processor handle.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU accelerator partition profile is being retrieved.
///
/// # Returns
///
/// * `AmdsmiResult<(AmdsmiAcceleratorPartitionProfileT, u32)>` - Returns `Ok((AmdsmiAcceleratorPartitionProfileT, u32))` containing the [`AmdsmiAcceleratorPartitionProfileT`] and partition ID if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     match amdsmi_get_gpu_accelerator_partition_profile(processor_handle) {
///         Ok((profile, partition_id)) => {
///             println!("Profile: {:?}", profile.profile_type);
///             println!("Partition ID: {}", partition_id);
///         },
///         Err(e) => println!("Failed to retrieve GPU accelerator partition profile: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_accelerator_partition_profile` call fails.
pub fn amdsmi_get_gpu_accelerator_partition_profile(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<(AmdsmiAcceleratorPartitionProfileT, u32)> {
    let mut profile = MaybeUninit::<AmdsmiAcceleratorPartitionProfileT>::uninit();
    let mut partition_id = MaybeUninit::<u32>::uninit();
    call_unsafe!(
        amdsmi_wrapper::amdsmi_get_gpu_accelerator_partition_profile(
            processor_handle,
            profile.as_mut_ptr(),
            partition_id.as_mut_ptr()
        )
    );
    let profile = unsafe { profile.assume_init() };
    let partition_id = unsafe { partition_id.assume_init() };
    Ok((profile, partition_id))
}

/// Retrieve the GPU KFD information.
///
/// This function retrieves the GPU KFD information for a given processor handle.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the GPU KFD information is being retrieved.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiKfdInfoT>` - Returns `Ok(AmdsmiKfdInfoT)` containing the [`AmdsmiKfdInfoT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     match amdsmi_get_gpu_kfd_info(processor_handle) {
///         Ok(info) => {
///             println!("KFD Info: {:?}", info);
///         },
///         Err(e) => println!("Failed to retrieve GPU KFD information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_gpu_kfd_info` call fails.
pub fn amdsmi_get_gpu_kfd_info(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiKfdInfoT> {
    let mut info = MaybeUninit::<AmdsmiKfdInfoT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_gpu_kfd_info(
        processor_handle,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Retrieve the GPU violation status.
///
/// This function retrieves the violation status for a given processor handle.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the violation status is being retrieved.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiViolationStatusT>` - Returns `Ok(AmdsmiViolationStatusT)` containing the [`AmdsmiViolationStatusT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///
///     match amdsmi_get_violation_status(processor_handle) {
///         Ok(info) => {
///             println!("Violation Status: {:?}", info);
///         },
///         Err(AmdsmiStatusT::AmdsmiStatusNotSupported) => println!("amdsmi_get_violation_status() not supported on this device"),
///         Err(e) => println!("Failed to retrieve violation status: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_violation_status` call fails.
pub fn amdsmi_get_violation_status(
    processor_handle: AmdsmiProcessorHandle,
) -> AmdsmiResult<AmdsmiViolationStatusT> {
    let mut info = MaybeUninit::<AmdsmiViolationStatusT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_violation_status(
        processor_handle,
        info.as_mut_ptr()
    ));
    let info = unsafe { info.assume_init() };
    Ok(info)
}

/// Retrieve the nearest link topology information.
///
/// This function retrieves the nearest link topology information for a given processor handle and link type.
///
/// # Arguments
///
/// * `processor_handle` - A handle to the processor for which the link topology information is being retrieved.
/// * `link_type` - The type of [`AmdsmiLinkTypeT`] for which the topology information is being retrieved.
///
/// # Returns
///
/// * `AmdsmiResult<AmdsmiTopologyNearestT>` - Returns `Ok(AmdsmiTopologyNearestT)` containing the [`AmdsmiTopologyNearestT`] if successful, or an error if it fails.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// #
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     // Example processor_handle, assuming the number of processors is greater than zero
///     let processor_handle = amdsmi_get_processor_handles!()[0];
///     let link_type = AmdsmiLinkTypeT::AmdsmiLinkTypePcie;
///
///     match amdsmi_get_link_topology_nearest(processor_handle, link_type) {
///         Ok(info) => {
///             println!("Topology Nearest Info: {:?}", info);
///         },
///         Err(e) => println!("Failed to retrieve link topology nearest information: {}", e),
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
///
/// # Errors
///
/// This function will return the error in [`AmdsmiStatusT`] if the underlying `amdsmi_wrapper::amdsmi_get_link_topology_nearest` call fails.
pub fn amdsmi_get_link_topology_nearest(
    processor_handle: AmdsmiProcessorHandle,
    link_type: AmdsmiLinkTypeT,
) -> AmdsmiResult<AmdsmiTopologyNearestT> {
    let mut topology_nearest_info = MaybeUninit::<AmdsmiTopologyNearestT>::uninit();
    call_unsafe!(amdsmi_wrapper::amdsmi_get_link_topology_nearest(
        processor_handle,
        link_type,
        topology_nearest_info.as_mut_ptr()
    ));
    let topology_nearest_info = unsafe { topology_nearest_info.assume_init() };
    Ok(topology_nearest_info)
}

/// A macro to get all the GPU processor handles directly.
///
/// This macro retrieves all the GPU processor handles by first getting the socket handles
/// and then getting the processor handles for each socket.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
///
/// # fn main() {
/// #   // Initialize the AMD SMI library
/// #   amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).expect("Failed to initialize AMD SMI");
/// #
///     let processor_handles = amdsmi_get_processor_handles!();
///     for handle in processor_handles {
///         // Perform operations with each processor handle
///         // ...
///     }
/// #
/// #   // Shut down the AMD SMI library
/// #   amdsmi_shut_down().expect("Failed to shut down AMD SMI");
/// # }
/// ```
#[macro_export]
macro_rules! amdsmi_get_processor_handles {
    () => {{
        // Get the socket handles
        let socket_handles = amdsmi_get_socket_handles().expect("Failed to get socket handles");

        let mut processor_handles = Vec::new();

        // Iterate over each socket handle to get the processor handles
        for socket_handle in socket_handles {
            // Call the amdsmi_get_processor_handles function directly
            let mut processors = amdsmi_get_processor_handles(socket_handle)
                .expect("Failed to get processor handles for socket");

            // Append the processor handles to the main vector
            processor_handles.append(&mut processors);
        }

        processor_handles
    }};
}
