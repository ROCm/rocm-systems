# AMD SMI Rust Binding

This rust crate provides Rust bindings for the AMD System Management Interface (AMD-SMI) library. It allows you to interact with AMD GPUs and retrieve various information using Rust.

## Table of Contents

- [Overview](#overview)
- [Hello World Example](#hello-world-example)
- [Directory Structure](#directory-structure)
- [Building](#building)
- [Regenerating FFI Bindings](#regenerating-ffi-bindings)
- [Running Tests](#running-tests)
- [Running Examples](#running-examples)
- [Generating Documentation](#generating-documentation)
- [Adding New API and Tests](#adding-new-api-and-tests)

## Overview

The AMD SMI Rust binding crate automates the generation of bindings and ensures safety, maintainability, and ease of use. The implementation consists of two main steps:

1. **Generating Bindings with `bindgen`**:
   - The `build.rs` script uses `bindgen` to automatically generate Rust FFI (Foreign Function Interface) bindings for the AMD SMI C library. This step exports all enums, structs, unions, and unsafe functions from the C library into Rust, providing a comprehensive low-level interface to the AMD SMI library.

2. **Implementing Safe Rust Wrappers**:
   - The generated bindings are then wrapped in safe Rust functions. These safe wrappers handle error checking, resource management, and provide a more idiomatic Rust interface. This ensures that users of the library can interact with the AMD SMI functions without dealing with unsafe code directly.

## Hello World Example
Here is a simple "Hello World" example to get you started with the AMD SMI Rust bindings. This example initializes the AMD SMI library, retrieves the GPU information, and prints it to the console.
```
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

            // Get GPU BDF using the processor handle
            match amdsmi_get_gpu_device_bdf(processor_handle) {
                Ok(gpu_bdf) => println!("GPU BDF: {}", gpu_bdf),
                Err(e) => eprintln!("Failed to get GPU BDF: {}", e),
            }

            println!();
        }
    }

    // Shutdown the AMD SMI library
    if let Err(e) = amdsmi_shut_down() {
        eprintln!("Failed to shutdown AMD SMI: {}", e);
    }
}
```

## Directory Structure

```
amd-smi/rust-interface
├── Cargo.toml               # Cargo configuration file
├── CMakeLists.txt           # CMake configuration file for building the AMD SMI library
├── build.rs                 # Build script for generating bindings
├── src/
│   ├── lib.rs               # Library entry point
│   ├── amdsmi.rs            # Main module for AMD SMI bindings
│   ├── utils.rs             # Utility functions and helpers
│   └── amdsmi_wrapper.rs    # Automatically generate FFI bindings by `bindgen`
└── examples                 # Example programs demonstrating usage
```

## Building

To build this Rust binding, you need to have Rust and Cargo installed.

### Option 1: Using CMake

Navigate to the project's root directory and use the project's CMake build system. You need to define the `BUILD_RUST_WRAPPER` option. This will build this Rust interface and integrate it with the rest of the project.

```sh
mkdir build
cd build
cmake -DBUILD_RUST_WRAPPER=ON ..
make
sudo make install
```

### Option 2: Using Cargo

Alternatively, you can navigate to the `rust-interface` folder and build the project using Cargo.

```sh
cd rust-interface
cargo build
```


### Regenerating FFI Bindings

The `amdsmi_wrapper.rs` file contains automatically generated FFI (Foreign Function Interface) bindings from the AMD SMI C library. **Do not edit this file manually** as any changes will be overwritten during regeneration. Bindings are not automatically regenerated during normal builds unless the `AMDSMI_GENERATE_RUST_WRAPPER` environment variable is explicitly set.

To force regeneration of the bindings, you can:

```sh
# Using CMake
cd build
cmake -DBUILD_RUST_WRAPPER=ON -DREGENERATE_RUST_WRAPPER=ON ..
make

# Or using Cargo with environment variable (recommended)
cd rust-interface
AMDSMI_GENERATE_RUST_WRAPPER=1 cargo build
```

## Running Tests

To run the tests, use the following command:

```sh
cargo test --doc -- --show-output --test-threads=1
```

This command will execute all the unit tests that are defined in the API's example code within the API's documentation comments.

## Running Examples

To run the examples, use the following command:

```sh
cargo run --example example_name
```

Replace `example_name` with the name of the example you want to run. For example, if you have an example named `amdsmi_get_gpu_info`, you can run it as follows:

```sh
cargo run --example amdsmi_get_gpu_info
```

## Generating Documentation

To generate the documentation for the project, use the following command:

```sh
cargo doc --open
```

This will generate the documentation and open it in your default web browser.

## Adding New API and Tests

### Adding New API

1. **Update Bindings**:
   - To regenerate the bindings using `bindgen` according to the latest `amdsmi.h`, set the `AMDSMI_GENERATE_RUST_WRAPPER` environment variable and rebuild (see [Regenerating FFI Bindings](#regenerating-ffi-bindings)).

2. **Define the API**:
   - Add the new API function in the appropriate Rust file, e.g., `src/amdsmi.rs`.
   - Implement the function to use the `call_unsafe!` macro to call the `bindgen` generated unsafe rust FFI function in `amdsmi_wrapper.rs`.

3. **Document the API**:
   - Use Rust doc comments `///` to document the new API function. Include a description, parameters, return values, and examples.

### Adding Tests

Include examples in the documentation comments of your functions. These examples will be compiled and run as tests when you run `cargo test`.


### Example of New API

Here is an example of adding a new API function and documenting it with a doc test:

```rust
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
```

