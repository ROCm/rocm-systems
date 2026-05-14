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

extern crate bindgen;
use std::env;
use std::fs;
use std::io::Result;
use std::path::{Path, PathBuf};
mod callbacks;

fn get_rocm_dir() -> Option<PathBuf> {
    // Look for the latest folder in /opt that begins with "rocm"
    let opt_path = Path::new("/opt");
    if let Ok(entries) = fs::read_dir(opt_path) {
        let mut rocm_dirs: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_dir()
                    && path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .starts_with("rocm")
            })
            .collect();

        // Sort the directories by name and get the latest one
        rocm_dirs.sort();
        if let Some(latest_rocm_dir) = rocm_dirs.last() {
            return Some(latest_rocm_dir.clone());
        }
    }
    None
}

fn get_amdsmi_lib_dir() -> Result<String> {
    let amdsmi_file = "libamd_smi.so";

    // Check the environment variable AMDSMI_LIB_DIR to get the library directory
    if let Ok(lib_dir) = env::var("AMDSMI_LIB_DIR") {
        if PathBuf::from(lib_dir.clone()).join(amdsmi_file).exists() {
            return Ok(lib_dir);
        }
    }

    // Check the current directory
    if let Ok(current_dir) = env::current_dir() {
        let current_lib_dir = current_dir.join("lib");
        if current_lib_dir.join(amdsmi_file).exists() {
            return Ok(current_lib_dir.to_string_lossy().into_owned());
        }
    }

    // Check the ROCm directory
    if let Some(lib_dir) = get_rocm_dir() {
        if lib_dir.join("lib").join(amdsmi_file).exists() {
            return Ok(lib_dir.join("lib").to_string_lossy().into_owned());
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "The libamd_smi.so library was not found. Please set the AMDSMI_LIB_DIR environment variable to the directory containing the library.",
    ))
}

fn get_amdsmi_header_file() -> Result<String> {
    let amdsmi_header_path = "include/amd_smi/amdsmi.h";
    // Use the project's header as the first priority
    let default_path = PathBuf::from("../").join(amdsmi_header_path);
    if default_path.exists() {
        return Ok(default_path.to_string_lossy().into_owned());
    }

    // Check the current directory
    if let Ok(current_dir) = env::current_dir() {
        let fallback_path = current_dir.join(amdsmi_header_path);
        if fallback_path.exists() {
            return Ok(fallback_path.to_string_lossy().into_owned());
        }
    }

    // Check the ROCm directory
    if let Some(rocm_dir) = get_rocm_dir() {
        let fallback_path = rocm_dir.join(amdsmi_header_path);
        if fallback_path.exists() {
            return Ok(fallback_path.to_string_lossy().into_owned());
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "The amdsmi.h header file was not found",
    ))
}

fn generate_amdsmi_wrapper(amdsmi_header_file: &str) {
    let bindings = bindgen::Builder::default()
        .header(amdsmi_header_file)
        .generate_comments(false)
        .prepend_enum_name(false)
        .rustified_enum("^(.*)$")
        .allowlist_type("^(amdsmi.*)$")
        .allowlist_function("^(amdsmi.*)$")
        .allowlist_var("^(AMDSMI.*)$")
        .parse_callbacks(Box::new(callbacks::UpperCamelCaseCallbacks))
        .raw_line("// Copyright (C) 2024 Advanced Micro Devices. All rights reserved.")
        .raw_line("//")
        .raw_line(
            "// Permission is hereby granted, free of charge, to any person obtaining a copy of",
        )
        .raw_line(
            "// this software and associated documentation files (the \"Software\"), to deal in",
        )
        .raw_line("// the Software without restriction, including without limitation the rights to")
        .raw_line(
            "// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of",
        )
        .raw_line(
            "// the Software, and to permit persons to whom the Software is furnished to do so,",
        )
        .raw_line("// subject to the following conditions:")
        .raw_line("//")
        .raw_line(
            "// The above copyright notice and this permission notice shall be included in all",
        )
        .raw_line("// copies or substantial portions of the Software.")
        .raw_line("//")
        .raw_line("// THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR")
        .raw_line(
            "// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS",
        )
        .raw_line(
            "// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR",
        )
        .raw_line(
            "// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER",
        )
        .raw_line("// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN")
        .raw_line("// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.")
        .raw_line("")
        .raw_line("#![allow(non_upper_case_globals)]")
        .generate()
        .expect("Unable to binding wrapper for amdsmi C interface!");

    let bindings_path = PathBuf::from("./src/amdsmi_wrapper.rs");
    bindings
        .write_to_file(&bindings_path)
        .expect("Couldn't write binding wrapper for amdsmi C interface!");
    println!("Wrapper generated at: {:?}", bindings_path);

}

fn main() {
    // Get the amd_smi library directory
    let amdsmi_lib_dir = get_amdsmi_lib_dir().expect("Failed to get the amd_smi library path");

    // Tell cargo to tell rustc to link the AMD-SMI library
    println!("cargo:rustc-link-lib=amd_smi");
    println!("cargo:rustc-link-search=native={}", amdsmi_lib_dir);

    let generate_wrapper = env::var("AMDSMI_GENERATE_RUST_WRAPPER").is_ok();
    if generate_wrapper {
        // Get the amdsmi.h header file path
        let amdsmi_header_file = get_amdsmi_header_file().expect("Failed to get the amd_smi header file");

        // Generate the amdsmi wrapper
        generate_amdsmi_wrapper(&amdsmi_header_file);
    }
}
