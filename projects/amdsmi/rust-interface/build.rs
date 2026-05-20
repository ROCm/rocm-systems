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
use std::process::Command;
mod callbacks;

// Attempts to locate `libamd_smi.so` by invoking `pkg-config --libs-only-L amd_smi`
// and returning the first directory advertised by pkg-config that actually
// contains the shared library. Returns `None` if pkg-config is missing, has no
// `amd_smi.pc`, or its reported directory does not contain the library.
fn try_pkg_config() -> Option<String> {
    let output = Command::new("pkg-config")
        .args(["--libs-only-L", "amd_smi"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for part in stdout.split_whitespace() {
        if let Some(path) = part.strip_prefix("-L") {
            let lib_path = PathBuf::from(path);
            if lib_path.join("libamd_smi.so").exists() {
                return Some(path.to_string());
            }
        }
    }
    None
}

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

    // 1. Explicit override via environment variable.
    if let Ok(lib_dir) = env::var("AMDSMI_LIB_DIR") {
        if PathBuf::from(&lib_dir).join(amdsmi_file).exists() {
            return Ok(lib_dir);
        }
    }

    // 2. ./lib/ subdirectory of the build, used by the CMake-driven source
    //    tarball flow that ships an in-tree libamd_smi.so.
    if let Ok(current_dir) = env::current_dir() {
        let current_lib_dir = current_dir.join("lib");
        if current_lib_dir.join(amdsmi_file).exists() {
            return Ok(current_lib_dir.to_string_lossy().into_owned());
        }
    }

    // 3. pkg-config (for distros that install amd_smi.pc).
    if let Some(lib_dir) = try_pkg_config() {
        return Ok(lib_dir);
    }

    // 4. The most recently installed /opt/rocm* directory.
    if let Some(rocm_dir) = get_rocm_dir() {
        let lib_path = rocm_dir.join("lib");
        if lib_path.join(amdsmi_file).exists() {
            return Ok(lib_path.to_string_lossy().into_owned());
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "Could not find libamd_smi.so. Please ensure AMD-SMI is installed.\n\
         \n\
         Search locations tried:\n\
         1. $AMDSMI_LIB_DIR (unset or did not contain libamd_smi.so)\n\
         2. ./lib/ (not found)\n\
         3. pkg-config amd_smi (not configured or missing)\n\
         4. /opt/rocm*/lib/ (not found)\n\
         \n\
         To fix this, either:\n\
         - Install ROCm: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/\n\
         - Set AMDSMI_LIB_DIR to a directory containing libamd_smi.so\n\
         - Build AMD-SMI from source and install it",
    ))
}

fn get_amdsmi_header_file() -> Result<String> {
    let amdsmi_header_path = "include/amd_smi/amdsmi.h";

    // 1. Explicit override via environment variable.
    if let Ok(include_dir) = env::var("AMDSMI_INCLUDE_DIR") {
        let header_path = PathBuf::from(&include_dir).join("amd_smi/amdsmi.h");
        if header_path.exists() {
            return Ok(header_path.to_string_lossy().into_owned());
        }
    }

    // 2. Header from the parent project tree (development builds).
    let default_path = PathBuf::from("../").join(amdsmi_header_path);
    if default_path.exists() {
        return Ok(default_path.to_string_lossy().into_owned());
    }

    // 3. Header next to the build (source-tarball flow).
    if let Ok(current_dir) = env::current_dir() {
        let fallback_path = current_dir.join(amdsmi_header_path);
        if fallback_path.exists() {
            return Ok(fallback_path.to_string_lossy().into_owned());
        }
    }

    // 4. Header from the most recently installed /opt/rocm*.
    if let Some(rocm_dir) = get_rocm_dir() {
        let fallback_path = rocm_dir.join(amdsmi_header_path);
        if fallback_path.exists() {
            return Ok(fallback_path.to_string_lossy().into_owned());
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "The amdsmi.h header file was not found. This is required for regenerating FFI bindings.\n\
         Set AMDSMI_INCLUDE_DIR to a directory containing amd_smi/amdsmi.h",
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
        .raw_line("#![allow(non_camel_case_types)]")
        // Suppress clippy lints over which we have no control because the
        // bindings are mechanically generated by bindgen.
        .raw_line("#![allow(clippy::useless_transmute)]")
        .raw_line("#![allow(clippy::unnecessary_cast)]")
        .raw_line("#![allow(clippy::enum_variant_names)]")
        .raw_line("#![allow(clippy::empty_docs)]")
        .generate()
        .expect("Unable to generate binding wrapper for amdsmi C interface!");

    // CARGO_MANIFEST_DIR is always set during build script execution and
    // points at the directory containing Cargo.toml, regardless of where
    // cargo was invoked from.
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let bindings_path = PathBuf::from(&manifest_dir).join("src/amdsmi_wrapper.rs");
    bindings
        .write_to_file(&bindings_path)
        .expect("Couldn't write binding wrapper for amdsmi C interface!");
    println!("cargo:warning=Wrapper generated at: {:?}", bindings_path);
}

fn main() {
    // Re-run the build script whenever any of the inputs that affect library
    // or header discovery change.
    println!("cargo:rerun-if-env-changed=AMDSMI_LIB_DIR");
    println!("cargo:rerun-if-env-changed=AMDSMI_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=AMDSMI_GENERATE_RUST_WRAPPER");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    // Always emit the link-lib directive so downstream binaries that depend
    // on `amdsmi` know to link against libamd_smi at final link time.
    println!("cargo:rustc-link-lib=amd_smi");

    // docs.rs builds documentation in a sandbox where libamd_smi.so is not
    // present. Detect that environment by the `DOCS_RS` env var that docs.rs
    // sets, and skip the library search so the doc build does not fail.
    // `cargo doc`/`rustdoc` does not actually link the final crate, so omitting
    // the search path here is harmless.
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    let amdsmi_lib_dir = get_amdsmi_lib_dir().expect("Failed to find AMD-SMI library");
    println!("cargo:rustc-link-search=native={}", amdsmi_lib_dir);

    // Bindgen regeneration is opt-in. Building from a published crate source
    // tree never regenerates because we ship a pre-generated wrapper.
    if env::var("AMDSMI_GENERATE_RUST_WRAPPER").is_ok() {
        let amdsmi_header_file =
            get_amdsmi_header_file().expect("Failed to get the amd_smi header file");
        println!("cargo:rerun-if-changed={}", amdsmi_header_file);
        generate_amdsmi_wrapper(&amdsmi_header_file);
    }
}
