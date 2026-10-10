// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
        .raw_line("// Copyright Advanced Micro Devices, Inc.")
        .raw_line("// SPDX-License-Identifier: MIT")
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

#[cfg(feature = "dynamic-loading")]
fn generate_runtime_bindings() {
    use quote::quote;
    use syn::{FnArg, ForeignItem, Item};

    // Reuse bindgen's checked-in signatures: ordinary builds need neither
    // libclang nor installed AMD SMI headers, and there is only one ABI source.
    let source = fs::read_to_string("src/amdsmi_wrapper.rs").expect("Cannot read AMD SMI bindings");
    let bindings = syn::parse_file(&source).expect("Cannot parse AMD SMI bindings");
    let mut wrappers = quote! {};
    for item in bindings.items {
        let Item::ForeignMod(block) = item else {
            continue;
        };
        let abi = block.abi;
        for item in block.items {
            let ForeignItem::Fn(function) = item else {
                continue;
            };
            let name = function.sig.ident;
            let inputs = function.sig.inputs;
            let output = function.sig.output;
            let (args, types): (Vec<_>, Vec<_>) = inputs
                .iter()
                .map(|input| {
                    let FnArg::Typed(arg) = input else {
                        unreachable!("C functions have no self argument");
                    };
                    (&arg.pat, &arg.ty)
                })
                .unzip();
            wrappers.extend(quote! {
                #[allow(non_snake_case)]
                pub unsafe fn #name(#inputs) #output {
                    type Function = unsafe #abi fn(#(#types),*) #output;
                    static FUNCTION: OnceLock<Result<Function, AmdsmiStatusT>> = OnceLock::new();
                    let function = FUNCTION.get_or_init(|| {
                        let library = library()?;
                        // SAFETY: the signature is taken from bindgen's declaration,
                        // and the process-wide library outlives every copied symbol.
                        unsafe {
                            library.get::<Function>(concat!(stringify!(#name), "\0").as_bytes())
                                .map(|symbol| *symbol)
                                .map_err(|_| AmdsmiStatusT::AmdsmiStatusFailLoadSymbol)
                        }
                    });
                    match function {
                        Ok(function) => unsafe { function(#(#args),*) },
                        Err(status) => *status,
                    }
                }
            });
        }
    }
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    fs::write(output.join("runtime_bindings.rs"), wrappers.to_string())
        .expect("Cannot write runtime AMD SMI bindings");
}

fn main() {
    println!("cargo:rerun-if-env-changed=AMDSMI_LIB_DIR");
    println!("cargo:rerun-if-env-changed=AMDSMI_GENERATE_RUST_WRAPPER");
    println!("cargo:rerun-if-changed=src/amdsmi_wrapper.rs");
    println!("cargo:rerun-if-changed=callbacks.rs");

    if !cfg!(feature = "dynamic-loading") {
        let amdsmi_lib_dir = get_amdsmi_lib_dir().expect("Failed to get the amd_smi library path");
        println!("cargo:rustc-link-lib=amd_smi");
        println!("cargo:rustc-link-search=native={}", amdsmi_lib_dir);
    }

    let generate_wrapper = env::var("AMDSMI_GENERATE_RUST_WRAPPER").is_ok();
    if generate_wrapper {
        // Get the amdsmi.h header file path
        let amdsmi_header_file =
            get_amdsmi_header_file().expect("Failed to get the amd_smi header file");
        println!("cargo:rerun-if-changed={}", amdsmi_header_file);

        // Generate the amdsmi wrapper
        generate_amdsmi_wrapper(&amdsmi_header_file);
    }

    #[cfg(feature = "dynamic-loading")]
    generate_runtime_bindings();
}
