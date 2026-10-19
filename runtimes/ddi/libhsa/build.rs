// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Build the HSA tool dispatch layout from the public ABI and exported symbols.

use std::collections::{HashMap, HashSet};
use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;

const TABLES: [(&str, &str); 6] = [
    ("core", "CoreApiTable"),
    ("amd_ext", "AmdExtTable"),
    ("finalizer_ext", "FinalizerExtTable"),
    ("image_ext", "ImageExtTable"),
    ("tools", "ToolsApiTable"),
    ("pc_sampling_ext", "PcSamplingExtTable"),
];

const PUBLIC_EXTENSION_SYMBOLS: [&str; 28] = [
    "hsa_ext_image_get_capability",
    "hsa_ext_image_data_get_info",
    "hsa_ext_image_create",
    "hsa_ext_image_destroy",
    "hsa_ext_image_copy",
    "hsa_ext_image_import",
    "hsa_ext_image_export",
    "hsa_ext_image_clear",
    "hsa_ext_sampler_create",
    "hsa_ext_sampler_destroy",
    "hsa_ext_image_get_capability_with_layout",
    "hsa_ext_image_data_get_info_with_layout",
    "hsa_ext_image_create_with_layout",
    "hsa_ext_sampler_create_v2",
    "hsa_ven_amd_loader_query_host_address",
    "hsa_ven_amd_loader_query_segment_descriptors",
    "hsa_ven_amd_loader_query_executable",
    "hsa_ven_amd_loader_executable_iterate_loaded_code_objects",
    "hsa_ven_amd_loader_loaded_code_object_get_info",
    "hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size",
    "hsa_ven_amd_loader_iterate_executables",
    "hsa_ven_amd_pcs_iterate_configuration",
    "hsa_ven_amd_pcs_create",
    "hsa_ven_amd_pcs_create_from_id",
    "hsa_ven_amd_pcs_destroy",
    "hsa_ven_amd_pcs_start",
    "hsa_ven_amd_pcs_stop",
    "hsa_ven_amd_pcs_flush",
];

const PRIVATE_TABLE_FUNCTIONS: [&str; 3] = [
    "hsa_amd_queue_intercept_create",
    "hsa_amd_queue_intercept_register",
    "hsa_amd_runtime_queue_create_register",
];

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn exported_symbols(source: &str) -> Result<Vec<String>, io::Error> {
    let exports: Vec<String> = source
        .lines()
        .filter_map(|line| line.strip_prefix(".symver "))
        .filter_map(|line| line.split_once(',').map(|(name, _)| name))
        .filter(|name| name.starts_with("hsa_"))
        .map(str::to_owned)
        .collect();
    let unique: HashSet<&str> = exports.iter().map(String::as_str).collect();
    if exports.len() != unique.len() {
        return Err(invalid("duplicate Linux HSA export"));
    }
    if PUBLIC_EXTENSION_SYMBOLS
        .iter()
        .any(|name| !unique.contains(name))
    {
        return Err(invalid("extension entry point missing from Linux exports"));
    }
    Ok(exports)
}

fn table_fields(header: &str, table: &str) -> Result<Vec<String>, io::Error> {
    let start = format!("struct {table} {{");
    let (_, rest) = header
        .split_once(&start)
        .ok_or_else(|| invalid(format!("missing {table}")))?;
    let (body, _) = rest
        .split_once("};")
        .ok_or_else(|| invalid(format!("unterminated {table}")))?;
    let fields: Vec<String> = body
        .split(';')
        .filter_map(|declaration| declaration.split_whitespace().last())
        .map(|name| name.trim_start_matches('*'))
        .filter_map(|name| name.strip_suffix("_fn"))
        .filter(|name| name.starts_with("hsa_"))
        .map(str::to_owned)
        .collect();
    if fields.is_empty() {
        return Err(invalid(format!("empty {table}")));
    }
    Ok(fields)
}

#[allow(clippy::too_many_lines)]
fn generate_rust(exports: &[String], tables: &[(&str, Vec<String>)]) -> String {
    let runtime_functions: HashSet<&str> = exports
        .iter()
        .map(String::as_str)
        .chain(PRIVATE_TABLE_FUNCTIONS)
        .collect();
    let positions: HashMap<&str, usize> = exports
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect();
    let mut lines = vec![
        "// Copyright (c) 2026 Advanced Micro Devices, Inc.".to_owned(),
        "// SPDX-License-Identifier: MIT".to_owned(),
        "// Generated from the public HSA tool ABI and Linux exports.".to_owned(),
        String::new(),
        "use std::ffi::c_void;".to_owned(),
        "use std::sync::atomic::{AtomicPtr, Ordering};".to_owned(),
        String::new(),
        format!("pub(super) const EXPORT_COUNT: usize = {};", exports.len()),
    ];
    for (member, fields) in tables {
        lines.push(format!(
            "pub(super) const {}_COUNT: usize = {};",
            member.to_ascii_uppercase(),
            fields.len()
        ));
    }
    lines.extend([String::new(), "unsafe extern \"C\" {".to_owned()]);
    for name in exports
        .iter()
        .map(String::as_str)
        .chain(PRIVATE_TABLE_FUNCTIONS)
    {
        lines.push(format!("    fn __hsa_impl_{name}();"));
    }
    for name in PUBLIC_EXTENSION_SYMBOLS {
        lines.push(format!(
            "    #[link_name = \"{name}\"] fn __hsa_export_{name}();"
        ));
    }
    lines.extend([
        "}".to_owned(),
        String::new(),
        "static HSA_DEFAULT_SLOTS: [AtomicPtr<c_void>; EXPORT_COUNT] = [".to_owned(),
    ]);
    for name in exports {
        lines.push(format!(
            "    AtomicPtr::new(__hsa_impl_{name} as *mut c_void),"
        ));
    }
    lines.extend([
        "];".to_owned(),
        String::new(),
        "#[unsafe(no_mangle)]".to_owned(),
        "static hsa_dispatch_slots: [AtomicPtr<c_void>; EXPORT_COUNT] = [".to_owned(),
    ]);
    for index in 0..exports.len() {
        lines.push(format!(
            "    AtomicPtr::new(std::ptr::addr_of!(HSA_DEFAULT_SLOTS[{index}]) as *mut c_void),"
        ));
    }
    lines.extend([
        "];".to_owned(),
        String::new(),
        "pub(super) fn initial_functions() -> InitialFunctions {".to_owned(),
        "    InitialFunctions {".to_owned(),
    ]);
    for (member, fields) in tables {
        lines.push(format!("        {member}: ["));
        for name in fields {
            let value = if runtime_functions.contains(name.as_str()) {
                format!("__hsa_impl_{name} as *mut c_void")
            } else {
                "std::ptr::null_mut()".to_owned()
            };
            lines.push(format!("            {value},"));
        }
        lines.push("        ],".to_owned());
    }
    lines.extend([
        "    }".to_owned(),
        "}".to_owned(),
        String::new(),
        "#[derive(Clone, Copy)]".to_owned(),
        "pub(super) struct InitialFunctions {".to_owned(),
    ]);
    for (member, _) in tables {
        lines.push(format!(
            "    pub(super) {member}: [*mut c_void; {}_COUNT],",
            member.to_ascii_uppercase()
        ));
    }
    lines.extend([
        "}".to_owned(),
        String::new(),
        "pub(super) fn valid(tables: &super::ApiTables) -> bool {".to_owned(),
    ]);
    let mut checks = Vec::new();
    for (member, fields) in tables {
        for (index, name) in fields.iter().enumerate() {
            if runtime_functions.contains(name.as_str()) {
                checks.push(format!("!tables.{member}.functions[{index}].is_null()"));
            }
        }
    }
    lines.push(format!("    {}", checks.join("\n        && ")));
    lines.extend([
        "}".to_owned(),
        String::new(),
        "pub(super) fn publish(tables: &super::ApiTables) {".to_owned(),
    ]);
    for (member, fields) in tables {
        for (index, name) in fields.iter().enumerate() {
            if let Some(position) = positions.get(name.as_str()) {
                lines.push(format!(
                    "    hsa_dispatch_slots[{position}].store((&raw const tables.{member}.functions[{index}]) as *mut c_void, Ordering::Release);"
                ));
            }
        }
    }
    lines.extend([
        "}".to_owned(),
        String::new(),
        "pub(super) fn reset() {".to_owned(),
    ]);
    for index in 0..exports.len() {
        lines.push(format!(
            "    hsa_dispatch_slots[{index}].store(std::ptr::addr_of!(HSA_DEFAULT_SLOTS[{index}]) as *mut c_void, Ordering::Release);"
        ));
    }
    lines.extend([
        "}".to_owned(),
        String::new(),
        "pub(super) fn public_address(name: &str) -> Option<usize> {".to_owned(),
        "    match name {".to_owned(),
    ]);
    for name in PUBLIC_EXTENSION_SYMBOLS {
        lines.push(format!(
            "        \"{name}\" => Some(__hsa_export_{name} as *const () as usize),"
        ));
    }
    lines.extend([
        "        _ => None,".to_owned(),
        "    }".to_owned(),
        "}".to_owned(),
        String::new(),
    ]);
    lines.join("\n")
}

fn generate_dispatch(exports: &[String]) -> String {
    let mut lines = vec![
        "/* Copyright (c) 2026 Advanced Micro Devices, Inc. */".to_owned(),
        "/* SPDX-License-Identifier: MIT */".to_owned(),
        "/* Generated from the Linux HSA exports. */".to_owned(),
        ".text".to_owned(),
        ".hidden hsa_dispatch_slots".to_owned(),
    ];
    for name in exports
        .iter()
        .map(String::as_str)
        .chain(PRIVATE_TABLE_FUNCTIONS)
    {
        lines.push(format!(".hidden __hsa_impl_{name}"));
    }
    for (index, name) in exports.iter().enumerate() {
        lines.extend([
            format!(".globl {name}"),
            format!(".type {name}, @function"),
            format!("{name}:"),
            format!(
                "    mov r11, qword ptr [rip + hsa_dispatch_slots + {}]",
                index * 8
            ),
            "    jmp qword ptr [r11]".to_owned(),
            format!(".size {name}, .-{name}"),
        ]);
    }
    lines.push(String::new());
    lines.join("\n")
}

fn main() -> Result<(), Box<dyn Error>> {
    let manifest = env::var("CARGO_MANIFEST_DIR")?;
    let manifest = Path::new(&manifest);
    let header = manifest.join("../../api-headers/include/hsa/hsa_api_trace.h");
    let exports = manifest.join("src/exports_linux.S");
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", exports.display());

    let header = fs::read_to_string(header)?;
    let exports = exported_symbols(&fs::read_to_string(exports)?)?;
    let tables = TABLES
        .iter()
        .map(|&(member, table)| table_fields(&header, table).map(|fields| (member, fields)))
        .collect::<Result<Vec<_>, _>>()?;
    let out = env::var("OUT_DIR")?;
    let out = Path::new(&out);
    fs::write(
        out.join("api_trace_generated.rs"),
        generate_rust(&exports, &tables),
    )?;
    fs::write(
        out.join("api_trace_dispatch_linux.S"),
        generate_dispatch(&exports),
    )?;
    Ok(())
}
